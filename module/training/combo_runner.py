"""
Generic (backbone, classifier_type) training orchestration.

This module is the generalized body of train_swin.py's main(): K-fold CV,
best-fold checkpointing, then a final held-out test evaluation. It is
deliberately classifier-agnostic — it calls the same four BaseClassifier
methods (__init__, fit, evaluate, save/load) regardless of whether
classifier_type is 'baseline', 'evidential', or 'progressive', relying on
BaseClassifier's contract for substitutability (Liskov Substitution): every
concrete classifier honors the same constructor kwargs and the same
`fit(train_loader, val_loader, epochs, lr, use_sam, primary_metric,
patience, min_delta)` signature (verified directly against
module/classifiers/*.py — see PROJECT_NOTES.md).

Adding a 5th combo later means adding one ComboConfig + one thin script
that calls train_combo() — this file does not change (Open/Closed).
"""

from __future__ import annotations

import gc
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import shutil

import numpy as np
import pandas as pd
import torch

from module.classifiers import get_classifier
from module.models import get_img_size
from module.training.data_split import SplitDataset, build_loader, load_and_split
from module.training.tracking import ExperimentTracker


@dataclass(frozen=True)
class ComboConfig:
    """Everything that varies between the four deployment combos."""

    combo_id: str              # short stable id, e.g. "effnet_progressive" — used in filenames/manifest
    display_name: str          # human-readable, e.g. "EfficientNet-B4 + Progressive"
    model_name: str            # timm identifier, e.g. "tf_efficientnet_b4.ns_jft_in1k"
    classifier_type: str       # 'baseline' | 'evidential' | 'progressive' (module.classifiers registry key)
    weights_filename: str      # e.g. "effnet_progressive_best.pth"
    class_names_filename: str  # e.g. "effnet_class_names.txt"


@dataclass
class TrainingRunResult:
    combo: ComboConfig
    img_size: int
    class_names: list[str]
    fold_metrics: list[dict]
    test_metrics: dict          # raw dict from BaseClassifier.evaluate() — see its docstring for keys
    weights_path: Path
    class_names_path: Path


def _instantiate(cfg: ComboConfig, num_classes: int, device: str,
                  checkpoint_path: Optional[str], class_weights_tensor):
    ClassifierClass = get_classifier(cfg.classifier_type)
    return ClassifierClass(
        model_name=cfg.model_name,
        num_classes=num_classes,
        device=device,
        checkpoint_path=checkpoint_path,
        class_weights_tensor=class_weights_tensor,
    )


def train_combo(
    cfg: ComboConfig,
    *,
    data_dir: str,
    save_dir: Path,
    device: str,
    epochs: int,
    nfolds: int,
    batch_size: int,
    lr: float,
    patience: int,
    min_delta: float,
    use_sam: bool,
    primary_metric: str,
    test_split: float,
    num_workers: int,
    pin_memory: bool,
    persistent_workers: bool,
    logger,
    tracker: ExperimentTracker,
) -> TrainingRunResult:
    save_dir.mkdir(parents=True, exist_ok=True)
    weights_path = save_dir / cfg.weights_filename
    class_names_path = save_dir / cfg.class_names_filename
    best_fold_path = save_dir / f"{cfg.combo_id}_best_fold.pth"

    img_size = get_img_size(cfg.model_name)

    logger.info(f"Combo      : {cfg.display_name} ({cfg.combo_id})")
    logger.info(f"Model      : {cfg.model_name}  (img_size={img_size})")
    logger.info(f"Classifier : {cfg.classifier_type}")
    logger.info(f"Device     : {device}")
    logger.info(f"Epochs     : {epochs}  |  Folds : {nfolds}  |  BS : {batch_size}")

    split: SplitDataset = load_and_split(data_dir, img_size, test_split, device)
    class_names = split.class_names
    logger.info(f"Classes ({len(class_names)}): {class_names}")
    class_names_path.write_text("\n".join(class_names))
    logger.info(f"Class names saved -> {class_names_path}")

    tracker.start_run(
        run_name=cfg.combo_id,
        params={
            "model_name": cfg.model_name,
            "classifier_type": cfg.classifier_type,
            "img_size": img_size,
            "epochs": epochs,
            "nfolds": nfolds,
            "batch_size": batch_size,
            "lr": lr,
            "primary_metric": primary_metric,
        },
    )

    fold_results = []
    best_fold_val = 0.0

    try:
        for fold, tr_idx, val_idx in split.kfold_splits(nfolds):
            logger.info(f"\n{'=' * 70}\nFOLD {fold}/{nfolds}\n{'=' * 70}")

            train_loader = build_loader(split.full_dataset, tr_idx, batch_size, True,
                                         num_workers, pin_memory, persistent_workers)
            val_loader = build_loader(split.full_dataset, val_idx, batch_size, False,
                                       num_workers, pin_memory, persistent_workers)

            clf = _instantiate(cfg, len(class_names), device, str(weights_path), split.class_weights_tensor)

            if hasattr(clf, "set_phases") :
                clf.set_phases(4)
                # clf.set_sequential_scheduler()

            clf.fit(
                train_loader=train_loader,
                val_loader=val_loader,
                epochs=epochs,
                lr=lr,
                use_sam=use_sam,
                primary_metric=primary_metric,
                patience=patience,
                min_delta=min_delta,
            )

            fold_metrics = {
                "fold": fold,
                f"val_{primary_metric}": clf.best_metric_value,
                "val_acc": clf.best_acc,
                "val_recall": clf.best_recall,
                "val_f1": clf.best_f1,
            }
            fold_results.append(fold_metrics)
            tracker.log_metrics(fold_metrics, step=fold)

            if clf.best_metric_value > best_fold_val:
                best_fold_val = clf.best_metric_value
                shutil.copy(str(weights_path), str(best_fold_path)) # copy the peak file
                logger.info(f"  * New best fold ({best_fold_val:.4f}) — checkpoint updated")

            del clf, train_loader, val_loader
            if device == "cuda":
                torch.cuda.empty_cache()
            gc.collect()

        df = pd.DataFrame(fold_results)
        col = f"val_{primary_metric}"
        logger.info("\nK-Fold Summary:\n" + df.to_string(index=False))
        logger.info(f"Mean {primary_metric}: {df[col].mean():.4f} +/- {df[col].std():.4f}")

        # ── Final held-out test evaluation ──────────────────────────────────
        logger.info("\nFinal held-out test evaluation...")

        if best_fold_path.exists():
            shutil.copy(str(best_fold_path), str(weights_path)) # sync best-fold peak back into weights_path
            logger.info(f"Synced best-fold checkpoint -> {weights_path} (best fold val={best_fold_val:.4f})")

        test_loader = build_loader(split.full_dataset, split.test_idx, batch_size, False,
                                    num_workers, pin_memory, persistent_workers)

        eval_clf = _instantiate(cfg, len(class_names), device, None, None)
        checkpoint = weights_path
        eval_clf.load(str(checkpoint))
        logger.info(f"Loaded checkpoint : {checkpoint}")

        test_metrics = eval_clf.evaluate(test_loader, class_names)
        logger.info(
            f"\nTest Results — Accuracy: {test_metrics['accuracy']:.2f}%  "
            f"Recall: {test_metrics['recall']:.4f}  Precision: {test_metrics['precision']:.4f}  "
            f"F1: {test_metrics['f1']:.4f}"
        )

        tracker.log_metrics({
            "test_accuracy": test_metrics["accuracy"],
            "test_recall": test_metrics["recall"],
            "test_precision": test_metrics["precision"],
            "test_f1": test_metrics["f1"],
        })
        if checkpoint.exists():
            tracker.log_artifact(str(checkpoint))

        return TrainingRunResult(
            combo=cfg,
            img_size=img_size,
            class_names=class_names,
            fold_metrics=fold_results,
            test_metrics=test_metrics,
            weights_path=checkpoint,
            class_names_path=class_names_path,
        )
    finally:
        tracker.end_run()
