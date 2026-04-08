"""
Training script for swin_base_patch4_window7_224.ms_in22k_ft_in1k
using ProgressiveClassifier from the existing module.

Usage:
    python train_swin.py

Outputs:
    saved_models/swin_progressive_best.pth   -- model weights
    saved_models/swin_class_names.txt        -- class names (one per line)
"""

import os, sys, gc
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split, StratifiedKFold

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from module.config import (
    DATA_DIR, DEVICE, EPOCHS, NFOLDS, BATCH_SIZE,
    NUM_WORKERS, PIN_MEMORY, PERSISTENT_WORKERS,
    TEST_SPLIT, PATIENCE, MIN_DELTA_METRIC, OPTIMIZE_METRIC, LR,
)
from module.utils       import FullDataset, Logger, get_base_transformations
from module.models      import get_img_size
from module.test        import test_model
from module.classifiers import get_classifier

# ─────────────────────────────────────────────────────────────────────────────
MODEL_NAME      = "swin_base_patch4_window7_224.ms_in22k_ft_in1k"
CLASSIFIER_TYPE = "progressive"
SAVE_DIR        = ROOT / "saved_models"
WEIGHTS_PATH    = SAVE_DIR / "swin_progressive_best.pth"
CLASS_NAMES_PATH= SAVE_DIR / "swin_class_names.txt"
# ─────────────────────────────────────────────────────────────────────────────


def main():
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    logger = Logger("swin_train", file_name="swin_train")
    logger.info(f"Model      : {MODEL_NAME}")
    logger.info(f"Classifier : {CLASSIFIER_TYPE}")
    logger.info(f"Device     : {DEVICE}")
    logger.info(f"Epochs : {EPOCHS}  |  Folds : {NFOLDS}  |  BS : {BATCH_SIZE}")

    # ── Dataset ──────────────────────────────────────────────────────────────
    img_size  = get_img_size(MODEL_NAME)
    transform = get_base_transformations(img_size)

    logger.info(f"Loading dataset : {DATA_DIR}  (img_size={img_size})")
    full_dataset = FullDataset(DATA_DIR, transform)

    targets     = np.array(full_dataset.targets)
    class_names = full_dataset.classes

    logger.info(f"Classes : {class_names}")
    logger.info(f"Samples : {len(targets)}")

    CLASS_NAMES_PATH.write_text("\n".join(class_names))
    logger.info(f"Class names saved -> {CLASS_NAMES_PATH}")

    # ── Train / test split ───────────────────────────────────────────────────
    train_val_idx, test_idx = train_test_split(
        np.arange(len(targets)),
        test_size=TEST_SPLIT,
        stratify=targets,
        random_state=42,
    )
    logger.info(f"Split : {len(train_val_idx)} train/val  |  {len(test_idx)} test")

    skf        = StratifiedKFold(n_splits=NFOLDS, shuffle=True, random_state=42)
    tv_targets = targets[train_val_idx]

    ClassifierClass = get_classifier(CLASSIFIER_TYPE)
    fold_results    = []
    best_fold_val   = 0.0
    best_fold_path  = SAVE_DIR / "swin_progressive_best_fold.pth"

    # ── K-Fold ───────────────────────────────────────────────────────────────
    for fold, (rel_tr, rel_val) in enumerate(skf.split(train_val_idx, tv_targets)):
        tr_idx  = train_val_idx[rel_tr]
        val_idx = train_val_idx[rel_val]

        def _loader(indices, shuffle):
            return DataLoader(
                Subset(full_dataset, indices),
                batch_size=BATCH_SIZE, shuffle=shuffle,
                num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
                persistent_workers=PERSISTENT_WORKERS and NUM_WORKERS > 0,
            )

        train_loader = _loader(tr_idx,  True)
        val_loader   = _loader(val_idx, False)

        logger.info(f"\n{'='*70}")
        logger.info(f"FOLD {fold+1}/{NFOLDS}")
        logger.info(f"{'='*70}")

        clf = ClassifierClass(
            model_name=MODEL_NAME,
            num_classes=len(class_names),
            device=DEVICE,
            checkpoint_path=str(WEIGHTS_PATH),
        )

        clf.fit(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=EPOCHS,
            lr=LR,
            use_sam=False,
            primary_metric=OPTIMIZE_METRIC,
            patience=PATIENCE,
            min_delta=MIN_DELTA_METRIC,
        )

        fold_results.append({
            "fold":           fold + 1,
            f"val_{OPTIMIZE_METRIC}": clf.best_metric_value,
            "val_acc":        clf.best_acc,
            "val_recall":     clf.best_recall,
            "val_f1":         clf.best_f1,
        })

        if clf.best_metric_value > best_fold_val:
            best_fold_val = clf.best_metric_value
            clf.save(str(best_fold_path))
            logger.info(f"  * New best fold ({best_fold_val:.4f}) -- fold checkpoint updated")

        del clf, train_loader, val_loader
        torch.cuda.empty_cache()
        gc.collect()

    # ── Summary ──────────────────────────────────────────────────────────────
    df = pd.DataFrame(fold_results)
    logger.info("\nK-Fold Summary:\n" + df.to_string(index=False))
    col = f"val_{OPTIMIZE_METRIC}"
    logger.info(f"Mean {OPTIMIZE_METRIC}: {df[col].mean():.4f} +/- {df[col].std():.4f}")

    # ── Final test evaluation ─────────────────────────────────────────────────
    logger.info("\nFinal held-out test evaluation...")

    test_loader = DataLoader(
        Subset(full_dataset, test_idx),
        batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
        persistent_workers=PERSISTENT_WORKERS and NUM_WORKERS > 0,
    )

    eval_clf = ClassifierClass(
        model_name=MODEL_NAME,
        num_classes=len(class_names),
        device=DEVICE,
    )
    checkpoint = WEIGHTS_PATH if WEIGHTS_PATH.exists() else best_fold_path
    eval_clf.load(str(checkpoint))
    logger.info(f"Loaded checkpoint : {checkpoint}")

    experiment_name = f"{MODEL_NAME}_classifier={CLASSIFIER_TYPE}_metric={OPTIMIZE_METRIC}"
    metrics = test_model(
        model_name=MODEL_NAME,
        model=eval_clf,
        loader=test_loader,
        classes=class_names,
        experiment_name=experiment_name,
        logger=logger,
        use_tta=False,
    )

    logger.info(f"\nTest Results:")
    logger.info(f"  Accuracy  : {metrics['accuracy']:.2f}%")
    logger.info(f"  Recall    : {metrics['recall']:.4f}")
    logger.info(f"  Precision : {metrics['precision']:.4f}")
    logger.info(f"  F1        : {metrics['f1']:.4f}")
    logger.info(f"\nWeights saved -> {checkpoint}")
    logger.info("Done.")


if __name__ == "__main__":
    main()
