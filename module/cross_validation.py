from torchvision import datasets
import torch
import torch.nn as nn
import torch.optim as optim
import os
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Subset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import gc
from sklearn.model_selection import train_test_split, StratifiedKFold

from module.models import get_model, get_img_size, check_model
from module.utils import *
from module.config import *
from module.visualization import Visualizer
from module.test import test_model
from module.classifiers import list_classifiers, get_classifier, BaseClassifier


def _build_loader_bda(
    disease_id: str,
    fold: int,
    split: str,
    img_size: int,
    registry_dir: str,
    shards_dir: str,
) -> DataLoader:
    """
    Build a DataLoader using the BDA pipeline (WebDataset shards or RegistryDataset).
    """
    from data_pipeline.loaders.webdataset_loader import get_dataloader
    return get_dataloader(
        disease=disease_id,
        fold=fold,
        split=split,
        img_size=img_size,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        registry_dir=registry_dir,
        shards_dir=shards_dir,
        pin_memory=PIN_MEMORY,
        persistent_workers=PERSISTENT_WORKERS,
    )


def _build_loader_legacy(
    full_dataset,
    indices: np.ndarray,
    shuffle: bool,
) -> DataLoader:
    """
    Build a DataLoader using the legacy FullDataset / Subset approach.
    """
    return DataLoader(
        Subset(full_dataset, indices),
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=PERSISTENT_WORKERS if NUM_WORKERS > 0 else False,
    )


class Cross_Validator:
    """
    Unified k-fold cross-validation for all classifier types.

    Supports two data modes (controlled by config.USE_BDA_PIPELINE):
      BDA mode   — DataLoader sourced from DiseaseRegistry + WebDataset shards.
      Legacy mode — DataLoader sourced from FullDataset / ImageFolder (ADNI only).

    Results are written to MLflow (if USE_MLFLOW=True) and the CSV fallback.
    """

    def __init__(self, model_names, logger: Logger, model_classifier_map=None):
        """
        Args:
            model_names: List of model names to train.
            logger: Logger instance.
            model_classifier_map: Dict mapping model_name -> classifier_type.
                                  If None or model not in map, uses 'baseline'.
        """
        self.model_names = model_names
        self.results = []
        self.logger = logger
        self.master_file = os.path.join(RESULTS_DIR, "master_results.csv")
        self.models_dir = os.path.join(RESULTS_DIR, "best_models")

        self.model_classifier_map = model_classifier_map or {}

        os.makedirs(self.models_dir, exist_ok=True)

        self.logger.debug(f"Models for cross-validation: {self.model_names}")
        self.logger.debug(f"Classifier mapping: {self.model_classifier_map}")
        self.logger.debug(f"Master results file: {self.master_file}")
        self.logger.debug(f"USE_BDA_PIPELINE: {USE_BDA_PIPELINE}")

    def run(self):
        """Main execution loop for cross-validation across all models."""
        master_df = None
        if os.path.exists(self.master_file):
            master_df = pd.read_csv(self.master_file)
            self.logger.debug(f"Master df exists -> {self.master_file}")

        # ── BDA path ─────────────────────────────────────────────────────────
        if USE_BDA_PIPELINE:
            self._run_bda(master_df)
        # ── Legacy path ───────────────────────────────────────────────────────
        else:
            self._run_legacy(master_df)

        self.logger.info("\n>>> Batch Complete.")

    # =========================================================================
    # BDA execution path
    # =========================================================================

    def _run_bda(self, master_df):
        """
        BDA-mode execution: data comes from DiseaseRegistry + WebDataset loader.
        """
        self.logger.info("\n" + "=" * 80)
        self.logger.info("BDA PIPELINE MODE")
        self.logger.info("=" * 80)
        self.logger.info(f"Disease ID  : {DISEASE_ID}")
        self.logger.info(f"Registry    : {REGISTRY_DIR}")
        self.logger.info(f"Shards      : {SHARDS_DIR}")

        # ── Ensure the BDA registry exists (auto-build if needed) ─────────────
        pipeline_ready = ensure_pipeline_ready(
            disease_id=DISEASE_ID,
            data_dir=DATA_DIR,
            layout=DATASET_LAYOUT,
            registry_dir=REGISTRY_DIR,
            shards_dir=SHARDS_DIR,
            nfolds=NFOLDS,
            csv_path=DATASET_CSV,
            auto_run=AUTO_RUN_PIPELINE,
        )

        if not pipeline_ready:
            self.logger.warning(
                "BDA registry not available. Falling back to legacy FullDataset."
            )
            self._run_legacy(master_df)
            return

        # ── Load metadata for class info and weights ──────────────────────────
        from data_pipeline.registry.disease_registry import DiseaseRegistry

        reg = DiseaseRegistry(REGISTRY_DIR, use_dask=False)
        classes = reg.class_names(DISEASE_ID)
        class_weights_np = reg.class_weights(DISEASE_ID, split="train")
        class_weights_tensor = torch.FloatTensor(class_weights_np).to(DEVICE)

        self.logger.info(f"Classes ({len(classes)}): {classes}")
        self.logger.info("Class weights: " + ", ".join(
            f"{cls}={w:.3f}" for cls, w in zip(classes, class_weights_np)
        ))

        # ── Log per-class counts from registry summary ────────────────────────
        summary = reg.summary(DISEASE_ID)

        # ── Train each model ──────────────────────────────────────────────────
        for model_name in self.model_names:
            classifier_type_input = self.model_classifier_map.get(model_name, 'baseline')

            if classifier_type_input == "all":
                classifiers_to_be_used = list_classifiers()
            elif isinstance(classifier_type_input, str):
                classifiers_to_be_used = [classifier_type_input]
            elif isinstance(classifier_type_input, (list, tuple)):
                classifiers_to_be_used = list(classifier_type_input)
            else:
                self.logger.error(
                    f"Classifier type not valid -> {classifier_type_input} "
                    f"({type(classifier_type_input)})"
                )
                continue

            for classifier_type in classifiers_to_be_used:
                self._run_classifier_bda(
                    model_name=model_name,
                    classifier_type=classifier_type,
                    classes=classes,
                    class_weights_tensor=class_weights_tensor,
                    master_df=master_df,
                )

    def _run_classifier_bda(
        self,
        model_name: str,
        classifier_type: str,
        classes: list,
        class_weights_tensor,
        master_df,
    ):
        """
        BDA-mode: single classifier training with k-fold cross-validation.
        DataLoaders are built from the BDA registry.
        """
        experiment_name = (
            f"{model_name}_classifier={classifier_type}"
            f"_disease={DISEASE_ID}_metric={OPTIMIZE_METRIC}"
        )
        checkpoint_path = os.path.join(
            self.models_dir, f"{experiment_name}_best_weights.pth"
        )
        fold_checkpoint_path = os.path.join(
            self.models_dir, f"{experiment_name}_best_fold.pth"
        )

        self.logger.info("\n" + "=" * 80)
        self.logger.info(f"STARTING: {experiment_name.upper()}")
        self.logger.info("=" * 80)

        # Skip if already completed
        if master_df is not None:
            existing = master_df[
                (master_df.get('model_name', pd.Series(dtype=str)) == model_name) &
                (master_df.get('classifier_type', pd.Series(dtype=str)) == classifier_type) &
                (master_df.get('disease_id', pd.Series(dtype=str)) == DISEASE_ID) &
                (master_df.get('optimize_metric', pd.Series(dtype=str)) == OPTIMIZE_METRIC)
            ]
            if not existing.empty:
                self.logger.info(">> Experiment already completed. Skipping.")
                return

        if not check_model(model_name=model_name):
            self.logger.error(f"timm does not support model: {model_name}")
            return

        img_size = get_img_size(model_name)
        self.logger.debug(f"Image size for {model_name}: {img_size}")

        classifier_class = get_classifier(classifier_type)

        fold_metrics = []
        training_history = []
        best_fold = 0
        best_fold_metric = 0.0

        # ── K-Fold cross-validation ───────────────────────────────────────────
        for fold_idx in range(NFOLDS):
            self.logger.info(f"\n  [Fold {fold_idx + 1}/{NFOLDS}]")

            train_loader = _build_loader_bda(
                disease_id=DISEASE_ID,
                fold=fold_idx,
                split="train",
                img_size=img_size,
                registry_dir=REGISTRY_DIR,
                shards_dir=SHARDS_DIR,
            )
            val_loader = _build_loader_bda(
                disease_id=DISEASE_ID,
                fold=fold_idx,
                split="val",
                img_size=img_size,
                registry_dir=REGISTRY_DIR,
                shards_dir=SHARDS_DIR,
            )

            classifier: BaseClassifier = classifier_class(
                model_name=model_name,
                num_classes=len(classes),
                device=DEVICE,
                checkpoint_path=checkpoint_path,
                class_weights_tensor=class_weights_tensor,
            )

            use_sam = classifier_type in ['clinical_grade', 'ultimate']

            history = classifier.fit(
                train_loader=train_loader,
                val_loader=val_loader,
                epochs=EPOCHS,
                lr=LR,
                use_sam=use_sam,
                primary_metric=OPTIMIZE_METRIC,
                patience=PATIENCE,
                min_delta=MIN_DELTA_METRIC,
            )
            training_history.append(history)

            fold_metric_value = classifier.best_metric_value

            self.logger.info(
                f"  Fold {fold_idx + 1} Best {OPTIMIZE_METRIC.capitalize()}: "
                f"{fold_metric_value:.4f} | Acc: {classifier.best_acc:.2f}% | "
                f"Recall: {classifier.best_recall:.4f}"
            )

            fold_metrics.append({
                'fold': fold_idx + 1,
                f'val_{OPTIMIZE_METRIC}': fold_metric_value,
                'val_acc': classifier.best_acc,
                'val_recall': classifier.best_recall,
                'val_f1': classifier.best_f1,
            })

            if fold_metric_value > best_fold_metric:
                best_fold = fold_idx
                best_fold_metric = fold_metric_value
                classifier.save(fold_checkpoint_path)
                self.logger.info(f"  ✓ Best fold so far! Checkpoint saved.")

            del classifier, train_loader, val_loader
            torch.cuda.empty_cache()
            gc.collect()
            self.logger.info(f"Fold {fold_idx + 1} cleanup completed")

        # ── Aggregate fold results ────────────────────────────────────────────
        fold_df = pd.DataFrame(fold_metrics)
        aggregate = {
            f'mean_fold_{OPTIMIZE_METRIC}': fold_df[f'val_{OPTIMIZE_METRIC}'].mean(),
            f'std_fold_{OPTIMIZE_METRIC}':  fold_df[f'val_{OPTIMIZE_METRIC}'].std(),
            'mean_fold_acc':    fold_df['val_acc'].mean(),
            'std_fold_acc':     fold_df['val_acc'].std(),
            'mean_fold_recall': fold_df['val_recall'].mean(),
            'mean_fold_f1':     fold_df['val_f1'].mean(),
        }
        self.logger.info(
            f"\n  K-Fold Summary:\n"
            f"    Mean {OPTIMIZE_METRIC.capitalize()}: "
            f"{aggregate[f'mean_fold_{OPTIMIZE_METRIC}']:.4f} "
            f"± {aggregate[f'std_fold_{OPTIMIZE_METRIC}']:.4f}\n"
            f"    Best Fold: {best_fold + 1} ({best_fold_metric:.4f})"
        )

        # ── Final test evaluation ─────────────────────────────────────────────
        self.logger.info("\n  Loading best fold model for final evaluation...")
        test_loader = _build_loader_bda(
            disease_id=DISEASE_ID,
            fold=-1,
            split="test",
            img_size=img_size,
            registry_dir=REGISTRY_DIR,
            shards_dir=SHARDS_DIR,
        )

        eval_classifier: BaseClassifier = classifier_class(
            model_name=model_name,
            num_classes=len(classes),
            device=DEVICE,
        )
        eval_classifier.load(checkpoint_path)

        metrics = test_model(
            model_name=model_name,
            model=eval_classifier,
            loader=test_loader,
            classes=classes,
            experiment_name=experiment_name,
            history=training_history,
            logger=self.logger,
            use_tta=False,
        )

        final_accuracy = metrics["accuracy"]
        final_recall   = metrics["recall"]

        self.logger.info(
            f"\n  Final Test Results:\n"
            f"    Accuracy : {final_accuracy:.2f}%\n"
            f"    Recall   : {final_recall:.4f}\n"
            f"    Precision: {metrics['precision']:.4f}\n"
            f"    F1       : {metrics['f1']:.4f}"
        )

        # ── MLflow + CSV logging ──────────────────────────────────────────────
        final_metrics = {
            'test_accuracy':  final_accuracy,
            'test_recall':    final_recall,
            'test_precision': metrics['precision'],
            'test_f1':        metrics['f1'],
        }
        config_dict = {
            'epochs': EPOCHS, 'batch_size': BATCH_SIZE,
            'lr': LR, 'n_folds': NFOLDS,
        }

        if USE_MLFLOW:
            try:
                from data_pipeline.tracking.mlflow_tracker import log_cross_validation_result
                log_cross_validation_result(
                    model_name=model_name,
                    classifier_type=classifier_type,
                    disease=DISEASE_ID,
                    fold_metrics=fold_metrics,
                    final_metrics=final_metrics,
                    config=config_dict,
                    model_path=checkpoint_path,
                    tracking_uri=MLFLOW_TRACKING_URI,
                    experiment_name=f"{DISEASE_ID}_cross_validation",
                )
            except Exception as e:
                self.logger.warning(f"MLflow logging failed (non-fatal): {e}")

        # Always write CSV fallback
        results_data = {
            'model_name':          [model_name],
            'classifier_type':     [classifier_type],
            'disease_id':          [DISEASE_ID],
            'optimize_metric':     [OPTIMIZE_METRIC],
            'pretrained':          [PRETRAINED],
            'best_val_metric':     [f"{best_fold_metric:.4f}"],
            'final_test_accuracy': [f"{final_accuracy:.2f}%"],
            'final_test_recall':   [f"{final_recall:.4f}"],
            'total_epochs':        [EPOCHS],
            'batch_size':          [BATCH_SIZE],
            'n_splits':            [NFOLDS],
            **{k: [v] for k, v in aggregate.items()},
        }
        pd.DataFrame(results_data).to_csv(
            self.master_file,
            mode='a',
            header=not os.path.exists(self.master_file),
            index=False,
        )
        self.logger.info(f"\nResults appended to {self.master_file}")
        self.logger.info(f"EXPERIMENT FINISHED: {experiment_name.upper()}")

        del eval_classifier, test_loader
        torch.cuda.empty_cache()
        gc.collect()

    # =========================================================================
    # Legacy execution path (original behaviour, unchanged logic)
    # =========================================================================

    def _run_legacy(self, master_df):
        """
        Legacy-mode execution: data comes from FullDataset / ImageFolder.
        Identical to the original cross_validation.py logic.
        """
        self.logger.info("\n" + "=" * 80)
        self.logger.info("LEGACY MODE — Loading dataset from ImageFolder")
        self.logger.info("=" * 80)
        self.logger.info(f"Loading dataset from: {DATA_DIR}")

        temp_transform = transforms.Compose([transforms.ToTensor()])
        base_dataset = FullDataset(DATA_DIR, temp_transform)

        targets = np.array(base_dataset.targets)
        classes = base_dataset.classes

        self.logger.info(f"Total samples: {len(targets)}")
        self.logger.debug(f"Classes found: {classes}")

        class_counts = np.bincount(targets)
        total_samples = len(targets)
        class_weights = total_samples / (len(classes) * class_counts)
        class_weights_tensor = torch.FloatTensor(class_weights).to(DEVICE)

        self.logger.info("=" * 80)
        self.logger.info("CLASS DISTRIBUTION & WEIGHTS:")
        for i, cls in enumerate(classes):
            self.logger.info(
                f"  {cls:<20}: {class_counts[i]:>4} samples "
                f"({100 * class_counts[i] / total_samples:>5.1f}%) | "
                f"Weight: {class_weights[i]:.3f}"
            )
        self.logger.info("=" * 80)

        train_val_indices, test_indices = train_test_split(
            np.arange(len(targets)),
            test_size=TEST_SPLIT,
            stratify=targets,
            random_state=42,
        )

        self.logger.info(
            f"Data split: {len(train_val_indices)} train/val, "
            f"{len(test_indices)} test"
        )

        train_val_targets = targets[train_val_indices]
        skf = StratifiedKFold(n_splits=NFOLDS, shuffle=True, random_state=42)

        for model_name in self.model_names:
            classifier_type_input = self.model_classifier_map.get(model_name, 'baseline')

            if classifier_type_input == "all":
                classifiers_to_be_used = list_classifiers()
            elif isinstance(classifier_type_input, str):
                classifiers_to_be_used = [classifier_type_input]
            elif isinstance(classifier_type_input, (list, tuple)):
                classifiers_to_be_used = list(classifier_type_input)
            else:
                self.logger.error(
                    f"Classifier type not valid -> {classifier_type_input} "
                    f"({type(classifier_type_input)})"
                )
                continue

            for classifier_type in classifiers_to_be_used:
                self._run_classifier_legacy(
                    model_name=model_name,
                    classifier_type=classifier_type,
                    base_dataset=base_dataset,
                    train_val_indices=train_val_indices,
                    test_indices=test_indices,
                    targets=targets,
                    classes=classes,
                    skf=skf,
                    master_df=master_df,
                    class_weights_tensor=class_weights_tensor,
                )

    def _run_classifier_legacy(
        self,
        model_name,
        classifier_type,
        base_dataset,
        train_val_indices,
        test_indices,
        targets,
        classes,
        skf,
        master_df,
        class_weights_tensor,
    ):
        """
        Legacy-mode: single classifier training.
        Identical to the original _run_classifier() implementation.
        """
        experiment_name = (
            f"{model_name}_classifier={classifier_type}_metric={OPTIMIZE_METRIC}"
        )
        checkpoint_path = os.path.join(
            self.models_dir, f"{experiment_name}_best_weights.pth"
        )
        fold_checkpoint_path = os.path.join(
            self.models_dir, f"{experiment_name}_best_fold.pth"
        )

        self.logger.info("\n" + "=" * 80)
        self.logger.info(f"STARTING EXPERIMENT: {experiment_name.upper()}")
        self.logger.info(f"Using Classifier: {classifier_type}")
        self.logger.info("=" * 80)

        if master_df is not None:
            existing = master_df[
                (master_df['model_name'] == model_name) &
                (master_df.get('classifier_type', pd.Series(dtype=str)) == classifier_type) &
                (master_df.get('optimize_metric', pd.Series(dtype=str)) == OPTIMIZE_METRIC)
            ]
            if not existing.empty:
                self.logger.info(">> Experiment already completed. Skipping.")
                return

        if not check_model(model_name=model_name):
            self.logger.error(f"timm does not support model: {model_name}")
            return

        img_size = get_img_size(model_name)
        model_transform = get_base_transformations(img_size)
        full_dataset = FullDataset(DATA_DIR, model_transform)

        self.logger.info(f"\n=== K-Fold: {model_name} with {classifier_type} ===")

        train_val_targets = targets[train_val_indices]
        fold_metrics = []
        training_history = []
        best_fold = 0
        best_fold_metric = 0.0

        classifier_class = get_classifier(classifier_type)

        for fold, (fold_train_idx_rel, fold_val_idx_rel) in enumerate(
            skf.split(train_val_indices, train_val_targets)
        ):
            self.logger.info(f"\n  [Fold {fold + 1}/{NFOLDS}]")

            train_idx = train_val_indices[fold_train_idx_rel]
            val_idx   = train_val_indices[fold_val_idx_rel]

            train_loader = _build_loader_legacy(full_dataset, train_idx, shuffle=True)
            val_loader   = _build_loader_legacy(full_dataset, val_idx,   shuffle=False)

            classifier: BaseClassifier = classifier_class(
                model_name=model_name,
                num_classes=len(classes),
                device=DEVICE,
                checkpoint_path=checkpoint_path,
                class_weights_tensor=class_weights_tensor,
            )

            use_sam = classifier_type in ['clinical_grade', 'ultimate']

            history = classifier.fit(
                train_loader=train_loader,
                val_loader=val_loader,
                epochs=EPOCHS,
                lr=LR,
                use_sam=use_sam,
                primary_metric=OPTIMIZE_METRIC,
                patience=PATIENCE,
                min_delta=MIN_DELTA_METRIC,
            )
            training_history.append(history)

            fold_metric_value = classifier.best_metric_value
            self.logger.info(
                f"  Fold {fold + 1} Best {OPTIMIZE_METRIC.capitalize()}: "
                f"{fold_metric_value:.4f} | Acc: {classifier.best_acc:.2f}% | "
                f"Recall: {classifier.best_recall:.4f}"
            )

            fold_metrics.append({
                'fold': fold + 1,
                f'val_{OPTIMIZE_METRIC}': fold_metric_value,
                'val_acc': classifier.best_acc,
                'val_recall': classifier.best_recall,
                'val_f1': classifier.best_f1,
            })

            if fold_metric_value > best_fold_metric:
                best_fold = fold
                best_fold_metric = fold_metric_value
                classifier.save(fold_checkpoint_path)
                self.logger.info(f"  ✓ Best fold so far! Checkpoint saved.")

            del classifier, train_loader, val_loader
            torch.cuda.empty_cache()
            gc.collect()
            self.logger.info(f"Fold {fold + 1} cleanup completed")

        fold_df = pd.DataFrame(fold_metrics)
        aggregate = {
            f'mean_fold_{OPTIMIZE_METRIC}': fold_df[f'val_{OPTIMIZE_METRIC}'].mean(),
            f'std_fold_{OPTIMIZE_METRIC}':  fold_df[f'val_{OPTIMIZE_METRIC}'].std(),
            'mean_fold_acc':    fold_df['val_acc'].mean(),
            'std_fold_acc':     fold_df['val_acc'].std(),
            'mean_fold_recall': fold_df['val_recall'].mean(),
            'mean_fold_f1':     fold_df['val_f1'].mean(),
        }
        self.logger.info(
            f"\n  K-Fold Summary:\n"
            f"    Mean {OPTIMIZE_METRIC.capitalize()}: "
            f"{aggregate[f'mean_fold_{OPTIMIZE_METRIC}']:.4f} "
            f"± {aggregate[f'std_fold_{OPTIMIZE_METRIC}']:.4f}\n"
            f"    Best Fold: {best_fold + 1} ({best_fold_metric:.4f})"
        )

        test_subset  = Subset(full_dataset, test_indices)
        test_loader  = DataLoader(
            test_subset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=PIN_MEMORY,
            persistent_workers=PERSISTENT_WORKERS if NUM_WORKERS > 0 else False,
        )

        eval_classifier: BaseClassifier = classifier_class(
            model_name=model_name,
            num_classes=len(classes),
            device=DEVICE,
        )
        eval_classifier.load(checkpoint_path)

        metrics = test_model(
            model_name=model_name,
            model=eval_classifier,
            loader=test_loader,
            classes=classes,
            experiment_name=experiment_name,
            history=training_history,
            logger=self.logger,
            use_tta=False,
        )

        final_accuracy = metrics["accuracy"]
        final_recall   = metrics["recall"]

        self.logger.info(
            f"\n  Final Test Results:\n"
            f"    Accuracy : {final_accuracy:.2f}%\n"
            f"    Recall   : {final_recall:.4f}\n"
            f"    Precision: {metrics['precision']:.4f}\n"
            f"    F1       : {metrics['f1']:.4f}"
        )

        results_data = {
            'model_name':          [model_name],
            'classifier_type':     [classifier_type],
            'optimize_metric':     [OPTIMIZE_METRIC],
            'pretrained':          [PRETRAINED],
            'best_val_metric':     [f"{best_fold_metric:.4f}"],
            'final_test_accuracy': [f"{final_accuracy:.2f}%"],
            'final_test_recall':   [f"{final_recall:.4f}"],
            'total_epochs':        [EPOCHS],
            'batch_size':          [BATCH_SIZE],
            'n_splits':            [NFOLDS],
            **{k: [v] for k, v in aggregate.items()},
        }
        pd.DataFrame(results_data).to_csv(
            self.master_file,
            mode='a',
            header=not os.path.exists(self.master_file),
            index=False,
        )
        self.logger.info(f"\nResults appended to {self.master_file}")
        self.logger.info(f"EXPERIMENT FINISHED: {experiment_name.upper()}")

        del eval_classifier, test_loader, full_dataset
        torch.cuda.empty_cache()
        gc.collect()
