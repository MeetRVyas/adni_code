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

from .models import get_model, get_img_size
from .utils import *
from .config import *
from .visualization import Visualizer
from .test import test_model
from classifiers import get_classifier, BaseClassifier


class Cross_Validator:
    """
    Unified k-fold cross-validation for all classifier types.
    
    Key features:
    - Single execution path for all classifiers
    - Stratified k-fold splitting
    - GPU memory management
    - Comprehensive metric tracking
    """
    
    def __init__(self, model_names, logger: Logger, model_classifier_map=None):
        """
        Args:
            model_names: List of model names to train
            logger: Logger instance
            model_classifier_map: Dict mapping model_name -> classifier_type
                                 If None or model not in map, uses 'simple' classifier
                                 Example: {'resnet18': 'progressive', 'efficientnet_b4': 'ultimate'}
        """
        self.model_names = model_names
        self.results = []
        self.logger = logger
        self.master_file = os.path.join(RESULTS_DIR, "master_results.csv")
        self.models_dir = os.path.join(RESULTS_DIR, "best_models")
        
        # Classifier mapping
        self.model_classifier_map = model_classifier_map or {}
        
        os.makedirs(self.models_dir, exist_ok = True)
        
        self.logger.debug(f"Models for cross-validation: {self.model_names}")
        self.logger.debug(f"Classifier mapping: {self.model_classifier_map}")
        self.logger.debug(f"Master results file: {self.master_file}")

    def run(self):
        """Main execution loop for cross-validation across all models"""
        master_df = None
        if os.path.exists(self.master_file):
            master_df = pd.read_csv(self.master_file)
            self.logger.debug(f"Master df exists -> {self.master_file}")
        
        # Load dataset ONCE
        self.logger.info("\n" + "="*80)
        self.logger.info("LOADING DATASET (ONCE FOR ALL MODELS)")
        self.logger.info("="*80)
        self.logger.info(f"Loading dataset from: {DATA_DIR}")
        
        temp_transform = transforms.Compose([transforms.ToTensor()])
        base_dataset = FullDataset(DATA_DIR, temp_transform)

        training_history = []
        
        targets = np.array(base_dataset.targets)
        classes = base_dataset.classes
        
        self.logger.info(f"Total samples: {len(targets)}")
        self.logger.debug(f"Classes found: {classes}")
        
        # Calculate Class Weights (same for all models)
        class_counts = np.bincount(targets)
        total_samples = len(targets)
        class_weights = total_samples / (len(classes) * class_counts)
        class_weights_tensor = torch.FloatTensor(class_weights).to(DEVICE)
        
        self.logger.info("="*80)
        self.logger.info("CLASS DISTRIBUTION & WEIGHTS:")
        for i, cls in enumerate(classes):
            self.logger.info(f"  {cls:<20}: {class_counts[i]:>4} samples "
                           f"({100*class_counts[i]/total_samples:>5.1f}%) | Weight: {class_weights[i]:.3f}")
        self.logger.info("="*80)
        
        # Train/Test Split ONCE (same test set for all models)
        train_val_indices, test_indices = train_test_split(
            np.arange(len(targets)),
            test_size=TEST_SPLIT,
            stratify=targets,
            random_state=42
        )
        
        self.logger.info(f"Data split: {len(train_val_indices)} train/val, {len(test_indices)} test")
        
        train_val_targets = targets[train_val_indices]
        
        skf = StratifiedKFold(n_splits=NFOLDS, shuffle=True, random_state=42)
        
        # Train each model
        for model_name in self.model_names:
            # Get classifier type (default to 'simple' if not specified)
            classifier_type = self.model_classifier_map.get(model_name, 'baseline')
            
            self._run_classifier(
                model_name=model_name,
                classifier_type=classifier_type,
                base_dataset=base_dataset,
                train_val_indices=train_val_indices,
                test_indices=test_indices,
                targets=targets,
                classes=classes,
                skf=skf,
                master_df=master_df
            )
        
        self.logger.info("\n>>> Batch Complete.")

    def _run_classifier(self, model_name, classifier_type, base_dataset, 
                       train_val_indices, test_indices, targets, classes, skf, master_df):
        """
        Unified training method for ALL classifier types.
        
        Flow:
        1. Check if already completed
        2. Get classifier class from registry
        3. Create model-specific dataset
        4. Run k-fold CV with classifier.fit()
        5. Evaluate on test set with classifier.evaluate()
        6. Generate visualizations (including GradCAM if model available)
        7. Save unified results
        """
        experiment_name = f"{model_name}_classifier={classifier_type}_metric={OPTIMIZE_METRIC}"
        checkpoint_path = os.path.join(self.models_dir, f"{experiment_name}_best_weights.pth")
        fold_checkpoint_path = os.path.join(self.models_dir, f"{experiment_name}_best_fold.pth")
        
        self.logger.info("\n" + "="*80)
        self.logger.info(f"STARTING EXPERIMENT: {experiment_name.upper()}")
        self.logger.info(f"Using Classifier: {classifier_type}")
        self.logger.info("="*80)

        # Skip if already completed
        if master_df is not None:
            existing = master_df[
                (master_df['model_name'] == model_name) &
                (master_df.get('classifier_type', '') == classifier_type) &
                (master_df.get('optimize_metric', '') == OPTIMIZE_METRIC)
            ]
            if not existing.empty:
                self.logger.info(">> Experiment already completed. Skipping.")
                self.logger.info(existing.to_string())
                self.logger.info("="*80)
                return

        # Get image size and transforms
        img_size = get_img_size(model_name)
        self.logger.debug(f"Image size for {model_name}: {img_size}")
        model_transform = get_base_transformations(img_size)

        # Load dataset
        self.logger.info(f"Loading dataset from: {DATA_DIR}")
        full_dataset = FullDataset(DATA_DIR, model_transform)

        self.logger.info(f"\n=== K-Fold Cross-Validation: {model_name} with {classifier_type} ===")
        
        train_val_targets = targets[train_val_indices]
        fold_metrics = []
        training_histoy = []
        best_fold = 0
        best_fold_metric = 0.0
        
        # Create fresh classifier instance
        classifier_class = get_classifier(classifier_type)
        
        # K-Fold Cross-Validation
        for fold, (fold_train_idx_rel, fold_val_idx_rel) in enumerate(skf.split(train_val_indices, train_val_targets)):
            self.logger.info(f"\n  [Fold {fold+1}/{NFOLDS}]")
            
            # Get absolute indices
            train_idx = train_val_indices[fold_train_idx_rel]
            val_idx = train_val_indices[fold_val_idx_rel]
            
            # Create datasets and loaders
            train_ds = Subset(full_dataset, train_idx)
            val_ds = Subset(full_dataset, val_idx)
            
            train_loader = DataLoader(
                train_ds,
                batch_size=BATCH_SIZE,
                shuffle=True,
                num_workers=NUM_WORKERS,
                pin_memory=PIN_MEMORY,
                persistent_workers=PERSISTENT_WORKERS if NUM_WORKERS > 0 else False
            )
            val_loader = DataLoader(
                val_ds,
                batch_size=BATCH_SIZE,
                shuffle=False,
                num_workers=NUM_WORKERS,
                pin_memory=PIN_MEMORY,
                persistent_workers=PERSISTENT_WORKERS if NUM_WORKERS > 0 else False
            )
            classifier : BaseClassifier = classifier_class(
                model_name=model_name,
                num_classes=len(classes),
                device=DEVICE
            )
            
            self.logger.info(f"  Training {classifier_type} classifier...")
            
            # Determine if should use SAM (only for clinical_grade and ultimate)
            use_sam = classifier_type in ['clinical_grade', 'ultimate']
            
            # Train classifier - ALL HAVE SAME INTERFACE!
            history = classifier.fit(
                train_loader=train_loader,
                val_loader=val_loader,
                epochs=EPOCHS,
                lr=LR,
                use_sam=use_sam,
                primary_metric=OPTIMIZE_METRIC,
                patience=PATIENCE,
                min_delta=MIN_DELTA_METRIC,
                checkpoint_path = checkpoint_path
            )
            training_histoy.append(history)
            
            # Get best metrics from this fold
            fold_metric_value = classifier.best_metric_value
            
            self.logger.info(
                f"  Fold {fold+1} Best {OPTIMIZE_METRIC.capitalize()}: {fold_metric_value:.4f} | "
                f"Acc: {classifier.best_acc:.2f}% | "
                f"Recall: {classifier.best_recall:.4f}"
            )
            
            fold_metrics.append({
                'fold': fold + 1,
                f'val_{OPTIMIZE_METRIC}': fold_metric_value,
                'val_acc': classifier.best_acc,
                'val_recall': classifier.best_recall,
                'val_f1': classifier.best_f1
            })
            
            # Track best fold
            if fold_metric_value > best_fold_metric:
                best_fold = fold
                best_fold_metric = fold_metric_value
                # Save best fold checkpoint
                classifier.save(fold_checkpoint_path)
                self.logger.info(f"  ✓ Best fold so far! Checkpoint saved.")
            
            # Cleanup
            del classifier, train_loader, val_loader
            torch.cuda.empty_cache()
            gc.collect()
            self.logger.info(f"Fold {fold + 1} cleanup completed")
        
        # Aggregate fold results
        fold_df = pd.DataFrame(fold_metrics)
        aggregate_fold_results = {
            f'mean_fold_{OPTIMIZE_METRIC}': fold_df[f'val_{OPTIMIZE_METRIC}'].mean(),
            f'std_fold_{OPTIMIZE_METRIC}': fold_df[f'val_{OPTIMIZE_METRIC}'].std(),
            'mean_fold_acc': fold_df['val_acc'].mean(),
            'std_fold_acc': fold_df['val_acc'].std(),
            'mean_fold_recall': fold_df['val_recall'].mean(),
            'mean_fold_f1': fold_df['val_f1'].mean(),
        }
        
        self.logger.info(f"\n  K-Fold Summary:")
        self.logger.info(f"    Mean {OPTIMIZE_METRIC.capitalize()}: {aggregate_fold_results[f'mean_fold_{OPTIMIZE_METRIC}']:.4f} "
                        f"± {aggregate_fold_results[f'std_fold_{OPTIMIZE_METRIC}']:.4f}")
        self.logger.info(f"    Best Fold: {best_fold+1} ({best_fold_metric:.4f})")
        
        # Test on held-out test set using best fold model
        self.logger.info(f"\n  Loading best fold model for final evaluation...")
        
        test_subset = Subset(full_dataset, test_indices)
        test_loader = DataLoader(
            test_subset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=PIN_MEMORY,
            persistent_workers=PERSISTENT_WORKERS if NUM_WORKERS > 0 else False
        )
        
        # Load best fold checkpoint
        eval_classifier : BaseClassifier = classifier_class(
            model_name=model_name,
            num_classes=len(classes),
            device=DEVICE
        )
        eval_classifier.load(checkpoint_path)
        # eval_classifier.load(fold_checkpoint_path)
        
        self.logger.info(f"\n  Final evaluation on test set...")
        metrics = test_model(
            model_name=model_name,
            model=eval_classifier,
            loader=test_loader,
            classes=classes,
            experiment_name=experiment_name,
            history=training_histoy,
            logger=self.logger,
            use_tta=False
        )
        
        final_accuracy = metrics["accuracy"]
        final_recall = metrics["recall"]
        
        self.logger.info(f"\n  Final Test Results:")
        self.logger.info(f"    Accuracy: {final_accuracy:.2f}%")
        self.logger.info(f"    Recall: {final_recall:.4f}")
        self.logger.info(f"    Precision: {metrics['precision']:.4f}")
        self.logger.info(f"    F1: {metrics['f1']:.4f}")
        
        # Save results in unified format
        results_data = {
            'model_name': [model_name],
            'classifier_type': [classifier_type],
            'optimize_metric': [OPTIMIZE_METRIC],
            'pretrained': [PRETRAINED],
            'best_val_metric': [f"{best_fold_metric:.4f}"],
            'final_test_accuracy': [f"{final_accuracy:.2f}%"],
            'final_test_recall': [f"{final_recall:.4f}"],
            'total_epochs': [EPOCHS],
            'batch_size': [BATCH_SIZE],
            'n_splits': [NFOLDS],
            **aggregate_fold_results
        }
        
        pd.DataFrame(results_data).to_csv(
            self.master_file,
            mode='a',
            header=not os.path.exists(self.master_file),
            index=False
        )
        self.logger.info(f"\nResults appended to {self.master_file}")
        self.logger.info(f"EXPERIMENT FINISHED: {experiment_name.upper()}")
        
        # Cleanup
        del eval_classifier, test_loader, full_dataset
        torch.cuda.empty_cache()
        gc.collect()
