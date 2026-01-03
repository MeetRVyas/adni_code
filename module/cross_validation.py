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
from .progressive_finetuning import ProgressiveFineTuner


class Cross_Validator:
    """
    Handles k-fold cross-validation for multiple model architectures
    
    Key features:
    - Stratified k-fold splitting to maintain class distribution
    - GPU memory management between folds
    - Checkpoint saving for best models
    - Comprehensive metric tracking
    """
    def __init__(self, model_names, logger: Logger, use_aug=False, strategy='simple_cosine', enable_ensemble=False):
        self.model_names = model_names
        self.results = []
        self.use_aug = use_aug
        self.strategy_name = strategy
        self.logger = logger
        self.enable_ensemble = enable_ensemble
        self.master_file = os.path.join(RESULTS_DIR, "master_results.csv")
        self.models_dir = os.path.join(RESULTS_DIR, "best_models")

        # Ensemble tracking
        self.ensemble_predictions = []  # List of (model_name, y_pred, y_prob)
        self.test_labels = None  # Shared test labels

        os.makedirs(self.models_dir, exist_ok = True)
        self.logger.debug(f"Models for cross-validation: {self.model_names}")
        self.logger.debug(f"GPU Augmentation: {self.use_aug}")
        self.logger.debug(f"Training Strategy: {self.strategy_name}")
        self.logger.debug(f"Ensemble mode: {self.enable_ensemble}")
        self.logger.debug(f"Master results file: {self.master_file}")

    def run(self):
        """Main execution loop for cross-validation across all models"""
        master_df = None
        if os.path.exists(self.master_file):
            master_df = pd.read_csv(self.master_file)
            self.logger.debug(f"Master df exists -> {self.master_file}")
        
        # Load dataset ONCE (file paths only)
        self.logger.info("\n" + "="*80)
        self.logger.info("LOADING DATASET (ONCE FOR ALL MODELS)")
        self.logger.info("="*80)
        self.logger.info(f"Loading dataset from: {DATA_DIR}")
        
        # Load with minimal transform (just to get file paths and labels)
        temp_transform = transforms.Compose([transforms.ToTensor()])
        base_dataset = FullDataset(DATA_DIR, temp_transform)

        training_history = []
        
        targets = np.array(base_dataset.targets)
        classes = base_dataset.classes
        file_paths = [sample[0] for sample in base_dataset.samples]  # Extract file paths
        
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
            random_state=42  # ← Deterministic split
        )
        
        self.logger.info(f"Data split: {len(train_val_indices)} train/val, {len(test_indices)} test")
        self.logger.info("✅ All models will use the SAME test set (enables ensembling)")
        
        # Store test labels for ensemble
        self.test_labels = targets[test_indices]
        
        train_val_targets = targets[train_val_indices]
        
        skf = StratifiedKFold(n_splits=NFOLDS, shuffle=True, random_state=42)
        
        # Train each model with model-specific transforms
        for model_name in self.model_names:
            experiment_name = f"{model_name}_pre={PRETRAINED}_aug={self.use_aug}_strat={self.strategy_name}"
            checkpoint_path = os.path.join(self.models_dir, f"{experiment_name}_best_model.pth")
            
            self.logger.info("\n" + "="*80)
            self.logger.info(f"STARTING EXPERIMENT: {experiment_name.upper()}")
            self.logger.info("="*80)

            # Skip if already completed
            if master_df is not None:
                existing = master_df[
                    (master_df['model_name'] == model_name) &
                    (master_df['pretrained'] == PRETRAINED) &
                    (master_df['augmentation'] == self.use_aug) &
                    (master_df['strategy'] == self.strategy_name)
                ]
                if not existing.empty:
                    self.logger.info(">> Experiment already completed. Skipping.")
                    self.logger.info(existing.to_string())
                    self.logger.info("="*80)
                    continue

            img_size = get_img_size(model_name)
            self.logger.debug(f"Image size for {model_name}: {img_size}")

            model_transform = get_base_transformations(img_size)

            # Load dataset
            self.logger.info(f"Loading dataset from: {DATA_DIR}")
            full_dataset = FullDataset(DATA_DIR, model_transform)

            self.logger.info(f"\n=== Cross-validating: {model_name} ===")
            fold_metrics = []

            self.logger.info("Initializing Fine tuner")

            finetuner = ProgressiveFineTuner(model_name, classes, self.logger, checkpoint_path)
            
            for fold, (fold_train_idx_rel, fold_val_idx_rel) in enumerate(skf.split(train_val_indices, train_val_targets)):
                self.logger.info(f"  [Fold {fold+1}/{NFOLDS}]")

                self.logger.info(f"Making Datasets{', Data Loaders and GPU Augmenter' if self.use_aug else ' and Data Loaders'}")
                
                train_idx = train_val_indices[fold_train_idx_rel]
                val_idx = train_val_indices[fold_val_idx_rel]
                
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

                history, best_stats = finetuner.fit(train_loader, val_loader)
                training_history.append(history)
                
                self.logger.info(
                    f"-> Best Val Acc: {best_stats.get('val_acc', 0):.2f}% | "
                    f"F1: {best_stats.get('val_f1', 0):.4f} | "
                    f"Precision: {best_stats.get('val_prec', 0):.4f} | "
                    f"Recall: {best_stats.get('val_rec', 0):.4f}"
                )

                fold_metrics.append({
                    'fold': fold + 1,
                    **best_stats
                })

                # Cleanup
                del train_loader, val_loader
                torch.cuda.empty_cache()
                gc.collect()
                self.logger.info(f"Fold {fold + 1} cleanup completed")
            
            test_subset = Subset(full_dataset, test_indices)
            test_loader = DataLoader(
                test_subset, 
                batch_size=BATCH_SIZE, 
                shuffle=False, 
                num_workers=NUM_WORKERS, 
                pin_memory=PIN_MEMORY,
                persistent_workers=PERSISTENT_WORKERS if NUM_WORKERS > 0 else False
            )
            
            # Post-training evaluation on test set
            visualizer = Visualizer(
                experiment_name=experiment_name, 
                model_name=model_name, 
                class_names=classes, 
                transform=model_transform, 
                logger=self.logger
            )
            
            self.logger.info(f"Loading best model from {checkpoint_path}")
            eval_model = get_model(model_name, num_classes=len(classes), pretrained=False).to(DEVICE)
            eval_model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE, weights_only=True))
            
            metrics = test_model(
                model_name, eval_model, test_loader, classes, 
                experiment_name, training_history, self.logger, visualizer
            )

            final_accuracy = metrics["accuracy"]
            self.logger.info(f"[{final_accuracy:.2f}%] Final test accuracy")

            # Save predictions for ensemble
            if self.enable_ensemble:
                self.ensemble_predictions.append({
                    'model_name': model_name,
                    'y_pred': metrics['y_pred'],  # Predicted labels
                    'y_prob': metrics['y_prob'],  # Predicted probabilities
                    'accuracy': final_accuracy
                })
                self.logger.info(f"✅ Predictions saved for ensemble ({len(self.ensemble_predictions)}/{len(self.model_names)})")
            
            # Aggregate fold results
            aggregate_fold_results = {}
            if fold_metrics:
                df = pd.DataFrame(fold_metrics)
                aggregate_fold_results = {
                    'mean_fold_loss': df['val_loss'].mean(),
                    'mean_fold_acc': df['val_acc'].mean(),
                    'std_fold_acc': df['val_acc'].std(),
                    'mean_fold_prec': df['val_prec'].mean(),
                    'mean_fold_rec': df['val_rec'].mean(),
                    'mean_fold_f1': df['val_f1'].mean(),
                }

            # Save results
            results_data = {
                'model_name': [model_name],
                'pretrained': [PRETRAINED],
                'augmentation': [self.use_aug],
                'strategy': [self.strategy_name],
                'best_val_accuracy': [f"{best_stats.get('val_acc', 0):.2f}%"],
                'final_test_accuracy': [f"{final_accuracy:.2f}%"],
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
            self.logger.info(f"Results appended to {self.master_file}")
            self.logger.info(f"EXPERIMENT FINISHED: {experiment_name.upper()}")

            # Final cleanup
            del full_dataset, eval_model, test_loader, finetuner
            torch.cuda.empty_cache()
            gc.collect()
        
         # ENSEMBLE: Combine predictions from all models
        if self.enable_ensemble and len(self.ensemble_predictions) > 1:
            self.logger.info("\n" + "="*80)
            self.logger.info("RUNNING ENSEMBLE PREDICTION")
            self.logger.info("="*80)
            self._run_ensemble(classes)
        
        self.logger.info("\n>>> Batch Complete.")
    
    def _run_ensemble(self, classes):
        """Ensemble predictions from multiple models."""
        self.logger.info(f"Ensembling {len(self.ensemble_predictions)} models:")
        for pred in self.ensemble_predictions:
            self.logger.info(f"  - {pred['model_name']}: {pred['accuracy']:.2f}%")
        
        # Average probabilities
        all_probs = np.array([pred['y_prob'] for pred in self.ensemble_predictions])
        ensemble_probs = all_probs.mean(axis=0)
        ensemble_preds = np.argmax(ensemble_probs, axis=1)
        
        # Calculate ensemble accuracy
        from sklearn.metrics import accuracy_score, classification_report
        ensemble_accuracy = accuracy_score(self.test_labels, ensemble_preds) * 100
        
        self.logger.info("\n" + "="*80)
        self.logger.info(f"🏆 ENSEMBLE ACCURACY: {ensemble_accuracy:.2f}%")
        self.logger.info("="*80)
        
        # Detailed report
        self.logger.info("\nEnsemble Classification Report:")
        report = classification_report(self.test_labels, ensemble_preds, target_names=classes, digits=4)
        self.logger.info("\n" + report)
        
        # Save ensemble results
        ensemble_results = {
            'model_name': ['ENSEMBLE'],
            'pretrained': [PRETRAINED],
            'augmentation': [self.use_aug],
            'strategy': [self.strategy_name],
            'num_models': [len(self.ensemble_predictions)],
            'final_test_accuracy': [f"{ensemble_accuracy:.2f}%"],
            'individual_accuracies': [', '.join([f"{p['accuracy']:.2f}%" for p in self.ensemble_predictions])],
        }
        
        pd.DataFrame(ensemble_results).to_csv(
            self.master_file, mode='a', header=False, index=False
        )
        
        self.logger.info(f"✅ Ensemble results saved to {self.master_file}")