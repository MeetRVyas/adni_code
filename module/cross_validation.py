from .models import get_model, get_img_size
from .utils import *
from .config import *
from .visualization import Visualizer
from .test import test_model

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


class Cross_Validator:
    """
    Handles k-fold cross-validation for multiple model architectures
    
    Key features:
    - Stratified k-fold splitting to maintain class distribution
    - GPU memory management between folds
    - Checkpoint saving for best models
    - Comprehensive metric tracking
    """
    
    def __init__(self, model_names, logger: Logger, use_aug=False):
        self.model_names = model_names
        self.results = []
        self.use_aug = use_aug
        self.logger = logger
        self.master_file = os.path.join(RESULTS_DIR, "master_results.csv")
        self.models_dir = os.path.join(RESULTS_DIR, "best_models")

        os.makedirs(self.models_dir, exist_ok = True)
        self.logger.debug(f"Models for cross-validation: {self.model_names}")
        self.logger.debug(f"GPU Augmentation: {self.use_aug}")
        self.logger.debug(f"Master results file: {self.master_file}")

    def run(self):
        """Main execution loop for cross-validation across all models"""
        
        master_df = None
        if os.path.exists(self.master_file):
            master_df = pd.read_csv(self.master_file)
            self.logger.debug(f"Loaded existing results from {self.master_file}")
        
        skf = StratifiedKFold(n_splits=NFOLDS, shuffle=True, random_state=42)
        
        for model_name in self.model_names:
            experiment_name = f"{model_name}_pre={PRETRAINED}_aug={self.use_aug}"
            checkpoint_path = os.path.join(self.models_dir, f"{experiment_name}_best_model.pth")
            
            self.logger.info("\n" + "="*80)
            self.logger.info(f"STARTING EXPERIMENT: {experiment_name.upper()}")
            self.logger.info("="*80)

            # Skip if already completed
            if master_df is not None:
                existing = master_df[
                    (master_df['model_name'] == model_name) & 
                    (master_df['pretrained'] == PRETRAINED) & 
                    (master_df['augmentation'] == self.use_aug)
                ]
                if not existing.empty:
                    self.logger.info(">> Experiment already completed. Skipping.")
                    self.logger.info(existing.to_string())
                    self.logger.info("="*80)
                    continue

            img_size = get_img_size(model_name)
            self.logger.debug(f"Image size for {model_name}: {img_size}")

            base_transform = get_base_transformations(img_size)

            # Load dataset
            self.logger.info(f"Loading dataset from: {DATA_DIR}")
            full_dataset = FullDataset(DATA_DIR, base_transform)
            targets = np.array(full_dataset.targets)
            classes = full_dataset.classes
            self.logger.debug(f"Classes found: {classes}")

            class_counts = np.bincount(targets)
            total_samples = len(targets)
            class_weights = total_samples / (len(classes) * class_counts)
            class_weights_tensor = torch.FloatTensor(class_weights).to(DEVICE)
            
            self.logger.info("="*80)
            self.logger.info("CLASS DISTRIBUTION & WEIGHTS:")
            for i, cls in enumerate(classes):
                self.logger.info(f"  {cls:<20}: {class_counts[i]:>4} samples ({100*class_counts[i]/total_samples:>5.1f}%) | Weight: {class_weights[i]:.3f}")
            self.logger.info("="*80)

            # Stratified train-test split
            train_val_indices, test_indices = train_test_split(
                np.arange(len(targets)),
                test_size=TEST_SPLIT,
                stratify=targets,
                random_state=42
            )
            
            self.logger.info(f"Data split: {len(train_val_indices)} train/val, {len(test_indices)} test")

            train_val_targets = targets[train_val_indices]

            self.logger.info(f"\n=== Cross-validating: {model_name} ===")
            fold_metrics = []
            
            for fold, (fold_train_idx_rel, fold_val_idx_rel) in enumerate(skf.split(train_val_indices, train_val_targets)):
                self.logger.info(f"  [Fold {fold+1}/{NFOLDS}]")

                self.logger.info(f"Making Datasets{', Data Loaders and GPU Augmenter' if self.use_aug else ' and Data Loaders'}")
                
                train_idx = train_val_indices[fold_train_idx_rel]
                val_idx = train_val_indices[fold_val_idx_rel]
                
                train_ds = Subset(full_dataset, train_idx)
                val_ds = Subset(full_dataset, val_idx)

                # GPU augmentation
                gpu_augmenter = get_gpu_augmentations(img_size) if self.use_aug else None
                
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

                self.logger.info("Initializing Model, Optimizer, Scaler and Scheduler")

                # Model Setup
                try:
                    model = get_model(model_name, num_classes=len(classes), pretrained=PRETRAINED)
                    model = model.to(DEVICE)
                except Exception as e:
                    self.logger.error(f"Failed to load {model_name}: {e}")
                    break

                # Use different learning rates for backbone vs classifier
                # Backbone (pretrained features): very low LR to preserve learned features
                # Classifier (new head): higher LR to adapt to new task
                
                # Get model parameters - this is architecture-agnostic
                if hasattr(model, 'get_classifier'):
                    # TIMM models
                    classifier_params = model.get_classifier().parameters()
                    backbone_params = [p for n, p in model.named_parameters() if 'classifier' not in n and 'fc' not in n and 'head' not in n]
                elif hasattr(model, 'fc'):
                    # ResNet, EfficientNet
                    classifier_params = model.fc.parameters()
                    backbone_params = [p for n, p in model.named_parameters() if 'fc' not in n]
                elif hasattr(model, 'classifier'):
                    # MobileNet, VGG
                    classifier_params = model.classifier.parameters()
                    backbone_params = [p for n, p in model.named_parameters() if 'classifier' not in n]
                elif hasattr(model, 'head'):
                    # ViT, Swin
                    classifier_params = model.head.parameters()
                    backbone_params = [p for n, p in model.named_parameters() if 'head' not in n]
                else:
                    # Fallback: treat all parameters equally
                    self.logger.warning(f"Could not separate backbone/classifier for {model_name}. Using uniform LR.")
                    classifier_params = model.parameters()
                    backbone_params = []
                
                # Optimizer with discriminative learning rates
                if backbone_params:
                    optimizer = optim.AdamW([
                        {'params': classifier_params, 'lr': 5e-4},  # Classifier: moderate LR
                        {'params': backbone_params, 'lr': 1e-5}     # Backbone: very low LR (20x lower)
                    ], weight_decay=0.01)
                    max_lr_scheduler = 5e-4  # For OneCycleLR
                else:
                    optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)
                    max_lr_scheduler = 5e-4
                
                criterion = nn.CrossEntropyLoss(
                    weight=class_weights_tensor,
                    label_smoothing=0.0  # Disabled for class imbalance (was 0.1)
                )
                
                scaler = torch.amp.GradScaler(enabled=USE_AMP)
                
                # OneCycleLR requires per-batch stepping
                scheduler = optim.lr_scheduler.OneCycleLR(
                    optimizer, 
                    max_lr=max_lr_scheduler, 
                    epochs=EPOCHS, 
                    steps_per_epoch=len(train_loader), 
                    pct_start=0.2,           # 20% warmup (was 30%)
                    div_factor=10.0,         # Initial LR = max_lr / 10 (was 25)
                    final_div_factor=100.0   # Final LR = Initial LR / 100
                )
                
                self.logger.debug(f"Using optimizer: AdamW with discriminative LR (classifier: 5e-4, backbone: 1e-5)")
                self.logger.debug(f"Using criterion: CrossEntropyLoss with class weights")
                self.logger.debug(f"Using scaler: {scaler}")
                self.logger.debug(f"Using scheduler: {scheduler}")

                best_acc = 0.0
                training_history = []
                best_stats = {}
                patience = 5
                patience_counter = 0
                min_delta = 0.5  # Minimum improvement of 0.5% to reset patience
                step = EPOCHS / 20
                curr = step

                self.logger.info("Starting training and validation epochs")
                print("\t\tProcessing [", end = "")
                
                for epoch in range(EPOCHS):
                    # Train
                    t_loss, t_acc = train_one_epoch(
                        model, train_loader, criterion, optimizer, scaler, gpu_augmenter, scheduler
                    )
                    # Validate
                    v_loss, v_acc, v_prec, v_rec, v_f1 = validate_one_epoch(
                        model, val_loader, criterion
                    )
                    
                    training_history.append({
                        'epoch': epoch, 
                        'train_acc': t_acc, 
                        'train_loss': t_loss, 
                        'val_acc': v_acc, 
                        'val_loss': v_loss, 
                        'val_prec': v_prec, 
                        'val_rec': v_rec, 
                        'val_f1': v_f1
                    })

                    # Check for improvement
                    if v_acc > best_acc + min_delta:
                        self.logger.debug(f"[Epoch {epoch+1}] New best validation accuracy: {v_acc:.2f}% (improved by {v_acc - best_acc:.2f}%)")
                        best_acc = v_acc
                        best_stats = {
                            'val_acc': v_acc, 'val_loss': v_loss,
                            'val_prec': v_prec, 'val_rec': v_rec, 'val_f1': v_f1,
                            'train_acc': t_acc, 'train_loss': t_loss
                        }
                        torch.save(model.state_dict(), checkpoint_path)
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        if patience_counter >= patience:
                            self.logger.info(f"[Epoch {epoch+1}] Early stopping triggered (no improvement for {patience} epochs)")
                            print("#" * int(20 - curr // step + 1), end = "")
                            break
                    
                    # Periodic GPU cache clearing
                    if (epoch + 1) % EMPTY_CACHE_FREQUENCY == 0:
                        torch.cuda.empty_cache()
                    
                    if epoch >= curr :
                        print("#", end = "")
                        curr += step
                print("]")
                
                self.logger.info(
                    f"-> Best Val Acc: {best_acc:.2f}% | "
                    f"F1: {best_stats['val_f1']:.4f} | "
                    f"Precision: {best_stats['val_prec']:.4f} | "
                    f"Recall: {best_stats['val_rec']:.4f}"
                )

                fold_metrics.append({
                    'fold': fold + 1,
                    **best_stats
                })

                # Cleanup
                del model, optimizer, scheduler, scaler, train_loader, val_loader
                if gpu_augmenter is not None:
                    del gpu_augmenter
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
                transform=base_transform, 
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
                'best_val_accuracy': [f"{best_acc:.2f}%"],
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
            del eval_model, test_loader
            torch.cuda.empty_cache()
            gc.collect()
        
        self.logger.info("\n>>> Batch Complete.")