from .models import RECOMMENDED_IMG_SIZES, get_model, get_img_size
from .utils import *

from torchvision import datasets
import torch, torch.nn as nn, torch.optim as optim, os, pandas as pd, numpy as np
from torch.utils.data import DataLoader, Subset
import matplotlib
matplotlib.use('Agg') # GUARANTEED FIX for Tkinter/main thread error
import matplotlib.pyplot as plt
from tqdm import tqdm
import gc
from sklearn.model_selection import train_test_split, StratifiedKFold

from .config import *
from .visualization import Visualizer
from .test import test_model

class Cross_Validator:
    def __init__(self, model_names, logger : Logger, use_aug = False):
        self.model_names = model_names
        self.results = []
        self.use_aug = use_aug
        self.logger = logger
        self.master_file = os.path.join(OUTPUT_DIR, "master_results.csv")
        self.models_dir = os.path.join(OUTPUT_DIR, "best_models")

        os.makedirs(self.models_dir, exist_ok = True)
        self.logger.debug(f"Models getting cross validated -> {self.model_names}")
        self.logger.debug(f"GPU Augmentation used -> {self.use_aug}")
        self.logger.debug(f"Master File -> {self.master_file}")
        self.logger.debug(f"Models Directory -> {self.models_dir}")

    def run(self):
        master_df = None
        if os.path.exists(self.master_file):
            master_df = pd.read_csv(self.master_file)
            self.logger.debug(f"Master df exists -> {self.master_file}")
        
        skf = StratifiedKFold(n_splits=NFOLDS, shuffle=True, random_state=42)
        
        for model_name in self.model_names:
            experiment_name = f"{model_name}_pre={PRETRAINED}_aug={self.use_aug}"
            checkpoint_path = os.path.join(self.models_dir, f"{experiment_name}_best_model.pth")
            self.logger.debug("\n" + "="*80 + f"\nSTARTING EXPERIMENT: {experiment_name.upper()}\n" + "="*80)

            if master_df is not None and not master_df[(master_df['model_name'] == model_name) & (master_df['pretrained'] == PRETRAINED) & (master_df['augmentation'] == self.use_aug)].empty:
                self.logger.info(">> Found completed entry in master results. Skipping.")
                self.logger.info("="*80)
                self.logger.info(master_df[(master_df['model_name'] == model_name) & (master_df['pretrained'] == PRETRAINED) & (master_df['augmentation'] == self.use_aug)].to_string())
                self.logger.info("\n" + "="*80 + "\n" + "="*80 + "\n")
                continue

            
            img_size = get_img_size(model_name)
            self.logger.debug(f"Image size for model {model_name} -> {img_size}")

            base_transform = get_base_transformations(img_size)

            # Load Base Dataset (No transforms yet)
            self.logger.info(f"Loading images from: {DATA_DIR}")
            # ImageFolder loads data in standard format: root/class_name/image.jpg
            full_dataset = FullDataset(DATA_DIR, base_transform)
            targets = np.array(full_dataset.targets) # Get labels for Stratified Split
            classes = full_dataset.classes
            self.logger.debug(f"Classes found -> {classes}")

            train_val_indices, test_indices = train_test_split(
                np.arange(len(targets)),
                test_size=TEST_SPLIT,
                stratify=targets,
                random_state=42
            )
            
            test_subset = Subset(full_dataset, test_indices)
            test_loader = DataLoader(
                test_subset, 
                batch_size=BATCH_SIZE, 
                shuffle=False, 
                num_workers=NUM_WORKERS, 
                pin_memory=True
            )
            self.logger.info(f"Data Split: {len(train_val_indices)} Train/Val samples, {len(test_indices)} Test samples")

            train_val_targets = targets[train_val_indices]

            self.logger.info(f"\n=== Validating Model: {model_name} ===")
            fold_metrics = [] 
            
            for fold, (fold_train_idx_rel, fold_val_idx_rel) in enumerate(skf.split(train_val_indices, train_val_targets)):
                self.logger.info(f"  [Fold {fold+1}/{NFOLDS}]  ")
                
                train_idx = train_val_indices[fold_train_idx_rel]
                val_idx = train_val_indices[fold_val_idx_rel]
                
                train_ds = Subset(full_dataset, train_idx)
                val_ds = Subset(full_dataset, val_idx)

                gpu_augmenter = get_gpu_augmentations(img_size) if self.use_aug else None
                
                train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
                val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

                # Model Setup (The >80% Strategy)
                try:
                    model = get_model(model_name, num_classes=len(classes), pretrained=PRETRAINED)
                    model = model.to(DEVICE)
                except Exception as e:
                    self.logger.error(f"FAILED to load {model_name}: {e}")
                    break

                # Optimizer Setup
                optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
                # optimizer = optim.AdamW([
                #     {'params': model.get_classifier().parameters(), 'lr': 1e-3},
                #     {'params': model.features.parameters(), 'lr': 1e-5} # or model.backbone
                # ], weight_decay=0.01)
                criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
                scaler = torch.amp.GradScaler(enabled=(DEVICE == 'cuda'))
                # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
                # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.2, patience=3, verbose=True)
                scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-3, epochs=EPOCHS, steps_per_epoch=len(train_loader), 
                            pct_start=0.3,                   # 30% of time spent warming up
                            div_factor=25.0,                 # Initial LR = max_lr / 25
                            final_div_factor=100.0           # Final LR = Initial LR / 100
                            )
                self.logger.debug(f"Using scaler -> {scaler}")
                self.logger.debug(f"Using scheduler -> {scheduler}")

                best_acc, training_history = 0.0, []
                best_stats = {}
                
                for epoch in range(EPOCHS):
                    # Train
                    t_loss, t_acc = train_one_epoch(
                        model, train_loader, criterion, optimizer, scaler, gpu_augmenter, scheduler
                    )
                    # Validate
                    v_loss, v_acc, v_prec, v_rec, v_f1 = validate_one_epoch(
                        model, val_loader, criterion
                    )
                    
                    # scheduler.step()
                    training_history.append({'epoch': epoch, 'train_acc': t_acc, 'train_loss': t_loss, 'val_acc': v_acc, 'val_loss': v_loss, 'val_prec' : v_prec, 'val_rec' : v_rec, 'val_f1' : v_f1})

                    if v_acc > best_acc:
                        self.logger.debug(f"[{v_acc}] Best model uptil now for model {model_name}")
                        best_acc = v_acc
                        best_stats = {
                            'val_acc': v_acc, 'val_loss': v_loss,
                            'val_prec': v_prec, 'val_rec': v_rec, 'val_f1': v_f1,
                            'train_acc': t_acc, 'train_loss': t_loss
                        }
                        torch.save(model.state_dict(), checkpoint_path)
                
                self.logger.info(f"-> Best Val Accuracy: {best_acc:.2f}% | F1: {best_stats['val_f1']:.4f} | Precision: {best_stats['val_prec']:.4f} | Recall: {best_stats['val_rec']:.4f}")

                
                fold_metrics.append({
                    'fold': fold + 1,
                    **best_stats
                })

                # Cleanup
                del model, optimizer, scheduler, scaler, train_loader, val_loader
                torch.cuda.empty_cache()
                gc.collect()
                self.logger.info(f"Cleanup for fold {fold + 1} done")
                self.logger.info("Training histoy plotted")

            visualizer = Visualizer(experiment_name = experiment_name, model_name = model_name, class_names = classes, transform = base_transform, logger = self.logger)
            # We load the weights into a fresh, non-pretrained model instance
            self.logger.info(f"Loading the best model for {model_name} from {checkpoint_path}")
            eval_model = get_model(model_name, num_classes=len(classes), pretrained=False).to(DEVICE)
            # Use weights_only=True for security, as recommended by the warning message
            eval_model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE, weights_only=True))
            metrics = test_model(model_name, eval_model, test_loader, classes, experiment_name, training_history, self.logger, visualizer)

            final_accuracy = metrics["accuracy"]
            self.logger.info(f"[{final_accuracy}] Test accuracy")
            
            aggregate_fold_results = {}
            # Aggregate Results
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
            pd.DataFrame(results_data).to_csv(self.master_file, mode='a', header=not os.path.exists(self.master_file), index=False)
            self.logger.info(f"Results appended to {self.master_file}\nEXPERIMENT FINISHED: {experiment_name.upper()}")

            del full_dataset, test_loader, eval_model
            torch.cuda.empty_cache()
            gc.collect()
        
        zip_and_empty(OUTPUT_DIR, "result.zip")

        self.logger.info(f"\n>>> Batch Complete.")