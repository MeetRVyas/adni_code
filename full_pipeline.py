import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from PIL import Image
import logging
import shutil
import torch
import timm
import torch
from torchvision import transforms, datasets
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from itertools import cycle
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, average_precision_score,
    confusion_matrix, classification_report, roc_auc_score,
    accuracy_score, precision_recall_fscore_support,
    precision_score, recall_score, f1_score,
    cohen_kappa_score, matthews_corrcoef, jaccard_score,
)
from abc import ABC, abstractmethod
from typing import Dict, Optional, List, Tuple
from tqdm import tqdm
import copy
import time
from pathlib import Path
from torch.utils.data import DataLoader, Subset
import gc
from sklearn.model_selection import train_test_split, StratifiedKFold
import sys


# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Directory structure
OUTPUT_DIR = "output"
RESULTS_DIR = os.path.join(OUTPUT_DIR, "results")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
REPORTS_DIR = os.path.join(RESULTS_DIR, "reports")
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")
DATA_DIR = "OriginalDataset"

# Classifier Configuration
OPTIMIZE_METRIC = 'recall'  # Primary metric: 'recall', 'accuracy', 'f1', 'precision'
MIN_DELTA_METRIC = 0.001  # Minimum improvement threshold for early stopping

# Training hyperparameters
EPOCHS = 30  # Total epochs (will be distributed in progressive training)
NFOLDS = 5
BATCH_SIZE = 32
NUM_WORKERS = 4
PRETRAINED = True
NUM_SAMPLES_TO_ANALYSE = 5  # For GradCAM/XAI visualization
TEST_SPLIT = 0.2
PATIENCE = 10
MIN_DELTA = 0.3  # For legacy compatibility
LR = 1e-4

# Optimization settings
USE_AMP = True  # Automatic Mixed Precision
PIN_MEMORY = True
PERSISTENT_WORKERS = True

# Memory management
EMPTY_CACHE_FREQUENCY = 1
SAVE_BEST_ONLY = True

# Timeout settings (in seconds)
SUBPROCESS_TIMEOUT = 8 * 3600

# Create necessary directories
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)


RECOMMENDED_IMG_SIZES = {
    "tf_efficientnet_b4": 380, 
    "tf_efficientnetv2_s": 300, 
    "inception_v3": 299, 
    "xception": 299,
    "vit_base_patch16_224": 224, 
    "swin_base_patch4_window7_224": 224, 
    "convnext_small": 224,
    "convnext_tiny": 224, 
    "maxvit_tiny_224": 224, 
    "resnet50": 224, 
    "resnext50_32x4d": 224,
    "densenet121": 224, 
    "coatnet_0_rw_224": 224, 
    "resnet18": 224, 
    "vgg16_bn": 224, 
    "efficientnet_b0": 224,
    "mobilenetv3_large_100": 224, 
    "vit_tiny_patch16_224": 224, 
    "poolformer_s12": 224,
    "efficientformer_l1": 224,
    "mobilevit_s": 224,
    "ghostnet_100": 224
}


def get_img_size(model_name):
    base_name = model_name.split(".")[0]
    return RECOMMENDED_IMG_SIZES.get(base_name, 224)


def get_model(model_name, num_classes, pretrained=True):
    try:
        model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
        
        # Disable auxiliary classifier for Inception v3 during training
        if "inception" in model_name and pretrained:
            model.aux_logits = False
            
        print(f"Loaded model: {model_name} | Pretrained: {pretrained} | Classes: {num_classes}")
        return model
        
    except Exception as e:
        print(f"Error loading model '{model_name}': {e}")
        raise e


class FullDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform):
        from torchvision.datasets import ImageFolder
        self.transform = transform
        data = ImageFolder(root=data_dir)
        self.samples = data.samples
        self.targets = data.targets
        self.classes = data.classes

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label


class Logger:
    def __init__(self, name: str = "Logger", file_name: str = "batch"):
        self.logger = logging.getLogger(name)
        self.current_log_dir = os.path.join(LOG_DIR, name)
        os.makedirs(self.current_log_dir, exist_ok=True)
        
        if not self.logger.handlers:
            self.logger.setLevel(logging.DEBUG)
            
            formatter = logging.Formatter("[%(levelname)s] %(asctime)s - %(name)s - %(message)s")
            console_formatter = logging.Formatter("[%(name)s] %(message)s")

            base_file_name = os.path.join(self.current_log_dir, file_name)

            info_path = f"{base_file_name}_debug.log"
            info_handler = logging.FileHandler(info_path, encoding="utf-8")
            info_handler.setLevel(logging.DEBUG)
            info_handler.setFormatter(formatter)
            
            error_path = f"{base_file_name}_error.log"
            error_handler = logging.FileHandler(error_path)
            error_handler.setLevel(logging.ERROR)
            error_handler.setFormatter(formatter)
            
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            console_handler.setFormatter(console_formatter)

            self.logger.addHandler(info_handler)
            self.logger.addHandler(error_handler)
            self.logger.addHandler(console_handler)

    def info(self, message: str) -> None:
        self.logger.info(message)

    def warning(self, message: str) -> None:
        self.logger.warning(message)

    def error(self, message: str, exc_info : bool = True) -> None:
        self.logger.error(message, exc_info)
    
    def debug(self, message: str) -> None:
        self.logger.debug(message)


# Sharpness-Aware Minimization (SAM)
class SAM(torch.optim.Optimizer):
    
    def __init__(self, params, base_optimizer, rho=0.05, **kwargs):
        if isinstance(params, (list, tuple)) and isinstance(params[0], dict):
            param_groups = params
        else:
            param_groups = [{'params': params}]

        defaults = dict(rho=rho, **kwargs)
        super().__init__(param_groups, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
    
    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            
            for p in group["params"]:
                if p.grad is None :
                    continue
                # Save current weights
                self.state[p]["old_p"] = p.data.clone()
                # Ascent step
                e_w = p.grad * scale.to(p)
                p.add_(e_w)
        
        if zero_grad :
            self.zero_grad()
    
    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None :
                    continue
                # Restore original weights
                p.data = self.state[p]["old_p"]
        
        # Now take actual optimizer step with new gradient
        self.base_optimizer.step()
        
        if zero_grad :
            self.zero_grad()
    
    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device
        norm = torch.norm(
            torch.stack([
                p.grad.norm(p=2).to(shared_device)
                for group in self.param_groups for p in group["params"]
                if p.grad is not None
            ]),
            p=2
        )
        return norm


def train_one_epoch(model, loader, criterion, optimizer, scaler, gpu_augmenter=None, scheduler=None):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    for images, labels in loader :
        images, labels = images.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)
        
        if gpu_augmenter:
            images = gpu_augmenter(images)
        
        optimizer.zero_grad(set_to_none=True)
        
        if isinstance(optimizer, SAM) :
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.first_step(zero_grad = True)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.second_step(zero_grad = True)
        else :
            with torch.amp.autocast(device_type = DEVICE, dtype=torch.float16, enabled=(DEVICE == 'cuda')):
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        if scheduler and isinstance(scheduler, (
                    optim.lr_scheduler.OneCycleLR,
                    optim.lr_scheduler.SequentialLR
                )):
            scheduler.step()

        running_loss += loss.detach().item() * images.size(0)
        
        _, predicted = outputs.detach().max(1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    total_loss = running_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds) * 100.0
    
    return total_loss, acc


def validate_one_epoch(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.inference_mode():
        for images, labels in loader :
            images, labels = images.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)
            
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=(DEVICE == 'cuda')):
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    total_loss = running_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds) * 100.0
    prec, rec, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted', zero_division=0)
    
    return total_loss, acc, prec, rec, f1


def get_base_transformations(img_size):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def _aggressive_empty_directory(folder_path):
    if not os.path.exists(folder_path):
        return

    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)

        if (os.path.isfile(item_path) or os.path.islink(item_path)) and not item.endswith(".zip"):
            os.remove(item_path)
        elif os.path.isdir(item_path):
            try:
                shutil.rmtree(item_path)
            except Exception:
                _aggressive_empty_directory(item_path)
                try:
                    os.rmdir(item_path)
                except:
                    pass


def zip_and_empty(source_dir, output_zip):
    import zipfile
    
    if not os.path.exists(source_dir):
        print(f"Directory '{source_dir}' does not exist. Skipping.")
        return

    print(f"📦 Zipping '{source_dir}' to '{output_zip}'...")

    try:
        with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(source_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    parent_dir = os.path.dirname(source_dir)
                    arcname = os.path.relpath(file_path, parent_dir)
                    zipf.write(file_path, arcname=arcname)
                    
        if os.path.exists(output_zip):
            print(f"✅ Zip created successfully: {output_zip}")
            print(f"🗑️ Emptying target folder: {source_dir}")
            _aggressive_empty_directory(source_dir)
            print("✅ Cleanup complete.")
        else:
            print("❌ Zip creation failed. Folder NOT emptied.")

    except Exception as e:
        print(f"❌ An error occurred during process: {e}")


def reshape_transform_swin(tensor, height=7, width=7):
    result = tensor.reshape(tensor.size(0), height, width, tensor.size(2))
    result = result.permute(0, 3, 1, 2)
    return result

class NativeGradCAM:
    def __init__(self, model, target_layer, reshape_transform=None):
        self.model = model.eval()
        self.reshape_transform = reshape_transform
        self.activations = None
        self.gradients = None
        self.hooks = []
        
        self.hooks.append(target_layer.register_forward_hook(self.save_activation))
        self.hooks.append(target_layer.register_full_backward_hook(self.save_gradient))

    def save_activation(self, module, input, output):
        # Clone to avoid inplace modification issues
        self.activations = output.clone() if isinstance(output, torch.Tensor) else output

    def save_gradient(self, module, grad_input, grad_output):
        # Clone to avoid inplace modification issues
        self.gradients = grad_output[0].clone() if isinstance(grad_output[0], torch.Tensor) else grad_output[0]

    def __call__(self, input_tensor):
        self.model.zero_grad()
        
        # Ensure model is in eval mode and disable inplace operations
        self.model.eval()
        
        with torch.enable_grad():
            input_tensor = input_tensor.clone().detach().requires_grad_(True)
            output = self.model(input_tensor)
            pred_index = output.argmax(dim=1)
            score = output[:, pred_index]
            score.backward()
        
        grads = self.gradients
        fmaps = self.activations
        
        if self.reshape_transform:
            grads = self.reshape_transform(grads)
            fmaps = self.reshape_transform(fmaps)
        
        weights = torch.mean(grads, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * fmaps, dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-7)
        
        return cam.detach().cpu().numpy()[0, 0]

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()


def generate_gradcam_plot(model, input_tensor, original_img_np, target_layer, reshape_transform=None, alpha=0.4):
    cam_engine = NativeGradCAM(model, target_layer, reshape_transform)
    
    try:
        import cv2
        heatmap = cam_engine(input_tensor)
        heatmap = cv2.resize(heatmap, (original_img_np.shape[1], original_img_np.shape[0]))
        heatmap_uint8 = np.uint8(255 * heatmap)
        heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        
        if original_img_np.max() <= 1.0:
            original_img_np = np.uint8(255 * original_img_np)
        else:
            original_img_np = np.uint8(original_img_np)
        
        superimposed_img = cv2.addWeighted(original_img_np, 1, heatmap_colored, alpha, 0)
        return superimposed_img
        
    finally:
        cam_engine.remove_hooks()


class Visualizer:
    
    def __init__(self, experiment_name, model_name, class_names, transform=None, logger=None):
        self.experiment_name = experiment_name
        self.model_name = model_name
        self.class_names = class_names
        self.logger = logger
        self.img_size = get_img_size(model_name)
        self.transform = transform or get_base_transformations(self.img_size)
        
        self.save_dir = os.path.join(PLOTS_DIR, experiment_name)
        os.makedirs(self.save_dir, exist_ok=True)

    def log(self, msg):
        if self.logger:
            self.logger.info(msg)
        else:
            print(msg)

    # =========================================================================
    # Plots
    # =========================================================================
    def plot_training_history(self, history):
        if not history:
            return

        epochs = [h['epoch'] + 1 for h in history]
        metrics = ['loss', 'acc', 'f1', 'prec', 'rec']
        titles = ['Cross Entropy Loss', 'Accuracy (%)', 'F1 Score', 'Precision', 'Recall']
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(f"Training Dynamics: {self.experiment_name}", fontsize=16, weight='bold')
        axes = axes.flatten()

        for i, metric in enumerate(metrics):
            ax = axes[i]
            train_key = f'train_{metric}'
            val_key = f'val_{metric}'
            
            if train_key in history[0]:
                train_vals = [h[train_key] for h in history]
                ax.plot(epochs, train_vals, 'o--', label='Train', color='cornflowerblue', linewidth=2)
            
            if val_key in history[0]:
                val_vals = [h[val_key] for h in history]
                ax.plot(epochs, val_vals, 'o-', label='Validation', color='darkorange', linewidth=2)
            
            ax.set_title(titles[i])
            ax.set_xlabel("Epochs")
            ax.grid(True, linestyle='--', alpha=0.6)
            ax.legend()

        axes[-1].axis('off')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        save_path = os.path.join(self.save_dir, "training_history.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_confusion_matrix(self, y_true, y_pred, normalize=False):
        cm = confusion_matrix(y_true, y_pred)
        fmt = 'd'
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2f'

        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues', 
                    xticklabels=self.class_names, yticklabels=self.class_names,
                    square=True, cbar_kws={"shrink": .8})
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.title(f'Confusion Matrix: {self.experiment_name}', fontsize=14)
        
        save_path = os.path.join(self.save_dir, "confusion_matrix.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_classwise_metrics(self, y_true, y_pred):
        report = classification_report(y_true, y_pred, target_names=self.class_names, output_dict=True)
        
        metrics_list = []
        for cls in self.class_names:
            if cls in report:
                metrics_list.append({
                    'Class': cls,
                    'Precision': report[cls]['precision'],
                    'Recall': report[cls]['recall'],
                    'F1-Score': report[cls]['f1-score']
                })
            else:
                self.logger.error(f"Class {cls} not found in classification report\n{report}")
        
        if not metrics_list:
            raise ValueError("No valid class metrics found")
        
        df = pd.DataFrame(metrics_list).set_index('Class')
        
        plt.figure(figsize=(8, len(self.class_names) * 0.8 + 2))
        sns.heatmap(df, annot=True, cmap='RdYlGn', fmt='.3f', vmin=0.0, vmax=1.0, linewidths=1)
        plt.title('Class-wise Performance Metrics', fontsize=14)
        plt.yticks(rotation=0)
        
        save_path = os.path.join(self.save_dir, "classwise_metrics.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_roc_curve(self, y_true, y_prob):
        n_classes = len(self.class_names)
        y_true_bin = pd.get_dummies(y_true).values
        
        # Check for NaN in y_prob
        if np.isnan(y_prob).any():
            self.log(f"⚠️  Warning: NaN values detected in predictions. Replacing with 0.25 (1/n_classes)")
            y_prob = np.nan_to_num(y_prob, nan=1.0/n_classes)
        
        plt.figure(figsize=(10, 8))
        colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green', 'red', 'purple'])

        classes_plotted = 0
        
        for i, color in zip(range(n_classes), colors):
            if i < y_prob.shape[1]:
                try:
                    fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
                    auc_score = auc(fpr, tpr)
                    plt.plot(fpr, tpr, color=color, lw=2, 
                             label=f'ROC of {self.class_names[i]} (AUC = {auc_score:0.2f})')
                    classes_plotted += 1
                except Exception as e:
                    self.log(f"⚠️  Skipping ROC for {self.class_names[i]}: {e}")
                    continue
        
        if classes_plotted == 0 :
            raise ValueError("No ROC curve plotted for any of the classes")
        
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curves: {self.experiment_name}')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        save_path = os.path.join(self.save_dir, "roc_curve.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_precision_recall_curve(self, y_true, y_prob):
        n_classes = len(self.class_names)
        y_true_bin = pd.get_dummies(y_true).values
        
        # Check for NaN in y_prob
        if np.isnan(y_prob).any():
            self.log(f"⚠️  Warning: NaN values detected in predictions. Replacing with 0.25 (1/n_classes)")
            y_prob = np.nan_to_num(y_prob, nan=1.0/n_classes)
    
        plt.figure(figsize=(10, 8))
        colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green', 'red', 'purple'])
        
        classes_plotted = 0
        
        for i, color in zip(range(n_classes), colors):
            if i < y_prob.shape[1]:
                try:
                    prec, rec, _ = precision_recall_curve(y_true_bin[:, i], y_prob[:, i])
                    avg_prec = average_precision_score(y_true_bin[:, i], y_prob[:, i])
                    plt.plot(rec, prec, color=color, lw=2, 
                             label=f'{self.class_names[i]} (AP = {avg_prec:0.2f})')
                    classes_plotted += 1
                except Exception as e:
                    self.log(f"⚠️  Skipping PR curve for {self.class_names[i]}: {e}")
                    continue
        
        if classes_plotted == 0 :
            raise ValueError("No PR curve plotted for any of the classes")
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curves: {self.experiment_name}')
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(self.save_dir, "precision_recall_curve.png"), dpi=300)
        plt.close()

    def plot_confidence_distribution(self, y_true, y_prob):
        y_pred = np.argmax(y_prob, axis=1)
        confidences = np.max(y_prob, axis=1).astype(np.float32)  # Convert to float32
        
        correct_mask = (y_pred == y_true)
        incorrect_mask = ~correct_mask
        
        plt.figure(figsize=(10, 6))
        
        sns.histplot(confidences[correct_mask], color='green', label='Correct Predictions', 
                     kde=True, bins=20, alpha=0.5, element="step")
        
        if np.any(incorrect_mask):
            sns.histplot(confidences[incorrect_mask], color='red', label='Incorrect Predictions', 
                         kde=True, bins=20, alpha=0.5, element="step")
            
        plt.xlabel("Prediction Confidence (Probability)")
        plt.ylabel("Count")
        plt.title(f"Confidence Distribution{' : Correct vs Incorrect' if np.any(incorrect_mask) else ''}")
        plt.legend()
        
        save_path = os.path.join(self.save_dir, "confidence_analysis.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_cumulative_gain(self, y_true, y_prob):
        n_classes = y_prob.shape[1]
        y_true_bin = pd.get_dummies(y_true).values
        percentages = np.arange(1, len(y_true) + 1) / len(y_true)

        plt.figure(figsize=(10, 8))
        plt.plot([0, 1], [0, 1], 'k--', label="Random Model")

        colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green', 'red', 'purple'])

        for i, color in zip(range(n_classes), colors):
            score = y_prob[:, i]
            true_class = y_true_bin[:, i]

            order = np.argsort(score)[::-1]
            true_sorted = true_class[order]
            cum_gains = np.cumsum(true_sorted)
            
            total_positives = np.sum(true_class)
            if total_positives > 0:
                cum_gains = cum_gains / total_positives
            else:
                cum_gains = np.zeros_like(cum_gains)

            plt.plot(percentages, cum_gains, color=color, lw=2, label=f'{self.class_names[i]}')

        plt.xlabel("Percentage of Sample Targeted")
        plt.ylabel("Cumulative Gain")
        plt.title(f"Cumulative Gain Curve: {self.experiment_name}")
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        save_path = os.path.join(self.save_dir, "cumulative_gain.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    # =========================================================================
    # XAI
    # =========================================================================
    def _get_target_layer(self, model):
        try:
            if "swin" in self.model_name:
                return model.layers[-1].blocks[-1].norm2
            elif "resnet" in self.model_name or "resnext" in self.model_name:
                return model.layer4[-1]
            elif "efficientnet" in self.model_name:
                return model.conv_head
            elif "densenet" in self.model_name:
                return model.features.norm5
            elif "convnext" in self.model_name:
                return model.stages[-1].blocks[-1].norm
            elif "mobilenet" in self.model_name:
                # MobileNet has special handling for inplace operations
                return model.conv_head
            else:
                # Recursive fallback to last Conv2d
                for name, module in list(model.named_modules())[::-1]:
                    if isinstance(module, torch.nn.Conv2d):
                        return module
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error determining target layer: {e}")
        return None

    def run_grad_cam(self, model, image_path, target_layer):
        # Disable inplace operations for models like MobileNetV3
        def set_inplace_false(m):
            for attr in dir(m):
                if 'inplace' in attr:
                    try:
                        setattr(m, attr, False)
                    except:
                        pass
        
        model.apply(set_inplace_false)
        model.eval()
        
        pil_img = Image.open(image_path).convert('RGB').resize((self.img_size, self.img_size))
        input_tensor = self.transform(pil_img).unsqueeze(0).to(DEVICE)
        original_img_np = np.array(pil_img)
        
        reshape = reshape_transform_swin if "swin" in str(type(model)).lower() else None
        viz_img = generate_gradcam_plot(model, input_tensor, original_img_np, target_layer, reshape)
        return viz_img

    def run_lime(self, model, image_path):
        from lime import lime_image
        from skimage.segmentation import mark_boundaries
        
        def batch_predict(images):
            pil_images = [Image.fromarray(img.astype('uint8')) for img in images]
            batch = torch.stack([self.transform(img) for img in pil_images], dim=0).to(DEVICE)
            
            with torch.inference_mode():
                model.eval()
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=(DEVICE == 'cuda')):
                    logits = model(batch)
                probs = F.softmax(logits, dim=1).cpu().numpy()
                
                # Handle NaN in predictions
                if np.isnan(probs).any():
                    n_classes = probs.shape[1]
                    probs = np.nan_to_num(probs, nan=1.0/n_classes)
                    # Normalize to ensure sum=1
                    probs = probs / probs.sum(axis=1, keepdims=True)
                
                return probs
        
        image_np = np.array(Image.open(image_path).convert('RGB').resize((self.img_size, self.img_size)))
        explainer = lime_image.LimeImageExplainer()
        
        explanation = explainer.explain_instance(
            image_np, 
            batch_predict, 
            top_labels=1, 
            hide_color=0, 
            num_samples=300
        )
        
        temp, mask = explanation.get_image_and_mask(
            explanation.top_labels[0], 
            positive_only=True, 
            num_features=5, 
            hide_rest=False
        )
        
        lime_viz = mark_boundaries(temp / 255.0, mask)
        return (np.clip(lime_viz, 0, 1) * 255).astype(np.uint8)

    def run_shap(self, model, image_path):
        import shap
        
        image = Image.open(image_path).convert('RGB').resize((self.img_size, self.img_size))
        input_tensor = self.transform(image).unsqueeze(0).to(DEVICE)
        background = torch.randn(5, 3, self.img_size, self.img_size).to(DEVICE)
        
        explainer = shap.GradientExplainer(model, background)
        shap_values = explainer.shap_values(input_tensor)
        
        with torch.inference_mode():
            output = model(input_tensor)
            pred_idx = torch.argmax(output).item()
        
        shap_numpy = np.swapaxes(np.swapaxes(shap_values[pred_idx], 1, -1), 1, 2)
        input_numpy = np.swapaxes(np.swapaxes(input_tensor.cpu().numpy(), 1, -1), 1, 2)
        
        temp_filename = f"temp_shap_{os.getpid()}_{np.random.randint(10000)}.png"
        
        fig = plt.figure(figsize=(4, 4))
        shap.image_plot(shap_numpy, -input_numpy, show=False)
        plt.savefig(temp_filename, bbox_inches='tight', pad_inches=0, dpi=100)
        plt.close(fig)
        
        if os.path.exists(temp_filename):
            shap_img = np.array(Image.open(temp_filename).convert('RGB'))
            os.remove(temp_filename)
            return shap_img
        else:
            raise FileNotFoundError(f"SHAP temp file not created")

    def generate_xai_comparison_plot(self, model, image_path, sample_id):
        model.eval()
        
        original_img = np.array(Image.open(image_path).convert('RGB').resize((self.img_size, self.img_size)))
        image_basename = os.path.splitext(os.path.basename(image_path))[0]
        self.log(f"  Generating XAI comparison for {image_basename}...")
        
        target_layer = self._get_target_layer(model)
        
        # Try each XAI method independently
        xai_results = {}
        
        # GradCAM
        if target_layer is not None:
            try:
                xai_results['gradcam'] = self.run_grad_cam(model, image_path, target_layer)
            except Exception as e:
                if self.logger:
                    self.logger.error(f"GradCAM failed for {image_path}: {e}")
                xai_results['gradcam'] = None
        
        # LIME
        try:
            xai_results['lime'] = self.run_lime(model, image_path)
        except Exception as e:
            if self.logger:
                self.logger.error(f"LIME failed for {image_path}: {e}")
            xai_results['lime'] = None
        
        # SHAP
        try:
            xai_results['shap'] = self.run_shap(model, image_path)
        except Exception as e:
            if self.logger:
                self.logger.error(f"SHAP failed for {image_path}: {e}")
            xai_results['shap'] = None
        
        successful_methods = sum(1 for v in xai_results.values() if v is not None)
        
        # If all methods failed, skip plot
        if successful_methods == 0:
            self.log(f"  ✗ All XAI methods failed for {sample_id}")
            return None
        
        # Create 2x2 comparison plot
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        fig.suptitle(f"XAI Comparison: {os.path.basename(image_path)} ({self.model_name})", fontsize=16, weight='bold')
        
        # [0,0] Original (top-left)
        axes[0, 0].imshow(original_img)
        axes[0, 0].set_title("Original", fontsize=14, color='black')
        axes[0, 0].axis('off')
        
        # [0,1] GradCAM (top-right)
        if xai_results['gradcam'] is not None:
            axes[0, 1].imshow(xai_results['gradcam'])
            axes[0, 1].set_title("GradCAM", fontsize=14, color='green')
        else:
            axes[0, 1].text(0.5, 0.5, "GradCAM\n(Failed)", ha='center', va='center', 
                           fontsize=16, color='red', weight='bold',
                           transform=axes[0, 1].transAxes)
            axes[0, 1].set_facecolor('#ffcccc')
            axes[0, 1].set_title("GradCAM", fontsize=14, color='red')
        axes[0, 1].axis('off')
        
        # [1,0] LIME (bottom-left)
        if xai_results['lime'] is not None:
            axes[1, 0].imshow(xai_results['lime'])
            axes[1, 0].set_title("LIME", fontsize=14, color='green')
        else:
            axes[1, 0].text(0.5, 0.5, "LIME\n(Failed)", ha='center', va='center',
                           fontsize=16, color='red', weight='bold',
                           transform=axes[1, 0].transAxes)
            axes[1, 0].set_facecolor('#ffcccc')
            axes[1, 0].set_title("LIME", fontsize=14, color='red')
        axes[1, 0].axis('off')
        
        # [1,1] SHAP (bottom-right)
        if xai_results['shap'] is not None:
            axes[1, 1].imshow(xai_results['shap'])
            axes[1, 1].set_title("SHAP", fontsize=14, color='green')
        else:
            axes[1, 1].text(0.5, 0.5, "SHAP\n(Failed)", ha='center', va='center',
                           fontsize=16, color='red', weight='bold',
                           transform=axes[1, 1].transAxes)
            axes[1, 1].set_facecolor('#ffcccc')
            axes[1, 1].set_title("SHAP", fontsize=14, color='red')
        axes[1, 1].axis('off')
        
        # 6. Save
        image_basename = os.path.splitext(os.path.basename(image_path))[0]
        output_filename = os.path.join(self.save_dir, f"xai_comparison_{sample_id}_{image_basename}.png")
        plt.savefig(output_filename, bbox_inches='tight', dpi=150)
        plt.close(fig)
        
        return output_filename, successful_methods

    # =========================================================================
    # 4. MASTER GENERATOR
    # =========================================================================
    def generate_all_plots(self, y_true, y_prob, history=None, model=None, test_loader=None, xai_samples=5):
        self.log(f"Generating visualizations for {self.experiment_name}...")
        
        # Check for NaN in y_prob at the start
        if np.isnan(y_prob).any():
            self.log(f"⚠️  WARNING: NaN detected in probability outputs! Model may have collapsed.")
            self.log(f"   This usually happens when the model predicts only one class.")
            self.log(f"   Replacing NaN with uniform distribution (1/n_classes) for visualization.")
        
        y_pred = np.argmax(y_prob, axis=1)
        
        # Results tracking
        results = {
            'xai_success': 0,
            'xai_total': 0,
            'plots': {}
        }

        # XAI Analysis
        if model and test_loader and xai_samples > 0:
            self.log(f"Generating XAI comparisons for {xai_samples} samples...")
            
            dataset = test_loader.dataset
            full_ds = dataset.dataset if hasattr(dataset, 'dataset') else dataset
            
            if hasattr(full_ds, 'samples'):
                indices = np.random.choice(len(dataset), min(xai_samples, len(dataset)), replace=False)
                
                for i, idx in enumerate(indices):
                    img_path = full_ds.samples[idx][0]
                    image_name = os.path.splitext(os.path.basename(img_path))[0]
                    
                    self.log(f"  Generating XAI comparison for {image_name}...")
                    results['xai_total'] += 1
                    
                    try:
                        result = self.generate_xai_comparison_plot(model, img_path, sample_id=f"sample_{i}")
                        if result is not None:
                            output_file, success_count = result
                            results['xai_success'] += 1
                            self.log(f"    ✓ XAI comparison saved ({success_count}/3 methods successful)")
                        else:
                            self.log(f"    ✗ XAI comparison failed (all methods failed)")
                    except Exception as e:
                        self.log(f"    ✗ XAI comparison failed: {e}")
                        if self.logger:
                            self.logger.error(f"XAI error for {img_path}: {e}", exc_info=True)
        
        # Clean up model from memory
        try:
            import gc
            del model, test_loader
            torch.cuda.empty_cache()
            gc.collect()
        except Exception as e:
            self.log(f"Cleanup failed : {e}")
            if self.logger:
                self.logger.error(f"Cleanup of model and test loader failed : {e}", exc_info=True)
        
        plot_functions = [
            ('training_history', lambda: self.plot_training_history(history) if history else None),
            ('confusion_matrix', lambda: self.plot_confusion_matrix(y_true, y_pred, normalize=True)),
            ('classwise_metrics', lambda: self.plot_classwise_metrics(y_true, y_pred)),
            ('roc_curve', lambda: self.plot_roc_curve(y_true, y_prob)),
            ('precision_recall', lambda: self.plot_precision_recall_curve(y_true, y_prob)),
            ('confidence', lambda: self.plot_confidence_distribution(y_true, y_prob)),
            ('cumulative_gain', lambda: self.plot_cumulative_gain(y_true, y_prob))
        ]
        
        # Execute each plot with error handling
        for plot_name, plot_func in plot_functions:
            try:
                plot_func()
                results['plots'][plot_name] = True
                self.log(f"✓ {plot_name.replace('_', ' ').title()} saved")
            except Exception as e:
                results['plots'][plot_name] = False
                self.log(f"✗ {plot_name.replace('_', ' ').title()} failed: {e}")
                if self.logger:
                    self.logger.error(f"{plot_name} error: {e}", exc_info=True)
                plt.close('all')
        
        
        # Summary
        successful_plots = sum(results['plots'].values())
        total_plots = len(results['plots'])
        
        self.log(f"✅ Visualization complete: {successful_plots}/{total_plots} plots successful")
        if results['xai_total'] > 0:
            self.log(f"   XAI: {results['xai_success']}/{results['xai_total']} samples")
        self.log(f"   All outputs saved to {self.save_dir}")
        
        return results


class BaseClassifier(ABC):
    
    def __init__(self, model_name: str, num_classes: int = 4, device: str = 'cuda', checkpoint_path : str = None):
        self.model_name = model_name
        self.num_classes = num_classes
        self.device = device
        # checkpoint_path: Path to store checkpoints (best model weights)
        self.checkpoint_path = checkpoint_path

        
        # Model (built by subclass)
        self.model = None
        
        # Training state
        self.best_metric_value = 0.0
        self.best_recall = 0.0
        self.best_acc = 0.0
        self.best_f1 = 0.0
        self.best_epoch = 0
        self.history = []
        
        # Build model
        self.build_model()
        
        if self.model is None:
            raise ValueError("build_model() must set self.model")
        
        self.model = self.model.to(device)
    
    @abstractmethod
    def build_model(self):
        pass
    
    @abstractmethod
    def forward(self, images: torch.Tensor):
        pass
    
    @abstractmethod
    def compute_loss(self, outputs, labels) -> torch.Tensor:
        pass
    
    def get_predictions(self, outputs) -> torch.Tensor:
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        return torch.argmax(outputs, dim=1)
    
    def _get_metric_value(self, labels: List, preds: List, metric: str) -> float:
        if metric == 'recall':
            return recall_score(labels, preds, average='macro', zero_division=0)
        elif metric == 'accuracy':
            return accuracy_score(labels, preds) * 100
        elif metric == 'f1':
            return f1_score(labels, preds, average='macro', zero_division=0)
        elif metric == 'precision':
            return precision_score(labels, preds, average='macro', zero_division=0)
        else:
            raise ValueError(f"Unknown metric: {metric}")
    
    def train_epoch(self, train_loader, optimizer, scaler=None, scheduler=None):
        self.model.train()
        running_loss = 0.0
        all_preds, all_labels = [], []
        
        # Check if SAM optimizer
        from .techniques import SAM
        is_sam = isinstance(optimizer, SAM)
        use_amp = not is_sam and self.device == 'cuda'
        
        for images, labels in tqdm(train_loader, desc="Training", leave=False):
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            
            if is_sam:
                # SAM: Two-step (no AMP)
                outputs = self.forward(images)
                loss = self.compute_loss(outputs, labels)
                loss.backward()
                optimizer.first_step(zero_grad=True)
                
                outputs = self.forward(images)
                loss = self.compute_loss(outputs, labels)
                loss.backward()
                optimizer.second_step(zero_grad=True)
            else:
                # Standard (with AMP if available)
                if use_amp and scaler:
                    with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                        outputs = self.forward(images)
                        loss = self.compute_loss(outputs, labels)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    outputs = self.forward(images)
                    loss = self.compute_loss(outputs, labels)
                    loss.backward()
                    optimizer.step()
            
            # Scheduler step (for OneCycleLR)
            if scheduler and isinstance(scheduler, (
                optim.lr_scheduler.OneCycleLR,
                optim.lr_scheduler.SequentialLR
            )):
                scheduler.step()
            
            # Metrics
            running_loss += loss.detach().item() * images.size(0)
            
            with torch.no_grad():
                preds = self.get_predictions(outputs)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        avg_loss = running_loss / len(train_loader.dataset)
        acc = accuracy_score(all_labels, all_preds) * 100
        recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
        
        return avg_loss, acc, recall
    
    def validate_epoch(self, val_loader, primary_metric : str = "recall"):
        self.model.eval()
        running_loss = 0.0
        all_preds, all_labels = [], []
        
        with torch.inference_mode():
            for images, labels in tqdm(val_loader, desc="Validating", leave=False):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.forward(images)
                loss = self.compute_loss(outputs, labels)
                
                running_loss += loss.item() * images.size(0)
                
                preds = self.get_predictions(outputs)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate all metrics
        avg_loss = running_loss / len(val_loader.dataset)
        acc = accuracy_score(all_labels, all_preds) * 100
        recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
        precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
        f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        primary_value = self._get_metric_value(all_labels, all_preds, primary_metric)
        
        return avg_loss, acc, recall, precision, f1, primary_value
    
    def fit(self, train_loader, val_loader, epochs: int = 30, 
            lr: float = 1e-4, use_sam: bool = False,
            primary_metric: str = 'recall',
            patience: int = 10, min_delta: float = 0.001):
        # Optimizer
        if use_sam:
            from .techniques import SAM
            optimizer = SAM(self.model.parameters(), optim.AdamW, lr=lr, weight_decay=0.01, rho=0.05)
        else:
            optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)
        
        # Scheduler
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer.base_optimizer if use_sam else optimizer,
            T_max=epochs,
            eta_min=1e-7
        )
        
        # Scaler (only if not SAM)
        scaler = torch.amp.GradScaler(enabled=(self.device == 'cuda' and not use_sam))
        
        # Training loop
        patience_counter = 0
        
        print(f"\n{'='*80}")
        print(f"Training {self.__class__.__name__} - {self.model_name}")
        print(f"Optimizing for: {primary_metric.upper()} (primary)")
        print(f"{'='*80}\n")
        
        for epoch in range(epochs):
            # Train
            train_loss, train_acc, train_recall = self.train_epoch(
                train_loader, optimizer, scaler if not use_sam else None, scheduler
            )
            
            # Validate
            val_loss, val_acc, val_recall, val_prec, val_f1, primary_value = self.validate_epoch(val_loader)
            
            # Record history
            self.history.append({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'train_recall': train_recall,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'val_recall': val_recall,
                'val_precision': val_prec,
                'val_f1': val_f1,
                f'val_{primary_metric}': primary_value
            })
            
            # Check improvement
            improved = False
            improvement_msg = []
            
            # Primary metric
            if primary_value > self.best_metric_value + min_delta:
                self.best_metric_value = primary_value
                improved = True
                improvement_msg.append(f"{primary_metric.capitalize()}: {primary_value:.4f} ★")
            
            # Track all metrics
            if val_recall > self.best_recall:
                self.best_recall = val_recall
                if primary_metric != 'recall':
                    improvement_msg.append(f"Recall: {val_recall:.4f}")
            
            if val_acc > self.best_acc:
                self.best_acc = val_acc
                if primary_metric != 'accuracy':
                    improvement_msg.append(f"Acc: {val_acc:.2f}%")
            
            if val_f1 > self.best_f1:
                self.best_f1 = val_f1
                if primary_metric != 'f1':
                    improvement_msg.append(f"F1: {val_f1:.4f}")
            
            # Print progress
            if improved:
                print(f"Epoch {epoch+1}/{epochs} - " + " | ".join(improvement_msg))
                self.best_epoch = epoch + 1
                self.save(self.checkpoint_path)
                patience_counter = 0
            else:
                print(f"Epoch {epoch+1}/{epochs} - {primary_metric}: {primary_value:.4f}, Acc: {val_acc:.2f}%")
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
            
            # Step scheduler
            if not isinstance(scheduler, (optim.lr_scheduler.OneCycleLR, optim.lr_scheduler.SequentialLR)):
                scheduler.step()
        
        print(f"\n{'='*80}")
        print(f"Training Complete!")
        print(f"Best Epoch: {self.best_epoch}")
        print(f"Best {primary_metric.capitalize()}: {self.best_metric_value:.4f} ★")
        print(f"Best Recall: {self.best_recall:.4f}")
        print(f"Best Accuracy: {self.best_acc:.2f}%")
        print(f"Best F1: {self.best_f1:.4f}")
        print(f"{'='*80}\n")
        
        return self.history
    
    def evaluate(self, test_loader, class_names: Optional[List[str]] = None):
        self.model.eval()
        all_preds, all_probs, all_labels = [], [], []
        
        with torch.inference_mode():
            for images, labels in tqdm(test_loader, desc="Testing"):
                images = images.to(self.device)
                outputs = self.forward(images)
                probs = torch.softmax(outputs, dim=1)
                preds = self.get_predictions(outputs)
                
                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate all metrics
        acc = accuracy_score(all_labels, all_preds) * 100
        recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
        precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
        f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        
        # Per-class metrics
        if class_names is None:
            class_names = [f"Class {i}" for i in range(self.num_classes)]
        
        cm = confusion_matrix(all_labels, all_preds)
        report = classification_report(all_labels, all_preds, target_names=class_names, zero_division=0)
        per_class_recall = recall_score(all_labels, all_preds, average=None, zero_division=0)
        
        # print(f"\n{'='*80}")
        # print(f"TEST RESULTS - {self.__class__.__name__}")
        # print(f"{'='*80}")
        # print(f"Overall Accuracy: {acc:.2f}%")
        # print(f"Overall Recall: {recall:.4f} ★ (PRIMARY METRIC)")
        # print(f"Overall Precision: {precision:.4f}")
        # print(f"Overall F1: {f1:.4f}")
        # print(f"\nPer-Class Recall:")
        # for i, (name, rec) in enumerate(zip(class_names, per_class_recall)):
        #     print(f"  {name}: {rec:.4f}")
        # print(f"\nConfusion Matrix:")
        # print(cm)
        # print(f"\nDetailed Report:")
        # print(report)
        # print(f"{'='*80}\n")
        
        return {
            'accuracy': acc,
            'recall': recall,
            'precision': precision,
            'f1': f1,
            'per_class_recall': per_class_recall,
            'confusion_matrix': cm,
            'report': report,
            'labels': all_labels,
            'preds': all_preds,
            'probs': all_probs
        }
    
    def save(self, path: str):
        torch.save(self.model.state_dict(), path)
    
    def load(self, path: str):
        self.model.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))


# ============================================================================
# EVIDENTIAL LAYER
# ============================================================================

class EvidentialLayer(nn.Module):
    
    def __init__(self, in_features: int, num_classes: int):
        super().__init__()
        self.num_classes = num_classes
        self.evidence_layer = nn.Linear(in_features, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        evidence = F.softplus(self.evidence_layer(x))
        return evidence
    
    def get_predictions_and_uncertainty(self, evidence: torch.Tensor) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        # Dirichlet parameters
        alpha = evidence + 1
        
        # Expected probabilities
        S = alpha.sum(dim=1, keepdim=True)
        probs = alpha / S
        
        # Uncertainty (normalized)
        K = self.num_classes
        uncertainty = K / S
        
        return probs, uncertainty, alpha


# ============================================================================
# EVIDENTIAL LOSS
# ============================================================================

class EvidentialLoss(nn.Module):
    
    def __init__(self, num_classes: int, lam: float = 0.5, epsilon: float = 1e-10):
        super().__init__()
        self.num_classes = num_classes
        self.lam = lam
        self.epsilon = epsilon
    
    def forward(self, evidence: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        alpha = evidence + 1
        S = alpha.sum(dim=1, keepdim=True)
        
        # One-hot encoding
        target_one_hot = F.one_hot(target, num_classes=self.num_classes).float()
        
        # Bayesian risk
        A = torch.sum((target_one_hot - alpha / S) ** 2, dim=1, keepdim=True)
        B = torch.sum(alpha * (S - alpha) / (S * S * (S + 1)), dim=1, keepdim=True)
        loss_mse = (A + B).mean()
        
        # KL divergence regularization
        alpha_tilde = target_one_hot + (1 - target_one_hot) * alpha
        S_tilde = alpha_tilde.sum(dim=1, keepdim=True)
        
        kl_div = torch.lgamma(S_tilde) - torch.lgamma(alpha_tilde).sum(dim=1, keepdim=True)
        kl_div += ((alpha_tilde - 1) * (torch.digamma(alpha_tilde) - torch.digamma(S_tilde))).sum(dim=1, keepdim=True)
        kl_div = kl_div.mean()
        
        return loss_mse + self.lam * kl_div


# ============================================================================
# ARCHITECTURE HANDLER
# ============================================================================

class ArchitectureHandler:
    
    @staticmethod
    def get_feature_extractor(model: nn.Module, model_name: str) -> Tuple[nn.Module, int]:
        model_name_lower = model_name.lower()
        
        # ResNet family
        if 'resnet' in model_name_lower or 'resnext' in model_name_lower:
            feature_extractor = nn.Sequential(
                model.conv1, model.bn1, model.relu, model.maxpool,
                model.layer1, model.layer2, model.layer3, model.layer4,
                model.global_pool, nn.Flatten()
            )
            in_features = model.fc.in_features
        
        # EfficientNet family
        elif 'efficientnet' in model_name_lower:
            if hasattr(model, 'conv_stem'):
                feature_extractor = nn.Sequential(
                    model.conv_stem, model.bn1, model.act1,
                    model.blocks, model.conv_head, model.bn2, model.act2,
                    model.global_pool, nn.Flatten()
                )
            else:
                modules = list(model.children())[:-1]
                feature_extractor = nn.Sequential(*modules, nn.Flatten())
            
            if hasattr(model, 'classifier'):
                in_features = model.classifier.in_features
            elif hasattr(model, 'fc'):
                in_features = model.fc.in_features
            else:
                dummy = torch.randn(1, 3, 224, 224)
                with torch.no_grad():
                    out = feature_extractor(dummy)
                in_features = out.shape[1]
        
        # Vision Transformer
        elif 'vit' in model_name_lower:
            class ViTFeatureExtractor(nn.Module):
                def __init__(self, vit_model):
                    super().__init__()
                    self.patch_embed = vit_model.patch_embed
                    self.cls_token = vit_model.cls_token
                    self.pos_embed = vit_model.pos_embed
                    self.pos_drop = vit_model.pos_drop if hasattr(vit_model, 'pos_drop') else nn.Identity()
                    self.blocks = vit_model.blocks
                    self.norm = vit_model.norm
                
                def forward(self, x):
                    x = self.patch_embed(x)
                    cls_token = self.cls_token.expand(x.shape[0], -1, -1)
                    x = torch.cat([cls_token, x], dim=1)
                    x = x + self.pos_embed
                    x = self.pos_drop(x)
                    x = self.blocks(x)
                    x = self.norm(x)
                    return x[:, 0]
            
            feature_extractor = ViTFeatureExtractor(model)
            in_features = model.head.in_features if hasattr(model.head, 'in_features') else model.embed_dim
        
        # Swin Transformer
        elif 'swin' in model_name_lower:
            class SwinFeatureExtractor(nn.Module):
                def __init__(self, swin_model):
                    super().__init__()
                    self.patch_embed = swin_model.patch_embed
                    self.layers = swin_model.layers
                    self.norm = swin_model.norm
                    self.avgpool = nn.AdaptiveAvgPool1d(1)
                
                def forward(self, x):
                    x = self.patch_embed(x)
                    x = self.layers(x)
                    x = self.norm(x)
                    x = x.transpose(1, 2)
                    x = self.avgpool(x)
                    x = x.flatten(1)
                    return x
            
            feature_extractor = SwinFeatureExtractor(model)
            
            if hasattr(model.head, 'fc'):
                in_features = model.head.fc.in_features
            elif hasattr(model.head, 'in_features'):
                in_features = model.head.in_features
            else:
                in_features = model.num_features
        
        # Generic fallback
        else:
            print(f"Warning: Unknown architecture '{model_name}', using generic extraction")
            modules = list(model.children())[:-1]
            feature_extractor = nn.Sequential(*modules, nn.Flatten())
            
            dummy = torch.randn(1, 3, 224, 224)
            with torch.no_grad():
                out = feature_extractor(dummy)
            in_features = out.shape[1]
        
        return feature_extractor, in_features


# ============================================================================
# UNIVERSAL EVIDENTIAL MODEL
# ============================================================================

class UniversalEvidentialModel(nn.Module):
    
    def __init__(self, model_name: str, num_classes: int = 4, pretrained: bool = True):
        super().__init__()
        
        self.model_name = model_name
        self.num_classes = num_classes
        
        # Load pretrained model
        base_model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
        
        # Extract feature extractor
        self.feature_extractor, in_features = ArchitectureHandler.get_feature_extractor(
            base_model, model_name
        )
        
        # Evidential head
        self.evidential_head = EvidentialLayer(in_features, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(x)
        evidence = self.evidential_head(features)
        return evidence
    
    def get_predictions_and_uncertainty(self, evidence: torch.Tensor) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        return self.evidential_head.get_predictions_and_uncertainty(evidence)


# ============================================================================
# SAM OPTIMIZER (Solution 9)
# ============================================================================

class SAM(torch.optim.Optimizer):
    
    def __init__(self, params, base_optimizer, rho=0.05, **kwargs):
        defaults = dict(rho=rho, **kwargs)
        super(SAM, self).__init__(params, defaults)
        
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
    
    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            
            for p in group["params"]:
                if p.grad is None:
                    continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = p.grad * scale.to(p)
                p.add_(e_w)
        
        if zero_grad:
            self.zero_grad()
    
    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.data = self.state[p]["old_p"]
        
        self.base_optimizer.step()
        
        if zero_grad:
            self.zero_grad()
    
    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device
        norm = torch.norm(
            torch.stack([
                p.grad.norm(p=2).to(shared_device)
                for group in self.param_groups for p in group["params"]
                if p.grad is not None
            ]),
            p=2
        )
        return norm


# ============================================================================
# CENTER LOSS (Solution 4)
# ============================================================================

class CenterLoss(nn.Module):
    
    def __init__(self, num_classes: int, embedding_dim: int, lambda_c: float = 1.0):
        super().__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.lambda_c = lambda_c
        
        # Learnable class centers
        self.centers = nn.Parameter(torch.randn(num_classes, embedding_dim))
    
    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        centers_batch = self.centers[labels]
        loss = F.mse_loss(embeddings, centers_batch)
        return self.lambda_c * loss


# ============================================================================
# TRIPLET LOSS (Solution 3)
# ============================================================================

class TripletLoss(nn.Module):
    
    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin
    
    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, 
                negative: torch.Tensor) -> torch.Tensor:
        distance_positive = F.pairwise_distance(anchor, positive, p=2)
        distance_negative = F.pairwise_distance(anchor, negative, p=2)
        
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean()


def create_triplet_batch(embeddings: torch.Tensor, labels: torch.Tensor) -> Tuple[
    Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]
]:
    triplets = []
    
    for i, label in enumerate(labels):
        anchor = embeddings[i]
        
        # Positive: same class, different sample
        positive_mask = (labels == label) & (torch.arange(len(labels), device=labels.device) != i)
        if positive_mask.sum() > 0:
            positive_idx = torch.where(positive_mask)[0][torch.randint(positive_mask.sum(), (1,))].item()
            positive = embeddings[positive_idx]
        else:
            continue
        
        # Negative: different class, hardest (closest to anchor)
        negative_mask = labels != label
        if negative_mask.sum() > 0:
            negative_candidates = embeddings[negative_mask]
            distances = F.pairwise_distance(anchor.unsqueeze(0), negative_candidates)
            hardest_idx = distances.argmin()
            negative = negative_candidates[hardest_idx]
        else:
            continue
        
        triplets.append((anchor, positive, negative))
    
    if len(triplets) == 0:
        return None, None, None
    
    anchors = torch.stack([t[0] for t in triplets])
    positives = torch.stack([t[1] for t in triplets])
    negatives = torch.stack([t[2] for t in triplets])
    
    return anchors, positives, negatives


# ============================================================================
# MANIFOLD MIXUP (Solution 5)
# ============================================================================

class ManifoldMixup(nn.Module):
    
    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha
    
    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, float
    ]:
        batch_size = embeddings.size(0)
        
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1
        
        index = torch.randperm(batch_size).to(embeddings.device)
        
        mixed_embeddings = lam * embeddings + (1 - lam) * embeddings[index]
        labels_a = labels
        labels_b = labels[index]
        
        return mixed_embeddings, labels_a, labels_b, lam


def manifold_mixup_loss(criterion, outputs: torch.Tensor, labels_a: torch.Tensor,
                        labels_b: torch.Tensor, lam: float) -> torch.Tensor:
    loss = lam * criterion(outputs, labels_a) + (1 - lam) * criterion(outputs, labels_b)
    return loss


# ============================================================================
# COSINE CLASSIFIER (Solution 6)
# ============================================================================

class CosineClassifier(nn.Module):
    
    def __init__(self, in_features: int, num_classes: int, scale: float = 30.0):
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.scale = scale
        
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # Normalize features and weights
        features_normalized = F.normalize(features, p=2, dim=1)
        weight_normalized = F.normalize(self.weight, p=2, dim=1)
        
        # Cosine similarity
        cosine = F.linear(features_normalized, weight_normalized)
        
        # Scale for stable training
        logits = self.scale * cosine
        
        return logits


# ============================================================================
# SE BLOCK (Solution 7)
# ============================================================================

class SEBlock(nn.Module):
    
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, _, _ = x.size()
        
        # Squeeze: Global information embedding
        y = self.squeeze(x).view(batch, channels)
        
        # Excitation: Channel attention
        y = self.excitation(y).view(batch, channels, 1, 1)
        
        # Scale
        return x * y.expand_as(x)


# ============================================================================
# DISTANCE-AWARE LABEL SMOOTHING (Solution 8)
# ============================================================================

class DistanceAwareLabelSmoothing(nn.Module):
    
    def __init__(self, num_classes: int, smoothing: float = 0.1):
        super().__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=1)
        batch_size = logits.size(0)
        
        # Create smooth labels
        smooth_labels = torch.zeros_like(log_probs)
        
        for i in range(batch_size):
            true_class = targets[i].item()
            
            # Confidence on true class
            smooth_labels[i, true_class] = self.confidence
            
            # Distribute remaining mass based on distance
            remaining = self.smoothing
            total_weight = 0.0
            
            for k in range(self.num_classes):
                if k != true_class:
                    distance = abs(k - true_class)
                    weight = 1.0 / (distance + 1)
                    total_weight += weight
            
            # Normalize weights
            for k in range(self.num_classes):
                if k != true_class:
                    distance = abs(k - true_class)
                    weight = 1.0 / (distance + 1)
                    smooth_labels[i, k] = remaining * (weight / total_weight)
        
        # KL divergence loss
        loss = -(smooth_labels * log_probs).sum(dim=1).mean()
        
        return loss


# ============================================================================
# PROTOTYPICAL NETWORK (Solution 2)
# ============================================================================

class PrototypicalNetwork(nn.Module):
    
    def __init__(self, encoder: nn.Module, embedding_dim: int = 512):
        super().__init__()
        self.encoder = encoder
        self.embedding_dim = embedding_dim
        self.prototypes = None
    
    def compute_prototypes(self, support_embeddings: torch.Tensor, 
                          support_labels: torch.Tensor, num_classes: int) -> torch.Tensor:
        prototypes = torch.zeros(num_classes, self.embedding_dim).to(support_embeddings.device)
        
        for k in range(num_classes):
            class_mask = (support_labels == k)
            class_embeddings = support_embeddings[class_mask]
            
            if class_embeddings.size(0) > 0:
                prototypes[k] = class_embeddings.mean(dim=0)
        
        return prototypes
    
    def euclidean_distance(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        n = x.size(0)
        m = y.size(0)
        d = x.size(1)
        
        x = x.unsqueeze(1).expand(n, m, d)
        y = y.unsqueeze(0).expand(n, m, d)
        
        return torch.pow(x - y, 2).sum(2)
    
    def forward(self, query_images: torch.Tensor, support_images: Optional[torch.Tensor] = None,
                support_labels: Optional[torch.Tensor] = None, num_classes: int = 4) -> torch.Tensor:
        query_embeddings = self.encoder(query_images)
        
        if support_images is not None:
            support_embeddings = self.encoder(support_images)
            prototypes = self.compute_prototypes(support_embeddings, support_labels, num_classes)
        else:
            prototypes = self.prototypes
        
        distances = self.euclidean_distance(query_embeddings, prototypes)
        log_probs = F.log_softmax(-distances, dim=1)
        
        return log_probs
    
    def update_prototypes(self, train_loader, num_classes: int = 4):
        self.encoder.eval()
        all_embeddings = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in train_loader:
                embeddings = self.encoder(images.to(next(self.encoder.parameters()).device))
                all_embeddings.append(embeddings)
                all_labels.append(labels)
        
        all_embeddings = torch.cat(all_embeddings, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        
        self.prototypes = self.compute_prototypes(all_embeddings, all_labels, num_classes)


class FocalLoss(nn.Module):
    
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        p_t = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - p_t) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class ArchitectureLayerGroups:
    
    @staticmethod
    def get_resnet_groups(model):
        return [
            list(model.conv1.parameters()) + list(model.bn1.parameters()) + list(model.layer1.parameters()),
            list(model.layer2.parameters()),
            list(model.layer3.parameters()),
            list(model.layer4.parameters()),
            list(model.fc.parameters())
        ]
    
    @staticmethod
    def get_vit_groups(model):
        if hasattr(model, 'blocks'):
            num_blocks = len(model.blocks)
        else:
            num_blocks = 12
        
        split1 = num_blocks // 4
        split2 = num_blocks // 2
        split3 = 3 * num_blocks // 4
        
        groups = []
        
        # Group 0: Patch embedding + early blocks
        group0 = []
        if hasattr(model, 'patch_embed'):
            group0.extend(list(model.patch_embed.parameters()))
        if hasattr(model, 'pos_embed'):
            group0.append(model.pos_embed)
        if hasattr(model, 'cls_token'):
            group0.append(model.cls_token)
        
        if hasattr(model, 'blocks'):
            for i in range(0, split1):
                group0.extend(list(model.blocks[i].parameters()))
        groups.append(group0)
        
        # Group 1-3: Block ranges
        if hasattr(model, 'blocks'):
            groups.append([p for i in range(split1, split2) for p in model.blocks[i].parameters()])
            groups.append([p for i in range(split2, split3) for p in model.blocks[i].parameters()])
            
            group3 = [p for i in range(split3, num_blocks) for p in model.blocks[i].parameters()]
            if hasattr(model, 'norm'):
                group3.extend(list(model.norm.parameters()))
            groups.append(group3)
        else:
            groups.extend([[], [], []])
        
        # Group 4: Head
        group4 = []
        if hasattr(model, 'head'):
            group4.extend(list(model.head.parameters()))
        elif hasattr(model, 'fc'):
            group4.extend(list(model.fc.parameters()))
        groups.append(group4)
        
        return groups
    
    @staticmethod
    def get_swin_groups(model):
        groups = []
        
        # Group 0: Patch embed + Stage 1
        group0 = []
        if hasattr(model, 'patch_embed'):
            group0.extend(list(model.patch_embed.parameters()))
        if hasattr(model, 'layers') and len(model.layers) > 0:
            group0.extend(list(model.layers[0].parameters()))
        groups.append(group0)
        
        # Groups 1-3: Stages 2-4
        if hasattr(model, 'layers'):
            for i in range(1, min(4, len(model.layers))):
                groups.append(list(model.layers[i].parameters()))
            
            # Pad if needed
            while len(groups) < 4:
                groups.append([])
            
            # Add norm to last group
            if hasattr(model, 'norm'):
                groups[3].extend(list(model.norm.parameters()))
        else:
            groups.extend([[], [], []])
        
        # Group 4: Head
        group4 = []
        if hasattr(model, 'head'):
            if hasattr(model.head, 'fc'):
                group4.extend(list(model.head.fc.parameters()))
            else:
                group4.extend(list(model.head.parameters()))
        elif hasattr(model, 'fc'):
            group4.extend(list(model.fc.parameters()))
        groups.append(group4)
        
        return groups
    
    @staticmethod
    def get_efficientnet_groups(model):
        children = list(model.children())
        n_children = len(children)
        n_per_group = max(1, n_children // 5)
        
        groups = []
        for i in range(4):
            start = i * n_per_group
            end = (i + 1) * n_per_group if i < 3 else n_children - 1
            group_params = []
            for child in children[start:end]:
                if hasattr(child, 'parameters'):
                    group_params.extend(list(child.parameters()))
            groups.append(group_params)
        
        # Last group (classifier)
        group4 = []
        for child in children[4*n_per_group:]:
            if hasattr(child, 'parameters'):
                group4.extend(list(child.parameters()))
        groups.append(group4)
        
        return groups
    
    @staticmethod
    def get_mobilenet_groups(model):
        if hasattr(model, 'features'):
            features = model.features
            n_features = len(features)
            
            return [
                list(features[:n_features//4].parameters()),
                list(features[n_features//4:n_features//2].parameters()),
                list(features[n_features//2:3*n_features//4].parameters()),
                list(features[3*n_features//4:].parameters()),
                list(model.classifier.parameters()) if hasattr(model, 'classifier') else []
            ]
        else:
            # Fallback
            all_params = list(model.parameters())
            n = len(all_params)
            return [
                all_params[:n//5],
                all_params[n//5:2*n//5],
                all_params[2*n//5:3*n//5],
                all_params[3*n//5:4*n//5],
                all_params[4*n//5:]
            ]
    
    @staticmethod
    def get_layer_groups(model, model_name):
        model_name_lower = model_name.lower()
        
        if 'resnet' in model_name_lower or 'resnext' in model_name_lower:
            return ArchitectureLayerGroups.get_resnet_groups(model)
        elif 'vit' in model_name_lower:
            return ArchitectureLayerGroups.get_vit_groups(model)
        elif 'swin' in model_name_lower:
            return ArchitectureLayerGroups.get_swin_groups(model)
        elif 'efficientnet' in model_name_lower:
            return ArchitectureLayerGroups.get_efficientnet_groups(model)
        elif 'mobilenet' in model_name_lower:
            return ArchitectureLayerGroups.get_mobilenet_groups(model)
        else:
            # Generic fallback
            all_params = list(model.parameters())
            n = len(all_params)
            return [
                all_params[:n//5],
                all_params[n//5:2*n//5],
                all_params[2*n//5:3*n//5],
                all_params[3*n//5:4*n//5],
                all_params[4*n//5:]
            ]


class ProgressiveClassifier(BaseClassifier):
    
    def build_model(self):
        self.model = timm.create_model(
            self.model_name,
            pretrained=True,
            num_classes=self.num_classes
        )
        
        # Get layer groups for discriminative LRs
        self.layer_groups = ArchitectureLayerGroups.get_layer_groups(
            self.model, self.model_name
        )
        
        # Detect architecture type for scheduler selection
        self.architecture_type = 'transformer' if any(
            x in self.model_name.lower() for x in ['vit', 'swin', 'transformer']
        ) else 'cnn'
    
    def forward(self, images):
        return self.model(images)
    
    def compute_loss(self, outputs, labels):
        if not hasattr(self, 'focal_loss'):
            self.focal_loss = FocalLoss(alpha=1.0, gamma=2.0).to(self.device)
        return self.focal_loss(outputs, labels)
    
    def _get_discriminative_params(self, base_lr):
        lr_multipliers = [1/100, 1/10, 1/3, 1.0, 10.0]
        
        param_groups = []
        for params, mult in zip(self.layer_groups, lr_multipliers):
            if params:
                param_groups.append({
                    'params': params,
                    'lr': base_lr * mult
                })
        
        return param_groups
    
    def fit(self, train_loader, val_loader, epochs=30, lr=1e-4,
            use_sam=False, primary_metric='recall',
            patience=10, min_delta=0.001):
        print(f"\n{'='*80}")
        print(f"PROGRESSIVE FINE-TUNING: {self.model_name}")
        print(f"Optimizing for: {primary_metric.upper()}")
        print(f"{'='*80}\n")
        
        # Phase 1: Classifier only (5 epochs)
        print("="*80)
        print("PHASE 1: Training Classifier Only (Backbone Frozen)")
        print("="*80)
        
        self._train_phase(
            phase=1,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=min(5, epochs),
            lr=lr * 10,  # Higher LR for random classifier
            freeze_mode='classifier_only',
            use_sam=False,
            primary_metric=primary_metric,
            patience=5,
            min_delta=min_delta
        )
        
        # Phase 2: Top 50% layers (10 epochs)
        remaining_epochs = max(0, epochs - 5)
        if remaining_epochs > 0:
            print("\n" + "="*80)
            print("PHASE 2: Fine-tuning Top Layers (Bottom 50% Frozen)")
            print("="*80)
            
            self._train_phase(
                phase=2,
                train_loader=train_loader,
                val_loader=val_loader,
                epochs=min(10, remaining_epochs),
                lr=lr,
                freeze_mode='top_50',
                use_sam=False,
                primary_metric=primary_metric,
                patience=7,
                min_delta=min_delta
            )
        
        # Phase 3: All layers with discriminative LRs (remaining epochs)
        remaining_epochs = max(0, epochs - 15)
        if remaining_epochs > 0:
            print("\n" + "="*80)
            print("PHASE 3: Discriminative Fine-Tuning (All Layers)")
            print("="*80)
            
            self._train_phase(
                phase=3,
                train_loader=train_loader,
                val_loader=val_loader,
                epochs=remaining_epochs,
                lr=lr,
                freeze_mode='all_discriminative',
                use_sam=use_sam,  # SAM only in phase 3
                primary_metric=primary_metric,
                patience=patience,
                min_delta=min_delta
            )
        
        print(f"\n{'='*80}")
        print("PROGRESSIVE FINE-TUNING COMPLETE")
        print(f"Final Best {primary_metric.capitalize()}: {self.best_metric_value:.4f} ★")
        print(f"Final Best Recall: {self.best_recall:.4f}")
        print(f"Final Best Accuracy: {self.best_acc:.2f}%")
        print(f"{'='*80}\n")
        
        return self.history
    
    def _train_phase(self, phase, train_loader, val_loader, epochs, lr,
                    freeze_mode, use_sam, primary_metric, patience, min_delta):
        
        # Freeze/unfreeze according to mode
        if freeze_mode == 'classifier_only':
            # Freeze all except classifier
            for param in self.model.parameters():
                param.requires_grad = False
            if self.layer_groups[-1]:
                for param in self.layer_groups[-1]:
                    param.requires_grad = True
                    
        elif freeze_mode == 'top_50':
            # Unfreeze top 50%
            all_params = list(self.model.parameters())
            n_params = len(all_params)
            for param in all_params[:n_params//2]:
                param.requires_grad = False
            for param in all_params[n_params//2:]:
                param.requires_grad = True
                
        elif freeze_mode == 'all_discriminative':
            # Unfreeze everything
            for param in self.model.parameters():
                param.requires_grad = True
        
        # Create optimizer
        if freeze_mode == 'all_discriminative':
            # Discriminative LRs
            param_groups = self._get_discriminative_params(lr)
            print(f"Discriminative LR groups:")
            for i, group in enumerate(param_groups):
                print(f"  Group {i}: {len(list(group['params']))} params, LR={group['lr']:.2e}")
        else:
            # Single LR
            param_groups = filter(lambda p: p.requires_grad, self.model.parameters())
        
        if use_sam:
            optimizer = SAM(param_groups, optim.AdamW, lr=lr, weight_decay=0.01, rho=0.05)
        else:
            optimizer = optim.AdamW(param_groups, lr=lr, weight_decay=0.01)
        
        # Create scheduler
        if self.architecture_type == 'cnn':
            scheduler = optim.lr_scheduler.OneCycleLR(
                optimizer.base_optimizer if use_sam else optimizer,
                max_lr=lr if freeze_mode != 'all_discriminative' else lr * 10,  # Max LR for classifier
                epochs=epochs,
                steps_per_epoch=len(train_loader),
                pct_start=0.3,
                div_factor=25.0,
                final_div_factor=1000.0
            )
        else:  # transformer
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer.base_optimizer if use_sam else optimizer,
                T_max=epochs,
                eta_min=1e-7
            )
        
        # Scaler
        scaler = torch.amp.GradScaler(enabled=(self.device == 'cuda' and not use_sam))
        
        # Training loop
        patience_counter = 0
        
        for epoch in range(epochs):
            # Train
            train_loss, train_acc, train_recall = self.train_epoch(
                train_loader, optimizer, scaler if not use_sam else None, scheduler
            )
            
            # Validate
            val_loss, val_acc, val_recall, val_prec, val_f1, primary_value = self.validate_epoch(val_loader)
            
            # Record history
            self.history.append({
                'phase': phase,
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'train_recall': train_recall,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'val_recall': val_recall,
                'val_precision': val_prec,
                'val_f1': val_f1,
                f'val_{primary_metric}': primary_value
            })
            
            # Check improvement
            improved = False
            if primary_value > self.best_metric_value + min_delta:
                self.best_metric_value = primary_value
                improved = True
            if val_recall > self.best_recall:
                self.best_recall = val_recall
            if val_acc > self.best_acc:
                self.best_acc = val_acc
            if val_f1 > self.best_f1:
                self.best_f1 = val_f1
            
            # Print
            if improved:
                print(f"  [Epoch {epoch+1}/{epochs}] {primary_metric}: {primary_value:.4f} ★, "
                      f"Acc: {val_acc:.2f}%, Recall: {val_recall:.4f}")
                self.best_epoch = epoch + 1
                torch.save(self.model.state_dict(),
                          f'ProgressiveClassifier_{self.model_name}_best.pth')
                patience_counter = 0
            else:
                print(f"  [Epoch {epoch+1}/{epochs}] {primary_metric}: {primary_value:.4f}, "
                      f"Acc: {val_acc:.2f}%")
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= patience:
                print(f"  Early stopping (patience={patience})")
                break
            
            # Step scheduler (if not OneCycleLR)
            if not isinstance(scheduler, optim.lr_scheduler.OneCycleLR):
                scheduler.step()
        
        print(f"âœ… Phase {phase} Complete - Best {primary_metric}: {self.best_metric_value:.4f}")


# ============================================================================
# 1. BASELINE CLASSIFIER
# ============================================================================

class BaselineClassifier(BaseClassifier):
    
    def build_model(self):
        self.model = timm.create_model(
            self.model_name,
            pretrained=True,
            num_classes=self.num_classes
        )
    
    def forward(self, images):
        return self.model(images)
    
    def compute_loss(self, outputs, labels):
        return F.cross_entropy(outputs, labels)


# ============================================================================
# 2. EVIDENTIAL CLASSIFIER
# ============================================================================

class EvidentialClassifier(BaseClassifier):
    
    def build_model(self):
        self.evidential_model = UniversalEvidentialModel(
            self.model_name,
            num_classes=self.num_classes,
            pretrained=True
        )
        self.model = self.evidential_model
        self.criterion = EvidentialLoss(self.num_classes, lam=0.5)
    
    def forward(self, images):
        return self.evidential_model(images)
    
    def compute_loss(self, evidence, labels):
        return self.criterion(evidence, labels)
    
    def get_predictions(self, evidence):
        probs, _, _ = self.evidential_model.get_predictions_and_uncertainty(evidence)
        return torch.argmax(probs, dim=1)


# ============================================================================
# 3. METRIC LEARNING CLASSIFIER
# ============================================================================

class MetricLearningClassifier(BaseClassifier):
    
    def build_model(self):
        # Base model
        base = timm.create_model(self.model_name, pretrained=True, num_classes=self.num_classes)
        
        # Get feature dimension
        if hasattr(base, 'fc'):
            in_features = base.fc.in_features
            self.feature_extractor = nn.Sequential(*list(base.children())[:-1], nn.Flatten())
        elif hasattr(base, 'head'):
            in_features = base.head.in_features
            self.feature_extractor = nn.Sequential(*list(base.children())[:-1], nn.Flatten())
        else:
            in_features = 512
            self.feature_extractor = nn.Sequential(*list(base.children())[:-1], nn.Flatten())
        
        # Embedding layer
        self.embedding_dim = 256
        self.embedding = nn.Sequential(
            nn.Linear(in_features, self.embedding_dim),
            nn.BatchNorm1d(self.embedding_dim),
            nn.ReLU()
        )
        
        # Classifier
        self.classifier = nn.Linear(self.embedding_dim, self.num_classes)
        
        # Losses
        self.ce_loss = nn.CrossEntropyLoss()
        self.center_loss = CenterLoss(self.num_classes, self.embedding_dim, lambda_c=0.1)
        self.triplet_loss = TripletLoss(margin=1.0)
        
        # Combined model
        self.model = nn.ModuleDict({
            'feature_extractor': self.feature_extractor,
            'embedding': self.embedding,
            'classifier': self.classifier,
            'center_loss': self.center_loss
        })
    
    def forward(self, images):
        features = self.feature_extractor(images)
        embeddings = self.embedding(features)
        logits = self.classifier(embeddings)
        return logits, embeddings
    
    def compute_loss(self, outputs, labels):
        logits, embeddings = outputs
        
        # CrossEntropy
        ce = self.ce_loss(logits, labels)
        
        # Center Loss
        center = self.center_loss(embeddings, labels)
        
        # Triplet Loss
        anchors, positives, negatives = create_triplet_batch(embeddings, labels)
        if anchors is not None:
            triplet = self.triplet_loss(anchors, positives, negatives)
        else:
            triplet = torch.tensor(0.0).to(embeddings.device)
        
        return ce + center + 0.1 * triplet
    
    def get_predictions(self, outputs):
        logits, _ = outputs
        return torch.argmax(logits, dim=1)


# ============================================================================
# 4. REGULARIZED CLASSIFIER
# ============================================================================

class RegularizedClassifier(BaseClassifier):
    
    def build_model(self):
        base = timm.create_model(self.model_name, pretrained=True, num_classes=self.num_classes)
        
        # Extract feature extractor
        if hasattr(base, 'fc'):
            in_features = base.fc.in_features
            self.feature_extractor = nn.Sequential(*list(base.children())[:-1], nn.Flatten())
        elif hasattr(base, 'head'):
            in_features = base.head.in_features
            self.feature_extractor = nn.Sequential(*list(base.children())[:-1], nn.Flatten())
        else:
            in_features = 512
            self.feature_extractor = nn.Sequential(*list(base.children())[:-1], nn.Flatten())
        
        self.classifier = nn.Linear(in_features, self.num_classes)
        
        # Regularizers
        self.manifold_mixup = ManifoldMixup(alpha=0.2)
        self.criterion = DistanceAwareLabelSmoothing(self.num_classes, smoothing=0.1)
        
        self.model = nn.ModuleDict({
            'feature_extractor': self.feature_extractor,
            'classifier': self.classifier
        })
        
        self.mixup_enabled = True
    
    def forward(self, images, labels=None):
        features = self.feature_extractor(images)
        
        # Apply manifold mixup during training
        if self.training and self.mixup_enabled and labels is not None:
            features, labels_a, labels_b, lam = self.manifold_mixup(features, labels)
            logits = self.classifier(features)
            return logits, labels_a, labels_b, lam
        else:
            logits = self.classifier(features)
            return logits
    
    def compute_loss(self, outputs, labels):
        if isinstance(outputs, tuple) and len(outputs) == 4:
            logits, labels_a, labels_b, lam = outputs
            loss = manifold_mixup_loss(self.criterion, logits, labels_a, labels_b, lam)
        else:
            logits = outputs
            loss = self.criterion(logits, labels)
        return loss
    
    def get_predictions(self, outputs):
        if isinstance(outputs, tuple):
            logits = outputs[0]
        else:
            logits = outputs
        return torch.argmax(logits, dim=1)
    
    def train_epoch(self, train_loader, optimizer, scaler=None, scheduler=None):
        self.model.train()
        self.mixup_enabled = True
        running_loss = 0.0
        all_preds, all_labels = [], []
        
        from .techniques import SAM
        is_sam = isinstance(optimizer, SAM)
        use_amp = not is_sam and self.device == 'cuda'
        
        for images, labels in train_loader:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            
            if is_sam:
                outputs = self.forward(images, labels)
                loss = self.compute_loss(outputs, labels)
                loss.backward()
                optimizer.first_step(zero_grad=True)
                
                outputs = self.forward(images, labels)
                loss = self.compute_loss(outputs, labels)
                loss.backward()
                optimizer.second_step(zero_grad=True)
            else:
                if use_amp and scaler:
                    with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                        outputs = self.forward(images, labels)
                        loss = self.compute_loss(outputs, labels)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    outputs = self.forward(images, labels)
                    loss = self.compute_loss(outputs, labels)
                    loss.backward()
                    optimizer.step()
            
            if scheduler and isinstance(scheduler, (
                torch.optim.lr_scheduler.OneCycleLR,
                torch.optim.lr_scheduler.SequentialLR
            )):
                scheduler.step()
            
            running_loss += loss.detach().item() * images.size(0)
            
            with torch.no_grad():
                preds = self.get_predictions(outputs)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        from sklearn.metrics import accuracy_score, recall_score
        avg_loss = running_loss / len(train_loader.dataset)
        acc = accuracy_score(all_labels, all_preds) * 100
        recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
        
        return avg_loss, acc, recall


# ============================================================================
# 5. ATTENTION-ENHANCED CLASSIFIER
# ============================================================================

class AttentionEnhancedClassifier(BaseClassifier):
    
    def build_model(self):
        base = timm.create_model(self.model_name, pretrained=True, num_classes=self.num_classes)
        
        # Add SE blocks (simplified - add to last layer)
        if hasattr(base, 'layer4'):
            for block in base.layer4:
                if hasattr(block, 'conv2'):
                    channels = block.conv2.out_channels
                    block.se = SEBlock(channels, reduction=16)
        
        # Get feature dimension
        if hasattr(base, 'fc'):
            in_features = base.fc.in_features
            self.feature_extractor = nn.Sequential(*list(base.children())[:-1], nn.Flatten())
        elif hasattr(base, 'head'):
            in_features = base.head.in_features
            self.feature_extractor = nn.Sequential(*list(base.children())[:-1], nn.Flatten())
        else:
            in_features = 512
            self.feature_extractor = nn.Sequential(*list(base.children())[:-1], nn.Flatten())
        
        # Cosine classifier
        self.cosine_classifier = CosineClassifier(in_features, self.num_classes, scale=30.0)
        
        self.model = nn.ModuleDict({
            'feature_extractor': self.feature_extractor,
            'cosine_classifier': self.cosine_classifier
        })
    
    def forward(self, images):
        features = self.feature_extractor(images)
        logits = self.cosine_classifier(features)
        return logits
    
    def compute_loss(self, outputs, labels):
        return F.cross_entropy(outputs, labels)


# ============================================================================
# 6. PROGRESSIVE EVIDENTIAL CLASSIFIER
# ============================================================================

class ProgressiveEvidentialClassifier(BaseClassifier):
    
    def build_model(self):
        self.evidential_model = UniversalEvidentialModel(
            self.model_name,
            num_classes=self.num_classes,
            pretrained=True
        )
        self.model = self.evidential_model
        self.criterion = EvidentialLoss(self.num_classes, lam=0.5)
    
    def forward(self, images):
        return self.evidential_model(images)
    
    def compute_loss(self, evidence, labels):
        return self.criterion(evidence, labels)
    
    def get_predictions(self, evidence):
        probs, _, _ = self.evidential_model.get_predictions_and_uncertainty(evidence)
        return torch.argmax(probs, dim=1)
    
    def fit(self, train_loader, val_loader, epochs: int = 30, lr: float = 1e-4,
            use_sam: bool = False, primary_metric: str = 'recall',
            patience: int = 10, min_delta: float = 0.001):
        print(f"\n{'='*80}")
        print(f"PROGRESSIVE EVIDENTIAL TRAINING")
        print(f"{'='*80}\n")
        
        # Phase 1: Classifier only
        print("Phase 1: Training Evidential Head Only (5 epochs)")
        for param in self.model.feature_extractor.parameters():
            param.requires_grad = False
        for param in self.model.evidential_head.parameters():
            param.requires_grad = True
        
        super().fit(train_loader, val_loader, epochs=5, lr=lr*10, use_sam=False,
                   primary_metric=primary_metric, patience=5, min_delta=min_delta)
        
        # Phase 2: All layers
        print("\nPhase 2: Fine-tuning All Layers (remaining epochs)")
        for param in self.model.parameters():
            param.requires_grad = True
        
        remaining_epochs = epochs - 5
        super().fit(train_loader, val_loader, epochs=remaining_epochs, lr=lr, use_sam=use_sam,
                   primary_metric=primary_metric, patience=patience, min_delta=min_delta)
        
        return self.history


# ============================================================================
# 7. CLINICAL-GRADE CLASSIFIER
# ============================================================================

class ClinicalGradeClassifier(BaseClassifier):
    
    def build_model(self):
        self.evidential_model = UniversalEvidentialModel(
            self.model_name,
            num_classes=self.num_classes,
            pretrained=True
        )
        
        # Center loss
        embedding_dim = 512
        self.center_loss = CenterLoss(self.num_classes, embedding_dim, lambda_c=0.1)
        
        # Manifold mixup
        self.manifold_mixup = ManifoldMixup(alpha=0.2)
        
        # Losses
        self.evidential_loss = EvidentialLoss(self.num_classes, lam=0.5)
        
        self.model = self.evidential_model
        self.mixup_enabled = True
    
    def forward(self, images, labels=None):
        features = self.evidential_model.feature_extractor(images)
        
        # Apply manifold mixup during training
        if self.training and self.mixup_enabled and labels is not None:
            features, labels_a, labels_b, lam = self.manifold_mixup(features, labels)
            evidence = self.evidential_model.evidential_head(features)
            return evidence, features, labels_a, labels_b, lam
        else:
            evidence = self.evidential_model.evidential_head(features)
            return evidence, features
    
    def compute_loss(self, outputs, labels):
        if len(outputs) == 5:
            evidence, features, labels_a, labels_b, lam = outputs
            loss_ev = manifold_mixup_loss(self.evidential_loss, evidence, labels_a, labels_b, lam)
            loss_center = self.center_loss(features, labels)
        else:
            evidence, features = outputs
            loss_ev = self.evidential_loss(evidence, labels)
            loss_center = self.center_loss(features, labels)
        
        return loss_ev + loss_center
    
    def get_predictions(self, outputs):
        if isinstance(outputs, tuple):
            evidence = outputs[0]
        else:
            evidence = outputs
        
        probs, _, _ = self.evidential_model.get_predictions_and_uncertainty(evidence)
        return torch.argmax(probs, dim=1)
    
    def fit(self, train_loader, val_loader, epochs: int = 30, lr: float = 1e-4,
            use_sam: bool = True, primary_metric: str = 'recall',
            patience: int = 10, min_delta: float = 0.001):
        return super().fit(train_loader, val_loader, epochs, lr, use_sam=True,
                          primary_metric=primary_metric, patience=patience, min_delta=min_delta)
    
    def train_epoch(self, train_loader, optimizer, scaler=None, scheduler=None):
        self.model.train()
        self.mixup_enabled = True
        running_loss = 0.0
        all_preds, all_labels = [], []
        
        from .techniques import SAM
        is_sam = isinstance(optimizer, SAM)
        
        for images, labels in train_loader:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            
            if is_sam:
                outputs = self.forward(images, labels)
                loss = self.compute_loss(outputs, labels)
                loss.backward()
                optimizer.first_step(zero_grad=True)
                
                outputs = self.forward(images, labels)
                loss = self.compute_loss(outputs, labels)
                loss.backward()
                optimizer.second_step(zero_grad=True)
            else:
                outputs = self.forward(images, labels)
                loss = self.compute_loss(outputs, labels)
                loss.backward()
                optimizer.step()
            
            if scheduler and isinstance(scheduler, (
                torch.optim.lr_scheduler.OneCycleLR,
                torch.optim.lr_scheduler.SequentialLR
            )):
                scheduler.step()
            
            running_loss += loss.detach().item() * images.size(0)
            
            with torch.no_grad():
                preds = self.get_predictions(outputs)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        from sklearn.metrics import accuracy_score, recall_score
        avg_loss = running_loss / len(train_loader.dataset)
        acc = accuracy_score(all_labels, all_preds) * 100
        recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
        
        return avg_loss, acc, recall


# ============================================================================
# 8. HYBRID TRANSFORMER CLASSIFIER
# ============================================================================

class HybridTransformerClassifier(BaseClassifier):
    
    def build_model(self):
        # Only works with CNN-based models
        if 'vit' in self.model_name.lower() or 'swin' in self.model_name.lower():
            print(f"Warning: HybridTransformer works best with CNN models. Using {self.model_name} as-is.")
            self.model = timm.create_model(self.model_name, pretrained=True, num_classes=self.num_classes)
            return
        
        # CNN backbone
        cnn_base = timm.create_model(self.model_name, pretrained=True, num_classes=self.num_classes)
        
        # Extract CNN features (before classifier)
        if hasattr(cnn_base, 'fc'):
            self.cnn_features = nn.Sequential(*list(cnn_base.children())[:-2])
            in_features = cnn_base.fc.in_features
        elif hasattr(cnn_base, 'head'):
            self.cnn_features = nn.Sequential(*list(cnn_base.children())[:-2])
            in_features = cnn_base.head.in_features
        else:
            self.cnn_features = nn.Sequential(*list(cnn_base.children())[:-1])
            in_features = 512
        
        # Simple transformer layer
        self.transformer = nn.TransformerEncoderLayer(
            d_model=in_features,
            nhead=8,
            dim_feedforward=in_features * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        
        # Classifier
        self.classifier = nn.Linear(in_features, self.num_classes)
        
        self.model = nn.ModuleDict({
            'cnn_features': self.cnn_features,
            'transformer': self.transformer,
            'classifier': self.classifier
        })
    
    def forward(self, images):
        # CNN features
        features = self.cnn_features(images)
        
        # Global average pooling
        if len(features.shape) == 4:
            features = F.adaptive_avg_pool2d(features, 1).flatten(1)
        
        # Add batch dimension for transformer
        features = features.unsqueeze(1)  # [B, 1, C]
        
        # Transformer
        features = self.transformer(features)
        features = features.squeeze(1)  # [B, C]
        
        # Classify
        logits = self.classifier(features)
        return logits
    
    def compute_loss(self, outputs, labels):
        return F.cross_entropy(outputs, labels)


# ============================================================================
# 9. ULTIMATE RECALL-OPTIMIZED CLASSIFIER
# ============================================================================

class UltimateRecallOptimizedClassifier(BaseClassifier):
    
    def build_model(self):
        # Base model with SE blocks
        base = timm.create_model(self.model_name, pretrained=True, num_classes=self.num_classes)
        
        # Add SE blocks
        if hasattr(base, 'layer4'):
            for block in base.layer4:
                if hasattr(block, 'conv2'):
                    channels = block.conv2.out_channels
                    block.se = SEBlock(channels, reduction=16)
        
        # Extract feature extractor
        if hasattr(base, 'fc'):
            in_features = base.fc.in_features
            self.feature_extractor = nn.Sequential(*list(base.children())[:-1], nn.Flatten())
        elif hasattr(base, 'head'):
            in_features = base.head.in_features
            self.feature_extractor = nn.Sequential(*list(base.children())[:-1], nn.Flatten())
        else:
            in_features = 512
            self.feature_extractor = nn.Sequential(*list(base.children())[:-1], nn.Flatten())
        
        # Embedding layer
        self.embedding_dim = 256
        self.embedding = nn.Sequential(
            nn.Linear(in_features, self.embedding_dim),
            nn.BatchNorm1d(self.embedding_dim),
            nn.ReLU()
        )
        
        # Evidential head
        self.evidential_head = EvidentialLayer(self.embedding_dim, self.num_classes)
        
        # Cosine classifier (alternative)
        self.cosine_classifier = CosineClassifier(self.embedding_dim, self.num_classes, scale=30.0)
        
        # Losses
        self.evidential_loss = EvidentialLoss(self.num_classes, lam=0.5)
        self.center_loss = CenterLoss(self.num_classes, self.embedding_dim, lambda_c=0.1)
        
        # Regularizers
        self.manifold_mixup = ManifoldMixup(alpha=0.2)
        
        self.model = nn.ModuleDict({
            'feature_extractor': self.feature_extractor,
            'embedding': self.embedding,
            'evidential_head': self.evidential_head,
            'cosine_classifier': self.cosine_classifier,
            'center_loss': self.center_loss
        })
        
        self.mixup_enabled = True
        self.use_evidential = True
    
    def forward(self, images, labels=None):
        # Features
        features = self.feature_extractor(images)
        embeddings = self.embedding(features)
        
        # Apply manifold mixup during training
        if self.training and self.mixup_enabled and labels is not None:
            embeddings, labels_a, labels_b, lam = self.manifold_mixup(embeddings, labels)
            
            if self.use_evidential:
                evidence = self.evidential_head(embeddings)
                return evidence, embeddings, labels_a, labels_b, lam
            else:
                logits = self.cosine_classifier(embeddings)
                return logits, embeddings, labels_a, labels_b, lam
        else:
            if self.use_evidential:
                evidence = self.evidential_head(embeddings)
                return evidence, embeddings
            else:
                logits = self.cosine_classifier(embeddings)
                return logits, embeddings
    
    def compute_loss(self, outputs, labels):
        if len(outputs) == 5:
            pred, embeddings, labels_a, labels_b, lam = outputs
            
            if self.use_evidential:
                loss_pred = lam * self.evidential_loss(pred, labels_a) + \
                           (1 - lam) * self.evidential_loss(pred, labels_b)
            else:
                loss_pred = lam * F.cross_entropy(pred, labels_a) + \
                           (1 - lam) * F.cross_entropy(pred, labels_b)
            
            loss_center = self.center_loss(embeddings, labels)
        else:
            pred, embeddings = outputs
            
            if self.use_evidential:
                loss_pred = self.evidential_loss(pred, labels)
            else:
                loss_pred = F.cross_entropy(pred, labels)
            
            loss_center = self.center_loss(embeddings, labels)
        
        return loss_pred + loss_center
    
    def get_predictions(self, outputs):
        if isinstance(outputs, tuple):
            pred = outputs[0]
        else:
            pred = outputs
        
        if self.use_evidential:
            alpha = pred + 1
            S = alpha.sum(dim=1, keepdim=True)
            probs = alpha / S
            return torch.argmax(probs, dim=1)
        else:
            return torch.argmax(pred, dim=1)
    
    def fit(self, train_loader, val_loader, epochs: int = 40, lr: float = 1e-4,
            use_sam: bool = True, primary_metric: str = 'recall',
            patience: int = 12, min_delta: float = 0.0005):
        return super().fit(train_loader, val_loader, epochs, lr, use_sam=True,
                          primary_metric=primary_metric, patience=patience, min_delta=min_delta)
    
    def train_epoch(self, train_loader, optimizer, scaler=None, scheduler=None):
        self.model.train()
        self.mixup_enabled = True
        running_loss = 0.0
        all_preds, all_labels = [], []
        
        from .techniques import SAM
        is_sam = isinstance(optimizer, SAM)
        
        for images, labels in train_loader:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            
            if is_sam:
                outputs = self.forward(images, labels)
                loss = self.compute_loss(outputs, labels)
                loss.backward()
                optimizer.first_step(zero_grad=True)
                
                outputs = self.forward(images, labels)
                loss = self.compute_loss(outputs, labels)
                loss.backward()
                optimizer.second_step(zero_grad=True)
            else:
                outputs = self.forward(images, labels)
                loss = self.compute_loss(outputs, labels)
                loss.backward()
                optimizer.step()
            
            if scheduler and isinstance(scheduler, (
                torch.optim.lr_scheduler.OneCycleLR,
                torch.optim.lr_scheduler.SequentialLR
            )):
                scheduler.step()
            
            running_loss += loss.detach().item() * images.size(0)
            
            with torch.no_grad():
                preds = self.get_predictions(outputs)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        from sklearn.metrics import accuracy_score, recall_score
        avg_loss = running_loss / len(train_loader.dataset)
        acc = accuracy_score(all_labels, all_preds) * 100
        recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
        
        return avg_loss, acc, recall


# Classifier registry
CLASSIFIER_MAP = {
    'baseline': BaselineClassifier,
    'progressive': ProgressiveClassifier,
    'evidential': EvidentialClassifier,
    'metric_learning': MetricLearningClassifier,
    'regularized': RegularizedClassifier,
    'attention_enhanced': AttentionEnhancedClassifier,
    'progressive_evidential': ProgressiveEvidentialClassifier,
    'clinical_grade': ClinicalGradeClassifier,
    'hybrid_transformer': HybridTransformerClassifier,
    'ultimate': UltimateRecallOptimizedClassifier,
}


def test_all_classifiers_on_model(
    model_name: str,
    train_loader,
    val_loader,
    test_loader,
    classifiers: List[str] = 'all',
    class_names: Optional[List[str]] = None,
    epochs: int = 30,
    lr: float = 1e-4,
    primary_metric: str = 'recall',
    device: str = 'cuda',
    save_dir: str = './classifier_comparison'
):
    
    # Create save directory
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    # Default class names
    if class_names is None:
        class_names = ['NonDemented', 'VeryMildDemented', 'MildDemented', 'ModerateDemented']
    
    # Handle 'all' option
    if classifiers == 'all':
        classifiers_to_test = list(CLASSIFIER_MAP.keys())
    else:
        classifiers_to_test = [c.lower().replace(' ', '_') for c in classifiers]
    
    # Validate classifier names
    invalid = [c for c in classifiers_to_test if c not in CLASSIFIER_MAP]
    if invalid:
        available = ', '.join(CLASSIFIER_MAP.keys())
        raise ValueError(f"Invalid classifiers: {invalid}. Available: {available}")
    
    # Results storage
    results = []
    
    print("\n" + "="*100)
    print(f"TESTING MULTIPLE CLASSIFIERS ON: {model_name.upper()}")
    print("="*100)
    print(f"Primary Metric: {primary_metric.upper()}")
    print(f"Testing {len(classifiers_to_test)} classifiers")
    print(f"Device: {device}")
    print("="*100 + "\n")
    
    # Test each classifier
    for clf_name in classifiers_to_test:
        print("\n" + "█"*100)
        print(f"█{'':^98}█")
        print(f"█{f'TESTING: {clf_name.upper()} on {model_name}':^98}█")
        print(f"█{'':^98}█")
        print("█"*100 + "\n")
        
        try:
            # Create classifier
            start_time = time.time()
            
            clf_class = CLASSIFIER_MAP[clf_name]
            classifier = clf_class(
                model_name=model_name,
                num_classes=len(class_names),
                device=device
            )
            
            # Train
            print(f"\n{'─'*100}")
            print("TRAINING PHASE")
            print(f"{'─'*100}\n")
            
            # Determine if should use SAM
            use_sam = clf_name in ['clinical_grade', 'ultimate']
            
            history = classifier.fit(
                train_loader=train_loader,
                val_loader=val_loader,
                epochs=epochs,
                lr=lr,
                use_sam=use_sam,
                primary_metric=primary_metric,
                patience=10,
                min_delta=0.001 if clf_name != 'ultimate' else 0.0005
            )
            
            training_time = time.time() - start_time
            
            # Test
            print(f"\n{'─'*100}")
            print("TESTING PHASE")
            print(f"{'─'*100}\n")
            
            test_results = classifier.evaluate(test_loader, class_names=class_names)
            
            # Store results
            result_entry = {
                'Classifier': clf_name,
                'Model': model_name,
                f'Test_{primary_metric.capitalize()}': test_results.get(primary_metric, 0.0),
                'Test_Recall': test_results['recall'],
                'Test_Accuracy': test_results['accuracy'],
                'Test_Precision': test_results['precision'],
                'Test_F1': test_results['f1'],
                f'Best_Val_{primary_metric.capitalize()}': classifier.best_metric_value,
                'Best_Val_Recall': classifier.best_recall,
                'Best_Val_Accuracy': classifier.best_acc,
                'Training_Time_Min': training_time / 60,
                'Total_Epochs': len(history) if isinstance(history, list) else epochs,
            }
            
            # Per-class recall
            for i, class_name in enumerate(class_names):
                result_entry[f'{class_name}_Recall'] = test_results['per_class_recall'][i]
            
            results.append(result_entry)
            
            # Save model
            save_path = f"{save_dir}/{clf_name}_{model_name}.pth"
            classifier.save(save_path)
            print(f"\n✓ Model saved to: {save_path}")
            
            # Success message
            print(f"\n{'╔'*100}")
            print(f"✓ {clf_name.upper()} COMPLETE")
            print(f"  Test {primary_metric.capitalize()}: {test_results[primary_metric if primary_metric in test_results else 'recall']:.4f} ★")
            print(f"  Test Accuracy: {test_results['accuracy']:.2f}%")
            print(f"  Training Time: {training_time/60:.1f} minutes")
            print(f"{'╚'*100}\n")
            
        except Exception as e:
            print(f"\n✗ ERROR testing {clf_name}: {str(e)}")
            import traceback
            traceback.print_exc()
            
            # Record failure
            results.append({
                'Classifier': clf_name,
                'Model': model_name,
                f'Test_{primary_metric.capitalize()}': 0.0,
                'Test_Recall': 0.0,
                'Test_Accuracy': 0.0,
                'Test_Precision': 0.0,
                'Test_F1': 0.0,
                f'Best_Val_{primary_metric.capitalize()}': 0.0,
                'Best_Val_Recall': 0.0,
                'Best_Val_Accuracy': 0.0,
                'Training_Time_Min': 0.0,
                'Total_Epochs': 0,
                'Error': str(e)
            })

        finally :
            # Cleanup
            del classifier
            torch.cuda.empty_cache()
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Sort by primary metric
    sort_col = f'Test_{primary_metric.capitalize()}'
    if sort_col not in results_df.columns:
        sort_col = 'Test_Recall'
    
    results_df = results_df.sort_values(sort_col, ascending=False).reset_index(drop=True)
    
    # Add rank
    results_df.insert(0, 'Rank', range(1, len(results_df) + 1))
    
    # Save results
    results_path = f"{save_dir}/comparison_{model_name}.csv"
    results_df.to_csv(results_path, index=False)
    
    # Print summary
    print("\n" + "="*100)
    print(f"FINAL RESULTS SUMMARY - {model_name.upper()} (Ranked by {primary_metric.upper()})")
    print("="*100 + "\n")
    
    # Display table
    display_cols = ['Rank', 'Classifier', sort_col, 'Test_Accuracy', 'Test_F1', 'Training_Time_Min']
    print(results_df[display_cols].to_string(index=False))
    
    print(f"\n{'='*100}")
    print(f"🏆 WINNER: {results_df.iloc[0]['Classifier'].upper()}")
    print(f"   {primary_metric.capitalize()}: {results_df.iloc[0][sort_col]:.4f}")
    print(f"   Accuracy: {results_df.iloc[0]['Test_Accuracy']:.2f}%")
    print(f"{'='*100}")
    
    print(f"\n📊 Full results saved to: {results_path}\n")
    
    return results_df


def test_single_classifier(
    classifier_name: str,
    model_name: str,
    train_loader,
    val_loader,
    test_loader,
    class_names: Optional[List[str]] = None,
    epochs: int = 30,
    lr: float = 1e-4,
    primary_metric: str = 'recall',
    device: str = 'cuda'
):
    
    classifier_name = classifier_name.lower().replace(' ', '_')
    
    if classifier_name not in CLASSIFIER_MAP:
        available = ', '.join(CLASSIFIER_MAP.keys())
        raise ValueError(f"Unknown classifier: '{classifier_name}'. Available: {available}")
    
    if class_names is None:
        class_names = ['NonDemented', 'VeryMildDemented', 'MildDemented', 'ModerateDemented']
    
    print(f"\n{'='*80}")
    print(f"Testing: {classifier_name.upper()} on {model_name}")
    print(f"{'='*80}\n")
    
    # Create classifier
    clf_class = CLASSIFIER_MAP[classifier_name]
    classifier = clf_class(
        model_name=model_name,
        num_classes=len(class_names),
        device=device
    )
    
    # Train
    use_sam = classifier_name in ['clinical_grade', 'ultimate']
    
    history = classifier.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        lr=lr,
        use_sam=use_sam,
        primary_metric=primary_metric,
        patience=10
    )
    
    # Test
    test_results = classifier.evaluate(test_loader, class_names=class_names)
    
    return test_results, classifier


def compare_classifiers(results_df: pd.DataFrame, save_path: Optional[str] = None):
    # Set style
    sns.set_style("whitegrid")
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Test Recall comparison
    ax = axes[0, 0]
    sns.barplot(data=results_df, x='Test_Recall', y='Classifier', ax=ax, palette='viridis')
    ax.set_title('Test Recall (PRIMARY METRIC)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Recall')
    ax.axvline(x=0.99, color='r', linestyle='--', label='Target (0.99)')
    ax.legend()
    
    # 2. Test Accuracy comparison
    ax = axes[0, 1]
    sns.barplot(data=results_df, x='Test_Accuracy', y='Classifier', ax=ax, palette='rocket')
    ax.set_title('Test Accuracy', fontsize=14, fontweight='bold')
    ax.set_xlabel('Accuracy (%)')
    
    # 3. Test F1 comparison
    ax = axes[1, 0]
    sns.barplot(data=results_df, x='Test_F1', y='Classifier', ax=ax, palette='mako')
    ax.set_title('Test F1 Score', fontsize=14, fontweight='bold')
    ax.set_xlabel('F1 Score')
    
    # 4. Training time comparison
    ax = axes[1, 1]
    sns.barplot(data=results_df, x='Training_Time_Min', y='Classifier', ax=ax, palette='flare')
    ax.set_title('Training Time', fontsize=14, fontweight='bold')
    ax.set_xlabel('Time (minutes)')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n📊 Comparison plot saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()


def list_available_classifiers():
    print("\n" + "="*80)
    print("AVAILABLE CLASSIFIERS (11 Total)")
    print("="*80)
    
    print("\nADVANCED (Research-Grade):")
    advanced = [
        ('baseline', 'Standard CrossEntropy'),
        ('evidential', 'Uncertainty quantification'),
        ('metric_learning', 'Prototypes + Triplet + Center Loss'),
        ('regularized', 'Manifold Mixup + Label Smoothing'),
        ('attention_enhanced', 'SE Blocks + Cosine Classifier'),
        ('progressive', 'Sophisticated 3-phase fine-tuning'),
        ('progressive_evidential', 'Progressive + Evidential'),
        ('clinical_grade', 'Clinical deployment (5 techniques + SAM) ⭐'),
        ('hybrid_transformer', 'CNN + Transformer hybrid'),
        ('ultimate', 'All 10 techniques (maximum recall) ⭐⭐⭐')
    ]
    
    for i, (name, desc) in enumerate(advanced, 3):
        print(f"  {i}. {name:<22} - {desc}")
    
    print("\n" + "="*80)
    print("Usage Examples:")
    print("  test_all_classifiers_on_model('resnet18', ..., classifiers='all')")
    print("  test_all_classifiers_on_model('resnet18', ..., classifiers=['simple', 'progressive'])")
    print("  test_single_classifier('ultimate', 'resnet18', ...)")
    print("="*80 + "\n")
    
    return list(CLASSIFIER_MAP.keys())


# Convenience function
def get_classifier_info(classifier_name: str):
    classifier_name = classifier_name.lower().replace(' ', '_')
    
    if classifier_name not in CLASSIFIER_MAP:
        available = ', '.join(CLASSIFIER_MAP.keys())
        raise ValueError(f"Unknown classifier: '{classifier_name}'. Available: {available}")
    
    info_map = {
        'baseline': {
            'name': 'BaselineClassifier',
            'category': 'Basic',
            'description': 'Basic timm wrapper with AdamW + OneCycleLR',
            'speed': 'Fast',
            'use_case': 'Simple baseline for comparisons'
        },
        'progressive': {
            'name': 'ProgressiveClassifier',
            'category': 'Basic',
            'description': 'Sophisticated 3-phase discriminative fine-tuning',
            'speed': 'Medium',
            'use_case': 'High-quality training with architecture awareness'
        },
        'evidential': {
            'name': 'EvidentialClassifier',
            'category': 'Advanced',
            'description': 'Evidential deep learning for uncertainty quantification',
            'speed': 'Medium',
            'use_case': 'When you need uncertainty scores'
        },
        'ultimate': {
            'name': 'UltimateRecallOptimizedClassifier',
            'category': 'Advanced',
            'description': 'All 10 research techniques combined',
            'speed': 'Slow',
            'use_case': 'Maximum recall for critical medical diagnosis'
        },
        # Add more as needed...
    }
    
    return info_map.get(classifier_name, {
        'name': CLASSIFIER_MAP[classifier_name].__name__,
        'category': 'Advanced',
        'description': 'Research-grade classifier',
        'speed': 'Medium',
        'use_case': 'Advanced training'
    })


# Classifier registry for string-based access
CLASSIFIER_REGISTRY = {
    'baseline': BaselineClassifier,
    'evidential': EvidentialClassifier,
    'metric_learning': MetricLearningClassifier,
    'regularized': RegularizedClassifier,
    'attention_enhanced': AttentionEnhancedClassifier,
    'progressive': ProgressiveClassifier,
    'progressive_evidential': ProgressiveEvidentialClassifier,
    'clinical_grade': ClinicalGradeClassifier,
    'hybrid_transformer': HybridTransformerClassifier,
    'ultimate': UltimateRecallOptimizedClassifier,
}


def get_classifier(classifier_type: str = 'simple') :
    classifier_type = classifier_type.lower().replace(' ', '_')
    
    if classifier_type not in CLASSIFIER_REGISTRY:
        available = ', '.join(CLASSIFIER_REGISTRY.keys())
        raise ValueError(
            f"Unknown classifier type: '{classifier_type}'. "
            f"Available: {available}"
        )
    
    return CLASSIFIER_REGISTRY[classifier_type]


def list_classifiers(display : bool = False):
    print("\n" + "="*80)
    print("AVAILABLE CLASSIFIERS")
    print("="*80)
    
    print("\nADVANCED:")
    print("  baseline            - Standard CrossEntropy")
    print("  evidential          - Uncertainty quantification")
    print("  metric_learning     - Prototypes + Triplet + Center Loss")
    print("  regularized         - Manifold Mixup + Label Smoothing")
    print("  attention_enhanced  - SE Blocks + Cosine Classifier")
    print("  progressive         - Sophisticated 3-phase fine-tuning")
    print("  progressive_evidential - Progressive + Evidential")
    print("  clinical_grade      - Clinical deployment (5 techniques) ⭐")
    print("  hybrid_transformer  - CNN + Transformer hybrid")
    print("  ultimate            - All 10 techniques (maximum recall) ⭐⭐⭐")
    
    print("\n" + "="*80 + "\n")
    
    return list(CLASSIFIER_REGISTRY.keys())


def test_model(model_name, model, loader, classes, experiment_name, logger, history = None,
               visualizer=None, use_tta=False):
    
    # Create visualizer (test.py creates its own - cross_validation doesn't need to!)
    if not visualizer :
        img_size = get_img_size(model_name)
        transform = get_base_transformations(img_size)
        visualizer = Visualizer(
            experiment_name=experiment_name,
            model_name=model_name,
            class_names=classes,
            transform=transform,
            logger=logger
        )
    
    
    # Check if this is a classifier object (has evaluate method)
    if hasattr(model, 'evaluate'):
        logger.info(f"--- Detected Classifier Object: Using classifier.evaluate() ---")
        
        # Use classifier's built-in evaluate method
        result = model.evaluate(loader, class_names=classes)
        y_true = np.array(result["labels"])
        y_pred = np.array(result["preds"])
        y_prob = np.array(result["probs"])
        accuracy = result["accuracy"]
        recall = result["recall"]
        precision = result["precision"]
        f1 = result["f1"]
        per_class_recall = result["per_class_recall"]
        cm = result["confusion_matrix"]
        classification_report_ = result["report"]
        pytorch_model = model.model if hasattr(model, 'model') else None
    
    else:
        # Raw PyTorch model - use legacy test logic
        logger.info(f"--- Detected Raw PyTorch Model: Using standard testing ---")
        
        from .augmentation import TTAWrapper
        
        model.eval()
        all_preds = []
        all_labels = []
        all_probs = []
        
        if use_tta:
            logger.info("Using Test-Time Augmentation (TTA)")
            tta_model = TTAWrapper(model, num_augmentations=5)
        
        # Inference Loop
        with torch.inference_mode():
            for images, labels in loader:
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)
                
                if use_tta:
                    probs = tta_model(images)
                    preds = torch.argmax(probs, dim=1)
                else:
                    outputs = model(images)
                    probs = torch.softmax(outputs, dim=1)
                    preds = torch.argmax(outputs, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        # Convert to numpy
        y_true = np.array(all_labels)
        y_pred = np.array(all_preds)
        y_prob = np.array(all_probs)
        
        # Calculate Metrics
        accuracy = accuracy_score(y_true, y_pred) * 100
        recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
        precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
        f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        per_class_recall = recall_score(all_labels, all_preds, average=None, zero_division=0)
        cm = confusion_matrix(y_true, y_pred)
        classification_report_ = classification_report(y_true, y_pred, target_names=classes, digits=4)
        pytorch_model = model
    
    # NaN handling
    if np.isnan(y_prob).any():
        logger.warning("⚠️  NaN detected in probability outputs!")
        n_classes = len(classes)
        y_prob = np.nan_to_num(y_prob, nan=1.0/n_classes)
    
    # Verify probabilities sum to 1.0
    prob_sums = y_prob.sum(axis=1)
    if not np.allclose(prob_sums, 1.0, atol=1e-3):
        logger.warning("⚠️  Normalizing probabilities to sum=1.0")
        y_prob = y_prob / prob_sums[:, np.newaxis]
    
    # ROC AUC
    try:
        if len(classes) == 2:
            roc_auc = roc_auc_score(y_true, y_prob[:, 1])
        else:
            roc_auc = roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')
    except:
        logger.warning(f"Unexpected error in ROC AUC calculation: {e}")
        roc_auc = 0.0
    
    kappa = cohen_kappa_score(y_true, y_pred)
    corrcoef = matthews_corrcoef(y_true, y_pred)
    jaccard = jaccard_score(y_true, y_pred, average="weighted")
    
    # Generate Report
    report_path = os.path.join(REPORTS_DIR, f"{experiment_name}.txt")
    
    with open(report_path, "w") as f:
        f.write(f"===== COMPREHENSIVE ANALYSIS REPORT: {experiment_name} =====\n\n")
        f.write(f"Test-Time Augmentation: {'Enabled' if use_tta else 'Disabled'}\n\n")
        f.write("--- Overall Performance ---\n")
        f.write(f"Overall Accuracy: {accuracy:.2f}%\n")
        f.write(f"Overall Recall: {recall:.4f}\n")
        f.write(f"Overall Precision: {precision:.4f}\n")
        f.write(f"Overall F1: {f1:.4f}\n")
        f.write(f"\nPer-Class Recall")
        for i, (name, rec) in enumerate(zip(classes, per_class_recall)):
            f.write(f"  {name}: {rec:.4f}")
        f.write(f"Macro ROC AUC:    {roc_auc:.4f}\n")
        f.write(f"Cohen's Kappa:    {kappa:.4f}\n")
        f.write(f"Matthews Correlation Coefficient (MCC): {corrcoef:.4f}\n")
        f.write(f"Jaccard Score:    {jaccard:.4f}\n\n")
        
        f.write("--- Detailed Per-Class Metrics ---\n")
        f.write(classification_report_)
        
        f.write("\n--- Per-Class Specificity & Confusion Matrix Stats ---\n")
        
        for i, class_name in enumerate(classes):
            tp = cm[i, i]
            fp = cm[:, i].sum() - tp
            fn = cm[i, :].sum() - tp
            tn = cm.sum() - (tp + fp + fn)
            
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            f.write(f"{class_name:<20}: Specificity: {specificity:.4f}, "
                    f"TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn}\n")
    
    logger.info(f"Report saved to: {report_path}")
    
    # Generate ALL visualizations
    logger.info("Generating comprehensive visualizations...")
    
    visualizer.generate_all_plots(
        y_true=y_true,
        y_prob=y_prob,
        history=history,
        model=pytorch_model,
        test_loader=loader,
        xai_samples=NUM_SAMPLES_TO_ANALYSE  # Enable GradCAM/LIME/SHAP
    )
    
    logger.info("Testing and visualization complete")
    
    return {
        'accuracy': accuracy,
        'recall': recall,
        'precision': precision,
        'f1': f1,
        'per_class_recall': per_class_recall,
        'roc_auc': roc_auc,
        'kappa': kappa,
        'corrcoef': corrcoef,
        'jaccard': jaccard,
        'y_true': y_true,
        'y_prob': y_prob,
        'y_pred': y_pred,
    }


class Cross_Validator:
    
    def __init__(self, model_names, logger: Logger, model_classifier_map=None):
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
            classifier_type_input = self.model_classifier_map.get(model_name, 'baseline')

            if classifier_type_input == "all" :
                classifiers_to_be_used = list_classifiers()
            elif isinstance(classifier_type_input, str) :
                classifiers_to_be_used = [classifier_type_input]
            
            for classifier_type in classifiers_to_be_used :
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


def run_batch():
    models = __models_list__
    classifier_map = __classifier_map__

    logger = Logger("batch_" + str(hash(str(models)))[:8])
    logger.info(f"Starting validation for {models}")
    logger.info(f"Classifier mapping: {classifier_map}")
    
    try:
        validator = Cross_Validator(
            models,
            logger,
            model_classifier_map=classifier_map
        )
        validator.run()
        logger.info("Validation complete")
    except Exception as e:
        logger.error(f"Batch failed: {e}")
        raise

if __name__ == "__main__":
    run_batch()