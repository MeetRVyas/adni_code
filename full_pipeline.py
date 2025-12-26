import os
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms, datasets
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
import os
import logging
import shutil

import seaborn as sns
import pandas as pd
import numpy as np
import torch.nn.functional as F
from itertools import cycle
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    roc_curve, auc, precision_recall_curve, average_precision_score,
    confusion_matrix, classification_report
)

import torch.nn as nn
import torch.optim as optim
import gc
from sklearn.model_selection import train_test_split, StratifiedKFold

# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Directory structure
OUTPUT_DIR = "output"
RESULTS_DIR = os.path.join(OUTPUT_DIR, "results")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
REPORTS_DIR = os.path.join(RESULTS_DIR, "reports")
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")
DATA_DIR = "OriginalDataset"

# Training hyperparameters
EPOCHS = 5
NFOLDS = 3
BATCH_SIZE = 32
NUM_WORKERS = 4  # Reduced from 8 to prevent CPU bottleneck
PRETRAINED = True
NUM_SAMPLES_TO_ANALYSE = 5
TEST_SPLIT = 0.2

# Optimization settings
USE_AMP = True  # Automatic Mixed Precision
PIN_MEMORY = True
PERSISTENT_WORKERS = True  # Keep workers alive between epochs

# Memory management
EMPTY_CACHE_FREQUENCY = 1  # Clear cache every N epochs
SAVE_BEST_ONLY = True  # Only save best checkpoint, not every improvement

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

    def error(self, message: str) -> None:
        self.logger.error(message, exc_info=True)
    
    def debug(self, message: str) -> None:
        self.logger.debug(message)


def get_gpu_augmentations(img_size):
    return torch.nn.Sequential(
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    ).to(DEVICE)


def train_one_epoch(model, loader, criterion, optimizer, scaler, gpu_augmenter=None, scheduler=None):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    for images, labels in tqdm(loader, desc="   Training", leave=False):
        images, labels = images.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)
        
        if gpu_augmenter:
            images = gpu_augmenter(images)
        
        optimizer.zero_grad(set_to_none=True)
        
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=(DEVICE == 'cuda')):
            outputs = model(images)
            loss = criterion(outputs, labels)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if scheduler:
            scheduler.step()

        running_loss += loss.item() * images.size(0)
        
        # Critical fix: detach before converting to prevent memory leak
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
        for images, labels in tqdm(loader, desc="   Validating", leave=False):
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
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def __call__(self, input_tensor):
        self.model.zero_grad()
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

    def plot_training_history(self, history):
        if not history:
            return

        try:
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
            self.log(f"✓ Training history saved")
            
        except Exception as e:
            self.log(f"Training history plot failed: {e}")
            if self.logger:
                self.logger.error(f"Training history visualization error: {e}")
            plt.close('all')

    def plot_confusion_matrix(self, y_true, y_pred, normalize=False):
        try:
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
            self.log(f"✓ Confusion matrix saved")
            
        except Exception as e:
            self.log(f"Confusion matrix plot failed: {e}")
            if self.logger:
                self.logger.error(f"Confusion matrix visualization error: {e}")
            plt.close('all')

    def plot_classwise_metrics(self, y_true, y_pred):
        try:
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
            
            if not metrics_list:
                self.log("No valid class metrics to plot")
                return
            
            df = pd.DataFrame(metrics_list).set_index('Class')
            
            plt.figure(figsize=(8, len(self.class_names) * 0.8 + 2))
            sns.heatmap(df, annot=True, cmap='RdYlGn', fmt='.3f', vmin=0.0, vmax=1.0, linewidths=1)
            plt.title('Class-wise Performance Metrics', fontsize=14)
            plt.yticks(rotation=0)
            
            save_path = os.path.join(self.save_dir, "classwise_metrics.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            self.log(f"✓ Class-wise metrics saved")
            
        except Exception as e:
            self.log(f"Class-wise metrics plot failed: {e}")
            if self.logger:
                self.logger.error(f"Class-wise metrics visualization error: {e}")
            plt.close('all')

    def plot_roc_curve(self, y_true, y_prob):
        try:
            n_classes = len(self.class_names)
            y_true_bin = pd.get_dummies(y_true).values
            
            plt.figure(figsize=(10, 8))
            colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green', 'red', 'purple'])
            
            for i, color in zip(range(n_classes), colors):
                if i < y_prob.shape[1]:
                    fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
                    auc_score = auc(fpr, tpr)
                    plt.plot(fpr, tpr, color=color, lw=2, 
                             label=f'{self.class_names[i]} (AUC = {auc_score:0.2f})')
            
            plt.plot([0, 1], [0, 1], 'k--', lw=2)
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curves: {self.experiment_name}')
            plt.legend(loc="lower right")
            plt.grid(True, alpha=0.3)
            
            save_path = os.path.join(self.save_dir, "roc_curve.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            self.log(f"✓ ROC curves saved")
            
        except Exception as e:
            self.log(f"ROC curve plot failed: {e}")
            if self.logger:
                self.logger.error(f"ROC curve visualization error: {e}")
            plt.close('all')

    def plot_precision_recall_curve(self, y_true, y_prob):
        try:
            n_classes = len(self.class_names)
            y_true_bin = pd.get_dummies(y_true).values
        
            plt.figure(figsize=(10, 8))
            colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green', 'red', 'purple'])
            
            for i, color in zip(range(n_classes), colors):
                if i < y_prob.shape[1]:
                    prec, rec, _ = precision_recall_curve(y_true_bin[:, i], y_prob[:, i])
                    avg_prec = average_precision_score(y_true_bin[:, i], y_prob[:, i])
                    plt.plot(rec, prec, color=color, lw=2, 
                             label=f'{self.class_names[i]} (AP = {avg_prec:0.2f})')
            
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title(f'Precision-Recall Curves: {self.experiment_name}')
            plt.legend(loc="lower left")
            plt.grid(True, alpha=0.3)
            
            save_path = os.path.join(self.save_dir, "precision_recall_curve.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            self.log(f"✓ Precision-Recall curves saved")
            
        except Exception as e:
            self.log(f"Precision-Recall curve plot failed: {e}")
            if self.logger:
                self.logger.error(f"Precision-Recall curve visualization error: {e}")
            plt.close('all')

    def plot_confidence_distribution(self, y_true, y_prob):
        try:
            y_pred = np.argmax(y_prob, axis=1)
            confidences = np.max(y_prob, axis=1)
            
            correct_mask = (y_pred == y_true)
            incorrect_mask = ~correct_mask
            
            plt.figure(figsize=(10, 6))
            
            sns.histplot(confidences[correct_mask], color='green', label='Correct Predictions', 
                         kde=True, bins=20, alpha=0.5, element="step")
            
            if np.any(incorrect_mask):
                sns.histplot(confidences[incorrect_mask], color='red', label='Incorrect Predictions', 
                             kde=True, bins=20, alpha=0.5, element="step")
                
            plt.xlabel("Prediction Confidence")
            plt.ylabel("Count")
            plt.title("Confidence Distribution: Correct vs Incorrect")
            plt.legend()
            
            save_path = os.path.join(self.save_dir, "confidence_analysis.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            self.log(f"✓ Confidence distribution saved")
            
        except Exception as e:
            self.log(f"Confidence distribution plot failed: {e}")
            if self.logger:
                self.logger.error(f"Confidence distribution visualization error: {e}")
            plt.close('all')

    def plot_cumulative_gain(self, y_true, y_prob):
        try:
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
            self.log(f"✓ Cumulative gain saved")
            
        except Exception as e:
            self.log(f"Cumulative gain plot failed: {e}")
            if self.logger:
                self.logger.error(f"Cumulative gain visualization error: {e}")
            plt.close('all')

    def _get_target_layer(self, model):
        try:
            model_type = str(type(model)).lower()
            
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
            else:
                # Fallback: find last Conv2d layer
                for name, module in list(model.named_modules())[::-1]:
                    if isinstance(module, torch.nn.Conv2d):
                        return module
        except Exception as e:
            self.logger.error(f"Error determining target layer: {e}")
        
        return None

    def run_lime(self, model, image_path):
        try:
            from lime import lime_image
            from skimage.segmentation import mark_boundaries
            
            def batch_predict(images):
                pil_images = [Image.fromarray(img.astype('uint8')) for img in images]
                batch = torch.stack([self.transform(img) for img in pil_images], dim=0).to(DEVICE)
                
                with torch.inference_mode():
                    with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=(DEVICE == 'cuda')):
                        logits = model(batch)
                    probs = F.softmax(logits, dim=1)
                
                return probs.cpu().numpy()
            
            image_np = np.array(Image.open(image_path).convert('RGB').resize((self.img_size, self.img_size)))
            explainer = lime_image.LimeImageExplainer()
            
            # Reduced num_samples for speed (default is 1000)
            explanation = explainer.explain_instance(
                image_np, 
                batch_predict, 
                top_labels=1, 
                hide_color=0, 
                num_samples=300  # Reduced from 500 for faster processing
            )
            
            temp, mask = explanation.get_image_and_mask(
                explanation.top_labels[0], 
                positive_only=True, 
                num_features=5, 
                hide_rest=False
            )
            
            lime_viz = mark_boundaries(temp / 255.0, mask)
            return np.clip(lime_viz, 0, 1)
            
        except Exception as e:
            self.log(f"LIME failed: {e}")
            if self.logger:
                self.logger.error(f"LIME visualization error for {image_path}: {e}")
            return None

    def run_shap(self, model, image_path):
        try:
            import shap
            
            # Prepare inputs
            image = Image.open(image_path).convert('RGB').resize((self.img_size, self.img_size))
            input_tensor = self.transform(image).unsqueeze(0).to(DEVICE)
            
            # Background: using random noise (fast) instead of dataset subset
            background = torch.randn(5, 3, self.img_size, self.img_size).to(DEVICE)
            
            explainer = shap.GradientExplainer(model, background)
            shap_values = explainer.shap_values(input_tensor)
            
            # Get predicted class
            with torch.inference_mode():
                output = model(input_tensor)
                pred_idx = torch.argmax(output).item()
            
            # Format for plotting
            shap_numpy = np.swapaxes(np.swapaxes(shap_values[pred_idx], 1, -1), 1, 2)
            input_numpy = np.swapaxes(np.swapaxes(input_tensor.cpu().numpy(), 1, -1), 1, 2)
            
            # Plot to temporary file
            temp_filename = f"temp_shap_{os.getpid()}_{np.random.randint(10000)}.png"
            
            fig = plt.figure(figsize=(4, 4))
            shap.image_plot(shap_numpy, -input_numpy, show=False)
            plt.savefig(temp_filename, bbox_inches='tight', pad_inches=0, dpi=100)
            plt.close(fig)
            
            # Read back as image
            if os.path.exists(temp_filename):
                shap_img = np.array(Image.open(temp_filename).convert('RGB'))
                os.remove(temp_filename)
                return shap_img
            else:
                return None
                
        except Exception as e:
            self.log(f"SHAP failed: {e}")
            if self.logger:
                self.logger.error(f"SHAP visualization error for {image_path}: {e}")
            return None

    def generate_xai_comparison_plot(self, model, image_path, sample_id):
        model.eval()
        
        try:
            # Load original image
            original_img = np.array(Image.open(image_path).convert('RGB').resize((self.img_size, self.img_size)))
            image_basename = os.path.splitext(os.path.basename(image_path))[0]
            
            self.log(f"  Generating XAI comparison for {image_basename}...")
            
            # Get target layer for GradCAM
            target_layer = self._get_target_layer(model)
            
            # Run each XAI method with individual error handling
            xai_results = {
                'original': original_img,
                'gradcam': None,
                'lime': None,
                'shap': None
            }
            
            # GradCAM
            if target_layer is not None:
                try:
                    pil_img = Image.open(image_path).convert('RGB').resize((self.img_size, self.img_size))
                    input_tensor = self.transform(pil_img).unsqueeze(0).to(DEVICE)
                    reshape = reshape_transform_swin if "swin" in str(type(model)).lower() else None
                    xai_results['gradcam'] = generate_gradcam_plot(
                        model, input_tensor, original_img, target_layer, reshape
                    )
                except Exception as e:
                    self.log(f"    GradCAM failed: {e}")
                    if self.logger:
                        self.logger.error(f"GradCAM error for {image_path}: {e}")
            else:
                self.log(f"    Could not determine target layer for GradCAM")
            
            # LIME
            try:
                lime_result = self.run_lime(model, image_path)
                if lime_result is not None:
                    xai_results['lime'] = (lime_result * 255).astype(np.uint8)
            except Exception as e:
                self.log(f"    LIME failed: {e}")
                if self.logger:
                    self.logger.error(f"LIME outer error for {image_path}: {e}")
            
            # SHAP
            try:
                xai_results['shap'] = self.run_shap(model, image_path)
            except Exception as e:
                self.log(f"    SHAP failed: {e}")
                if self.logger:
                    self.logger.error(f"SHAP outer error for {image_path}: {e}")
            
            # Create comparison plot
            fig, axes = plt.subplots(1, 4, figsize=(24, 6))
            titles = ['Original', 'GradCAM', 'LIME', 'SHAP']
            
            for idx, (key, title) in enumerate(zip(['original', 'gradcam', 'lime', 'shap'], titles)):
                if xai_results[key] is not None:
                    axes[idx].imshow(xai_results[key])
                    axes[idx].set_title(title, fontsize=16, color='green')
                else:
                    # Show blank/error placeholder
                    axes[idx].text(0.5, 0.5, f'{title}\n(Failed)', 
                                   ha='center', va='center', fontsize=14, color='red',
                                   transform=axes[idx].transAxes)
                    axes[idx].set_title(title, fontsize=16, color='red')
                
                axes[idx].axis('off')
            
            fig.suptitle(f"XAI Comparison: {image_basename} ({self.model_name})", fontsize=20)
            
            # Save
            output_filename = os.path.join(self.save_dir, f"xai_comparison_{sample_id}_{image_basename}.png")
            plt.savefig(output_filename, bbox_inches='tight', dpi=200)
            plt.close(fig)
            
            self.log(f"    ✓ XAI comparison saved to {output_filename}")
            return output_filename
            
        except Exception as e:
            self.log(f"  XAI comparison plot failed: {e}")
            if self.logger:
                self.logger.error(f"XAI comparison error for {image_path}: {e}")
            return None

    def generate_all_plots(self, y_true, y_prob, history=None, model=None, test_loader=None, xai_samples=5):
        self.log(f"Generating visualizations for {self.experiment_name}...")
        
        y_pred = np.argmax(y_prob, axis=1)
        
        # Track which visualizations succeeded/failed
        results = {
            'xai': 0,
            'training_history': False,
            'confusion_matrix': False,
            'classwise_metrics': False,
            'roc_curve': False,
            'precision_recall': False,
            'confidence': False,
            'cumulative_gain': False
        }
        
        # XAI Analysis (if model provided)
        if model and test_loader and xai_samples > 0:
            self.log(f"Generating XAI comparisons for {xai_samples} samples...")
            
            try:
                dataset = test_loader.dataset
                indices = np.random.choice(len(dataset), min(xai_samples, len(dataset)), replace=False)
                
                full_ds = dataset.dataset if hasattr(dataset, 'dataset') else dataset
                
                xai_success = 0
                for i, idx in enumerate(indices):
                    if hasattr(full_ds, 'samples'):
                        img_path = full_ds.samples[idx][0]
                        result = self.generate_xai_comparison_plot(model, img_path, sample_id=f"sample_{i}")
                        if result is not None:
                            xai_success += 1
                    else:
                        self.log("Dataset does not support path retrieval for XAI")
                        break
                
                results['xai'] = xai_success
                self.log(f"XAI: {xai_success}/{xai_samples} samples completed successfully")
                
            except Exception as e:
                self.log(f"XAI analysis failed: {e}")
                if self.logger:
                    self.logger.error(f"XAI batch processing error: {e}")
        
        # Clean up model from memory
        if model and test_loader:
            try:
                import gc
                del model, test_loader
                torch.cuda.empty_cache()
                gc.collect()
            except Exception as e:
                self.log(f"Model cleanup warning: {e}")
        
        # Statistical plots - each with independent error handling
        if history:
            try:
                self.plot_training_history(history=history)
                results['training_history'] = True
            except Exception as e:
                self.log(f"Training history failed: {e}")
                if self.logger:
                    self.logger.error(f"Training history error: {e}")
        
        try:
            self.plot_confusion_matrix(y_true, y_pred, normalize=True)
            results['confusion_matrix'] = True
        except Exception as e:
            self.log(f"Confusion matrix failed: {e}")
            if self.logger:
                self.logger.error(f"Confusion matrix error: {e}")
        
        try:
            self.plot_classwise_metrics(y_true, y_pred)
            results['classwise_metrics'] = True
        except Exception as e:
            self.log(f"Classwise metrics failed: {e}")
            if self.logger:
                self.logger.error(f"Classwise metrics error: {e}")
        
        try:
            self.plot_roc_curve(y_true, y_prob)
            results['roc_curve'] = True
        except Exception as e:
            self.log(f"ROC curve failed: {e}")
            if self.logger:
                self.logger.error(f"ROC curve error: {e}")
        
        try:
            self.plot_precision_recall_curve(y_true, y_prob)
            results['precision_recall'] = True
        except Exception as e:
            self.log(f"Precision-Recall curve failed: {e}")
            if self.logger:
                self.logger.error(f"Precision-Recall curve error: {e}")
        
        try:
            self.plot_confidence_distribution(y_true, y_prob)
            results['confidence'] = True
        except Exception as e:
            self.log(f"Confidence distribution failed: {e}")
            if self.logger:
                self.logger.error(f"Confidence distribution error: {e}")
        
        try:
            self.plot_cumulative_gain(y_true, y_prob)
            results['cumulative_gain'] = True
        except Exception as e:
            self.log(f"Cumulative gain failed: {e}")
            if self.logger:
                self.logger.error(f"Cumulative gain error: {e}")
        
        # Summary report
        successful_plots = sum([1 for k, v in results.items() if k != 'xai' and v])
        total_plots = len(results) - 1  # Excluding XAI count
        
        self.log(f"✅ Visualization complete: {successful_plots}/{total_plots} plots successful")
        self.log(f"   XAI: {results['xai']}/{xai_samples} samples")
        self.log(f"   All outputs saved to {self.save_dir}")
        
        return results


from sklearn.metrics import (
    accuracy_score, 
    roc_auc_score, 
    cohen_kappa_score, 
    classification_report, 
    confusion_matrix,
    matthews_corrcoef,
    jaccard_score
)

def test_model(model_name, model, loader, classes, experiment_name, history, logger: Logger, visualizer: Visualizer = None):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    logger.info(f"--- Starting Evaluation: {experiment_name} ---")
    
    # Inference loop with proper memory management
    with torch.inference_mode():
        for images, labels in tqdm(loader, desc="Testing"):
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)
            
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=(DEVICE == 'cuda')):
                outputs = model(images)
            
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)
            
            # Transfer to CPU after detaching
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # Convert to numpy arrays
    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)
    
    # Calculate comprehensive metrics
    accuracy = accuracy_score(y_true, y_pred) * 100
    
    try:
        if len(classes) == 2:
            roc_auc = roc_auc_score(y_true, y_prob[:, 1])
        else:
            roc_auc = roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')
    except Exception as e:
        logger.warning(f"ROC AUC calculation failed: {e}")
        roc_auc = 0.0
    
    kappa = cohen_kappa_score(y_true, y_pred)
    corrcoef = matthews_corrcoef(y_true, y_pred)
    jaccard = jaccard_score(y_true, y_pred, average="weighted")
    
    # Generate comprehensive text report
    report_path = os.path.join(REPORTS_DIR, f"{experiment_name}.txt")
    
    with open(report_path, "w") as f:
        f.write(f"===== COMPREHENSIVE ANALYSIS REPORT: {experiment_name} =====\n\n")
        f.write("--- Overall Performance ---\n")
        f.write(f"Overall Accuracy: {accuracy:.2f}%\n")
        f.write(f"Macro ROC AUC: {roc_auc:.4f}\n")
        f.write(f"Cohen's Kappa: {kappa:.4f}\n")
        f.write(f"Matthews Correlation Coefficient (MCC): {corrcoef:.4f}\n")
        f.write(f"Jaccard Score: {jaccard:.4f}\n\n")
        
        f.write("--- Detailed Per-Class Metrics ---\n")
        f.write(classification_report(y_true, y_pred, target_names=classes, digits=4))
        
        f.write("\n--- Per-Class Specificity & Confusion Matrix Stats ---\n")
        cm = confusion_matrix(y_true, y_pred)
        
        for i, class_name in enumerate(classes):
            tp = cm[i, i]
            fp = cm[:, i].sum() - tp
            fn = cm[i, :].sum() - tp
            tn = cm.sum() - (tp + fp + fn)
            
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            f.write(f"{class_name:<20}: Specificity: {specificity:.4f}, TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn}\n")
    
    logger.info(f"✅ Report saved to: {report_path}")

    # Generate all visualization plots
    if visualizer is None:
        visualizer = Visualizer(
            experiment_name=experiment_name, 
            model_name=model_name, 
            class_names=classes, 
            logger=logger
        )
    
    visualizer.generate_all_plots(
        y_true=y_true, 
        y_prob=y_prob, 
        history=history, 
        model=model, 
        test_loader=loader, 
        xai_samples=NUM_SAMPLES_TO_ANALYSE
    )

    metrics_dict = {
        'accuracy': accuracy,
        'roc_auc': roc_auc,
        'kappa': kappa,
        'corrcoef': corrcoef,
        'jaccard': jaccard,
        'y_true': y_true,
        'y_prob': y_prob
    }
    
    return metrics_dict


class Cross_Validator:
    
    def __init__(self, model_names, logger: Logger, use_aug=False):
        self.model_names = model_names
        self.results = []
        self.use_aug = use_aug
        self.logger = logger
        self.master_file = os.path.join(OUTPUT_DIR, "master_results.csv")
        self.models_dir = os.path.join(OUTPUT_DIR, "best_models")

        os.makedirs(self.models_dir, exist_ok=True)
        self.logger.debug(f"Models for cross-validation: {self.model_names}")
        self.logger.debug(f"GPU Augmentation: {self.use_aug}")
        self.logger.debug(f"Master results file: {self.master_file}")

    def run(self):
        
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

            # Stratified train-test split
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
                pin_memory=PIN_MEMORY,
                persistent_workers=PERSISTENT_WORKERS if NUM_WORKERS > 0 else False
            )
            self.logger.info(f"Data split: {len(train_val_indices)} train/val, {len(test_indices)} test")

            train_val_targets = targets[train_val_indices]

            self.logger.info(f"\n=== Cross-validating: {model_name} ===")
            fold_metrics = []
            
            for fold, (fold_train_idx_rel, fold_val_idx_rel) in enumerate(skf.split(train_val_indices, train_val_targets)):
                self.logger.info(f"  [Fold {fold+1}/{NFOLDS}]")
                
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

                # Model initialization
                try:
                    model = get_model(model_name, num_classes=len(classes), pretrained=PRETRAINED)
                    model = model.to(DEVICE)
                except Exception as e:
                    self.logger.error(f"Failed to load {model_name}: {e}")
                    break

                # Optimizer and scheduler setup
                optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
                criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
                scaler = torch.amp.GradScaler(enabled=USE_AMP)
                
                # OneCycleLR requires per-batch stepping
                scheduler = optim.lr_scheduler.OneCycleLR(
                    optimizer, 
                    max_lr=1e-3, 
                    epochs=EPOCHS, 
                    steps_per_epoch=len(train_loader), 
                    pct_start=0.3,
                    div_factor=25.0,
                    final_div_factor=100.0
                )

                best_acc = 0.0
                training_history = []
                best_stats = {}
                
                for epoch in range(EPOCHS):
                    # Training phase
                    t_loss, t_acc = train_one_epoch(
                        model, train_loader, criterion, optimizer, scaler, gpu_augmenter, scheduler
                    )
                    
                    # Validation phase
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

                    # Save best model
                    if v_acc > best_acc:
                        self.logger.debug(f"New best validation accuracy: {v_acc:.2f}%")
                        best_acc = v_acc
                        best_stats = {
                            'val_acc': v_acc, 'val_loss': v_loss,
                            'val_prec': v_prec, 'val_rec': v_rec, 'val_f1': v_f1,
                            'train_acc': t_acc, 'train_loss': t_loss
                        }
                        torch.save(model.state_dict(), checkpoint_path)
                    
                    # Periodic GPU cache clearing
                    if (epoch + 1) % EMPTY_CACHE_FREQUENCY == 0:
                        torch.cuda.empty_cache()
                
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

                # Critical: cleanup GPU memory between folds
                del model, optimizer, scheduler, scaler, train_loader, val_loader
                if gpu_augmenter is not None:
                    del gpu_augmenter
                torch.cuda.empty_cache()
                gc.collect()
                self.logger.info(f"Fold {fold + 1} cleanup completed")

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
            self.logger.info(f"Final test accuracy: {final_accuracy:.2f}%")
            
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
        
        # Archive results
        zip_and_empty(OUTPUT_DIR, "result.zip")
        self.logger.info("\n>>> Batch Complete.")

if __name__ == "__main__":
    models = __models_list__
    use_aug = __augmentation__
    
    logger = Logger("batch_" + str(hash(str(models)))[:8])
    logger.info(f"Starting validation for {models}")
    logger.info(f"Augmentation: {use_aug}")
    
    try:
        validator = Cross_Validator(models, logger, use_aug=use_aug)
        validator.run()
        logger.info("✓ Batch complete")
    except Exception as e:
        logger.error(f"Batch failed: {e}")
        raise