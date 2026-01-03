import os
import torch
from torchvision import transforms, datasets
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import logging
import shutil
import timm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from itertools import cycle
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, average_precision_score,
    confusion_matrix, classification_report, roc_auc_score,
    accuracy_score, precision_recall_fscore_support
)
from torch.utils.data import DataLoader, Subset
import gc
from sklearn.model_selection import train_test_split, StratifiedKFold


# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Directory structure
OUTPUT_DIR = "/kaggle/working/output"
RESULTS_DIR = os.path.join(OUTPUT_DIR, "results")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
REPORTS_DIR = os.path.join(RESULTS_DIR, "reports")
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")
DATA_DIR = "/kaggle/input/augmented-alzheimer-mri-dataset/OriginalDataset"

# Training hyperparameters
EPOCHS = 25
NFOLDS = 3
BATCH_SIZE = 32
NUM_WORKERS = 4  # Reduced from 8 to prevent CPU bottleneck
PRETRAINED = True
NUM_SAMPLES_TO_ANALYSE = 5
TEST_SPLIT = 0.2
PATIENCE = 10
MIN_DELTA = 0.3
LR = 1e-4

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


def get_gpu_augmentations(img_size):
    return torch.nn.Sequential(
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    ).to(DEVICE)


# ADVANCED AUGMENTATION TECHNIQUES
class Mixup:
    def __init__(self, alpha=0.2):
        self.alpha = alpha
    
    def __call__(self, x, y):
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1
        
        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(x.device)
        
        mixed_x = lam * x + (1 - lam) * x[index]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam
    
    def loss_function(self, criterion, outputs, y_a, y_b, lam):
        return lam * criterion(outputs, y_a) + (1 - lam) * criterion(outputs, y_b)


class CutMix:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
    
    def __call__(self, x, y):
        lam = np.random.beta(self.alpha, self.alpha)
        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(x.device)
        
        # Get random box
        W, H = x.size(2), x.size(3)
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)
        
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        
        # Apply cutmix
        x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
        
        # Adjust lambda
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
        
        y_a, y_b = y, y[index]
        return x, y_a, y_b, lam
    
    def loss_function(self, criterion, outputs, y_a, y_b, lam):
        return lam * criterion(outputs, y_a) + (1 - lam) * criterion(outputs, y_b)


# TEST-TIME AUGMENTATION (TTA)
class TTAWrapper:
    
    def __init__(self, model, num_augmentations=5):
        self.model = model
        self.num_augmentations = num_augmentations
        
        # Define TTA transformations
        self.tta_transforms = [
            transforms.Compose([]),  # Original (no transform)
            transforms.Compose([transforms.RandomHorizontalFlip(p=1.0)]),
            transforms.Compose([transforms.RandomRotation(degrees=10)]),
            transforms.Compose([
                transforms.ColorJitter(brightness=0.1, contrast=0.1)
            ]),
        ]
    
    def __call__(self, images):
        self.model.eval()
        batch_size = images.size(0)
        all_outputs = []
        
        with torch.no_grad():
            for transform in self.tta_transforms[:self.num_augmentations]:
                # Apply transform to each image in batch
                augmented_batch = torch.stack([
                    transform(img) for img in images
                ])
                
                # Get predictions
                outputs = self.model(augmented_batch)
                probs = F.softmax(outputs, dim=1)
                all_outputs.append(probs)
        
        # Average all predictions
        avg_probs = torch.stack(all_outputs).mean(dim=0)
        return avg_probs


class Training_Strategy:
    
    @staticmethod
    def get_strategy(name) :
        strategies = {
            'simple_cosine': SimpleCosineLRStrategy,
            'discriminative_onecycle': DiscriminativeOneCycleStrategy,
            'full_finetune_cosine': FullFinetuneCosineStrategy,
            'aggressive_warmup': AggressiveWarmupStrategy,
            'conservative_sgd': ConservativeSGDStrategy,
        }
        
        strategy_class = strategies.get(name, FullFinetuneCosineStrategy)
        return strategy_class
    
    def __init__(self, model, img_size, lr, epochs, steps_per_epoch, num_classes, use_aug, class_weights_tensor=None):
        self.model = model
        self.img_size = img_size
        self.lr = lr
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch
        self.num_classes = num_classes
        self.use_aug = use_aug
        self.class_weights_tensor = class_weights_tensor
        
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        self.use_mixup = False
        self.use_cutmix = False
        self.use_simple_augments = use_aug
        self.gradient_accumulation_steps = 1
        
    def setup(self):
        raise NotImplementedError


class SimpleCosineLRStrategy(Training_Strategy):
    def setup(self):
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=0.01)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.epochs)
        self.criterion = nn.CrossEntropyLoss(weight=self.class_weights_tensor) if self.class_weights_tensor is not None else nn.CrossEntropyLoss()
        return "AdamW + CosineAnnealing + Full Fine-tuning"


class DiscriminativeOneCycleStrategy(Training_Strategy):
    def setup(self):
        # Separate backbone from classifier
        classifier_params, backbone_params = self._separate_parameters()
        
        if backbone_params:
            self.optimizer = optim.AdamW([
                {'params': classifier_params, 'lr': self.lr},
                {'params': backbone_params, 'lr': self.lr / 50}  # 50x lower
            ], weight_decay=0.01)
        else:
            self.optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=0.01)
        
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer, max_lr=self.lr, epochs=self.epochs, 
            steps_per_epoch=self.steps_per_epoch, pct_start=0.2,
            div_factor=10.0, final_div_factor=100.0
        )
        
        self.criterion = nn.CrossEntropyLoss(
            weight=self.class_weights_tensor,
            label_smoothing=0.0
        )
        
        return "AdamW + OneCycleLR + Discriminative LR"
    
    def _separate_parameters(self):
        try:
            # Try different methods to get classifier
            classifier = None
            if hasattr(self.model, 'get_classifier'):
                classifier = self.model.get_classifier()
                # Handle tuple return (EfficientFormer, ConvNeXt)
                if isinstance(classifier, tuple):
                    classifier = classifier[0] if len(classifier) > 0 else None
            elif hasattr(self.model, 'fc'):
                classifier = self.model.fc
            elif hasattr(self.model, 'classifier'):
                classifier = self.model.classifier
            elif hasattr(self.model, 'head'):
                classifier = self.model.head
            
            if classifier is None or not isinstance(classifier, nn.Module):
                return list(self.model.parameters()), []
            
            classifier_params = list(classifier.parameters())
            classifier_param_ids = {id(p) for p in classifier_params}
            backbone_params = [p for p in self.model.parameters() if id(p) not in classifier_param_ids]
            
            return classifier_params, backbone_params
        except:
            return list(self.model.parameters()), []


class FullFinetuneCosineStrategy(Training_Strategy):
    def setup(self):
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=0.01)
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=self.epochs // 3, T_mult=1, eta_min=self.lr / 100
        )
        self.criterion = nn.CrossEntropyLoss(
            weight=self.class_weights_tensor,
            label_smoothing=0.1
        )
        if self.use_aug :
            self.use_mixup = True  # Enable mixup
            self.use_simple_augments = False
        return "AdamW + CosineWarmRestarts + Mixup"


class AggressiveWarmupStrategy(Training_Strategy):
    def setup(self):
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.lr * 2, weight_decay=0.01)
        
        # Linear warmup + CosineAnnealing
        warmup_epochs = max(1, self.epochs // 10)
        warmup_scheduler = optim.lr_scheduler.LinearLR(
            self.optimizer, start_factor=0.1, total_iters=warmup_epochs * self.steps_per_epoch
        )
        cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=(self.epochs - warmup_epochs) * self.steps_per_epoch
        )
        self.scheduler = optim.lr_scheduler.SequentialLR(
            self.optimizer, 
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_epochs * self.steps_per_epoch]
        )
        
        self.criterion = nn.CrossEntropyLoss(
            weight=self.class_weights_tensor,
            label_smoothing=0.1
        )
        if self.use_aug :
            self.use_cutmix = True
            self.use_simple_augments = False
        self.gradient_accumulation_steps = 2  # Effective batch size 2x
        return "AdamW + Warmup + CutMix + GradAccum"


class ConservativeSGDStrategy(Training_Strategy):
    def setup(self):
        self.optimizer = optim.SGD(
            self.model.parameters(), lr=self.lr, momentum=0.9, 
            weight_decay=0.01, nesterov=True
        )
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=self.epochs // 3, gamma=0.1)
        self.criterion = nn.CrossEntropyLoss(weight=self.class_weights_tensor)
        return "SGD + StepLR + Nesterov"


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


def train_one_epoch(model, loader, criterion, optimizer, scaler, strategy, scheduler=None):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    mixup = Mixup(alpha=0.2) if strategy.use_mixup else None
    cutmix = CutMix(alpha=1.0) if strategy.use_cutmix else None
    simple_augments = get_gpu_augmentations(strategy.img_size) if strategy.use_simple_augments else None
    grad_accum_steps = strategy.gradient_accumulation_steps
    
    for batch_idx, (images, labels) in enumerate(loader):
        images, labels = images.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)
        
        # Apply augmentation
        if mixup and np.random.rand() < 0.5:
            images, labels_a, labels_b, lam = mixup(images, labels)
            use_mixed = True
        elif cutmix and np.random.rand() < 0.5:
            images, labels_a, labels_b, lam = cutmix(images, labels)
            use_mixed = True
        elif simple_augments and np.random.rand() < 0.5:
            images = simple_augments(images)
            use_mixed = False
        else:
            use_mixed = False
        
        # optimizer.zero_grad(set_to_none=True)
        
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=(DEVICE == 'cuda')):
            outputs = model(images)
            
            if use_mixed:
                loss = (mixup or cutmix).loss_function(criterion, outputs, labels_a, labels_b, lam)
            else:
                loss = criterion(outputs, labels)
            
            loss = loss / grad_accum_steps
        
        scaler.scale(loss).backward()
        
        # Gradient accumulation
        if (batch_idx + 1) % grad_accum_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            
            if scheduler and isinstance(scheduler, (
                optim.lr_scheduler.OneCycleLR,
                optim.lr_scheduler.SequentialLR
            )):
                scheduler.step()
        
        running_loss += loss.item() * grad_accum_steps * images.size(0)
        
        # Collect stats (use original labels for mixup/cutmix)
        if not use_mixed:
            _, predicted = outputs.detach().max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    total_loss = running_loss / len(loader.dataset)
    
    if all_preds:  # Only calculate accuracy if we have non-mixed batches
        acc = accuracy_score(all_labels, all_preds) * 100.0
    else:
        acc = 0.0  # All batches were mixed
    
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

from sklearn.metrics import (
    accuracy_score, 
    roc_auc_score, 
    cohen_kappa_score, 
    classification_report, 
    confusion_matrix,
    matthews_corrcoef,
    jaccard_score
)


def test_model(model_name, model, loader, classes, experiment_name, history, logger: Logger, 
               visualizer: Visualizer = None, use_tta=True):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    logger.info(f"--- Starting Evaluation: {experiment_name} ---")
    if use_tta:
        logger.info("Using Test-Time Augmentation (TTA) for improved accuracy")
        tta_model = TTAWrapper(model, num_augmentations=5)
    
    # Inference Loop
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            
            if use_tta:
                # Use TTA
                probs = tta_model(images)
                preds = torch.argmax(probs, dim=1)
            else:
                # Standard inference
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
    
    # ============================================================================
    # NaN Detection and Handling
    # ============================================================================
    if np.isnan(y_prob).any():
        logger.warning("⚠️  NaN detected in probability outputs!")
        logger.warning("   This usually indicates model collapse to single-class prediction.")
        logger.warning("   Replacing NaN with uniform distribution for metric calculation.")
        n_classes = len(classes)
        y_prob = np.nan_to_num(y_prob, nan=1.0/n_classes)
    
    # Verify probabilities sum to 1.0 (within tolerance)
    prob_sums = y_prob.sum(axis=1)
    if not np.allclose(prob_sums, 1.0, atol=1e-3):
        logger.warning("⚠️  Probability distributions don't sum to 1.0!")
        logger.warning(f"   Sum range: [{prob_sums.min():.4f}, {prob_sums.max():.4f}]")
        # Normalize to ensure sum=1
        y_prob = y_prob / prob_sums[:, np.newaxis]
        logger.warning("   Probabilities normalized to sum=1.0")
    
    # ============================================================================
    # Calculate Metrics
    # ============================================================================
    accuracy = accuracy_score(y_true, y_pred) * 100
    
    # ROC AUC with improved error handling
    try:
        if len(classes) == 2:
            roc_auc = roc_auc_score(y_true, y_prob[:, 1])
        else:
            roc_auc = roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')
    except ValueError as e:
        logger.warning(f"ROC AUC calculation failed: {e}")
        roc_auc = 0.0
    except Exception as e:
        logger.warning(f"Unexpected error in ROC AUC calculation: {e}")
        roc_auc = 0.0
        
    kappa = cohen_kappa_score(y_true, y_pred)
    corrcoef = matthews_corrcoef(y_true, y_pred)
    jaccard = jaccard_score(y_true, y_pred, average="weighted")
    
    # ============================================================================
    # Generate Report
    # ============================================================================
    report_path = os.path.join(REPORTS_DIR, f"{experiment_name}.txt")
    
    with open(report_path, "w") as f:
        f.write(f"===== COMPREHENSIVE ANALYSIS REPORT: {experiment_name} =====\n\n")
        f.write(f"Test-Time Augmentation (TTA): {'Enabled' if use_tta else 'Disabled'}\n\n")
        f.write("--- Overall Performance ---\n")
        f.write(f"Overall Accuracy: {accuracy:.2f}%\n")
        f.write(f"Macro ROC AUC:    {roc_auc:.4f}\n")
        f.write(f"Cohen's Kappa:    {kappa:.4f}\n")
        f.write(f"Matthews Correlation Coefficient (MCC):    {corrcoef:.4f}\n")
        f.write(f"Jaccard Score:    {jaccard:.4f}\n\n")
        
        f.write("--- Detailed Per-Class Metrics ---\n")
        f.write(classification_report(y_true, y_pred, target_names=classes, digits=4))
        
        f.write("\n--- Per-Class Specificity & Confusion Matrix Stats ---\n")
        cm = confusion_matrix(y_true, y_pred)
        
        # Calculate Specificity per class
        for i, class_name in enumerate(classes):
            tp = cm[i, i]
            fp = cm[:, i].sum() - tp
            fn = cm[i, :].sum() - tp
            tn = cm.sum() - (tp + fp + fn)
            
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            f.write(f"{class_name:<20}: Specificity: {specificity:.4f}, "
                   f"TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn}\n")
            
    logger.info(f"✅ Report saved to: {report_path}")

    # ============================================================================
    # Generate Visualizations
    # ============================================================================
    visualizer = visualizer or Visualizer(
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
        'y_prob': y_prob,
        'y_pred': y_pred,  # Add predictions for ensemble
    }
    
    return metrics_dict


class Cross_Validator:
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
            # targets = np.array(full_dataset.targets)
            # classes = full_dataset.classes
            # self.logger.debug(f"Classes found: {classes}")

            # # Calculate Class Weights
            # class_counts = np.bincount(targets)
            # total_samples = len(targets)
            # class_weights = total_samples / (len(classes) * class_counts)
            # class_weights_tensor = torch.FloatTensor(class_weights).to(DEVICE)
            
            # self.logger.info("="*80)
            # self.logger.info("CLASS DISTRIBUTION & WEIGHTS:")
            # for i, cls in enumerate(classes):
            #     self.logger.info(f"  {cls:<20}: {class_counts[i]:>4} samples ({100*class_counts[i]/total_samples:>5.1f}%) | Weight: {class_weights[i]:.3f}")
            # self.logger.info("="*80)

            # # Stratified train-test split
            # train_val_indices, test_indices = train_test_split(
            #     np.arange(len(targets)),
            #     test_size=TEST_SPLIT,
            #     stratify=targets,
            #     random_state=42
            # )
            
            # self.logger.info(f"Data split: {len(train_val_indices)} train/val, {len(test_indices)} test")

            # train_val_targets = targets[train_val_indices]

            self.logger.info(f"\n=== Cross-validating: {model_name} ===")
            fold_metrics = []
            
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

                self.logger.info("Initializing Model, Optimizer, Scaler and Scheduler")

                # Model Setup
                try:
                    model = get_model(model_name, num_classes=len(classes), pretrained=PRETRAINED)
                    model = model.to(DEVICE)
                except Exception as e:
                    self.logger.error(f"Failed to load {model_name}: {e}")
                    break

                # Get training strategy
                strategy = Training_Strategy.get_strategy(self.strategy_name)(
                    model,
                    img_size,
                    lr=LR,
                    epochs=EPOCHS,
                    steps_per_epoch=len(train_loader),
                    num_classes=len(classes),
                    use_aug = self.use_aug,
                    class_weights_tensor=class_weights_tensor if self.strategy_name != 'simple_cosine' else None
                )
                
                strategy_desc = strategy.setup()
                
                scaler = torch.amp.GradScaler(enabled=USE_AMP)

                self.logger.info(f"Using Strategy: {strategy_desc}")
                self.logger.debug(f"Using criterion: CrossEntropyLoss with class weights")
                self.logger.debug(f"Using scaler: {scaler}")

                best_acc = 0.0
                training_history = []
                best_stats = {}
                patience_counter = 0
                step = EPOCHS / 20
                curr = step

                self.logger.info("Starting training and validation epochs")
                print("\t\tProcessing [", end = "")
                
                for epoch in range(EPOCHS):
                    # Train
                    t_loss, t_acc = train_one_epoch(
                        model, train_loader, strategy.criterion, strategy.optimizer, 
                        scaler, strategy, strategy.scheduler
                    )
                    
                    # Validate
                    v_loss, v_acc, v_prec, v_rec, v_f1 = validate_one_epoch(
                        model, val_loader, strategy.criterion
                    )
                    
                    # Step epoch-based schedulers
                    if strategy.scheduler and not isinstance(strategy.scheduler, (
                        optim.lr_scheduler.OneCycleLR,
                        optim.lr_scheduler.SequentialLR
                    )):
                        strategy.scheduler.step()
                    
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
                    if v_acc > best_acc + MIN_DELTA:
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
                        if patience_counter >= PATIENCE:
                            self.logger.info(f"[Epoch {epoch+1}] Early stopping triggered (no improvement for {PATIENCE} epochs)")
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
                    f"F1: {best_stats.get('val_f1', 0):.4f} | "
                    f"Precision: {best_stats.get('val_prec', 0):.4f} | "
                    f"Recall: {best_stats.get('val_rec', 0):.4f}"
                )

                fold_metrics.append({
                    'fold': fold + 1,
                    **best_stats
                })

                # Cleanup
                del model, strategy, scaler, train_loader, val_loader
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
            del full_dataset, eval_model, test_loader
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

def run_batch():
    models = __models_list__
    use_aug = __augmentation__
    strategy = __strategy__
    enable_ensemble = __ensemble__

    logger = Logger("batch_" + str(hash(str(models)))[:8])
    logger.info(f"Starting validation for {models}")
    
    try:
        validator = Cross_Validator(models, logger, use_aug=use_aug, strategy = strategy, enable_ensemble = enable_ensemble)
        validator.run()
        logger.info("Validation complete")
    except Exception as e:
        logger.error(f"Batch failed: {e}")
        raise

if __name__ == "__main__":
    run_batch()