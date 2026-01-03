import torch
from torchvision import transforms
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
from .config import DEVICE


def get_gpu_augmentations(img_size):
    """GPU-based augmentations applied to tensor batches"""
    return torch.nn.Sequential(
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    ).to(DEVICE)


# ADVANCED AUGMENTATION TECHNIQUES
class Mixup:
    """Mixup augmentation: mixes two samples and their labels."""
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
    """CutMix augmentation: cuts and pastes patches between samples."""
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
    """Test-Time Augmentation wrapper for +2-5% accuracy boost."""
    
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
        """
        Apply TTA and average predictions.
        Args:
            images: Batch of images [B, C, H, W]
        Returns:
            Averaged probabilities [B, num_classes]
        """
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