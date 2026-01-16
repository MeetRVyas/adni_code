"""
Core Research Techniques

All techniques from published papers:
- SAM (Sharpness-Aware Minimization) - ICLR 2021
- Center Loss - ECCV 2016
- Triplet Loss - CVPR 2015
- Manifold Mixup - ICML 2019
- Cosine Classifier - CVPR 2018
- SE Blocks - CVPR 2018
- Distance-Aware Label Smoothing - Custom for ordinal problems
- Prototypical Networks - NeurIPS 2017
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional


# ============================================================================
# SAM OPTIMIZER (Solution 9)
# ============================================================================

class SAM(torch.optim.Optimizer):
    """
    Sharpness-Aware Minimization optimizer.
    
    Research: "Sharpness-Aware Minimization for Efficiently Improving Generalization" (ICLR 2021)
    Expected: +3-5% accuracy, better generalization
    """
    
    def __init__(self, params, base_optimizer, rho=0.05, **kwargs):
        defaults = dict(rho=rho, **kwargs)
        super(SAM, self).__init__(params, defaults)
        
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
    
    @torch.no_grad()
    def first_step(self, zero_grad=False):
        """Ascent step (find adversarial weights)."""
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
        """Descent step (update with sharpness-aware gradient)."""
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.data = self.state[p]["old_p"]
        
        self.base_optimizer.step()
        
        if zero_grad:
            self.zero_grad()
    
    def _grad_norm(self):
        """Compute gradient norm across all parameters."""
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
    """
    Center Loss for discriminative feature learning.
    
    Research: "A Discriminative Feature Learning Approach for Deep Face Recognition" (ECCV 2016)
    Expected: +4-6% accuracy, tight class clusters
    """
    
    def __init__(self, num_classes: int, embedding_dim: int, lambda_c: float = 1.0):
        super().__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.lambda_c = lambda_c
        
        # Learnable class centers
        self.centers = nn.Parameter(torch.randn(num_classes, embedding_dim))
    
    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeddings: [batch_size, embedding_dim]
            labels: [batch_size]
        
        Returns:
            loss: Center loss
        """
        centers_batch = self.centers[labels]
        loss = F.mse_loss(embeddings, centers_batch)
        return self.lambda_c * loss


# ============================================================================
# TRIPLET LOSS (Solution 3)
# ============================================================================

class TripletLoss(nn.Module):
    """
    Triplet Loss for metric learning.
    
    Research: "FaceNet: A Unified Embedding for Face Recognition" (CVPR 2015)
    Expected: +5-7% accuracy
    """
    
    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin
    
    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, 
                negative: torch.Tensor) -> torch.Tensor:
        """
        Args:
            anchor: [N, embedding_dim]
            positive: [N, embedding_dim] (same class as anchor)
            negative: [N, embedding_dim] (different class)
        """
        distance_positive = F.pairwise_distance(anchor, positive, p=2)
        distance_negative = F.pairwise_distance(anchor, negative, p=2)
        
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean()


def create_triplet_batch(embeddings: torch.Tensor, labels: torch.Tensor) -> Tuple[
    Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]
]:
    """
    Create triplets from a batch for Triplet Loss.
    
    Returns:
        (anchors, positives, negatives) or (None, None, None) if can't create triplets
    """
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
    """
    Manifold Mixup: Mix in embedding space, not image space.
    
    Research: "Manifold Mixup: Learning Better Representations by Interpolating Hidden States" (ICML 2019)
    Expected: +3-5% accuracy without destroying medical semantics
    """
    
    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha
    
    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, float
    ]:
        """
        Apply manifold mixup to embeddings.
        
        Returns:
            mixed_embeddings, labels_a, labels_b, lam
        """
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
    """
    Compute mixed loss for manifold mixup.
    
    Loss = λ * Loss(output, label_a) + (1-λ) * Loss(output, label_b)
    """
    loss = lam * criterion(outputs, labels_a) + (1 - lam) * criterion(outputs, labels_b)
    return loss


# ============================================================================
# COSINE CLASSIFIER (Solution 6)
# ============================================================================

class CosineClassifier(nn.Module):
    """
    Cosine similarity classifier instead of linear.
    
    Research: "CosFace: Large Margin Cosine Loss for Deep Face Recognition" (CVPR 2018)
    Expected: +3-4% accuracy, robust to intensity variations
    """
    
    def __init__(self, in_features: int, num_classes: int, scale: float = 30.0):
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.scale = scale
        
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: [batch_size, in_features]
        
        Returns:
            logits: [batch_size, num_classes]
        """
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
    """
    Squeeze-and-Excitation block for channel attention.
    
    Research: "Squeeze-and-Excitation Networks" (CVPR 2018, won ImageNet)
    Expected: +3-4% accuracy
    """
    
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
        """
        Args:
            x: [batch, channels, height, width]
        
        Returns:
            x_scaled: [batch, channels, height, width] with channel attention
        """
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
    """
    Label smoothing that respects class ordering for ordinal problems.
    
    Perfect for medical severity: NonDemented < VeryMild < Mild < Moderate
    Expected: +2-3% accuracy on ordinal problems
    """
    
    def __init__(self, num_classes: int, smoothing: float = 0.1):
        super().__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: [batch_size, num_classes]
            targets: [batch_size]
        
        Returns:
            loss: Cross-entropy with distance-aware smoothing
        """
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
    """
    Prototypical Networks for few-shot learning.
    
    Research: "Prototypical Networks for Few-shot Learning" (NeurIPS 2017)
    Expected: +6-9% with limited data
    """
    
    def __init__(self, encoder: nn.Module, embedding_dim: int = 512):
        super().__init__()
        self.encoder = encoder
        self.embedding_dim = embedding_dim
        self.prototypes = None
    
    def compute_prototypes(self, support_embeddings: torch.Tensor, 
                          support_labels: torch.Tensor, num_classes: int) -> torch.Tensor:
        """
        Compute class prototypes from support set.
        
        Returns:
            prototypes: [num_classes, embedding_dim]
        """
        prototypes = torch.zeros(num_classes, self.embedding_dim).to(support_embeddings.device)
        
        for k in range(num_classes):
            class_mask = (support_labels == k)
            class_embeddings = support_embeddings[class_mask]
            
            if class_embeddings.size(0) > 0:
                prototypes[k] = class_embeddings.mean(dim=0)
        
        return prototypes
    
    def euclidean_distance(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Euclidean distance matrix.
        
        Args:
            x: [N, D] query embeddings
            y: [M, D] prototype embeddings
        
        Returns:
            distances: [N, M]
        """
        n = x.size(0)
        m = y.size(0)
        d = x.size(1)
        
        x = x.unsqueeze(1).expand(n, m, d)
        y = y.unsqueeze(0).expand(n, m, d)
        
        return torch.pow(x - y, 2).sum(2)
    
    def forward(self, query_images: torch.Tensor, support_images: Optional[torch.Tensor] = None,
                support_labels: Optional[torch.Tensor] = None, num_classes: int = 4) -> torch.Tensor:
        """
        Forward pass with prototypical classification.
        
        Returns:
            log_probs: Log probabilities for query images
        """
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
        """Update prototypes using entire training set."""
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
