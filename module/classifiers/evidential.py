"""
Evidential Deep Learning

Universal implementation that works with ANY architecture.

Research: "Evidential Deep Learning to Quantify Classification Uncertainty" (NeurIPS 2018)
Expected: +5-8% accuracy + uncertainty quantification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from typing import Tuple


# ============================================================================
# EVIDENTIAL LAYER
# ============================================================================

class EvidentialLayer(nn.Module):
    """
    Evidential output layer that outputs evidence instead of logits.
    
    Evidence → Dirichlet distribution → Uncertainty quantification
    """
    
    def __init__(self, in_features: int, num_classes: int):
        super().__init__()
        self.num_classes = num_classes
        self.evidence_layer = nn.Linear(in_features, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns evidence (must be positive).
        
        Args:
            x: [batch_size, in_features]
        
        Returns:
            evidence: [batch_size, num_classes]
        """
        evidence = F.softplus(self.evidence_layer(x))
        return evidence
    
    def get_predictions_and_uncertainty(self, evidence: torch.Tensor, class_weights: torch.Tensor = None) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        """
        Convert evidence to predictions and uncertainty.
        
        Returns:
            probs: [batch_size, num_classes] expected probabilities
            uncertainty: [batch_size, 1] uncertainty score (0=certain, 1=uncertain)
            alpha: [batch_size, num_classes] Dirichlet parameters
        """
        # Dirichlet parameters
        alpha = evidence + 1
        
        # Expected probabilities
        S = alpha.sum(dim=1, keepdim=True)
        K = self.num_classes
        uncertainty = K / S
        
        # Expected probabilities
        if class_weights is not None:
            if class_weights.device != evidence.device:
                class_weights = class_weights.to(evidence.device)
            # Weighted expected probabilities for recall bias
            # p_k = (w_k * alpha_k) / sum(w_j * alpha_j)
            weighted_alpha = alpha * class_weights.view(1, -1)
            S = weighted_alpha.sum(dim=1, keepdim=True)
            probs = weighted_alpha / S
        else:
            probs = alpha / S
        
        return probs, uncertainty, alpha


# ============================================================================
# EVIDENTIAL LOSS
# ============================================================================

class EvidentialLoss(nn.Module):
    """
    Loss function for evidential deep learning.
    
    Combines:
    1. Bayesian risk (classification loss)
    2. KL divergence (regularization to prevent overconfidence)
    """
    
    def __init__(self, num_classes: int, lam: float = 0.5, epsilon: float = 1e-10, class_weights: torch.Tensor = None):
        super().__init__()
        self.num_classes = num_classes
        self.lam = lam
        self.epsilon = epsilon
        self.class_weights = class_weights
    
    def forward(self, evidence: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            evidence: [batch_size, num_classes]
            target: [batch_size] class indices
        
        Returns:
            loss: Scalar loss
        """
        alpha = evidence + 1
        S = alpha.sum(dim=1, keepdim=True)
        
        # One-hot encoding
        target_one_hot = F.one_hot(target, num_classes=self.num_classes).float()
        
        # Bayesian risk
        A = torch.sum((target_one_hot - alpha / S) ** 2, dim=1, keepdim=True)
        B = torch.sum(alpha * (S - alpha) / (S * S * (S + 1)), dim=1, keepdim=True)
        
        # Apply class weights to MSE loss
        loss_mse = A + B
        if self.class_weights is not None:
            if self.class_weights.device != evidence.device:
                self.class_weights = self.class_weights.to(evidence.device)
            
            # Weight per sample based on target class
            weights = self.class_weights[target].view(-1, 1)
            loss_mse = loss_mse * weights
            
        loss_mse = loss_mse.mean()
        
        # KL divergence regularization
        alpha_tilde = target_one_hot + (1 - target_one_hot) * alpha
        S_tilde = alpha_tilde.sum(dim=1, keepdim=True)
        
        kl_div = torch.lgamma(S_tilde) - torch.lgamma(alpha_tilde).sum(dim=1, keepdim=True)
        kl_div += ((alpha_tilde - 1) * (torch.digamma(alpha_tilde) - torch.digamma(S_tilde))).sum(dim=1, keepdim=True)
        kl_div = kl_div.mean()
        
        return loss_mse + self.lam * kl_div


# ============================================================================
# UNIVERSAL EVIDENTIAL MODEL
# ============================================================================

class UniversalEvidentialModel(nn.Module):
    """
    Universal evidential wrapper for ANY architecture.
    
    Usage:
        model = UniversalEvidentialModel('resnet18', num_classes=4)
        evidence = model(images)
        probs, uncertainty, alpha = model.get_predictions_and_uncertainty(evidence)
    """
    
    def __init__(self, model_name: str, num_classes: int = 4, pretrained: bool = True):
        super().__init__()
        
        self.model_name = model_name
        self.num_classes = num_classes

        self.feature_extractor = timm.create_model(self.model_name, pretrained=pretrained, num_classes=0)
        in_features = self.feature_extractor.num_features
        
        # Evidential head
        self.evidential_head = EvidentialLayer(in_features, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns evidence (not probabilities!)."""
        features = self.feature_extractor(x)
        evidence = self.evidential_head(features)
        return evidence
    
    def get_predictions_and_uncertainty(self, evidence: torch.Tensor, class_weights: torch.Tensor = None) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        """Convert evidence to predictions and uncertainty."""
        return self.evidential_head.get_predictions_and_uncertainty(evidence, class_weights)