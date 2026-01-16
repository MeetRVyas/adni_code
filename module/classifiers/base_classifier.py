"""
Base Classifier with Metric-Agnostic Optimization

All classifiers inherit from this base class to ensure:
1. Consistent API
2. Configurable optimization metric (recall, accuracy, f1, precision)
3. Unified evaluation
4. SAM-aware training
"""

import torch
import torch.nn as nn
import torch.optim as optim
from abc import ABC, abstractmethod
from typing import Dict, Optional, List
import numpy as np
from sklearn.metrics import (
    accuracy_score, recall_score, precision_score, f1_score,
    classification_report, confusion_matrix
)
from tqdm import tqdm


class BaseClassifier(ABC):
    """
    Abstract base class for all classifiers.
    
    All classifiers must implement:
    - build_model()
    - forward()
    - compute_loss()
    """
    
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
        """Build the model architecture. Must set self.model."""
        pass
    
    @abstractmethod
    def forward(self, images: torch.Tensor):
        """Forward pass through model."""
        pass
    
    @abstractmethod
    def compute_loss(self, outputs, labels) -> torch.Tensor:
        """Compute loss from model outputs and labels."""
        pass
    
    def get_predictions(self, outputs) -> torch.Tensor:
        """Convert model outputs to class predictions."""
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        return torch.argmax(outputs, dim=1)
    
    def _get_metric_value(self, labels: List, preds: List, metric: str) -> float:
        """Calculate specific metric value."""
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
        """Train for one epoch."""
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
        """Validate for one epoch."""
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
        """
        Complete training loop with metric-agnostic optimization.
        
        Args:
            train_loader: Training data
            val_loader: Validation data
            epochs: Maximum epochs
            lr: Learning rate
            use_sam: Use SAM optimizer
            primary_metric: Metric to optimize ('recall', 'accuracy', 'f1', 'precision')
            patience: Early stopping patience
            min_delta: Minimum improvement threshold
        
        Returns:
            history: Training history
        """
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
        """Comprehensive evaluation on test set."""
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
        """Save model weights."""
        torch.save(self.model.state_dict(), path)
    
    def load(self, path: str):
        """Load model weights."""
        self.model.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))
