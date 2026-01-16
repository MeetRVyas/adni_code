"""
Progressive Classifier - Your sophisticated ProgressiveFineTuner converted to BaseClassifier

Features:
- Architecture-aware layer grouping (ResNet, ViT, Swin, EfficientNet, MobileNet)
- 3-phase discriminative fine-tuning
- Focal Loss for hard examples
- SAM optimizer support in Phase 3
- Research-grade training strategy
"""

import torch
import torch.nn as nn
import torch.optim as optim
import timm
import numpy as np

from .base_classifier import BaseClassifier
from .techniques import SAM


class FocalLoss(nn.Module):
    """
    Focal Loss for handling hard examples.
    From: "Focal Loss for Dense Object Detection" (Lin et al., 2017)
    """
    
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
    """
    Architecture-aware layer grouping for discriminative learning rates.
    
    Groups layers from early (general features) to late (task-specific).
    """
    
    @staticmethod
    def get_resnet_groups(model):
        """ResNet family layer groups."""
        return [
            list(model.conv1.parameters()) + list(model.bn1.parameters()) + list(model.layer1.parameters()),
            list(model.layer2.parameters()),
            list(model.layer3.parameters()),
            list(model.layer4.parameters()),
            list(model.fc.parameters())
        ]
    
    @staticmethod
    def get_vit_groups(model):
        """Vision Transformer layer groups."""
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
        """Swin Transformer layer groups."""
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
        """EfficientNet layer groups."""
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
        """MobileNet layer groups."""
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
        """Automatically detect architecture and return layer groups."""
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
    """
    Progressive Fine-Tuning Classifier.
    
    Your sophisticated ProgressiveFineTuner converted to match BaseClassifier interface.
    
    Features:
    - Phase 1 (5 epochs): Classifier only
    - Phase 2 (10 epochs): Top 50% layers
    - Phase 3 (15 epochs): All layers with discriminative LRs + optional SAM
    - Architecture-aware layer grouping
    - Focal Loss
    """
    
    def build_model(self):
        """Load pretrained model."""
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
        """Standard forward pass."""
        return self.model(images)
    
    def compute_loss(self, outputs, labels):
        """Focal Loss for hard examples."""
        if not hasattr(self, 'focal_loss'):
            self.focal_loss = FocalLoss(alpha=1.0, gamma=2.0).to(self.device)
        return self.focal_loss(outputs, labels)
    
    def _get_discriminative_params(self, base_lr):
        """
        Create parameter groups with discriminative learning rates.
        
        LR multipliers:
        - Group 0 (early): base_lr / 100
        - Group 1 (mid-early): base_lr / 10
        - Group 2 (mid-late): base_lr / 3
        - Group 3 (late): base_lr
        - Group 4 (classifier): base_lr * 10
        """
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
        """
        3-phase progressive fine-tuning.
        
        Total epochs distributed: 5 (phase1) + 10 (phase2) + remaining (phase3)
        """
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
        """Train a single phase."""
        
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
