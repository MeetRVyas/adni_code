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

from module.classifiers.base_classifier import BaseClassifier
from module.classifiers.techniques import SAM


class FocalLoss(nn.Module):
    """
    Focal Loss for handling hard examples.
    From: "Focal Loss for Dense Object Detection" (Lin et al., 2017)
    """
    
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean', weights=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.weights = weights
    
    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        p_t = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - p_t) ** self.gamma * ce_loss
        
        if self.weights is not None:
             if self.weights.device != inputs.device:
                self.weights = self.weights.to(inputs.device)
             
             # Apply class weights
             weight_per_sample = self.weights[targets]
             focal_loss = focal_loss * weight_per_sample
        
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
        groups = [
            list(model.conv1.parameters()) + list(model.bn1.parameters()) + list(model.layer1.parameters()),
            list(model.layer2.parameters()),
            list(model.layer3.parameters()),
            list(model.layer4.parameters()),
        ]
        if hasattr(model, 'fc'):
            groups.append(list(model.fc.parameters()))
        else :
            groups.append([])
        return groups
    
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
        # We initialize 5 groups as per your original logic
        groups = [[] for _ in range(5)]
        
        # We use a set to track parameter IDs to ensure 100% coverage
        param_ids = set()
        
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
                
            # --- Logic matching your original context ---
            
            # Group 4: Head
            # timm swin models name the classifier "head"
            if name.startswith('head.'):
                groups[4].append(param)
                
            # Group 0: Patch Embed (Stem)
            elif name.startswith('patch_embed.') or name.startswith('absolute_pos_embed'):
                groups[0].append(param)
                
            # Layers (Stages 1-4)
            elif name.startswith('layers.'):
                # name format is "layers.X.blocks..."
                # We parse X to determine the group
                try:
                    # Extract the layer index (0, 1, 2, or 3)
                    layer_idx = int(name.split('.')[1])
                    
                    if layer_idx == 0:
                        # Context: Group 0 includes Stage 1
                        groups[0].append(param)
                    elif layer_idx == 1:
                        # Context: Group 1 is Stage 2
                        groups[1].append(param)
                    elif layer_idx == 2:
                        # Context: Group 2 is Stage 3
                        groups[2].append(param)
                    elif layer_idx == 3:
                        # Context: Group 3 is Stage 4
                        groups[3].append(param)
                    else:
                        # Fallback: If model has >4 stages (rare), put in Group 3
                        groups[3].append(param)
                        
                except (IndexError, ValueError):
                    # Fallback: If parsing fails, put in Group 0 (safest default)
                    print(f"Warning: Could not parse layer index for {name}. Assigning to Group 0.")
                    groups[0].append(param)
    
            # Group 3: Final Norm
            # Context: Your code put model.norm in Group 3
            elif name.startswith('norm.'):
                groups[3].append(param)
                
            # Catch-all
            else:
                print(f"Warning: Unknown parameter found: '{name}'. Assigning to Group 0.")
                groups[0].append(param)
            
            # Track ID
            param_ids.add(id(param))
    
        # --- Robustness Check ---
        # Ensure every single trainable parameter was assigned to a group
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        if len(param_ids) != len(trainable_params):
            raise RuntimeError(
                f"Grouping Error: Model has {len(trainable_params)} trainable params, "
                f"but function only grouped {len(param_ids)}. "
                "Check for frozen layers or shared parameters."
        )

        for i, group in enumerate(groups) :
            print(f"Group {i} -> {len(group)}")
    
        return groups
    
    @staticmethod
    def get_efficientnet_groups(model):
        """
        EfficientNet layer groups.
        Drills into model.blocks for balanced parameter distribution.
        """
        # Initialize 5 parameter groups
        groups = [[], [], [], [], []]
        
        # Helper to safely add parameters from a module
        def append_params(group_idx, module):
            if module is not None:
                groups[group_idx].extend(list(module.parameters()))
    
        # --- 1. STEM & EARLY ENTRY (Group 0) ---
        # The Stem and BN1 handle the raw image input.
        if hasattr(model, 'conv_stem'): append_params(0, model.conv_stem)
        if hasattr(model, 'bn1'):       append_params(0, model.bn1)
    
        # --- 2. BACKBONE BLOCKS (Groups 0, 1, 2, 3) ---
        if hasattr(model, 'blocks'):
            # Flatten the blocks if they are nested Sequential (common in timm)
            all_stages = list(model.blocks.children())
            total_stages = len(all_stages)
            
            for i, stage in enumerate(all_stages):
                # EfficientNet B4 typically has 7 stages (indices 0 to 6)
                # We map these stages to groups to maintain semantic gradient.
                
                if i == 0:
                    # Stage 0 is usually stride 1, keeping high resolution.
                    # It contextually belongs with the Stem.
                    append_params(0, stage)
                
                elif i <= 2:
                    # Stages 1 & 2: First significant downsampling.
                    append_params(1, stage)
                    
                elif i <= 4:
                    # Stages 3 & 4: The "Body" of the network.
                    append_params(2, stage)
                    
                else:
                    # Stages 5 & 6+: The Deepest features. 
                    # These are the most complex semantic features before the head.
                    append_params(3, stage)
    
        # --- 3. THE HEAD (Group 4) ---
        # This is the "Adapter". In Transfer Learning, we want the 
        # feature projection (conv_head) AND the classifier to learn fastest.
        
        # The Conv Head projects features to the final channel dimension
        if hasattr(model, 'conv_head'): append_params(4, model.conv_head)
        if hasattr(model, 'bn2'):       append_params(4, model.bn2)
        
        # The Global Pooling is parameter-less, so we skip to Classifier
        if hasattr(model, 'global_pool'): pass 
        
        # The Classifier (Linear Layer)
        if hasattr(model, 'classifier'): append_params(4, model.classifier)
        elif hasattr(model, 'fc'):       append_params(4, model.fc) # Legacy fallback

        for i, group in enumerate(groups) :
            print(f"Group {i} -> {len(group)}")
    
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
            self.focal_loss = FocalLoss(alpha=1.0, gamma=2.0, weights=self.class_weights_tensor).to(self.device)
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
            use_sam=True, primary_metric='recall',
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
            for param in self.model.parameters() :
                param.requires_grad = False
            top_groups = self.layer_groups[2:]  # Groups 2, 3, 4
            for group in top_groups:
                for param in group:
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
            lr_multipliers = [1/100, 1/10, 1/3, 1.0, 10.0]
            max_lr = [lr * m for m in lr_multipliers[:len(param_groups)]]

            scheduler = optim.lr_scheduler.OneCycleLR(
                optimizer.base_optimizer if use_sam else optimizer,
                max_lr=max_lr if freeze_mode != 'all_discriminative' else lr * 10,  # Max LR for classifier
                epochs=epochs,
                steps_per_epoch=len(train_loader),
                pct_start=0.3,
                div_factor=25.0,
                final_div_factor=1000.0
            )
        else:  # transformer
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer.base_optimizer if use_sam else optimizer,
                T_0 = (epochs // 7) + 1,
                T_mult = 2,
                eta_min = 1e-7
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
