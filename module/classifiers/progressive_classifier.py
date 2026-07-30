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
    def _verify_full_coverage(model, groups, arch_name):
        """
        Shared safeguard, factored out so every architecture-specific grouping
        function gets it for free instead of reimplementing it (previously only
        get_swin_groups had this check).

        Confirms every trainable parameter tensor in `model` was assigned to
        exactly one discriminative-LR group. Raises loudly on any gap instead of
        letting an incomplete grouping silently freeze part of the network
        during progressive fine-tuning (a missing parameter simply never
        receives a param_group / gradient update).
        """
        covered_ids = set()
        duplicate_ids = set()
        for group in groups:
            for p in group:
                if id(p) in covered_ids:
                    duplicate_ids.add(id(p))
                covered_ids.add(id(p))

        trainable = [p for p in model.parameters() if p.requires_grad]
        trainable_ids = {id(p) for p in trainable}
        missing_ids = trainable_ids - covered_ids

        if missing_ids:
            name_by_id = {id(p): n for n, p in model.named_parameters()}
            sample = [name_by_id.get(i, "<unnamed>") for i in list(missing_ids)[:10]]
            raise RuntimeError(
                f"[{arch_name}] Layer-grouping coverage check failed: "
                f"{len(missing_ids)} of {len(trainable)} trainable parameter tensors "
                f"were not assigned to any discriminative-LR group, e.g. {sample}. "
                f"Refusing to start training with an incomplete grouping."
            )

        if duplicate_ids:
            name_by_id = {id(p): n for n, p in model.named_parameters()}
            sample = [name_by_id.get(i, "<unnamed>") for i in list(duplicate_ids)[:10]]
            print(
                f"[{arch_name}] Warning: {len(duplicate_ids)} parameter tensor(s) were "
                f"assigned to more than one group, e.g. {sample}. This does not raise "
                f"because it can't silently drop gradients the way a missing parameter "
                f"can, but it does mean that parameter's effective LR depends on optimizer "
                f"param-group ordering — worth resolving before treating this mapping as final."
            )

        for i, group in enumerate(groups):
            print(f"[{arch_name}] Group {i} -> {len(group)} parameter tensors")

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
        ArchitectureLayerGroups._verify_full_coverage(model, groups, "resnet")
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
        
        ArchitectureLayerGroups._verify_full_coverage(model, groups, "vit")
        return groups
    
    @staticmethod
    def get_swin_groups(model):
        """Swin Transformer layer groups."""
        # We initialize 5 groups as per your original logic
        groups = [[] for _ in range(5)]

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
            
        ArchitectureLayerGroups._verify_full_coverage(model, groups, "swin")
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

        ArchitectureLayerGroups._verify_full_coverage(model, groups, "efficientnet")
        return groups
    
    @staticmethod
    def get_mobilenet_groups(model):
        """MobileNet layer groups."""
        if hasattr(model, 'features'):
            features = model.features
            n_features = len(features)

            groups = [
                list(features[:n_features//4].parameters()),
                list(features[n_features//4:n_features//2].parameters()),
                list(features[n_features//2:3*n_features//4].parameters()),
                list(features[3*n_features//4:].parameters()),
                list(model.classifier.parameters()) if hasattr(model, 'classifier') else []
            ]
        else:
            # Fallback: unstructured even split. Coverage is trivially 100% here
            # by construction (every trainable param is sliced into exactly one
            # group), but the LR "grouping" carries no architectural meaning —
            # kept only so mobilenet variants without a `.features` attribute
            # don't crash outright.
            all_params = [p for p in model.parameters() if p.requires_grad]
            n = len(all_params)
            groups = [
                all_params[:n//5],
                all_params[n//5:2*n//5],
                all_params[2*n//5:3*n//5],
                all_params[3*n//5:4*n//5],
                all_params[4*n//5:]
            ]

        ArchitectureLayerGroups._verify_full_coverage(model, groups, "mobilenet")
        return groups
    
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
    - Phase 1 (5 epochs): Classifier only, trained on class-balanced batches
      (WeightedRandomSampler via _build_balanced_loader) with an unweighted loss
    - Phase 2 (10 epochs): Top 50% layers, natural distribution, unweighted loss
    - Phase 3 (15 epochs): All layers with discriminative LRs + optional SAM,
      natural distribution, full effective-number class weights
    - Architecture-aware layer grouping
    - Focal Loss, with LDAM-DRW-style deferred reweighting across the 3 phases
      (see compute_loss/_phase_class_weights)
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

        self.phases = 3
        self.lr_multipliers = [1/100, 1/10, 1/3, 1.0, 10.0]

        def _get_scheduler(optimizer, use_sam, epochs, **kwargs) :
            return optim.lr_scheduler.OneCycleLR(
                optimizer.base_optimizer if use_sam else optimizer,
                max_lr=kwargs["max_lr"],
                epochs=epochs,
                steps_per_epoch=kwargs["steps_per_epoch"],
                pct_start=0.3,
                div_factor=25.0,
                final_div_factor=1000.0
            )

        self._get_scheduler = _get_scheduler

    def set_phases(self, phases : int) -> None :
        self.phases = max(1, min(phases, 5))
        print(f"Total Phases set: {self.phases}")

    def set_sequential_scheduler(self) -> None :
        def _get_scheduler(optimizer, use_sam, epochs, **kwargs) :
            base_sched = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer.base_optimizer if use_sam else optimizer,
                T_0=(epochs // 7) + 1, T_mult=2, eta_min=1e-7
            )
            warmup_sched = optim.lr_scheduler.LinearLR(
                optimizer.base_optimizer if use_sam else optimizer,
                start_factor=0.01, end_factor=1.0, total_iters=1  # ramps over epoch 1, given epoch-level .step()
            )
            return optim.lr_scheduler.SequentialLR(
                optimizer.base_optimizer if use_sam else optimizer,
                schedulers=[warmup_sched, base_sched], milestones=[1]
            )
        self._get_scheduler = _get_scheduler
    
    def forward(self, images):
        """Standard forward pass."""
        return self.model(images)
    
    def compute_loss(self, outputs, labels):
        """
        Focal Loss for hard examples, with LDAM-DRW-style deferred
        reweighting (Cao et al., "Learning Imbalanced Datasets with
        Label-Distribution-Aware Margin Loss", https://arxiv.org/abs/1906.07413):
        applying full class weights from epoch 1 can hurt the initial
        representation, so weight strength is scheduled by phase rather than
        fixed for the whole run.
        """
        phase = getattr(self, 'current_phase', 3)  # default to full weighting if called outside _train_phase
        target_weights = self._phase_class_weights(phase)

        weights_changed = (
            not hasattr(self, 'focal_loss')
            or getattr(self, '_focal_loss_weights_id', None) is not (
                id(target_weights) if target_weights is not None else None
            )
        )
        if weights_changed:
            self.focal_loss = FocalLoss(alpha=1.0, gamma=2.0, weights=target_weights).to(self.device)
            self._focal_loss_weights_id = id(target_weights) if target_weights is not None else None
        return self.focal_loss(outputs, labels)

    def _phase_class_weights(self, phase):
        """
        Phase 1/2: no class weighting (uniform) — classifier warm-up (Phase 1)
        and top-layer fine-tuning (Phase 2) train on the natural distribution
        with an unweighted loss, so early representation learning isn't
        distorted by aggressive reweighting before the model has learned
        anything useful to reweight.
        Phase 3: full effective-number class weights.
        """
        if self.class_weights_tensor is None:
            return None
        return self.class_weights_tensor if phase == self.phases else None
    
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
        
        param_groups = []
        for params, mult in zip(self.layer_groups, self.lr_multipliers):
            if params:
                param_groups.append({
                    'params': params,
                    'lr': base_lr * mult
                })
        
        return param_groups

    def _build_balanced_loader(self, train_loader):
        """
        Phase 1 only: retrain the classifier head on a class-balanced view of
        the data rather than the natural distribution, following the
        "decoupling representation and classifier" approach for long-tailed
        recognition (Kang et al., https://arxiv.org/abs/1910.09217.
        """
        from torch.utils.data import DataLoader
        from module.training.data_split import class_balanced_sampler

        subset = train_loader.dataset  # a torch.utils.data.Subset(full_dataset, indices)
        sampler = class_balanced_sampler(subset.dataset, np.array(subset.indices))
        return DataLoader(
            subset,
            batch_size=train_loader.batch_size,
            sampler=sampler,
            num_workers=train_loader.num_workers,
            pin_memory=train_loader.pin_memory,
        )
    
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

        remaining_epochs = epochs

        for i in range(1, self.phases) :
            print("="*80)
            print(f"PHASE {i}")
            print("="*80)
            
            self._train_phase(
                phase=i,
                train_loader=self._build_balanced_loader(train_loader),
                val_loader=val_loader,
                epochs=min(5 * i, remaining_epochs),
                lr=lr,  # Higher LR for random classifier
                use_sam=False,
                primary_metric=primary_metric,
                patience=5 * i,
                min_delta=min_delta
            )

            remaining_epochs = max(0, remaining_epochs - 5 * i)
            if remaining_epochs == 0 :
                break
        else :
            # Final Phase: All layers with discriminative LRs (remaining epochs)
            print("\n" + "="*80)
            print("Final PHASE: Discriminative Fine-Tuning (All Layers)")
            print("="*80)
            
            self._train_phase(
                phase=self.phases,
                train_loader=train_loader,
                val_loader=val_loader,
                epochs=remaining_epochs,
                lr=lr,
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
                    use_sam, primary_metric, patience, min_delta):
        """Train a single phase."""
        self.current_phase = phase  # read by compute_loss() -> _phase_class_weights()

        if phase > 1:
            self.load(self.checkpoint_path)
            print(f"  Restored best checkpoint (recall={self.best_recall:.4f}) before Phase {phase}")
        
        # Freeze/unfreeze according to mode
        if phase != self.phases:
            for param in self.model.parameters():
                param.requires_grad = False
            grps = self.layer_groups[5 - phase:]
            for grp in grps :
                if grp:
                    for param in grp:
                        param.requires_grad = True
                    
            # Single LR
            param_groups = filter(lambda p: p.requires_grad, self.model.parameters())

            lr = lr * self.lr_multipliers[5 - phase]

        else:
            # Unfreeze everything
            for param in self.model.parameters():
                param.requires_grad = True
            # Discriminative LRs
            param_groups = self._get_discriminative_params(lr)
            print(f"Discriminative LR groups:")
            for i, group in enumerate(param_groups):
                print(f"  Group {i}: {len(list(group['params']))} params, LR={group['lr']:.2e}")
        
        if use_sam:
            optimizer = SAM(param_groups, optim.AdamW, lr=lr, weight_decay=0.01, rho=0.05)
        else:
            optimizer = optim.AdamW(param_groups, lr=lr, weight_decay=0.01)
        
        # Create scheduler
        max_lr = [lr * m for m in self.lr_multipliers[:len(list(param_groups))]]
        scheduler = self._get_scheduler(
            optimizer = optimizer,
            use_sam = use_sam,
            epochs = epochs,
            max_lr = max_lr if phase == self.phases else lr * 10,
            steps_per_epoch = len(train_loader),
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
            val_loss, val_acc, val_recall, val_prec, val_f1, primary_value, per_class_recall = self.validate_epoch(val_loader)
            
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
                'val_per_class_recall': per_class_recall,
                f'val_{primary_metric}': primary_value
            })

            print(f"[Epoch {epoch+1}] **{val_recall:.3f}**")
            
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
                self.save(self.checkpoint_path)
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

        for h in self.history :
            if h["phase"] == phase :
                print(
                    f"Train: L={h['train_loss']:.4f} A={h['train_acc']:.2f}% R={h['train_recall']:.3f} | "
                    f"Val: L={h['val_loss']:.4f} A={h['val_acc']:.2f}% "
                    f"P={h['val_prec']:.3f} R={h['val_recall']:.3f} F1={h['val_f1']:.3f}"
                )
                print(f"Per-class Recall: {[f'{r:.3f}' for r in h['per_class_recall']]}")
        
        print(f"âœ… Phase {phase} Complete - Best {primary_metric}: {self.best_metric_value:.4f}")
