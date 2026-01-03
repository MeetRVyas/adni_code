import torch.nn as nn
import torch.optim as optim

class Training_Strategy:
    """Base class for training strategies."""
    
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
        """Must be implemented by subclasses."""
        raise NotImplementedError


class SimpleCosineLRStrategy(Training_Strategy):
    """Simple full fine-tuning with CosineAnnealing."""
    def setup(self):
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=0.01)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.epochs)
        self.criterion = nn.CrossEntropyLoss(weight=self.class_weights_tensor) if self.class_weights_tensor is not None else nn.CrossEntropyLoss()
        return "AdamW + CosineAnnealing + Full Fine-tuning"


class DiscriminativeOneCycleStrategy(Training_Strategy):
    """Discriminative LR with OneCycleLR."""
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
        """Safely separate model parameters."""
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
    """Full fine-tuning with CosineAnnealingWarmRestarts + Class Weights."""
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
    """Aggressive training with high LR, warmup, and augmentations."""
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
    """Conservative SGD with momentum - for difficult datasets."""
    def setup(self):
        self.optimizer = optim.SGD(
            self.model.parameters(), lr=self.lr, momentum=0.9, 
            weight_decay=0.01, nesterov=True
        )
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=self.epochs // 3, gamma=0.1)
        self.criterion = nn.CrossEntropyLoss(weight=self.class_weights_tensor)
        return "SGD + StepLR + Nesterov"