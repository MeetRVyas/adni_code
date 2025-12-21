import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from itertools import cycle
from PIL import Image
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, average_precision_score,
    confusion_matrix, classification_report
)

from config import DEVICE, PLOTS_DIR
from utils import get_base_transformations
from models import get_img_size


def reshape_transform_swin(tensor, height=7, width=7):
    """Custom reshape for Swin Transformer feature maps"""
    result = tensor.reshape(tensor.size(0), height, width, tensor.size(2))
    result = result.permute(0, 3, 1, 2)
    return result


class NativeGradCAM:
    """
    Lightweight GradCAM implementation without external dependencies
    
    More efficient than pytorch_grad_cam for batch processing
    """
    
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
    """
    Generates GradCAM visualization with OpenCV overlay
    
    Args:
        model: PyTorch model
        input_tensor: Preprocessed input (1, C, H, W)
        original_img_np: Original image as numpy array (H, W, 3)
        target_layer: Layer to visualize
        reshape_transform: Optional transform for feature maps
        alpha: Overlay transparency
    
    Returns:
        np.ndarray: Superimposed heatmap visualization
    """
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
    """
    Comprehensive visualization suite for model analysis
    
    Generates:
    - Training curves (loss, accuracy, F1, precision, recall)
    - Confusion matrices
    - ROC and Precision-Recall curves
    - Per-class performance heatmaps
    - Confidence distribution analysis
    - Cumulative gain curves
    - XAI visualizations (GradCAM only for efficiency)
    """
    
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
        """Plots comprehensive training dynamics across all metrics"""
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
        """Plots confusion matrix with optional normalization"""
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
        """Generates heatmap of per-class precision, recall, F1"""
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
        """Plots multi-class ROC curves"""
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
        """Plots multi-class Precision-Recall curves"""
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
        """Analyzes model confidence for correct vs incorrect predictions"""
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
        """Plots cumulative gain curves for model ranking ability"""
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
        """
        Heuristic layer selection for GradCAM
        
        Targets the last convolutional/attention layer in the architecture
        """
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
        """
        Generates LIME explanation for model predictions
        
        Returns:
            np.ndarray: LIME visualization or None if failed
        """
        try:
            from lime import lime_image
            from skimage.segmentation import mark_boundaries
            
            def batch_predict(images):
                """Prediction function for LIME"""
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
        """
        Generates SHAP explanation for model predictions
        
        Returns:
            np.ndarray: SHAP visualization image or None if failed
        """
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
        """
        Generates comprehensive XAI comparison: Original, GradCAM, LIME, SHAP
        
        Args:
            model: Trained PyTorch model
            image_path: Path to input image
            sample_id: Unique identifier for this sample
        
        Returns:
            str: Path to saved comparison plot, or None if failed
        """
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
        """
        Master function to generate all visualization outputs with comprehensive error handling
        
        Each visualization is wrapped in try-except to ensure one failure doesn't stop others
        
        Args:
            y_true: Ground truth labels
            y_prob: Predicted probabilities
            history: Training history (optional)
            model: Trained model (optional, for XAI)
            test_loader: Test data loader (optional, for XAI)
            xai_samples: Number of samples for XAI analysis
        """
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
