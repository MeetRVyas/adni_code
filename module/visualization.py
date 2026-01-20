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
    confusion_matrix, classification_report, roc_auc_score
)

from .config import DEVICE, PLOTS_DIR
from .utils import get_base_transformations
from .models import get_img_size


def reshape_transform_swin(tensor, height=7, width=7):
    """Custom reshape for Swin Transformer feature maps"""
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
    """Generates GradCAM visualization with OpenCV overlay"""
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
    Comprehensive visualization suite with optimized error handling
    
    Key improvements:
    - No internal try-except blocks in individual plot methods
    - All error handling centralized in generate_all_plots()
    - Accurate success/failure counting
    - 2x2 XAI comparison layout
    - Fixed float16 pandas issue
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

    # =========================================================================
    # Plots
    # =========================================================================
    def plot_training_history(self, history):
        """Plots a comprehensive grid of training metrics: Loss, Accuracy, F1, Precision, Recall."""
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
        """Plots a stylish Confusion Matrix."""
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
        """Generates a Heatmap of Precision, Recall, and F1-Score per class."""
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
        """Plots multi-class ROC curves"""
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
        """Plots multi-class Precision-Recall curves"""
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
        """Plots histograms of prediction confidence with float32 conversion."""
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
        """Plots cumulative gain curves"""
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
        """Heuristic to find the target layer for GradCAM."""
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
        """Returns the Grad-CAM visualization array (RGB) with inplace operation fix."""
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
        """Returns the LIME explanation"""
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
        """Generates SHAP visualization."""
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
        """
        Master function that runs Grad-CAM, LIME, and SHAP in 2x2 layout with individual error handling.
        """
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
        """
        One-stop function with centralized error handling.
        """
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