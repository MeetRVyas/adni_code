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

# Custom Reshape function for Swin Transformers
def reshape_transform_swin(tensor, height=7, width=7):
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
        for h in self.hooks: h.remove()

def generate_gradcam_plot(model, input_tensor, original_img_np, target_layer, reshape_transform=None, alpha=0.4):
    """
    Visualization logic using OpenCV.
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
    def __init__(self, experiment_name, model_name, class_names, transform = None, logger=None):
        """
        Initializes the visualizer for a specific experiment.
        
        Args:
            experiment_name (str): Unique identifier for the experiment/fold.
            model_name (str): Name of the architecture (used for XAI layer detection).
            class_names (list): List of class string names.
            logger: Optional logger instance.
        """
        self.experiment_name = experiment_name
        self.model_name = model_name
        self.class_names = class_names
        self.logger = logger
        self.img_size = get_img_size(model_name)
        self.transform = transform or get_base_transformations(self.img_size)
        
        # Create output directory
        self.save_dir = os.path.join(PLOTS_DIR, experiment_name)
        os.makedirs(self.save_dir, exist_ok=True)

    def log(self, msg):
        if self.logger: self.logger.info(msg)
        else: print(msg)

    # =========================================================================
    # 1. TRAINING HISTORY (MERGED & IMPROVED)
    # =========================================================================
    def plot_training_history(self, history):
        """
        Plots a comprehensive grid of training metrics: Loss, Accuracy, F1, Precision, Recall.
        Accepts a list of dictionaries (from your training loop).
        """
        if not history: return

        # Extract data
        epochs = [h['epoch'] + 1 for h in history]
        metrics = ['loss', 'acc', 'f1', 'prec', 'rec']
        titles = ['Cross Entropy Loss', 'Accuracy (%)', 'F1 Score', 'Precision', 'Recall']
        
        # Prepare figure
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(f"Training Dynamics: {self.experiment_name}", fontsize=16, weight='bold')
        axes = axes.flatten()

        for i, metric in enumerate(metrics):
            ax = axes[i]
            # Handle Train vs Val keys
            train_key = f'train_{metric}'
            val_key = f'val_{metric}'
            
            # Plot Training if available
            if train_key in history[0]:
                train_vals = [h[train_key] for h in history]
                ax.plot(epochs, train_vals, 'o--', label='Train', color='cornflowerblue', linewidth=2)
            
            # Plot Validation
            if val_key in history[0]:
                val_vals = [h[val_key] for h in history]
                ax.plot(epochs, val_vals, 'o-', label='Validation', color='darkorange', linewidth=2)
            
            ax.set_title(titles[i])
            ax.set_xlabel("Epochs")
            ax.grid(True, linestyle='--', alpha=0.6)
            ax.legend()

        # Hide empty subplot if any (we have 5 metrics, 6 slots)
        # (if len(metrics) < 6)
        axes[-1].axis('off')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        save_path = os.path.join(self.save_dir, "training_history.png")
        try :
            plt.savefig(save_path, dpi=300)
        except Exception as e :
            self.logger.error(f"Error while plotting -> {e}")
        plt.close()

    # =========================================================================
    # 2. MODEL PERFORMANCE PLOTS (IMPROVED "OTHERS")
    # =========================================================================
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
        try :
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        except Exception as e :
            self.logger.error(f"Error while plotting -> {e}")
        plt.close()

    def plot_classwise_metrics(self, y_true, y_pred):
        """
        Generates a Heatmap of Precision, Recall, and F1-Score per class.
        This is cleaner than bar charts for multi-class problems.
        """
        report = classification_report(y_true, y_pred, target_names=self.class_names, output_dict=True)
        
        # Prepare Dataframe
        metrics_list = []
        for cls in self.class_names:
            if cls in report:
                metrics_list.append({
                    'Class': cls,
                    'Precision': report[cls]['precision'],
                    'Recall': report[cls]['recall'],
                    'F1-Score': report[cls]['f1-score']
                })
            else :
                self.logger.error(f"Class {cls} not found in classification report\n{report}")
        
        df = pd.DataFrame(metrics_list).set_index('Class')
        
        plt.figure(figsize=(8, len(self.class_names) * 0.8 + 2))
        sns.heatmap(df, annot=True, cmap='RdYlGn', fmt='.3f', vmin=0.0, vmax=1.0, linewidths=1)
        plt.title(f'Class-wise Performance Metrics', fontsize=14)
        plt.yticks(rotation=0)
        
        save_path = os.path.join(self.save_dir, "classwise_metrics.png")
        try :
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        except Exception as e :
            self.logger.error(f"Error while plotting -> {e}")
        plt.close()

    def plot_roc_curve(self, y_true, y_prob):
        """Plots both ROC curve."""
        n_classes = len(self.class_names)
        y_true_bin = pd.get_dummies(y_true).values
        
        plt.figure(figsize=(10, 8))
        colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green', 'red', 'purple'])
        for i, color in zip(range(n_classes), colors):
            if i < y_prob.shape[1]:
                fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
                auc_score = auc(fpr, tpr)
                plt.plot(fpr, tpr, color=color, lw=2, 
                         label=f'ROC of {self.class_names[i]} (AUC = {auc_score:0.2f})')
        
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curves: {self.experiment_name}')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        try :
            plt.savefig(os.path.join(self.save_dir, "roc_curve.png"), dpi=300)
        except Exception as e :
            self.logger.error(f"Error while plotting -> {e}")
        plt.close()

    def plot_precision_recall_curve(self, y_true, y_prob):
        """Plots both PR curve."""
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
        try :
            plt.savefig(os.path.join(self.save_dir, "precision_recall_curve.png"), dpi=300)
        except Exception as e :
            self.logger.error(f"Error while plotting -> {e}")
        plt.close()

    def plot_confidence_distribution(self, y_true, y_prob):
        """
        Plots histograms of prediction confidence, split by Correct vs Incorrect predictions.
        Crucial for analyzing model calibration.
        """
        y_pred = np.argmax(y_prob, axis=1)
        confidences = np.max(y_prob, axis=1)
        
        correct_mask = (y_pred == y_true)
        incorrect_mask = ~correct_mask
        
        plt.figure(figsize=(10, 6))
        
        # Plot Correct
        sns.histplot(confidences[correct_mask], color='green', label='Correct Predictions', 
                     kde=True, bins=20, alpha=0.5, element="step")
        
        # Plot Incorrect
        if np.any(incorrect_mask):
            sns.histplot(confidences[incorrect_mask], color='red', label='Incorrect Predictions', 
                         kde=True, bins=20, alpha=0.5, element="step")
            
        plt.xlabel("Prediction Confidence (Probability)")
        plt.ylabel("Count")
        plt.title(f"Confidence Distribution{' : Correct vs Incorrect' if np.any(incorrect_mask) else ''}")
        plt.legend()
        
        save_path = os.path.join(self.save_dir, "confidence_analysis.png")
        try :
            plt.savefig(save_path, dpi=300)
        except Exception as e :
            self.logger.error(f"Error while plotting -> {e}")
        plt.close()

    def plot_cumulative_gain(self, y_true, y_prob):
        """
        Plots cumulative gain curve and saves to file.
        """
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

            class_label = self.class_names[i] if self.class_names else f"Class {i}"
            plt.plot(percentages, cum_gains, color=color, lw=2, label=f'{class_label}')

        plt.xlabel("Percentage of Sample Targeted")
        plt.ylabel("Cumulative Gain")
        plt.title(f"Cumulative Gain Curve: {self.experiment_name}")
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        save_path = os.path.join(self.save_dir, "cumulative_gain.png")
        try :
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
        except Exception as e :
            self.logger.error(f"Error while plotting -> {e}")
        plt.close()
        self.logger.info(f"Cumulative Gain curve saved to {save_path}")

    # =========================================================================
    # 3. EXPLAINABILITY (XAI) MODULE
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
            else:
                # Recursive fallback to last Conv2d
                for name, module in list(model.named_modules())[::-1]:
                    if isinstance(module, torch.nn.Conv2d):
                        return module
        except Exception as e :
            self.logger.error(f"Error determining target layer: {e}")
        return None

    def run_grad_cam(self, model, image_path, target_layer):
        """Returns the Grad-CAM visualization array (RGB)."""
        # try:
        #     from pytorch_grad_cam import GradCAM
        #     from pytorch_grad_cam.utils.image import show_cam_on_image
            
        #     rgb_img = Image.open(image_path).convert('RGB').resize((self.img_size, self.img_size))
        #     input_tensor = self.transform(rgb_img).unsqueeze(0).to(DEVICE)
            
        #     # Check for Swin specific reshape
        #     reshape = reshape_transform_swin if "swin" in model.default_cfg['architecture'] else None
            
        #     with GradCAM(model=model, target_layers=[target_layer], reshape_transform=reshape) as cam:
        #         grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0, :]
        #         # Combine image and heatmap
        #         grad_cam_img = show_cam_on_image(np.float32(rgb_img) / 255, grayscale_cam, use_rgb=True)
        #         return grad_cam_img
        # except Exception as e:
        #     self.logger.error(f"[Grad-CAM Error]: {e}")
        #     self.log(f"Trying GradCAM from scratch for {self.experiment_name}")
        try:
            pil_img = Image.open(image_path).convert('RGB').resize((self.img_size, self.img_size))
            input_tensor = self.transform(pil_img).unsqueeze(0).to(DEVICE)
            original_img_np = np.array(pil_img)
            reshape = reshape_transform_swin if "swin" in str(type(model)).lower() else None
            viz_img = generate_gradcam_plot(model, input_tensor, original_img_np, target_layer, reshape)
            return viz_img
            
        except Exception as e:
            print(f"[Grad-CAM Error]: {e}")
            return np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)

    def run_lime(self, model, image_path):
        """Returns the LIME explanation image array."""
        try:
            from lime import lime_image
            from skimage.segmentation import mark_boundaries
            
            def batch_predict(images):
                # LIME passes numpy arrays, need to convert to Tensor
                pil_images = [Image.fromarray(image) for image in images]
                batch = torch.stack([self.transform(img) for img in pil_images], dim=0).to(DEVICE)
                with torch.no_grad():
                    logits = model(batch)
                return F.softmax(logits, dim=1).cpu().numpy()
            
            image_np = np.array(Image.open(image_path).convert('RGB').resize((self.img_size, self.img_size)))
            explainer = lime_image.LimeImageExplainer()
            
            # Reduced num_samples for speed in automated runs (default is 1000)
            explanation = explainer.explain_instance(image_np, batch_predict, top_labels=1, hide_color=0, num_samples=500)
            
            temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=5, hide_rest=False)
            lime_viz = mark_boundaries(temp / 2 + 0.5, mask)
            return np.clip(lime_viz, 0, 1)
        except Exception as e:
            self.logger.error(f"[LIME Error]: {e}")
            return np.zeros((self.img_size, self.img_size, 3))

    def run_shap(self, model, image_path):
        """
        Generates SHAP plot, saves it temporarily, reads it back as an image array.
        This is necessary because SHAP plots directly to matplotlib axes.
        """
        try:
            import shap
            
            # Prepare inputs
            image = Image.open(image_path).convert('RGB').resize((self.img_size, self.img_size))
            input_tensor = self.transform(image).unsqueeze(0).to(DEVICE)
            
            # Background for GradientExplainer (random noise or subset of data)
            # Using random noise here for speed, ideally use X_train summary
            background = torch.randn(5, 3, self.img_size, self.img_size).to(DEVICE)
            
            explainer = shap.GradientExplainer(model, background)
            shap_values = explainer.shap_values(input_tensor)
            
            # SHAP returns a list (one per class). We take the one for the predicted class.
            with torch.no_grad():
                output = model(input_tensor)
                pred_idx = torch.argmax(output).item()
                
            # Format for plotting
            shap_numpy = np.swapaxes(np.swapaxes(shap_values[pred_idx], 1, -1), 1, 2)
            # shap_numpy = [np.swapaxes(np.swapaxes(s, 1, -1), 1, 2) for s in shap_values]
            input_numpy = np.swapaxes(np.swapaxes(input_tensor.cpu().numpy(), 1, -1), 1, 2)
            
            # Plot to a temporary file
            fig = plt.figure(figsize=(4, 4))
            shap.image_plot(shap_numpy, -input_numpy, show=False)
            
            temp_filename = f"temp_shap_{os.getpid()}.png"
            try :
                plt.savefig(temp_filename, bbox_inches='tight', pad_inches=0)
            except Exception as e :
                self.logger.error(f"Error while plotting -> {e}")
            plt.close(fig)
            
            # Read back
            shap_img = np.array(Image.open(temp_filename).convert('RGB'))
            if os.path.exists(temp_filename):
                os.remove(temp_filename)
                
            return shap_img
        except Exception as e:
            print(f"[SHAP Error]: {e}")
            return np.zeros((self.img_size, self.img_size, 3))
    
    # --- INTEGRATION FUNCTION ---

    def generate_xai_comparison_plot(self, model, image_path, sample_id):
        """
        Master function that runs Grad-CAM, LIME, and SHAP, then stitches them into a 4-panel plot.
        """
        model.eval()
        
        # 1. Get Target Layer
        target_layer = self._get_target_layer(model)
        
        # 2. Run Individual XAI
        self.log(f"  Running XAI for sample {sample_id}...")
        grad_cam_viz = self.run_grad_cam(model, image_path, target_layer)
        lime_viz = self.run_lime(model, image_path)
        shap_viz = self.run_shap(model, image_path)
        
        # 3. Original Image
        original_img = np.array(Image.open(image_path).convert('RGB').resize((self.img_size, self.img_size)))
        
        # 4. Plotting
        fig, axes = plt.subplots(1, 4, figsize=(24, 6))
        
        axes[0].imshow(original_img)
        axes[0].set_title("Original", fontsize=16)
        
        axes[1].imshow(grad_cam_viz)
        axes[1].set_title("Grad-CAM", fontsize=16)
        
        axes[2].imshow(lime_viz)
        axes[2].set_title("LIME", fontsize=16)
        
        axes[3].imshow(shap_viz)
        axes[3].set_title("SHAP", fontsize=16)
        
        for ax in axes: ax.axis('off')
        
        image_basename = os.path.splitext(os.path.basename(image_path))[0]
        fig.suptitle(f"XAI Analysis: {image_basename}.jpg ({self.model_name})", fontsize=20)
        
        # Save
        output_filename = os.path.join(self.save_dir, f"xai_comparision_{sample_id}_{image_basename}.png")
        try :
            plt.savefig(output_filename, bbox_inches='tight')
        except Exception as e :
            self.logger.error(f"Error while plotting -> {e}")
        plt.close(fig)
        
        return output_filename

    # =========================================================================
    # 4. MASTER GENERATOR
    # =========================================================================
    def generate_all_plots(self, y_true, y_prob, history = None, model=None, test_loader=None, xai_samples = 5):
        """
        One-stop function to generate all statistical and visual reports.
        """
        self.log(f"Generating visualizations for {self.experiment_name}...")
        
        y_pred = np.argmax(y_prob, axis=1)
        
        if model and test_loader and xai_samples > 0:
            self.log(f"Generating XAI reports for {xai_samples} samples...")
            
            # Get samples correctly
            dataset = test_loader.dataset
            indices = np.random.choice(len(dataset), xai_samples, replace=False)
            
            import copy
            
            # Find image paths
            # If Subset, we map indices
            full_ds = dataset.dataset if hasattr(dataset, 'dataset') else dataset
            
            for i, idx in enumerate(indices):
                # ImageFolder stores path in .samples
                if hasattr(full_ds, 'samples'):
                    img_path = full_ds.samples[idx][0]
                    self.generate_xai_comparison_plot(model, img_path, sample_id=f"sample_{i}")
                else:
                    self.log("Dataset does not support path retrieval for XAI.")
                    break
        
        if model and test_loader :
            import gc
            del model, test_loader
            torch.cuda.empty_cache()
            gc.collect()
        
        self.plot_training_history(history = history)
        self.plot_confusion_matrix(y_true, y_pred, normalize=True)
        self.plot_classwise_metrics(y_true, y_pred)
        self.plot_roc_curve(y_true, y_prob)
        self.plot_precision_recall_curve(y_true, y_prob)
        self.plot_confidence_distribution(y_true, y_prob)
        self.plot_cumulative_gain(y_true, y_prob)