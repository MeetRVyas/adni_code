from sklearn.metrics import (
    accuracy_score, 
    roc_auc_score, 
    cohen_kappa_score, 
    classification_report, 
    confusion_matrix,
    matthews_corrcoef,
    jaccard_score,
    precision_score,
    recall_score,
    f1_score
)

import os
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from .config import DEVICE, REPORTS_DIR, NUM_SAMPLES_TO_ANALYSE
from .visualization import Visualizer
from .utils import get_base_transformations
from .models import get_img_size
from .classifiers import BaseClassifier


def test_model(model_name, model, loader, classes, experiment_name, logger, history = None,
               visualizer=None, use_tta=False):
    """
    Unified testing function for all model types.
    
    Automatically detects if model is a classifier object (has evaluate() method)
    or a raw PyTorch model.
    
    Args:
        model_name: Name of the model architecture
        model: PyTorch model OR classifier object
        loader: Test data loader
        classes: List of class names
        experiment_name: Name for saving results
        history: Training history
        logger: Logger instance
        visualizer: Visualizer instance (optional)
        use_tta: If True, use TTA (only for raw PyTorch models)
    
    Returns:
        metrics_dict: Dictionary with all metrics
    """
    
    # Create visualizer (test.py creates its own - cross_validation doesn't need to!)
    if not visualizer :
        img_size = get_img_size(model_name)
        transform = get_base_transformations(img_size)
        visualizer = Visualizer(
            experiment_name=experiment_name,
            model_name=model_name,
            class_names=classes,
            transform=transform,
            logger=logger
        )
    
    
    # Check if this is a classifier object (has evaluate method)
    if hasattr(model, 'evaluate'):
        logger.info(f"--- Detected Classifier Object: Using classifier.evaluate() ---")
        
        # Use classifier's built-in evaluate method
        result = model.evaluate(loader, class_names=classes)
        y_true = np.array(result["labels"])
        y_pred = np.array(result["preds"])
        y_prob = np.array(result["probs"])
        accuracy = result["accuracy"]
        recall = result["recall"]
        precision = result["precision"]
        f1 = result["f1"]
        per_class_recall = result["per_class_recall"]
        cm = result["confusion_matrix"]
        classification_report_ = result["report"]
        pytorch_model = model.model if hasattr(model, 'model') else None
    
    else:
        # Raw PyTorch model - use legacy test logic
        logger.info(f"--- Detected Raw PyTorch Model: Using standard testing ---")
        
        from .augmentation import TTAWrapper
        
        model.eval()
        all_preds = []
        all_labels = []
        all_probs = []
        
        if use_tta:
            logger.info("Using Test-Time Augmentation (TTA)")
            tta_model = TTAWrapper(model, num_augmentations=5)
        
        # Inference Loop
        with torch.inference_mode():
            for images, labels in loader:
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)
                
                if use_tta:
                    probs = tta_model(images)
                    preds = torch.argmax(probs, dim=1)
                else:
                    outputs = model(images)
                    probs = torch.softmax(outputs, dim=1)
                    preds = torch.argmax(outputs, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        # Convert to numpy
        y_true = np.array(all_labels)
        y_pred = np.array(all_preds)
        y_prob = np.array(all_probs)
        
        # Calculate Metrics
        accuracy = accuracy_score(y_true, y_pred) * 100
        recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
        precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
        f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        per_class_recall = recall_score(all_labels, all_preds, average=None, zero_division=0)
        cm = confusion_matrix(y_true, y_pred)
        classification_report_ = classification_report(y_true, y_pred, target_names=classes, digits=4)
        pytorch_model = model
    
    # NaN handling
    if np.isnan(y_prob).any():
        logger.warning("⚠️  NaN detected in probability outputs!")
        n_classes = len(classes)
        y_prob = np.nan_to_num(y_prob, nan=1.0/n_classes)
    
    # Verify probabilities sum to 1.0
    prob_sums = y_prob.sum(axis=1)
    if not np.allclose(prob_sums, 1.0, atol=1e-3):
        logger.warning("⚠️  Normalizing probabilities to sum=1.0")
        y_prob = y_prob / prob_sums[:, np.newaxis]
    
    # ROC AUC
    try:
        if len(classes) == 2:
            roc_auc = roc_auc_score(y_true, y_prob[:, 1])
        else:
            roc_auc = roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')
    except:
        logger.warning(f"Unexpected error in ROC AUC calculation: {e}")
        roc_auc = 0.0
    
    kappa = cohen_kappa_score(y_true, y_pred)
    corrcoef = matthews_corrcoef(y_true, y_pred)
    jaccard = jaccard_score(y_true, y_pred, average="weighted")
    
    # Generate Report
    report_path = os.path.join(REPORTS_DIR, f"{experiment_name}.txt")
    
    with open(report_path, "w") as f:
        f.write(f"===== COMPREHENSIVE ANALYSIS REPORT: {experiment_name} =====\n\n")
        f.write(f"Test-Time Augmentation: {'Enabled' if use_tta else 'Disabled'}\n\n")
        f.write("--- Overall Performance ---\n")
        f.write(f"Overall Accuracy: {accuracy:.2f}%\n")
        f.write(f"Overall Recall: {recall:.4f}\n")
        f.write(f"Overall Precision: {precision:.4f}\n")
        f.write(f"Overall F1: {f1:.4f}\n")
        f.write(f"\nPer-Class Recall")
        for i, (name, rec) in enumerate(zip(classes, per_class_recall)):
            f.write(f"  {name}: {rec:.4f}")
        f.write(f"Macro ROC AUC:    {roc_auc:.4f}\n")
        f.write(f"Cohen's Kappa:    {kappa:.4f}\n")
        f.write(f"Matthews Correlation Coefficient (MCC): {corrcoef:.4f}\n")
        f.write(f"Jaccard Score:    {jaccard:.4f}\n\n")
        
        f.write("--- Detailed Per-Class Metrics ---\n")
        f.write(classification_report_)
        
        f.write("\n--- Per-Class Specificity & Confusion Matrix Stats ---\n")
        
        for i, class_name in enumerate(classes):
            tp = cm[i, i]
            fp = cm[:, i].sum() - tp
            fn = cm[i, :].sum() - tp
            tn = cm.sum() - (tp + fp + fn)
            
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            f.write(f"{class_name:<20}: Specificity: {specificity:.4f}, "
                    f"TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn}\n")
    
    logger.info(f"Report saved to: {report_path}")
    
    # Generate ALL visualizations
    logger.info("Generating comprehensive visualizations...")
    
    visualizer.generate_all_plots(
        y_true=y_true,
        y_prob=y_prob,
        history=history,
        model=pytorch_model,
        test_loader=loader,
        xai_samples=NUM_SAMPLES_TO_ANALYSE  # Enable GradCAM/LIME/SHAP
    )
    
    logger.info("Testing and visualization complete")
    
    return {
        'accuracy': accuracy,
        'recall': recall,
        'precision': precision,
        'f1': f1,
        'per_class_recall': per_class_recall,
        'roc_auc': roc_auc,
        'kappa': kappa,
        'corrcoef': corrcoef,
        'jaccard': jaccard,
        'y_true': y_true,
        'y_prob': y_prob,
        'y_pred': y_pred,
    }
