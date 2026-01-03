from sklearn.metrics import (
    accuracy_score, 
    roc_auc_score, 
    cohen_kappa_score, 
    classification_report, 
    confusion_matrix,
    matthews_corrcoef,
    jaccard_score
)

import os
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from torchvision import transforms
from .config import DEVICE, REPORTS_DIR, NUM_SAMPLES_TO_ANALYSE
from .visualization import Visualizer
from .utils import Logger
from .augmentation import TTAWrapper


def test_model(model_name, model, loader, classes, experiment_name, history, logger: Logger, 
               visualizer: Visualizer = None, use_tta=False):
    """
    Performs comprehensive evaluation with optional Test-Time Augmentation.
    
    Args:
        use_tta: If True, use TTA for +2-5% accuracy boost (default: True)
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    logger.info(f"--- Starting Evaluation: {experiment_name} ---")
    if use_tta:
        logger.info("Using Test-Time Augmentation (TTA) for improved accuracy")
        tta_model = TTAWrapper(model, num_augmentations=5)
    
    # Inference Loop
    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            
            if use_tta:
                # Use TTA
                probs = tta_model(images)
                preds = torch.argmax(probs, dim=1)
            else:
                # Standard inference
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
    
    # ============================================================================
    # NaN Detection and Handling
    # ============================================================================
    if np.isnan(y_prob).any():
        logger.warning("⚠️  NaN detected in probability outputs!")
        logger.warning("   This usually indicates model collapse to single-class prediction.")
        logger.warning("   Replacing NaN with uniform distribution for metric calculation.")
        n_classes = len(classes)
        y_prob = np.nan_to_num(y_prob, nan=1.0/n_classes)
    
    # Verify probabilities sum to 1.0 (within tolerance)
    prob_sums = y_prob.sum(axis=1)
    if not np.allclose(prob_sums, 1.0, atol=1e-3):
        logger.warning("⚠️  Probability distributions don't sum to 1.0!")
        logger.warning(f"   Sum range: [{prob_sums.min():.4f}, {prob_sums.max():.4f}]")
        # Normalize to ensure sum=1
        y_prob = y_prob / prob_sums[:, np.newaxis]
        logger.warning("   Probabilities normalized to sum=1.0")
    
    # ============================================================================
    # Calculate Metrics
    # ============================================================================
    accuracy = accuracy_score(y_true, y_pred) * 100
    
    # ROC AUC with improved error handling
    try:
        if len(classes) == 2:
            roc_auc = roc_auc_score(y_true, y_prob[:, 1])
        else:
            roc_auc = roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')
    except ValueError as e:
        logger.warning(f"ROC AUC calculation failed: {e}")
        roc_auc = 0.0
    except Exception as e:
        logger.warning(f"Unexpected error in ROC AUC calculation: {e}")
        roc_auc = 0.0
        
    kappa = cohen_kappa_score(y_true, y_pred)
    corrcoef = matthews_corrcoef(y_true, y_pred)
    jaccard = jaccard_score(y_true, y_pred, average="weighted")
    
    # ============================================================================
    # Generate Report
    # ============================================================================
    report_path = os.path.join(REPORTS_DIR, f"{experiment_name}.txt")
    
    with open(report_path, "w") as f:
        f.write(f"===== COMPREHENSIVE ANALYSIS REPORT: {experiment_name} =====\n\n")
        f.write(f"Test-Time Augmentation (TTA): {'Enabled' if use_tta else 'Disabled'}\n\n")
        f.write("--- Overall Performance ---\n")
        f.write(f"Overall Accuracy: {accuracy:.2f}%\n")
        f.write(f"Macro ROC AUC:    {roc_auc:.4f}\n")
        f.write(f"Cohen's Kappa:    {kappa:.4f}\n")
        f.write(f"Matthews Correlation Coefficient (MCC):    {corrcoef:.4f}\n")
        f.write(f"Jaccard Score:    {jaccard:.4f}\n\n")
        
        f.write("--- Detailed Per-Class Metrics ---\n")
        f.write(classification_report(y_true, y_pred, target_names=classes, digits=4))
        
        f.write("\n--- Per-Class Specificity & Confusion Matrix Stats ---\n")
        cm = confusion_matrix(y_true, y_pred)
        
        # Calculate Specificity per class
        for i, class_name in enumerate(classes):
            tp = cm[i, i]
            fp = cm[:, i].sum() - tp
            fn = cm[i, :].sum() - tp
            tn = cm.sum() - (tp + fp + fn)
            
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            f.write(f"{class_name:<20}: Specificity: {specificity:.4f}, "
                   f"TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn}\n")
            
    logger.info(f"✅ Report saved to: {report_path}")

    # ============================================================================
    # Generate Visualizations
    # ============================================================================
    visualizer = visualizer or Visualizer(
        experiment_name=experiment_name, 
        model_name=model_name, 
        class_names=classes, 
        logger=logger
    )
    visualizer.generate_all_plots(
        y_true=y_true, 
        y_prob=y_prob, 
        history=history, 
        model=model, 
        test_loader=loader, 
        xai_samples=NUM_SAMPLES_TO_ANALYSE
    )

    metrics_dict = {
        'accuracy': accuracy,
        'roc_auc': roc_auc,
        'kappa': kappa,
        'corrcoef': corrcoef,
        'jaccard': jaccard,
        'y_true': y_true,
        'y_prob': y_prob,
        'y_pred': y_pred,  # Add predictions for ensemble
    }
    
    return metrics_dict
