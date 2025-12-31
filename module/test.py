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
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from .config import DEVICE, REPORTS_DIR, NUM_SAMPLES_TO_ANALYSE
from .visualization import Visualizer
from .utils import Logger

def test_model(model_name, model, loader, classes, experiment_name, history, logger: Logger, visualizer: Visualizer = None):
    """
    Comprehensive model evaluation on test set
    
    Generates:
    - Detailed performance metrics (accuracy, AUC, kappa, MCC, Jaccard)
    - Per-class analysis (precision, recall, F1, specificity)
    - Confusion matrix
    - Visualization plots (ROC curves, PR curves, etc.)
    - XAI analysis on sample images
    
    Args:
        model_name: Architecture name
        model: Trained model instance
        loader: Test data loader
        classes: List of class names
        experiment_name: Unique identifier for this experiment
        history: Training history (list of dicts with epoch metrics)
        logger: Logger instance
        visualizer: Visualizer instance for plotting
    
    Returns:
        dict: Comprehensive metrics including y_true and y_prob for further analysis
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    step = len(loader) / 20
    curr = step
    
    logger.info(f"--- Starting Evaluation: {experiment_name} ---")
    print("\t\tProcessing [", end = "")
    
    # Inference loop with proper memory management
    with torch.inference_mode():
        for images, labels in loader :
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)
            
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=(DEVICE == 'cuda')):
                outputs = model(images)
            
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)
            
            # Transfer to CPU after detaching
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            if len(all_labels) >= curr :
                print("#", end = "")
                curr += step
        print("]")
    
    # Convert to numpy arrays
    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)
    
    # Check for NaN in probabilities and handle gracefully
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
    
    # Calculate Metrics
    accuracy = accuracy_score(y_true, y_pred) * 100
    
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
    
    # Generate comprehensive text report
    report_path = os.path.join(REPORTS_DIR, f"{experiment_name}.txt")
    
    with open(report_path, "w") as f:
        f.write(f"===== COMPREHENSIVE ANALYSIS REPORT: {experiment_name} =====\n\n")
        f.write("--- Overall Performance ---\n")
        f.write(f"Overall Accuracy: {accuracy:.2f}%\n")
        f.write(f"Macro ROC AUC: {roc_auc:.4f}\n")
        f.write(f"Cohen's Kappa: {kappa:.4f}\n")
        f.write(f"Matthews Correlation Coefficient (MCC): {corrcoef:.4f}\n")
        f.write(f"Jaccard Score: {jaccard:.4f}\n\n")
        
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
            f.write(f"{class_name:<20}: Specificity: {specificity:.4f}, TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn}\n")
    
    logger.info(f"✅ Report saved to: {report_path}")

    # Generate all visualization plots
    if visualizer is None:
        visualizer = Visualizer(
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
        'y_prob': y_prob
    }
    
    return metrics_dict