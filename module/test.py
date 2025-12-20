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
from tqdm import tqdm
from .config import DEVICE, REPORTS_DIR, NUM_SAMPLES_TO_ANALYSE
from .visualization import Visualizer
from .utils import Logger

def test_model(model_name, model, loader, classes, experiment_name, history, logger : Logger, visualizer : Visualizer = None):
    """
    Performs a comprehensive evaluation of the model on the provided loader.
    Generates a text report and visualizations.
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    print(f"--- Starting Evaluation: {experiment_name} ---")
    
    # 1. Inference Loop
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Testing"):
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            
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
    
    # 2. Calculate Metrics
    accuracy = accuracy_score(y_true, y_pred) * 100
    try:
        # Handle binary vs multi-class AUC
        if len(classes) == 2:
            roc_auc = roc_auc_score(y_true, y_prob[:, 1])
        else:
            roc_auc = roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')
    except Exception:
        roc_auc = 0.0
        
    kappa = cohen_kappa_score(y_true, y_pred)
    corrcoef = matthews_corrcoef(y_true, y_pred)
    jaccard = jaccard_score(y_true, y_pred, average = "weighted")
    
    # 3. Generate Report
    report_path = os.path.join(REPORTS_DIR, f"{experiment_name}.txt")
    
    with open(report_path, "w") as f:
        f.write(f"===== COMPREHENSIVE ANALYSIS REPORT FOR: {experiment_name} =====\n\n")
        f.write("--- Overall Performance ---\n")
        f.write(f"Overall Accuracy: {accuracy:.2f}%\n")
        f.write(f"Macro ROC AUC:    {roc_auc:.4f}\n")
        f.write(f"Cohen's Kappa:    {kappa:.4f}\n\n")
        f.write(f"Matthews Correlation Coefficient (MCC):    {corrcoef:.2f}\n\n")
        f.write(f"Jaccard Score:    {jaccard:.4f}\n\n")
        
        f.write("--- Detailed Per-Class Metrics ---\n")
        f.write(classification_report(y_true, y_pred, target_names=classes, digits=4))
        
        f.write("\n--- Per-Class Specificity & Counts ---\n")
        cm = confusion_matrix(y_true, y_pred)
        
        # Calculate Specificity per class
        for i, class_name in enumerate(classes):
            tp = cm[i, i]
            fp = cm[:, i].sum() - tp
            fn = cm[i, :].sum() - tp
            tn = cm.sum() - (tp + fp + fn)
            
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            f.write(f"{class_name:<20}: Specificity: {specificity:.4f}, TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn}\n")
            
    print(f"✅ Report saved to: {report_path}")

    # 4. Generate Visualizations
    # Use the Visualizer class if provided, else run ad-hoc plots
    visualizer = visualizer or Visualizer(experiment_name = experiment_name, model_name = model_name, class_names = classes, logger = logger)
    visualizer.generate_all_plots(y_true = y_true, y_prob = y_prob, history = history, model = model, test_loader = loader, xai_samples = NUM_SAMPLES_TO_ANALYSE)

    metrics_dict = {
        'accuracy': accuracy,
        'roc_auc': roc_auc,
        'kappa': kappa,
        'corrcoef': corrcoef,
        'jaccard': jaccard,
        'y_true': y_true, # useful for master visualizer later
        'y_prob': y_prob
    }
    
    return metrics_dict