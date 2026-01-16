"""
Testing Framework for Advanced Classifiers
"""

import torch
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import time
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

from . import (
    ProgressiveClassifier,
    BaselineClassifier,
    EvidentialClassifier,
    MetricLearningClassifier,
    RegularizedClassifier,
    AttentionEnhancedClassifier,
    ProgressiveEvidentialClassifier,
    ClinicalGradeClassifier,
    HybridTransformerClassifier,
    UltimateRecallOptimizedClassifier
)


# Classifier registry
CLASSIFIER_MAP = {
    'baseline': BaselineClassifier,
    'progressive': ProgressiveClassifier,
    'evidential': EvidentialClassifier,
    'metric_learning': MetricLearningClassifier,
    'regularized': RegularizedClassifier,
    'attention_enhanced': AttentionEnhancedClassifier,
    'progressive_evidential': ProgressiveEvidentialClassifier,
    'clinical_grade': ClinicalGradeClassifier,
    'hybrid_transformer': HybridTransformerClassifier,
    'ultimate': UltimateRecallOptimizedClassifier,
}


def test_all_classifiers_on_model(
    model_name: str,
    train_loader,
    val_loader,
    test_loader,
    classifiers: List[str] = 'all',
    class_names: Optional[List[str]] = None,
    epochs: int = 30,
    lr: float = 1e-4,
    primary_metric: str = 'recall',
    device: str = 'cuda',
    save_dir: str = './classifier_comparison'
):
    """
    Test multiple classifiers on ONE model (Option C).
    
    Args:
        model_name: Base model (e.g., 'resnet18', 'efficientnet_b4')
        train_loader: Training data
        val_loader: Validation data
        test_loader: Test data
        classifiers: List of classifier names or 'all'
        class_names: Class names for reporting
        epochs: Training epochs per classifier
        lr: Learning rate
        primary_metric: Metric to optimize ('recall', 'accuracy', 'f1', 'precision')
        device: 'cuda' or 'cpu'
        save_dir: Directory to save results
    
    Returns:
        results_df: DataFrame with all results, ranked by primary_metric
    
    Example:
        # Test specific classifiers
        results = test_all_classifiers_on_model(
            'resnet18',
            train_loader, val_loader, test_loader,
            classifiers=['baseline', 'evidential', 'clinical_grade', 'ultimate']
        )
        
        # Test ALL classifiers
        results = test_all_classifiers_on_model(
            'resnet18',
            train_loader, val_loader, test_loader,
            classifiers='all'
        )
    """
    
    # Create save directory
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    # Default class names
    if class_names is None:
        class_names = ['NonDemented', 'VeryMildDemented', 'MildDemented', 'ModerateDemented']
    
    # Handle 'all' option
    if classifiers == 'all':
        classifiers_to_test = list(CLASSIFIER_MAP.keys())
    else:
        classifiers_to_test = [c.lower().replace(' ', '_') for c in classifiers]
    
    # Validate classifier names
    invalid = [c for c in classifiers_to_test if c not in CLASSIFIER_MAP]
    if invalid:
        available = ', '.join(CLASSIFIER_MAP.keys())
        raise ValueError(f"Invalid classifiers: {invalid}. Available: {available}")
    
    # Results storage
    results = []
    
    print("\n" + "="*100)
    print(f"TESTING MULTIPLE CLASSIFIERS ON: {model_name.upper()}")
    print("="*100)
    print(f"Primary Metric: {primary_metric.upper()}")
    print(f"Testing {len(classifiers_to_test)} classifiers")
    print(f"Device: {device}")
    print("="*100 + "\n")
    
    # Test each classifier
    for clf_name in classifiers_to_test:
        print("\n" + "█"*100)
        print(f"█{'':^98}█")
        print(f"█{f'TESTING: {clf_name.upper()} on {model_name}':^98}█")
        print(f"█{'':^98}█")
        print("█"*100 + "\n")
        
        try:
            # Create classifier
            start_time = time.time()
            
            clf_class = CLASSIFIER_MAP[clf_name]
            classifier = clf_class(
                model_name=model_name,
                num_classes=len(class_names),
                device=device
            )
            
            # Train
            print(f"\n{'─'*100}")
            print("TRAINING PHASE")
            print(f"{'─'*100}\n")
            
            # Determine if should use SAM
            use_sam = clf_name in ['clinical_grade', 'ultimate']
            
            history = classifier.fit(
                train_loader=train_loader,
                val_loader=val_loader,
                epochs=epochs,
                lr=lr,
                use_sam=use_sam,
                primary_metric=primary_metric,
                patience=10,
                min_delta=0.001 if clf_name != 'ultimate' else 0.0005
            )
            
            training_time = time.time() - start_time
            
            # Test
            print(f"\n{'─'*100}")
            print("TESTING PHASE")
            print(f"{'─'*100}\n")
            
            test_results = classifier.evaluate(test_loader, class_names=class_names)
            
            # Store results
            result_entry = {
                'Classifier': clf_name,
                'Model': model_name,
                f'Test_{primary_metric.capitalize()}': test_results.get(primary_metric, 0.0),
                'Test_Recall': test_results['recall'],
                'Test_Accuracy': test_results['accuracy'],
                'Test_Precision': test_results['precision'],
                'Test_F1': test_results['f1'],
                f'Best_Val_{primary_metric.capitalize()}': classifier.best_metric_value,
                'Best_Val_Recall': classifier.best_recall,
                'Best_Val_Accuracy': classifier.best_acc,
                'Training_Time_Min': training_time / 60,
                'Total_Epochs': len(history) if isinstance(history, list) else epochs,
            }
            
            # Per-class recall
            for i, class_name in enumerate(class_names):
                result_entry[f'{class_name}_Recall'] = test_results['per_class_recall'][i]
            
            results.append(result_entry)
            
            # Save model
            save_path = f"{save_dir}/{clf_name}_{model_name}.pth"
            classifier.save(save_path)
            print(f"\n✓ Model saved to: {save_path}")
            
            # Success message
            print(f"\n{'╔'*100}")
            print(f"✓ {clf_name.upper()} COMPLETE")
            print(f"  Test {primary_metric.capitalize()}: {test_results[primary_metric if primary_metric in test_results else 'recall']:.4f} ★")
            print(f"  Test Accuracy: {test_results['accuracy']:.2f}%")
            print(f"  Training Time: {training_time/60:.1f} minutes")
            print(f"{'╚'*100}\n")
            
        except Exception as e:
            print(f"\n✗ ERROR testing {clf_name}: {str(e)}")
            import traceback
            traceback.print_exc()
            
            # Record failure
            results.append({
                'Classifier': clf_name,
                'Model': model_name,
                f'Test_{primary_metric.capitalize()}': 0.0,
                'Test_Recall': 0.0,
                'Test_Accuracy': 0.0,
                'Test_Precision': 0.0,
                'Test_F1': 0.0,
                f'Best_Val_{primary_metric.capitalize()}': 0.0,
                'Best_Val_Recall': 0.0,
                'Best_Val_Accuracy': 0.0,
                'Training_Time_Min': 0.0,
                'Total_Epochs': 0,
                'Error': str(e)
            })

        finally :
            # Cleanup
            del classifier
            torch.cuda.empty_cache()
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Sort by primary metric
    sort_col = f'Test_{primary_metric.capitalize()}'
    if sort_col not in results_df.columns:
        sort_col = 'Test_Recall'
    
    results_df = results_df.sort_values(sort_col, ascending=False).reset_index(drop=True)
    
    # Add rank
    results_df.insert(0, 'Rank', range(1, len(results_df) + 1))
    
    # Save results
    results_path = f"{save_dir}/comparison_{model_name}.csv"
    results_df.to_csv(results_path, index=False)
    
    # Print summary
    print("\n" + "="*100)
    print(f"FINAL RESULTS SUMMARY - {model_name.upper()} (Ranked by {primary_metric.upper()})")
    print("="*100 + "\n")
    
    # Display table
    display_cols = ['Rank', 'Classifier', sort_col, 'Test_Accuracy', 'Test_F1', 'Training_Time_Min']
    print(results_df[display_cols].to_string(index=False))
    
    print(f"\n{'='*100}")
    print(f"🏆 WINNER: {results_df.iloc[0]['Classifier'].upper()}")
    print(f"   {primary_metric.capitalize()}: {results_df.iloc[0][sort_col]:.4f}")
    print(f"   Accuracy: {results_df.iloc[0]['Test_Accuracy']:.2f}%")
    print(f"{'='*100}")
    
    print(f"\n📊 Full results saved to: {results_path}\n")
    
    return results_df


def test_single_classifier(
    classifier_name: str,
    model_name: str,
    train_loader,
    val_loader,
    test_loader,
    class_names: Optional[List[str]] = None,
    epochs: int = 30,
    lr: float = 1e-4,
    primary_metric: str = 'recall',
    device: str = 'cuda'
):
    """
    Test a single classifier (quick testing).
    
    Args:
        classifier_name: Name of classifier (e.g., 'simple', 'progressive', 'clinical_grade')
        model_name: Base model (e.g., 'resnet18')
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        test_loader: Test DataLoader
        class_names: List of class names
        epochs: Training epochs
        lr: Learning rate
        primary_metric: Metric to optimize
        device: 'cuda' or 'cpu'
    
    Returns:
        test_results: Test metrics dict
        classifier: Trained classifier
    
    Example:
        results, model = test_single_classifier(
            'ultimate', 'resnet18',
            train_loader, val_loader, test_loader
        )
    """
    
    classifier_name = classifier_name.lower().replace(' ', '_')
    
    if classifier_name not in CLASSIFIER_MAP:
        available = ', '.join(CLASSIFIER_MAP.keys())
        raise ValueError(f"Unknown classifier: '{classifier_name}'. Available: {available}")
    
    if class_names is None:
        class_names = ['NonDemented', 'VeryMildDemented', 'MildDemented', 'ModerateDemented']
    
    print(f"\n{'='*80}")
    print(f"Testing: {classifier_name.upper()} on {model_name}")
    print(f"{'='*80}\n")
    
    # Create classifier
    clf_class = CLASSIFIER_MAP[classifier_name]
    classifier = clf_class(
        model_name=model_name,
        num_classes=len(class_names),
        device=device
    )
    
    # Train
    use_sam = classifier_name in ['clinical_grade', 'ultimate']
    
    history = classifier.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        lr=lr,
        use_sam=use_sam,
        primary_metric=primary_metric,
        patience=10
    )
    
    # Test
    test_results = classifier.evaluate(test_loader, class_names=class_names)
    
    return test_results, classifier


def compare_classifiers(results_df: pd.DataFrame, save_path: Optional[str] = None):
    """
    Plot comparison of classifiers.
    
    Args:
        results_df: DataFrame from test_all_classifiers_on_model()
        save_path: Where to save the plot (optional)
    """
    # Set style
    sns.set_style("whitegrid")
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Test Recall comparison
    ax = axes[0, 0]
    sns.barplot(data=results_df, x='Test_Recall', y='Classifier', ax=ax, palette='viridis')
    ax.set_title('Test Recall (PRIMARY METRIC)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Recall')
    ax.axvline(x=0.99, color='r', linestyle='--', label='Target (0.99)')
    ax.legend()
    
    # 2. Test Accuracy comparison
    ax = axes[0, 1]
    sns.barplot(data=results_df, x='Test_Accuracy', y='Classifier', ax=ax, palette='rocket')
    ax.set_title('Test Accuracy', fontsize=14, fontweight='bold')
    ax.set_xlabel('Accuracy (%)')
    
    # 3. Test F1 comparison
    ax = axes[1, 0]
    sns.barplot(data=results_df, x='Test_F1', y='Classifier', ax=ax, palette='mako')
    ax.set_title('Test F1 Score', fontsize=14, fontweight='bold')
    ax.set_xlabel('F1 Score')
    
    # 4. Training time comparison
    ax = axes[1, 1]
    sns.barplot(data=results_df, x='Training_Time_Min', y='Classifier', ax=ax, palette='flare')
    ax.set_title('Training Time', fontsize=14, fontweight='bold')
    ax.set_xlabel('Time (minutes)')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n📊 Comparison plot saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()


def list_available_classifiers():
    """List all available classifiers."""
    print("\n" + "="*80)
    print("AVAILABLE CLASSIFIERS (11 Total)")
    print("="*80)
    
    print("\nADVANCED (Research-Grade):")
    advanced = [
        ('baseline', 'Standard CrossEntropy'),
        ('evidential', 'Uncertainty quantification'),
        ('metric_learning', 'Prototypes + Triplet + Center Loss'),
        ('regularized', 'Manifold Mixup + Label Smoothing'),
        ('attention_enhanced', 'SE Blocks + Cosine Classifier'),
        ('progressive', 'Sophisticated 3-phase fine-tuning'),
        ('progressive_evidential', 'Progressive + Evidential'),
        ('clinical_grade', 'Clinical deployment (5 techniques + SAM) ⭐'),
        ('hybrid_transformer', 'CNN + Transformer hybrid'),
        ('ultimate', 'All 10 techniques (maximum recall) ⭐⭐⭐')
    ]
    
    for i, (name, desc) in enumerate(advanced, 3):
        print(f"  {i}. {name:<22} - {desc}")
    
    print("\n" + "="*80)
    print("Usage Examples:")
    print("  test_all_classifiers_on_model('resnet18', ..., classifiers='all')")
    print("  test_all_classifiers_on_model('resnet18', ..., classifiers=['simple', 'progressive'])")
    print("  test_single_classifier('ultimate', 'resnet18', ...)")
    print("="*80 + "\n")
    
    return list(CLASSIFIER_MAP.keys())


# Convenience function
def get_classifier_info(classifier_name: str):
    """
    Get information about a specific classifier.
    
    Args:
        classifier_name: Name of classifier
    
    Returns:
        dict: Classifier information
    
    Example:
        info = get_classifier_info('progressive')
        print(info['description'])
    """
    classifier_name = classifier_name.lower().replace(' ', '_')
    
    if classifier_name not in CLASSIFIER_MAP:
        available = ', '.join(CLASSIFIER_MAP.keys())
        raise ValueError(f"Unknown classifier: '{classifier_name}'. Available: {available}")
    
    info_map = {
        'baseline': {
            'name': 'BaselineClassifier',
            'category': 'Basic',
            'description': 'Basic timm wrapper with AdamW + OneCycleLR',
            'speed': 'Fast',
            'use_case': 'Simple baseline for comparisons'
        },
        'progressive': {
            'name': 'ProgressiveClassifier',
            'category': 'Basic',
            'description': 'Sophisticated 3-phase discriminative fine-tuning',
            'speed': 'Medium',
            'use_case': 'High-quality training with architecture awareness'
        },
        'evidential': {
            'name': 'EvidentialClassifier',
            'category': 'Advanced',
            'description': 'Evidential deep learning for uncertainty quantification',
            'speed': 'Medium',
            'use_case': 'When you need uncertainty scores'
        },
        'ultimate': {
            'name': 'UltimateRecallOptimizedClassifier',
            'category': 'Advanced',
            'description': 'All 10 research techniques combined',
            'speed': 'Slow',
            'use_case': 'Maximum recall for critical medical diagnosis'
        },
        # Add more as needed...
    }
    
    return info_map.get(classifier_name, {
        'name': CLASSIFIER_MAP[classifier_name].__name__,
        'category': 'Advanced',
        'description': 'Research-grade classifier',
        'speed': 'Medium',
        'use_case': 'Advanced training'
    })
