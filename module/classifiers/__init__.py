"""
Classifiers Package

Complete classifier library for deep learning research.

Available Classifiers:
    BASIC:
    - SimpleClassifier: Basic timm model wrapper (DEFAULT)
    - ProgressiveClassifier: Your sophisticated progressive fine-tuning
    
    ADVANCED:
    - BaselineClassifier: Standard CrossEntropy
    - EvidentialClassifier: Uncertainty quantification
    - MetricLearningClassifier: Prototypes + Triplet + Center Loss
    - RegularizedClassifier: Manifold Mixup + Label Smoothing
    - AttentionEnhancedClassifier: SE Blocks + Cosine Classifier
    - ProgressiveEvidentialClassifier: Progressive + Evidential
    - ClinicalGradeClassifier: Clinical deployment (5 techniques)
    - HybridTransformerClassifier: CNN + Transformer hybrid
    - UltimateRecallOptimizedClassifier: All 10 techniques

Usage:
    from classifiers import SimpleClassifier, ProgressiveClassifier
    
    # Default (simple)
    clf = SimpleClassifier('resnet18', num_classes=4)
    
    # Progressive
    clf = ProgressiveClassifier('resnet18', num_classes=4)
    
    # Advanced
    from classifiers import ClinicalGradeClassifier
    clf = ClinicalGradeClassifier('resnet18', num_classes=4)
    
    # All have same interface
    clf.fit(train_loader, val_loader, primary_metric='recall')
    results = clf.evaluate(test_loader)
"""

from .base_classifier import BaseClassifier

# Basic classifiers
from .progressive_classifier import ProgressiveClassifier

# Advanced classifiers
from .advanced_classifiers import (
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

# Testing framework
from .testing import (
    test_all_classifiers_on_model,
    test_single_classifier,
    compare_classifiers,
    list_available_classifiers
)

__all__ = [
    # Base
    'BaseClassifier',
    
    # Advanced
    'BaselineClassifier',
    'EvidentialClassifier',
    'MetricLearningClassifier',
    'RegularizedClassifier',
    'AttentionEnhancedClassifier',
    'ProgressiveClassifier',
    'ProgressiveEvidentialClassifier',
    'ClinicalGradeClassifier',
    'HybridTransformerClassifier',
    'UltimateRecallOptimizedClassifier',
    
    # Testing
    'test_all_classifiers_on_model',
    'test_single_classifier',
    'compare_classifiers',
    'list_available_classifiers',
]

# Classifier registry for string-based access
CLASSIFIER_REGISTRY = {
    'baseline': BaselineClassifier,
    'evidential': EvidentialClassifier,
    'metric_learning': MetricLearningClassifier,
    'regularized': RegularizedClassifier,
    'attention_enhanced': AttentionEnhancedClassifier,
    'progressive': ProgressiveClassifier,
    'progressive_evidential': ProgressiveEvidentialClassifier,
    'clinical_grade': ClinicalGradeClassifier,
    'hybrid_transformer': HybridTransformerClassifier,
    'ultimate': UltimateRecallOptimizedClassifier,
}


def get_classifier(classifier_type: str = 'simple') :
    """
    Get classifier class by name.
    
    Args:
        classifier_type: Name of classifier (default: 'simple')
            Options: 'simple', 'progressive', 'baseline', 'evidential',
                    'metric_learning', 'regularized', 'attention_enhanced',
                    'progressive_evidential', 'clinical_grade',
                    'hybrid_transformer', 'ultimate'
    
    Returns:
        Classifier class
    
    Raises:
        ValueError: If classifier type not found
    
    Examples:
        # Get default classifier
        clf_class = get_classifier()  # Returns SimpleClassifier
        
        # Get specific classifier
        clf_class = get_classifier('progressive')
        clf_class = get_classifier('clinical_grade')
    """
    classifier_type = classifier_type.lower().replace(' ', '_')
    
    if classifier_type not in CLASSIFIER_REGISTRY:
        available = ', '.join(CLASSIFIER_REGISTRY.keys())
        raise ValueError(
            f"Unknown classifier type: '{classifier_type}'. "
            f"Available: {available}"
        )
    
    return CLASSIFIER_REGISTRY[classifier_type]


def list_classifiers(display : bool = False):
    """List all available classifiers with categories."""
    print("\n" + "="*80)
    print("AVAILABLE CLASSIFIERS")
    print("="*80)
    
    print("\nADVANCED:")
    print("  baseline            - Standard CrossEntropy")
    print("  evidential          - Uncertainty quantification")
    print("  metric_learning     - Prototypes + Triplet + Center Loss")
    print("  regularized         - Manifold Mixup + Label Smoothing")
    print("  attention_enhanced  - SE Blocks + Cosine Classifier")
    print("  progressive         - Sophisticated 3-phase fine-tuning")
    print("  progressive_evidential - Progressive + Evidential")
    print("  clinical_grade      - Clinical deployment (5 techniques) ⭐")
    print("  hybrid_transformer  - CNN + Transformer hybrid")
    print("  ultimate            - All 10 techniques (maximum recall) ⭐⭐⭐")
    
    print("\n" + "="*80 + "\n")
    
    return list(CLASSIFIER_REGISTRY.keys())
