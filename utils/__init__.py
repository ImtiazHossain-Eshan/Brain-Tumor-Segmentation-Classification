"""
Utilities package for BRISC 2025 project
Author: Imtiaz Hossain (ID: 23101137)
"""

from .data_loader import (
    BRISCDatasetInfo,
    BRISCSegmentationDataset,
    BRISCClassificationDataset,
    BRISCJointDataset,
    get_train_transforms,
    get_val_transforms,
    create_data_loaders
)

from .metrics import (
    dice_coefficient,
    iou_score,
    pixel_accuracy,
    DiceLoss,
    DiceBCELoss,
    SegmentationMetrics,
    ClassificationMetrics,
    MultiTaskMetrics
)

from .visualization import (
    visualize_sample,
    visualize_batch,
    plot_training_curves,
    plot_confusion_matrix,
    plot_class_distribution,
    plot_metrics_comparison,
    create_inference_visualization
)

__all__ = [
    # Data loaders
    'BRISCDatasetInfo',
    'BRISCSegmentationDataset',
    'BRISCClassificationDataset',
    'BRISCJointDataset',
    'get_train_transforms',
    'get_val_transforms',
    'create_data_loaders',
    
    # Metrics
    'dice_coefficient',
    'iou_score',
    'pixel_accuracy',
    'DiceLoss',
    'DiceBCELoss',
    'SegmentationMetrics',
    'ClassificationMetrics',
    'MultiTaskMetrics',
    
    # Visualization
    'visualize_sample',
    'visualize_batch',
    'plot_training_curves',
    'plot_confusion_matrix',
    'plot_class_distribution',
    'plot_metrics_comparison',
    'create_inference_visualization'
]