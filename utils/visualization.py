"""
Visualization utilities for BRISC 2025 dataset
Author: Imtiaz Hossain (ID: 23101137)
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
from pathlib import Path
from typing import List, Tuple, Optional
import seaborn as sns
from matplotlib.gridspec import GridSpec

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import *


def visualize_sample(
    image: np.ndarray,
    mask: Optional[np.ndarray] = None,
    pred_mask: Optional[np.ndarray] = None,
    title: str = "",
    save_path: Optional[Path] = None
):
    """
    Visualize image, ground truth mask, and predicted mask
    
    Args:
        image: Input image (H, W) or (H, W, 3)
        mask: Ground truth mask (H, W)
        pred_mask: Predicted mask (H, W)
        title: Plot title
        save_path: Path to save figure
    """
    num_images = 1 + (mask is not None) + (pred_mask is not None)
    
    fig, axes = plt.subplots(1, num_images, figsize=(5 * num_images, 5))
    if num_images == 1:
        axes = [axes]
    
    idx = 0
    
    # Original image
    axes[idx].imshow(image, cmap='gray')
    axes[idx].set_title('Original Image')
    axes[idx].axis('off')
    idx += 1
    
    # Ground truth mask
    if mask is not None:
        axes[idx].imshow(mask, cmap='hot')
        axes[idx].set_title('Ground Truth Mask')
        axes[idx].axis('off')
        idx += 1
    
    # Predicted mask
    if pred_mask is not None:
        axes[idx].imshow(pred_mask, cmap='hot')
        axes[idx].set_title('Predicted Mask')
        axes[idx].axis('off')
    
    if title:
        fig.suptitle(title, fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    
    plt.show()


def visualize_batch(
    images: torch.Tensor,
    masks: torch.Tensor,
    pred_masks: Optional[torch.Tensor] = None,
    num_samples: int = 4,
    save_path: Optional[Path] = None
):
    """
    Visualize a batch of images with masks
    
    Args:
        images: Batch of images (B, 1, H, W)
        masks: Batch of masks (B, 1, H, W)
        pred_masks: Batch of predicted masks (B, 1, H, W)
        num_samples: Number of samples to visualize
        save_path: Path to save figure
    """
    batch_size = min(images.shape[0], num_samples)
    num_cols = 3 if pred_masks is not None else 2
    
    fig, axes = plt.subplots(batch_size, num_cols, figsize=(5 * num_cols, 5 * batch_size))
    
    if batch_size == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(batch_size):
        # Convert to numpy and squeeze channel dimension
        img = images[i, 0].cpu().numpy()
        mask = masks[i, 0].cpu().numpy()
        
        # Original image
        axes[i, 0].imshow(img, cmap='gray')
        axes[i, 0].set_title(f'Sample {i+1}: Image')
        axes[i, 0].axis('off')
        
        # Ground truth mask
        axes[i, 1].imshow(mask, cmap='hot')
        axes[i, 1].set_title(f'Sample {i+1}: GT Mask')
        axes[i, 1].axis('off')
        
        # Predicted mask (if available)
        if pred_masks is not None:
            pred = pred_masks[i, 0].cpu().numpy()
            axes[i, 2].imshow(pred, cmap='hot')
            axes[i, 2].set_title(f'Sample {i+1}: Pred Mask')
            axes[i, 2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight')
        print(f"Saved batch visualization to {save_path}")
    
    plt.show()


def plot_training_curves(
    history: dict,
    metrics: List[str] = ['loss'],
    save_path: Optional[Path] = None
):
    """
    Plot training and validation curves
    
    Args:
        history: Dictionary with training history
        metrics: List of metrics to plot
        save_path: Path to save figure
    """
    num_metrics = len(metrics)
    fig, axes = plt.subplots(1, num_metrics, figsize=(7 * num_metrics, 5))
    
    if num_metrics == 1:
        axes = [axes]
    
    for idx, metric in enumerate(metrics):
        train_key = f'train_{metric}'
        val_key = f'val_{metric}'
        
        if train_key in history:
            axes[idx].plot(history[train_key], label='Train', linewidth=2)
        if val_key in history:
            axes[idx].plot(history[val_key], label='Validation', linewidth=2)
        
        axes[idx].set_xlabel('Epoch', fontsize=12)
        axes[idx].set_ylabel(metric.replace('_', ' ').title(), fontsize=12)
        axes[idx].set_title(f'{metric.replace("_", " ").title()} over Epochs', fontsize=14, fontweight='bold')
        axes[idx].legend(fontsize=10)
        axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight')
        print(f"Saved training curves to {save_path}")
    
    plt.show()


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    save_path: Optional[Path] = None,
    normalize: bool = False
):
    """
    Plot confusion matrix
    
    Args:
        cm: Confusion matrix
        class_names: List of class names
        save_path: Path to save figure
        normalize: Whether to normalize the confusion matrix
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Count' if not normalize else 'Proportion'}
    )
    
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    plt.ylabel('True Label', fontsize=12, fontweight='bold')
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight')
        print(f"Saved confusion matrix to {save_path}")
    
    plt.show()


def plot_class_distribution(
    class_counts: dict,
    title: str = "Class Distribution",
    save_path: Optional[Path] = None
):
    """
    Plot class distribution bar chart
    
    Args:
        class_counts: Dictionary with class names as keys and counts as values
        title: Plot title
        save_path: Path to save figure
    """
    plt.figure(figsize=(10, 6))
    
    classes = list(class_counts.keys())
    counts = list(class_counts.values())
    
    bars = plt.bar(classes, counts, color=sns.color_palette("husl", len(classes)))
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.,
            height,
            f'{int(height)}',
            ha='center',
            va='bottom',
            fontsize=11,
            fontweight='bold'
        )
    
    plt.xlabel('Class', fontsize=12, fontweight='bold')
    plt.ylabel('Count', fontsize=12, fontweight='bold')
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight')
        print(f"Saved class distribution to {save_path}")
    
    plt.show()


def plot_metrics_comparison(
    results: dict,
    metric_name: str,
    save_path: Optional[Path] = None
):
    """
    Compare a metric across different models or experiments
    
    Args:
        results: Dictionary with experiment names as keys and metrics as values
        metric_name: Name of the metric to compare
        save_path: Path to save figure
    """
    plt.figure(figsize=(12, 6))
    
    experiments = list(results.keys())
    values = [results[exp][metric_name] for exp in experiments]
    
    bars = plt.bar(experiments, values, color=sns.color_palette("Set2", len(experiments)))
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.,
            height,
            f'{height:.4f}',
            ha='center',
            va='bottom',
            fontsize=10,
            fontweight='bold'
        )
    
    plt.xlabel('Experiment / Model', fontsize=12, fontweight='bold')
    plt.ylabel(metric_name.replace('_', ' ').title(), fontsize=12, fontweight='bold')
    plt.title(f'{metric_name.replace("_", " ").title()} Comparison', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight')
        print(f"Saved metrics comparison to {save_path}")
    
    plt.show()


def create_inference_visualization(
    image_path: Path,
    model,
    device: torch.device,
    save_path: Optional[Path] = None
):
    """
    Create visualization for inference on a single image
    This is the format required by the project guidelines
    
    Args:
        image_path: Path to input image
        model: Trained segmentation model
        device: Device to run inference on
        save_path: Path to save visualization
    """
    # Load image
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    original_shape = image.shape
    
    # Resize for model
    image_resized = cv2.resize(image, IMAGE_SIZE)
    image_normalized = image_resized.astype(np.float32) / 255.0
    
    # Convert to tensor
    image_tensor = torch.from_numpy(image_normalized).unsqueeze(0).unsqueeze(0).to(device)
    
    # Get prediction
    model.eval()
    with torch.no_grad():
        pred = model(image_tensor)
        pred_mask = torch.sigmoid(pred)
        pred_mask = (pred_mask > 0.5).float()
    
    # Convert back to numpy
    pred_mask_np = pred_mask.squeeze().cpu().numpy()
    
    # Resize prediction back to original size
    pred_mask_resized = cv2.resize(pred_mask_np, (original_shape[1], original_shape[0]))
    
    # Create visualization
    fig = plt.figure(figsize=(15, 5))
    
    # Original image
    plt.subplot(1, 3, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Original Image', fontsize=14, fontweight='bold')
    plt.axis('off')
    
    # Predicted mask
    plt.subplot(1, 3, 2)
    plt.imshow(pred_mask_resized, cmap='hot')
    plt.title('Predicted Segmentation Mask', fontsize=14, fontweight='bold')
    plt.axis('off')
    
    # Overlay
    plt.subplot(1, 3, 3)
    plt.imshow(image, cmap='gray')
    plt.imshow(pred_mask_resized, cmap='hot', alpha=0.5)
    plt.title('Overlay', fontsize=14, fontweight='bold')
    plt.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight')
        print(f"Saved inference visualization to {save_path}")
    
    plt.show()
    
    return pred_mask_resized