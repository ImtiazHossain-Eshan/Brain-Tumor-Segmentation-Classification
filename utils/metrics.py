"""
Evaluation metrics for segmentation and classification tasks
Author: Imtiaz Hossain (ID: 23101137)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, confusion_matrix, classification_report
)
from typing import Tuple, Dict


# ============================================================================
# SEGMENTATION METRICS
# ============================================================================

def dice_coefficient(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6) -> torch.Tensor:
    """
    Calculate Dice coefficient (F1 score for segmentation)
    
    Args:
        pred: Predicted masks (B, 1, H, W)
        target: Ground truth masks (B, 1, H, W)
        smooth: Smoothing factor to avoid division by zero
    
    Returns:
        Dice coefficient
    """
    pred = pred.contiguous().view(-1)
    target = target.contiguous().view(-1)
    
    intersection = (pred * target).sum()
    dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    
    return dice


def iou_score(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6) -> torch.Tensor:
    """
    Calculate Intersection over Union (IoU) / Jaccard Index
    
    Args:
        pred: Predicted masks (B, 1, H, W)
        target: Ground truth masks (B, 1, H, W)
        smooth: Smoothing factor
    
    Returns:
        IoU score
    """
    pred = pred.contiguous().view(-1)
    target = target.contiguous().view(-1)
    
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    
    iou = (intersection + smooth) / (union + smooth)
    
    return iou


def pixel_accuracy(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Calculate pixel-wise accuracy
    
    Args:
        pred: Predicted masks (B, 1, H, W)
        target: Ground truth masks (B, 1, H, W)
    
    Returns:
        Pixel accuracy
    """
    pred = (pred > 0.5).float()
    target = (target > 0.5).float()
    
    correct = (pred == target).float().sum()
    total = torch.numel(pred) # Total number of pixels
    
    return correct / total


class DiceLoss(nn.Module):
    """Dice Loss for binary segmentation"""
    
    def __init__(self, smooth: float = 1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return 1 - dice_coefficient(pred, target, self.smooth)


class DiceBCELoss(nn.Module):
    """Combined Dice and BCE loss for better convergence"""
    
    def __init__(self, dice_weight: float = 0.5, bce_weight: float = 0.5):
        super(DiceBCELoss, self).__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.dice_loss = DiceLoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Apply sigmoid for dice loss
        pred_sigmoid = torch.sigmoid(pred)
        
        dice = self.dice_loss(pred_sigmoid, target)
        bce = self.bce_loss(pred, target)
        
        return self.dice_weight * dice + self.bce_weight * bce


class SegmentationMetrics:
    """Class to track and compute segmentation metrics"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all metrics"""
        self.dice_scores = []
        self.iou_scores = []
        self.pixel_accs = []
    
    def update(self, pred: torch.Tensor, target: torch.Tensor):
        """
        Update metrics with a batch
        
        Args:
            pred: Predictions (B, 1, H, W)
            target: Ground truth (B, 1, H, W)
        """
        # Apply sigmoid and threshold
        pred = torch.sigmoid(pred)
        pred = (pred > 0.5).float()
        target = (target > 0.5).float()
        
        # Calculate metrics
        dice = dice_coefficient(pred, target).item()
        iou = iou_score(pred, target).item()
        pixel_acc = pixel_accuracy(pred, target).item()
        
        self.dice_scores.append(dice)
        self.iou_scores.append(iou)
        self.pixel_accs.append(pixel_acc)
    
    def get_metrics(self) -> Dict[str, float]:
        """Get average metrics"""
        return {
            'dice_coefficient': np.mean(self.dice_scores) if self.dice_scores else 0.0,
            'mIoU': np.mean(self.iou_scores) if self.iou_scores else 0.0,
            'pixel_accuracy': np.mean(self.pixel_accs) if self.pixel_accs else 0.0
        }


# ============================================================================
# CLASSIFICATION METRICS
# ============================================================================

class ClassificationMetrics:
    """Class to track and compute classification metrics"""
    
    def __init__(self, num_classes: int = 4):
        self.num_classes = num_classes
        self.reset()
    
    def reset(self):
        """Reset all metrics"""
        self.all_preds = []
        self.all_targets = []
    
    def update(self, pred: torch.Tensor, target: torch.Tensor):
        """
        Update metrics with a batch
        
        Args:
            pred: Predictions (B, num_classes) - logits or probabilities
            target: Ground truth labels (B,)
        """
        # Get predicted classes
        if pred.dim() > 1:
            pred_classes = torch.argmax(pred, dim=1)
        else:
            pred_classes = pred
        
        self.all_preds.extend(pred_classes.cpu().numpy())
        self.all_targets.extend(target.cpu().numpy())
    
    def get_metrics(self) -> Dict[str, float]:
        """Calculate and return all classification metrics"""
        if not self.all_preds:
            return {
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0
            }
        
        preds = np.array(self.all_preds)
        targets = np.array(self.all_targets)
        
        metrics = {
            'accuracy': accuracy_score(targets, preds),
            'precision': precision_score(targets, preds, average='weighted', zero_division=0),
            'recall': recall_score(targets, preds, average='weighted', zero_division=0),
            'f1_score': f1_score(targets, preds, average='weighted', zero_division=0)
        }
        
        return metrics
    
    def get_confusion_matrix(self) -> np.ndarray:
        """Get confusion matrix"""
        if not self.all_preds:
            return np.zeros((self.num_classes, self.num_classes))
        
        return confusion_matrix(self.all_targets, self.all_preds)
    
    def get_classification_report(self, class_names: list = None) -> str:
        """Get detailed classification report"""
        if not self.all_preds:
            return "No predictions available"
        
        return classification_report(
            self.all_targets,
            self.all_preds,
            target_names=class_names,
            zero_division=0
        )


# ============================================================================
# MULTI-TASK METRICS (for joint training)
# ============================================================================

class MultiTaskMetrics:
    """Metrics for multi-task learning (segmentation + classification)"""
    
    def __init__(self, num_classes: int = 4):
        self.seg_metrics = SegmentationMetrics()
        self.cls_metrics = ClassificationMetrics(num_classes)
    
    def reset(self):
        """Reset all metrics"""
        self.seg_metrics.reset()
        self.cls_metrics.reset()
    
    def update(
        self,
        seg_pred: torch.Tensor,
        seg_target: torch.Tensor,
        cls_pred: torch.Tensor,
        cls_target: torch.Tensor
    ):
        """Update both segmentation and classification metrics"""
        self.seg_metrics.update(seg_pred, seg_target)
        self.cls_metrics.update(cls_pred, cls_target)
    
    def get_metrics(self) -> Dict[str, float]:
        """Get combined metrics"""
        seg_metrics = self.seg_metrics.get_metrics()
        cls_metrics = self.cls_metrics.get_metrics()
        
        # Combine with prefixes
        combined = {}
        for key, val in seg_metrics.items():
            combined[f'seg_{key}'] = val
        for key, val in cls_metrics.items():
            combined[f'cls_{key}'] = val
        
        return combined