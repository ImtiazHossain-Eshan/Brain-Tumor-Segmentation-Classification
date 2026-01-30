"""
Complete Test Dataset Evaluation for Segmentation Models
Evaluates trained U-Net and Attention U-Net models on the test dataset
Author: Imtiaz Hossain (ID: 23101137)
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm
import time
from datetime import datetime

import sys
sys.path.append(str(Path(__file__).parent))

from config import *
from models import UNet, AttentionUNet
from utils.data_loader import BRISCSegmentationDataset, get_val_transforms
from utils.metrics import DiceBCELoss, SegmentationMetrics


def evaluate_model(model, test_loader, criterion, device, model_name):
    """
    Evaluate a model on the test dataset
    
    Args:
        model: Trained model to evaluate
        test_loader: DataLoader for test dataset
        criterion: Loss function
        device: Device to run evaluation on
        model_name: Name of the model for logging
        
    Returns:
        Dictionary containing test metrics and detailed results
    """
    print(f"\n{'='*80}")
    print(f"Evaluating {model_name} on Test Dataset")
    print(f"{'='*80}")
    
    model.eval()
    metrics = SegmentationMetrics()
    total_loss = 0
    sample_losses = []
    
    start_time = time.time()
    
    with torch.no_grad():
        pbar = tqdm(test_loader, desc=f'Testing {model_name}')
        for batch_idx, (images, masks) in enumerate(pbar):
            images = images.to(device)
            masks = masks.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            # Update metrics
            total_loss += loss.item()
            sample_losses.append(loss.item())
            metrics.update(outputs, masks)
            
            # Update progress bar
            current_metrics = metrics.get_metrics()
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'dice': f'{current_metrics["dice_coefficient"]:.4f}'
            })
    
    eval_time = time.time() - start_time
    
    # Calculate final metrics
    avg_loss = total_loss / len(test_loader)
    final_metrics = metrics.get_metrics()
    
    # Print detailed results
    print(f"\n{'-'*80}")
    print(f"Test Evaluation Results for {model_name}")
    print(f"{'-'*80}")
    print(f"Total test samples: {len(test_loader.dataset)}")
    print(f"Evaluation time: {eval_time:.2f} seconds")
    print(f"\nTest Loss: {avg_loss:.6f}")
    print(f"\nSegmentation Metrics:")
    print(f"  Dice Coefficient: {final_metrics['dice_coefficient']:.6f}")
    print(f"  mIoU (Mean IoU):  {final_metrics['mIoU']:.6f}")
    print(f"  Pixel Accuracy:   {final_metrics['pixel_accuracy']:.6f}")
    
    # Create comprehensive results dictionary
    results = {
        'model_name': model_name,
        'evaluation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'dataset_info': {
            'total_samples': len(test_loader.dataset),
            'batch_size': test_loader.batch_size,
            'image_size': IMAGE_SIZE
        },
        'test_metrics': {
            'loss': float(avg_loss),
            'dice_coefficient': float(final_metrics['dice_coefficient']),
            'mIoU': float(final_metrics['mIoU']),
            'pixel_accuracy': float(final_metrics['pixel_accuracy'])
        },
        'statistics': {
            'mean_loss': float(avg_loss),
            'min_loss': float(min(sample_losses)),
            'max_loss': float(max(sample_losses)),
            'std_loss': float(np.std(sample_losses))
        },
        'evaluation_time_seconds': float(eval_time)
    }
    
    return results


def test_unet(device=None):
    """Test U-Net model on test dataset"""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Using device: {device}")
    
    # Load test dataset
    test_dataset = BRISCSegmentationDataset(
        images_dir=SEGMENTATION_TEST / 'images',
        masks_dir=SEGMENTATION_TEST / 'masks',
        transform=get_val_transforms()
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=4, 
        pin_memory=True
    )
    
    # Create model
    model = UNet(in_channels=1, out_channels=1, base_filters=64)
    model = model.to(device)
    
    # Load trained weights
    checkpoint_path = MODELS_DIR / 'unet_best.pth'
    print(f"\nLoading model from: {checkpoint_path}")
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    print("✓ Model loaded successfully")
    
    # Define loss function
    criterion = DiceBCELoss()
    
    # Evaluate
    results = evaluate_model(model, test_loader, criterion, device, 'U-Net')
    
    # Save results
    results_path = RESULTS_DIR / 'unet_test_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"\n✓ Results saved to: {results_path}")
    
    return results


def test_attention_unet(device=None):
    """Test Attention U-Net model on test dataset"""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Using device: {device}")
    
    # Load test dataset
    test_dataset = BRISCSegmentationDataset(
        images_dir=SEGMENTATION_TEST / 'images',
        masks_dir=SEGMENTATION_TEST / 'masks',
        transform=get_val_transforms()
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=4, 
        pin_memory=True
    )
    
    # Create model
    model = AttentionUNet(in_channels=1, out_channels=1, base_filters=64)
    model = model.to(device)
    
    # Load trained weights
    checkpoint_path = MODELS_DIR / 'attention_unet_best.pth'
    print(f"\nLoading model from: {checkpoint_path}")
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    print("✓ Model loaded successfully")
    
    # Define loss function
    criterion = DiceBCELoss()
    
    # Evaluate
    results = evaluate_model(model, test_loader, criterion, device, 'Attention U-Net')
    
    # Save results
    results_path = RESULTS_DIR / 'attention_unet_test_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"\n✓ Results saved to: {results_path}")
    
    return results


def compare_models(unet_results, attn_unet_results):
    """Compare test results of both models"""
    print("\n" + "="*80)
    print("MODEL COMPARISON - TEST DATASET RESULTS")
    print("="*80)
    
    print(f"\n{'Metric':<25} {'U-Net':<15} {'Attention U-Net':<15} {'Winner'}")
    print("-"*80)
    
    metrics_to_compare = [
        ('Dice Coefficient', 'dice_coefficient', True),
        ('mIoU', 'mIoU', True),
        ('Pixel Accuracy', 'pixel_accuracy', True),
        ('Loss', 'loss', False)
    ]
    
    comparison = {
        'unet_wins': 0,
        'attention_unet_wins': 0,
        'ties': 0
    }
    
    for metric_name, metric_key, higher_is_better in metrics_to_compare:
        unet_val = unet_results['test_metrics'][metric_key]
        attn_val = attn_unet_results['test_metrics'][metric_key]
        
        if higher_is_better:
            if unet_val > attn_val:
                winner = "U-Net ✓"
                comparison['unet_wins'] += 1
            elif attn_val > unet_val:
                winner = "Attention U-Net ✓"
                comparison['attention_unet_wins'] += 1
            else:
                winner = "Tie"
                comparison['ties'] += 1
        else:  # Lower is better (for loss)
            if unet_val < attn_val:
                winner = "U-Net ✓"
                comparison['unet_wins'] += 1
            elif attn_val < unet_val:
                winner = "Attention U-Net ✓"
                comparison['attention_unet_wins'] += 1
            else:
                winner = "Tie"
                comparison['ties'] += 1
        
        print(f"{metric_name:<25} {unet_val:<15.6f} {attn_val:<15.6f} {winner}")
    
    print("\n" + "-"*80)
    print(f"Overall: U-Net wins {comparison['unet_wins']} metrics, "
          f"Attention U-Net wins {comparison['attention_unet_wins']} metrics, "
          f"{comparison['ties']} ties")
    
    # Save comparison
    comparison_results = {
        'comparison_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'unet_results': unet_results['test_metrics'],
        'attention_unet_results': attn_unet_results['test_metrics'],
        'summary': comparison
    }
    
    comparison_path = RESULTS_DIR / 'test_comparison.json'
    with open(comparison_path, 'w') as f:
        json.dump(comparison_results, f, indent=4)
    print(f"\n✓ Comparison saved to: {comparison_path}")
    
    return comparison_results


if __name__ == "__main__":
    # Set random seeds for reproducibility
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    
    print("\n" + "="*80)
    print("COMPLETE TEST DATASET EVALUATION")
    print("Brain Tumor Segmentation - CSE428 BRISC 2025")
    print("="*80)
    
    # Test U-Net
    print("\n[1/2] Testing U-Net Model...")
    unet_results = test_unet()
    
    # Test Attention U-Net
    print("\n[2/2] Testing Attention U-Net Model...")
    attn_unet_results = test_attention_unet()
    
    # Compare models
    comparison = compare_models(unet_results, attn_unet_results)
    
    print("\n" + "="*80)
    print("✓ TEST EVALUATION COMPLETED SUCCESSFULLY")
    print("="*80)
    print("\nResults saved in:")
    print(f"  - {RESULTS_DIR / 'unet_test_results.json'}")
    print(f"  - {RESULTS_DIR / 'attention_unet_test_results.json'}")
    print(f"  - {RESULTS_DIR / 'test_comparison.json'}")
    print("\n")
