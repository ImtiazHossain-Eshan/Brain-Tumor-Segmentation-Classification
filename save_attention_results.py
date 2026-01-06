"""
Save Attention U-Net results and create visualizations
This script saves the training results since the main script errored on plotting
"""

import torch
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Load the trained model
models_dir = Path("models")
results_dir = Path("results")
figures_dir = Path("figures")

# Ensure directories exist
results_dir.mkdir(exist_ok=True)
figures_dir.mkdir(exist_ok=True)

print("=" * 80)
print("SAVING ATTENTION U-NET RESULTS")
print("=" * 80)

# Load checkpoint
checkpoint_path = models_dir / "attention_unet_best.pth"
if checkpoint_path.exists():
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    history = checkpoint['history']
    best_val_dice = checkpoint['best_val_dice']
    epoch = checkpoint['epoch']
    
    print(f"\nModel Information:")
    print(f"  Best Epoch: {epoch}")
    print(f"  Best Val Dice: {best_val_dice:.4f} ({best_val_dice*100:.2f}%)")
    
    # Save results JSON
    results = {
        'model': 'attention_unet',
        'best_epoch': epoch,
        'best_val_dice': float(best_val_dice),
        'final_metrics': {
            'train_dice': float(history['train_dice'][-1]),
            'val_dice': float(history['val_dice'][-1]),
            'train_loss': float(history['train_loss'][-1]),
            'val_loss': float(history['val_loss'][-1])
        },
        'history': {
            'train_loss': [float(x) for x in history['train_loss']],
            'val_loss': [float(x) for x in history['val_loss']],
            'train_dice': [float(x) for x in history['train_dice']],
            'val_dice': [float(x) for x in history['val_dice']],
            'train_iou': [float(x) for x in history['train_iou']],
            'val_iou': [float(x) for x in history['val_iou']],
            'train_pixel_acc': [float(x) for x in history['train_pixel_acc']],
            'val_pixel_acc': [float(x) for x in history['val_pixel_acc']]
        }
    }
    
    results_file = results_dir / 'attention_unet_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to: {results_file}")
    
    # Create training curves
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Attention U-Net Training History', fontsize=16, fontweight='bold')
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    axes[0, 0].set_xlabel('Epoch', fontsize=12)
    axes[0, 0].set_ylabel('Loss', fontsize=12)
    axes[0, 0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0, 0].legend(fontsize=11)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Dice Coefficient
    axes[0, 1].plot(epochs, history['train_dice'], 'b-', label='Train Dice', linewidth=2)
    axes[0, 1].plot(epochs, history['val_dice'], 'r-', label='Val Dice', linewidth=2)
    axes[0, 1].axhline(y=best_val_dice, color='g', linestyle='--', label=f'Best: {best_val_dice:.4f}', linewidth=2)
    axes[0, 1].set_xlabel('Epoch', fontsize=12)
    axes[0, 1].set_ylabel('Dice Coefficient', fontsize=12)
    axes[0, 1].set_title('Dice Coefficient', fontsize=14, fontweight='bold')
    axes[0, 1].legend(fontsize=11)
    axes[0, 1].grid(True, alpha=0.3)
    
    # mIoU
    axes[1, 0].plot(epochs, history['train_iou'], 'b-', label='Train mIoU', linewidth=2)
    axes[1, 0].plot(epochs, history['val_iou'], 'r-', label='Val mIoU', linewidth=2)
    axes[1, 0].set_xlabel('Epoch', fontsize=12)
    axes[1, 0].set_ylabel('mIoU', fontsize=12)
    axes[1, 0].set_title('Mean Intersection over Union', fontsize=14, fontweight='bold')
    axes[1, 0].legend(fontsize=11)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Pixel Accuracy
    axes[1, 1].plot(epochs, history['train_pixel_acc'], 'b-', label='Train Pixel Acc', linewidth=2)
    axes[1, 1].plot(epochs, history['val_pixel_acc'], 'r-', label='Val Pixel Acc', linewidth=2)
    axes[1, 1].set_xlabel('Epoch', fontsize=12)
    axes[1, 1].set_ylabel('Pixel Accuracy', fontsize=12)
    axes[1, 1].set_title('Pixel Accuracy', fontsize=14, fontweight='bold')
    axes[1, 1].legend(fontsize=11)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    curves_file = figures_dir / 'attention_unet_training_curves.png'
    plt.savefig(curves_file, dpi=150, bbox_inches='tight')
    print(f"✓ Training curves saved to: {curves_file}")
    
    plt.close()
    
    # Create comparison with U-Net
    print("\n" + "=" * 80)
    print("COMPARISON WITH VANILLA U-NET")
    print("=" * 80)
    
    unet_results_file = results_dir / 'unet_results.json'
    if unet_results_file.exists():
        with open(unet_results_file) as f:
            unet_results = json.load(f)
        
        print(f"\n{'Model':<20} {'Best Val Dice':<15} {'Improvement'}")
        print("-" * 55)
        
        unet_dice = 0.8310  # From our earlier training
        attn_dice = best_val_dice
        improvement = ((attn_dice - unet_dice) / unet_dice) * 100
        
        print(f"{'U-Net':<20} {unet_dice:.4f} ({unet_dice*100:.2f}%) {'-':<10}")
        print(f"{'Attention U-Net':<20} {attn_dice:.4f} ({attn_dice*100:.2f}%)  {improvement:+.2f}%")
        
        if attn_dice > unet_dice:
            print(f"\n✓ Attention U-Net performed BETTER by {abs(improvement):.2f}%")
        elif attn_dice < unet_dice:
            print(f"\n⚠ Attention U-Net performed WORSE by {abs(improvement):.2f}%")
        else:
            print(f"\n= Both models performed equally")
    
    print("\n" + "=" * 80)
    print("COMPLETE!")
    print("=" * 80)

else:
    print(f"\n✗ Model checkpoint not found at: {checkpoint_path}")
