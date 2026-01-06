"""
Bonus Task 3: Hyperparameter Optimization Study
Test different optimizers and learning rates on U-Net segmentation
"""

import sys
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from tqdm import tqdm
import json
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(str(Path(__file__).parent))

from config import *
from models import UNet
from utils.data_loader import create_data_loaders
from utils.metrics import DiceBCELoss, SegmentationMetrics

# Hyperparameter configurations to test
OPTIMIZERS = {
    'Adam': optim.Adam,
    'SGD': optim.SGD,
    'AdamW': optim.AdamW,
    'RMSprop': optim.RMSprop
}

LEARNING_RATES = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3]

# Reduced epochs for faster experimentation
EXPERIMENT_EPOCHS = 20

def train_model(model, optimizer, criterion, train_loader, val_loader, device, epochs=20):
    """Train model and return best validation Dice"""
    best_val_dice = 0.0
    history = []
    
    print("  Training progress:")
    for epoch in range(epochs):
        # Training
        model.train()
        train_metrics = SegmentationMetrics()
        train_loss = 0.0
        
        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            with torch.no_grad():
                preds = torch.sigmoid(outputs) > 0.5
                train_metrics.update(preds, masks)
        
        train_loss /= len(train_loader)
        train_dice = train_metrics.get_metrics()['dice_coefficient']
        
        # Validation
        model.eval()
        val_metrics = SegmentationMetrics()
        val_loss = 0.0
        
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()
                
                preds = torch.sigmoid(outputs) > 0.5
                val_metrics.update(preds, masks)
        
        val_loss /= len(val_loader)
        val_dice = val_metrics.get_metrics()['dice_coefficient']
        
        best_val_dice = max(best_val_dice, val_dice)
        
        history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_dice': train_dice,
            'val_dice': val_dice
        })
        
        # Print EVERY epoch
        print(f"    Epoch {epoch+1:2d}/{epochs}: Train Dice={train_dice:.4f}, Val Dice={val_dice:.4f}, Best={best_val_dice:.4f}")
    
    return best_val_dice, history

def run_hyperparameter_study():
    print("\n" + "=" * 80)
    print("BONUS TASK 3: HYPERPARAMETER OPTIMIZATION STUDY")
    print("=" * 80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    loaders = create_data_loaders(
        batch_size=BATCH_SIZE,
        val_split=VALIDATION_SPLIT,
        num_workers=2
    )
    
    train_loader = loaders['train']
    val_loader = loaders['val']
    
    print(f"\nDataset: {len(train_loader.dataset)} train, {len(val_loader.dataset)} val")
    print(f"Testing {len(OPTIMIZERS)} optimizers × {len(LEARNING_RATES)} learning rates")
    print(f"Total experiments: {len(OPTIMIZERS) * len(LEARNING_RATES)}")
    print(f"Epochs per experiment: {EXPERIMENT_EPOCHS}")
    
    # Store results
    results = []
    
    # Test each combination
    experiment_num = 0
    total_experiments = len(OPTIMIZERS) * len(LEARNING_RATES)
    
    print(f"\n{'=' * 80}")
    print("STARTING HYPERPARAMETER GRID SEARCH")
    print("=" * 80)
    
    for opt_name, opt_class in tqdm(OPTIMIZERS.items(), desc="Optimizers", position=0):
        for lr in LEARNING_RATES:
            experiment_num += 1
            print(f"\n{'=' * 80}")
            print(f"Experiment {experiment_num}/{total_experiments} - {opt_name} @ LR={lr:.0e}")
            print("=" * 80)
            
            # Create fresh model
            model = UNet(
                in_channels=1,
                out_channels=1,
                base_filters=64
            ).to(device)
            
            # Create optimizer with specific config
            if opt_name == 'SGD':
                optimizer = opt_class(model.parameters(), lr=lr, momentum=0.9)
            else:
                optimizer = opt_class(model.parameters(), lr=lr)
            
            criterion = DiceBCELoss()
            
            # Train
            start_time = time.time()
            best_dice, history = train_model(
                model, optimizer, criterion, 
                train_loader, val_loader, device,
                epochs=EXPERIMENT_EPOCHS
            )
            training_time = time.time() - start_time
            
            # Store results
            result = {
                'optimizer': opt_name,
                'learning_rate': lr,
                'best_val_dice': float(best_dice),
                'training_time_minutes': training_time / 60,
                'history': history
            }
            results.append(result)
            
            print(f"\n  ✓ COMPLETE: Best Val Dice={best_dice:.4f}, Time={training_time/60:.2f}min")
            print(f"  Progress: {experiment_num}/{total_experiments} experiments done ({experiment_num/total_experiments*100:.1f}%)")
    
    # Save results
    with open(RESULTS_DIR / 'hyperparameter_study_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "=" * 80)
    print("ALL EXPERIMENTS COMPLETE")
    print("=" * 80)
    
    # Analysis
    analyze_results(results)
    
    return results

def analyze_results(results):
    """Analyze and visualize hyperparameter study results"""
    print("\n" + "=" * 80)
    print("ANALYSIS")
    print("=" * 80)
    
    # Create DataFrame
    df = pd.DataFrame([
        {
            'Optimizer': r['optimizer'],
            'Learning Rate': r['learning_rate'],
            'Best Val Dice': r['best_val_dice'],
            'Training Time (min)': r['training_time_minutes']
        }
        for r in results
    ])
    
    # Find best configuration
    best_idx = df['Best Val Dice'].idxmax()
    best = df.iloc[best_idx]
    
    print("\n🏆 BEST CONFIGURATION:")
    print(f"  Optimizer: {best['Optimizer']}")
    print(f"  Learning Rate: {best['Learning Rate']}")
    print(f"  Best Val Dice: {best['Best Val Dice']:.4f}")
    print(f"  Training Time: {best['Training Time (min)']:.2f} minutes")
    
    # Summary by optimizer
    print("\n📊 RESULTS BY OPTIMIZER:")
    print("-" * 80)
    for opt in OPTIMIZERS.keys():
        opt_results = df[df['Optimizer'] == opt]
        best_dice = opt_results['Best Val Dice'].max()
        best_lr = opt_results.loc[opt_results['Best Val Dice'].idxmax(), 'Learning Rate']
        avg_dice = opt_results['Best Val Dice'].mean()
        print(f"{opt:10} | Best: {best_dice:.4f} (LR={best_lr:.0e}) | Avg: {avg_dice:.4f}")
    
    # Summary by learning rate
    print("\n📊 RESULTS BY LEARNING RATE:")
    print("-" * 80)
    for lr in LEARNING_RATES:
        lr_results = df[df['Learning Rate'] == lr]
        best_dice = lr_results['Best Val Dice'].max()
        best_opt = lr_results.loc[lr_results['Best Val Dice'].idxmax(), 'Optimizer']
        avg_dice = lr_results['Best Val Dice'].mean()
        print(f"LR={lr:.0e} | Best: {best_dice:.4f} ({best_opt}) | Avg: {avg_dice:.4f}")
    
    # Create visualizations
    create_visualizations(df, results)

def create_visualizations(df, results):
    """Create comprehensive visualizations"""
    
    # 1. Heatmap: Optimizer vs Learning Rate
    pivot = df.pivot(index='Optimizer', columns='Learning Rate', values='Best Val Dice')
    
    plt.figure(figsize=(12, 6))
    sns.heatmap(pivot, annot=True, fmt='.4f', cmap='RdYlGn', 
                vmin=0.5, vmax=0.9, cbar_kws={'label': 'Best Val Dice'})
    plt.title('Hyperparameter Study: Best Validation Dice Coefficient', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Learning Rate', fontsize=12)
    plt.ylabel('Optimizer', fontsize=12)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'bonus3_heatmap.png', dpi=150)
    print(f"\n✓ Saved heatmap to figures/bonus3_heatmap.png")
    plt.close()
    
    # 2. Learning curves for best configuration
    best_config = max(results, key=lambda x: x['best_val_dice'])
    history = best_config['history']
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = [h['epoch'] for h in history]
    train_loss = [h['train_loss'] for h in history]
    val_loss = [h['val_loss'] for h in history]
    train_dice = [h['train_dice'] for h in history]
    val_dice = [h['val_dice'] for h in history]
    
    # Loss
    axes[0].plot(epochs, train_loss, 'b-', label='Train Loss', linewidth=2)
    axes[0].plot(epochs, val_loss, 'r-', label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Loss Curves (Best Config)', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Dice
    axes[1].plot(epochs, train_dice, 'b-', label='Train Dice', linewidth=2)
    axes[1].plot(epochs, val_dice, 'r-', label='Val Dice', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Dice Coefficient', fontsize=12)
    axes[1].set_title(f'Dice Curves (Best: {best_config["optimizer"]}, LR={best_config["learning_rate"]:.0e})', 
                     fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'bonus3_best_curves.png', dpi=150)
    print(f"✓ Saved learning curves to figures/bonus3_best_curves.png")
    plt.close()
    
    # 3. Bar chart comparison
    fig, ax = plt.subplots(figsize=(14, 6))
    
    x = range(len(df))
    colors = ['blue', 'green', 'red', 'orange']
    opt_colors = {opt: colors[i] for i, opt in enumerate(OPTIMIZERS.keys())}
    bar_colors = [opt_colors[opt] for opt in df['Optimizer']]
    
    bars = ax.bar(x, df['Best Val Dice'], color=bar_colors, alpha=0.7, edgecolor='black')
    ax.set_xlabel('Configuration', fontsize=12)
    ax.set_ylabel('Best Validation Dice', fontsize=12)
    ax.set_title('Hyperparameter Study: All Configurations', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f"{row['Optimizer']}\n{row['Learning Rate']:.0e}" 
                        for _, row in df.iterrows()], rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=color, label=opt, alpha=0.7) 
                      for opt, color in opt_colors.items()]
    ax.legend(handles=legend_elements, loc='lower right')
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'bonus3_comparison.png', dpi=150)
    print(f"✓ Saved comparison chart to figures/bonus3_comparison.png")
    plt.close()

if __name__ == '__main__':
    results = run_hyperparameter_study()
    
    print("\n" + "=" * 80)
    print("HYPERPARAMETER STUDY COMPLETE!")
    print("=" * 80)
    print("\nResults saved to:")
    print("  - results/hyperparameter_study_results.json")
    print("  - figures/bonus3_heatmap.png")
    print("  - figures/bonus3_best_curves.png")
    print("  - figures/bonus3_comparison.png")
