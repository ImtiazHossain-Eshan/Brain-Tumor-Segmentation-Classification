"""
Train only Attention U-Net model
This script trains just the Attention U-Net without retraining vanilla U-Net
"""

import sys
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from tqdm import tqdm
import json
import time

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from config import *
from models import AttentionUNet
from utils.data_loader import create_data_loaders
from utils.metrics import DiceBCELoss, SegmentationMetrics
from utils.visualization import plot_training_curves

class AttentionUNetTrainer:
    def __init__(self, model, criterion, optimizer, device, train_loader, val_loader):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.best_val_dice = 0.0
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_dice': [],
            'val_dice': [],
            'train_iou': [],
            'val_iou': [],
            'train_pixel_acc': [],
            'val_pixel_acc': []
        }
        
    def train_epoch(self):
        self.model.train()
        metrics = SegmentationMetrics()
        total_loss = 0.0
        
        pbar = tqdm(self.train_loader, desc="Training")
        for images, masks in pbar:
            images = images.to(self.device)
            masks = masks.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, masks)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # Calculate metrics
            with torch.no_grad():
                preds = torch.sigmoid(outputs) > 0.5
                metrics.update(preds, masks)
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / len(self.train_loader)
        epoch_metrics = metrics.get_metrics()
        
        return avg_loss, epoch_metrics
    
    def validate(self):
        self.model.eval()
        metrics = SegmentationMetrics()
        total_loss = 0.0
        
        with torch.no_grad():
            for images, masks in tqdm(self.val_loader, desc="Validation"):
                images = images.to(self.device)
                masks = masks.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                total_loss += loss.item()
                
                preds = torch.sigmoid(outputs) > 0.5
                metrics.update(preds, masks)
        
        avg_loss = total_loss / len(self.val_loader)
        epoch_metrics = metrics.get_metrics()
        
        return avg_loss, epoch_metrics
    
    def train(self, epochs, patience=15):
        print(f"\nTraining Attention U-Net")
        print("=" * 80)
        
        best_epoch = 0
        patience_counter = 0
        start_time = time.time()
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            print("-" * 80)
            
            # Train
            train_loss, train_metrics = self.train_epoch()
            
            # Validate
            val_loss, val_metrics = self.validate()
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_dice'].append(train_metrics['dice_coefficient'])
            self.history['val_dice'].append(val_metrics['dice_coefficient'])
            self.history['train_iou'].append(train_metrics['mIoU'])
            self.history['val_iou'].append(val_metrics['mIoU'])
            self.history['train_pixel_acc'].append(train_metrics['pixel_accuracy'])
            self.history['val_pixel_acc'].append(val_metrics['pixel_accuracy'])
            
            # Print metrics
            print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            print(f"Train Dice: {train_metrics['dice_coefficient']:.4f} | Val Dice: {val_metrics['dice_coefficient']:.4f}")
            
            # Save best model
            if val_metrics['dice_coefficient'] > self.best_val_dice:
                self.best_val_dice = val_metrics['dice_coefficient']
                best_epoch = epoch + 1
                patience_counter = 0
                
                checkpoint = {
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'best_val_dice': self.best_val_dice,
                    'history': self.history
                }
                torch.save(checkpoint, MODELS_DIR / 'attention_unet_best.pth')
                print(f"✓ Saved best model (Dice: {self.best_val_dice:.4f})")
            else:
                patience_counter += 1
                print(f"No improvement ({patience_counter}/{patience})")
            
            # Early stopping
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
        
        training_time = (time.time() - start_time) / 60
        print(f"\nTraining completed in {training_time:.2f} minutes")
        print(f"Best validation Dice: {self.best_val_dice:.4f} at epoch {best_epoch}")
        
        return self.history

def main():
    print("\n" + "=" * 80)
    print("TRAINING ATTENTION U-NET")
    print("=" * 80)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create data loaders
    loaders = create_data_loaders(
        batch_size=BATCH_SIZE,
        val_split=VALIDATION_SPLIT,
        num_workers=2
    )
    
    train_loader = loaders['train']
    val_loader = loaders['val']
    test_loader = loaders['test']
    
    # Create model
    model = AttentionUNet(
        in_channels=1,
        out_channels=1,
        base_filters=64  # First layer filters
    ).to(device)
    
    # Loss and optimizer
    criterion = DiceBCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Create trainer
    trainer = AttentionUNetTrainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        train_loader=train_loader,
        val_loader=val_loader
    )
    
    # Train
    history = trainer.train(epochs=EPOCHS, patience=EARLY_STOPPING_PATIENCE)
    
    # Save results
    results = {
        'model': 'attention_unet',
        'best_val_dice': trainer.best_val_dice,
        'history': history
    }
    
    with open(RESULTS_DIR / 'attention_unet_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Plot training curves
    plot_training_curves(
        history,
        save_path=FIGURES_DIR / 'attention_unet_training_curves.png'
    )
    
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE!")
    print("=" * 80)
    print(f"Best model saved to: {MODELS_DIR / 'attention_unet_best.pth'}")
    print(f"Results saved to: {RESULTS_DIR / 'attention_unet_results.json'}")
    print(f"Training curves saved to: {FIGURES_DIR / 'attention_unet_training_curves.png'}")

if __name__ == '__main__':
    main()
