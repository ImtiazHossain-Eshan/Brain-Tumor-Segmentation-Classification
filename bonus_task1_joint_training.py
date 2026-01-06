"""
Bonus Task 1: Joint vs Separate Training Analysis
Train UNetWithClassifier for joint segmentation + classification
Compare with separate training approach
"""

import sys
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from tqdm import tqdm
import json
import time
import numpy as np

sys.path.append(str(Path(__file__).parent))

from config import *
from models import UNetWithClassifier
from utils.data_loader import create_data_loaders
from utils.metrics import DiceBCELoss, MultiTaskMetrics
from utils.visualization import plot_training_curves

class JointTrainer:
    def __init__(self, model, seg_criterion, cls_criterion, optimizer, device, train_loader, val_loader):
        self.model = model
        self.seg_criterion = seg_criterion
        self.cls_criterion = cls_criterion
        self.optimizer = optimizer
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.best_val_dice = 0.0
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_seg_loss': [],
            'val_seg_loss': [],
            'train_cls_loss': [],
            'val_cls_loss': [],
            'train_dice': [],
            'val_dice': [],
            'train_accuracy': [],
            'val_accuracy': []
        }
        
    def train_epoch(self):
        self.model.train()
        metrics = MultiTaskMetrics(num_classes=NUM_CLASSES)
        total_loss = 0.0
        total_seg_loss = 0.0
        total_cls_loss = 0.0
        
        pbar = tqdm(self.train_loader, desc="Training")
        for images, masks, labels in pbar:
            images = images.to(self.device)
            masks = masks.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            seg_outputs, cls_outputs = self.model(images)
            
            # Calculate losses
            seg_loss = self.seg_criterion(seg_outputs, masks)
            cls_loss = self.cls_criterion(cls_outputs, labels)
            
            # Combined loss (weighted)
            loss = SEGMENTATION_WEIGHT * seg_loss + CLASSIFICATION_WEIGHT * cls_loss
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Track losses
            total_loss += loss.item()
            total_seg_loss += seg_loss.item()
            total_cls_loss += cls_loss.item()
            
            # Calculate metrics
            with torch.no_grad():
                seg_preds = torch.sigmoid(seg_outputs) > 0.5
                cls_preds = torch.argmax(cls_outputs, dim=1)
                metrics.update(seg_preds, masks, cls_preds, labels)
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / len(self.train_loader)
        avg_seg_loss = total_seg_loss / len(self.train_loader)
        avg_cls_loss = total_cls_loss / len(self.train_loader)
        epoch_metrics = metrics.get_metrics()
        
        return avg_loss, avg_seg_loss, avg_cls_loss, epoch_metrics
    
    def validate(self):
        self.model.eval()
        metrics = MultiTaskMetrics(num_classes=NUM_CLASSES)
        total_loss = 0.0
        total_seg_loss = 0.0
        total_cls_loss = 0.0
        
        with torch.no_grad():
            for images, masks, labels in tqdm(self.val_loader, desc="Validation"):
                images = images.to(self.device)
                masks = masks.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                seg_outputs, cls_outputs = self.model(images)
                
                # Calculate losses
                seg_loss = self.seg_criterion(seg_outputs, masks)
                cls_loss = self.cls_criterion(cls_outputs, labels)
                loss = SEGMENTATION_WEIGHT * seg_loss + CLASSIFICATION_WEIGHT * cls_loss
                
                total_loss += loss.item()
                total_seg_loss += seg_loss.item()
                total_cls_loss += cls_loss.item()
                
                # Metrics
                seg_preds = torch.sigmoid(seg_outputs) > 0.5
                cls_preds = torch.argmax(cls_outputs, dim=1)
                metrics.update(seg_preds, masks, cls_preds, labels)
        
        avg_loss = total_loss / len(self.val_loader)
        avg_seg_loss = total_seg_loss / len(self.val_loader)
        avg_cls_loss = total_cls_loss / len(self.val_loader)
        epoch_metrics = metrics.get_metrics()
        
        return avg_loss, avg_seg_loss, avg_cls_loss, epoch_metrics
    
    def train(self, epochs, patience=15):
        print("\nTraining Joint Model (U-Net + Classifier)")
        print("=" * 80)
        
        best_epoch = 0
        patience_counter = 0
        start_time = time.time()
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            print("-" * 80)
            
            # Train
            train_loss, train_seg_loss, train_cls_loss, train_metrics = self.train_epoch()
            
            # Validate
            val_loss, val_seg_loss, val_cls_loss, val_metrics = self.validate()
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_seg_loss'].append(train_seg_loss)
            self.history['val_seg_loss'].append(val_seg_loss)
            self.history['train_cls_loss'].append(train_cls_loss)
            self.history['val_cls_loss'].append(val_cls_loss)
            self.history['train_dice'].append(train_metrics.get('seg_dice_coefficient', 0.0))
            self.history['val_dice'].append(val_metrics.get('seg_dice_coefficient', 0.0))
            self.history['train_accuracy'].append(train_metrics.get('cls_accuracy', 0.0))
            self.history['val_accuracy'].append(val_metrics.get('cls_accuracy', 0.0))
            
            # Print metrics
            train_dice = train_metrics.get('seg_dice_coefficient', 0.0)
            val_dice = val_metrics.get('seg_dice_coefficient', 0.0)
            train_acc = train_metrics.get('cls_accuracy', 0.0)
            val_acc = val_metrics.get('cls_accuracy', 0.0)
            
            print(f"Train Loss: {train_loss:.4f} (Seg: {train_seg_loss:.4f}, Cls: {train_cls_loss:.4f}) | Val Loss: {val_loss:.4f}")
            print(f"Train Dice: {train_dice:.4f} | Val Dice: {val_dice:.4f}")
            print(f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
            
            # Save best model
            if val_dice > self.best_val_dice:
                self.best_val_dice = val_dice
                best_epoch = epoch + 1
                patience_counter = 0
                
                checkpoint = {
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'best_val_dice': self.best_val_dice,
                    'val_accuracy': val_acc,
                    'history': self.history
                }
                torch.save(checkpoint, MODELS_DIR / 'joint_unet_classifier_best.pth')
                print(f"✓ Saved best model (Dice: {self.best_val_dice:.4f}, Acc: {val_acc:.4f})")
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
    print("BONUS TASK 1: JOINT VS SEPARATE TRAINING")
    print("=" * 80)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create joint data loaders
    from utils.data_loader import BRISCJointDataset, get_train_transforms, get_val_transforms
    import torch.utils.data as data
    
    # Training dataset
    train_dataset = BRISCJointDataset(
        images_dir=SEGMENTATION_TRAIN / 'images',
        masks_dir=SEGMENTATION_TRAIN / 'masks',
        transform=get_train_transforms()
    )
    
    # Split into train and validation
    train_size = int((1 - VALIDATION_SPLIT) * len(train_dataset))
    val_size = len(train_dataset) - train_size
    
    train_dataset, val_dataset = data.random_split(
        train_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(RANDOM_SEED)
    )
    
    train_loader = data.DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    
    val_loader = data.DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    print(f"Found {len(train_dataset)} training samples")
    print(f"Found {len(val_dataset)} validation samples")
    
    # Create model
    model = UNetWithClassifier(
        in_channels=1,
        num_classes=NUM_CLASSES,
        base_filters=64
    ).to(device)
    
    # Loss and optimizer
    seg_criterion = DiceBCELoss()
    cls_criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Create trainer
    trainer = JointTrainer(
        model=model,
        seg_criterion=seg_criterion,
        cls_criterion=cls_criterion,
        optimizer=optimizer,
        device=device,
        train_loader=train_loader,
        val_loader=val_loader
    )
    
    # Train (use fewer epochs for joint training)
    history = trainer.train(epochs=50, patience=EARLY_STOPPING_PATIENCE)
    
    # Save results
    results = {
        'approach': 'joint_training',
        'best_val_dice': float(trainer.best_val_dice),
        'history': {k: [float(x) for x in v] for k, v in history.items()}
    }
    
    with open(RESULTS_DIR / 'joint_training_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "=" * 80)
    print("COMPARISON: JOINT VS SEPARATE")
    print("=" * 80)
    
    # Compare with separate training
    print("\n📊 Segmentation Performance:")
    print(f"  Separate (U-Net only):   {0.8310:.4f} Dice")
    print(f"  Joint (U-Net + Cls):     {trainer.best_val_dice:.4f} Dice")
    diff_seg = ((trainer.best_val_dice - 0.8310) / 0.8310) * 100
    print(f"  Difference:              {diff_seg:+.2f}%")
    
    print("\n📊 Classification Performance:")
    print(f"  Separate (DenseNet):     97.50% Accuracy")
    print(f"  Joint (U-Net + Cls):     {history['val_accuracy'][-1]*100:.2f}% Accuracy")
    
    print("\n" + "=" * 80)
    print("COMPLETE!")
    print("=" * 80)

if __name__ == '__main__':
    main()
