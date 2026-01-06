"""
Training script for classification models  
Bonus Task 2: Compare multiple classifier architectures
Author: Imtiaz Hossain (ID: 23101137)
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm
import time

import sys
sys.path.append(str(Path(__file__).parent.parent))

from config import *
from models import get_classifier
from utils.data_loader import BRISCClassificationDataset, get_train_transforms, get_val_transforms
from utils.metrics import ClassificationMetrics
from utils.visualization import plot_training_curves, plot_confusion_matrix


class ClassificationTrainer:
    """Trainer for classification models"""
    
    def __init__(
        self,
        model,
        device,
        train_loader,
        val_loader,
        test_loader,
        optimizer,
        criterion,
        scheduler=None,
        model_name="classifier",
        save_dir=MODELS_DIR
    ):
        self.model = model.to(device)
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.model_name = model_name
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True, parents=True)
        
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }
        
        self.best_val_acc = 0.0
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        metrics = ClassificationMetrics(NUM_CLASSES)
        total_loss = 0
        
        pbar = tqdm(self.train_loader, desc='Training')
        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            metrics.update(outputs.detach(), labels)
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / len(self.train_loader)
        epoch_metrics = metrics.get_metrics()
        
        return avg_loss, epoch_metrics
    
    def validate(self, loader=None):
        """Validate model"""
        if loader is None:
            loader = self.val_loader
        
        self.model.eval()
        metrics = ClassificationMetrics(NUM_CLASSES)
        total_loss = 0
        
        with torch.no_grad():
            for images, labels in tqdm(loader, desc='Validation'):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                metrics.update(outputs, labels)
        
        avg_loss = total_loss / len(loader)
        epoch_metrics = metrics.get_metrics()
        
        return avg_loss, epoch_metrics, metrics
    
    def train(self, epochs):
        """Complete training loop"""
        print(f"\nTraining {self.model_name}")
        print("=" * 80)
        
        start_time = time.time()
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            print("-" * 80)
            
            train_loss, train_metrics = self.train_epoch()
            val_loss, val_metrics, val_metrics_obj = self.validate()
            
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_metrics['accuracy'])
            self.history['val_acc'].append(val_metrics['accuracy'])
            
            print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            print(f"Train Acc: {train_metrics['accuracy']:.4f} | Val Acc: {val_metrics['accuracy']:.4f}")
            print(f"Train F1: {train_metrics['f1_score']:.4f} | Val F1: {val_metrics['f1_score']:.4f}")
            
            if self.scheduler:
                self.scheduler.step(val_loss)
            
            if val_metrics['accuracy'] > self.best_val_acc:
                self.best_val_acc = val_metrics['accuracy']
                self.save_checkpoint('best')
                print("✓ Saved best model")
        
        training_time = time.time() - start_time
        print(f"\nTraining completed in {training_time / 60:.2f} minutes")
        
        # Test
        self.load_checkpoint('best')
        test_loss, test_metrics, test_metrics_obj = self.validate(self.test_loader)
        
        print("\nTest Set Results:")
        for key, val in test_metrics.items():
            print(f"{key}: {val:.4f}")
        
        # Plot confusion matrix
        cm = test_metrics_obj.get_confusion_matrix()
        plot_confusion_matrix(
            cm,
            CLASS_NAMES,
            save_path=FIGURES_DIR / f'{self.model_name}_confusion_matrix.png'
        )
        
        results = {
            'model_name': self.model_name,
            'training_time': training_time,
            'best_val_acc': self.best_val_acc,
            'test_metrics': test_metrics,
            'history': self.history
        }
        
        with open(self.save_dir / f"{self.model_name}_results.json", 'w') as f:
            json.dump(results, f, indent=4)
        
        return results
    
    def save_checkpoint(self, name='checkpoint'):
        checkpoint_path = self.save_dir / f"{self.model_name}_{name}.pth"
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history
        }, checkpoint_path)
    
    def load_checkpoint(self, name='checkpoint'):
        checkpoint_path = self.save_dir / f"{self.model_name}_{name}.pth"
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])


def train_classifier(classifier_name='mobilenet', device=None, epochs=EPOCHS):
    """Train a specific classifier"""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create datasets
    train_dataset = BRISCClassificationDataset(
        root_dir=CLASSIFICATION_TRAIN,
        transform=get_train_transforms()
    )
    
    val_size = int(VALIDATION_SPLIT * len(train_dataset))
    train_size = len(train_dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(RANDOM_SEED)
    )
    
    test_dataset = BRISCClassificationDataset(
        root_dir=CLASSIFICATION_TEST,
        transform=get_val_transforms()
    )
    
    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    # Model
    model = get_classifier(classifier_name, num_classes=NUM_CLASSES)
    
    # Optimizer and loss
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=REDUCE_LR_PATIENCE, factor=0.5)
    
    # Trainer
    trainer = ClassificationTrainer(
        model=model,
        device=device,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        model_name=f'{classifier_name}_classifier'
    )
    
    results = trainer.train(epochs=epochs)
    
    plot_training_curves(
        trainer.history,
        metrics=['loss', 'acc'],
        save_path=FIGURES_DIR / f'{classifier_name}_training_curves.png'
    )
    
    return model, results


if __name__ == "__main__":
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    
    # Train all classifiers for comparison (Bonus Task 2)
    classifiers = ['mobilenet', 'efficientnet', 'densenet']
    all_results = {}
    
    for clf_name in classifiers:
        print(f"\n{'=' * 80}")
        print(f"TRAINING {clf_name.upper()} CLASSIFIER")
        print(f"{'=' * 80}")
        _, results = train_classifier(clf_name, epochs=50)
        all_results[clf_name] = results
    
    # Compare results
    print(f"\n{'=' * 80}")
    print("CLASSIFIER COMPARISON (BONUS TASK 2)")
    print(f"{'=' * 80}\n")
    
    for name, results in all_results.items():
        print(f"{name.upper()}:")
        print(f"  Test Accuracy: {results['test_metrics']['accuracy']:.4f}")
        print(f"  Test F1:       {results['test_metrics']['f1_score']:.4f}")
        print()
