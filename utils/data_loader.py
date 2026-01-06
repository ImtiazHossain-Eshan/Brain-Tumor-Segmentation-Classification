"""
Data loading and preprocessing utilities for BRISC 2025 dataset
Author: Imtiaz Hossain (ID: 23101137)
"""

import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, List, Dict, Optional
from sklearn.model_selection import train_test_split
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
from torch.utils.data import Dataset, DataLoader
import json

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import *


class BRISCDatasetInfo:
    """Utility class to analyze and display BRISC dataset information"""
    
    def __init__(self, data_root: Path):
        self.data_root = Path(data_root)
        self.classification_root = self.data_root / "classification_task"
        self.segmentation_root = self.data_root / "segmentation_task"
        
    def analyze_dataset(self) -> Dict:
        """Analyze and return dataset statistics"""
        stats = {
            'classification': self._analyze_classification(),
            'segmentation': self._analyze_segmentation()
        }
        return stats
    
    def _analyze_classification(self) -> Dict:
        """Analyze classification dataset"""
        stats = {'train': {}, 'test': {}}
        
        for split in ['train', 'test']:
            split_path = self.classification_root / split
            stats[split]['total'] = 0
            stats[split]['classes'] = {}
            
            for class_name in CLASS_NAMES:
                class_path = split_path / class_name
                if class_path.exists():
                    num_images = len(list(class_path.glob('*.jpg')))
                    stats[split]['classes'][class_name] = num_images
                    stats[split]['total'] += num_images
                else:
                    stats[split]['classes'][class_name] = 0
                    
        return stats
    
    def _analyze_segmentation(self) -> Dict:
        """Analyze segmentation dataset"""
        stats = {'train': {}, 'test': {}}
        
        for split in ['train', 'test']:
            images_path = self.segmentation_root / split / 'images'
            masks_path = self.segmentation_root / split / 'masks'
            
            if images_path.exists():
                num_images = len(list(images_path.glob('*.jpg')))
                stats[split]['num_images'] = num_images
            else:
                stats[split]['num_images'] = 0
                
            if masks_path.exists():
                num_masks = len(list(masks_path.glob('*.png')))
                stats[split]['num_masks'] = num_masks
            else:
                stats[split]['num_masks'] = 0
                
        return stats
    
    def print_summary(self):
        """Print dataset summary"""
        stats = self.analyze_dataset()
        
        print("=" * 80)
        print("BRISC 2025 Dataset Summary")
        print("=" * 80)
        
        print("\n CLASSIFICATION TASK:")
        print("-" * 80)
        for split in ['train', 'test']:
            print(f"\n{split.upper()}:")
            print(f"  Total images: {stats['classification'][split]['total']}")
            print(f"  Class distribution:")
            for class_name, count in stats['classification'][split]['classes'].items():
                print(f"    - {class_name:12s}: {count:5d} images")
        
        print("\n SEGMENTATION TASK:")
        print("-" * 80)
        for split in ['train', 'test']:
            print(f"\n{split.upper()}:")
            print(f"  Images: {stats['segmentation'][split]['num_images']}")
            print(f"  Masks:  {stats['segmentation'][split]['num_masks']}")
        
        print("\n" + "=" * 80)


class BRISCSegmentationDataset(Dataset):
    """PyTorch Dataset for BRISC segmentation task"""
    
    def __init__(
        self, 
        images_dir: Path, 
        masks_dir: Path,
        transform: Optional[A.Compose] = None,
        image_size: Tuple[int, int] = IMAGE_SIZE
    ):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.transform = transform
        self.image_size = image_size
        
        # Get all image filenames
        self.image_files = sorted(list(self.images_dir.glob('*.jpg')))
        
        # Verify corresponding masks exist
        self.valid_samples = []
        for img_file in self.image_files:
            mask_file = self.masks_dir / f"{img_file.stem}.png"
            if mask_file.exists():
                self.valid_samples.append((img_file, mask_file))
        
        print(f"Found {len(self.valid_samples)} valid image-mask pairs")
    
    def __len__(self) -> int:
        return len(self.valid_samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path, mask_path = self.valid_samples[idx]
        
        # Load image and mask
        image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        
        # Resize
        image = cv2.resize(image, self.image_size)
        mask = cv2.resize(mask, self.image_size, interpolation=cv2.INTER_NEAREST)
        
        # Normalize image to [0, 1]
        image = image.astype(np.float32) / 255.0
        mask = mask.astype(np.float32) / 255.0
        
        # Apply transformations
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
        else:
            # Convert to tensor
            image = torch.from_numpy(image).unsqueeze(0)  # Add channel dimension
            mask = torch.from_numpy(mask).unsqueeze(0)
        
        # Ensure channel dimension exists for mask
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)  # Add channel dimension if missing
        
        # Ensure mask is binary
        mask = (mask > 0.5).float()
        
        return image, mask


class BRISCClassificationDataset(Dataset):
    """PyTorch Dataset for BRISC classification task"""
    
    def __init__(
        self,
        root_dir: Path,
        transform: Optional[A.Compose] = None,
        image_size: Tuple[int, int] = IMAGE_SIZE
    ):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.image_size = image_size
        
        # Collect all images with labels
        self.samples = []
        for class_name, class_idx in CLASS_LABELS.items():
            class_dir = self.root_dir / class_name
            if class_dir.exists():
                for img_file in class_dir.glob('*.jpg'):
                    self.samples.append((img_file, class_idx))
        
        print(f"Found {len(self.samples)} classification samples")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        
        # Load image
        image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        
        # Resize
        image = cv2.resize(image, self.image_size)
        
        # Normalize to [0, 1]
        image = image.astype(np.float32) / 255.0
        
        # Apply transformations
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']
        else:
            # Convert to tensor
            image = torch.from_numpy(image).unsqueeze(0)  # Add channel dimension
        
        return image, label


class BRISCJointDataset(Dataset):
    """
    Joint dataset for multi-task learning (segmentation + classification)
    Loads images from segmentation task with both mask and classification label
    """
    
    def __init__(
        self,
        images_dir: Path,
        masks_dir: Path,
        transform: Optional[A.Compose] = None,
        image_size: Tuple[int, int] = IMAGE_SIZE
    ):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.transform = transform
        self.image_size = image_size
        
        # Get all image-mask pairs
        self.image_files = sorted(list(self.images_dir.glob('*.jpg')))
        self.samples = []
        
        for img_file in self.image_files:
            mask_file = self.masks_dir / f"{img_file.stem}.png"
            if mask_file.exists():
                # Extract class from filename
                # Format: brisc2025_train_00001_gl_ax_t1.jpg
                # Tumor type is at index 3: gl, me, pi, nt
                parts = img_file.stem.split('_')
                tumor_code = parts[3]
                
                # Map tumor code to class
                tumor_map = {'gl': 0, 'me': 1, 'pi': 2, 'nt': 3}
                class_idx = tumor_map.get(tumor_code, -1)
                
                if class_idx != -1:
                    self.samples.append((img_file, mask_file, class_idx))
        
        print(f"Found {len(self.samples)} joint training samples")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        img_path, mask_path, class_idx = self.samples[idx]
        
        # Load image and mask
        image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        
        # Resize
        image = cv2.resize(image, self.image_size)
        mask = cv2.resize(mask, self.image_size, interpolation=cv2.INTER_NEAREST)
        
        # Normalize
        image = image.astype(np.float32) / 255.0
        mask = mask.astype(np.float32) / 255.0
        
        # Apply transformations
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
        else:
            image = torch.from_numpy(image).unsqueeze(0)
            mask = torch.from_numpy(mask).unsqueeze(0)
        
        # Ensure channel dimension exists for mask
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)  # Add channel dimension if missing
        
        # Ensure mask is binary
        mask = (mask > 0.5).float()
        
        return image, mask, class_idx


def get_train_transforms() -> A.Compose:
    """Get training augmentation pipeline"""
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=15, p=0.5),
        A.Affine(
            translate_percent={'x': (-0.1, 0.1), 'y': (-0.1, 0.1)},
            scale=(0.9, 1.1),
            p=0.5
        ),
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=0.5
        ),
        A.GaussNoise(p=0.3),
        ToTensorV2()
    ])


def get_val_transforms() -> A.Compose:
    """Get validation/test transforms (no augmentation)"""
    return A.Compose([
        ToTensorV2()
    ])


def create_data_loaders(
    batch_size: int = BATCH_SIZE,
    val_split: float = VALIDATION_SPLIT,
    num_workers: int = 4
) -> Dict[str, DataLoader]:
    """
    Create data loaders for training, validation, and testing
    
    Returns:
        Dictionary containing data loaders for different splits
    """
    # For segmentation task
    seg_train_dataset = BRISCSegmentationDataset(
        images_dir=SEGMENTATION_TRAIN / 'images',
        masks_dir=SEGMENTATION_TRAIN / 'masks',
        transform=get_train_transforms()
    )
    
    # Split into train and validation
    train_size = int((1 - val_split) * len(seg_train_dataset))
    val_size = len(seg_train_dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        seg_train_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(RANDOM_SEED)
    )
    
    # Test dataset
    seg_test_dataset = BRISCSegmentationDataset(
        images_dir=SEGMENTATION_TEST / 'images',
        masks_dir=SEGMENTATION_TEST / 'masks',
        transform=get_val_transforms()
    )
    
    # Create data loaders
    loaders = {
        'train': DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        ),
        'val': DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        ),
        'test': DataLoader(
            seg_test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
    }
    
    return loaders