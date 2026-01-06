"""
Configuration file for CSE428 BRISC 2025 Project
Author: Imtiaz Hossain (ID: 23101137)
"""

import os
from pathlib import Path

# ============================================================================
# PROJECT PATHS
# ============================================================================
PROJECT_ROOT = Path(__file__).parent
DATA_ROOT = PROJECT_ROOT / "brisc2025"
CLASSIFICATION_ROOT = DATA_ROOT / "classification_task"
SEGMENTATION_ROOT = DATA_ROOT / "segmentation_task"

# Training and testing directories
CLASSIFICATION_TRAIN = CLASSIFICATION_ROOT / "train"
CLASSIFICATION_TEST = CLASSIFICATION_ROOT / "test"
SEGMENTATION_TRAIN = SEGMENTATION_ROOT / "train"
SEGMENTATION_TEST = SEGMENTATION_ROOT / "test"

# Output directories
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"
LOGS_DIR = PROJECT_ROOT / "logs"

# Create output directories if they don't exist
for dir_path in [MODELS_DIR, RESULTS_DIR, FIGURES_DIR, LOGS_DIR]:
    dir_path.mkdir(exist_ok=True, parents=True)

# ============================================================================
# DATA CONFIGURATION
# ============================================================================
# Class labels for classification
CLASS_LABELS = {
    'glioma': 0,
    'meningioma': 1,
    'pituitary': 2,
    'no_tumor': 3
}

CLASS_NAMES = ['glioma', 'meningioma', 'pituitary', 'no_tumor']
NUM_CLASSES = len(CLASS_LABELS)

# Image properties
IMAGE_SIZE = (256, 256)  # Standard size for MRI images
IMAGE_CHANNELS = 1  # Grayscale images
MASK_CHANNELS = 1   # Binary masks

# ============================================================================
# TRAINING HYPERPARAMETERS
# ============================================================================

# General training parameters
BATCH_SIZE = 16
EPOCHS = 100
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
VALIDATION_SPLIT = 0.15  # 15% of training data for validation

# Early stopping
EARLY_STOPPING_PATIENCE = 15
REDUCE_LR_PATIENCE = 7
MIN_LR = 1e-7

# Data augmentation parameters
AUGMENTATION_PARAMS = {
    'horizontal_flip': True,
    'vertical_flip': True,
    'rotation_range': 15,
    'width_shift_range': 0.1,
    'height_shift_range': 0.1,
    'zoom_range': 0.1,
    'brightness_range': [0.8, 1.2],
}

# ============================================================================
# MODEL ARCHITECTURES CONFIGURATION
# ============================================================================

# U-Net configuration
UNET_FILTERS = [64, 128, 256, 512, 1024]  # Filters at each level
UNET_DROPOUT = 0.3

# Attention U-Net configuration
ATTENTION_UNET_FILTERS = [64, 128, 256, 512, 1024]
ATTENTION_UNET_DROPOUT = 0.3

# Classification head configuration
CLASSIFIER_DROPOUT = 0.5
CLASSIFIER_DENSE_UNITS = [512, 256, 128]

# ============================================================================
# LOSS FUNCTIONS
# ============================================================================
SEGMENTATION_LOSS = 'dice_bce'  # Dice + BCE combined loss
CLASSIFICATION_LOSS = 'categorical_crossentropy'

# Loss weights for multi-task learning
SEGMENTATION_WEIGHT = 1.0
CLASSIFICATION_WEIGHT = 1.0

# ============================================================================
# EVALUATION METRICS
# ============================================================================

# Segmentation metrics
SEGMENTATION_METRICS = ['mIoU', 'dice_coefficient', 'pixel_accuracy']

# Classification metrics
CLASSIFICATION_METRICS = ['accuracy', 'precision', 'recall', 'f1_score']

# ============================================================================
# BONUS TASK CONFIGURATIONS
# ============================================================================

# Bonus 2: Multiple classifier architectures
CLASSIFIER_ARCHITECTURES = ['MobileNet', 'EfficientNetB0', 'DenseNet121']

# Bonus 3: Hyperparameter optimization
OPTIMIZERS_TO_TEST = ['Adam', 'SGD', 'AdamW', 'RMSprop']
LEARNING_RATES_TO_TEST = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2]
BATCH_SIZES_TO_TEST = [8, 16, 32, 64]

# Bonus 4: EfficientDet configuration
EFFICIENTDET_BIFPN_CHANNELS = 64
EFFICIENTDET_NUM_BIFPN_LAYERS = 3

# ============================================================================
# RANDOM SEED FOR REPRODUCIBILITY
# ============================================================================
RANDOM_SEED = 42

# ============================================================================
# DEVICE CONFIGURATION
# ============================================================================
# Will be set automatically based on availability (CUDA/CPU)
DEVICE = None  # Set in training scripts

# ============================================================================
# LOGGING AND CHECKPOINTING
# ============================================================================
CHECKPOINT_FREQUENCY = 5  # Save checkpoint every N epochs
LOG_FREQUENCY = 10  # Log metrics every N batches
SAVE_BEST_ONLY = True  # Save only the best model

# ============================================================================
# VISUALIZATION SETTINGS
# ============================================================================
CMAP_IMAGE = 'gray'
CMAP_MASK = 'hot'
FIGURE_DPI = 150
FIGURE_SIZE = (15, 5)