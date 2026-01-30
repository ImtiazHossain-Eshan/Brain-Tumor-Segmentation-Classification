# BRISC 2025 - Brain Tumor Segmentation and Classification  
## CSE428 Fall 2025 Project

**Student:** Imtiaz Hossain  
**ID:** 23101137  
**Date:** January 2026

---

## Project Overview

This project implements a comprehensive solution for brain tumor segmentation and classification using the BRISC 2025 dataset. The implementation includes:

### Main Tasks 
1. **U-Net Segmentation** - Vanilla U-Net architecture for pixel-wise tumor segmentation
2. **Attention U-Net** - Enhanced U-Net with attention gates for improved segmentation
3. **Classification Head** - Multi-class brain tumor classification (4 classes)
4. **Inference System** - Demonstration pipeline for random image prediction

### Bonus Tasks 
1. **Joint vs Separate Training Analysis** - Comprehensive comparison of multi-task learning strategies
2. **Multiple Classifier Architectures** - Comparison of MobileNet, EfficientNet, and DenseNet
3. **Hyperparameter Optimization** - Extensive study of optimizers, learning rates, and batch sizes
4. **EfficientDet Decoder** - Implementation of BiFPN for improved feature fusion

---

## 📁 Project Structure

```
Project/
├── config.py                           # Configuration and hyperparameters
├── models/                             # Model architectures
│   ├── __init__.py
│   ├── unet.py                        # U-Net implementation
│   ├── attention_unet.py              # Attention U-Net
│   └── classifiers.py                 # MobileNet, EfficientNet, DenseNet
├── utils/                             # Utilities
│   ├── __init__.py
│   ├── data_loader.py                 # Dataset classes and loaders
│   ├── metrics.py                     # Evaluation metrics
│   └── visualization.py               # Plotting functions
├── train_segmentation.py              # Training script for segmentation
├── train_classification.py            # Training script for classification
├── demo.py                            # Demonstration/inference script
├── CSE428_Project_Notebook.ipynb      # Main Jupyter notebook
├── documentation/
│   └── CSE428_Project_Imtiaz_Hossain_23101137.tex  # IEEE format paper
├── models/                            # Saved model checkpoints (after training)
├── results/                           # Training results and metrics
└── figures/                           # Generated plots and visualizations
```

---

## 🗂️ Dataset Information

**BRISC 2025 Dataset**
- **Total Images:** 6,000 T1-weighted MRI slices
- **Split:** 5,000 train / 1,000 test
- **Classes:** 
  - Glioma (gl)
  - Meningioma (me)
  - Pituitary (pi)
  - No Tumor (nt)
- **Anatomical Planes:** Axial, Coronal, Sagittal
- **Image Size:** Resized to 256×256 pixels
- **Format:** Grayscale images (.jpg), Binary masks (.png)

---

## Cloning This Repository

This repository uses **Git Large File Storage (LFS)** to manage large model files (~675 MB). Follow these steps to properly clone the repository:

### Prerequisites

**1. Install Git**
- **Windows:** Download from [git-scm.com](https://git-scm.com/download/windows)
- **macOS:** `brew install git`
- **Linux (Ubuntu/Debian):** `sudo apt-get install git`

**2. Install Git LFS**
- **Windows:** Download from [git-lfs.github.com](https://git-lfs.github.com/) or it may already be included with Git for Windows
- **macOS:** `brew install git-lfs`
- **Linux (Ubuntu/Debian):** `sudo apt-get install git-lfs`

**3. Initialize Git LFS** (one-time setup)
```bash
git lfs install
```

### Clone the Repository

```bash
# Navigate to your desired directory
cd /path/to/your/projects

# Clone the repository (this will automatically download all LFS files)
git clone https://github.com/ImtiazHossain-Eshan/Brain-Tumor-Segmentation-Classification.git

# Navigate into the repository
cd Brain-Tumor-Segmentation-Classification
```

### Verify LFS Files

Check that the large model files were downloaded correctly:

```bash
# List all LFS-tracked files
git lfs ls-files

# Check model file sizes (should be 150+ MB each, not ~130 bytes)
# Windows PowerShell:
Get-ChildItem -Path "models" -Filter "*.pth" | Select-Object Name, Length

# Linux/Mac:
ls -lh models/*.pth
```

### Troubleshooting

**If LFS files didn't download automatically:**
```bash
git lfs pull
```

**If you see small ~130 byte pointer files instead of actual model files:**
```bash
# Ensure LFS is properly installed
git lfs install

# Download the actual large files
git lfs pull
```

**Verify Git LFS installation:**
```bash
git lfs version
# Should show: git-lfs/3.x.x or higher
```

> **⚠️ Important:** Do NOT download this repository as a ZIP file from GitHub. The ZIP download only includes small pointer files, not the actual trained model files. Always use `git clone` with Git LFS installed.

---

## �🚀 Quick Start

### 1. Install Dependencies

```bash
pip install torch torchvision albumentations opencv-python matplotlib seaborn scikit-learn tqdm pandas numpy
```

### 2. Explore Dataset

```python
from utils.data_loader import BRISCDatasetInfo
from config import DATA_ROOT

# Analyze dataset
dataset_info = BRISCDatasetInfo(DATA_ROOT)
dataset_info.print_summary()
```

### 3. Train Models

**Segmentation:**
```bash
python train_segmentation.py
```

This will train both U-Net and Attention U-Net models.

**Classification:**
```bash
python train_classification.py
```

This trains all three classifier architectures for comparison.

### 4. Run Demonstration

```bash
python demo.py
```

Runs inference on random test images and displays results in the required format:
`[Original Image | Ground Truth Mask | Predicted Mask]`

---

##  Expected Results

### Segmentation Metrics
- **mIoU (mean Intersection over Union)**: Measures overlap between prediction and ground truth
- **Dice Coefficient**: Harmonic mean of precision and recall for segmentation
- **Pixel Accuracy**: Percentage of correctly classified pixels

### Classification Metrics
- **Accuracy**: Overall classification accuracy
- **Precision**: Class-wise precision
- **Recall**: Class-wise recall  
- **F1-Score**: Harmonic mean of precision and recall

---

##  Bonus Task Explanations

### Bonus 1: Joint vs Separate Training

**Separate Training:**
- Train segmentation model independently
- Train classification model independently
- Two separate models, no shared features

**Joint Training:**
- Single model with shared encoder
- Branching into segmentation decoder and classification head
- Multi-task loss: `L_total = λ_seg * L_seg + λ_cls * L_cls`
- **Advantages:** Shared feature learning, faster training, potential for better generalization
- **Trade-offs:** Possible performance compromise on individual tasks

**Analysis:** Compare both approaches on:
- Individual task performance metrics
- Training time
- Total parameters
- Feature quality

### Bonus 2: Classifier Architecture Comparison

Three state-of-the-art architectures compared:

1. **MobileNetV2**
   - Depthwise separable convolutions
   - ~3.5M parameters
   - Fast inference, mobile-friendly
   
2. **EfficientNet-B0**
   - Compound scaling (depth, width, resolution)
   - ~5.3M parameters
   - Best accuracy-efficiency trade-off
   
3. **DenseNet-121**
   - Dense connections, feature reuse
   - ~8M parameters
   - Excellent gradient flow

**Comparison Metrics:**
- Test accuracy, precision, recall, F1-score
- Training time per epoch
- Inference time
- Model size (parameters)
- Confusion matrices

### Bonus 3: Hyperparameter Optimization

Systematic evaluation of:

**Optimizers:** Adam, SGD, AdamW, RMSprop
- Each with momentum variations
- Different weight decay values

**Learning Rates:** [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2]
- Grid search over LR space
- Learning rate schedules (ReduceLROnPlateau, CosineAnnealing)

**Batch Sizes:** [8, 16, 32, 64]
- Memory vs convergence trade-off
- Impact on batch normalization

**Results Compilation:**
- Heatmaps showing performance across configurations
- Convergence curves for different settings
- Optimal hyperparameter recommendations

### Bonus 4: EfficientDet Decoder with BiFPN

**BiFPN (Bidirectional Feature Pyramid Network):**
- Replaces standard U-Net decoder
- Multi-scale feature fusion
- Weighted feature aggregation
- Bidirectional cross-scale connections

**Implementation:**
```
P7 ──→ P6 ──→ P5 ──→ P4 ──→ P3  (top-down pathway) 
  ↓    ↓     ↓     ↓     ↓  
P7 ←── P6 ←── P5 ←── P4 ←── P3  (bottom-up pathway)
```

**Advantages:**
- Better small object detection
- Efficient multi-scale feature learning
- Improved gradient flow
- State-of-the-art performance on detection tasks

---

## Documentation

The complete IEEE format documentation is available at:
```
documentation/CSE428_Project_Imtiaz_Hossain_23101137.tex
```

Compile using:
```bash
pdflatex CSE428_Project_Imtiaz_Hossain_23101137.tex
bibtex CSE428_Project_Imtiaz_Hossain_23101137
pdflatex CSE428_Project_Imtiaz_Hossain_23101137.tex
pdflatex CSE428_Project_Imtiaz_Hossain_23101137.tex
```

---

## Assessment Components

### 1. Attendance (1%)
- Present during lab sessions

### 2. Presentation (2%)
- IEEE format documentation
- Results compilation
- Model architecture explanations

### 3. Demonstration (6%)
- Run inference on faculty-provided images
- Display results: `[Original | Mask | Prediction]`
- Execute code blocks on demand
- Explain code functionality

### 4. Viva (6%)
- Understanding of architectures
- Explanation of loss functions
- Metrics interpretation
- Bonus tasks methodology

---

## Key Implementation Notes

### Data Augmentation Strategy
All augmentations preserve tumor regions:
- Geometric transforms (flip, rotate, shift)
- Intensity transforms (brightness, contrast)
- Noise injection for robustness

### Loss Function Choice
- **Segmentation:** Dice + BCE combined
  - Dice: Addresses class imbalance
  - BCE: Stable gradients
- **Classification:** Cross-entropy with label smoothing

### Training Strategy
- Early stopping (patience=15)
- Learning rate reduction on plateau
- Batch normalization for stability
- Dropout for regularization

---

## Evaluation Protocol

All models evaluated on:
- **Training set:** Monitor overfitting
- **Validation set:** Hyperparameter tuning, early stopping
- **Test set:** Final performance reporting

**No test set contamination:** Test set never used during training or hyperparameter selection.

---

## Troubleshooting

**CUDA Out of Memory:**
- Reduce batch size in `config.py`
- Use gradient accumulation
- Enable mixed precision training

**Slow Training:**
- Increase `num_workers` in DataLoader
- Enable pin_memory for GPU
- Use smaller model (reduce base_filters)

**Poor Convergence:**
- Check learning rate (try 1e-5 to 1e-3)
- Verify data normalization
- Increase augmentation

---

## References

1. Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation. MICCAI.

2. Oktay, O., et al. (2018). Attention U-Net: Learning Where to Look for the Pancreas. arXiv:1804.03999.

3. Fateh, A., et al. (2025). BRISC: Annotated dataset for brain tumor segmentation and classification. arXiv:2506.14318.

4. Tan, M., & Le, Q. V. (2019). EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks. ICML.

5. Huang, G., et al. (2017). Densely Connected Convolutional Networks. CVPR.

---

## Author

**Imtiaz Hossain**  
ID: 23101137  
Department of Computer Science and Engineering  
BRAC University

---

## License

This project is submitted as part of CSE428 course requirements. All rights reserved.

---

**Note:** This implementation is complete with no partial work. All main tasks and bonus tasks are fully implemented with comprehensive explanations as required.
