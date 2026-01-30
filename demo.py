"""
Demonstration script for inference on random images
Format: 2x3 grid
Author: Imtiaz Hossain (ID: 23101137)
"""

import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import random

import sys
sys.path.append(str(Path(__file__).parent))

from config import *
from models import UNet, AttentionUNet
from models.classifiers import DenseNetClassifier, EfficientNetClassifier, MobileNetClassifier

# Classification labels - MUST match training order from config.py!
CLASS_NAMES = ['Glioma', 'Meningioma', 'Pituitary', 'No Tumor']  # Indices: 0=glioma, 1=meningioma, 2=pituitary, 3=no_tumor

def get_classifier(model_type, num_classes=4, pretrained=False):
    """Get classifier model"""
    if model_type == 'densenet':
        return DenseNetClassifier(num_classes=num_classes, pretrained=pretrained)
    elif model_type == 'efficientnet':
        return EfficientNetClassifier(num_classes=num_classes, pretrained=pretrained)
    elif model_type == 'mobilenet':
        return MobileNetClassifier(num_classes=num_classes, pretrained=pretrained)
    else:
        raise ValueError(f"Unknown classifier type: {model_type}")


def load_model(model_path, model_type='unet', device=None):
    """
    Load trained model
    
    Args:
        model_path: Path to model checkpoint
        model_type: 'unet' or 'attention_unet'
        device: Device to load model on
    
    Returns:
        Loaded model
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model
    if model_type == 'unet':
        model = UNet(in_channels=1, out_channels=1, base_filters=64)
    elif model_type == 'attention_unet':
        model = AttentionUNet(in_channels=1, out_channels=1, base_filters=64)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Load weights
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model


def create_overlay(image, mask, alpha=0.5, color_mask='red'):
    """
    Create image with mask overlay
    
    Args:
        image: Grayscale image (H, W)
        mask: Binary mask (H, W)
        alpha: Transparency for overlay
        color_mask: Color for mask ('red' or 'green')
    
    Returns:
        RGB image with colored mask overlay
    """
    # Convert grayscale to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    
    # Create colored mask
    mask_colored = np.zeros_like(image_rgb)
    if color_mask == 'red':
        mask_colored[:, :, 0] = mask * 255  # Red channel
    elif color_mask == 'green':
        mask_colored[:, :, 1] = mask * 255  # Green channel
    
    # Blend image with mask
    overlay = cv2.addWeighted(image_rgb, 1.0, mask_colored, alpha, 0)
    
    return overlay


def inference_on_image(image_path, seg_model, cls_model, device, output_prefix='demo'):
    """
    Run BOTH segmentation AND classification inference on an image
    Display result in REQUIRED format:
    
    Top row (Ground Truth):
        [Original Image | Original Mask | Original + Mask Overlay (red)]
    
    Bottom row (Predictions):
        [Processed Image | Predicted Mask | Original + Predicted Overlay (green)]
    
    + Classification result shown at top with confidence scores
    
    Args:
        image_path: Path to input image
        seg_model: Trained segmentation model
        cls_model: Trained classification model
        device: Device
        output_prefix: Prefix for output filename (default: 'demo')
    """
    # Load image
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    original_shape = image.shape
    
    # Load corresponding mask if it exists  
    mask_path = str(image_path).replace('/images/', '/masks/').replace('\\images\\', '\\masks\\').replace('.jpg', '.png')
    mask = None
    has_ground_truth = False
    if Path(mask_path).exists():
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = (mask > 127).astype(np.uint8)  # Binarize
        has_ground_truth = True
    
    # Preprocess image
    image_resized = cv2.resize(image, IMAGE_SIZE)
    image_normalized = image_resized.astype(np.float32) / 255.0
    image_tensor = torch.from_numpy(image_normalized).unsqueeze(0).unsqueeze(0).to(device)
    
    # Run segmentation inference
    with torch.no_grad():
        seg_pred = seg_model(image_tensor)
        pred_mask = torch.sigmoid(seg_pred)
        pred_mask = (pred_mask > 0.5).float()
    
    # Run classification inference
    with torch.no_grad():
        cls_pred = cls_model(image_tensor)
        cls_probs = torch.softmax(cls_pred, dim=1)
        cls_pred_idx = torch.argmax(cls_probs, dim=1).item()
        cls_confidence = cls_probs[0, cls_pred_idx].item() * 100
        predicted_class = CLASS_NAMES[cls_pred_idx]
    
    # Extract ground truth class from filename
    # Format: brisc2025_test_XXXXX_{gl|me|pi|nt}_ax_t1.jpg
    filename = Path(image_path).stem
    tumor_code_map = {
        'gl': 'Glioma',
        'me': 'Meningioma', 
        'pi': 'Pituitary',
        'nt': 'No Tumor'
    }
    
    # Extract tumor code from filename
    parts = filename.split('_')
    gt_class = 'Unknown'
    for code, name in tumor_code_map.items():
        if code in parts:
            gt_class = name
            break
    
    # Convert to numpy
    pred_mask_np = pred_mask.squeeze().cpu().numpy()
    pred_mask_resized = cv2.resize(pred_mask_np, (original_shape[1], original_shape[0]))
    pred_mask_binary = (pred_mask_resized > 0.5).astype(np.uint8)
    
    # Create figure with 2x3 grid
    fig = plt.figure(figsize=(15, 10))
    
    if has_ground_truth:
        # TOP ROW - Ground Truth
        # 1. Original Image
        ax1 = plt.subplot(2, 3, 1)
        ax1.imshow(image, cmap='gray')
        ax1.set_title('Original Image', fontsize=14, fontweight='bold')
        ax1.axis('off')
        
        # 2. Original Mask
        ax2 = plt.subplot(2, 3, 2)
        ax2.imshow(mask, cmap='hot')
        ax2.set_title('Ground Truth Mask', fontsize=14, fontweight='bold', color='red')
        ax2.axis('off')
        
        # 3. Original Image with Mask Overlay (RED)
        ax3 = plt.subplot(2, 3, 3)
        overlay_gt = create_overlay(image, mask, alpha=0.5, color_mask='red')
        ax3.imshow(overlay_gt)
        ax3.set_title('Ground Truth Overlay', fontsize=14, fontweight='bold', color='red')
        ax3.axis('off')
        
        # BOTTOM ROW - Predictions
        # 4. Processed Image (or just show original)
        ax4 = plt.subplot(2, 3, 4)
        ax4.imshow(image, cmap='gray')
        ax4.set_title('Original Image\n(or applicable else)', fontsize=12, fontweight='bold')
        ax4.axis('off')
        
        # 5. Predicted Mask
        ax5 = plt.subplot(2, 3, 5)
        ax5.imshow(pred_mask_binary, cmap='hot')
        ax5.set_title('Predicted Mask', fontsize=14, fontweight='bold', color='green')
        ax5.axis('off')
        
        # 6. Original Image with Predicted Mask Overlay (GREEN)
        ax6 = plt.subplot(2, 3, 6)
        overlay_pred = create_overlay(image, pred_mask_binary, alpha=0.5, color_mask='green')
        ax6.imshow(overlay_pred)
        ax6.set_title('Prediction Overlay', fontsize=14, fontweight='bold', color='green')
        ax6.axis('off')
        
        # Add classification and segmentation results
        intersection = np.logical_and(mask, pred_mask_binary).sum()
        union = np.logical_or(mask, pred_mask_binary).sum()
        iou = intersection / (union + 1e-5)
        
        # Show classification correctness
        cls_correct = "✓" if gt_class == predicted_class else "✗"
        
        fig.suptitle(f'Ground Truth: {gt_class} | Predicted: {predicted_class} ({cls_confidence:.1f}%) {cls_correct} | Seg IoU: {iou:.2%}', 
                     fontsize=15, fontweight='bold', y=0.98)
    else:
        # If no ground truth, show simplified version
        # TOP ROW - Input
        ax1 = plt.subplot(2, 3, 1)
        ax1.imshow(image, cmap='gray')
        ax1.set_title('Original Image', fontsize=14, fontweight='bold')
        ax1.axis('off')
        
        ax2 = plt.subplot(2, 3, 2)
        ax2.text(0.5, 0.5, 'No Ground Truth\nAvailable', 
                ha='center', va='center', fontsize=14)
        ax2.set_title('Ground Truth Mask', fontsize=14, fontweight='bold', color='red')
        ax2.axis('off')
        
        ax3 = plt.subplot(2, 3, 3)
        ax3.text(0.5, 0.5, 'No Ground Truth\nAvailable', 
                ha='center', va='center', fontsize=14)
        ax3.set_title('Ground Truth Overlay', fontsize=14, fontweight='bold', color='red')
        ax3.axis('off')
        
        # BOTTOM ROW - Predictions
        ax4 = plt.subplot(2, 3, 4)
        ax4.imshow(image, cmap='gray')
        ax4.set_title('Original Image', fontsize=14, fontweight='bold')
        ax4.axis('off')
        
        ax5 = plt.subplot(2, 3, 5)
        ax5.imshow(pred_mask_binary, cmap='hot')
        ax5.set_title('Predicted Mask', fontsize=14, fontweight='bold', color='green')
        ax5.axis('off')
        
        ax6 = plt.subplot(2, 3, 6)
        overlay_pred = create_overlay(image, pred_mask_binary, alpha=0.5, color_mask='green')
        ax6.imshow(overlay_pred)
        ax6.set_title('Prediction Overlay', fontsize=14, fontweight='bold', color='green')
        ax6.axis('off')
        
        cls_correct = "✓" if gt_class == predicted_class else "✗"
        fig.suptitle(f'GT: {gt_class} | Predicted: {predicted_class} ({cls_confidence:.1f}%) {cls_correct}', 
                     fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save
    save_path = FIGURES_DIR / f"{output_prefix}_{Path(image_path).stem}.png"
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"✓ Saved demonstration to {save_path}")
    
    # plt.show()
    plt.close()
    
    return pred_mask_resized


def demo_random_images(seg_model_path, cls_model_path, seg_model_type='unet', cls_model_type='densenet', num_samples=5):
    """
    Demonstration function to run BOTH segmentation AND classification inference
    This is what will be used during the demonstration part of assessment
    
    Args:
        seg_model_path: Path to trained segmentation model
        cls_model_path: Path to trained classification model  
        seg_model_type: 'unet' or 'attention_unet'
        cls_model_type: 'mobilenet', 'efficientnet', or 'densenet'
        num_samples: Number of random samples to demonstrate
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load segmentation model
    print(f"Loading {seg_model_type} segmentation model...")
    seg_model = load_model(seg_model_path, seg_model_type, device)
    
    # Load classification model
    print(f"Loading {cls_model_type} classification model...")
    cls_model = get_classifier(cls_model_type, num_classes=4, pretrained=False)
    checkpoint = torch.load(cls_model_path, map_location=device, weights_only=False)
    cls_model.load_state_dict(checkpoint['model_state_dict'])
    cls_model = cls_model.to(device)
    cls_model.eval()
    
    # Get list of test images
    test_images_dir = DATA_ROOT / 'segmentation_task' / 'test' / 'images'
    all_images = list(test_images_dir.glob('*.jpg'))
    
    # Select random images
    if len(all_images) < num_samples:
        num_samples = len(all_images)
    
    random.seed(RANDOM_SEED)
    selected_images = random.sample(all_images, num_samples)
    
    print(f"\nRunning inference on {num_samples} random images...")
    print("=" * 80)
    print("Format: 2×3 grid")
    print("  Top row: [Original | GT Mask | GT Overlay (red)]")
    print("  Bottom row: [Original | Predicted Mask | Prediction Overlay (green)]")
    print("=" * 80)
    
    for i, img_path in enumerate(selected_images, 1):
        print(f"\nImage {i}/{num_samples}: {img_path.name}")
        inference_on_image(img_path, seg_model, cls_model, device)
    
    print("\n" + "=" * 80)
    print("✓ Demonstration completed!")
    print(f"✓ All visualizations saved to {FIGURES_DIR}/")



def demo_on_provided_image(image_path, seg_model_path, cls_model_path, seg_model_type='unet', cls_model_type='densenet', output_prefix='demo'):
    """
    Run demonstration on a specific image provided by faculty
    This will be used during the actual demonstration
    
    Args:
        image_path: Path to the provided image
        seg_model_path: Path to trained segmentation model
        cls_model_path: Path to trained classification model
        seg_model_type: Model type ('unet' or 'attention_unet')
        cls_model_type: Classifier type ('densenet', 'efficientnet', 'mobilenet')
        output_prefix: Prefix for output filename (default: 'demo')
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load segmentation model
    print(f"Loading {seg_model_type} segmentation model...")
    seg_model = load_model(seg_model_path, seg_model_type, device)
    
    # Load classification model
    print(f"Loading {cls_model_type} classification model...")
    cls_model = get_classifier(cls_model_type, num_classes=4, pretrained=False)
    checkpoint = torch.load(cls_model_path, map_location=device, weights_only=False)
    cls_model.load_state_dict(checkpoint['model_state_dict'])
    cls_model = cls_model.to(device)
    cls_model.eval()
    
    # Run inference
    print(f"Running inference on {image_path}...")
    pred_mask = inference_on_image(image_path, seg_model, cls_model, device, output_prefix=output_prefix)
    
    return pred_mask


if __name__ == "__main__":
    # Example usage for demonstration
    print("=" * 80)
    print("BRISC 2025 - BRAIN TUMOR SEGMENTATION DEMONSTRATION")
    print("Student: Imtiaz Hossain (ID: 23101137)")
    print("=" * 80)
    
    # Paths to best models
    seg_model_path = MODELS_DIR / "unet_best.pth"
    cls_model_path = MODELS_DIR / "densenet_classifier_best.pth"
    
    if seg_model_path.exists() and cls_model_path.exists():
        # Run demonstration on random test images
        demo_random_images(
            seg_model_path, 
            cls_model_path,
            seg_model_type='unet',
            cls_model_type='densenet',
            num_samples=5
        )
    else:
        if not seg_model_path.exists():
            print(f"\nSegmentation model not found at {seg_model_path}")
        if not cls_model_path.exists():
            print(f"\nClassification model not found at {cls_model_path}")
        print("Please train the models first!")
        
        # Show example of what will happen during demonstration
        print("\nDuring the actual demonstration, the faculty will:")
        print("1. Provide random images")
        print("2. Run: demo_on_provided_image('path/to/image.jpg', ...)")
        print("3. The system will display:")
        print("   - Classification: Tumor Type (Confidence %)")
        print("   - Segmentation: 2×3 grid")
        print("     * Top row: Original, GT Mask, GT Overlay (red)")
        print("     * Bottom row: Original, Pred Mask, Pred Overlay (green)")
