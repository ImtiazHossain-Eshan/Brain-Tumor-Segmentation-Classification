"""
Quick demonstration script for faculty assessment
Usage: python quick_demo.py path/to/image.jpg
Author: Imtiaz Hossain (ID: 23101137)
"""

import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from demo import demo_on_provided_image
from config import MODELS_DIR

def main():
    if len(sys.argv) < 2:
        print("=" * 80)
        print("BRISC 2025 - Quick Demonstration")
        print("Student: Imtiaz Hossain (ID: 23101137)")
        print("=" * 80)
        print("\nUsage: python quick_demo.py <path/to/image.jpg>")
        print("\nExample:")
        print("  python quick_demo.py brisc2025/segmentation_task/test/images/sample.jpg")
        print("\nThis will:")
        print("  1. Classify the tumor type (Glioma/Meningioma/Pituitary/No Tumor)")
        print("  2. Segment the tumor region")
        print("  3. Display 2×3 visualization with GT vs Prediction")
        print("=" * 80)
        return
    
    # Get image path from command line
    image_path = sys.argv[1]
    
    if not Path(image_path).exists():
        print(f"Error: Image not found at {image_path}")
        return
    
    # Model paths
    seg_model_path = MODELS_DIR / "unet_best.pth"
    cls_model_path = MODELS_DIR / "densenet_classifier_best.pth"
    
    if not seg_model_path.exists():
        print(f"Error: Segmentation model not found at {seg_model_path}")
        return
    
    if not cls_model_path.exists():
        print(f"Error: Classification model not found at {cls_model_path}")
        return
    
    print("=" * 80)
    print("BRISC 2025 - Brain Tumor Analysis")
    print("Student: Imtiaz Hossain (ID: 23101137)")
    print("=" * 80)
    print(f"\nProcessing: {image_path}")
    print("\nLoading models...")
    print("  ✓ U-Net Segmentation (83.10% Dice)")
    print("  ✓ DenseNet-121 Classification (97.50% Accuracy)")
    print("\nRunning inference...\n")
    
    # Run demonstration
    demo_on_provided_image(image_path, seg_model_path, cls_model_path)
    
    print("\n" + "=" * 80)
    print("✓ Demonstration complete!")
    print("  Check figures/ folder for visualization")
    print("=" * 80)

if __name__ == "__main__":
    main()
