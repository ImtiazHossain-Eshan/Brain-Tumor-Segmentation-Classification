"""
Quick demonstration script for faculty assessment using Attention U-Net
Usage: python quick_demo_attention_unet.py path/to/image.jpg
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
        print("BRISC 2025 - Attention U-Net Demonstration")
        print("Student: Imtiaz Hossain (ID: 23101137)")
        print("=" * 80)
        print("\nUsage: python quick_demo_attention_unet.py <path/to/image.jpg>")
        print("\nExample:")
        print("  python quick_demo_attention_unet.py brisc2025/segmentation_task/test/images/sample.jpg")
        print("\nThis will:")
        print("  1. Classify the tumor type (Glioma/Meningioma/Pituitary/No Tumor)")
        print("  2. Segment the tumor region using Attention U-Net")
        print("  3. Display 2×3 visualization with GT vs Prediction")
        print("\nModel Architecture:")
        print("  • Segmentation: Attention U-Net (with Attention Gates)")
        print("  • Classification: DenseNet-121")
        print("=" * 80)
        return
    
    # Get image path from command line
    image_path = sys.argv[1]
    
    if not Path(image_path).exists():
        print(f"Error: Image not found at {image_path}")
        return
    
    # Model paths
    seg_model_path = MODELS_DIR / "attention_unet_best.pth"
    cls_model_path = MODELS_DIR / "densenet_classifier_best.pth"
    
    if not seg_model_path.exists():
        print(f"Error: Attention U-Net model not found at {seg_model_path}")
        return
    
    if not cls_model_path.exists():
        print(f"Error: Classification model not found at {cls_model_path}")
        return
    
    print("=" * 80)
    print("BRISC 2025 - Brain Tumor Analysis (Attention U-Net)")
    print("Student: Imtiaz Hossain (ID: 23101137)")
    print("=" * 80)
    print(f"\nProcessing: {image_path}")
    print("\nLoading models...")
    print("  ✓ Attention U-Net Segmentation (82.29% Dice)")
    print("    - Architecture: U-Net + Attention Gates")
    print("    - Attention mechanism improves focus on tumor regions")
    print("  ✓ DenseNet-121 Classification (97.50% Accuracy)")
    print("\nRunning inference...\n")
    
    # Run demonstration with Attention U-Net
    demo_on_provided_image(
        image_path, 
        seg_model_path, 
        cls_model_path,
        seg_model_type='attention_unet',  # Use Attention U-Net instead of regular U-Net
        cls_model_type='densenet',
        output_prefix='demo_attention'  
    )
    
    print("\n" + "=" * 80)
    print("✓ Demonstration complete!")
    print("  Check figures/ folder for visualization")
    print("\nModel Comparison:")
    print("  • Regular U-Net:     83.10% Dice")
    print("  • Attention U-Net:   82.29% Dice")
    print("  Note: Results are comparable; Attention U-Net may perform better on difficult cases")
    print("=" * 80)

if __name__ == "__main__":
    main()
