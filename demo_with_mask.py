"""
Enhanced demonstration script - supports optional mask path
Usage: 
  python demo_with_mask.py <image_path>
  python demo_with_mask.py <image_path> <mask_path>
Author: Imtiaz Hossain (ID: 23101137)
"""

import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from demo import demo_on_provided_image
from config import MODELS_DIR
import shutil
import tempfile

def main():
    if len(sys.argv) < 2:
        print("=" * 80)
        print("BRISC 2025 - Enhanced Demonstration")
        print("Student: Imtiaz Hossain (ID: 23101137)")
        print("=" * 80)
        print("\nUsage:")
        print("  python demo_with_mask.py <image_path>")
        print("  python demo_with_mask.py <image_path> <mask_path>")
        print("\nExamples:")
        print("  # Auto-find mask:")
        print("  python demo_with_mask.py brain_scan.jpg")
        print("")
        print("  # Explicit mask path:")
        print("  python demo_with_mask.py brain_scan.jpg tumor_mask.png")
        print("=" * 80)
        return
    
    # Get paths from command line
    image_path = sys.argv[1]
    mask_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    if not Path(image_path).exists():
        print(f"Error: Image not found at {image_path}")
        return
    
    # If explicit mask provided, temporarily copy it to expected location
    temp_mask = None
    if mask_path and Path(mask_path).exists():
        print(f"Using provided mask: {mask_path}")
        
        # Create expected mask path
        img_path_obj = Path(image_path)
        expected_mask_path = str(img_path_obj).replace('/images/', '/masks/').replace('\\images\\', '\\masks\\')
        expected_mask_path = expected_mask_path.replace('.jpg', '.png').replace('.jpeg', '.png')
        
        # Check if source and destination are the same
        if Path(mask_path).resolve() == Path(expected_mask_path).resolve():
            print(f"→ Mask is already in correct location!")
        else:
            # Copy mask to expected location temporarily
            expected_mask_dir = Path(expected_mask_path).parent
            expected_mask_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(mask_path, expected_mask_path)
            temp_mask = expected_mask_path
            print(f"→ Copied mask to: {expected_mask_path}")
    elif mask_path:
        print(f"Warning: Mask not found at {mask_path}")
        print("Proceeding without ground truth mask...")
    
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
    print(f"\n📁 Image: {image_path}")
    if mask_path:
        print(f"📁 Mask:  {mask_path}")
    print("\n🤖 Loading models...")
    print("  ✓ U-Net Segmentation (83.10% Dice)")
    print("  ✓ DenseNet-121 Classification (97.50% Accuracy)")
    print("\n⚙️  Running inference...\n")
    
    try:
        # Run demonstration
        demo_on_provided_image(image_path, seg_model_path, cls_model_path)
        
        print("\n" + "=" * 80)
        print("✅ Demonstration complete!")
        print("📊 Check figures/ folder for visualization")
        print("=" * 80)
    finally:
        # Cleanup temporary mask if created
        if temp_mask and Path(temp_mask).exists():
            try:
                Path(temp_mask).unlink()
                print(f"\n🧹 Cleaned up temporary mask")
            except:
                pass

if __name__ == "__main__":
    main()
