"""
Verify trained model quality
"""
import torch
import json
from pathlib import Path

models_dir = Path("models")
results_dir = Path("results")

print("=" * 80)
print("TRAINING COMPLETION CHECK")
print("=" * 80)

# Check models
print("\n📦 TRAINED MODELS:")
print("-" * 80)

unet_model = models_dir / "unet_best.pth"
attn_model = models_dir / "attention_unet_best.pth"

if unet_model.exists():
    size_mb = unet_model.stat().st_size / (1024 * 1024)
    print(f"✅ U-Net: {size_mb:.1f} MB")
    
    # Try to load and check
    try:
        checkpoint = torch.load(unet_model, map_location='cpu', weights_only=False)
        if 'history' in checkpoint:
            history = checkpoint['history']
            if 'val_dice' in history and len(history['val_dice']) > 0:
                best_dice = max(history['val_dice'])
                epochs_trained = len(history['val_dice'])
                print(f"   Epochs trained: {epochs_trained}")
                print(f"   Best Val Dice: {best_dice:.4f}")
        else:
            print("   ⚠️  No training history found")
    except Exception as e:
        print(f"   ⚠️  Could not load checkpoint: {str(e)[:50]}")
else:
    print("❌ U-Net model not found")

if attn_model.exists():
    size_mb = attn_model.stat().st_size / (1024 * 1024)
    print(f"✅ Attention U-Net: {size_mb:.1f} MB")
else:
    print("❌ Attention U-Net model not found")

# Check results
print("\n📊 RESULTS FILES:")
print("-" * 80)

unet_results = results_dir / "unet_results.json"
attn_results = results_dir / "attention_unet_results.json"

if unet_results.exists():
    with open(unet_results) as f:
        data = json.load(f)
        print("✅ U-Net results available")
        if 'test_metrics' in data:
            print(f"   Test Dice: {data['test_metrics'].get('dice_coefficient', 'N/A')}")
            print(f"   Test mIoU: {data['test_metrics'].get('mIoU', 'N/A')}")
else:
    print("❌ U-Net results not found")

if attn_results.exists():
    print("✅ Attention U-Net results available")
else:
    print("❌ Attention U-Net results not found")

# Summary
print("\n" + "=" * 80)
print("SUMMARY:")
print("=" * 80)

if unet_model.exists() and attn_model.exists():
    print("✅ SUCCESS: Both models trained!")
elif unet_model.exists():
    print("⚠️  PARTIAL: Only U-Net completed")
    print("\nThe training encountered an error during saving/testing.")
    print("The U-Net model IS trained and can be used!")
    print("\nNext steps:")
    print("  1. Test the U-Net model with demo.py")
    print("  2. Optionally retrain Attention U-Net separately")
else:
    print("❌ FAILED: No models found")

print("=" * 80)
