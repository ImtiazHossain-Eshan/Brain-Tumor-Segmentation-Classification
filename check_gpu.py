"""
Check GPU/CUDA availability for training
"""

import torch
import sys

print("=" * 80)
print("GPU/CUDA Availability Check")
print("=" * 80)

# Check CUDA availability
cuda_available = torch.cuda.is_available()
print(f"\nCUDA Available: {cuda_available}")

if cuda_available:
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"\nNumber of GPUs: {torch.cuda.device_count()}")
    
    for i in range(torch.cuda.device_count()):
        print(f"\nGPU {i}:")
        print(f"  Name: {torch.cuda.get_device_name(i)}")
        print(f"  Memory Total: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")
        print(f"  Memory Allocated: {torch.cuda.memory_allocated(i) / 1e9:.2f} GB")
        print(f"  Memory Cached: {torch.cuda.memory_reserved(i) / 1e9:.2f} GB")
    
    print("\n✅ GPU is available! Training will be much faster (~2-4 hours per model)")
    print("   The training scripts will automatically use GPU.")
    
else:
    print("\n❌ No GPU detected. Training will run on CPU.")
    print("   This will be VERY SLOW (~10-20 hours per model)")
    print("\nOptions:")
    print("  1. Install CUDA-enabled PyTorch:")
    print("     pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
    print("  2. Use Google Colab with free GPU")
    print("  3. Reduce epochs for testing (set EPOCHS=10 in config.py)")

# Test tensor creation on available device
device = torch.device('cuda' if cuda_available else 'cpu')
print(f"\nDefault device: {device}")

# Quick performance test
print("\n" + "=" * 80)
print("Quick Performance Test")
print("=" * 80)

import time

# Create random tensors
size = 1000
x = torch.randn(size, size)
y = torch.randn(size, size)

# CPU test
start = time.time()
z_cpu = torch.matmul(x, y)
cpu_time = time.time() - start
print(f"CPU Matrix Multiplication ({size}x{size}): {cpu_time:.4f} seconds")

if cuda_available:
    # GPU test
    x_gpu = x.to('cuda')
    y_gpu = y.to('cuda')
    torch.cuda.synchronize()
    
    start = time.time()
    z_gpu = torch.matmul(x_gpu, y_gpu)
    torch.cuda.synchronize()
    gpu_time = time.time() - start
    
    print(f"GPU Matrix Multiplication ({size}x{size}): {gpu_time:.4f} seconds")
    print(f"\n⚡ GPU Speedup: {cpu_time/gpu_time:.2f}x faster than CPU")

print("\n" + "=" * 80)
