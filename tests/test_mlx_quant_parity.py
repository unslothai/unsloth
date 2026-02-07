# Apply Mac compatibility patches BEFORE importing unsloth
import platform
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if platform.system() == "Darwin":
    from patcher import patch_for_mac
    patch_for_mac()

import torch
import mlx.core as mx
import numpy as np
from unsloth.kernels.mlx.bridge import torch_to_mlx, mlx_to_torch
from unsloth.kernels.mlx.quantization import quantize_4bit
from unsloth.kernels.mlx.fast_lora import quantized_matmul

def test_quant_parity():
    print("Testing MLX 4-bit Quantization Parity...")
    
    # 1. Setup
    torch.manual_seed(42)
    device = "cpu" # Test can run on CPU for math verification if mlx supports it, but better on Mac
    out_features = 4096
    in_features = 4096
    batch_size = 1
    
    # Create FP16 weights and inputs
    W = torch.randn(out_features, in_features, dtype=torch.float16)
    X = torch.randn(batch_size, in_features, dtype=torch.float16)
    
    # 2. Reference (PyTorch FP16)
    expected = torch.matmul(X, W.T)
    
    # 3. Quantize using MLX
    W_q = quantize_4bit(W, group_size=64)
    
    # 4. MLX Forward
    X_mlx = torch_to_mlx(X)
    # Note: quantized_matmul handles MLXQuantizedWeight objects
    actual_mlx = quantized_matmul(X_mlx, W_q)
    actual = mlx_to_torch(actual_mlx)
    
    # 5. Measure Error
    # 4-bit quantization will have some error, so we expect some difference
    diff = (expected - actual).abs()
    mean_diff = diff.mean().item()
    max_diff = diff.max().item()
    
    print(f"Mean Abs Error (FP16 vs 4-bit): {mean_diff:.6f}")
    print(f"Max Abs Error: {max_diff:.6f}")
    
    # Standard check for 4-bit: mean error should be small
    if mean_diff < 0.05:
        print("✅ Parity Check Passed (within quantization noise)")
    else:
        print("❌ Parity Check Failed (Error too high)")

if __name__ == "__main__":
    try:
        test_quant_parity()
    except Exception as e:
        print(f"Error during test: {e}")
