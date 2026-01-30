# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""
SwiGLU Correctness Test: Verify fused kernel matches PyTorch reference.

Tests numerical accuracy of:
1. Fused Metal kernel vs PyTorch reference (silu(e) * g)
2. MLX composed vs PyTorch reference
"""

import sys
import platform

print("=" * 75)
print("SwiGLU Correctness Verification")
print("=" * 75)

IS_DARWIN = platform.system() == "Darwin"
if not IS_DARWIN:
    print("❌ Requires macOS with Apple Silicon")
    sys.exit(1)

import torch
import numpy as np

if not torch.backends.mps.is_available():
    print("❌ MPS not available")
    sys.exit(1)

import mlx.core as mx

print(f"PyTorch: {torch.__version__}")
print(f"MLX:     {mx.__version__}")
print()


def pytorch_swiglu_reference(e: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    """Ground truth SwiGLU: silu(e) * g"""
    return torch.nn.functional.silu(e) * g


def mlx_swiglu_composed(e: mx.array, g: mx.array) -> mx.array:
    """MLX composed SwiGLU"""
    return mx.sigmoid(e) * e * g


def get_fused_kernel():
    """Get the fused Metal kernel"""
    kernel_source = '''
        uint gid = thread_position_in_grid.x;
        uint n = n_ptr[0];
        if (gid >= n) return;
        
        float e_val = float(e[gid]);
        float g_val = float(g[gid]);
        
        float sigmoid_e = 1.0f / (1.0f + exp(-e_val));
        float silu_e = e_val * sigmoid_e;
        
        h[gid] = half(silu_e * g_val);
    '''
    return mx.fast.metal_kernel(
        name="swiglu_fused_test",
        input_names=["e", "g", "n_ptr"],
        output_names=["h"],
        source=kernel_source,
    )


def run_correctness_tests():
    """Run comprehensive correctness tests."""
    
    test_shapes = [
        (1, 128, 256, "Small"),
        (2, 512, 1024, "Medium"),
        (4, 2048, 4096, "Large (LLM-sized)"),
    ]
    
    # Get fused kernel
    fused_kernel = get_fused_kernel()
    
    all_passed = True
    
    for batch, seq, dim, name in test_shapes:
        print(f"Testing: {name} - shape ({batch}, {seq}, {dim})")
        print("-" * 50)
        
        # Create test data (use same random seed for reproducibility)
        torch.manual_seed(42)
        e_torch = torch.randn(batch, seq, dim, dtype=torch.float32)
        g_torch = torch.randn(batch, seq, dim, dtype=torch.float32)
        
        # =====================================================================
        # 1. PyTorch Reference (ground truth - float32)
        # =====================================================================
        ref_output = pytorch_swiglu_reference(e_torch, g_torch)
        
        # =====================================================================
        # 2. MLX Composed
        # =====================================================================
        e_mlx = mx.array(e_torch.numpy())
        g_mlx = mx.array(g_torch.numpy())
        
        mlx_output = mlx_swiglu_composed(e_mlx, g_mlx)
        mx.eval(mlx_output)
        mlx_output_np = np.array(mlx_output)
        
        # Compare
        mlx_diff = np.abs(mlx_output_np - ref_output.numpy())
        mlx_max_diff = mlx_diff.max()
        mlx_mean_diff = mlx_diff.mean()
        mlx_pass = mlx_max_diff < 1e-5
        
        print(f"  MLX Composed vs PyTorch:")
        print(f"    Max diff:  {mlx_max_diff:.2e} {'✅' if mlx_pass else '❌'}")
        print(f"    Mean diff: {mlx_mean_diff:.2e}")
        
        # =====================================================================
        # 3. Fused Metal Kernel (float16 - will have more error)
        # =====================================================================
        e_mlx_f16 = e_mlx.astype(mx.float16).flatten()
        g_mlx_f16 = g_mlx.astype(mx.float16).flatten()
        n_elements = batch * seq * dim
        n_arr = mx.array([n_elements], dtype=mx.uint32)
        
        fused_output = fused_kernel(
            inputs=[e_mlx_f16, g_mlx_f16, n_arr],
            output_shapes=[(n_elements,)],
            output_dtypes=[mx.float16],
            grid=(n_elements, 1, 1),
            threadgroup=(min(256, n_elements), 1, 1),
        )
        mx.eval(fused_output[0])
        fused_output_np = np.array(fused_output[0]).reshape(batch, seq, dim)
        
        # Compare against float16 reference (fair comparison)
        ref_f16 = pytorch_swiglu_reference(e_torch.half(), g_torch.half()).float().numpy()
        
        fused_diff = np.abs(fused_output_np.astype(np.float32) - ref_f16)
        fused_max_diff = fused_diff.max()
        fused_mean_diff = fused_diff.mean()
        # FP16 tolerance is higher due to precision limits
        fused_pass = fused_max_diff < 1e-2  # ~1% max error acceptable for fp16
        
        print(f"  Fused Metal (fp16) vs PyTorch (fp16):")
        print(f"    Max diff:  {fused_max_diff:.2e} {'✅' if fused_pass else '❌'}")
        print(f"    Mean diff: {fused_mean_diff:.2e}")
        
        if not (mlx_pass and fused_pass):
            all_passed = False
        
        print()
    
    # Summary
    print("=" * 75)
    if all_passed:
        print("✅ ALL CORRECTNESS TESTS PASSED")
        print()
        print("The fused Metal kernel produces numerically correct results.")
        print("FP16 has slightly higher error which is expected due to precision limits.")
    else:
        print("❌ SOME TESTS FAILED - Check the kernel implementation!")
    print("=" * 75)


if __name__ == "__main__":
    run_correctness_tests()
