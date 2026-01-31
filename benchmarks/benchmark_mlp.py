"""
MLP Block Benchmark Suite (Fused vs Native)
Tests the "Maximum Fusion" strategy: Up/Gate -> SwiGLU -> Down in a single MLX chain.
"""

import sys
import time
import platform
import torch
import torch.nn.functional as F
import mlx.core as mx
import numpy as np

# Add parent directory for imports
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from unsloth.kernels.mps.fast_lora import mps_apply_lora_mlp_swiglu
from unsloth.kernels.mlx.bridge import torch_to_mlx, mlx_to_torch, mlx_context


def benchmark_fn(fn, warmup = 10, iterations = 50):
    for _ in range(warmup):
        fn()
    torch.mps.synchronize()
    start = time.perf_counter()
    for _ in range(iterations):
        fn()
    torch.mps.synchronize()
    return (time.perf_counter() - start) / iterations * 1000


def run_benchmark():
    print("=" * 60)
    print("Unsloth MLP Fusion Benchmark")
    print("=" * 60)

    # Configs: Llama-3 8B style
    # Batch, Seq, Hidden, Inter size
    configs = [
        (1, 1, 4096, 14336, "Llama-3 8B Decoding (GEMV)"),
        (1, 2048, 4096, 14336, "Llama-3 8B Inference"),
        (4, 512, 4096, 14336, "Llama-3 8B Training"),
        (1, 8192, 4096, 14336, "Long Context 8K"),
    ]

    for b, s, h_dim, i_dim, name in configs:
        print(f"\n📊 {name} ({b}x{s} tokens)")
        print(f"   Hidden: {h_dim}, Intermediate: {i_dim}")

        # Inputs
        X = torch.randn(b, s, h_dim, device = "mps", dtype = torch.float16)

        # Weights (Transposed for linear layer logic: Out x In)
        # unsloth mps_matmul_lora uses X @ W.T, so W shape is [Out, In]
        upW = torch.randn(i_dim, h_dim, device = "mps", dtype = torch.float16)
        gateW = torch.randn(i_dim, h_dim, device = "mps", dtype = torch.float16)
        downW = torch.randn(h_dim, i_dim, device = "mps", dtype = torch.float16)

        # LoRA placeholders (None for now as we test base MLP speed)
        # But our fusion supports them!

        # 1. PyTorch Native Baseline
        def torch_mlp():
            # Standard Llama MLP
            up = F.linear(X, upW)
            gate = F.linear(X, gateW)
            act = F.silu(gate) * up
            out = F.linear(act, downW)
            return out

        t_torch = benchmark_fn(torch_mlp)
        print(f"   PyTorch Native:   {t_torch:7.3f} ms")

        # 2. Unsloth Fused MLP (Simulated Chain)
        # We manually attach _mlx_cache to X to trigger the "Sandwich" path
        X_chained = X.clone()
        with mlx_context():
            X_chained._mlx_cache = torch_to_mlx(X)

        def unsloth_mlp():
            return mps_apply_lora_mlp_swiglu(
                X_chained,
                gateW,
                None,
                None,
                None,
                1.0,  # Gate
                upW,
                None,
                None,
                None,
                1.0,  # Up
                downW,
                None,
                None,
                None,
                1.0,  # Down
            )

        t_unsloth = benchmark_fn(unsloth_mlp)
        speedup = t_torch / t_unsloth
        print(f"   Unsloth Fused:    {t_unsloth:7.3f} ms | {speedup:.2f}x Speedup")

        if speedup > 3.0:
             print("   🚀 Massive Speedup Detected!")
        elif speedup > 1.5:
             print("   ✅ Significant Improvement")
             
        # 4. Unsloth Fused (Merged Gate/Up)
        # Idea: X @ [Gate, Up] is faster than X@Gate + X@Up
        with mlx_context():
            W_merged = mx.concatenate([torch_to_mlx(gateW), torch_to_mlx(upW)], axis=0) # [2*I, H]
        
        # We need a custom merged MLP function for the benchmark
        from unsloth.kernels.metal.swiglu import mlx_swiglu_forward
        
        def unsloth_merged_mlp():
            with mlx_context():
                X_mlx = _get_mlx_cached(X_chained)
                # Merged Projection
                # W_merged is (2*mid, hidden). X is (..., hidden).
                # X @ W.T -> (..., 2*mid)
                eg_mixed = X_mlx @ W_merged.T
                
                # Split (MLX split is cheap? Or strided view?)
                # mid is dim/2 of the last axis
                # Actually eg_mixed shape last dim is 2*intermediate.
                # We need to split into e and g.
                split_idx = eg_mixed.shape[-1] // 2
                e_mlx = eg_mixed[..., :split_idx]
                g_mlx = eg_mixed[..., split_idx:]
                
                h_mlx = mlx_swiglu_forward(e_mlx, g_mlx)
                
                # Down proj
                W_down_mlx = _get_mlx_cached(downW)
                out_mlx = h_mlx @ W_down_mlx.T
                return out_mlx
                
        # Warmup for merged
        for _ in range(3): unsloth_merged_mlp()
        
        t_merged = benchmark_fn(unsloth_merged_mlp)
        speedup_merged = t_torch / t_merged
        print(f"   Unsloth Merged:   {t_merged:7.3f} ms | {speedup_merged:.2f}x Speedup (Merged Gate/Up)")

        # 5. Unsloth Compiled
        # Compile the merged function
        compiled_fn = mx.compile(unsloth_merged_mlp)
        
        # Warmup compiled
        for _ in range(5): compiled_fn()
        
        t_compiled = benchmark_fn(compiled_fn)
        speedup_compiled = t_torch / t_compiled
        print(f"   Unsloth Compiled: {t_compiled:7.3f} ms | {speedup_compiled:.2f}x Speedup (mx.compile)")
        elif speedup < 1.0:
             print("   ⚠️ Slower (Check overhead)")

        # 3. Unsloth Fused MLP (4-bit Quantized)
        # Quantize weights on the fly
        from unsloth.kernels.mlx.quantization import quantize_4bit
        
        # We need to manually inject quantized weights into the cache
        # The benchmark function mps_apply_lora_mlp_swiglu takes raw tensors
        # and calls _mlx_matmul, which calls _get_mlx_cached.
        # We need to pre-populate _mlx_cache on the weight tensors with MLXQuantizedWeight objects.
        
        # Clone for isolation
        upW_q = upW.clone()
        gateW_q = gateW.clone()
        downW_q = downW.clone()
        
        with mlx_context():
            upW_q._mlx_cache = quantize_4bit(upW)
            gateW_q._mlx_cache = quantize_4bit(gateW)
            downW_q._mlx_cache = quantize_4bit(downW)
            
        def unsloth_mlp_4bit():
            return mps_apply_lora_mlp_swiglu(
                X_chained,
                gateW_q, None, None, None, 1.0,
                upW_q, None, None, None, 1.0,
                downW_q, None, None, None, 1.0,
            )

        t_unsloth_4bit = benchmark_fn(unsloth_mlp_4bit)
        speedup_4bit = t_torch / t_unsloth_4bit
        print(f"   Unsloth 4-bit:    {t_unsloth_4bit:7.3f} ms | {speedup_4bit:.2f}x Speedup (vs FP16 Torch)")

    print("\n" + "=" * 60)
    print("Correctness Verification")
    print("=" * 60)

    # Test on small batch
    b, s, h, i = 2, 128, 1024, 4096
    X_small = torch.randn(b, s, h, device = "mps", dtype = torch.float16)
    upW_small = torch.randn(i, h, device = "mps", dtype = torch.float16)
    gateW_small = torch.randn(i, h, device = "mps", dtype = torch.float16)
    downW_small = torch.randn(h, i, device = "mps", dtype = torch.float16)

    # Torch Reference
    up = F.linear(X_small, upW_small)
    gate = F.linear(X_small, gateW_small)
    act = F.silu(gate) * up
    ref = F.linear(act, downW_small)

    # Unsloth Fused
    X_chained = X_small.clone()
    with mlx_context():
        X_chained._mlx_cache = torch_to_mlx(X_small)

    out_unsloth = mps_apply_lora_mlp_swiglu(
        X_chained,
        gateW_small,
        None,
        None,
        None,
        1.0,
        upW_small,
        None,
        None,
        None,
        1.0,
        downW_small,
        None,
        None,
        None,
        1.0,
    )

    diff = (out_unsloth - ref).abs().max()
    print(f"MLP Block Output Diff: {diff:.2e}")
    if diff < 1e-2:
        print("✅ Correctness Passed")
    else:
        print("❌ Correctness Failed")


if __name__ == "__main__":
    run_benchmark()
