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
        elif speedup < 1.0:
            print("   ⚠️ Slower (Check overhead)")


if __name__ == "__main__":
    run_benchmark()
