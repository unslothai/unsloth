# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

import sys
import torch
import time
import platform
from typing import Callable


# -----------------------------------------------------------------------------
# 1. Standalone Kernels (Bypass all Unsloth/Transformers/Zoo dependencies)
# -----------------------------------------------------------------------------
def standalone_mps_swiglu_forward(e: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    """PyTorch-native SwiGLU forward pass for baseline comparison."""
    return torch.nn.functional.silu(e) * g


def benchmark_fn(fn: Callable, warmup: int = 15, iterations: int = 100) -> float:
    """Benchmark a function and return average latency in ms."""
    for _ in range(warmup):
        fn()

    if torch.backends.mps.is_available():
        torch.mps.synchronize()

    start = time.perf_counter()
    for _ in range(iterations):
        fn()

    if torch.backends.mps.is_available():
        torch.mps.synchronize()

    return (time.perf_counter() - start) / iterations * 1000  # ms


def run_benchmarks():
    if not torch.backends.mps.is_available():
        print("❌ MPS backend not available (Required for M4 Max)")
        sys.exit(1)

    print("=" * 70)
    print("SwiGLU Activation Standalone Benchmark (M4 Max Baseline)")
    print("=" * 70)
    print(f"Platform: {platform.platform()}")
    print(f"PyTorch: {torch.__version__}")
    print()

    # Standard Activation sizes for Llama-3 (8B) and deep context
    configs = [
        (1, 2048, 8192),  # Small batch, typical seq
        (4, 512, 14336),  # Llama-3 8B specific hidden_dimension
        (1, 16384, 8192),  # Large seq (Long context)
    ]

    dtype = torch.float16  # Standard for Apple Silicon training
    print(f"Testing with dtype={dtype}")
    print()

    for batch, seq, dim in configs:
        elements = batch * seq * dim
        print(
            f"Config: bsz={batch}, seq={seq}, dim={dim} ({elements/1e6:.2f}M elements)"
        )
        print("-" * 60)

        # SwiGLU inputs: e (Gate), g (Up)
        e = torch.randn(batch, seq, dim, device = "mps", dtype = dtype)
        g = torch.randn(batch, seq, dim, device = "mps", dtype = dtype)

        # 1. PyTorch Native Reference (In-place/Optimized where possible)
        # silu(e) * g
        t_ref = benchmark_fn(lambda: torch.nn.functional.silu(e) * g)
        print(f"  PyTorch Native: {t_ref:.3f} ms")

        # 2. Standalone Fallback Implementation
        t_fallback = benchmark_fn(lambda: standalone_mps_swiglu_forward(e, g))
        print(f"  MPS Fallback:   {t_fallback:.3f} ms")

        # Calculate Throughput (GB/s)
        # Bytes read: e (2 bytes/elem) + g (2 bytes/elem). Path: 2 * numel * 2
        # Bytes written: result (2 bytes/elem). Path: 1 * numel * 2
        # Total: 3 * numel * 2
        size_bytes = 3 * elements * 2  # 2 for float16
        throughput = (size_bytes / 1e9) / (t_ref / 1000)

        print(f"  Throughput:     {throughput:.2f} GB/s")
        print()

    print(
        "Done. Use these baseline numbers to verify if custom Metal kernels improve bandwidth."
    )


if __name__ == "__main__":
    run_benchmarks()
