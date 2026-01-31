# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""
GEGLU Kernel Comprehensive Benchmark Suite
"""

import argparse
import sys
import time
import platform
from typing import Callable

# Add parent directory to sys.path to allow absolute imports of unsloth
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

IS_DARWIN = platform.system() == "Darwin"
if not IS_DARWIN:
    print("❌ This benchmark requires macOS with Apple Silicon")
    sys.exit(1)

import torch
import numpy as np

if not torch.backends.mps.is_available():
    print("❌ MPS backend not available")
    sys.exit(1)

import mlx.core as mx
import mlx.nn as nn_mlx  # Separate module for neural network ops


def print_header():
    """Print system information header."""
    print("=" * 75)
    print("GEGLU Kernel Benchmark Suite")
    print("=" * 75)
    print(f"Platform:  {platform.platform()}")
    print(f"Processor: {platform.processor()}")
    print(f"PyTorch:   {torch.__version__}")
    print(f"MLX:       {mx.__version__}")
    print()


# ==============================================================================
# Utility Functions
# ==============================================================================


def benchmark_fn(fn: Callable, warmup: int = 10, iterations: int = 50) -> float:
    """Benchmark a function and return average latency in ms."""
    for _ in range(warmup):
        result = fn()

    if isinstance(result, (mx.array, list)):
        mx.eval(result)
    torch.mps.synchronize()

    start = time.perf_counter()
    for _ in range(iterations):
        result = fn()
        if isinstance(result, (mx.array, list)):
            mx.eval(result)

    torch.mps.synchronize()
    return (time.perf_counter() - start) / iterations * 1000


def calculate_throughput(
    elements: int, latency_ms: float, dtype_size: int = 2
) -> float:
    """Calculate memory throughput in GB/s. GEGLU: 3 tensors."""
    bytes_total = 3 * elements * dtype_size
    return (bytes_total / 1e9) / (latency_ms / 1000.0)


# ==============================================================================
# Performance Benchmark
# ==============================================================================


def run_performance_benchmark():
    """Run performance benchmarks comparing implementations."""
    print("=" * 75)
    print("PERFORMANCE BENCHMARK")
    print("=" * 75)
    print()

    configs = [
        (1, 2048, 8192, "Llama-3 8B (inference)"),
        (4, 512, 14336, "Llama-3 8B (training)"),
    ]

    import importlib.util
    import os

    kernel_path = os.path.join(
        os.path.dirname(__file__), "..", "unsloth", "kernels", "metal", "geglu.py"
    )
    spec = importlib.util.spec_from_file_location("metal_geglu", kernel_path)
    metal_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(metal_module)

    results = []

    for batch, seq, dim, desc in configs:
        elements = batch * seq * dim
        print(f"📊 {desc} (Exact)")
        print(f"   Shape: ({batch}, {seq}, {dim}) = {elements / 1e6:.2f}M elements")

        # MLX Composed (Baseline)
        e_mlx = mx.random.normal((batch, seq, dim)).astype(mx.float16)
        g_mlx = mx.random.normal((batch, seq, dim)).astype(mx.float16)
        mx.eval(e_mlx)
        mx.eval(g_mlx)

        def mlx_composed():
            return nn_mlx.gelu(e_mlx) * g_mlx

        t_mlx = benchmark_fn(mlx_composed)
        tp_mlx = calculate_throughput(elements, t_mlx)
        print(f"   MLX Composed:       {t_mlx:7.3f} ms | {tp_mlx:7.2f} GB/s")

        # Fused Metal (pure MLX - fair comparison, no conversion overhead)
        def mlx_fused():
            return metal_module.mlx_geglu_exact_forward(e_mlx, g_mlx)

        t_fused = benchmark_fn(mlx_fused)
        tp_fused = calculate_throughput(elements, t_fused)
        speedup = t_mlx / t_fused
        print(f"   Fused Metal:        {t_fused:7.3f} ms | {tp_fused:7.2f} GB/s  ({speedup:.2f}x)")
        print()



def run_correctness_tests():
    """Verify numerical correctness."""
    print("=" * 75)
    print("CORRECTNESS VERIFICATION")
    print("=" * 75)
    print()

    import importlib.util
    import os

    kernel_path = os.path.join(
        os.path.dirname(__file__), "..", "unsloth", "kernels", "metal", "geglu.py"
    )
    spec = importlib.util.spec_from_file_location("metal_geglu", kernel_path)
    metal_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(metal_module)

    batch, seq, dim = 2, 512, 1024
    torch.manual_seed(42)
    e = torch.randn(batch, seq, dim, dtype = torch.float32)
    g = torch.randn(batch, seq, dim, dtype = torch.float32)

    # Convert to MLX
    e_mlx = mx.array(e.numpy()).astype(mx.float16)
    g_mlx = mx.array(g.numpy()).astype(mx.float16)

    # Exact
    print("Testing GEGLU Exact...")
    ref = torch.nn.functional.gelu(e, approximate = "none") * g
    out = metal_module.mlx_geglu_exact_forward(e_mlx, g_mlx)
    mx.eval(out)
    diff = np.abs(np.array(out).astype(np.float32) - ref.numpy())
    print(
        f"  Exact Grad parity: max_diff={diff.max():.2e} {'✅' if diff.max() < 1e-2 else '❌'}"
    )

    # Approx
    print("Testing GEGLU Approx...")
    ref = torch.nn.functional.gelu(e, approximate = "tanh") * g
    out = metal_module.mlx_geglu_approx_forward(e_mlx, g_mlx)
    mx.eval(out)
    diff = np.abs(np.array(out).astype(np.float32) - ref.numpy())
    print(
        f"  Approx Grad parity: max_diff={diff.max():.2e} {'✅' if diff.max() < 1e-2 else '❌'}"
    )
    print()



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--perf", action = "store_true")
    parser.add_argument("--correctness", action = "store_true")
    args = parser.parse_args()

    print_header()
    if not args.perf and not args.correctness:
        run_correctness_tests()
        run_performance_benchmark()
    else:
        if args.correctness:
            run_correctness_tests()
        if args.perf:
            run_performance_benchmark()


if __name__ == "__main__":
    main()
