# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""
SwiGLU Kernel Comprehensive Benchmark Suite

This is the single source of truth for SwiGLU kernel testing on Apple Silicon.
Includes both performance benchmarks and correctness verification.

Usage:
    python benchmarks/benchmark_swiglu.py              # Run all tests
    python benchmarks/benchmark_swiglu.py --perf       # Performance only
    python benchmarks/benchmark_swiglu.py --correctness # Correctness only
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


def print_header():
    """Print system information header."""
    print("=" * 75)
    print("SwiGLU Kernel Benchmark Suite")
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

    # Sync depending on result type
    if isinstance(result, mx.array):
        mx.eval(result)
    torch.mps.synchronize()

    start = time.perf_counter()
    for _ in range(iterations):
        result = fn()
        if isinstance(result, mx.array):
            mx.eval(result)

    torch.mps.synchronize()
    return (time.perf_counter() - start) / iterations * 1000


def calculate_throughput(
    elements: int, latency_ms: float, dtype_size: int = 2
) -> float:
    """Calculate memory throughput in GB/s. SwiGLU: 3 tensors (read e, g; write h)."""
    bytes_total = 3 * elements * dtype_size
    return (bytes_total / 1e9) / (latency_ms / 1000.0)


def mlx_swiglu_composed(e: mx.array, g: mx.array) -> mx.array:
    """MLX composed SwiGLU: silu(e) * g - NOT fused."""
    return mx.sigmoid(e) * e * g


def pytorch_swiglu_reference(e: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    """Ground truth SwiGLU: silu(e) * g"""
    return torch.nn.functional.silu(e) * g


def get_fused_kernel():
    """Create the fused Metal SwiGLU kernel."""
    kernel_source = """
        uint gid = thread_position_in_grid.x;
        uint n = n_ptr[0];
        if (gid >= n) return;
        
        float e_val = float(e[gid]);
        float g_val = float(g[gid]);
        
        float sigmoid_e = 1.0f / (1.0f + exp(-e_val));
        float silu_e = e_val * sigmoid_e;
        
        h[gid] = half(silu_e * g_val);
    """
    return mx.fast.metal_kernel(
        name = "swiglu_fused",
        input_names = ["e", "g", "n_ptr"],
        output_names = ["h"],
        source = kernel_source,
    )


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
        (1, 8192, 8192, "Long context 8K"),
        (8, 2048, 4096, "Smaller model batch"),
    ]

    import importlib.util

    kernel_path = os.path.join(
        os.path.dirname(__file__), "..", "unsloth", "kernels", "metal", "swiglu.py"
    )
    spec = importlib.util.spec_from_file_location("metal_swiglu", kernel_path)
    metal_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(metal_module)

    results = []

    for batch, seq, dim, desc in configs:
        elements = batch * seq * dim
        print(f"📊 {desc}")
        print(f"   Shape: ({batch}, {seq}, {dim}) = {elements / 1e6:.2f}M elements")
        print("-" * 70)

        # MLX Composed (baseline)
        e_mlx = mx.random.normal((batch, seq, dim)).astype(mx.float16)
        g_mlx = mx.random.normal((batch, seq, dim)).astype(mx.float16)
        mx.eval(e_mlx)
        mx.eval(g_mlx)

        t_mlx = benchmark_fn(lambda: mlx_swiglu_composed(e_mlx, g_mlx))
        tp_mlx = calculate_throughput(elements, t_mlx)
        print(f"   MLX Composed:       {t_mlx:7.3f} ms | {tp_mlx:7.2f} GB/s")

        # Fused Metal Kernel (pure MLX - fair comparison)
        def mlx_fused():
            return metal_module.mlx_swiglu_forward(e_mlx, g_mlx)

        t_fused = benchmark_fn(mlx_fused)
        tp_fused = calculate_throughput(elements, t_fused)
        speedup = t_mlx / t_fused
        print(f"   Fused Metal (MLX):  {t_fused:7.3f} ms | {tp_fused:7.2f} GB/s  ({speedup:.2f}x)")

        # PyTorch Tensors
        e_torch = torch.randn(batch, seq, dim, device = "mps", dtype = torch.float16)
        g_torch = torch.randn(batch, seq, dim, device = "mps", dtype = torch.float16)
        torch.mps.synchronize()

        # Fused Metal (PyTorch path - includes conversion overhead)
        t_metal = benchmark_fn(
            lambda: metal_module.metal_swiglu_forward(e_torch, g_torch)
        )
        tp_metal = calculate_throughput(elements, t_metal)
        print(f"   Fused Metal (PyTorch): {t_metal:7.3f} ms | {tp_metal:7.2f} GB/s  (includes conversion)")

        # PyTorch MPS Reference
        t_torch = benchmark_fn(lambda: pytorch_swiglu_reference(e_torch, g_torch))
        tp_torch = calculate_throughput(elements, t_torch)
        print(f"   PyTorch MPS:        {t_torch:7.3f} ms | {tp_torch:7.2f} GB/s")

        results.append(
            {
                "config": desc,
                "mlx": tp_mlx,
                "fused": tp_fused,
                "torch": tp_torch,
                "speedup": speedup,
            }
        )
        print()


    # Summary
    if results:
        avg_speedup = sum(r["speedup"] for r in results) / len(results)
        avg_fused = sum(r["fused"] for r in results) / len(results)

        print("=" * 75)
        print(f"Average Fused Throughput: {avg_fused:.2f} GB/s")
        print(f"Average Speedup vs MLX:   {avg_speedup:.2f}x")
        if avg_speedup > 1.0:
            print(f"✅ Fused kernel is {avg_speedup:.2f}x faster than MLX composed!")
        print("=" * 75)


# ==============================================================================
# Correctness Verification
# ==============================================================================


def run_correctness_tests():
    """Verify numerical correctness against PyTorch reference."""
    print()
    print("=" * 75)
    print("CORRECTNESS VERIFICATION")
    print("=" * 75)
    print()

    test_shapes = [
        (1, 128, 256, "Small"),
        (2, 512, 1024, "Medium"),
        (4, 2048, 4096, "Large (LLM-sized)"),
    ]

    fused_kernel = get_fused_kernel()
    all_passed = True

    for batch, seq, dim, name in test_shapes:
        print(f"Testing: {name} - shape ({batch}, {seq}, {dim})")
        print("-" * 50)

        torch.manual_seed(42)
        e_torch = torch.randn(batch, seq, dim, dtype = torch.float32)
        g_torch = torch.randn(batch, seq, dim, dtype = torch.float32)

        # PyTorch Reference
        ref_output = pytorch_swiglu_reference(e_torch, g_torch)

        # MLX Composed
        e_mlx = mx.array(e_torch.numpy())
        g_mlx = mx.array(g_torch.numpy())
        mlx_output = mlx_swiglu_composed(e_mlx, g_mlx)
        mx.eval(mlx_output)

        mlx_diff = np.abs(np.array(mlx_output) - ref_output.numpy())
        mlx_pass = mlx_diff.max() < 1e-5
        print(
            f"  MLX Composed:  max={mlx_diff.max():.2e} mean={mlx_diff.mean():.2e} {'✅' if mlx_pass else '❌'}"
        )

        # Fused Metal (Production)
        import importlib.util

        kernel_path = os.path.join(
            os.path.dirname(__file__), "..", "unsloth", "kernels", "metal", "swiglu.py"
        )
        spec = importlib.util.spec_from_file_location("metal_swiglu", kernel_path)
        metal_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(metal_module)

        fused_output = metal_module.metal_swiglu_forward(
            e_torch.to("mps", torch.float16), g_torch.to("mps", torch.float16)
        )
        fused_np = fused_output.cpu().float().numpy()

        ref_f16 = (
            pytorch_swiglu_reference(e_torch.half(), g_torch.half()).float().numpy()
        )
        fused_diff = np.abs(fused_np - ref_f16)
        fused_pass = fused_diff.max() < 1e-2  # 1% tolerance for FP16
        print(
            f"  Fused (fp16):  max={fused_diff.max():.2e} mean={fused_diff.mean():.2e} {'✅' if fused_pass else '❌'}"
        )

        if not (mlx_pass and fused_pass):
            all_passed = False
        print()

    print("=" * 75)
    if all_passed:
        print("✅ ALL CORRECTNESS TESTS PASSED")
    else:
        print("❌ SOME TESTS FAILED")
    print("=" * 75)

    return all_passed


# ==============================================================================
# Main
# ==============================================================================


def main():
    parser = argparse.ArgumentParser(description = "SwiGLU Kernel Benchmark Suite")
    parser.add_argument(
        "--perf", action = "store_true", help = "Run performance benchmarks only"
    )
    parser.add_argument(
        "--correctness", action = "store_true", help = "Run correctness tests only"
    )
    args = parser.parse_args()

    print_header()

    run_perf = not args.correctness or args.perf
    run_correct = not args.perf or args.correctness

    if run_perf:
        run_performance_benchmark()

    if run_correct:
        run_correctness_tests()

    print("\nDone.")


if __name__ == "__main__":
    main()
