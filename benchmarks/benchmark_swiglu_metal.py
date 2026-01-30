# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""
Metal SwiGLU Kernel Benchmark

Compares throughput of:
1. PyTorch Native (silu(e) * g)
2. MPS Fallback (PyTorch-native on MPS)
3. Custom Metal Kernel (vectorized float4/half4)

Target: 200+ GB/s on M4 Max (546 GB/s peak bandwidth)
Baseline: ~61 GB/s with MPS fallbacks
"""

import sys
import time
import platform
from typing import Callable, Optional

import torch


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


def calculate_throughput(elements: int, latency_ms: float, dtype: torch.dtype) -> float:
    """
    Calculate memory throughput in GB/s.

    SwiGLU Memory:
        Read:  e (elements * dtype_size) + g (elements * dtype_size)
        Write: h (elements * dtype_size)
        Total: 3 * elements * dtype_size
    """
    dtype_size = 2 if dtype in (torch.float16, torch.bfloat16) else 4
    bytes_total = 3 * elements * dtype_size
    latency_s = latency_ms / 1000.0
    return (bytes_total / 1e9) / latency_s


def run_benchmarks():
    """Run comprehensive SwiGLU benchmark suite."""
    if not torch.backends.mps.is_available():
        print("❌ MPS backend not available (Required for Apple Silicon testing)")
        print("   This benchmark is designed for macOS with Apple Silicon.")
        sys.exit(1)

    print("=" * 75)
    print("Metal SwiGLU Kernel Benchmark")
    print("=" * 75)
    print(f"Platform:  {platform.platform()}")
    print(f"Processor: {platform.processor()}")
    print(f"PyTorch:   {torch.__version__}")
    print()

    # Check Metal kernel availability
    metal_available = False
    try:
        from unsloth.kernels.metal import is_metal_swiglu_available

        metal_available = is_metal_swiglu_available()
        print(
            f"Metal SwiGLU Available: {'✅ Yes' if metal_available else '❌ No (using MPS fallback)'}"
        )
    except ImportError:
        print("Metal SwiGLU Available: ❌ No (module not found)")

    print()

    # Standard configurations for LLM workloads
    configs = [
        # (batch, seq_len, hidden_dim, description)
        (1, 2048, 8192, "Llama-3 8B (Single Query)"),
        (4, 512, 14336, "Llama-3 8B (Batch Training)"),
        (1, 16384, 8192, "Long Context (16K)"),
        (8, 2048, 4096, "Smaller Model Batch"),
    ]

    dtype = torch.float16
    print(f"Dtype: {dtype}")
    print()

    # Results storage for summary
    results = []

    for batch, seq, dim, description in configs:
        elements = batch * seq * dim
        print(f"Config: {description}")
        print(f"  Shape: ({batch}, {seq}, {dim}) = {elements / 1e6:.2f}M elements")
        print("-" * 70)

        # Create test tensors
        e = torch.randn(batch, seq, dim, device = "mps", dtype = dtype)
        g = torch.randn(batch, seq, dim, device = "mps", dtype = dtype)

        # 1. PyTorch Native Reference
        t_native = benchmark_fn(lambda: torch.nn.functional.silu(e) * g)
        throughput_native = calculate_throughput(elements, t_native, dtype)
        print(f"  PyTorch Native:  {t_native:7.3f} ms | {throughput_native:7.2f} GB/s")

        # 2. MPS Fallback (explicit import)
        try:
            from unsloth.kernels.mps.swiglu import mps_swiglu_forward

            t_mps = benchmark_fn(lambda: mps_swiglu_forward(e, g))
            throughput_mps = calculate_throughput(elements, t_mps, dtype)
            print(f"  MPS Fallback:    {t_mps:7.3f} ms | {throughput_mps:7.2f} GB/s")
        except ImportError:
            throughput_mps = None
            print("  MPS Fallback:    (not available)")

        # 3. Metal Kernel
        if metal_available:
            try:
                from unsloth.kernels.metal.swiglu import metal_swiglu_forward

                t_metal = benchmark_fn(lambda: metal_swiglu_forward(e, g))
                throughput_metal = calculate_throughput(elements, t_metal, dtype)
                speedup = (
                    throughput_metal / throughput_native if throughput_native > 0 else 0
                )
                print(
                    f"  Metal Kernel:    {t_metal:7.3f} ms | {throughput_metal:7.2f} GB/s | {speedup:.2f}x speedup"
                )

                results.append(
                    {
                        "config": description,
                        "elements": elements,
                        "native": throughput_native,
                        "metal": throughput_metal,
                        "speedup": speedup,
                    }
                )
            except Exception as ex:
                print(f"  Metal Kernel:    ❌ Error: {ex}")
        else:
            print("  Metal Kernel:    (not available)")

        print()

    # Summary
    if results:
        print("=" * 75)
        print("SUMMARY")
        print("=" * 75)
        avg_throughput = sum(r["metal"] for r in results) / len(results)
        avg_speedup = sum(r["speedup"] for r in results) / len(results)
        max_throughput = max(r["metal"] for r in results)

        print(f"Average Metal Throughput: {avg_throughput:.2f} GB/s")
        print(f"Peak Metal Throughput:    {max_throughput:.2f} GB/s")
        print(f"Average Speedup:          {avg_speedup:.2f}x vs PyTorch Native")
        print()

        # Target check
        target = 200.0
        if max_throughput >= target:
            print(f"✅ TARGET MET: {max_throughput:.2f} GB/s >= {target} GB/s")
        else:
            print(f"⚠️  TARGET NOT MET: {max_throughput:.2f} GB/s < {target} GB/s")
            print(f"   M4 Max Peak Bandwidth: 546 GB/s")
            print(f"   Current Efficiency: {max_throughput / 546 * 100:.1f}%")

    print()
    print("Done.")


def run_backward_benchmark():
    """Benchmark backward pass specifically."""
    print()
    print("=" * 75)
    print("Metal SwiGLU Backward Pass Benchmark")
    print("=" * 75)

    if not torch.backends.mps.is_available():
        print("❌ MPS not available")
        return

    metal_available = False
    try:
        from unsloth.kernels.metal import is_metal_swiglu_available

        metal_available = is_metal_swiglu_available()
    except ImportError:
        pass

    dtype = torch.float16
    batch, seq, dim = 4, 2048, 8192
    elements = batch * seq * dim

    print(f"Config: ({batch}, {seq}, {dim}) = {elements / 1e6:.2f}M elements")
    print()

    # Flatten for backward (as per Triton kernel)
    dw = torch.randn(batch * seq, dim, device = "mps", dtype = dtype)
    e = torch.randn(batch * seq, dim, device = "mps", dtype = dtype)
    g = torch.randn(batch * seq, dim, device = "mps", dtype = dtype)

    # MPS Fallback
    try:
        from unsloth.kernels.mps.swiglu import mps_swiglu_backward

        def run_mps():
            dw_c, e_c, g_c = dw.clone(), e.clone(), g.clone()
            return mps_swiglu_backward(dw_c, e_c, g_c)

        t_mps = benchmark_fn(run_mps)
        # Backward reads: dw, e, g. Writes: h, df, de. Total: 6 * elements * dtype_size
        bytes_total = 6 * elements * 2
        throughput = (bytes_total / 1e9) / (t_mps / 1000)
        print(f"MPS Fallback:   {t_mps:7.3f} ms | {throughput:7.2f} GB/s")
    except ImportError:
        print("MPS Fallback:   (not available)")

    # Metal Kernel
    if metal_available:
        try:
            from unsloth.kernels.metal.swiglu import metal_swiglu_backward

            def run_metal():
                dw_c, e_c, g_c = dw.clone(), e.clone(), g.clone()
                return metal_swiglu_backward(dw_c, e_c, g_c)

            t_metal = benchmark_fn(run_metal)
            bytes_total = 6 * elements * 2
            throughput = (bytes_total / 1e9) / (t_metal / 1000)
            print(f"Metal Kernel:   {t_metal:7.3f} ms | {throughput:7.2f} GB/s")
        except Exception as ex:
            print(f"Metal Kernel:   ❌ Error: {ex}")


if __name__ == "__main__":
    run_benchmarks()
    run_backward_benchmark()
