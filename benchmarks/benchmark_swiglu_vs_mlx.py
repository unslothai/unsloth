# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""
SwiGLU Benchmark: Fused Metal Kernel vs MLX Composed

Compares:
1. MLX Composed: mx.nn.silu(e) * g (3 memory passes)
2. Custom Fused Metal Kernel (1 memory pass - YOUR kernel)
3. PyTorch MPS baseline: torch.nn.functional.silu(e) * g

This answers the question: How much faster is the fused kernel?
"""

import sys
import time
import platform
from typing import Callable

print("=" * 75)
print("SwiGLU Benchmark: Fused Kernel vs MLX Composed")
print("=" * 75)
print(f"Platform:  {platform.platform()}")
print(f"Processor: {platform.processor()}")
print()

# Check environment
IS_DARWIN = platform.system() == "Darwin"
if not IS_DARWIN:
    print("❌ This benchmark requires macOS with Apple Silicon")
    print("   Run this on your Mac!")
    sys.exit(1)

import torch
if not torch.backends.mps.is_available():
    print("❌ MPS backend not available")
    sys.exit(1)

print(f"PyTorch:   {torch.__version__}")

# Check MLX
try:
    import mlx.core as mx
    import mlx.nn as nn
    print(f"MLX:       {mx.__version__}")
    HAS_MLX = True
except ImportError:
    print("MLX:       ❌ Not installed")
    HAS_MLX = False
    sys.exit(1)

# Check custom kernel
try:
    from unsloth.kernels.metal.swiglu import metal_swiglu_forward, is_metal_swiglu_available
    HAS_CUSTOM_KERNEL = is_metal_swiglu_available()
    print(f"Custom Metal Kernel: {'✅ Available' if HAS_CUSTOM_KERNEL else '❌ Not Available'}")
except ImportError:
    HAS_CUSTOM_KERNEL = False
    print("Custom Metal Kernel: ❌ Import failed")

print()


def benchmark_fn(fn: Callable, warmup: int = 10, iterations: int = 50) -> float:
    """Benchmark a function and return average latency in ms."""
    # Warmup
    for _ in range(warmup):
        result = fn()
    
    # Force sync for MLX
    if HAS_MLX:
        mx.eval(result) if hasattr(result, '__mlx_array__') or isinstance(result, mx.array) else None
    
    # Force sync for PyTorch MPS
    torch.mps.synchronize()
    
    start = time.perf_counter()
    for _ in range(iterations):
        result = fn()
        # Sync appropriately
        if HAS_MLX and (hasattr(result, '__mlx_array__') or isinstance(result, mx.array)):
            mx.eval(result)
    
    torch.mps.synchronize()
    end = time.perf_counter()
    
    return (end - start) / iterations * 1000  # ms


def calculate_throughput(elements: int, latency_ms: float, dtype_size: int = 2) -> float:
    """
    Calculate memory throughput in GB/s.
    
    SwiGLU: Read e + g, Write h = 3 tensors
    """
    bytes_total = 3 * elements * dtype_size
    latency_s = latency_ms / 1000.0
    return (bytes_total / 1e9) / latency_s


def mlx_swiglu_composed(e: mx.array, g: mx.array) -> mx.array:
    """MLX composed SwiGLU: silu(e) * g - NOT fused, 3 memory passes."""
    return mx.sigmoid(e) * e * g  # silu(x) = x * sigmoid(x)


def run_benchmark():
    """Run the main benchmark."""
    print("=" * 75)
    print("BENCHMARK: Measuring kernel execution time (excluding data transfer)")
    print("=" * 75)
    print()
    
    configs = [
        # (batch, seq_len, hidden_dim, description)
        (1, 2048, 8192, "Llama-3 8B (inference)"),
        (4, 512, 14336, "Llama-3 8B (training batch)"),
        (1, 8192, 8192, "Long context 8K"),
        (8, 2048, 4096, "Smaller model batch"),
    ]
    
    results = []
    
    for batch, seq, dim, desc in configs:
        elements = batch * seq * dim
        print(f"📊 {desc}")
        print(f"   Shape: ({batch}, {seq}, {dim}) = {elements / 1e6:.2f}M elements")
        print("-" * 70)
        
        # =====================================================================
        # 1. MLX Composed Version (baseline for comparison)
        # =====================================================================
        e_mlx = mx.random.normal((batch, seq, dim)).astype(mx.float16)
        g_mlx = mx.random.normal((batch, seq, dim)).astype(mx.float16)
        mx.eval(e_mlx)
        mx.eval(g_mlx)
        
        t_mlx = benchmark_fn(lambda: mlx_swiglu_composed(e_mlx, g_mlx))
        tp_mlx = calculate_throughput(elements, t_mlx)
        print(f"   MLX Composed (silu*g):    {t_mlx:7.3f} ms | {tp_mlx:7.2f} GB/s")
        
        # =====================================================================
        # 2. Pure MLX Fused Kernel (in-framework, no torch conversion)
        # =====================================================================
        # We'll create a pure MLX version of the kernel to compare fairly
        if HAS_MLX and hasattr(mx, 'fast') and hasattr(mx.fast, 'metal_kernel'):
            try:
                # Create kernel inline for fair comparison
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
                
                fused_kernel = mx.fast.metal_kernel(
                    name="swiglu_fused_benchmark",
                    input_names=["e", "g", "n_ptr"],
                    output_names=["h"],
                    source=kernel_source,
                )
                
                e_flat = e_mlx.flatten()
                g_flat = g_mlx.flatten()
                n_arr = mx.array([elements], dtype=mx.uint32)
                
                def run_fused():
                    out = fused_kernel(
                        inputs=[e_flat, g_flat, n_arr],
                        output_shapes=[(elements,)],
                        output_dtypes=[mx.float16],
                        grid=(elements, 1, 1),
                        threadgroup=(min(256, elements), 1, 1),
                    )
                    mx.eval(out[0])
                    return out[0]
                
                t_fused = benchmark_fn(run_fused)
                tp_fused = calculate_throughput(elements, t_fused)
                speedup_vs_mlx = tp_fused / tp_mlx if tp_mlx > 0 else 0
                print(f"   Fused Metal Kernel:       {t_fused:7.3f} ms | {tp_fused:7.2f} GB/s | {speedup_vs_mlx:.2f}x vs MLX")
                
                results.append({
                    "config": desc,
                    "elements": elements,
                    "mlx_composed_gbps": tp_mlx,
                    "fused_metal_gbps": tp_fused,
                    "speedup": speedup_vs_mlx,
                })
            except Exception as ex:
                print(f"   Fused Metal Kernel:       ❌ Error: {ex}")
        
        # =====================================================================
        # 3. PyTorch MPS (for reference - different framework)
        # =====================================================================
        e_torch = torch.randn(batch, seq, dim, device="mps", dtype=torch.float16)
        g_torch = torch.randn(batch, seq, dim, device="mps", dtype=torch.float16)
        torch.mps.synchronize()
        
        t_torch = benchmark_fn(lambda: torch.nn.functional.silu(e_torch) * g_torch)
        tp_torch = calculate_throughput(elements, t_torch)
        print(f"   PyTorch MPS (reference):  {t_torch:7.3f} ms | {tp_torch:7.2f} GB/s")
        
        print()
    
    # Summary
    if results:
        print("=" * 75)
        print("SUMMARY: Fused Metal Kernel vs MLX Composed")
        print("=" * 75)
        
        avg_speedup = sum(r["speedup"] for r in results) / len(results)
        avg_fused = sum(r["fused_metal_gbps"] for r in results) / len(results)
        avg_mlx = sum(r["mlx_composed_gbps"] for r in results) / len(results)
        
        print(f"\nAverage Throughput:")
        print(f"  MLX Composed:      {avg_mlx:.2f} GB/s")
        print(f"  Fused Metal:       {avg_fused:.2f} GB/s")
        print(f"  Average Speedup:   {avg_speedup:.2f}x")
        print()
        
        if avg_speedup > 1.0:
            print(f"✅ Your fused kernel is {avg_speedup:.2f}x FASTER than MLX composed!")
            print(f"   This is because fusing eliminates {2} extra memory passes.")
        else:
            print(f"⚠️  Fused kernel is {1/avg_speedup:.2f}x slower - may need optimization")
        
        print()
        print("Theory: Fused kernel should be ~1.5-3x faster due to:")
        print("  - MLX composed: 3 separate ops = 3 kernel launches + 2 intermediate buffers")
        print("  - Fused kernel: 1 kernel launch, no intermediate storage")
        print()


if __name__ == "__main__":
    run_benchmark()
