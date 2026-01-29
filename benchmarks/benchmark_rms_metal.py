# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Benchmark script for Metal RMS LayerNorm kernel.

Compares performance of:
1. PyTorch MPS native
2. Custom Metal kernel (via MLX)
3. MPS fallback (PyTorch-based)

Run on macOS with Apple Silicon:
    python benchmarks/benchmark_rms_metal.py
"""

import platform
import time
import sys
from typing import Callable

import torch


def benchmark_fn(fn: Callable, warmup: int = 10, iterations: int = 100) -> float:
    """Benchmark a function and return average latency in ms."""
    # Warmup
    for _ in range(warmup):
        fn()
    
    # Sync before timing
    if torch.backends.mps.is_available():
        torch.mps.synchronize()
    
    start = time.perf_counter()
    for _ in range(iterations):
        fn()
    
    # Sync after timing
    if torch.backends.mps.is_available():
        torch.mps.synchronize()
    
    end = time.perf_counter()
    return (end - start) / iterations * 1000  # ms


def reference_rms_layernorm(X: torch.Tensor, W: torch.Tensor, eps: float) -> torch.Tensor:
    """PyTorch reference implementation."""
    X_f32 = X.to(torch.float32)
    variance = X_f32.pow(2).mean(-1, keepdim=True)
    rms_inv = torch.rsqrt(variance + eps)
    X_norm = (X_f32 * rms_inv).to(X.dtype)
    return W.to(X.dtype) * X_norm


def run_benchmarks():
    """Run benchmarks comparing different RMS LayerNorm implementations."""
    if platform.system() != "Darwin":
        print("❌ This benchmark requires macOS with Apple Silicon")
        sys.exit(1)
    
    if not torch.backends.mps.is_available():
        print("❌ MPS backend not available")
        sys.exit(1)
    
    print("=" * 70)
    print("RMS LayerNorm Metal Kernel Benchmark")
    print("=" * 70)
    print(f"Platform: {platform.platform()}")
    print(f"PyTorch: {torch.__version__}")
    print()
    
    # Test configurations
    configs = [
        # (batch_size, seq_len, hidden_dim)
        (1, 2048, 4096),    # Single batch, typical Llama-7B
        (4, 512, 4096),     # Small batch
        (8, 1024, 4096),    # Medium batch
        (1, 4096, 4096),    # Long sequence
        (1, 2048, 8192),    # Larger hidden dim (Llama-70B)
    ]
    
    # Check Metal kernel availability
    try:
        from unsloth.kernels.metal import is_metal_available, metal_rms_layernorm
        metal_available = is_metal_available()
    except ImportError:
        metal_available = False
    
    # Check MPS fallback availability
    try:
        from unsloth.kernels.mps.rms_layernorm import mps_rms_layernorm
        mps_fallback_available = True
    except ImportError:
        mps_fallback_available = False
    
    print("Available implementations:")
    print(f"  • PyTorch MPS (reference): ✅")
    print(f"  • Custom Metal kernel:     {'✅' if metal_available else '❌'}")
    print(f"  • MPS fallback:            {'✅' if mps_fallback_available else '❌'}")
    print()
    
    eps = 1e-5
    dtype = torch.float32
    
    results = []
    
    for batch_size, seq_len, hidden_dim in configs:
        print(f"Config: batch={batch_size}, seq={seq_len}, hidden={hidden_dim}")
        print("-" * 50)
        
        X = torch.randn(batch_size, seq_len, hidden_dim, device="mps", dtype=dtype)
        W = torch.randn(hidden_dim, device="mps", dtype=dtype)
        
        # Benchmark PyTorch MPS reference
        t_ref = benchmark_fn(lambda: reference_rms_layernorm(X, W, eps))
        print(f"  PyTorch MPS:     {t_ref:.3f} ms")
        
        # Benchmark Metal kernel
        if metal_available:
            t_metal = benchmark_fn(lambda: metal_rms_layernorm(X, W, eps))
            speedup = t_ref / t_metal
            print(f"  Metal kernel:    {t_metal:.3f} ms ({speedup:.2f}x)")
        else:
            t_metal = None
        
        # Benchmark MPS fallback
        if mps_fallback_available:
            t_mps = benchmark_fn(lambda: mps_rms_layernorm(X, W, eps))
            speedup = t_ref / t_mps
            print(f"  MPS fallback:    {t_mps:.3f} ms ({speedup:.2f}x)")
        else:
            t_mps = None
        
        results.append({
            "config": (batch_size, seq_len, hidden_dim),
            "pytorch_mps": t_ref,
            "metal_kernel": t_metal,
            "mps_fallback": t_mps,
        })
        print()
    
    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    if metal_available:
        avg_speedup = sum(
            r["pytorch_mps"] / r["metal_kernel"]
            for r in results if r["metal_kernel"]
        ) / len(results)
        print(f"Average Metal kernel speedup over PyTorch MPS: {avg_speedup:.2f}x")
    else:
        print("Metal kernel not available - install MLX for benchmarking")
    
    return results


if __name__ == "__main__":
    run_benchmarks()
