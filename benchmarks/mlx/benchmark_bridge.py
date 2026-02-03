#!/usr/bin/env python3
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
Benchmark MLX <-> PyTorch tensor conversion overhead.

This script measures the time cost of converting tensors between
PyTorch and MLX to help determine optimal dispatch granularity.

Usage:
    python benchmarks/mlx/benchmark_bridge.py

Requirements:
    - Apple Silicon Mac
    - MLX installed: pip install mlx>=0.6.0
"""

import sys
import time
from typing import List, Tuple


def check_environment() -> bool:
    """Check if we can run MLX benchmarks."""
    if sys.platform != "darwin":
        print("[X] MLX benchmarks require macOS (Apple Silicon)")
        return False

    try:
        import mlx.core as mx
        import torch

        print(
            f"[OK] MLX version: {mx.__version__ if hasattr(mx, '__version__') else 'unknown'}"
        )
        print(f"[OK] PyTorch version: {torch.__version__}")
        print(f"[OK] MPS available: {torch.backends.mps.is_available()}")
        return True
    except ImportError as e:
        print(f"[X] Missing dependency: {e}")
        return False


def benchmark_torch_to_mlx(sizes: List[Tuple[int, ...]], n_iterations: int = 100):
    """Benchmark PyTorch -> MLX conversion."""
    import torch
    from unsloth.kernels.mlx import torch_to_mlx, synchronize_mps

    print("\n" + "=" * 60)
    print("PyTorch -> MLX Conversion")
    print("=" * 60)
    print(f"{'Shape':<20} {'Total (ms)':<15} {'Per-iter (us)':<15}")
    print("-" * 60)

    for shape in sizes:
        # Create tensor on MPS
        tensor = torch.randn(shape, device="mps", dtype=torch.float32)
        synchronize_mps()

        # Warmup
        for _ in range(5):
            _ = torch_to_mlx(tensor)

        # Benchmark
        start = time.perf_counter()
        for _ in range(n_iterations):
            _ = torch_to_mlx(tensor)
        elapsed = time.perf_counter() - start

        total_ms = elapsed * 1000
        per_iter_us = (elapsed / n_iterations) * 1_000_000

        print(f"{str(shape):<20} {total_ms:<15.2f} {per_iter_us:<15.2f}")


def benchmark_mlx_to_torch(sizes: List[Tuple[int, ...]], n_iterations: int = 100):
    """Benchmark MLX -> PyTorch conversion."""
    import mlx.core as mx
    from unsloth.kernels.mlx import mlx_to_torch

    print("\n" + "=" * 60)
    print("MLX -> PyTorch Conversion")
    print("=" * 60)
    print(f"{'Shape':<20} {'Total (ms)':<15} {'Per-iter (us)':<15}")
    print("-" * 60)

    for shape in sizes:
        # Create MLX array
        arr = mx.random.normal(shape=shape)
        mx.eval(arr)

        # Warmup
        for _ in range(5):
            _ = mlx_to_torch(arr, device="mps")

        # Benchmark
        start = time.perf_counter()
        for _ in range(n_iterations):
            _ = mlx_to_torch(arr, device="mps")
        elapsed = time.perf_counter() - start

        total_ms = elapsed * 1000
        per_iter_us = (elapsed / n_iterations) * 1_000_000

        print(f"{str(shape):<20} {total_ms:<15.2f} {per_iter_us:<15.2f}")


def benchmark_roundtrip(sizes: List[Tuple[int, ...]], n_iterations: int = 100):
    """Benchmark full roundtrip conversion."""
    import torch
    import mlx.core as mx
    from unsloth.kernels.mlx import torch_to_mlx, mlx_to_torch, synchronize_mps

    print("\n" + "=" * 60)
    print("Roundtrip: PyTorch -> MLX -> PyTorch")
    print("=" * 60)
    print(f"{'Shape':<20} {'Total (ms)':<15} {'Per-iter (us)':<15}")
    print("-" * 60)

    for shape in sizes:
        # Create tensor on MPS
        tensor = torch.randn(shape, device="mps", dtype=torch.float32)
        synchronize_mps()

        # Warmup
        for _ in range(5):
            arr = torch_to_mlx(tensor)
            _ = mlx_to_torch(arr, device="mps")

        # Benchmark
        start = time.perf_counter()
        for _ in range(n_iterations):
            arr = torch_to_mlx(tensor)
            _ = mlx_to_torch(arr, device="mps")
        elapsed = time.perf_counter() - start

        total_ms = elapsed * 1000
        per_iter_us = (elapsed / n_iterations) * 1_000_000

        print(f"{str(shape):<20} {total_ms:<15.2f} {per_iter_us:<15.2f}")


def main():
    """Run all benchmarks."""
    print("=" * 60)
    print("MLX <-> PyTorch Bridge Benchmark")
    print("=" * 60)

    if not check_environment():
        sys.exit(1)

    # Test various tensor sizes
    sizes = [
        (64,),  # Small 1D
        (512,),  # Medium 1D
        (4096,),  # Large 1D
        (64, 64),  # Small 2D
        (256, 256),  # Medium 2D
        (1024, 1024),  # Large 2D
        (32, 128, 128),  # 3D (batch, seq, hidden)
        (8, 32, 4096),  # LLM-like shapes
    ]

    n_iterations = 100

    print(f"\nRunning {n_iterations} iterations per size...")

    benchmark_torch_to_mlx(sizes, n_iterations)
    benchmark_mlx_to_torch(sizes, n_iterations)
    benchmark_roundtrip(sizes, n_iterations)

    print("\n" + "=" * 60)
    print("Benchmark Complete")
    print("=" * 60)
    print("\nKey insight: If roundtrip overhead > kernel speedup,")
    print("consider batching multiple MLX ops together.")


if __name__ == "__main__":
    main()
