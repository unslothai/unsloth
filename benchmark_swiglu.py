# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""
SwiGLU Kernel Comprehensive Benchmark Suite

This is the single source of truth for SwiGLU kernel testing on Apple Silicon.
Includes both performance benchmarks and correctness verification.
Tests forward AND backward passes across all implementations.

Implementations benchmarked:
  - MLX Composed:   mx.sigmoid(e) * e * g  (3 separate ops, lazy)
  - MLX Compiled:   mx.compile'd composed  (fused by MLX compiler)
  - Fused Metal:    Hand-written Metal kernel via mx.fast.metal_kernel
  - PyTorch MPS:    torch.nn.functional.silu(e) * g
  - Chained:        Fused Metal with _mlx_cache bypass (no bridge cost)
  - Full Bridge:    Fused Metal with torch->mlx conversion

Usage:
    python benchmarks/benchmark_swiglu.py              # Run all tests
    python benchmarks/benchmark_swiglu.py --perf       # Performance only
    python benchmarks/benchmark_swiglu.py --correctness # Correctness only
    python benchmarks/benchmark_swiglu.py --forward    # Forward perf only
    python benchmarks/benchmark_swiglu.py --backward   # Backward perf only
"""

import argparse
import sys
import time
import platform
import types
import importlib.util
from pathlib import Path
from typing import Callable, Dict, Any, Optional
from dataclasses import dataclass

# Add parent directory to sys.path to allow absolute imports of unsloth
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

IS_DARWIN = platform.system() == "Darwin"
if not IS_DARWIN:
    print("This benchmark requires macOS with Apple Silicon")
    sys.exit(1)

import torch
import numpy as np

if not torch.backends.mps.is_available():
    print("MPS backend not available")
    sys.exit(1)

import mlx.core as mx

# Check for mx.synchronize (available MLX >= 0.6)
_HAS_MX_SYNC = hasattr(mx, "synchronize")


def _full_sync():
    """Synchronize both MPS and MLX command queues."""
    torch.mps.synchronize()
    if _HAS_MX_SYNC:
        mx.synchronize()
    else:
        # Fallback: eval a small tensor to force queue drain
        mx.eval(mx.zeros(1))


def _ensure_pkg(name: str, path: Path) -> None:
    if name in sys.modules:
        return
    module = types.ModuleType(name)
    module.__path__ = [str(path)]
    sys.modules[name] = module


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_metal_modules():
    """Load metal benchmark modules without importing unsloth/__init__.py."""
    repo_root = Path(__file__).resolve().parents[1]
    unsloth_dir = repo_root / "unsloth"
    kernels_dir = unsloth_dir / "kernels"
    mlx_dir = kernels_dir / "mlx"
    metal_dir = kernels_dir / "metal"

    _ensure_pkg("unsloth", unsloth_dir)
    _ensure_pkg("unsloth.kernels", kernels_dir)
    _ensure_pkg("unsloth.kernels.mlx", mlx_dir)
    _ensure_pkg("unsloth.kernels.metal", metal_dir)

    _load_module("unsloth.kernels.mlx.quantization", mlx_dir / "quantization.py")
    _load_module("unsloth.kernels.mlx.utils", mlx_dir / "utils.py")
    _load_module("unsloth.kernels.mlx.bridge", mlx_dir / "bridge.py")

    metal_swiglu = _load_module(
        "unsloth.kernels.metal.swiglu", metal_dir / "swiglu.py"
    )
    metal_rms = _load_module(
        "unsloth.kernels.metal.rms_layernorm", metal_dir / "rms_layernorm.py"
    )
    return metal_swiglu, metal_rms


def print_header():
    """Print system information header."""
    print("=" * 80)
    print("SwiGLU Kernel Benchmark Suite")
    print("=" * 80)
    print(f"Platform:       {platform.platform()}")
    print(f"Processor:      {platform.processor()}")
    print(f"PyTorch:        {torch.__version__}")
    print(f"MLX:            {mx.__version__}")
    print(f"mx.synchronize: {'available' if _HAS_MX_SYNC else 'UNAVAILABLE (using fallback)'}")
    print()


# ==============================================================================
# Benchmark Timing
# ==============================================================================


@dataclass
class BenchResult:
    mean_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    trimmed_mean_ms: float  # After dropping top/bottom 10%


def _force_eval(result):
    """Force evaluation of any MLX arrays in a result."""
    if isinstance(result, mx.array):
        mx.eval(result)
    elif isinstance(result, (tuple, list)):
        mlx_arrays = [r for r in result if isinstance(r, mx.array)]
        if mlx_arrays:
            mx.eval(*mlx_arrays)


def benchmark_fn(
    fn: Callable,
    warmup: int = 15,
    iterations: int = 100,
) -> BenchResult:
    """
    Benchmark a function with per-iteration synchronization and variance reporting.

    Every iteration is individually timed with a full GPU sync before start
    and after completion, ensuring no pipeline overlap inflates results.
    """
    # --- Warmup with full sync each iteration ---
    for _ in range(warmup):
        result = fn()
        _force_eval(result)
    _full_sync()

    # --- Timed iterations with per-iteration sync ---
    times = []
    for _ in range(iterations):
        _full_sync()  # Drain both queues before timing

        start = time.perf_counter()
        result = fn()
        _force_eval(result)
        _full_sync()  # Drain both queues after work
        end = time.perf_counter()

        times.append((end - start) * 1000)

    times_arr = np.array(times)

    # Trimmed mean: drop top and bottom 10%
    trim_n = max(1, len(times) // 10)
    sorted_times = np.sort(times_arr)
    trimmed = sorted_times[trim_n:-trim_n] if trim_n < len(sorted_times) // 2 else sorted_times

    return BenchResult(
        mean_ms=float(np.mean(times_arr)),
        std_ms=float(np.std(times_arr)),
        min_ms=float(np.min(times_arr)),
        max_ms=float(np.max(times_arr)),
        trimmed_mean_ms=float(np.mean(trimmed)),
    )


def calculate_throughput(
    elements: int, latency_ms: float, dtype_size: int = 2, n_tensors: int = 3
) -> float:
    """
    Calculate memory throughput in GB/s.
    Forward:  n_tensors=3 (read e, g; write h)
    Backward: n_tensors=6 (read dw, e, g; write h, de, dg)
    """
    bytes_total = n_tensors * elements * dtype_size
    return (bytes_total / 1e9) / (latency_ms / 1000.0)


def format_result(
    label: str, res: BenchResult, elements: int, n_tensors: int = 3
) -> str:
    """Format a benchmark result line with throughput and variance."""
    tp = calculate_throughput(elements, res.trimmed_mean_ms, n_tensors=n_tensors)
    return (
        f"   {label:<26s} "
        f"{res.trimmed_mean_ms:7.3f} ms "
        f"(+/- {res.std_ms:.3f}) "
        f"[min={res.min_ms:.3f} max={res.max_ms:.3f}] "
        f"| {tp:7.2f} GB/s"
    )


# ==============================================================================
# SwiGLU Implementations — Forward
# ==============================================================================


def mlx_swiglu_composed(e: mx.array, g: mx.array) -> mx.array:
    """MLX composed SwiGLU: silu(e) * g - NOT fused."""
    return mx.sigmoid(e) * e * g


def _mlx_swiglu_compiled_inner(e: mx.array, g: mx.array) -> mx.array:
    """Inner function for mx.compile - same math as composed."""
    return mx.sigmoid(e) * e * g


# Build the compiled forward once at module level
_mlx_swiglu_compiled = mx.compile(_mlx_swiglu_compiled_inner)


def mlx_swiglu_compiled(e: mx.array, g: mx.array) -> mx.array:
    """MLX compiled SwiGLU: same ops but traced/compiled by mx.compile."""
    return _mlx_swiglu_compiled(e, g)


def pytorch_swiglu_reference(e: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    """Ground truth SwiGLU: silu(e) * g"""
    return torch.nn.functional.silu(e) * g


# ==============================================================================
# SwiGLU Implementations — Backward
# ==============================================================================


def mlx_swiglu_composed_backward(dw: mx.array, e: mx.array, g: mx.array):
    """
    MLX composed backward via mx.vjp.
    Returns (h, de, dg) to match the fused kernel interface.
    """
    # vjp returns (primals_out, vjp_fn)
    h, vjp_fn = mx.vjp(mlx_swiglu_composed, [e, g], [dw])
    de, dg = vjp_fn
    return h, de, dg


def _mlx_swiglu_bwd_inner(dw: mx.array, e: mx.array, g: mx.array):
    """
    Inner function for compiled backward: vjp the UNCOMPILED composed function,
    then the whole fwd+bwd graph gets compiled as one unit.
    """
    h, vjp_fn = mx.vjp(mlx_swiglu_composed, [e, g], [dw])
    de, dg = vjp_fn
    return h, de, dg


# Compile the entire fwd+bwd graph — differentiate first, compile last
_mlx_swiglu_compiled_backward = mx.compile(_mlx_swiglu_bwd_inner)


def mlx_swiglu_compiled_backward(dw: mx.array, e: mx.array, g: mx.array):
    """MLX compiled backward: vjp traced then compiled."""
    return _mlx_swiglu_compiled_backward(dw, e, g)


def pytorch_swiglu_backward(
    dw: torch.Tensor, e: torch.Tensor, g: torch.Tensor
):
    """
    PyTorch MPS backward via autograd.
    Returns (de, dg) gradients.
    """
    e_ag = e.detach().requires_grad_(True)
    g_ag = g.detach().requires_grad_(True)
    h = torch.nn.functional.silu(e_ag) * g_ag
    h.backward(dw)
    return e_ag.grad, g_ag.grad


# ==============================================================================
# Performance Benchmark — Forward
# ==============================================================================


def run_forward_benchmark(metal_module, configs):
    """Run forward pass benchmarks."""
    print("=" * 80)
    print("FORWARD PASS BENCHMARK")
    print("=" * 80)
    print()
    print("Timing: per-iteration sync, 100 iterations, trimmed mean (drop top/bottom 10%)")
    print("Throughput: 3 tensors x elements x 2 bytes (read e, g; write h)")
    print()

    FWD_TENSORS = 3
    results = []

    for batch, seq, dim, desc in configs:
        elements = batch * seq * dim
        print(f"--- {desc} ---")
        print(f"    Shape: ({batch}, {seq}, {dim}) = {elements / 1e6:.2f}M elements")
        print()

        # ---- Create inputs ----
        e_mlx = mx.random.normal((batch, seq, dim)).astype(mx.float16)
        g_mlx = mx.random.normal((batch, seq, dim)).astype(mx.float16)
        mx.eval(e_mlx, g_mlx)

        e_torch = torch.randn(batch, seq, dim, device="mps", dtype=torch.float16)
        g_torch = torch.randn(batch, seq, dim, device="mps", dtype=torch.float16)
        torch.mps.synchronize()
        _full_sync()

        # 1. MLX Composed
        res_composed = benchmark_fn(lambda: mlx_swiglu_composed(e_mlx, g_mlx))
        print(format_result("MLX Composed:", res_composed, elements, FWD_TENSORS))

        # 2. MLX Compiled
        _warm = mlx_swiglu_compiled(e_mlx, g_mlx)
        mx.eval(_warm)
        _full_sync()
        res_compiled = benchmark_fn(lambda: mlx_swiglu_compiled(e_mlx, g_mlx))
        print(format_result("MLX Compiled:", res_compiled, elements, FWD_TENSORS))

        # 3. Fused Metal Kernel (pure MLX)
        res_fused = benchmark_fn(lambda: metal_module.mlx_swiglu_forward(e_mlx, g_mlx))
        speedup = res_composed.trimmed_mean_ms / res_fused.trimmed_mean_ms
        line = format_result("Fused Metal (MLX):", res_fused, elements, FWD_TENSORS)
        print(f"{line}  ({speedup:.2f}x vs composed)")

        # 4. PyTorch MPS
        res_mps = benchmark_fn(lambda: pytorch_swiglu_reference(e_torch, g_torch))
        print(format_result("PyTorch MPS:", res_mps, elements, FWD_TENSORS))

        # 5. Chained (bridge bypass)
        e_chained = e_torch.clone()
        g_chained = g_torch.clone()
        e_chained._mlx_cache = e_mlx
        g_chained._mlx_cache = g_mlx
        res_chained = benchmark_fn(
            lambda: metal_module.metal_swiglu_forward(e_chained, g_chained)
        )
        print(format_result("Chained (bridge skip):", res_chained, elements, FWD_TENSORS))

        # 6. Full bridge
        e_fresh = torch.randn(batch, seq, dim, device="mps", dtype=torch.float16)
        g_fresh = torch.randn(batch, seq, dim, device="mps", dtype=torch.float16)
        torch.mps.synchronize()
        res_bridge = benchmark_fn(
            lambda: metal_module.metal_swiglu_forward(e_fresh, g_fresh)
        )
        print(format_result("Torch->MLX (full bridge):", res_bridge, elements, FWD_TENSORS))

        print()

        results.append({
            "config": desc,
            "elements": elements,
            "composed": res_composed,
            "compiled": res_compiled,
            "fused": res_fused,
            "mps": res_mps,
            "chained": res_chained,
            "bridge": res_bridge,
        })

    _print_summary_table("FORWARD", results, FWD_TENSORS)
    return results


# ==============================================================================
# Performance Benchmark — Backward
# ==============================================================================


def run_backward_benchmark(metal_module, configs):
    """Run backward pass benchmarks."""
    print()
    print("=" * 80)
    print("BACKWARD PASS BENCHMARK")
    print("=" * 80)
    print()
    print("Timing: per-iteration sync, 100 iterations, trimmed mean (drop top/bottom 10%)")
    print("Throughput: 6 tensors x elements x 2 bytes (read dw, e, g; write h, de, dg)")
    print()

    BWD_TENSORS = 6
    results = []

    for batch, seq, dim, desc in configs:
        elements = batch * seq * dim
        print(f"--- {desc} ---")
        print(f"    Shape: ({batch}, {seq}, {dim}) = {elements / 1e6:.2f}M elements")
        print()

        # ---- Create inputs ----
        e_mlx = mx.random.normal((batch, seq, dim)).astype(mx.float16)
        g_mlx = mx.random.normal((batch, seq, dim)).astype(mx.float16)
        dw_mlx = mx.random.normal((batch, seq, dim)).astype(mx.float16)
        mx.eval(e_mlx, g_mlx, dw_mlx)

        e_torch = torch.randn(batch, seq, dim, device="mps", dtype=torch.float16)
        g_torch = torch.randn(batch, seq, dim, device="mps", dtype=torch.float16)
        dw_torch = torch.randn(batch, seq, dim, device="mps", dtype=torch.float16)
        torch.mps.synchronize()
        _full_sync()

        # 1. MLX Composed backward (vjp, not compiled)
        # Warmup vjp trace
        _warm = mlx_swiglu_composed_backward(dw_mlx, e_mlx, g_mlx)
        _force_eval(_warm)
        _full_sync()
        res_composed = benchmark_fn(
            lambda: mlx_swiglu_composed_backward(dw_mlx, e_mlx, g_mlx)
        )
        print(format_result("MLX Composed vjp:", res_composed, elements, BWD_TENSORS))

        # 2. MLX Compiled backward (vjp then compile)
        _warm = mlx_swiglu_compiled_backward(dw_mlx, e_mlx, g_mlx)
        _force_eval(_warm)
        _full_sync()
        res_compiled = benchmark_fn(
            lambda: mlx_swiglu_compiled_backward(dw_mlx, e_mlx, g_mlx)
        )
        print(format_result("MLX Compiled vjp:", res_compiled, elements, BWD_TENSORS))

        # 3. Fused Metal backward kernel (pure MLX)
        res_fused = benchmark_fn(
            lambda: metal_module.mlx_swiglu_backward(dw_mlx, e_mlx, g_mlx)
        )
        speedup = res_composed.trimmed_mean_ms / res_fused.trimmed_mean_ms
        line = format_result("Fused Metal bwd (MLX):", res_fused, elements, BWD_TENSORS)
        print(f"{line}  ({speedup:.2f}x vs composed)")

        # 4. PyTorch MPS backward (autograd)
        res_mps = benchmark_fn(
            lambda: pytorch_swiglu_backward(dw_torch, e_torch, g_torch)
        )
        print(format_result("PyTorch MPS autograd:", res_mps, elements, BWD_TENSORS))

        # 5. Fused Metal backward via torch wrapper (bridge bypass)
        e_chained = e_torch.clone()
        g_chained = g_torch.clone()
        dw_chained = dw_torch.clone()
        e_chained._mlx_cache = e_mlx
        g_chained._mlx_cache = g_mlx
        dw_chained._mlx_cache = dw_mlx
        res_chained = benchmark_fn(
            lambda: metal_module.metal_swiglu_backward(dw_chained, e_chained, g_chained)
        )
        print(format_result("Chained bwd (bridge skip):", res_chained, elements, BWD_TENSORS))

        # 6. Fused Metal backward via torch wrapper (full bridge)
        e_fresh = torch.randn(batch, seq, dim, device="mps", dtype=torch.float16)
        g_fresh = torch.randn(batch, seq, dim, device="mps", dtype=torch.float16)
        dw_fresh = torch.randn(batch, seq, dim, device="mps", dtype=torch.float16)
        torch.mps.synchronize()
        res_bridge = benchmark_fn(
            lambda: metal_module.metal_swiglu_backward(dw_fresh, e_fresh, g_fresh)
        )
        print(format_result("Torch->MLX bwd (bridge):", res_bridge, elements, BWD_TENSORS))

        print()

        results.append({
            "config": desc,
            "elements": elements,
            "composed": res_composed,
            "compiled": res_compiled,
            "fused": res_fused,
            "mps": res_mps,
            "chained": res_chained,
            "bridge": res_bridge,
        })

    _print_summary_table("BACKWARD", results, BWD_TENSORS)
    return results


# ==============================================================================
# Summary Table
# ==============================================================================


def _print_summary_table(pass_name: str, results: list, n_tensors: int):
    """Print a summary table for a set of benchmark results."""
    if not results:
        return

    print("=" * 80)
    print(f"SUMMARY — {pass_name} (trimmed mean GB/s)")
    print("=" * 80)
    header = (
        f"{'Config':<28s} "
        f"{'Composed':>9s} "
        f"{'Compiled':>9s} "
        f"{'Fused':>9s} "
        f"{'MPS':>9s} "
        f"{'Chained':>9s} "
        f"{'Bridge':>9s}"
    )
    print(header)
    print("-" * len(header))

    for r in results:
        n = r["elements"]
        tp = lambda res: calculate_throughput(n, res.trimmed_mean_ms, n_tensors=n_tensors)
        line = f"{r['config']:<28s}"
        line += f" {tp(r['composed']):>8.1f}"
        line += f" {tp(r['compiled']):>8.1f}"
        line += f" {tp(r['fused']):>8.1f}"
        line += f" {tp(r['mps']):>8.1f}"
        line += f" {tp(r['chained']):>8.1f}"
        line += f" {tp(r['bridge']):>8.1f}"
        print(line)

    print()

    # Winner per config (core implementations only, not bridge variants)
    for r in results:
        n = r["elements"]
        candidates = {
            "MLX Composed": r["composed"].trimmed_mean_ms,
            "MLX Compiled": r["compiled"].trimmed_mean_ms,
            "Fused Metal": r["fused"].trimmed_mean_ms,
            "PyTorch MPS": r["mps"].trimmed_mean_ms,
        }
        winner = min(candidates, key=candidates.get)
        tp = calculate_throughput(n, candidates[winner], n_tensors=n_tensors)
        print(f"  {r['config']:<28s} Winner: {winner} ({tp:.1f} GB/s)")

    print("=" * 80)


# ==============================================================================
# Performance Benchmark (combined entry point)
# ==============================================================================


def run_performance_benchmark(forward=True, backward=True):
    """Run performance benchmarks."""
    configs = [
        (1, 2048, 8192, "Llama-3 8B (inference)"),
        (4, 512, 14336, "Llama-3 8B (training)"),
        (1, 8192, 8192, "Long context 8K"),
        (8, 2048, 4096, "Smaller model batch"),
    ]

    metal_module, metal_rms = _load_metal_modules()

    fwd_results = None
    bwd_results = None

    if forward:
        fwd_results = run_forward_benchmark(metal_module, configs)

    if backward:
        bwd_results = run_backward_benchmark(metal_module, configs)

    return fwd_results, bwd_results


# ==============================================================================
# Correctness Verification
# ==============================================================================


def run_correctness_tests():
    """Verify numerical correctness of forward AND backward against PyTorch reference."""
    print()
    print("=" * 80)
    print("CORRECTNESS VERIFICATION")
    print("=" * 80)
    print()

    test_shapes = [
        (1, 128, 256, "Small"),
        (2, 512, 1024, "Medium"),
        (4, 2048, 4096, "Large (LLM-sized)"),
    ]

    metal_module, _ = _load_metal_modules()
    all_passed = True

    for batch, seq, dim, name in test_shapes:
        print(f"Testing: {name} - shape ({batch}, {seq}, {dim})")
        print("-" * 65)

        torch.manual_seed(42)
        e_torch = torch.randn(batch, seq, dim, dtype=torch.float32)
        g_torch = torch.randn(batch, seq, dim, dtype=torch.float32)
        dw_torch = torch.randn(batch, seq, dim, dtype=torch.float32)

        # =================================================================
        # Forward correctness
        # =================================================================
        print("  FORWARD:")
        ref_fwd = pytorch_swiglu_reference(e_torch, g_torch)

        # MLX Composed (fp32)
        e_mlx = mx.array(e_torch.numpy())
        g_mlx = mx.array(g_torch.numpy())
        out = mlx_swiglu_composed(e_mlx, g_mlx)
        mx.eval(out)
        diff = np.abs(np.array(out) - ref_fwd.numpy())
        p = diff.max() < 1e-5
        all_passed &= p
        print(f"    MLX Composed (fp32):  max={diff.max():.2e}  mean={diff.mean():.2e}  {'PASS' if p else 'FAIL'}")

        # MLX Compiled (fp32)
        out = mlx_swiglu_compiled(e_mlx, g_mlx)
        mx.eval(out)
        diff = np.abs(np.array(out) - ref_fwd.numpy())
        p = diff.max() < 1e-5
        all_passed &= p
        print(f"    MLX Compiled (fp32):  max={diff.max():.2e}  mean={diff.mean():.2e}  {'PASS' if p else 'FAIL'}")

        # Fused Metal (fp16)
        e_f16 = e_mlx.astype(mx.float16)
        g_f16 = g_mlx.astype(mx.float16)
        out = metal_module.mlx_swiglu_forward(e_f16, g_f16)
        mx.eval(out)
        ref_fwd_f16 = pytorch_swiglu_reference(e_torch.half(), g_torch.half()).float().numpy()
        diff = np.abs(np.array(out).astype(np.float32) - ref_fwd_f16)
        p = diff.max() < 1e-2
        all_passed &= p
        print(f"    Fused Metal  (fp16):  max={diff.max():.2e}  mean={diff.mean():.2e}  {'PASS' if p else 'FAIL'}")

        # =================================================================
        # Backward correctness
        # =================================================================
        print("  BACKWARD:")

        # PyTorch reference backward (fp32)
        e_ag = e_torch.detach().requires_grad_(True)
        g_ag = g_torch.detach().requires_grad_(True)
        h_ref = torch.nn.functional.silu(e_ag) * g_ag
        h_ref.backward(dw_torch)
        ref_de = e_ag.grad.numpy()
        ref_dg = g_ag.grad.numpy()

        # MLX Composed vjp (fp32)
        dw_mlx = mx.array(dw_torch.numpy())
        h_out, de_out, dg_out = mlx_swiglu_composed_backward(dw_mlx, e_mlx, g_mlx)
        mx.eval(h_out, de_out, dg_out)

        de_diff = np.abs(np.array(de_out) - ref_de)
        dg_diff = np.abs(np.array(dg_out) - ref_dg)
        p_de = de_diff.max() < 1e-4
        p_dg = dg_diff.max() < 1e-4
        all_passed &= (p_de and p_dg)
        print(
            f"    MLX Composed vjp:     de max={de_diff.max():.2e}  dg max={dg_diff.max():.2e}  "
            f"{'PASS' if p_de and p_dg else 'FAIL'}"
        )

        # MLX Compiled vjp (fp32)
        h_out, de_out, dg_out = mlx_swiglu_compiled_backward(dw_mlx, e_mlx, g_mlx)
        mx.eval(h_out, de_out, dg_out)

        de_diff = np.abs(np.array(de_out) - ref_de)
        dg_diff = np.abs(np.array(dg_out) - ref_dg)
        p_de = de_diff.max() < 1e-4
        p_dg = dg_diff.max() < 1e-4
        all_passed &= (p_de and p_dg)
        print(
            f"    MLX Compiled vjp:     de max={de_diff.max():.2e}  dg max={dg_diff.max():.2e}  "
            f"{'PASS' if p_de and p_dg else 'FAIL'}"
        )

        # Fused Metal backward (fp16)
        # The fused kernel returns (h, df, de) where df=dw*silu(e) and de=dw*g*silu'(e)
        # This matches: dg = dw * silu(e) = df, de = dw * g * silu'(e) = de
        # Reference in fp16
        e_ag16 = e_torch.half().detach().requires_grad_(True)
        g_ag16 = g_torch.half().detach().requires_grad_(True)
        h_ref16 = torch.nn.functional.silu(e_ag16) * g_ag16
        h_ref16.backward(dw_torch.half())
        ref_de16 = e_ag16.grad.float().numpy()
        ref_dg16 = g_ag16.grad.float().numpy()

        dw_f16 = dw_mlx.astype(mx.float16) if dw_mlx.dtype != mx.float16 else dw_mlx
        e_f16_bwd = e_mlx.astype(mx.float16) if e_mlx.dtype != mx.float16 else e_mlx
        g_f16_bwd = g_mlx.astype(mx.float16) if g_mlx.dtype != mx.float16 else g_mlx
        h_out, df_out, de_out = metal_module.mlx_swiglu_backward(dw_f16, e_f16_bwd, g_f16_bwd)
        mx.eval(h_out, df_out, de_out)

        # df_out = dg (grad w.r.t. g), de_out = de (grad w.r.t. e)
        de_diff = np.abs(np.array(de_out).astype(np.float32) - ref_de16)
        dg_diff = np.abs(np.array(df_out).astype(np.float32) - ref_dg16)
        p_de = de_diff.max() < 0.05  # fp16 backward tolerance
        p_dg = dg_diff.max() < 0.05
        all_passed &= (p_de and p_dg)
        print(
            f"    Fused Metal bwd:      de max={de_diff.max():.2e}  dg max={dg_diff.max():.2e}  "
            f"{'PASS' if p_de and p_dg else 'FAIL'}"
        )

        print()

    print("=" * 80)
    if all_passed:
        print("ALL CORRECTNESS TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
    print("=" * 80)

    return all_passed


# ==============================================================================
# Main
# ==============================================================================


def main():
    parser = argparse.ArgumentParser(description="SwiGLU Kernel Benchmark Suite")
    parser.add_argument(
        "--perf", action="store_true", help="Run performance benchmarks only"
    )
    parser.add_argument(
        "--correctness", action="store_true", help="Run correctness tests only"
    )
    parser.add_argument(
        "--forward", action="store_true", help="Run forward benchmarks only (implies --perf)"
    )
    parser.add_argument(
        "--backward", action="store_true", help="Run backward benchmarks only (implies --perf)"
    )
    args = parser.parse_args()

    print_header()

    # If --forward or --backward specified, only run perf
    if args.forward or args.backward:
        args.perf = True

    run_perf = not args.correctness or args.perf
    run_correct = not args.perf or args.correctness

    # Determine which passes to benchmark
    do_forward = True
    do_backward = True
    if args.forward and not args.backward:
        do_backward = False
    elif args.backward and not args.forward:
        do_forward = False

    if run_perf:
        run_performance_benchmark(forward=do_forward, backward=do_backward)

    if run_correct:
        run_correctness_tests()

    print("\nDone.")


if __name__ == "__main__":
    main()
