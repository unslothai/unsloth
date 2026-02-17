#!/usr/bin/env python3
"""
Benchmark custom MLX kernels vs MLX compile vs normal MLX (eager).

This script compares:
1. MLX eager mode (baseline)
2. MLX compiled (@mx.compile decorator)
3. MLX fast ops (mx.fast.* - Apple's optimized kernels)

Usage:
    python benchmarks/mlx/benchmark_kernels.py

Requirements:
    - Apple Silicon Mac
    - MLX installed: pip install mlx>=0.6.0
"""

import sys
import time
import argparse
from typing import Callable, Any

import mlx.core as mx


def check_environment() -> bool:
    """Check if we can run MLX benchmarks."""
    if sys.platform != "darwin":
        print("[X] MLX benchmarks require macOS (Apple Silicon)")
        return False
    
    print(f"[OK] MLX version: {mx.__version__ if hasattr(mx, '__version__') else 'unknown'}")
    return True


def benchmark_function(
    func: Callable,
    *args,
    name: str = "Operation",
    iters: int = 100,
    warmup: int = 10,
    **kwargs
) -> float:
    """Benchmark a function and return average time in ms."""
    for _ in range(warmup):
        _ = func(*args, **kwargs)
    mx.eval(func(*args, **kwargs))
    
    start = time.perf_counter()
    for _ in range(iters):
        out = func(*args, **kwargs)
    mx.eval(out)
    elapsed = time.perf_counter() - start
    
    return (elapsed / iters) * 1000


def print_benchmark_header(title: str):
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)


def print_result(name: str, time_ms: float, baseline: float = None):
    speedup = f"{baseline/time_ms:.2f}x" if baseline else ""
    print(f"  {name:<30} {time_ms:>8.3f} ms  {speedup}")


def benchmark_layer_norm(
    B: int = 1, S: int = 512, H: int = 4096,
    iters: int = 100
):
    """Benchmark LayerNorm implementations."""
    print_benchmark_header(f"LayerNorm [B={B}, S={S}, H={H}]")
    print(f"  {'Implementation':<30} {'Time':>12}  {'Speedup'}")
    print("-" * 52)
    
    X = mx.random.normal(shape=(B, S, H))
    W = mx.random.normal(shape=(H,))
    b = mx.random.normal(shape=(H,))
    eps = 1e-5
    
    baseline_time = benchmark_function(
        lambda: mx.fast.layer_norm(X, W, b, eps),
        name="mx.fast.layer_norm", iters=iters
    )
    print_result("mx.fast.layer_norm (Apple)", baseline_time)
    
    @mx.compile
    def compiled_layer_norm(x, w, b, eps):
        mean = mx.mean(x, axis=-1, keepdims=True)
        var = mx.var(x, axis=-1, keepdims=True)
        x_centered = x - mean
        norm = x_centered / mx.sqrt(var + eps)
        return w * norm + b
    
    compiled_time = benchmark_function(
        lambda: compiled_layer_norm(X, W, b, eps),
        name="Compiled LayerNorm", iters=iters
    )
    print_result("mx.compile", compiled_time, baseline_time)
    
    def eager_layer_norm(x, w, b, eps):
        mean = mx.mean(x, axis=-1, keepdims=True)
        var = mx.var(x, axis=-1, keepdims=True)
        x_centered = x - mean
        norm = x_centered / mx.sqrt(var + eps)
        return w * norm + b
    
    eager_time = benchmark_function(
        lambda: eager_layer_norm(X, W, b, eps),
        name="Eager LayerNorm", iters=iters
    )
    print_result("Eager (no compile)", eager_time, baseline_time)


def benchmark_rms_norm(
    B: int = 1, S: int = 512, H: int = 4096,
    iters: int = 100
):
    """Benchmark RMSNorm implementations."""
    print_benchmark_header(f"RMSNorm [B={B}, S={S}, H={H}]")
    print(f"  {'Implementation':<30} {'Time':>12}  {'Speedup'}")
    print("-" * 52)
    
    X = mx.random.normal(shape=(B, S, H))
    W = mx.random.normal(shape=(H,))
    eps = 1e-5
    
    baseline_time = benchmark_function(
        lambda: mx.fast.rms_norm(X, W, eps),
        name="mx.fast.rms_norm", iters=iters
    )
    print_result("mx.fast.rms_norm (Apple)", baseline_time)
    
    @mx.compile
    def compiled_rms_norm(x, w, eps):
        var = mx.mean(x * x, axis=-1, keepdims=True)
        norm = x * mx.rsqrt(var + eps)
        return w * norm
    
    compiled_time = benchmark_function(
        lambda: compiled_rms_norm(X, W, eps),
        name="Compiled RMSNorm", iters=iters
    )
    print_result("mx.compile", compiled_time, baseline_time)
    
    def eager_rms_norm(x, w, eps):
        var = mx.mean(x * x, axis=-1, keepdims=True)
        norm = x * mx.rsqrt(var + eps)
        return w * norm
    
    eager_time = benchmark_function(
        lambda: eager_rms_norm(X, W, eps),
        name="Eager RMSNorm", iters=iters
    )
    print_result("Eager (no compile)", eager_time, baseline_time)


def benchmark_rope(
    B: int = 1, H: int = 32, S: int = 512, D: int = 128,
    iters: int = 100
):
    """Benchmark RoPE implementations."""
    print_benchmark_header(f"RoPE [B={B}, H={H}, S={S}, D={D}]")
    print(f"  {'Implementation':<30} {'Time':>12}  {'Speedup'}")
    print("-" * 52)
    
    Q = mx.random.normal(shape=(B, H, S, D))
    cos = mx.random.normal(shape=(S, D))
    sin = mx.random.normal(shape=(S, D))
    cos = mx.concatenate([cos, cos], axis=-1)
    sin = mx.concatenate([sin, sin], axis=-1)
    
    @mx.compile
    def custom_rope_kernel(q, cos, sin):
        half = q.shape[-1] // 2
        q1 = q[..., :half]
        q2 = q[..., half:]
        cos1 = cos[..., :half]
        cos2 = cos[..., half:]
        sin1 = sin[..., :half]
        sin2 = sin[..., half:]
        out1 = q1 * cos1 - q2 * sin1
        out2 = q2 * cos2 + q1 * sin2
        return mx.concatenate([out1, out2], axis=-1)
    
    custom_time = benchmark_function(
        lambda: custom_rope_kernel(Q, cos, sin),
        name="Custom Kernel", iters=iters
    )
    print_result("Custom (unsloth)", custom_time)
    
    @mx.compile
    def compiled_rope(q, cos, sin):
        half = q.shape[-1] // 2
        q1 = q[..., :half]
        q2 = q[..., half:]
        cos1 = cos[..., :half]
        cos2 = cos[..., half:]
        sin1 = sin[..., :half]
        sin2 = sin[..., half:]
        out1 = q1 * cos1 - q2 * sin1
        out2 = q2 * cos2 + q1 * sin2
        return mx.concatenate([out1, out2], axis=-1)
    
    compiled_time = benchmark_function(
        lambda: compiled_rope(Q, cos, sin),
        name="Compiled RoPE", iters=iters
    )
    print_result("mx.compile", compiled_time, custom_time)
    
    def eager_rope(q, cos, sin):
        half = q.shape[-1] // 2
        q1 = q[..., :half]
        q2 = q[..., half:]
        cos1 = cos[..., :half]
        cos2 = cos[..., half:]
        sin1 = sin[..., :half]
        sin2 = sin[..., half:]
        out1 = q1 * cos1 - q2 * sin1
        out2 = q2 * cos2 + q1 * sin2
        return mx.concatenate([out1, out2], axis=-1)
    
    eager_time = benchmark_function(
        lambda: eager_rope(Q, cos, sin),
        name="Eager (no compile)", iters=iters
    )
    print_result("Eager (no compile)", eager_time, custom_time)


def benchmark_swiglu(
    B: int = 1, S: int = 512, H: int = 4096, hidden: int = 11008,
    iters: int = 100
):
    """Benchmark SwiGLU implementations."""
    print_benchmark_header(f"SwiGLU [B={B}, S={S}, H={H}, hidden={hidden}]")
    print(f"  {'Implementation':<30} {'Time':>12}  {'Speedup'}")
    print("-" * 52)
    
    gate = mx.random.normal(shape=(B, S, hidden))
    up = mx.random.normal(shape=(B, S, hidden))
    
    @mx.compile
    def custom_swiglu_kernel(e, g):
        return (e * mx.sigmoid(e)) * g
    
    custom_time = benchmark_function(
        lambda: custom_swiglu_kernel(gate, up),
        name="Custom SwiGLU", iters=iters
    )
    print_result("Custom (unsloth)", custom_time)
    
    @mx.compile
    def compiled_swiglu(gate, up):
        return (gate * mx.sigmoid(gate)) * up
    
    compiled_time = benchmark_function(
        lambda: compiled_swiglu(gate, up),
        name="mx.compile", iters=iters
    )
    print_result("mx.compile", compiled_time, custom_time)
    
    def eager_swiglu(gate, up):
        return (gate * mx.sigmoid(gate)) * up
    
    eager_time = benchmark_function(
        lambda: eager_swiglu(gate, up),
        name="Eager (no compile)", iters=iters
    )
    print_result("Eager (no compile)", eager_time, custom_time)


def benchmark_swiglu_mlp(
    B: int = 1, S: int = 128, H: int = 4096, hidden: int = 11008,
    iters: int = 100
):
    """Benchmark full SwiGLU MLP."""
    print_benchmark_header(f"SwiGLU MLP [B={B}, S={S}, H={H}, hidden={hidden}]")
    print(f"  {'Implementation':<30} {'Time':>12}  {'Speedup'}")
    print("-" * 52)
    
    X = mx.random.normal(shape=(B, S, H))
    gateW = mx.random.normal(shape=(hidden, H))
    upW = mx.random.normal(shape=(hidden, H))
    downW = mx.random.normal(shape=(H, hidden))
    
    @mx.compile
    def compiled_mlp(x, gateW, upW, downW):
        gate = x @ gateW.T
        up = x @ upW.T
        act = gate * mx.sigmoid(gate) * up
        return act @ downW.T
    
    compiled_time = benchmark_function(
        lambda: compiled_mlp(X, gateW, upW, downW),
        name="Compiled MLP", iters=iters
    )
    print_result("mx.compile", compiled_time)
    
    def eager_mlp(x, gateW, upW, downW):
        gate = x @ gateW.T
        up = x @ upW.T
        act = gate * mx.sigmoid(gate) * up
        return act @ downW.T
    
    eager_time = benchmark_function(
        lambda: eager_mlp(X, gateW, upW, downW),
        name="Eager MLP", iters=iters
    )
    print_result("Eager (no compile)", eager_time, compiled_time)


def benchmark_attention(
    B: int = 1, H: int = 32, S: int = 512, D: int = 128,
    iters: int = 100
):
    """Benchmark attention implementations."""
    print_benchmark_header(f"Attention [B={B}, H={H}, S={S}, D={D}]")
    print(f"  {'Implementation':<30} {'Time':>12}  {'Speedup'}")
    print("-" * 52)
    
    Q = mx.random.normal(shape=(B, H, S, D))
    K = mx.random.normal(shape=(B, H, S, D))
    V = mx.random.normal(shape=(B, H, S, D))
    scale = 1.0 / (D ** 0.5)
    
    baseline_time = benchmark_function(
        lambda: mx.fast.scaled_dot_product_attention(Q, K, V, scale=scale),
        name="mx.fast.sdpa", iters=iters
    )
    print_result("mx.fast.sdpa (Apple)", baseline_time)
    
    @mx.compile
    def compiled_attention(q, k, v, scale):
        scores = (q @ mx.swapaxes(k, -2, -1)) * scale
        attn_weights = mx.softmax(scores, axis=-1)
        return attn_weights @ v
    
    compiled_time = benchmark_function(
        lambda: compiled_attention(Q, K, V, scale),
        name="Compiled Attention", iters=iters
    )
    print_result("mx.compile", compiled_time, baseline_time)
    
    def eager_attention(q, k, v, scale):
        scores = (q @ mx.swapaxes(k, -2, -1)) * scale
        attn_weights = mx.softmax(scores, axis=-1)
        return attn_weights @ v
    
    eager_time = benchmark_function(
        lambda: eager_attention(Q, K, V, scale),
        name="Eager Attention", iters=iters
    )
    print_result("Eager (no compile)", eager_time, baseline_time)


def benchmark_grouped_gemm(
    num_tokens: int = 1024,
    num_experts: int = 8,
    topk: int = 2,
    hidden_dim: int = 4096,
    expert_dim: int = 14336,
    iters: int = 100
):
    """Benchmark grouped GEMM for MoE."""
    print_benchmark_header(
        f"Grouped GEMM [tokens={num_tokens}, experts={num_experts}, topk={topk}]"
    )
    print(f"  {'Implementation':<30} {'Time':>12}  {'Speedup'}")
    print("-" * 52)
    
    tokens_per_expert = num_tokens // num_experts
    total_selected = num_tokens * topk
    X = mx.random.normal(shape=(total_selected, hidden_dim))
    W = mx.random.normal(shape=(num_experts, expert_dim, hidden_dim))
    
    @mx.compile
    def compiled_grouped_gemm(x, W):
        outputs = []
        for i in range(num_experts):
            start_idx = i * tokens_per_expert * topk
            end_idx = (i + 1) * tokens_per_expert * topk
            X_exp = x[start_idx:end_idx]
            W_exp = W[i]
            Y_exp = X_exp @ W_exp.T
            outputs.append(Y_exp)
        return mx.concatenate(outputs, axis=0)
    
    compiled_time = benchmark_function(
        lambda: compiled_grouped_gemm(X, W),
        name="mx.compile", iters=iters
    )
    print_result("mx.compile", compiled_time)
    
    def eager_grouped_gemm(x, W):
        outputs = []
        for i in range(num_experts):
            start_idx = i * tokens_per_expert * topk
            end_idx = (i + 1) * tokens_per_expert * topk
            X_exp = x[start_idx:end_idx]
            W_exp = W[i]
            Y_exp = X_exp @ W_exp.T
            outputs.append(Y_exp)
        return mx.concatenate(outputs, axis=0)
    
    eager_time = benchmark_function(
        lambda: eager_grouped_gemm(X, W),
        name="Eager (no compile)", iters=iters
    )
    print_result("Eager (no compile)", eager_time, compiled_time)


def run_all_benchmarks(args):
    """Run all benchmark suites."""
    print("\n" + "=" * 70)
    print(" MLX Kernels Benchmark Suite")
    print(" Comparing: Eager vs Compiled vs mx.fast ops")
    print("=" * 70)
    print(f"\nIterations: {args.iters}, Warmup: {args.warmup}")
    
    if args.benchmark == "all" or args.benchmark == "layer_norm":
        benchmark_layer_norm(iters=args.iters)
    
    if args.benchmark == "all" or args.benchmark == "rms_norm":
        benchmark_rms_norm(iters=args.iters)
    
    if args.benchmark == "all" or args.benchmark == "rope":
        benchmark_rope(iters=args.iters)
    
    if args.benchmark == "all" or args.benchmark == "swiglu":
        benchmark_swiglu(iters=args.iters)
    
    if args.benchmark == "all" or args.benchmark == "mlp":
        benchmark_swiglu_mlp(iters=args.iters)
    
    if args.benchmark == "all" or args.benchmark == "attention":
        benchmark_attention(iters=args.iters)
    
    if args.benchmark == "all" or args.benchmark == "grouped_gemm":
        benchmark_grouped_gemm(iters=args.iters)
    
    print("\n" + "=" * 70)
    print(" Benchmark Complete")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark MLX kernels: eager vs compiled vs fast ops"
    )
    parser.add_argument(
        "-b", "--benchmark",
        default="all",
        choices=["all", "layer_norm", "rms_norm", "rope", "swiglu", "mlp", "attention", "grouped_gemm"],
        help="Which benchmark to run (default: all)"
    )
    parser.add_argument(
        "-i", "--iters",
        type=int,
        default=100,
        help="Number of iterations (default: 100)"
    )
    parser.add_argument(
        "-w", "--warmup",
        type=int,
        default=10,
        help="Number of warmup iterations (default: 10)"
    )
    
    args = parser.parse_args()
    
    if not check_environment():
        sys.exit(1)
    
    run_all_benchmarks(args)


if __name__ == "__main__":
    main()
