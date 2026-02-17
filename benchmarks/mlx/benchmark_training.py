#!/usr/bin/env python3
"""Real training benchmark for MLX kernels - forward + backward + memory."""

import argparse
import gc
import time
from typing import Callable, Optional

import mlx.core as mx


def get_vram_mb() -> float:
    """Get peak VRAM usage in MB using MLX."""
    # Reset peak memory to get the peak of the current batch
    mx.metal.reset_peak_memory()
    return mx.metal.get_peak_memory() / (1024 * 1024)


def benchmark_training_step(
    name: str,
    forward_backward_fn: Callable,
    iters: int = 100,
    warmup: int = 10,
) -> tuple[float, float]:
    """Benchmark a training step (forward + backward) and return average time and peak VRAM."""
    
    # Warmup - compile and cache everything
    for _ in range(warmup):
        _ = forward_backward_fn()
    mx.eval()
    gc.collect()

    # Reset peak and measure
    mx.metal.reset_peak_memory()
    start = time.perf_counter()
    for _ in range(iters):
        _ = forward_backward_fn()
    mx.eval()
    end = time.perf_counter()
    
    peak_vram = mx.metal.get_peak_memory() / (1024 * 1024)
    avg_time = (end - start) / iters * 1000
    
    return avg_time, peak_vram


def print_result(name: str, time_ms: float, memory_mb: Optional[float] = None, baseline_time: Optional[float] = None):
    """Print a benchmark result."""
    if memory_mb is not None:
        mem_str = f"{memory_mb:>8.1f} MB VRAM"
    else:
        mem_str = " " * 15

    if baseline_time is not None and baseline_time > 0:
        speedup = baseline_time / time_ms
        print(f"  {name:<30} {time_ms:>8.3f} ms  {mem_str}  {speedup:>5.2f}x")
    else:
        print(f"  {name:<30} {time_ms:>8.3f} ms  {mem_str}")


def benchmark_layer_norm_training(B: int = 4, S: int = 2048, H: int = 4096, iters: int = 100, warmup: int = 10):
    """Benchmark LayerNorm forward + backward training."""
    print(f"\n{'='*60}")
    print(f" LayerNorm Training [B={B}, S={S}, H={H}]")
    print(f"{'='*60}")
    print(f"  {'Implementation':<30} {'Time':>10}  {'Peak RAM':>15}  {'Speedup'}")
    print("-" * 75)

    x = mx.random.normal(shape=(B, S, H))
    weight = mx.random.normal(shape=(H,))
    bias = mx.random.normal(shape=(H,))
    eps = 1e-5

    def eager_layer_norm():
        x_centered = x - x.mean(axis=-1, keepdims=True)
        var = x_centered.square().mean(axis=-1, keepdims=True)
        inv_std = 1 / mx.sqrt(var + eps)
        output = (x_centered * inv_std) * weight + bias
        return output

    # Full backward pass benchmark
    grad_fn_eager = mx.grad(lambda x, w, b: eager_layer_norm().sum(), argnums=[0, 1, 2])
    
    time_eager, mem_eager = benchmark_training_step("Eager (no compile)", lambda: grad_fn_eager(x, weight, bias), iters, warmup)
    print_result("Eager (no compile)", time_eager, mem_eager)
    baseline_time = time_eager

    compiled_fn = mx.compile(grad_fn_eager)
    time_compiled, mem_compiled = benchmark_training_step("mx.compile", lambda: compiled_fn(x, weight, bias), iters, warmup)
    print_result("mx.compile", time_compiled, mem_compiled, baseline_time)

    try:
        def fast_ln_fn(x, w, b):
            return mx.fast.layer_norm(x, w, b, eps)
        
        grad_fn_fast = mx.grad(lambda x, w, b: fast_ln_fn(x, w, b).sum(), argnums=[0, 1, 2])
        time_fast, mem_fast = benchmark_training_step(
            "mx.fast.layer_norm",
            lambda: grad_fn_fast(x, weight, bias),
            iters,
            warmup
        )
        print_result("mx.fast.layer_norm", time_fast, mem_fast, baseline_time)
    except Exception as e:
        print(f"  mx.fast.layer_norm not available: {e}")


def benchmark_rmsnorm_training(B: int = 4, S: int = 2048, H: int = 4096, iters: int = 100, warmup: int = 10):
    """Benchmark RMSNorm forward + backward training."""
    print(f"\n{'='*60}")
    print(f" RMSNorm Training [B={B}, S={S}, H={H}]")
    print(f"{'='*60}")
    print(f"  {'Implementation':<30} {'Time':>10}  {'Peak RAM':>15}  {'Speedup'}")
    print("-" * 75)

    x = mx.random.normal(shape=(B, S, H))
    weight = mx.random.normal(shape=(H,))
    eps = 1e-5

    def eager_rms_norm():
        rms = mx.sqrt(x.square().mean(axis=-1, keepdims=True) + eps)
        return (x / rms) * weight

    grad_fn = mx.grad(lambda x, w: eager_rms_norm().sum(), argnums=[0, 1])

    time_eager, mem_eager = benchmark_training_step("Eager (no compile)", lambda: grad_fn(x, weight), iters, warmup)
    print_result("Eager (no compile)", time_eager, mem_eager)
    baseline_time = time_eager

    compiled_fn = mx.compile(grad_fn)
    time_compiled, mem_compiled = benchmark_training_step("mx.compile", lambda: compiled_fn(x, weight), iters, warmup)
    print_result("mx.compile", time_compiled, mem_compiled, baseline_time)

    try:
        grad_fn_fast = mx.grad(lambda x, w: mx.fast.rms_norm(x, w, eps).sum(), argnums=[0, 1])
        time_fast, mem_fast = benchmark_training_step(
            "mx.fast.rms_norm",
            lambda: grad_fn_fast(x, weight),
            iters,
            warmup
        )
        print_result("mx.fast.rms_norm", time_fast, mem_fast, baseline_time)
    except Exception as e:
        print(f"  mx.fast.rms_norm not available: {e}")


def benchmark_swiglu_training(B: int = 4, S: int = 2048, H: int = 4096, hidden: int = 11008, iters: int = 100, warmup: int = 10):
    """Benchmark SwiGLU forward + backward training."""
    print(f"\n{'='*60}")
    print(f" SwiGLU Training [B={B}, S={S}, H={H}, hidden={hidden}]")
    print(f"{'='*60}")
    print(f"  {'Implementation':<30} {'Time':>10}  {'Peak RAM':>15}  {'Speedup'}")
    print("-" * 75)

    x = mx.random.normal(shape=(B, S, H))
    w1 = mx.random.normal(shape=(H, hidden))
    w3 = mx.random.normal(shape=(H, hidden))
    w2 = mx.random.normal(shape=(hidden, H))

    def eager_swiglu(x, w1, w3, w2):
        gate = x @ w1
        up = x @ w3
        silu = gate * mx.sigmoid(gate)
        return (silu * up) @ w2

    grad_fn = mx.grad(lambda x, w1, w3, w2: eager_swiglu(x, w1, w3, w2).sum(), argnums=[0, 1, 2, 3])

    time_eager, mem_eager = benchmark_training_step("Eager (no compile)", lambda: grad_fn(x, w1, w3, w2), iters, warmup)
    print_result("Eager (no compile)", time_eager, mem_eager)
    baseline_time = time_eager

    compiled_fn = mx.compile(grad_fn)
    time_compiled, mem_compiled = benchmark_training_step("mx.compile", lambda: compiled_fn(x, w1, w3, w2), iters, warmup)
    print_result("mx.compile", time_compiled, mem_compiled, baseline_time)


def benchmark_attention_training(B: int = 4, H: int = 32, S: int = 1024, D: int = 128, iters: int = 100, warmup: int = 10):
    """Benchmark Attention forward + backward training."""
    print(f"\n{'='*60}")
    print(f" Attention Training [B={B}, H={H}, S={S}, D={D}]")
    print(f"{'='*60}")
    print(f"  {'Implementation':<30} {'Time':>10}  {'Peak RAM':>15}  {'Speedup'}")
    print("-" * 75)

    Q = mx.random.normal(shape=(B, H, S, D))
    K = mx.random.normal(shape=(B, H, S, D))
    V = mx.random.normal(shape=(B, H, S, D))
    scale = 1.0 / (D ** 0.5)

    def eager_attention(q, k, v):
        attn_scores = (q @ k.transpose(0, 1, 3, 2)) * scale
        attn_weights = mx.softmax(attn_scores, axis=-1)
        return attn_weights @ v

    grad_fn = mx.grad(lambda q, k, v: eager_attention(q, k, v).sum(), argnums=[0, 1, 2])

    time_eager, mem_eager = benchmark_training_step("Eager (no compile)", lambda: grad_fn(Q, K, V), iters, warmup)
    print_result("Eager (no compile)", time_eager, mem_eager)
    baseline_time = time_eager

    compiled_fn = mx.compile(grad_fn)
    time_compiled, mem_compiled = benchmark_training_step("mx.compile", lambda: compiled_fn(Q, K, V), iters, warmup)
    print_result("mx.compile", time_compiled, mem_compiled, baseline_time)

    try:
        grad_fn_fast = mx.grad(lambda q, k, v: mx.fast.scaled_dot_product_attention(q, k, v, scale=scale).sum(), argnums=[0, 1, 2])
        time_fast, mem_fast = benchmark_training_step(
            "mx.fast.sdpa",
            lambda: grad_fn_fast(Q, K, V),
            iters,
            warmup
        )
        print_result("mx.fast.sdpa", time_fast, mem_fast, baseline_time)
    except Exception as e:
        print(f"  mx.fast.sdpa not available: {e}")


def run_all_benchmarks(args):
    """Run all training benchmarks."""
    print("=" * 60)
    print(" MLX Training Benchmark Suite (Corrected)")
    print(" Forward + Backward Pass + VRAM Tracking")
    print("=" * 60)
    print(f"Iterations: {args.iters}, Warmup: {args.warmup}")
    print(f"Device: {mx.default_device()}")

    # Higher defaults for M4
    b, s = args.batch_size, args.seq_len

    if args.benchmark == "all" or args.benchmark == "layernorm":
        benchmark_layer_norm_training(B=b, S=s, iters=args.iters, warmup=args.warmup)
    if args.benchmark == "all" or args.benchmark == "rmsnorm":
        benchmark_rmsnorm_training(B=b, S=s, iters=args.iters, warmup=args.warmup)
    if args.benchmark == "all" or args.benchmark == "swiglu":
        benchmark_swiglu_training(B=b, S=s, iters=args.iters, warmup=args.warmup)
    if args.benchmark == "all" or args.benchmark == "attention":
        benchmark_attention_training(B=b, S=s if s <= 2048 else 2048, iters=args.iters, warmup=args.warmup)

    print("\n" + "=" * 60)
    print(" Benchmark Complete")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="MLX Training Kernel Benchmark")
    parser.add_argument("-b", "--benchmark", default="all", 
                        choices=["all", "layernorm", "rmsnorm", "swiglu", "attention", "mlp", "finetune"],
                        help="Which benchmark to run")
    parser.add_argument("-i", "--iters", type=int, default=50, help="Number of iterations")
    parser.add_argument("-w", "--warmup", type=int, default=10, help="Warmup iterations")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--seq-len", type=int, default=2048, help="Sequence length")
    args = parser.parse_args()

    print(f"[OK] MLX version: {mx.__version__}")
    run_all_benchmarks(args)


if __name__ == "__main__":
    main()
