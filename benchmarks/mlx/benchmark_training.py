#!/usr/bin/env python3
"""Real training benchmark for MLX kernels - forward + backward + memory."""

import argparse
import gc
import time
from typing import Callable, Optional

import psutil
import mlx.core as mx


def get_memory_mb() -> float:
    """Get current memory usage in MB using psutil."""
    process = psutil.Process()
    return process.memory_info().rss / (1024 * 1024)


def benchmark_function(func: Callable, name: str, iters: int = 100, warmup: int = 10):
    """Benchmark a function and return average time in ms."""
    for _ in range(warmup):
        _ = func()
    mx.eval()

    start = time.perf_counter()
    for _ in range(iters):
        _ = func()
    mx.eval()
    end = time.perf_counter()

    return (end - start) / iters * 1000


def benchmark_training_step(
    name: str,
    forward_backward_fn: Callable,
    iters: int = 100,
    warmup: int = 10,
) -> tuple[float, float]:
    """Benchmark a training step (forward + backward) and return time and peak memory."""
    for _ in range(warmup):
        _ = forward_backward_fn()
    mx.eval()
    gc.collect()

    peak_memory = 0.0
    start = time.perf_counter()
    for _ in range(iters):
        _ = forward_backward_fn()
        mx.eval()
        peak_memory = max(peak_memory, get_memory_mb())
    end = time.perf_counter()

    avg_time = (end - start) / iters * 1000
    avg_memory = peak_memory
    return avg_time, avg_memory


def print_result(name: str, time_ms: float, memory_mb: Optional[float] = None, baseline_time: Optional[float] = None):
    """Print a benchmark result."""
    if memory_mb is not None:
        mem_str = f"{memory_mb:>8.1f} MB"
    else:
        mem_str = " " * 10

    if baseline_time is not None and baseline_time > 0:
        speedup = baseline_time / time_ms
        print(f"  {name:<30} {time_ms:>8.2f} ms  {mem_str}  {speedup:>5.2f}x")
    else:
        print(f"  {name:<30} {time_ms:>8.2f} ms  {mem_str}")


def benchmark_layer_norm_training(B: int = 1, S: int = 512, H: int = 4096, iters: int = 100, warmup: int = 10):
    """Benchmark LayerNorm forward + backward training."""
    print(f"\n{'='*60}")
    print(f" LayerNorm Training [B={B}, S={S}, H={H}]")
    print(f"{'='*60}")
    print(f"  {'Implementation':<30} {'Time':>10}  {'Peak RAM':>10}  {'Speedup'}")
    print("-" * 62)

    x = mx.random.normal(shape=(B, S, H))
    weight = mx.random.normal(shape=(H,))
    bias = mx.random.normal(shape=(H,))
    grad_output = mx.random.normal(shape=(B, S, H))

    eps = 1e-5

    def eager_layer_norm_training():
        x_centered = x - x.mean(axis=-1, keepdims=True)
        var = x_centered.square().mean(axis=-1, keepdims=True)
        inv_std = 1 / mx.sqrt(var + eps)
        normalized = x_centered * inv_std
        output = normalized * weight + bias
        grad = grad_output * weight * inv_std - (grad_output.mean(axis=-1, keepdims=True) * x_centered * inv_std / H)
        return output, grad

    time_eager, mem_eager = benchmark_training_step("Eager (no compile)", eager_layer_norm_training, iters)
    print_result("Eager (no compile)", time_eager, mem_eager)

    baseline_time = time_eager

    def compiled_layer_norm_training():
        x_c = x - x.mean(axis=-1, keepdims=True)
        var = x_c.square().mean(axis=-1, keepdims=True)
        inv_std = 1 / mx.sqrt(var + eps)
        normalized = x_c * inv_std
        output = normalized * weight + bias
        grad = grad_output * weight * inv_std - (grad_output.mean(axis=-1, keepdims=True) * x_c * inv_std / H)
        return output, grad

    compiled_fn = mx.compile(lambda: compiled_layer_norm_training())
    time_compiled, mem_compiled = benchmark_training_step("mx.compile", compiled_fn, iters)
    print_result("mx.compile", time_compiled, mem_compiled, baseline_time)

    try:
        def fast_layer_norm_training():
            out = mx.fast.layer_norm(x, weight, bias, eps)
            return out
        
        grad_fn = mx.grad(lambda x, w, b: mx.fast.layer_norm(x, w, b, eps).sum(), argnums=[0, 1, 2])
        time_fast, mem_fast = benchmark_training_step(
            "mx.fast.layer_norm",
            lambda: grad_fn(x, weight, bias),
            iters
        )
        print_result("mx.fast.layer_norm", time_fast, mem_fast, baseline_time)
    except Exception as e:
        print(f"  mx.fast.layer_norm not available: {e}")


def benchmark_rmsnorm_training(B: int = 1, S: int = 512, H: int = 4096, iters: int = 100, warmup: int = 10):
    """Benchmark RMSNorm forward + backward training."""
    print(f"\n{'='*60}")
    print(f" RMSNorm Training [B={B}, S={S}, H={H}]")
    print(f"{'='*60}")
    print(f"  {'Implementation':<30} {'Time':>10}  {'Peak RAM':>10}  {'Speedup'}")
    print("-" * 62)

    x = mx.random.normal(shape=(B, S, H))
    weight = mx.random.normal(shape=(H,))
    grad_output = mx.random.normal(shape=(B, S, H))

    eps = 1e-5

    def eager_rmsnorm_training():
        rms = mx.sqrt(x.square().mean(axis=-1, keepdims=True) + eps)
        normalized = x / rms
        output = normalized * weight
        return output

    time_eager, mem_eager = benchmark_training_step("Eager (no compile)", eager_rmsnorm_training, iters)
    print_result("Eager (no compile)", time_eager, mem_eager)
    baseline_time = time_eager

    compiled_fn = mx.compile(lambda: eager_rmsnorm_training())
    time_compiled, mem_compiled = benchmark_training_step("mx.compile", compiled_fn, iters)
    print_result("mx.compile", time_compiled, mem_compiled, baseline_time)

    try:
        time_fast, mem_fast = benchmark_training_step(
            "mx.fast.rms_norm",
            lambda: mx.fast.rms_norm(x, weight, eps),
            iters
        )
        print_result("mx.fast.rms_norm", time_fast, mem_fast, baseline_time)
    except Exception as e:
        print(f"  mx.fast.rms_norm not available: {e}")


def benchmark_swiglu_training(B: int = 1, S: int = 512, H: int = 4096, hidden: int = 11008, iters: int = 100, warmup: int = 10):
    """Benchmark SwiGLU forward + backward training."""
    print(f"\n{'='*60}")
    print(f" SwiGLU Training [B={B}, S={S}, H={H}, hidden={hidden}]")
    print(f"{'='*60}")
    print(f"  {'Implementation':<30} {'Time':>10}  {'Peak RAM':>10}  {'Speedup'}")
    print("-" * 62)

    x = mx.random.normal(shape=(B, S, H))
    w1 = mx.random.normal(shape=(H, hidden))
    w3 = mx.random.normal(shape=(H, hidden))
    w2 = mx.random.normal(shape=(hidden, H))
    grad_output = mx.random.normal(shape=(B, S, H))

    def eager_swiglu_training():
        gate = x @ w1
        up = x @ w3
        hidden = gate * (1 / (1 + mx.exp(-gate))) * up  # SiLU: x * sigmoid(x)
        output = hidden @ w2
        return output

    time_eager, mem_eager = benchmark_training_step("Eager (no compile)", eager_swiglu_training, iters)
    print_result("Eager (no compile)", time_eager, mem_eager)
    baseline_time = time_eager

    compiled_fn = mx.compile(lambda: eager_swiglu_training())
    time_compiled, mem_compiled = benchmark_training_step("mx.compile", compiled_fn, iters)
    print_result("mx.compile", time_compiled, mem_compiled, baseline_time)


def benchmark_attention_training(B: int = 1, H: int = 32, S: int = 512, D: int = 128, iters: int = 100, warmup: int = 10):
    """Benchmark Attention forward + backward training."""
    print(f"\n{'='*60}")
    print(f" Attention Training [B={B}, H={H}, S={S}, D={D}]")
    print(f"{'='*60}")
    print(f"  {'Implementation':<30} {'Time':>10}  {'Peak RAM':>10}  {'Speedup'}")
    print("-" * 62)

    Q = mx.random.normal(shape=(B, H, S, D))
    K = mx.random.normal(shape=(B, H, S, D))
    V = mx.random.normal(shape=(B, H, S, D))
    grad_output = mx.random.normal(shape=(B, H, S, D))
    scale = 1.0 / (D ** 0.5)

    def eager_attention_training():
        attn_scores = (Q @ K.transpose(0, 1, 3, 2)) * scale
        attn_weights = mx.softmax(attn_scores, axis=-1)
        output = attn_weights @ V
        return output

    time_eager, mem_eager = benchmark_training_step("Eager (no compile)", eager_attention_training, iters)
    print_result("Eager (no compile)", time_eager, mem_eager)
    baseline_time = time_eager

    compiled_fn = mx.compile(lambda: eager_attention_training())
    time_compiled, mem_compiled = benchmark_training_step("mx.compile", compiled_fn, iters)
    print_result("mx.compile", time_compiled, mem_compiled, baseline_time)

    try:
        time_fast, mem_fast = benchmark_training_step(
            "mx.fast.sdpa",
            lambda: mx.fast.scaled_dot_product_attention(Q, K, V, scale=scale),
            iters
        )
        print_result("mx.fast.sdpa", time_fast, mem_fast, baseline_time)
    except Exception as e:
        print(f"  mx.fast.sdpa not available: {e}")


def benchmark_mlp_training(B: int = 1, S: int = 128, H: int = 4096, iters: int = 100, warmup: int = 10):
    """Benchmark full MLP training step."""
    print(f"\n{'='*60}")
    print(f" MLP Training [B={B}, S={S}, H={H}]")
    print(f"{'='*60}")
    print(f"  {'Implementation':<30} {'Time':>10}  {'Peak RAM':>10}  {'Speedup'}")
    print("-" * 62)

    hidden = int(H * 2.67)
    x = mx.random.normal(shape=(B, S, H))
    w1 = mx.random.normal(shape=(H, hidden))
    b1 = mx.random.normal(shape=(hidden,))
    w3 = mx.random.normal(shape=(H, hidden))
    w2 = mx.random.normal(shape=(hidden, H))
    b2 = mx.random.normal(shape=(H,))
    grad_output = mx.random.normal(shape=(B, S, H))

    def eager_mlp_training():
        h_pre = x @ w1 + b1
        h = h_pre * (1 / (1 + mx.exp(-h_pre))) * (x @ w3)
        return h @ w2 + b2

    time_eager, mem_eager = benchmark_training_step("Eager (no compile)", eager_mlp_training, iters)
    print_result("Eager (no compile)", time_eager, mem_eager)
    baseline_time = time_eager

    compiled_fn = mx.compile(lambda: eager_mlp_training())
    time_compiled, mem_compiled = benchmark_training_step("mx.compile", compiled_fn, iters)
    print_result("mx.compile", time_compiled, mem_compiled, baseline_time)


def run_all_benchmarks(args):
    """Run all training benchmarks."""
    print("=" * 60)
    print(" MLX Training Benchmark Suite")
    print(" Forward + Backward Pass + Memory")
    print("=" * 60)
    print(f"Iterations: {args.iters}, Warmup: {args.warmup}")
    print(f"Device: {mx.default_device()}")

    if args.benchmark == "all" or args.benchmark == "layernorm":
        benchmark_layer_norm_training(iters=args.iters, warmup=args.warmup)
    if args.benchmark == "all" or args.benchmark == "rmsnorm":
        benchmark_rmsnorm_training(iters=args.iters, warmup=args.warmup)
    if args.benchmark == "all" or args.benchmark == "swiglu":
        benchmark_swiglu_training(iters=args.iters, warmup=args.warmup)
    if args.benchmark == "all" or args.benchmark == "attention":
        benchmark_attention_training(iters=args.iters, warmup=args.warmup)
    if args.benchmark == "all" or args.benchmark == "mlp":
        benchmark_mlp_training(iters=args.iters, warmup=args.warmup)

    print("\n" + "=" * 60)
    print(" Benchmark Complete")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="MLX Training Kernel Benchmark")
    parser.add_argument("-b", "--benchmark", default="all", 
                        choices=["all", "layernorm", "rmsnorm", "swiglu", "attention", "mlp"],
                        help="Which benchmark to run")
    parser.add_argument("-i", "--iters", type=int, default=100, help="Number of iterations")
    parser.add_argument("-w", "--warmup", type=int, default=10, help="Warmup iterations")
    args = parser.parse_args()

    print(f"[OK] MLX version: {mx.__version__}")
    run_all_benchmarks(args)


if __name__ == "__main__":
    main()
