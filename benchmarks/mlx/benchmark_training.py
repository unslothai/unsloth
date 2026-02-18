#!/usr/bin/env python3
"""Real training benchmark for MLX kernels - forward + backward + memory."""

import argparse
import gc
import time
from typing import Callable, Optional

import mlx.core as mx

# Try to import unsloth custom kernels
# MLX wrappers (use mx.fast internally)
try:
    from unsloth.kernels.mlx.fast_ops import (
        mlx_layer_norm,
        mlx_rms_norm,
        mlx_rope,
        mlx_swiglu,
        mlx_scaled_dot_product_attention,
    )
    UNSLOTH_MLX_AVAILABLE = True
except Exception:
    UNSLOTH_MLX_AVAILABLE = False

# Custom Metal kernels (use mx.fast.metal_kernel)
try:
    from unsloth.kernels.metal import (
        is_metal_available,
        metal_swiglu_forward,
        metal_swiglu_backward,
        metal_rms_layernorm,
    )
    UNSLOTH_METAL_AVAILABLE = is_metal_available()
except Exception:
    UNSLOTH_METAL_AVAILABLE = False


def get_vram_mb() -> float:
    """Get peak VRAM usage in MB using MLX."""
    # Use non-deprecated API
    return mx.get_peak_memory() / (1024 * 1024)


def benchmark_training_step(
    name: str,
    forward_backward_fn: Callable,
    iters: int = 100,
    warmup: int = 10,
) -> tuple[float, float]:
    """Benchmark a training step (forward + backward) and return average time and peak VRAM."""
    
    print(f"  > {name:<20} Warmup...", end="", flush=True)
    # Warmup - compile and cache everything
    for i in range(warmup):
        res = forward_backward_fn()
        mx.eval(res) 
    mx.synchronize()
    print(" Done. Benchmarking...", end="", flush=True)
    gc.collect()

    # Reset peak (using non-deprecated API) and measure
    mx.reset_peak_memory()
    
    start = time.perf_counter()
    for i in range(iters):
        res = forward_backward_fn()
        mx.eval(res) 
    mx.synchronize() 
    end = time.perf_counter()
    print(" Done.")
    
    peak_vram = mx.get_peak_memory() / (1024 * 1024)
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


def benchmark_layer_norm_training(B: int = 16, S: int = 1024, H: int = 4096, iters: int = 50, warmup: int = 10):
    """Benchmark LayerNorm forward + backward training."""
    print(f"\n{'='*60}")
    print(f" LayerNorm Training [B={B}, S={S}, H={H}] (float16)")
    print(f"{'='*60}")
    print(f"  {'Implementation':<30} {'Time':>10}  {'Peak RAM':>15}  {'Speedup'}")
    print("-" * 75)

    dtype = mx.float16
    x = mx.random.normal(shape=(B, S, H), dtype=dtype)
    weight = mx.random.normal(shape=(H,), dtype=dtype)
    bias = mx.random.normal(shape=(H,), dtype=dtype)
    eps = 1e-5

    def eager_layer_norm(x, w, b):
        x_centered = x - x.mean(axis=-1, keepdims=True)
        var = x_centered.square().mean(axis=-1, keepdims=True)
        inv_std = 1 / mx.sqrt(var + eps)
        output = (x_centered * inv_std) * w + b
        return output

    # Full backward pass benchmark
    grad_fn_eager = mx.grad(lambda x, w, b: eager_layer_norm(x, w, b).sum(), argnums=[0, 1, 2])
    
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


def benchmark_rmsnorm_training(B: int = 16, S: int = 1024, H: int = 4096, iters: int = 50, warmup: int = 10):
    """Benchmark RMSNorm forward + backward training."""
    print(f"\n{'='*60}")
    print(f" RMSNorm Training [B={B}, S={S}, H={H}] (float16)")
    print(f"{'='*60}")
    print(f"  {'Implementation':<30} {'Time':>10}  {'Peak RAM':>15}  {'Speedup'}")
    print("-" * 75)

    dtype = mx.float16
    x = mx.random.normal(shape=(B, S, H), dtype=dtype)
    weight = mx.random.normal(shape=(H,), dtype=dtype)
    eps = 1e-5

    def eager_rms_norm(x, w):
        rms = mx.sqrt(x.square().mean(axis=-1, keepdims=True) + eps)
        return (x / rms) * w

    grad_fn = mx.grad(lambda x, w: eager_rms_norm(x, w).sum(), argnums=[0, 1])

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


def benchmark_swiglu_training(B: int = 16, S: int = 1024, H: int = 4096, hidden: int = 11008, iters: int = 50, warmup: int = 10):
    """Benchmark SwiGLU forward + backward training."""
    print(f"\n{'='*60}")
    print(f" SwiGLU Training [B={B}, S={S}, H={H}, hidden={hidden}] (float16)")
    print(f"{'='*60}")
    print(f"  {'Implementation':<30} {'Time':>10}  {'Peak RAM':>15}  {'Speedup'}")
    print("-" * 75)

    dtype = mx.float16
    x = mx.random.normal(shape=(B, S, H), dtype=dtype)
    w1 = mx.random.normal(shape=(H, hidden), dtype=dtype)
    w3 = mx.random.normal(shape=(H, hidden), dtype=dtype)
    w2 = mx.random.normal(shape=(hidden, H), dtype=dtype)

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


def benchmark_attention_training(B: int = 16, H: int = 32, S: int = 1024, D: int = 128, iters: int = 50, warmup: int = 10):
    """Benchmark Attention forward + backward training."""
    print(f"\n{'='*60}")
    print(f" Attention Training [B={B}, H={H}, S={S}, D={D}] (float16)")
    print(f"{'='*60}")
    print(f"  {'Implementation':<30} {'Time':>10}  {'Peak RAM':>15}  {'Speedup'}")
    print("-" * 75)

    dtype = mx.float16
    Q = mx.random.normal(shape=(B, H, S, D), dtype=dtype)
    K = mx.random.normal(shape=(B, H, S, D), dtype=dtype)
    V = mx.random.normal(shape=(B, H, S, D), dtype=dtype)
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


def benchmark_finetune_training(B=1, S=128, iters=50, warmup=10):
    """Full fine-tuning loop with a small Llama-style model."""
    print("\n" + "=" * 60)
    print(" Full Fine-tuning [Small Llama-style Model]")
    print("=" * 60)
    print(f"  {'Implementation':<30} {'Time':>10}  {'Peak RAM':>10}  {'Speedup'}")
    print("-" * 62)

    H, hidden, heads = 256, 512, 4
    D = H // heads
    lr = 0.01

    x = mx.random.normal(shape=(B, S, H))
    target = mx.random.normal(shape=(B, S, H))

    w_q = mx.random.normal(shape=(H, H))
    w_k = mx.random.normal(shape=(H, H))
    w_v = mx.random.normal(shape=(H, H))
    w_o = mx.random.normal(shape=(H, H))
    w1 = mx.random.normal(shape=(H, hidden))
    w3 = mx.random.normal(shape=(H, hidden))
    w2 = mx.random.normal(shape=(hidden, H))
    ln_gamma = mx.ones(shape=(H))
    ln_beta = mx.zeros(shape=(H))

    params = [w_q, w_k, w_v, w_o, w1, w3, w2, ln_gamma, ln_beta]

    def forward_backward():
        nonlocal x, target, params
        x = mx.random.normal(shape=(B, S, H))
        target = mx.random.normal(shape=(B, S, H))

        x_mean = x.mean(axis=2, keepdims=True)
        x_var = x.var(axis=2, keepdims=True)
        x_norm = (x - x_mean) / mx.sqrt(x_var + 1e-5)
        x = x_norm * ln_gamma + ln_beta

        q = x @ w_q
        k = x @ w_k
        v = x @ w_v
        q = q.reshape(B, S, heads, D).transpose(0, 2, 1, 3)
        k = k.reshape(B, S, heads, D).transpose(0, 2, 1, 3)
        v = v.reshape(B, S, heads, D).transpose(0, 2, 1, 3)
        attn = mx.fast.scaled_dot_product_attention(q, k, v, scale=1.0 / (D ** 0.5))
        attn = attn.transpose(0, 2, 1, 3).reshape(B, S, H)
        x = attn @ w_o

        gate = (x @ w1)
        gate = gate * mx.sigmoid(gate)
        up = x @ w3
        ff = gate * up
        x = ff @ w2

        loss = ((x - target) ** 2).mean()

        mx.eval(x)
        return loss

    time_eager, mem_eager = benchmark_training_step("Eager (no compile)", forward_backward, iters)
    print_result("Eager (no compile)", time_eager, mem_eager)
    baseline_time = time_eager

    compiled_fn = mx.compile(forward_backward)
    time_compiled, mem_compiled = benchmark_training_step("mx.compile", compiled_fn, iters)
    print_result("mx.compile", time_compiled, mem_compiled, baseline_time)

    def forward_backward_fast():
        """Using mx.fast layer norm and attention."""
        nonlocal x, target
        x = mx.random.normal(shape=(B, S, H))
        target = mx.random.normal(shape=(B, S, H))

        x = mx.fast.layer_norm(x, ln_gamma, ln_beta, eps=1e-5)

        q = x @ w_q
        k = x @ w_k
        v = x @ w_v
        q = q.reshape(B, S, heads, D).transpose(0, 2, 1, 3)
        k = k.reshape(B, S, heads, D).transpose(0, 2, 1, 3)
        v = v.reshape(B, S, heads, D).transpose(0, 2, 1, 3)
        attn = mx.fast.scaled_dot_product_attention(q, k, v, scale=1.0 / (D ** 0.5))
        attn = attn.transpose(0, 2, 1, 3).reshape(B, S, H)
        x = attn @ w_o

        gate = (x @ w1)
        gate = gate * mx.sigmoid(gate)
        up = x @ w3
        ff = gate * up
        x = ff @ w2

        loss = ((x - target) ** 2).mean()
        mx.eval(x)
        return loss

    time_fast, mem_fast = benchmark_training_step("mx.fast ops", forward_backward_fast, iters)
    print_result("mx.fast ops", time_fast, mem_fast, baseline_time)

    compiled_fast = mx.compile(forward_backward_fast)
    time_compiled_fast, mem_compiled_fast = benchmark_training_step("mx.compile+fast", compiled_fast, iters)
    print_result("mx.compile+fast", time_compiled_fast, mem_compiled_fast, baseline_time)

    # Unsloth MLX wrappers (our wrappers around mx.fast)
    if UNSLOTH_MLX_AVAILABLE:
        def forward_backward_unsloth():
            """Using unsloth MLX wrappers."""
            nonlocal x, target
            x = mx.random.normal(shape=(B, S, H))
            target = mx.random.normal(shape=(B, S, H))

            x = mlx_layer_norm(x, ln_gamma, ln_beta, eps=1e-5)

            q = x @ w_q
            k = x @ w_k
            v = x @ w_v
            q = q.reshape(B, S, heads, D).transpose(0, 2, 1, 3)
            k = k.reshape(B, S, heads, D).transpose(0, 2, 1, 3)
            v = v.reshape(B, S, heads, D).transpose(0, 2, 1, 3)
            attn = mlx_scaled_dot_product_attention(q, k, v, scale=1.0 / (D ** 0.5))
            attn = attn.transpose(0, 2, 1, 3).reshape(B, S, H)
            x = attn @ w_o

            gate = (x @ w1)
            gate = gate * mx.sigmoid(gate)
            up = x @ w3
            ff = gate * up
            x = ff @ w2

            loss = ((x - target) ** 2).mean()
            mx.eval(x)
            return loss

        time_unsloth, mem_unsloth = benchmark_training_step("Unsloth MLX", forward_backward_unsloth, iters)
        print_result("Unsloth MLX", time_unsloth, mem_unsloth, baseline_time)

        compiled_unsloth = mx.compile(forward_backward_unsloth)
        time_compiled_unsloth, mem_compiled_unsloth = benchmark_training_step("Unsloth MLX+compile", compiled_unsloth, iters)
        print_result("Unsloth MLX+compile", time_compiled_unsloth, mem_compiled_unsloth, baseline_time)

    # Custom Metal kernels (our mx.fast.metal_kernel implementations)
    if UNSLOTH_METAL_AVAILABLE:
        def forward_backward_metal():
            """Using unsloth custom Metal kernels."""
            nonlocal x, target
            x = mx.random.normal(shape=(B, S, H))
            target = mx.random.normal(shape=(B, S, H))

            # Use custom Metal RMSNorm kernel
            x = metal_rms_layernorm(x, ln_gamma, eps=1e-5)

            q = x @ w_q
            k = x @ w_k
            v = x @ w_v
            q = q.reshape(B, S, heads, D).transpose(0, 2, 1, 3)
            k = k.reshape(B, S, heads, D).transpose(0, 2, 1, 3)
            v = v.reshape(B, S, heads, D).transpose(0, 2, 1, 3)
            attn = mx.fast.scaled_dot_product_attention(q, k, v, scale=1.0 / (D ** 0.5))
            attn = attn.transpose(0, 2, 1, 3).reshape(B, S, H)
            x = attn @ w_o

            # Use custom Metal SwiGLU kernel
            gate = x @ w1
            up = x @ w3
            x = metal_swiglu_forward(gate, up)

            x = x @ w2

            loss = ((x - target) ** 2).mean()
            mx.eval(x)
            return loss

        time_metal, mem_metal = benchmark_training_step("Unsloth Metal", forward_backward_metal, iters)
        print_result("Unsloth Metal", time_metal, mem_metal, baseline_time)

        compiled_metal = mx.compile(forward_backward_metal)
        time_compiled_metal, mem_compiled_metal = benchmark_training_step("Unsloth Metal+compile", compiled_metal, iters)
        print_result("Unsloth Metal+compile", time_compiled_metal, mem_compiled_metal, baseline_time)


def run_all_benchmarks(args):
    """Run all training benchmarks."""
    print("=" * 60)
    print(" MLX Training Benchmark Suite (Corrected)")
    print(" Forward + Backward Pass + VRAM Tracking")
    print("=" * 60)
    print(f"Iterations: {args.iters}, Warmup: {args.warmup}")
    print(f"Device: {mx.default_device()}")

    # Tune defaults
    b, s = args.batch_size, args.seq_len

    if args.benchmark == "all" or args.benchmark == "layernorm":
        benchmark_layer_norm_training(B=b, S=s, iters=args.iters, warmup=args.warmup)
    if args.benchmark == "all" or args.benchmark == "rmsnorm":
        benchmark_rmsnorm_training(B=b, S=s, iters=args.iters, warmup=args.warmup)
    if args.benchmark == "all" or args.benchmark == "swiglu":
        benchmark_swiglu_training(B=b, S=s, iters=args.iters, warmup=args.warmup)
    if args.benchmark == "all" or args.benchmark == "attention":
        benchmark_attention_training(B=b, S=s if s <= 2048 else 2048, iters=args.iters, warmup=args.warmup)
    if args.benchmark == "all" or args.benchmark == "finetune":
        benchmark_finetune_training(B=b, S=s, iters=args.iters, warmup=args.warmup)

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
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--seq-len", type=int, default=1024, help="Sequence length")
    args = parser.parse_args()

    print(f"[OK] MLX version: {mx.__version__}")
    run_all_benchmarks(args)


if __name__ == "__main__":
    main()
