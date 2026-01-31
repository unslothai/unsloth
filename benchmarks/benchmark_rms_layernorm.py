import torch
import mlx.core as mx
import numpy as np
import time
import os
import sys
import platform
import argparse

# Ensure we can import unsloth
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from unsloth.kernels.mlx.bridge import torch_to_mlx, mlx_to_torch, mlx_context
from unsloth.kernels.rms_layernorm import fast_rms_layernorm


def print_header():
    print("=" * 75)
    print("RMSLayerNorm Kernel Benchmark Suite")
    print("=" * 75)
    print(f"Platform:  {platform.platform()}")
    print(f"Processor: {platform.processor()}")
    print(f"PyTorch:   {torch.__version__}")
    try:
        import mlx.core as mx

        print(f"MLX:       {mx.__version__}")
    except:
        print("MLX:       Not installed")
    print("=" * 75)
    print()


def benchmark_fn(fn, warmup = 10, iterations = 50):
    # Warmup
    for _ in range(warmup):
        res = fn()
        if isinstance(res, (mx.array, list, tuple)):
            mx.eval(res)

    if platform.system() == "Darwin":
        torch.mps.synchronize()
        mx.eval()

    start = time.perf_counter()
    for _ in range(iterations):
        res = fn()
        if isinstance(res, (mx.array, list, tuple)):
            mx.eval(res)

    if platform.system() == "Darwin":
        torch.mps.synchronize()
        mx.eval()
    end = time.perf_counter()

    return (end - start) * 1000 / iterations


def calculate_throughput(elements, time_ms, dtype_size = 2):
    # RMSNorm is memory bound. Read X, W. Write Y. r is secondary.
    # Total bytes = elements * dtype_size * 2 (read X, write Y) + dim * dtype_size (read W)
    # Plus whatever internal overhead.
    # We use a simple elements * dtype_size * 2 / time
    return (elements * dtype_size * 2) / (time_ms / 1000) / 1e9


def run_performance_benchmark():
    if platform.system() != "Darwin":
        print("Performance benchmarks only available on macOS (Apple Silicon).")
        return

    import unsloth.kernels.metal.rms_layernorm as metal_module

    # Common configurations
    configs = [
        {"name": "Llama-3 8B (inference)", "batch": 1, "seq": 2048, "dim": 4096},
        {"name": "Llama-3 8B (training)", "batch": 4, "seq": 512, "dim": 4096},
        {"name": "Long context 8K", "batch": 1, "seq": 8192, "dim": 8192},
    ]

    print("=" * 75)
    print("PERFORMANCE BENCHMARK (Forward Pass)")
    print("=" * 75)
    print()

    for cfg in configs:
        batch, seq, dim = cfg["batch"], cfg["seq"], cfg["dim"]
        elements = batch * seq * dim
        eps = 1e-5

        print(f"📊 {cfg['name']}")
        print(f"   Shape: ({batch}, {seq}, {dim}) = {elements/1e6:.2f}M elements")
        print("-" * 70)

        # PyTorch MPS Reference
        X_torch = torch.randn(batch, seq, dim, device = "mps", dtype = torch.float16)
        W_torch = torch.randn(dim, device = "mps", dtype = torch.float16)

        # We use a mock LayerNorm object for fast_rms_layernorm
        class MockLN:
            def __init__(self, weight, eps):
                self.weight = weight
                self.eps = eps

        ln = MockLN(W_torch, eps)

        def pytorch_mps():
            return fast_rms_layernorm(ln, X_torch, gemma = False)

        t_torch = benchmark_fn(pytorch_mps)
        tp_torch = calculate_throughput(elements, t_torch)
        print(f"   PyTorch MPS Dispatch: {t_torch:7.3f} ms | {tp_torch:7.2f} GB/s")

        # MLX Composed (mx.fast.rms_norm)
        X_mlx = torch_to_mlx(X_torch)
        W_mlx = torch_to_mlx(W_torch)

        def mlx_native():
            # MLX's native RMSNorm - note it might not match our exact behavior if we add Gemma etc.
            return mx.fast.rms_norm(X_mlx, W_mlx, eps)

        t_mlx = benchmark_fn(mlx_native)
        tp_mlx = calculate_throughput(elements, t_mlx)
        print(f"   MLX Native (fast):  {t_mlx:7.3f} ms | {tp_mlx:7.2f} GB/s")

        # Fused Metal (unsloth) - Forward
        def unsloth_metal_forward():
            return metal_module.metal_rms_layernorm(X_torch, W_torch, eps, gemma = False)

        t_metal_fwd = benchmark_fn(unsloth_metal_forward)
        tp_metal_fwd = calculate_throughput(elements, t_metal_fwd)
        speedup_fwd = t_torch / t_metal_fwd
        print(
            f"   Unsloth Metal Forward (Fused): {t_metal_fwd:7.3f} ms | {tp_metal_fwd:7.2f} GB/s ({speedup_fwd:.2f}x vs PyTorch)"
        )

        # Fused Metal (unsloth) - Backward
        Y_torch, r_torch = unsloth_metal_forward()
        dY_torch = torch.randn_like(Y_torch)

        def unsloth_metal_backward():
            return metal_module.metal_rms_layernorm_backward(
                dY_torch, X_torch, W_torch, r_torch, eps, gemma = False
            )

        t_metal_bwd = benchmark_fn(unsloth_metal_backward)
        # Backward is more complex: reads dY, X, W, r. Writes dX, dW.
        # Approx TP using simple metric
        tp_metal_bwd = calculate_throughput(elements, t_metal_bwd)
        print(
            f"   Unsloth Metal Backward (Fused): {t_metal_bwd:7.3f} ms | {tp_metal_bwd:7.2f} GB/s"
        )
        print()


def run_correctness_tests():
    print("=" * 75)
    print("CORRECTNESS VERIFICATION")
    print("=" * 75)

    batch, seq, dim = 2, 512, 1024
    eps = 1e-5

    X = torch.randn(batch, seq, dim, dtype = torch.float32)
    W = torch.randn(dim, dtype = torch.float32)

    # Reference (PyTorch Float32 CPU)
    variance = X.pow(2).mean(-1, keepdim = True)
    rms_inv = torch.rsqrt(variance + eps)
    ref = (X * rms_inv) * W

    if platform.system() == "Darwin":
        X_mps = X.to("mps", torch.float16)
        W_mps = W.to("mps", torch.float16)

        import unsloth.kernels.metal.rms_layernorm as metal_module

        # 1. Forward Correctness
        out_metal, r_metal = metal_module.metal_rms_layernorm(
            X_mps, W_mps, eps, gemma = False
        )
        diff = np.abs(out_metal.cpu().float().numpy() - ref.numpy())
        print(
            f"  Forward Parity (fp16): max_diff={diff.max():.2e} mean={diff.mean():.2e} {'✅' if diff.max() < 1e-2 else '❌'}"
        )

        # 2. Backward Correctness
        dY_mps = torch.randn_like(out_metal)
        dX_metal, dW_metal = metal_module.metal_rms_layernorm_backward(
            dY_mps, X_mps, W_mps, r_metal, eps, gemma = False
        )

        # Ref Backward (Simplified)
        X_f32 = X.to(torch.float32)
        dY_f32 = dY_mps.cpu().float()
        W_f32 = W.to(torch.float32)

        X_norm = X_f32 * rms_inv
        dX_norm = dY_f32 * W_f32
        rowsum = (dX_norm * X_norm).sum(-1, keepdim = True)
        ref_dX = rms_inv * (dX_norm - (X_norm / dim) * rowsum)
        ref_dW = (dY_f32 * X_norm).sum(0)

        diff_dX = np.abs(dX_metal.cpu().float().numpy() - ref_dX.numpy())
        diff_dW = np.abs(dW_metal.cpu().float().numpy() - ref_dW.numpy())

        print(
            f"  Backward dX Parity:    max_diff={diff_dX.max():.2e} {'✅' if diff_dX.max() < 1e-2 else '❌'}"
        )
        print(
            f"  Backward dW Parity:    max_diff={diff_dW.max():.2e} {'✅' if diff_dW.max() < 5e-2 else '❌'}"
        )

        # Gemma mode
        ref_gemma = (X * rms_inv) * (W + 1.0)
        out_gemma, _ = metal_module.metal_rms_layernorm(X_mps, W_mps, eps, gemma = True)
        diff_gemma = np.abs(out_gemma.cpu().float().numpy() - ref_gemma.numpy())
        print(
            f"  Unsloth Gemma Forward:  max_diff={diff_gemma.max():.2e} {'✅' if diff_gemma.max() < 1e-2 else '❌'}"
        )
    else:
        print("Correctness tests for Metal kernels skipped (not on macOS).")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--perf", action = "store_true")
    parser.add_argument("--correctness", action = "store_true")
    args = parser.parse_args()

    print_header()

    if args.correctness or not args.perf:
        run_correctness_tests()
        print()

    if args.perf or not args.correctness:
        run_performance_benchmark()


if __name__ == "__main__":
    main()
