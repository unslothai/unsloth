import torch
import torch.nn as nn
import time
import numpy as np
import os
import gc

try:
    import mlx.core as mx
    from unsloth.kernels.mlx.bridge import (
        torch_to_mlx,
        mlx_to_torch,
        synchronize_mps,
        mlx_context,
    )
    from unsloth.kernels.mlx.fast_lora import (
        apply_lora_mlp_swiglu,
        apply_lora_qkv,
        apply_lora_o,
    )
    from unsloth.kernels.mlx.quantization import quantize_4bit

    HAS_MLX = True
except ImportError:
    HAS_MLX = False


# Colors for terminal output
class colors:
    HEADER = "\033[95m"
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"


def get_mem_stats():
    if torch.backends.mps.is_available():
        # MPS doesn't have a direct memory query like CUDA, but we can use psutil or just dummy
        return 0
    return 0


def log_header(text):
    print(f"\n{colors.HEADER}{colors.BOLD}{'='*80}")
    print(f" {text}".center(80))
    print(f"{'='*80}{colors.ENDC}")


def log_result(name, latency, error = None, extra = ""):
    err_str = f" | Err: {error:.2e}" if error is not None else ""
    print(f" {name:<35} | Latency: {latency:>8.3f} ms{err_str} {extra}")


class ComprehensiveBenchmark:
    def __init__(self, device = "mps", dtype = torch.float16):
        if not torch.backends.mps.is_available() and device == "mps":
            print(
                f"{colors.RED}Warning: MPS not available, switching to CPU.{colors.ENDC}"
            )
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)
        self.dtype = dtype
        self.results = []

    def run_mlp_benchmark(self, batch_size = 1, seq_len = 1, dim = 4096, hidden_dim = 11008):
        log_header(
            f"MLP Benchmark: B={batch_size}, S={seq_len}, D={dim}, H={hidden_dim}"
        )

        # 1. Setup Data
        X = torch.randn(batch_size, seq_len, dim, device = self.device, dtype = self.dtype)
        gateW = torch.randn(hidden_dim, dim, device = self.device, dtype = self.dtype) / (
            dim**0.5
        )
        upW = torch.randn(hidden_dim, dim, device = self.device, dtype = self.dtype) / (
            dim**0.5
        )
        downW = torch.randn(dim, hidden_dim, device = self.device, dtype = self.dtype) / (
            hidden_dim**0.5
        )

        # LoRA adapters
        A = torch.randn(8, dim, device = self.device, dtype = self.dtype) / (dim**0.5)
        B = torch.randn(hidden_dim, 8, device = self.device, dtype = self.dtype) / (8**0.5)
        S = 1.0

        # Reference (PyTorch Naive)
        def torch_mlp(x):
            g = torch.nn.functional.linear(x, gateW) + (x @ A.T) @ B.T * S
            u = torch.nn.functional.linear(x, upW) + (x @ A.T) @ B.T * S
            act = torch.nn.functional.silu(g) * u
            # Correct shape: (batch, seq, hidden_dim) @ (hidden_dim, rank) -> (8)
            # then (rank) @ (rank, dim) -> (dim)
            return torch.nn.functional.linear(act, downW) + (act @ B) @ A * S

        # Warmup & Reference Output
        with torch.no_grad():
            y_ref = torch_mlp(X)
            if self.device.type == "mps":
                torch.mps.synchronize()

        iters = 50
        start = time.time()
        for _ in range(iters):
            _ = torch_mlp(X)
        if self.device.type == "mps":
            torch.mps.synchronize()
        torch_latency = (time.time() - start) * 1000 / iters
        log_result("PyTorch Native (FP16)", torch_latency)

        if HAS_MLX:
            # 2. MLX 16-Bit
            # Cache weights to show true compute speed
            gateW._mlx_cache = torch_to_mlx(gateW)
            upW._mlx_cache = torch_to_mlx(upW)
            downW._mlx_cache = torch_to_mlx(downW)
            A._mlx_cache = torch_to_mlx(A)
            B._mlx_cache = torch_to_mlx(B)

            # Warmup
            with torch.no_grad():
                y_mlx = apply_lora_mlp_swiglu(
                    X,
                    gateW,
                    None,
                    A,
                    B,
                    S,
                    upW,
                    None,
                    A,
                    B,
                    S,
                    downW,
                    None,
                    B.T,
                    A.T,
                    S,
                )

            start = time.time()
            for _ in range(iters):
                _ = apply_lora_mlp_swiglu(
                    X,
                    gateW,
                    None,
                    A,
                    B,
                    S,
                    upW,
                    None,
                    A,
                    B,
                    S,
                    downW,
                    None,
                    B.T,
                    A.T,
                    S,
                )
            mlx_16_latency = (time.time() - start) * 1000 / iters

            error = (y_ref - y_mlx).abs().max().item()
            log_result("MLX 16-Bit (Cached)", mlx_16_latency, error)

            # 3. MLX 4-Bit
            # Cache quantized weights
            gateW._mlx_cache = quantize_4bit(gateW)
            upW._mlx_cache = quantize_4bit(upW)
            downW._mlx_cache = quantize_4bit(downW)

            # Warmup
            with torch.no_grad():
                y_q = apply_lora_mlp_swiglu(
                    X,
                    gateW,
                    None,
                    A,
                    B,
                    S,
                    upW,
                    None,
                    A,
                    B,
                    S,
                    downW,
                    None,
                    B.T,
                    A.T,
                    S,
                )

            start = time.time()
            for _ in range(iters):
                _ = apply_lora_mlp_swiglu(
                    X,
                    gateW,
                    None,
                    A,
                    B,
                    S,
                    upW,
                    None,
                    A,
                    B,
                    S,
                    downW,
                    None,
                    B.T,
                    A.T,
                    S,
                )
            mlx_4_latency = (time.time() - start) * 1000 / iters

            error_q = (y_ref - y_q).abs().max().item()
            log_result(
                "MLX 4-Bit (Cached)",
                mlx_4_latency,
                error_q,
                extra = f"{colors.GREEN}[Quantized]{colors.ENDC}",
            )

    def run_qkv_benchmark(self, batch_size = 1, seq_len = 1, dim = 4096, hidden_dim = 4096):
        log_header(f"QKV Benchmark: B={batch_size}, S={seq_len}, D={dim}")

        QW = torch.randn(dim, dim, device = self.device, dtype = self.dtype) / (dim**0.5)
        KW = torch.randn(dim, dim, device = self.device, dtype = self.dtype) / (dim**0.5)
        VW = torch.randn(dim, dim, device = self.device, dtype = self.dtype) / (dim**0.5)
        X = torch.randn(batch_size, seq_len, dim, device = self.device, dtype = self.dtype)

        # LoRA Dummy
        QA = torch.randn(8, dim, device = self.device, dtype = self.dtype) / (dim**0.5)
        QB = torch.randn(dim, 8, device = self.device, dtype = self.dtype) / (8**0.5)
        QS = 1.0

        # PyTorch
        def torch_qkv(x):
            q = torch.nn.functional.linear(x, QW) + (x @ QA.T) @ QB.T * QS
            k = torch.nn.functional.linear(x, KW) + (x @ QA.T) @ QB.T * QS
            v = torch.nn.functional.linear(x, VW) + (x @ QA.T) @ QB.T * QS
            return q, k, v

        iters = 50
        # Reference
        q_ref, k_ref, v_ref = torch_qkv(X)
        if self.device.type == "mps":
            torch.mps.synchronize()

        start = time.time()
        for _ in range(iters):
            _ = torch_qkv(X)
        if self.device.type == "mps":
            torch.mps.synchronize()
        torch_latency = (time.time() - start) * 1000 / iters
        log_result("PyTorch Native", torch_latency)

        if HAS_MLX:
            # MLX 16-bit
            # Cache weights to show true compute speed
            QW._mlx_cache = torch_to_mlx(QW)
            KW._mlx_cache = torch_to_mlx(KW)
            VW._mlx_cache = torch_to_mlx(VW)
            QA._mlx_cache = torch_to_mlx(QA)
            QB._mlx_cache = torch_to_mlx(QB)

            q_mlx, k_mlx, v_mlx = apply_lora_qkv(
                X, QW, None, QA, QB, QS, KW, None, QA, QB, QS, VW, None, QA, QB, QS
            )

            start = time.time()
            for _ in range(iters):
                _ = apply_lora_qkv(
                    X, QW, None, QA, QB, QS, KW, None, QA, QB, QS, VW, None, QA, QB, QS
                )
            mlx_latency = (time.time() - start) * 1000 / iters

            error = (q_ref - q_mlx).abs().max().item()
            log_result("MLX Optimized", mlx_latency, error)

    def diagnose(self):
        log_header("System Diagnosis")
        print(f" Torch Version: {torch.__version__}")
        print(f" MPS Available: {torch.backends.mps.is_available()}")
        if HAS_MLX:
            print(f" MLX Version:  {mx.__version__}")
            print(f" MLX Device:   {mx.default_device()}")
        else:
            print(f" {colors.RED}MLX NOT FOUND{colors.ENDC}")

        # Test Bridge
        if HAS_MLX:
            log_header("Bridge & Kernel Validation")
            X = torch.ones(2, 2, device = self.device, dtype = self.dtype)
            try:
                X_mlx = torch_to_mlx(X)
                X_back = mlx_to_torch(X_mlx)
                diff = (X - X_back).abs().max().item()
                status = (
                    f"{colors.GREEN}OK{colors.ENDC}"
                    if diff < 1e-4
                    else f"{colors.RED}FAIL (Diff: {diff}){colors.ENDC}"
                )
                print(f" Zero-copy Bridge: {status}")
            except Exception as e:
                print(f" Zero-copy Bridge: {colors.RED}ERROR: {e}{colors.ENDC}")

            # Test GEMV Kernel
            try:
                from unsloth.kernels.metal.gemv import fast_gemv

                W = mx.array(np.random.randn(128, 128).astype(np.float16))
                v = mx.array(np.random.randn(1, 128).astype(np.float16))
                res = fast_gemv(v, W)
                mx.eval(res)
                print(f" Custom Metal GEMV Kernel: {colors.GREEN}OK{colors.ENDC}")
            except Exception as e:
                print(
                    f" Custom Metal GEMV Kernel: {colors.RED}FAILED/UNAVAILABLE: {e}{colors.ENDC}"
                )


if __name__ == "__main__":
    benchmark = ComprehensiveBenchmark()
    benchmark.diagnose()

    # 1. Decoding Cases (Batch=1, Seq=1)
    # This triggers the specialized GEMV Metal Kernel
    benchmark.run_mlp_benchmark(batch_size = 1, seq_len = 1)
    benchmark.run_qkv_benchmark(batch_size = 1, seq_len = 1)

    # 2. Small Context Cases (Batch=1, Seq=128)
    # This triggers the Compiled MLX Graph
    benchmark.run_mlp_benchmark(batch_size = 1, seq_len = 128)

    # 3. Large Workload Cases (Multi-batch)
    benchmark.run_mlp_benchmark(batch_size = 8, seq_len = 512)

    log_header("Benchmark Summary")
    print(f"{colors.YELLOW}Notes:{colors.ENDC}")
    print("- B=1 cases utilize the specialized SIMD Metal GEMV kernel.")
    print("- B>1 cases utilize the Compiled MLX graph fusion.")
    print("- 4-bit cases use on-the-fly dequantization within the MLX graph.")
