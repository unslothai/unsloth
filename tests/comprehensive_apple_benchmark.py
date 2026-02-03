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
    res = {"mps": 0, "mlx_active": 0, "mlx_peak": 0}
    if torch.backends.mps.is_available():
        res["mps"] = torch.mps.current_allocated_memory() / 1e6
    if HAS_MLX:
        res["mlx_active"] = mx.get_active_memory() / 1e6
        res["mlx_peak"] = mx.get_peak_memory() / 1e6
    return res


def log_header(text):
    print(f"\n{colors.HEADER}{colors.BOLD}{'='*80}")
    print(f" {text}".center(80))
    print(f"{'='*80}{colors.ENDC}")


def log_result(name, latency, error=None, extra="", mem=None):
    err_str = f" | Err: {error:.2e}" if error is not None else ""
    # mem is expected to be a dict or a float
    if isinstance(mem, dict) and "mlx_active" in mem:
        mem_str = (
            f" | VRAM (Active/Peak): {mem['mlx_active']:.1f}/{mem['mlx_peak']:.1f} MB"
        )
    elif mem is not None:
        mem_str = f" | VRAM: {mem:.1f} MB"
    else:
        mem_str = ""
    print(f" {name:<35} | Latency: {latency:>8.3f} ms{err_str}{mem_str} {extra}")


class ComprehensiveBenchmark:
    def __init__(self, device="mps", dtype=torch.float16):
        if not torch.backends.mps.is_available() and device == "mps":
            print(
                f"{colors.RED}Warning: MPS not available, switching to CPU.{colors.ENDC}"
            )
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)
        self.dtype = dtype
        self.results = []

    def run_mlp_benchmark(self, batch_size=1, seq_len=1, dim=4096, hidden_dim=11008):
        log_header(
            f"MLP Benchmark: B={batch_size}, S={seq_len}, D={dim}, H={hidden_dim}"
        )

        # 1. Setup Data
        X = torch.randn(batch_size, seq_len, dim, device=self.device, dtype=self.dtype)
        gateW = torch.randn(hidden_dim, dim, device=self.device, dtype=self.dtype) / (
            dim**0.5
        )
        upW = torch.randn(hidden_dim, dim, device=self.device, dtype=self.dtype) / (
            dim**0.5
        )
        downW = torch.randn(dim, hidden_dim, device=self.device, dtype=self.dtype) / (
            hidden_dim**0.5
        )

        # LoRA adapters
        A = torch.randn(8, dim, device=self.device, dtype=self.dtype) / (dim**0.5)
        B = torch.randn(hidden_dim, 8, device=self.device, dtype=self.dtype) / (8**0.5)
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
        # Increase iters for small batches
        actual_iters = iters * 10 if batch_size == 1 else iters
        
        start = time.time()
        for _ in range(actual_iters):
             _ = torch_mlp(X)
        if self.device.type == "mps":
             torch.mps.synchronize()
        torch_latency = (time.time() - start) * 1000 / actual_iters
        log_result("PyTorch Native (FP16)", torch_latency)

        if HAS_MLX:
            # 2. MLX 16-Bit
            # Cache weights to show true compute speed
            gateW._mlx_cache = torch_to_mlx(gateW)
            upW._mlx_cache = torch_to_mlx(upW)
            downW._mlx_cache = torch_to_mlx(downW)
            A._mlx_cache = torch_to_mlx(A)
            B._mlx_cache = torch_to_mlx(B)

            # Pre-convert X for raw kernel speed
            X_mlx = torch_to_mlx(X)

            # Warmup
            res = apply_lora_mlp_swiglu(
                X_mlx,
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
            mx.eval(res)
            y_mlx = mlx_to_torch(res)

            # Increase iters for small batches
            actual_iters = iters * 10 if batch_size == 1 else iters
            
            start = time.time()
            for _ in range(actual_iters):
                res = apply_lora_mlp_swiglu(
                    X_mlx, gateW, None, A, B, S, upW, None, A, B, S, downW, None, B.T, A.T, S
                )
            mx.eval(res)
            mlx_16_latency = (time.time() - start) * 1000 / actual_iters

            error = (y_ref - y_mlx).abs().max().item()
            log_result(
                "MLX 16-Bit (Cached)", mlx_16_latency, error, mem=get_mem_stats()
            )

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
            # Increase iters for small batches
            actual_iters = iters * 10 if batch_size == 1 else iters
            
            start = time.time()
            for _ in range(actual_iters):
                res = apply_lora_mlp_swiglu(
                    X, gateW, None, A, B, S, upW, None, A, B, S, downW, None, B.T, A.T, S
                )
            mx.eval(res)
            mlx_4_latency = (time.time() - start) * 1000 / actual_iters

            error_q = (y_ref - y_q).abs().max().item()
            log_result(
                "MLX 4-Bit (Cached)",
                mlx_4_latency,
                error_q,
                extra=f"{colors.GREEN}[Quantized]{colors.ENDC}",
                mem=get_mem_stats(),
            )

    def run_qkv_benchmark(self, batch_size=1, seq_len=1, dim=4096, hidden_dim=4096):
        log_header(f"QKV Benchmark: B={batch_size}, S={seq_len}, D={dim}")

        QW = torch.randn(dim, dim, device=self.device, dtype=self.dtype) / (dim**0.5)
        KW = torch.randn(dim, dim, device=self.device, dtype=self.dtype) / (dim**0.5)
        VW = torch.randn(dim, dim, device=self.device, dtype=self.dtype) / (dim**0.5)
        X = torch.randn(batch_size, seq_len, dim, device=self.device, dtype=self.dtype)

        # LoRA Dummy
        QA = torch.randn(8, dim, device=self.device, dtype=self.dtype) / (dim**0.5)
        QB = torch.randn(dim, 8, device=self.device, dtype=self.dtype) / (8**0.5)
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

            # Pre-convert X for raw kernel speed
            X_mlx = torch_to_mlx(X)

            q_mlx, k_mlx, v_mlx = apply_lora_qkv(
                X_mlx, QW, None, QA, QB, QS, KW, None, QA, QB, QS, VW, None, QA, QB, QS
            )
            mx.eval(q_mlx, k_mlx, v_mlx)
            q_mlx = mlx_to_torch(q_mlx)
            k_mlx = mlx_to_torch(k_mlx)
            v_mlx = mlx_to_torch(v_mlx)

            start = time.time()
            # Increase iters for small batches
            actual_iters = iters * 10 if batch_size == 1 else iters
            
            start = time.time()
            for _ in range(actual_iters):
                res = apply_lora_qkv(
                    X_mlx, QW, None, QA, QB, QS, KW, None, QA, QB, QS, VW, None, QA, QB, QS
                )
            # Force final eval
            mx.eval(*res)
            mlx_latency = (time.time() - start) * 1000 / actual_iters

            error = (q_ref - q_mlx).abs().max().item()
            log_result("MLX Optimized", mlx_latency, error, mem=get_mem_stats())

    def run_llama3_benchmark(self, batch_size=1, seq_len=1):
        # Llama-3-8B Scale: D=4096, H=14336
        log_header(f"Llama-3-8B Scale Benchmark (B={batch_size}, S={seq_len})")
        self.run_mlp_benchmark(
            batch_size=batch_size, seq_len=seq_len, dim=4096, hidden_dim=14336
        )

    def run_swiglu_benchmark(self, batch_size=1, seq_len=1, dim=4096):
        log_header(f"SwiGLU Benchmark: B={batch_size}, S={seq_len}, D={dim}")
        
        # Imports
        from unsloth.kernels.metal.swiglu import mlx_swiglu_forward
        
        elements = batch_size * seq_len * dim
        e = torch.randn(batch_size, seq_len, dim, device=self.device, dtype=self.dtype)
        g = torch.randn(batch_size, seq_len, dim, device=self.device, dtype=self.dtype)
        
        # PyTorch Native
        def torch_swiglu():
            return torch.nn.functional.silu(e) * g
            
        iters = 50
        # Increase iters for small batches
        actual_iters = iters * 10 if batch_size == 1 else iters
        
        start = time.time()
        for _ in range(actual_iters):
             _ = torch_swiglu()
        if self.device.type == "mps":
             torch.mps.synchronize()
        torch_lat = (time.time() - start) * 1000 / actual_iters
        log_result("PyTorch Native", torch_lat)

        if HAS_MLX:
            e_mlx = torch_to_mlx(e)
            g_mlx = torch_to_mlx(g)
            
            # Increase iters for small batches
            actual_iters = iters * 10 if batch_size == 1 else iters
            
            # MLX Composed
            def mlx_composed():
                return mx.sigmoid(e_mlx) * e_mlx * g_mlx
            
            start = time.time()
            for _ in range(actual_iters):
                res = mlx_composed()
            mx.eval(res)
            mlx_lat = (time.time() - start) * 1000 / actual_iters
            log_result("MLX Composed", mlx_lat)
            
            # Unsloth Fused
            def mlx_fused():
                return mlx_swiglu_forward(e_mlx, g_mlx)
                
            start = time.time()
            for _ in range(actual_iters):
                res = mlx_fused()
            mx.eval(res)
            fused_lat = (time.time() - start) * 1000 / actual_iters
            speedup = mlx_lat / fused_lat
            log_result("Unsloth Fused Metal", fused_lat, extra=f"({speedup:.2f}x Speedup)", mem=get_mem_stats())

        
    def run_rope_benchmark(self, batch_size=1, seq_len=1, dim=4096, n_heads=32):
        log_header(f"RoPE Benchmark: B={batch_size}, S={seq_len}, D={dim}")
        
        head_dim = dim // n_heads
        Q = torch.randn(batch_size, n_heads, seq_len, head_dim, device=self.device, dtype=self.dtype)
        K = torch.randn(batch_size, n_heads, seq_len, head_dim, device=self.device, dtype=self.dtype)
        
        # RoPE cos/sin
        cos = torch.randn(seq_len, head_dim, device=self.device, dtype=self.dtype)
        sin = torch.randn(seq_len, head_dim, device=self.device, dtype=self.dtype)
        
        # PyTorch Fallback (Simulated Reference)
        # Using the slow implementation manually for baseline
        from unsloth.kernels.rope_embedding import Slow_RoPE_Embedding
        
        def torch_rope(q, k, c, s):
            # Slow_RoPE_Embedding expects (cos, sin) to be broadcastable to [batch, seq, heads, dim]
            # Input Q/K are (batch, heads, seq, head_dim)
            # Transposed they are (batch, seq, heads, head_dim)
            # cos/sin are (seq, head_dim). Need to unsqueeze to (1, seq, 1, head_dim)
            c_fixed = c.unsqueeze(0).unsqueeze(2)
            s_fixed = s.unsqueeze(0).unsqueeze(2)
            
            q_trans = q.transpose(1, 2)
            # Use clone to avoid in-place issues if needed, though Slow_RoPE handles it
            q_out = Slow_RoPE_Embedding.apply(q_trans, c_fixed, s_fixed, None).transpose(1, 2)
            
            k_trans = k.transpose(1, 2)
            k_out = Slow_RoPE_Embedding.apply(k_trans, c_fixed, s_fixed, None).transpose(1, 2)
            return q_out, k_out

        # Warmup
        q_ref, k_ref = torch_rope(Q.clone(), K.clone(), cos, sin)
        if self.device.type == "mps":
             torch.mps.synchronize()

        iters = 50
        # Increase iters for small batches
        actual_iters = iters * 10 if batch_size == 1 else iters

        start = time.time()
        for _ in range(actual_iters):
            _ = torch_rope(Q.clone(), K.clone(), cos, sin)
        if self.device.type == "mps":
            torch.mps.synchronize()
        torch_latency = (time.time() - start) * 1000 / actual_iters
        log_result("PyTorch Native", torch_latency)

        if HAS_MLX:
            # MLX Optimized
            from unsloth.kernels.mlx.fast_ops import mlx_rope_qk
            
            # Pre-convert inputs
            Q_mlx = torch_to_mlx(Q)
            K_mlx = torch_to_mlx(K)
            cos_mlx = torch_to_mlx(cos)
            sin_mlx = torch_to_mlx(sin)
            
            # Warmup
            q_out, k_out = mlx_rope_qk(Q_mlx, K_mlx, cos_mlx, sin_mlx)
            mx.eval(q_out, k_out)
            
            # Increase iters for small batches
            actual_iters = iters * 10 if batch_size == 1 else iters
            
            start = time.time()
            for _ in range(actual_iters):
                 res = mlx_rope_qk(Q_mlx, K_mlx, cos_mlx, sin_mlx)
            # Final sync
            mx.eval(res[0], res[1])
            mlx_latency = (time.time() - start) * 1000 / actual_iters
            
            # Error check
            # res is (q_out_mlx, k_out_mlx)
            q_res_torch = mlx_to_torch(res[0])
            err_q = (q_ref - q_res_torch).abs().max().item()
            
            log_result("MLX Optimized", mlx_latency, err_q, mem=get_mem_stats())

    def run_geglu_benchmark(self, batch_size=1, seq_len=1, dim=4096):
        log_header(f"GEGLU Benchmark: B={batch_size}, S={seq_len}, D={dim}")
         # Imports
        try:
            from unsloth.kernels.metal.geglu import mlx_geglu_exact_forward
        except ImportError:
            print("GEGLU kernel not found.")
            return

        e = torch.randn(batch_size, seq_len, dim, device=self.device, dtype=self.dtype)
        g = torch.randn(batch_size, seq_len, dim, device=self.device, dtype=self.dtype)

        # PyTorch Native
        def torch_geglu():
            return torch.nn.functional.gelu(e) * g

        iters = 50
        # Increase iters for small batches
        actual_iters = iters * 10 if batch_size == 1 else iters
        
        start = time.time()
        for _ in range(actual_iters):
             _ = torch_geglu()
        if self.device.type == "mps":
             torch.mps.synchronize()
        torch_lat = (time.time() - start) * 1000 / actual_iters
        log_result("PyTorch Native", torch_lat)

        if HAS_MLX:
            e_mlx = torch_to_mlx(e)
            g_mlx = torch_to_mlx(g)
            
            # unsloth fused
            def mlx_fused():
                return mlx_geglu_exact_forward(e_mlx, g_mlx)

            # Increase iters for small batches
            actual_iters = iters * 10 if batch_size == 1 else iters
            
            start = time.time()
            for _ in range(actual_iters):
                res = mlx_fused()
            mx.eval(res)
            fused_lat = (time.time() - start) * 1000 / actual_iters
            log_result("Unsloth Fused Metal", fused_lat, mem=get_mem_stats())

    def run_rms_benchmark(self, batch_size=1, seq_len=1, dim=4096):
        log_header(f"RMS Norm Benchmark: B={batch_size}, S={seq_len}, D={dim}")
        try:
             from unsloth.kernels.metal import metal_rms_layernorm
        except ImportError:
            print("RMS kernel not found.")
            return

        x = torch.randn(batch_size, seq_len, dim, device=self.device, dtype=self.dtype)
        w = torch.randn(dim, device=self.device, dtype=self.dtype)
        eps = 1e-6

        # PyTorch Native
        def torch_rms():
            return torch.no_grad()(lambda: x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w)()

        iters = 50
        # Increase iters for small batches
        actual_iters = iters * 10 if batch_size == 1 else iters
        
        start = time.time()
        for _ in range(actual_iters):
             _ = torch_rms()
        if self.device.type == "mps":
             torch.mps.synchronize()
        torch_lat = (time.time() - start) * 1000 / actual_iters
        log_result("PyTorch Native (Simulated)", torch_lat)

        if HAS_MLX:
            x_mlx = torch_to_mlx(x)
            w_mlx = torch_to_mlx(w)
            
            # Increase iters for small batches
            actual_iters = iters * 10 if batch_size == 1 else iters
            
            def mlx_fused():
                return metal_rms_layernorm(x_mlx, w_mlx, eps)
            
            start = time.time()
            for _ in range(actual_iters):
                res = mlx_fused()
            mx.eval(res)
            fused_lat = (time.time() - start) * 1000 / actual_iters
            log_result("Unsloth Fused Metal", fused_lat, mem=get_mem_stats())


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
            X = torch.ones(2, 2, device=self.device, dtype=self.dtype)
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
    # Llama-3-8B Scale
    benchmark.run_llama3_benchmark(batch_size=1, seq_len=1)
    benchmark.run_qkv_benchmark(batch_size=1, seq_len=1)
    benchmark.run_swiglu_benchmark(batch_size=1, seq_len=1, dim=14336)
    benchmark.run_rms_benchmark(batch_size=1, seq_len=1, dim=4096)

    # 2. Large Workload Cases (Multi-batch)
    benchmark.run_llama3_benchmark(batch_size=8, seq_len=512)
    benchmark.run_qkv_benchmark(batch_size=8, seq_len=512)
    benchmark.run_swiglu_benchmark(batch_size=8, seq_len=512, dim=14336)
    benchmark.run_geglu_benchmark(batch_size=8, seq_len=512)
    benchmark.run_rms_benchmark(batch_size=8, seq_len=512, dim=4096)
    benchmark.run_rope_benchmark(batch_size=8, seq_len=512)

    print("\n" + "="*80)
    print(f"{'Benchmark Summary':^80}")
    print("="*80)
    print(f"{colors.YELLOW}Notes:{colors.ENDC}")
    print("- B=1 cases utilize the specialized SIMD Metal GEMV kernel.")
    print("- B>1 cases utilize the Compiled MLX graph fusion.")
    print("- 4-bit cases use direct quantized_matmul within the MLX graph.")
    print("- VRAM estimates reflect active GPU memory in MB.")

