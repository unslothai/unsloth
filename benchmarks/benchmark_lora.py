"""
LoRA Benchmark Suite (Fused MLX vs PyTorch MPS)
Tests QKV and Output projections with LoRA adapters.
"""

# Apply Mac compatibility patches BEFORE importing unsloth
import platform
import sys
import os
if platform.system() == "Darwin":
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
    from patcher import patch_for_mac
    patch_for_mac()

import sys
import time
import torch
import torch.nn.functional as F
import mlx.core as mx
import numpy as np
import os

# Add parent directory for imports
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from unsloth.kernels.mps.fast_lora import mps_apply_lora_qkv, mps_apply_lora_o
from unsloth.kernels.mlx.bridge import torch_to_mlx, mlx_context, mlx_to_torch

def get_peak_memory_mb():
    if torch.backends.mps.is_available():
        return torch.mps.current_allocated_memory() / 1024 / 1024
    return 0

def benchmark_fn(fn, warmup=10, iterations=50, check_memory=True):
    # Warmup
    for _ in range(warmup):
        fn()
    torch.mps.synchronize()
    
    start_mem = get_peak_memory_mb() if check_memory else 0
    start = time.perf_counter()
    
    for _ in range(iterations):
        fn()
        
    torch.mps.synchronize()
    end = time.perf_counter()
    end_mem = get_peak_memory_mb() if check_memory else 0
    
    avg_lat = (end - start) / iterations * 1000
    peak_mem_delta = max(0, end_mem - start_mem) # Rough estimate
    
    return avg_lat, peak_mem_delta

def run_benchmark():
    if not torch.backends.mps.is_available():
        print("❌ MPS backend not available")
        return

    print("=" * 70)
    print("Unsloth Fast LoRA Benchmark (MPS/MLX)")
    print("=" * 70)
    
    resolutions = [
        (1, 2048, 4096, 64, "Llama-3 8B Inference (BS=1)"),
        (4, 512, 4096, 64, "Llama-3 8B Training (BS=4)"),
        (1, 8192, 4096, 64, "Long Context 8K (BS=1)"),
    ]
    
    dtype = torch.float16 # or bfloat16
    
    for b, s, h, r, name in resolutions:
        print(f"\n📊 {name} | {b}x{s} | H={h} | R={r}")
        
        # -----------------------------
        # QKV Projection Benchmark
        # -----------------------------
        print(f"   [QKV Projection]")
        
        # Setup Tensors
        X = torch.randn(b, s, h, device="mps", dtype=dtype)
        
        # Weights (Out, In)
        QW = torch.randn(h, h, device="mps", dtype=dtype)
        KW = torch.randn(h, h, device="mps", dtype=dtype)
        VW = torch.randn(h, h, device="mps", dtype=dtype)
        
        # LoRA Adapters
        # A: (R, In), B: (Out, R)
        QA = torch.randn(r, h, device="mps", dtype=dtype)
        QB = torch.randn(h, r, device="mps", dtype=dtype)
        QS = 1.0
        
        KA = torch.randn(r, h, device="mps", dtype=dtype)
        KB = torch.randn(h, r, device="mps", dtype=dtype)
        KS = 1.0
        
        VA = torch.randn(r, h, device="mps", dtype=dtype)
        VB = torch.randn(h, r, device="mps", dtype=dtype)
        VS = 1.0

        # 1. PyTorch Standard (Simulated unoptimized)
        def torch_qkv():
            # Standard: Linear + LoRA
            # LoRA = (x @ A.T) @ B.T * s
            # Q = F.linear(X, QW) + (X @ QA.t() @ QB.t()) * QS
            # Doing it naively
            q_base = F.linear(X, QW)
            q_lora = (X @ QA.t()) @ QB.t() * QS
            q = q_base + q_lora
            
            k_base = F.linear(X, KW)
            k_lora = (X @ KA.t()) @ KB.t() * KS
            k = k_base + k_lora
            
            v_base = F.linear(X, VW)
            v_lora = (X @ VA.t()) @ VB.t() * VS
            v = v_base + v_lora
            return q, k, v

        t_torch, m_torch = benchmark_fn(torch_qkv)
        print(f"   PyTorch Native:   {t_torch:7.3f} ms")

        # 2. Unsloth MPS (PyTorch Fallback - what we fixed)
        # We need to ensure NO MLX cache exists to force the fallback path
        # But we also want to test the dispatcher which checks for cache.
        # So we pass clean tensors.
        def unsloth_mps_qkv():
             return mps_apply_lora_qkv(
                X, QW, None, QA, QB, QS,
                KW, None, KA, KB, KS,
                VW, None, VA, VB, VS
             )

        t_mps, m_mps = benchmark_fn(unsloth_mps_qkv)
        speedup_mps = t_torch / t_mps
        print(f"   Unsloth MPS Only: {t_mps:7.3f} ms | {speedup_mps:.2f}x Speedup")

        # 3. Unsloth MLX (Fused)
        # We trigger this by attaching _mlx_cache
        X_mlx = X.clone()
        with mlx_context():
            X_mlx._mlx_cache = torch_to_mlx(X)
            
        def unsloth_mlx_qkv():
            return mps_apply_lora_qkv(
                X_mlx, QW, None, QA, QB, QS,
                KW, None, KA, KB, KS,
                VW, None, VA, VB, VS
             )

        t_mlx, m_mlx = benchmark_fn(unsloth_mlx_qkv)
        speedup_mlx = t_torch / t_mlx
        print(f"   Unsloth MLX:      {t_mlx:7.3f} ms | {speedup_mlx:.2f}x Speedup")
        
        # Throughput
        flops_per_token = 3 * (2*h*h + 2*r*h) # 3 projs (Base + LoRA)
        total_flops = b * s * flops_per_token
        tflops = (total_flops / (t_mlx / 1000)) / 1e12
        print(f"   MLX TFLOPS:       {tflops:.2f} TFLOPS")


        # -----------------------------
        # Output Projection Benchmark
        # -----------------------------
        print(f"   [Output Projection]")
        
        OW = torch.randn(h, h, device="mps", dtype=dtype)
        OA = torch.randn(r, h, device="mps", dtype=dtype)
        OB = torch.randn(h, r, device="mps", dtype=dtype)
        OS = 1.0
        
        def torch_o():
            base = F.linear(X, OW)
            lora = (X @ OA.t()) @ OB.t() * OS
            return base + lora
            
        t_torch_o, _ = benchmark_fn(torch_o)
        
        def unsloth_mlx_o():
            return mps_apply_lora_o(X_mlx, OW, None, OA, OB, OS)
            
        t_mlx_o, _ = benchmark_fn(unsloth_mlx_o)
        speedup_o = t_torch_o / t_mlx_o
        print(f"   PyTorch Native:   {t_torch_o:7.3f} ms")
        print(f"   Unsloth MLX:      {t_mlx_o:7.3f} ms | {speedup_o:.2f}x Speedup")


if __name__ == "__main__":
    run_benchmark()
