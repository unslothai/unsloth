
import torch
import time
import argparse

try:
    import mlx.core as mx
    print("MLX found. Benchmarking MLX vs PyTorch.")
except ImportError:
    print("MLX not found. This benchmark requires MLX.")
    exit(0)

from unsloth.kernels.mlx.fast_lora import apply_lora_mlp_swiglu, apply_lora_qkv

def benchmark_mlp(batch_size=1, seq_len=128, dim=4096, hidden_dim=11008, iters=100):
    print(f"\n--- Benchmarking MLP [B={batch_size}, S={seq_len}, D={dim}] ---")
    
    # Setup PyTorch Tensors (CPU for now, would be MPS on Mac)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    dtype = torch.float16
    
    X = torch.randn(batch_size, seq_len, dim, device=device, dtype=dtype)
    
    # Weights
    gateW = torch.randn(hidden_dim, dim, device=device, dtype=dtype)
    gateA = torch.randn(8, dim, device=device, dtype=dtype)
    gateB = torch.randn(hidden_dim, 8, device=device, dtype=dtype)
    gateS = 1.0
    
    upW = torch.randn(hidden_dim, dim, device=device, dtype=dtype)
    upA = torch.randn(8, dim, device=device, dtype=dtype)
    upB = torch.randn(hidden_dim, 8, device=device, dtype=dtype)
    upS = 1.0
    
    downW = torch.randn(dim, hidden_dim, device=device, dtype=dtype)
    downA = torch.randn(8, hidden_dim, device=device, dtype=dtype)
    downB = torch.randn(dim, 8, device=device, dtype=dtype)
    downS = 1.0

    # --- PyTorch Baseline Implementation (Naive) ---
    def torch_mlp(x):
        # Gate
        gate = torch.nn.functional.linear(x, gateW) + (x @ gateA.T) @ gateB.T * gateS
        # Up
        up = torch.nn.functional.linear(x, upW) + (x @ upA.T) @ upB.T * upS
        # SwiGLU
        act = torch.nn.functional.silu(gate) * up # Standard SwiGLU is silu(gate) * up vs gate * sigmoid(gate) * up? 
        # Unsloth uses: e = e * sigmoid(e) aka swish/silu, then h = f * g. 
        # Wait, Unsloth code said: f = e * sigmoid(e). g = X @ U. h = f * g. 
        # So it IS SwiGLU.
        
        # Down
        out = torch.nn.functional.linear(act, downW) + (act @ downA.T) @ downB.T * downS
        return out

    # Warmup PyTorch
    for _ in range(10):
        _ = torch_mlp(X)
    if device.type == "mps": torch.mps.synchronize()

    start = time.time()
    for _ in range(iters):
        _ = torch_mlp(X)
    if device.type == "mps": torch.mps.synchronize()
    torch_time_ms = (time.time() - start) * 1000 / iters
    print(f"PyTorch (Naive): {torch_time_ms:.3f} ms / iter")

    # --- MLX Implementation ---
    # Warmup
    out_mlx = apply_lora_mlp_swiglu(
        X, 
        gateW, None, gateA, gateB, gateS,
        upW, None, upA, upB, upS,
        downW, None, downA, downB, downS
    )
    for _ in range(10):
        apply_lora_mlp_swiglu(
            X, 
            gateW, None, gateA, gateB, gateS,
            upW, None, upA, upB, upS,
            downW, None, downA, downB, downS
        )

    start = time.time()
    for _ in range(iters):
        apply_lora_mlp_swiglu(
            X, 
            gateW, None, gateA, gateB, gateS,
            upW, None, upA, upB, upS,
            downW, None, downA, downB, downS
        )
    # The function already synchronizes/evals internally if written correctly or we rely on bridge sync.
    # Our `apply_lora_mlp_swiglu` converts back to Torch, so it forces sync implicitly via dlpack usually?
    # Actually `mlx_to_torch` usually doesn't block unless we read data.
    # But `mx.eval` was called inside `apply_lora_mlp_swiglu`.
    
    mlx_time_ms = (time.time() - start) * 1000 / iters
    print(f"Propsoed MLX:    {mlx_time_ms:.3f} ms / iter")
    
    speedup = torch_time_ms / mlx_time_ms
    print(f"Speedup: {speedup:.2f}x")

if __name__ == "__main__":
    benchmark_mlp()
