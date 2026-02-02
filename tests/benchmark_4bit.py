import torch
import time
import mlx.core as mx
from unsloth.kernels.mlx.fast_lora import apply_lora_mlp_swiglu
from unsloth.kernels.mlx.utils import fast_quantize

def benchmark_4bit_vs_16bit(dim=4096, hidden_dim=11008, iters=50):
    print(f"--- Benchmarking 4-Bit vs 16-Bit MLX [D={dim}, H={hidden_dim}] ---")
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    dtype = torch.float16
    
    # 1. Setup Layer
    layer = torch.nn.Linear(dim, hidden_dim).to(device).to(dtype)
    X = torch.randn(1, 1, dim, device=device, dtype=dtype)
    
    # LoRA params (dummies)
    A = torch.randn(8, dim, device=device, dtype=dtype)
    B = torch.randn(hidden_dim, 8, device=device, dtype=dtype)
    S = 1.0
    
    # 2. Benchmark 16-Bit
    print("\nRunning 16-Bit MLX Benchmark...")
    # Warmup
    for _ in range(5):
        _ = apply_lora_mlp_swiglu(
            X, 
            layer.weight, None, A, B, S, # gate
            layer.weight, None, A, B, S, # up
            layer.weight, None, A, B, S  # down
        )
    
    start = time.time()
    for _ in range(iters):
        _ = apply_lora_mlp_swiglu(
            X, 
            layer.weight, None, A, B, S, # gate
            layer.weight, None, A, B, S, # up
            layer.weight, None, A, B, S  # down
        )
    time_16 = (time.time() - start) * 1000 / iters
    print(f"16-Bit Latency: {time_16:.3f} ms")
    
    # 3. Quantize to 4-Bit
    print("\nQuantizing to 4-Bit...")
    # Simulate Unsloth 4-bit cache
    from unsloth.kernels.mlx.quantization import quantize_4bit
    layer.weight._mlx_cache = quantize_4bit(layer.weight)
    
    # 4. Benchmark 4-Bit
    print("Running 4-Bit MLX Benchmark...")
    # Warmup
    for _ in range(5):
        _ = apply_lora_mlp_swiglu(
            X, 
            layer.weight, None, A, B, S, # gate
            layer.weight, None, A, B, S, # up
            layer.weight, None, A, B, S  # down
        )
        
    start = time.time()
    for _ in range(iters):
        _ = apply_lora_mlp_swiglu(
            X, 
            layer.weight, None, A, B, S, # gate
            layer.weight, None, A, B, S, # up
            layer.weight, None, A, B, S  # down
        )
    time_4 = (time.time() - start) * 1000 / iters
    print(f"4-Bit Latency:  {time_4:.3f} ms")
    
    print(f"\nSpeedup: {time_16 / time_4:.2f}x")

if __name__ == "__main__":
    if not torch.backends.mps.is_available():
        print("⚠️ MPS not available. Results on CPU will not reflect Apple Silicon performance.")
    
    benchmark_4bit_vs_16bit()
