
import mlx.core as mx
import numpy as np
import time
from unsloth.kernels.metal.gemv import fast_gemv

def test_gemv_correctness():
    print("Testing GEMV Correctness...")
    K = 4096
    N = 11008
    
    X = mx.random.normal((1, K)).astype(mx.float16)
    W = mx.random.normal((N, K)).astype(mx.float16)
    
    # Expected
    Y_ref = X @ W.T
    
    # Custom
    Y_out = fast_gemv(X, W)
    mx.eval(Y_out)
    
    # Compare
    diff = mx.abs(Y_ref - Y_out).max()
    print(f"Max Diff: {diff.item()}")
    
    if diff.item() > 1e-2: # Relaxed tolerance for float16 accumulation diffs
        print("❌ Correctness Failed!")
    else:
        print("✅ Correctness Passed!")

def benchmark_gemv():
    print("\nBenchmarking GEMV (B=1)...")
    K = 4096
    N = 11008
    
    X = mx.random.normal((1, K)).astype(mx.float16)
    W = mx.random.normal((N, K)).astype(mx.float16)
    mx.eval(X, W)
    
    # Baseline (MLX Matmul)
    start = time.time()
    for _ in range(100):
        y = X @ W.T
        mx.eval(y)
    base_time = (time.time() - start) * 1000 / 100
    print(f"MLX Native Matmul: {base_time:.3f} ms")
    
    # Custom GEMV
    # Warmup
    _ = fast_gemv(X, W)
    
    start = time.time()
    for _ in range(100):
        y = fast_gemv(X, W)
        mx.eval(y)
    custom_time = (time.time() - start) * 1000 / 100
    print(f"Custom GEMV:       {custom_time:.3f} ms")
    
    print(f"Speedup: {base_time / custom_time:.2f}x")

if __name__ == "__main__":
    test_gemv_correctness()
    benchmark_gemv()
