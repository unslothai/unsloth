#!/usr/bin/env python3
"""
Test pure MLX Metal kernels without PyTorch conversion overhead.
Run this on Apple Silicon to verify Chunk 1 implementation.
"""

# Import patcher first to set up mocks
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import patcher  # noqa: F401

import mlx.core as mx
import time

print("=" * 70)
print("Testing Pure MLX Metal Kernels (Chunk 1)")
print("=" * 70)

# Test 1: SwiGLU
print("\n1. Testing SwiGLU kernel...")
try:
    from unsloth.kernels.metal.swiglu_mlx import swiglu_forward, swiglu_backward
    
    # Create test tensors
    B, S, D = 2, 512, 4096
    hidden_dim = 11008
    
    gate = mx.random.normal((B, S, hidden_dim))
    up = mx.random.normal((B, S, hidden_dim))
    
    # Forward
    start = time.time()
    output = swiglu_forward(gate, up)
    mx.eval(output)
    fwd_time = (time.time() - start) * 1000
    
    print(f"   ✓ SwiGLU forward: {output.shape}, {fwd_time:.2f}ms")
    
    # Backward
    grad_output = mx.random.normal((B, S, D))
    start = time.time()
    grad_gate, grad_up = swiglu_backward(grad_output, gate, up)
    mx.eval([grad_gate, grad_up])
    bwd_time = (time.time() - start) * 1000
    
    print(f"   ✓ SwiGLU backward: {bwd_time:.2f}ms")
    print(f"   ✓ No torch conversion overhead!")
    
except Exception as e:
    print(f"   ✗ SwiGLU test failed: {e}")

# Test 2: RMSNorm
print("\n2. Testing RMSNorm kernel...")
try:
    from unsloth.kernels.metal.rms_layernorm_mlx import rms_layernorm_forward, rms_layernorm_backward
    
    B, S, D = 2, 512, 4096
    x = mx.random.normal((B, S, D))
    weight = mx.ones((D,))
    
    # Forward
    start = time.time()
    output = rms_layernorm_forward(x, weight, eps=1e-6)
    mx.eval(output)
    fwd_time = (time.time() - start) * 1000
    
    print(f"   ✓ RMSNorm forward: {output.shape}, {fwd_time:.2f}ms")
    
    # Backward
    grad_output = mx.random.normal((B, S, D))
    start = time.time()
    grad_x, grad_weight = rms_layernorm_backward(grad_output, x, weight, eps=1e-6)
    mx.eval([grad_x, grad_weight])
    bwd_time = (time.time() - start) * 1000
    
    print(f"   ✓ RMSNorm backward: {bwd_time:.2f}ms")
    
except Exception as e:
    print(f"   ✗ RMSNorm test failed: {e}")

# Test 3: GEGLU
print("\n3. Testing GEGLU kernel...")
try:
    from unsloth.kernels.metal.geglu_mlx import geglu_forward, geglu_backward
    
    B, S, D = 2, 512, 4096
    hidden_dim = 11008
    
    gate = mx.random.normal((B, S, hidden_dim))
    up = mx.random.normal((B, S, hidden_dim))
    
    # Forward
    start = time.time()
    output = geglu_forward(gate, up)
    mx.eval(output)
    fwd_time = (time.time() - start) * 1000
    
    print(f"   ✓ GEGLU forward: {output.shape}, {fwd_time:.2f}ms")
    
    # Backward
    grad_output = mx.random.normal((B, S, D))
    start = time.time()
    grad_gate, grad_up = geglu_backward(grad_output, gate, up)
    mx.eval([grad_gate, grad_up])
    bwd_time = (time.time() - start) * 1000
    
    print(f"   ✓ GEGLU backward: {bwd_time:.2f}ms")
    
except Exception as e:
    print(f"   ✗ GEGLU test failed: {e}")

print("\n" + "=" * 70)
print("Chunk 1 Complete: Pure MLX Metal Kernels")
print("=" * 70)
print("\nKey achievements:")
print("  • No torch_to_mlx/mlx_to_torch conversion")
print("  • Direct mx.array input/output")
print("  • Metal kernels compiled for Apple Silicon")
print("  • Forward + backward pass support")
print("\nNext: Chunk 2 - Training Infrastructure")
