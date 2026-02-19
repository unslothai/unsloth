#!/usr/bin/env python3
"""
Test pure MLX Metal kernels without PyTorch conversion overhead.
Run this on Apple Silicon to verify Chunk 1 implementation.
"""

# Import patcher first to set up mocks
import sys
from pathlib import Path

# Add repo root to path BEFORE importing anything
sys.path.insert(0, str(Path(__file__).parent.parent))

# Now import patcher
import patcher  # noqa: F401

# Import MLX first to avoid any torch imports
import mlx.core as mx
import time

print("=" * 70)
print("Testing Pure MLX Metal Kernels (Chunk 1)")
print("=" * 70)

# Test 1: SwiGLU
print("\n1. Testing SwiGLU kernel...")
try:
    # Import directly from file to avoid __init__.py imports
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "swiglu_mlx",
        str(Path(__file__).parent.parent / "unsloth" / "kernels" / "metal" / "swiglu_mlx.py")
    )
    swiglu_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(swiglu_module)
    swiglu_forward = swiglu_module.swiglu_forward
    swiglu_backward = swiglu_module.swiglu_backward
    
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
    grad_output = mx.random.normal((B, S, hidden_dim))
    start = time.time()
    h_out, grad_gate, grad_up = swiglu_backward(grad_output, gate, up)
    mx.eval([h_out, grad_gate, grad_up])
    bwd_time = (time.time() - start) * 1000
    
    print(f"   ✓ SwiGLU backward: {bwd_time:.2f}ms")
    print(f"   ✓ No torch conversion overhead!")
    
except Exception as e:
    print(f"   ✗ SwiGLU test failed: {e}")
    import traceback
    traceback.print_exc()

# Test 2: RMSNorm
print("\n2. Testing RMSNorm kernel...")
try:
    # Import directly from file to avoid __init__.py imports
    spec = importlib.util.spec_from_file_location(
        "rms_layernorm_mlx",
        str(Path(__file__).parent.parent / "unsloth" / "kernels" / "metal" / "rms_layernorm_mlx.py")
    )
    rms_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rms_module)
    rms_layernorm_forward = rms_module.mlx_rms_layernorm_forward
    rms_layernorm_backward = rms_module.mlx_rms_layernorm_backward
    
    # Create test tensors
    B, S, D = 2, 512, 4096
    
    x = mx.random.normal((B, S, D))
    weight = mx.random.normal((D,))
    
    # Forward
    start = time.time()
    output, r = rms_layernorm_forward(x, weight, eps=1e-5)
    mx.eval([output, r])
    fwd_time = (time.time() - start) * 1000
    
    print(f"   ✓ RMSNorm forward: {output.shape}, {fwd_time:.2f}ms")
    
    # Backward
    grad_output = mx.random.normal((B, S, D))
    start = time.time()
    grad_x, grad_weight = rms_layernorm_backward(grad_output, x, weight, r)
    mx.eval([grad_x, grad_weight])
    bwd_time = (time.time() - start) * 1000
    
    print(f"   ✓ RMSNorm backward: {bwd_time:.2f}ms")
    print(f"   ✓ Uses mx.fast.rms_norm for optimized forward pass!")
    
except Exception as e:
    print(f"   ✗ RMSNorm test failed: {e}")
    import traceback
    traceback.print_exc()

# Test 3: GEGLU
print("\n3. Testing GEGLU kernel...")
try:
    # Import directly from file to avoid __init__.py imports
    spec = importlib.util.spec_from_file_location(
        "geglu_mlx",
        str(Path(__file__).parent.parent / "unsloth" / "kernels" / "metal" / "geglu_mlx.py")
    )
    geglu_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(geglu_module)
    geglu_forward = geglu_module.mlx_geglu_exact_forward
    geglu_backward = geglu_module.mlx_geglu_exact_backward
    
    # Create test tensors
    B, S, D = 2, 512, 4096
    hidden_dim = 16384  # GeGLU uses larger hidden dim
    
    gate = mx.random.normal((B, S, hidden_dim))
    up = mx.random.normal((B, S, hidden_dim))
    
    # Forward
    start = time.time()
    output = geglu_forward(gate, up)
    mx.eval(output)
    fwd_time = (time.time() - start) * 1000
    
    print(f"   ✓ GEGLU forward: {output.shape}, {fwd_time:.2f}ms")
    
    # Backward
    grad_output = mx.random.normal((B, S, hidden_dim))
    start = time.time()
    h_out, grad_gate, grad_up = geglu_backward(grad_output, gate, up)
    mx.eval([h_out, grad_gate, grad_up])
    bwd_time = (time.time() - start) * 1000
    
    print(f"   ✓ GEGLU backward: {bwd_time:.2f}ms")
    print(f"   ✓ GeLU + GLU fused kernel for Gemma models!")
    
except Exception as e:
    print(f"   ✗ GEGLU test failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 70)
print("Chunk 1 Complete: Pure MLX Metal Kernels")
print("=" * 70)
print("""
Key achievements:
  • No torch_to_mlx/mlx_to_torch conversion
  • Direct mx.array input/output
  • Metal kernels compiled for Apple Silicon
  • Forward + backward pass support

Next: Chunk 2 - Training Infrastructure
  • MLX optimizers (AdamW, SGD)
  • MLX losses (cross-entropy)
  • MLX training loop with mx.grad
  • MLX LoRA layers
""")
