# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
# Licensed under the Apache License, Version 2.0.

from unsloth.kernels.mlx.bridge import torch_to_mlx, mlx_to_torch, mlx_context
from functools import lru_cache
from typing import Tuple, TYPE_CHECKING, Optional, Dict

if TYPE_CHECKING:
    import torch

# Metal Shader for RMSNorm Backward
# -----------------------------------------------------------------------------

# Metal Shader for RMSNorm Backward (Optimized)
# -----------------------------------------------------------------------------

RMS_BACKWARD_BODY = """
    uint row = threadgroup_position_in_grid.x;
    uint lid = thread_position_in_threadgroup.x;
    uint tpg = threads_per_threadgroup.x;
    
    uint R = n_rows[0];
    uint C = n_cols[0];
    bool GEMMA = (gemma[0] != 0);
    
    if (row >= R) return;
    
    device const half* dY_row = dY + row * C;
    device const half* X_row = X + row * C;
    device half* dX_out = dX_ptr + row * C;
    device float* dW_out = dW_rows_ptr + row * C;
    
    float inv_var = r[row];
    float dot = 0.0f;
    float N_inv = 1.0f / float(C);
    
    // 1. Calculate row-wise components and populate dW
    for (uint i = lid; i < C; i += tpg) {
        float dy = float(dY_row[i]);
        float x = float(X_row[i]);
        float w = float(W[i]);
        float w_eff = GEMMA ? (w + 1.0f) : w;
        
        float normed = x * inv_var;
        float dy_w = dy * w_eff;
        
        dW_out[i] = dy * normed;
        dot += dy_w * normed;
    }
    
    // 2. Parallel reduction for dot product in threadgroup
    threadgroup float shared_dots[256];
    shared_dots[lid] = dot;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    for (uint s = 128; s > 0; s >>= 1) {
        if (lid < s && lid + s < tpg) {
            shared_dots[lid] += shared_dots[lid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    float total_dot_n = shared_dots[0] * N_inv;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // 3. Compute dX
    for (uint i = lid; i < C; i += tpg) {
        float dy = float(dY_row[i]);
        float x = float(X_row[i]);
        float w = float(W[i]);
        float w_eff = GEMMA ? (w + 1.0f) : w;
        
        float normed = x * inv_var;
        dX_out[i] = half(inv_var * (dy * w_eff - normed * total_dot_n));
    }
"""


@lru_cache(maxsize=1)
def _get_backward_kernel():
    import mlx.core as mx

    return mx.fast.metal_kernel(
        name="rms_backward_v3",
        input_names=["dY", "X", "W", "r", "n_rows", "n_cols", "gemma"],
        output_names=["dX_ptr", "dW_rows_ptr"],
        source=RMS_BACKWARD_BODY,
    )


def mlx_rms_layernorm_forward(X_mlx, W_mlx, eps, gemma=False):
    """
    Optimized Forward using MLX.
    Uses float32 for intermediate variance/rsqrt for high parity.
    """
    import mlx.core as mx

    # Cast to float32 for robust variance calculation
    X_f32 = X_mlx.astype(mx.float32)

    # r = 1 / sqrt(mean(X^2) + eps)
    r = mx.rsqrt(mx.mean(mx.square(X_f32), axis=-1) + eps)

    if not gemma:
        # Standard RMSNorm using native where possible for speed
        # But native doesn't return r, so we might as well do it in float32 for parity
        Y = (X_f32 * r[..., None]) * W_mlx.astype(mx.float32)
    else:
        # Gemma uses (W + 1)
        Y = (X_f32 * r[..., None]) * (W_mlx.astype(mx.float32) + 1.0)

    return Y.astype(X_mlx.dtype), r


def mlx_rms_layernorm_backward(dY_mlx, X_mlx, W_mlx, r_mlx, gemma=False):
    """
    Optimized Backward using pure MLX primitives.
    MLX's graph optimizer fuses these operations for high performance.
    """
    import mlx.core as mx

    # Cast to float32 for high numerical precision
    X_f32 = X_mlx.astype(mx.float32)
    W_f32 = W_mlx.astype(mx.float32)
    dY_f32 = dY_mlx.astype(mx.float32)
    r_f32 = r_mlx.astype(mx.float32)[..., None]

    W_eff = (W_f32 + 1.0) if gemma else W_f32

    # 1. dW = sum(dY * X_norm)
    X_norm = X_f32 * r_f32
    # Sum over all dimensions except the last one (dim)
    # This handles both (B, S, D) and (N, D) shapes correctly
    dW = mx.sum(dY_f32 * X_norm, axis=list(range(X_mlx.ndim - 1)))

    # 2. dX = r * (dy_w - X_norm * mean(dy_w * X_norm))
    # This formula is numerically stable and matches Triton/PyTorch.
    dy_w = dY_f32 * W_eff
    # Row-wise mean of (dy_w * X_norm)
    mean_dot = mx.mean(dy_w * X_norm, axis=-1, keepdims=True)
    dX = r_f32 * (dy_w - X_norm * mean_dot)

    return dX.astype(X_mlx.dtype), dW.astype(W_mlx.dtype)


def metal_rms_layernorm(X, W, eps, gemma=False):
    """Fused Metal RMS LayerNorm (PyTorch interface)."""
    with mlx_context():
        X_mlx = torch_to_mlx(X)

        # Check for cached MLX weight
        W_mlx = getattr(W, "_mlx_cache", None)
        if W_mlx is None:
            W_mlx = torch_to_mlx(W)
            W._mlx_cache = W_mlx

        Y_mlx, r_mlx = mlx_rms_layernorm_forward(X_mlx, W_mlx, eps, gemma)

        Y_torch = mlx_to_torch(Y_mlx)
        r_torch = mlx_to_torch(r_mlx)

        # Fast path - Cache the MLX version of r to avoid re-converting in backward
        r_torch._mlx_tensor = r_mlx

        # CHAINING: Attach MLX output to PyTorch tensor so next layer (SwiGLU)
        # can use it directly without converting back from PyTorch
        Y_torch._mlx_cache = Y_mlx

        return Y_torch, r_torch


def metal_rms_layernorm_backward(dY, X, W, r, eps, gemma=False):
    """Fused Metal RMS LayerNorm Backward (PyTorch interface)."""
    with mlx_context():
        dY_mlx = torch_to_mlx(dY)
        X_mlx = torch_to_mlx(X)

        # Check for cached MLX weight
        W_mlx = getattr(W, "_mlx_cache", None)
        if W_mlx is None:
            W_mlx = torch_to_mlx(W)
            # We can cache it here too if it wasn't already (though forward usually catches it)
            W._mlx_cache = W_mlx

        # Fast path - Retrieve cached MLX version of r if available
        r_mlx = getattr(r, "_mlx_tensor", None)
        if r_mlx is None:
            r_mlx = torch_to_mlx(r)

        dX_mlx, dW_mlx = mlx_rms_layernorm_backward(dY_mlx, X_mlx, W_mlx, r_mlx, gemma)
        return mlx_to_torch(dX_mlx), mlx_to_torch(dW_mlx)
