# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
# Licensed under the Apache License, Version 2.0.

import torch
from unsloth.kernels.mlx.bridge import torch_to_mlx, mlx_to_torch, mlx_context
from functools import lru_cache
from typing import Tuple, TYPE_CHECKING, Optional

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
    device half* dX_row = dX + row * C;
    device float* dW_row = dW + row * C;
    
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
        
        dW_row[i] = dy * normed;
        dot += dy_w * normed;
    }
    
    // 2. Parallel reduction for dot product in threadgroup
    // We use a robust reduction that handles non-power-of-2 tpg (though tpg is usually 256 here)
    threadgroup float shared_dots[256];
    shared_dots[lid] = dot;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    for (uint s = 128; s > 0; s >>= 1) {
        if (lid < s && lid + s < tpg) {
            shared_dots[lid] += shared_dots[lid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // The result is in shared_dots[0]
    float total_dot_n = shared_dots[0] * N_inv;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // 3. Compute dX
    for (uint i = lid; i < C; i += tpg) {
        float dy = float(dY_row[i]);
        float x = float(X_row[i]);
        float w = float(W[i]);
        float w_eff = GEMMA ? (w + 1.0f) : w;
        
        float normed = x * inv_var;
        dX_row[i] = half(inv_var * (dy * w_eff - normed * total_dot_n));
    }
"""


@lru_cache(maxsize = 1)
def _get_backward_kernel():
    import mlx.core as mx

    return mx.fast.metal_kernel(
        name = "rms_backward_v2",
        input_names = ["dY", "X", "W", "r", "n_rows", "n_cols", "gemma"],
        output_names = ["dX", "dW"],
        source = RMS_BACKWARD_BODY,
    )


def mlx_rms_layernorm_forward(X_mlx, W_mlx, eps, gemma = False):
    """
    Optimized Forward using MLX.
    Uses float32 for intermediate variance/rsqrt for high parity.
    """
    import mlx.core as mx

    # Cast to float32 for robust variance calculation
    X_f32 = X_mlx.astype(mx.float32)

    # r = 1 / sqrt(mean(X^2) + eps)
    r = mx.rsqrt(mx.mean(mx.square(X_f32), axis = -1) + eps)

    if not gemma:
        # Standard RMSNorm using native where possible for speed
        # But native doesn't return r, so we might as well do it in float32 for parity
        Y = (X_f32 * r[..., None]) * W_mlx.astype(mx.float32)
    else:
        # Gemma uses (W + 1)
        Y = (X_f32 * r[..., None]) * (W_mlx.astype(mx.float32) + 1.0)

    return Y.astype(X_mlx.dtype), r


def mlx_rms_layernorm_backward(dY_mlx, X_mlx, W_mlx, r_mlx, gemma = False):
    """Custom Metal kernel for the backward pass with dW reduction."""
    import mlx.core as mx

    shape = X_mlx.shape
    dim = shape[-1]
    dY_flat = dY_mlx.reshape(-1, dim)
    X_flat = X_mlx.reshape(-1, dim)
    r_flat = r_mlx.flatten()
    W_contig = W_mlx

    n_rows, n_cols = X_flat.shape

    kernel = _get_backward_kernel()

    n_rows_mx = mx.array([n_rows], dtype = mx.uint32)
    n_cols_mx = mx.array([n_cols], dtype = mx.uint32)
    gemma_mx = mx.array([1 if gemma else 0], dtype = mx.uint32)

    # Simple threadgroup size
    tpg = min(256, ((n_cols + 31) // 32) * 32)

    outputs = kernel(
        inputs = [dY_flat, X_flat, W_contig, r_flat, n_rows_mx, n_cols_mx, gemma_mx],
        output_shapes = [(n_rows, n_cols), (n_rows, n_cols)],
        output_dtypes = [X_mlx.dtype, mx.float32],
        grid = (n_rows, 1, 1),
        threadgroup = (tpg, 1, 1),
    )

    dX = outputs[0].reshape(shape)
    dW = mx.sum(outputs[1], axis = 0).astype(W_mlx.dtype)
    return dX, dW


def metal_rms_layernorm(X, W, eps, gemma = False):
    """Fused Metal RMS LayerNorm (PyTorch interface)."""
    with mlx_context():
        X_mlx = torch_to_mlx(X)
        W_mlx = torch_to_mlx(W)
        Y_mlx, r_mlx = mlx_rms_layernorm_forward(X_mlx, W_mlx, eps, gemma)
        return mlx_to_torch(Y_mlx), mlx_to_torch(r_mlx)


def metal_rms_layernorm_backward(dY, X, W, r, eps, gemma = False):
    """Fused Metal RMS LayerNorm Backward (PyTorch interface)."""
    with mlx_context():
        dY_mlx = torch_to_mlx(dY)
        X_mlx = torch_to_mlx(X)
        W_mlx = torch_to_mlx(W)
        r_mlx = torch_to_mlx(r)
        dX_mlx, dW_mlx = mlx_rms_layernorm_backward(dY_mlx, X_mlx, W_mlx, r_mlx, gemma)
        return mlx_to_torch(dX_mlx), mlx_to_torch(dW_mlx)
