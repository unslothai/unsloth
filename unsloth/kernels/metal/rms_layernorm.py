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
    
    // Scalar loop for absolute correctness verification
    for (uint i = lid; i < C; i += tpg) {
        float dy = float(dY_row[i]);
        float x = float(X_row[i]);
        float w = float(W[i]);
        float w_eff = GEMMA ? (w + 1.0f) : w;
        
        float normed = x * inv_var;
        float dy_w = dy * w_eff;
        
        // Store row-wise dW components
        dW_row[i] = dy * normed;
        
        // Accumulate row-wise dot product: sum(dy * w_eff * normed)
        dot += dy_w * normed;
    }
    
    // Reduction for dot product across threadgroup
    threadgroup float tg_dots[32];
    float grp_dot = simd_sum(dot);
    if ((lid & 31) == 0) tg_dots[lid >> 5] = grp_dot;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    if (lid == 0) {
        float total_dot = 0.0f;
        for (uint i = 0; i < (tpg + 31) / 32; ++i) total_dot += tg_dots[i];
        tg_dots[0] = total_dot;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    float total_dot_n = tg_dots[0] * N_inv;
    
    // Second pass: Compute dX
    for (uint i = lid; i < C; i += tpg) {
        float dy = float(dY_row[i]);
        float x = float(X_row[i]);
        float w = float(W[i]);
        float w_eff = GEMMA ? (w + 1.0f) : w;
        
        float normed = x * inv_var;
        // dX = inv_var * (dy_w - normed * (sum(dy_w * normed) / n))
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
    Optimized Forward using MLX Native RMSNorm where possible.
    Includes explicit r computation for backward pass.
    """
    import mlx.core as mx

    # Important: Cast to float32 for stable computation
    X_f32 = X_mlx.astype(mx.float32)
    # mean(X^2) in float32
    r = mx.rsqrt(mx.mean(mx.square(X_f32), axis = -1) + eps)

    if not gemma:
        Y = mx.fast.rms_norm(X_mlx, W_mlx, eps)
    else:
        # Gemma uses (1 + W). We do it in float32 for accuracy.
        W_f32 = W_mlx.astype(mx.float32)
        Y = (X_f32 * r[..., None]) * (W_f32 + 1.0)
        Y = Y.astype(X_mlx.dtype)

    return Y, r


def mlx_rms_layernorm_backward(dY_mlx, X_mlx, W_mlx, r_mlx, gemma = False):
    """Custom Metal kernel for the backward pass with dW reduction."""
    import mlx.core as mx

    shape = X_mlx.shape
    dim = shape[-1]
    dY_flat = dY_mlx.reshape(-1, dim).contiguous()
    X_flat = X_mlx.reshape(-1, dim).contiguous()
    r_flat = r_mlx.flatten().contiguous()
    W_contig = W_mlx.contiguous()

    n_rows, n_cols = X_flat.shape

    kernel = _get_backward_kernel()

    n_rows_mx = mx.array([n_rows], dtype = mx.uint32)
    n_cols_mx = mx.array([n_cols], dtype = mx.uint32)
    gemma_mx = mx.array([1 if gemma else 0], dtype = mx.uint32)

    # Simple threadgroup size
    tpg = min(256, ((n_cols + 31) // 32) * 32)

    outputs = kernel(
        inputs = [dY_flat, X_flat, W_contig, r_flat, n_rows_mx, n_cols_mx, gemma_mx],
        output_shapes = [(n_rows * n_cols,), (n_rows * n_cols,)],
        output_dtypes = [X_mlx.dtype, mx.float32],
        grid = (n_rows, 1, 1),
        threadgroup = (tpg, 1, 1),
    )

    dX = outputs[0].reshape(shape)
    dW = mx.sum(outputs[1].reshape(n_rows, n_cols), axis = 0).astype(W_mlx.dtype)
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
