# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
# Licensed under the Apache License, Version 2.0.

import torch
import mlx.core as mx
from unsloth.kernels.mlx.bridge import torch_to_mlx, mlx_to_torch, mlx_context
from functools import lru_cache
from typing import Tuple, TYPE_CHECKING

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
    device float* dW_row = dW_rows + row * C;
    
    float inv_var = r[row];
    float dot = 0.0f;
    float N_inv = 1.0f / float(C);
    
    // Vectorized processing using half4
    device const half4* dY_v4 = (device const half4*)dY_row;
    device const half4* X_v4 = (device const half4*)X_row;
    device const half4* W_v4 = (device const half4*)W;
    
    uint num_vec = C / 4;
    for (uint i = lid; i < num_vec; i += tpg) {
        float4 dy = float4(dY_v4[i]);
        float4 x = float4(X_v4[i]);
        float4 w = float4(W_v4[i]);
        float4 w_eff = GEMMA ? (w + 1.0f) : w;
        
        float4 normed = x * inv_var;
        float4 dy_w = dy * w_eff;
        
        // Store intermediate dW components (row-wise)
        // dW_row is float*
        device float4* dW_v4 = (device float4*)(dW_row + i * 4);
        dW_v4[0] = dy * normed;
        
        // dot = sum(dy * w_eff * x * inv_var)
        dot += dy_w.x * normed.x + dy_w.y * normed.y + dy_w.z * normed.z + dy_w.w * normed.w;
    }
    
    // Handle tail
    for (uint i = num_vec * 4 + lid; i < C; i += tpg) {
        float dy = float(dY_row[i]);
        float x = float(X_row[i]);
        float w = float(W[i]);
        float w_eff = GEMMA ? (w + 1.0f) : w;
        float normed = x * inv_var;
        dW_row[i] = dy * normed;
        dot += (dy * w_eff) * normed;
    }
    
    // Reduction for dot product across threadgroup
    threadgroup float tg_dots[32];
    float grp_dot = simd_sum(dot);
    if ((lid & 31) == 0) tg_dots[lid >> 5] = grp_dot;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    if (lid == 0) {
        float total_dot = 0.0f;
        for (uint i = 0; i < (tpg + 31) / 32; ++i) total_dot += tg_dots[i];
        tg_dots[0] = total_dot; // Reuse first element for result
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    float total_dot_n = tg_dots[0] * N_inv;
    
    // Compute dX
    device half4* dX_v4 = (device half4*)dX_row;
    for (uint i = lid; i < num_vec; i += tpg) {
        float4 dy = float4(dY_v4[i]);
        float4 x = float4(X_v4[i]);
        float4 w = float4(W_v4[i]);
        float4 w_eff = GEMMA ? (w + 1.0f) : w;
        
        float4 normed = x * inv_var;
        // dX = inv_var * (dy_w - normed * (sum(dy_w * normed) / n))
        dX_v4[i] = half4(inv_var * (dy * w_eff - normed * tg_dots[0] * N_inv));
    }
    
    // Handle tail for dX
    for (uint i = num_vec * 4 + lid; i < C; i += tpg) {
        float dy = float(dY_row[i]);
        float x = float(X_row[i]);
        float w = float(W[i]);
        float w_eff = GEMMA ? (w + 1.0f) : w;
        float normed = x * inv_var;
        dX_row[i] = half(inv_var * (dy * w_eff - normed * tg_dots[0] * N_inv));
    }
"""


@lru_cache(maxsize = 1)
def _get_backward_kernel():
    return mx.fast.metal_kernel(
        name = "rms_backward_v2",
        input_names = ["dY", "X", "W", "r", "n_rows", "n_cols", "gemma"],
        output_names = ["dX", "dW_rows"],
        source = RMS_BACKWARD_BODY,
    )


def mlx_rms_layernorm_forward(X_mlx, W_mlx, eps, gemma = False):
    """
    Optimized Forward using MLX Native RMSNorm where possible.
    Includes explicit r computation for backward pass.
    """
    # Important: Cast to float32 for stable rsqrt computation
    X_f32 = X_mlx.astype(mx.float32)
    r = mx.rsqrt(mx.mean(mx.square(X_f32), axis = -1) + eps)

    if not gemma:
        Y = mx.fast.rms_norm(X_mlx, W_mlx, eps)
    else:
        # Gemma uses (1 + W)
        Y = mx.fast.rms_norm(X_mlx, W_mlx + 1.0, eps)

    return Y, r


def mlx_rms_layernorm_backward(dY_mlx, X_mlx, W_mlx, r_mlx, gemma = False):
    """Custom Metal kernel for the backward pass with dW reduction."""
    shape = X_mlx.shape
    dim = shape[-1]
    dY_flat = dY_mlx.reshape(-1, dim)
    X_flat = X_mlx.reshape(-1, dim)
    n_rows, n_cols = X_flat.shape

    kernel = _get_backward_kernel()

    n_rows_mx = mx.array([n_rows], dtype = mx.uint32)
    n_cols_mx = mx.array([n_cols], dtype = mx.uint32)
    gemma_mx = mx.array([1 if gemma else 0], dtype = mx.uint32)

    outputs = kernel(
        inputs = [dY_flat, X_flat, W_mlx, r_mlx, n_rows_mx, n_cols_mx, gemma_mx],
        output_shapes = [X_flat.shape, X_flat.shape],
        output_dtypes = [X_mlx.dtype, mx.float32],
        grid = (n_rows, 1, 1),
        threadgroup = (min(256, (n_cols + 3) // 4), 1, 1),
    )

    # Final reduction for dW across rows
    dW = mx.sum(outputs[1], axis = 0).astype(W_mlx.dtype)
    return outputs[0].reshape(shape), dW


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
