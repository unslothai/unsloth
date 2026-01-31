# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
# Licensed under the Apache License, Version 2.0.

import torch
import mlx.core as mx
from unsloth.kernels.mlx.bridge import torch_to_mlx, mlx_to_torch, mlx_context
from functools import lru_cache
from typing import Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    import torch

# Optimized Metal Shaders for RMSNorm (Robust SIMD-based bodies)
# -----------------------------------------------------------------------------

RMS_FORWARD_BODY = """
    uint row = threadgroup_position_in_grid.x;
    uint lid = thread_position_in_threadgroup.x;
    uint tpg = threads_per_threadgroup.x;
    
    uint R = n_rows[0];
    uint C = n_cols[0];
    float EPS = eps[0];
    bool GEMMA = (gemma[0] != 0);
    
    if (row >= R) return;
    
    device const half* X_row = X + row * C;
    device half* Y_row = Y + row * C;
    
    float acc = 0.0f;
    for (uint i = lid; i < C; i += tpg) {
        float v = float(X_row[i]);
        acc = fma(v, v, acc);
    }
    
    // SIMD-aware reduction
    float grp_sum = simd_sum(acc);
    threadgroup float tg_sums[8]; // Max 256 threads / 32 = 8 SIMDs
    if ((lid & 31) == 0) {
        tg_sums[lid >> 5] = grp_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    float total_sum = (lid < (tpg + 31) / 32) ? tg_sums[lid] : 0.0f;
    total_sum = simd_sum(total_sum);
    
    float inv_var = precise::rsqrt(total_sum / float(C) + EPS);
    if (lid == 0) r[row] = inv_var;
    
    for (uint i = lid; i < C; i += tpg) {
        float x = float(X_row[i]);
        float w = float(W[i]);
        float weight = GEMMA ? (w + 1.0f) : w;
        Y_row[i] = half(x * inv_var * weight);
    }
"""

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
    
    for (uint i = lid; i < C; i += tpg) {
        float dy = float(dY_row[i]);
        float x = float(X_row[i]);
        float w = float(W[i]);
        float w_eff = GEMMA ? (w + 1.0f) : w;
        
        float normed = x * inv_var;
        float dy_w = dy * w_eff;
        
        // Accumulate row-wise dW components
        dW_row[i] = dy * normed;
        
        dot = fma(dy_w, normed, dot);
    }
    
    // SIMD-aware reduction for dot product
    float grp_dot = simd_sum(dot);
    threadgroup float tg_dots[8];
    if ((lid & 31) == 0) {
        tg_dots[lid >> 5] = grp_dot;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    float total_dot = (lid < (tpg + 31) / 32) ? tg_dots[lid] : 0.0f;
    total_dot = simd_sum(total_dot);
    
    float N_inv = 1.0f / float(C);
    
    for (uint i = lid; i < C; i += tpg) {
        float dy = float(dY_row[i]);
        float x = float(X_row[i]);
        float w = float(W[i]);
        float w_eff = GEMMA ? (w + 1.0f) : w;
        
        float normed = x * inv_var;
        float dy_w = dy * w_eff;
        
        // dX = inv_var * (dy_w - normed * total_dot / N)
        dX_row[i] = half(inv_var * (dy_w - normed * total_dot * N_inv));
    }
"""

@lru_cache(maxsize = 1)
def _get_forward_kernel():
    return mx.fast.metal_kernel(
        name = "rms_forward",
        input_names = ["X", "W", "n_rows", "n_cols", "eps", "gemma"],
        output_names = ["Y", "r"],
        source = RMS_FORWARD_BODY,
    )

@lru_cache(maxsize = 1)
def _get_backward_kernel():
    return mx.fast.metal_kernel(
        name = "rms_backward",
        input_names = ["dY", "X", "W", "r", "n_rows", "n_cols", "gemma"],
        output_names = ["dX", "dW_rows"],
        source = RMS_BACKWARD_BODY,
    )

def mlx_rms_layernorm_forward(X_mlx, W_mlx, eps, gemma = False):
    """Pure MLX RMSNorm forward."""
    shape = X_mlx.shape
    dim = shape[-1]
    X_flat = X_mlx.reshape(-1, dim)
    n_rows, n_cols = X_flat.shape
    
    kernel = _get_forward_kernel()
    
    n_rows_mx = mx.array([n_rows], dtype = mx.uint32)
    n_cols_mx = mx.array([n_cols], dtype = mx.uint32)
    eps_mx = mx.array([eps], dtype = mx.float32)
    gemma_mx = mx.array([1 if gemma else 0], dtype = mx.uint32)
    
    outputs = kernel(
        inputs = [X_flat, W_mlx, n_rows_mx, n_cols_mx, eps_mx, gemma_mx],
        output_shapes = [(n_rows * n_cols,), (n_rows,)],
        output_dtypes = [X_mlx.dtype, mx.float32],
        grid = (n_rows, 1, 1),
        threadgroup = (min(256, n_cols), 1, 1),
    )
    return outputs[0].reshape(shape), outputs[1]

def mlx_rms_layernorm_backward(dY_mlx, X_mlx, W_mlx, r_mlx, gemma = False):
    """Pure MLX RMSNorm backward."""
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
        output_shapes = [(n_rows * n_cols,), (n_rows * n_cols,)],
        output_dtypes = [X_mlx.dtype, mx.float32],
        grid = (n_rows, 1, 1),
        threadgroup = (min(256, n_cols), 1, 1),
    )
    
    # Final reduction for dW across rows
    dW = mx.sum(outputs[1].reshape(n_rows, n_cols), axis = 0).astype(W_mlx.dtype)
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
