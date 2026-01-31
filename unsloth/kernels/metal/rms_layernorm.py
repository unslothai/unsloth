# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
# Licensed under the Apache License, Version 2.0.

import torch
import mlx.core as mx
from unsloth.kernels.mlx.bridge import torch_to_mlx, mlx_to_torch, mlx_context
from functools import lru_cache
from typing import Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    import torch

# Optimized Metal Shaders for RMSNorm (Vectorized bodies)
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
    
    device const half4* X_vec = (device const half4*)(X + row * C);
    device half4* Y_vec = (device half4*)(Y + row * C);
    device const half4* W_vec = (device const half4*)W;
    
    uint C_vec = C >> 2;
    float acc = 0.0f;
    for (uint i = lid; i < C_vec; i += tpg) {
        float4 v = float4(X_vec[i]);
        acc = fma(v.x, v.x, acc);
        acc = fma(v.y, v.y, acc);
        acc = fma(v.z, v.z, acc);
        acc = fma(v.w, v.w, acc);
    }
    
    // Tail handling for C not multiple of 4
    if (lid == 0) {
        for (uint i = C_vec << 2; i < C; ++i) {
            float v = float(X[row * C + i]);
            acc = fma(v, v, acc);
        }
    }
    
    // Reduce acc across threadgroup
    float grp_sum = simd_sum(acc);
    threadgroup float tg_sums[8];
    if ((lid & 31) == 0) tg_sums[lid >> 5] = grp_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    threadgroup float shared_inv_var;
    if (lid == 0) {
        float total_sum = 0.0f;
        for (uint i = 0; i < (tpg + 31) / 32; ++i) total_sum += tg_sums[i];
        shared_inv_var = precise::rsqrt(total_sum / float(C) + EPS);
        r[row] = shared_inv_var;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    float inv_var = shared_inv_var;
    for (uint i = lid; i < C_vec; i += tpg) {
        float4 x = float4(X_vec[i]);
        float4 w = float4(W_vec[i]);
        float4 weight = GEMMA ? (w + 1.0f) : w;
        Y_vec[i] = half4(x * inv_var * weight);
    }
    
    if (lid == 0) {
        for (uint i = C_vec << 2; i < C; ++i) {
            float x = float(X[row * C + i]);
            float w = float(W[i]);
            float weight = GEMMA ? (w + 1.0f) : w;
            Y[row * C + i] = half(x * inv_var * weight);
        }
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
    
    device const half4* dY_vec = (device const half4*)(dY + row * C);
    device const half4* X_vec = (device const half4*)(X + row * C);
    device half4* dX_vec = (device half4*)(dX + row * C);
    device float4* dW_row_vec = (device float4*)(dW_rows + row * C);
    device const half4* W_vec = (device const half4*)W;
    
    float inv_var = r[row];
    uint C_vec = C >> 2;
    float dot = 0.0f;
    
    for (uint i = lid; i < C_vec; i += tpg) {
        float4 dy = float4(dY_vec[i]);
        float4 x = float4(X_vec[i]);
        float4 w = float4(W_vec[i]);
        float4 w_eff = GEMMA ? (w + 1.0f) : w;
        
        float4 normed = x * inv_var;
        float4 dy_w = dy * w_eff;
        
        // Save row-wise dW components
        dW_row_vec[i] = dy * normed;
        
        dot = fma(dy_w.x, normed.x, dot);
        dot = fma(dy_w.y, normed.y, dot);
        dot = fma(dy_w.z, normed.z, dot);
        dot = fma(dy_w.w, normed.w, dot);
    }
    
    if (lid == 0) {
        for (uint i = C_vec << 2; i < C; ++i) {
            float dy = float(dY[row * C + i]);
            float x = float(X[row * C + i]);
            float w = float(W[i]);
            float w_eff = GEMMA ? (w + 1.0f) : w;
            float normed = x * inv_var;
            dW_rows[row * C + i] = dy * normed;
            dot = fma(dy * w_eff, normed, dot);
        }
    }
    
    // Reduce dot across threadgroup
    float grp_dot = simd_sum(dot);
    threadgroup float tg_dots[8];
    if ((lid & 31) == 0) tg_dots[lid >> 5] = grp_dot;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    threadgroup float shared_total_dot;
    if (lid == 0) {
        float total_dot = 0.0f;
        for (uint i = 0; i < (tpg + 31) / 32; ++i) total_dot += tg_dots[i];
        shared_total_dot = total_dot;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    float total_dot = shared_total_dot;
    float N_inv = 1.0f / float(C);
    
    for (uint i = lid; i < C_vec; i += tpg) {
        float4 dy = float4(dY_vec[i]);
        float4 x = float4(X_vec[i]);
        float4 w = float4(W_vec[i]);
        float4 w_eff = GEMMA ? (w + 1.0f) : w;
        
        float4 normed = x * inv_var;
        float4 dy_w = dy * w_eff;
        
        dX_vec[i] = half4(inv_var * (dy_w - normed * total_dot * N_inv));
    }
    
    if (lid == 0) {
        for (uint i = C_vec << 2; i < C; ++i) {
            float dy = float(dY[row * C + i]);
            float x = float(X[row * C + i]);
            float w = float(W[i]);
            float w_eff = GEMMA ? (w + 1.0f) : w;
            float normed = x * inv_var;
            dX[row * C + i] = half(inv_var * (dy * w_eff - normed * total_dot * N_inv));
        }
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
