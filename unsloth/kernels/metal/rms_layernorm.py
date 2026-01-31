# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
# Licensed under the Apache License, Version 2.0.

import torch
import mlx.core as mx
from unsloth.kernels.mlx.bridge import torch_to_mlx, mlx_to_torch, mlx_context
from functools import lru_cache
from typing import Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    import torch

# Optimized Metal Shaders for RMSNorm
# -----------------------------------------------------------------------------

RMS_NORM_MSL = """
#include <metal_stdlib>
using namespace metal;

// Forward pass: Y = X * rsqrt(mean(X^2) + eps) * (1 + W)
// lid 0 is responsible for tail handling and computing shared variance.
kernel void rms_forward(
    device const half* X [[buffer(0)]],
    device const half* W [[buffer(1)]],
    device half* Y [[buffer(2)]],
    device float* r [[buffer(3)]],
    constant uint& n_rows [[buffer(4)]],
    constant uint& n_cols [[buffer(5)]],
    constant float& eps [[buffer(6)]],
    constant bool& gemma [[buffer(7)]],
    uint row [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint tpg [[threads_per_threadgroup]]
) {
    if (row >= n_rows) return;
    
    device const half* X_row = X + row * n_cols;
    device half* Y_row = Y + row * n_cols;
    
    float acc = 0.0f;
    for (uint i = lid; i < n_cols; i += tpg) {
        float v = float(X_row[i]);
        acc = fma(v, v, acc);
    }
    
    // Reduce acc across threadgroup
    threadgroup float tg_sums[256];
    tg_sums[lid] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    for (uint s = tpg / 2; s > 0; s >>= 1) {
        if (lid < s) tg_sums[lid] += tg_sums[lid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    float inv_var = precise::rsqrt(tg_sums[0] / float(n_cols) + eps);
    if (lid == 0) r[row] = inv_var;
    
    for (uint i = lid; i < n_cols; i += tpg) {
        float x = float(X_row[i]);
        float w = float(W[i]);
        float weight = gemma ? (w + 1.0f) : w;
        Y_row[i] = half(x * inv_var * weight);
    }
}

kernel void rms_backward(
    device const half* dY [[buffer(0)]],
    device const half* X [[buffer(1)]],
    device const half* W [[buffer(2)]],
    device const float* r [[buffer(3)]],
    device half* dX [[buffer(4)]],
    device float* dW_rows [[buffer(5)]],
    constant uint& n_rows [[buffer(6)]],
    constant uint& n_cols [[buffer(7)]],
    constant bool& gemma [[buffer(8)]],
    uint row [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint tpg [[threads_per_threadgroup]]
) {
    if (row >= n_rows) return;
    
    device const half* dY_row = dY + row * n_cols;
    device const half* X_row = X + row * n_cols;
    device half* dX_row = dX + row * n_cols;
    device float* dW_row = dW_rows + row * n_cols;
    
    float inv_var = r[row];
    
    // rowsum_dY_normed = sum((dY * W_eff) * (X * inv_var))
    float dot = 0.0f;
    for (uint i = lid; i < n_cols; i += tpg) {
        float dy = float(dY_row[i]);
        float x = float(X_row[i]);
        float w = float(W[i]);
        float w_eff = gemma ? (w + 1.0f) : w;
        
        float normed = x * inv_var;
        float dy_w = dy * w_eff;
        
        // Accumulate dW row-wise for later reduction
        dW_row[i] = dy * normed;
        
        dot = fma(dy_w, normed, dot);
    }
    
    threadgroup float tg_sums[256];
    tg_sums[lid] = dot;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    for (uint s = tpg / 2; s > 0; s >>= 1) {
        if (lid < s) tg_sums[lid] += tg_sums[lid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    float rowsum = tg_sums[0];
    float N_inv = 1.0f / float(n_cols);
    
    for (uint i = lid; i < n_cols; i += tpg) {
        float dy = float(dY_row[i]);
        float x = float(X_row[i]);
        float w = float(W[i]);
        float w_eff = gemma ? (w + 1.0f) : w;
        
        float normed = x * inv_var;
        float dy_w = dy * w_eff;
        
        // dX = (inv_var / N) * (N * dY_W - normed * rowsum_dY_normed)
        dX_row[i] = half(inv_var * (dy_w - normed * rowsum * N_inv));
    }
}
"""


@lru_cache(maxsize = 1)
def _get_forward_kernel():
    return mx.fast.metal_kernel(
        name = "rms_forward",
        input_names = ["X", "W", "n_rows", "n_cols", "eps", "gemma"],
        output_names = ["Y", "r"],
        source = RMS_NORM_MSL,
    )


@lru_cache(maxsize = 1)
def _get_backward_kernel():
    return mx.fast.metal_kernel(
        name = "rms_backward",
        input_names = ["dY", "X", "W", "r", "n_rows", "n_cols", "gemma"],
        output_names = ["dX", "dW_rows"],
        source = RMS_NORM_MSL,
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
    gemma_mx = mx.array([gemma], dtype = mx.bool_)

    outputs = kernel(
        inputs = [X_flat, W_mlx, n_rows_mx, n_cols_mx, eps_mx, gemma_mx],
        output_shapes = [(n_rows, n_cols), (n_rows,)],
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
    gemma_mx = mx.array([gemma], dtype = mx.bool_)

    outputs = kernel(
        inputs = [dY_flat, X_flat, W_mlx, r_mlx, n_rows_mx, n_cols_mx, gemma_mx],
        output_shapes = [(n_rows, n_cols), (n_rows, n_cols)],
        output_dtypes = [X_mlx.dtype, mx.float32],
        grid = (n_rows, 1, 1),
        threadgroup = (min(256, n_cols), 1, 1),
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
