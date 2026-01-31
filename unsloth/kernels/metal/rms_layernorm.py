# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
# Licensed under the Apache License, Version 2.0.

import torch
import mlx.core as mx
from unsloth.kernels.mlx.bridge import torch_to_mlx, mlx_to_torch, mlx_context
from functools import lru_cache
import os

# Load MSL source
_LAYERNORM_METAL_PATH = os.path.join(os.path.dirname(__file__), "layernorm.metal")
with open(_LAYERNORM_METAL_PATH, "r") as f:
    LAYERNORM_METAL_SOURCE = f.read()

@lru_cache(maxsize=1)
def _get_rms_kernel_f16():
    return mx.fast.metal_kernel(
        name="rms_layernorm_gemma_forward_f16",
        input_names=["X", "W", "n_rows", "n_cols", "eps"],
        output_names=["Y", "rms_inv_out"],
        source=LAYERNORM_METAL_SOURCE,
    )

@lru_cache(maxsize=1)
def _get_rms_kernel_f32():
    return mx.fast.metal_kernel(
        name="rms_layernorm_gemma_forward",
        input_names=["X", "W", "n_rows", "n_cols", "eps"],
        output_names=["Y", "rms_inv_out"],
        source=LAYERNORM_METAL_SOURCE,
    )

def metal_rms_layernorm(X, W, eps, gemma=False):
    """
    Fused Metal RMS LayerNorm (Forward only currently).
    Supports both Standard and Gemma (1+W) scaling.
    """
    if not gemma:
        # Standard RMSNorm: Y = X * rsqrt(mean(X^2) + eps) * W
        # The Gemma kernel uses (1+W). To get Standard, we pass (W-1).
        W = W - 1.0

    shape = X.shape
    dim = shape[-1]
    X_flat = X.view(-1, dim)
    n_rows, n_cols = X_flat.shape
    
    with mlx_context():
        X_mlx = torch_to_mlx(X_flat)
        W_mlx = torch_to_mlx(W)
        
        n_rows_mx = mx.array([n_rows], dtype=mx.uint32)
        n_cols_mx = mx.array([n_cols], dtype=mx.uint32)
        eps_mx = mx.array([eps], dtype=mx.float32)
        
        if X.dtype == torch.float32:
            kernel = _get_rms_kernel_f32()
            out_dtype = mx.float32
        else:
            kernel = _get_rms_kernel_f16()
            out_dtype = mx.float16
            
        outputs = kernel(
            inputs=[X_mlx, W_mlx, n_rows_mx, n_cols_mx, eps_mx],
            output_shapes=[(n_rows, n_cols), (n_rows,)],
            output_dtypes=[out_dtype, mx.float32],
            grid=(n_rows, 1, 1),
            threadgroup=(256, 1, 1),
        )
        
        Y = mlx_to_torch(outputs[0])
        return Y.view(*shape)
