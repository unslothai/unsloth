# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

# Decode residual add + RMSNorm in one launch, rounding exactly as `X += residual` + fast_rms_layernorm_inference.

import os
import torch
import triton
import triton.language as tl
from .utils import MAX_FUSED_SIZE, torch_gpu_device
from .rms_layernorm import _launch, _eager_kernel, _settings

__all__ = [
    "fast_add_rms_layernorm_inference",
    "fast_add_rms_layernorm_inference_ok",
]

FUSED_ADD_RMSNORM = os.environ.get("UNSLOTH_FUSED_ADD_RMSNORM", "1") != "0"


@triton.jit
def _add_rms_layernorm_inference(
    X,
    R,
    W,
    X_row_stride,
    R_row_stride,
    n_cols: tl.constexpr,
    eps: tl.constexpr,
    HAS_RESIDUAL: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols
    X += row * X_row_stride
    R += row * R_row_stride

    h = tl.load(X + cols, mask = mask, other = 0)
    if HAS_RESIDUAL:
        r = tl.load(R + cols, mask = mask, other = 0)
        h = (h.to(tl.float32) + r.to(tl.float32)).to(X.dtype.element_ty)
    tl.store(R + cols, h.to(R.dtype.element_ty), mask = mask)

    h = h.to(tl.float32)
    # torch.mean multiplies by 1 / n.
    variance = tl.sum(h * h, axis = 0) * (1.0 / n_cols)
    inv_var = tl.math.rsqrt(variance + tl.full((), eps, tl.float32))
    normed = (h * inv_var).to(X.dtype.element_ty)
    w = tl.load(W + cols, mask = mask, other = 0)
    y = normed.to(tl.float32) * w.to(tl.float32)
    tl.store(X + cols, y.to(X.dtype.element_ty), mask = mask)


def _rows(X):
    # Unit-stride (rows, n_cols) view or None: the batch > 1 decode MLP output is a strided slice.
    if X.stride(-1) != 1:
        return None
    try:
        return X.view(-1, X.shape[-1])
    except RuntimeError:
        return None


def fast_add_rms_layernorm_inference_ok(X, residual, weight):
    return (
        FUSED_ADD_RMSNORM
        and X.device.type == "cuda"
        and residual.shape == X.shape
        and residual.device == X.device
        and weight.device == X.device
        and weight.is_contiguous()
        and weight.numel() == X.shape[-1]
        and X.shape[-1] <= MAX_FUSED_SIZE
        and _rows(X) is not None
        and _rows(residual) is not None
        and not torch.compiler.is_compiling()
    )


def fast_add_rms_layernorm_inference(
    X,
    residual,
    weight,
    eps,
    add = True,
):
    """Decode step: residual = X + residual (rounded to X's dtype; add = False: residual = X),
    then X = RMSNorm(residual) * weight in place. Returns X. No autograd."""
    n_cols = X.shape[-1]
    BLOCK_SIZE, num_warps = _settings(n_cols)
    device = X.device
    with torch_gpu_device(device):
        _launch(
            _eager_kernel,
            _add_rms_layernorm_inference,
            (X.numel() // n_cols,),
            (X, residual, weight, _rows(X).stride(0), _rows(residual).stride(0)),
            3,
            dict(
                n_cols = n_cols,
                eps = eps,
                HAS_RESIDUAL = add,
                BLOCK_SIZE = BLOCK_SIZE,
            ),
            device,
            num_warps = num_warps,
            enable_fp_fusion = False,
        )
    return X
