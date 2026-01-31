# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""
Metal-accelerated GEGLU kernels for Apple Silicon.

Uses MLX's mx.fast.metal_kernel for maximum performance with inline shader bodies.
Supports both Exact and Approximate (Tanh) variants.
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING, Tuple, Optional

if TYPE_CHECKING:
    import torch

__all__ = [
    "metal_geglu_exact_forward",
    "metal_geglu_exact_backward",
    "metal_geglu_approx_forward",
    "metal_geglu_approx_backward",
    "is_metal_geglu_available",
]


from unsloth.kernels.mlx.bridge import torch_to_mlx, mlx_to_torch, mlx_context

_METAL_GEGLU_AVAILABLE: Optional[bool] = None


def is_metal_geglu_available() -> bool:
    """Check if Metal GEGLU kernels are available."""
    global _METAL_GEGLU_AVAILABLE
    if _METAL_GEGLU_AVAILABLE is not None:
        return _METAL_GEGLU_AVAILABLE

    try:
        import platform

        if platform.system() != "Darwin":
            _METAL_GEGLU_AVAILABLE = False
            return False

        import mlx.core as mx

        if not hasattr(mx, "fast") or not hasattr(mx.fast, "metal_kernel"):
            _METAL_GEGLU_AVAILABLE = False
            return False

        _METAL_GEGLU_AVAILABLE = True
        return True
    except Exception:
        _METAL_GEGLU_AVAILABLE = False
        return False


# -----------------------------------------------------------------------------
# Optimized Metal Shaders (Scalar operations - valid Metal code)
# -----------------------------------------------------------------------------

# Note: Metal does NOT support half8/float8, only up to half4/float4.
# We use scalar processing for correctness.

GEGLU_EXACT_FORWARD_BODY = """
    uint gid = thread_position_in_grid.x;
    uint n = n_ptr[0];
    if (gid >= n) return;
    
    // Read inputs
    float ev = float(e[gid]);
    float gv = float(g[gid]);
    
    // Compute erf approximation inline (Abramowitz & Stegun)
    float x = ev * 0.70710678118f;  // x / sqrt(2)
    float a = abs(x);
    float t = 1.0f / fma(a, 0.3275911f, 1.0f);
    float erf_val = 1.0f - (((((1.061405429f * t - 1.453152027f) * t + 1.421413741f) * t - 0.284496736f) * t + 0.254829592f) * t * exp(-a * a));
    erf_val = x < 0.0f ? -erf_val : erf_val;
    
    // GELU(e) * g
    float gelu_e = 0.5f * ev * (1.0f + erf_val);
    h[gid] = half(gelu_e * gv);
"""

GEGLU_EXACT_BACKWARD_BODY = """
    uint gid = thread_position_in_grid.x;
    uint n = n_ptr[0];
    if (gid >= n) return;
    
    float dwv = float(dw_in[gid]);
    float ev = float(e_in[gid]);
    float gv = float(g_in[gid]);
    
    // Compute erf approximation inline
    float x = ev * 0.70710678118f;
    float a = abs(x);
    float t = 1.0f / fma(a, 0.3275911f, 1.0f);
    float erf_val = 1.0f - (((((1.061405429f * t - 1.453152027f) * t + 1.421413741f) * t - 0.284496736f) * t + 0.254829592f) * t * exp(-a * a));
    erf_val = x < 0.0f ? -erf_val : erf_val;
    
    float f_partial = 0.5f * (erf_val + 1.0f);
    float f = f_partial * ev;
    
    h_out[gid] = half(f * gv);
    df_out[gid] = half(dwv * f);
    
    float dgv = dwv * gv;
    float df_de = f_partial + 0.3989422804f * ev * exp(-0.5f * ev * ev);
    de_out[gid] = half(dgv * df_de);
"""

GEGLU_APPROX_FORWARD_BODY = """
    uint gid = thread_position_in_grid.x;
    uint n = n_ptr[0];
    if (gid >= n) return;
    
    float ev = float(e[gid]);
    float gv = float(g[gid]);
    
    // Tanh GELU approximation
    float inner = 0.7978845608f * ev * (1.0f + 0.044715f * ev * ev);
    float gelu_e = 0.5f * ev * (1.0f + tanh(inner));
    h[gid] = half(gelu_e * gv);
"""

GEGLU_APPROX_BACKWARD_BODY = """
    uint gid = thread_position_in_grid.x;
    uint n = n_ptr[0];
    if (gid >= n) return;
    
    float dwv = float(dw_in[gid]);
    float ev = float(e_in[gid]);
    float gv = float(g_in[gid]);
    
    float s = 0.7978845608f;
    float T = 1.0f + tanh(s * ev * (1.0f + 0.044715f * ev * ev));
    float T2 = 0.5f * T;
    float f = T2 * ev;
    
    h_out[gid] = half(f * gv);
    df_out[gid] = half(dwv * f);
    
    float Q2 = -T2 * (T - 2.0f) * (s * ev + 3.0f * (s * ev * 0.044715f * ev * ev));
    de_out[gid] = half(dwv * gv * (T2 + Q2));
"""


@lru_cache(maxsize = 1)
def _get_exact_forward():
    import mlx.core as mx

    return mx.fast.metal_kernel(
        name = "geglu_exact_forward_v8",
        input_names = ["e", "g", "n_ptr"],
        output_names = ["h"],
        source = GEGLU_EXACT_FORWARD_BODY,
    )


@lru_cache(maxsize = 1)
def _get_exact_backward():
    import mlx.core as mx

    return mx.fast.metal_kernel(
        name = "geglu_exact_backward_v8",
        input_names = ["dw_in", "e_in", "g_in", "n_ptr"],
        output_names = ["h_out", "df_out", "de_out"],
        source = GEGLU_EXACT_BACKWARD_BODY,
    )


@lru_cache(maxsize = 1)
def _get_approx_forward():
    import mlx.core as mx

    return mx.fast.metal_kernel(
        name = "geglu_approx_forward_v8",
        input_names = ["e", "g", "n_ptr"],
        output_names = ["h"],
        source = GEGLU_APPROX_FORWARD_BODY,
    )


@lru_cache(maxsize = 1)
def _get_approx_backward():
    import mlx.core as mx

    return mx.fast.metal_kernel(
        name = "geglu_approx_backward_v8",
        input_names = ["dw_in", "e_in", "g_in", "n_ptr"],
        output_names = ["h_out", "df_out", "de_out"],
        source = GEGLU_APPROX_BACKWARD_BODY,
    )


def _metal_geglu_forward(e, g, kernel_fn):
    import mlx.core as mx

    shape = e.shape
    n = e.numel()
    with mlx_context():
        e_mlx = torch_to_mlx(e).flatten()
        g_mlx = torch_to_mlx(g).flatten()
        n_arr = mx.array([n], dtype = mx.uint32)
        grid_size = n  # Scalar ops: one thread per element
        out = kernel_fn(
            inputs = [e_mlx, g_mlx, n_arr],
            output_shapes = [(n,)],
            output_dtypes = [mx.float16],
            grid = (grid_size, 1, 1),
            threadgroup = (min(256, grid_size), 1, 1),
        )
        return mlx_to_torch(out[0]).view(*shape)


def _metal_geglu_backward(dw, e, g, kernel_fn):
    import mlx.core as mx

    shape = e.shape
    n = e.numel()
    with mlx_context():
        dw_mlx = torch_to_mlx(dw).flatten()
        e_mlx = torch_to_mlx(e).flatten()
        g_mlx = torch_to_mlx(g).flatten()
        n_arr = mx.array([n], dtype = mx.uint32)
        grid_size = n  # Scalar ops: one thread per element
        outs = kernel_fn(
            inputs = [dw_mlx, e_mlx, g_mlx, n_arr],
            output_shapes = [(n,), (n,), (n,)],
            output_dtypes = [mx.float16, mx.float16, mx.float16],
            grid = (grid_size, 1, 1),
            threadgroup = (min(256, grid_size), 1, 1),
        )
        h = mlx_to_torch(outs[0]).view(*shape)
        df = mlx_to_torch(outs[1]).view(*shape)
        de = mlx_to_torch(outs[2]).view(*shape)
        return h, df, de


def metal_geglu_exact_forward(e, g):
    return _metal_geglu_forward(e, g, _get_exact_forward())


def metal_geglu_exact_backward(dw, e, g):
    return _metal_geglu_backward(dw, e, g, _get_exact_backward())


def metal_geglu_approx_forward(e, g):
    return _metal_geglu_forward(e, g, _get_approx_forward())


def metal_geglu_approx_backward(dw, e, g):
    return _metal_geglu_backward(dw, e, g, _get_approx_backward())
