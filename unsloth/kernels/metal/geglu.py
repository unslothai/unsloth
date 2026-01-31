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
# Optimized Metal Shaders (Vectorized half8)
# -----------------------------------------------------------------------------

ERF_DEF = """
float erf_approx(float x) {
    float a = abs(x);
    float t = 1.0f / fma(a, 0.3275911f, 1.0f);
    float y = 1.0f - (((((1.061405429f * t - 1.453152027f) * t + 1.421413741f) * t - 0.284496736f) * t + 0.254829592f) * t * exp(-a * a));
    return select(y, -y, x < 0.0f);
}

float8 erf_approx_v8(float8 x) {
    float8 xa = abs(x);
    float8 t = 1.0f / fma(xa, 0.3275911f, 1.0f);
    float8 y = 1.0f - (((((1.061405429f * t - 1.453152027f) * t + 1.421413741f) * t - 0.284496736f) * t + 0.254829592f) * t * exp(-xa * xa));
    return select(y, -y, x < 0.0f);
}
"""

GEGLU_EXACT_FORWARD_BODY = (
    ERF_DEF
    + """
    uint gid = thread_position_in_grid.x;
    uint n = n_ptr[0];
    uint start = gid * 8;
    if (start >= n) return;
    
    if (start + 7 < n) {
        device const half8* e_vec = (device const half8*)e;
        device const half8* g_vec = (device const half8*)g;
        device half8* h_vec = (device half8*)h;
        float8 ev = float8(e_vec[gid]);
        float8 gv = float8(g_vec[gid]);
        float8 gelu_ev = 0.5f * ev * (1.0f + erf_approx_v8(ev * 0.70710678118f));
        h_vec[gid] = half8(gelu_ev * gv);
    } else {
        for (uint i = start; i < n; i++) {
            float ev = float(e[i]);
            float gv = float(g[i]);
            float gelu_e = 0.5f * ev * (1.0f + erf_approx(ev * 0.70710678118f));
            h[i] = half(gelu_e * gv);
        }
    }
"""
)

GEGLU_EXACT_BACKWARD_BODY = (
    ERF_DEF
    + """
    uint gid = thread_position_in_grid.x;
    uint n = n_ptr[0];
    uint start = gid * 8;
    if (start >= n) return;
    
    if (start + 7 < n) {
        device const half8* dw_vec = (device const half8*)dw_in;
        device const half8* e_vec = (device const half8*)e_in;
        device const half8* g_vec = (device const half8*)g_in;
        
        float8 dwv = float8(dw_vec[gid]);
        float8 ev = float8(e_vec[gid]);
        float8 gv = float8(g_vec[gid]);
        
        float8 inv_sqrt2 = 0.70710678118f;
        float8 t_const = 0.3989422804f; 
        
        float8 erf_val = erf_approx_v8(ev * inv_sqrt2);
        float8 f_partial = 0.5f * (erf_val + 1.0f);
        float8 f = f_partial * ev;
        
        ((device half8*)h_out)[gid] = half8(f * gv);
        ((device half8*)df_out)[gid] = half8(dwv * f);
        float8 dgv = dwv * gv;
        float8 df_de = f_partial + t_const * ev * exp(-0.5f * ev * ev);
        ((device half8*)de_out)[gid] = half8(dgv * df_de);
    } else {
        for (uint i = start; i < n; i++) {
            float dwv = float(dw_in[i]);
            float ev = float(e_in[i]);
            float gv = float(g[i]);
            float f_partial = 0.5f * (erf_approx(ev * 0.70710678118f) + 1.0f);
            float f = f_partial * ev;
            h_out[i] = half(f * gv);
            df_out[i] = half(dwv * f);
            float df_de = f_partial + 0.3989422804f * ev * exp(-0.5f * ev * ev);
            de_out[i] = half(dwv * gv * df_de);
        }
    }
"""
)

GEGLU_APPROX_FORWARD_BODY = """
    uint gid = thread_position_in_grid.x;
    uint n = n_ptr[0];
    uint start = gid * 8;
    if (start >= n) return;
    
    if (start + 7 < n) {
        device const half8* e_vec = (device const half8*)e;
        device const half8* g_vec = (device const half8*)g;
        device half8* h_vec = (device half8*)h;
        float8 ev = float8(e_vec[gid]);
        float8 gv = float8(g_vec[gid]);
        float8 s = 0.7978845608f;
        float8 inner = s * ev * fma(0.044715f * ev, ev, 1.0f);
        float8 gelu_ev = 0.5f * ev * (1.0f + tanh(inner));
        h_vec[gid] = half8(gelu_ev * gv);
    } else {
        for (uint i = start; i < n; i++) {
            float ev = float(e[i]);
            float gv = float(g[i]);
            float inner = 0.7978845608f * ev * (1.0f + 0.044715f * ev * ev);
            float gelu_e = 0.5f * ev * (1.0f + tanh(inner));
            h[i] = half(gelu_e * gv);
        }
    }
"""

GEGLU_APPROX_BACKWARD_BODY = """
    uint gid = thread_position_in_grid.x;
    uint n = n_ptr[0];
    uint start = gid * 8;
    if (start >= n) return;
    
    if (start + 7 < n) {
        device const half8* dw_vec = (device const half8*)dw_in;
        device const half8* e_vec = (device const half8*)e_in;
        device const half8* g_vec = (device const half8*)g_in;
        
        float8 dwv = float8(dw_vec[gid]);
        float8 ev = float8(e_vec[gid]);
        float8 gv = float8(g_vec[gid]);
        
        float8 s = 0.7978845608f;
        float8 a = s * ev;
        float8 b = a * 0.044715f * ev * ev;
        float8 T = 1.0f + tanh(a + b);
        float8 T2 = 0.5f * T;
        float8 f = T2 * ev;
        
        ((device half8*)h_out)[gid] = half8(f * gv);
        ((device half8*)df_out)[gid] = half8(dwv * f);
        float8 dgv = dwv * gv;
        float8 Q2 = -T2 * (T - 2.0f) * (a + 3.0f * b);
        ((device half8*)de_out)[gid] = half8(dgv * (T2 + Q2));
    } else {
        for (uint i = start; i < n; i++) {
            float dwv = float(dw_in[i]);
            float ev = float(e_in[i]);
            float gv = float(g[i]);
            float s = 0.7978845608f;
            float T = 1.0f + tanh(s * ev * (1.0f + 0.044715f * ev * ev));
            float T2 = 0.5f * T;
            float f = T2 * ev;
            h_out[i] = half(f * gv);
            df_out[i] = half(dwv * f);
            float Q2 = -T2 * (T - 2.0f) * (s * ev + 3.0f * (s * ev * 0.044715f * ev * ev));
            de_out[i] = half(dwv * gv * (T2 + Q2));
        }
    }
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
        grid_size = (n + 7) // 8
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
        grid_size = (n + 7) // 8
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
