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
import torch

__all__ = [
    "metal_geglu_exact_forward",
    "metal_geglu_exact_backward",
    "metal_geglu_approx_forward",
    "metal_geglu_approx_backward",
    "mlx_geglu_exact_forward",
    "mlx_geglu_exact_backward",
    "mlx_geglu_approx_forward",
    "mlx_geglu_approx_backward",
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
    
    float f;
    float df_de;
    if (ev > 10.0f) {
        f = ev;
        df_de = 1.0f;
    } else if (ev < -10.0f) {
        f = 0.0f;
        df_de = 0.0f;
    } else {
        // Compute erf approximation inline
        float x = ev * 0.70710678118f;
        float a = abs(x);
        float t = 1.0f / fma(a, 0.3275911f, 1.0f);
        float erf_val = 1.0f - (((((1.061405429f * t - 1.453152027f) * t + 1.421413741f) * t - 0.284496736f) * t + 0.254829592f) * t * exp(-a * a));
        erf_val = x < 0.0f ? -erf_val : erf_val;
        float f_partial = 0.5f * (erf_val + 1.0f);
        f = f_partial * ev;
        df_de = f_partial + 0.3989422804f * ev * exp(-0.5f * ev * ev);
    }
    
    h_out[gid] = half(f * gv);
    df_out[gid] = half(dwv * f);
    de_out[gid] = half(dwv * gv * df_de);
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
    
    float df_de;
    float f;
    if (ev > 10.0f) {
        df_de = 1.0f;
        f = ev;
    } else if (ev < -10.0f) {
        df_de = 0.0f;
        f = 0.0f;
    } else {
        float s = 0.7978845608f;
        float inner = s * ev * fma(ev, ev * 0.044715f, 1.0f);
        float t = tanh(inner);
        float T = 1.0f + t;
        float T2 = 0.5f * T;
        f = T2 * ev;
        float sech2 = 1.0f - t * t;
        float Q2 = 0.5f * ev * sech2 * s * fma(ev, ev * 0.134145f, 1.0f);
        df_de = T2 + Q2;
    }
    
    h_out[gid] = half(f * gv);
    df_out[gid] = half(dwv * f);
    de_out[gid] = half(dwv * gv * df_de);
"""


@lru_cache(maxsize=1)
def _get_exact_forward():
    import mlx.core as mx

    return mx.fast.metal_kernel(
        name="geglu_exact_forward_v8",
        input_names=["e", "g", "n_ptr"],
        output_names=["h"],
        source=GEGLU_EXACT_FORWARD_BODY,
    )


@lru_cache(maxsize=1)
def _get_exact_backward():
    import mlx.core as mx

    return mx.fast.metal_kernel(
        name="geglu_exact_backward_v8",
        input_names=["dw_in", "e_in", "g_in", "n_ptr"],
        output_names=["h_out", "df_out", "de_out"],
        source=GEGLU_EXACT_BACKWARD_BODY,
    )


@lru_cache(maxsize=1)
def _get_approx_forward():
    import mlx.core as mx

    return mx.fast.metal_kernel(
        name="geglu_approx_forward_v8",
        input_names=["e", "g", "n_ptr"],
        output_names=["h"],
        source=GEGLU_APPROX_FORWARD_BODY,
    )


@lru_cache(maxsize=1)
def _get_approx_backward():
    import mlx.core as mx

    return mx.fast.metal_kernel(
        name="geglu_approx_backward_v8",
        input_names=["dw_in", "e_in", "g_in", "n_ptr"],
        output_names=["h_out", "df_out", "de_out"],
        source=GEGLU_APPROX_BACKWARD_BODY,
    )


# =============================================================================
# Pure MLX wrappers (recommended for performance)
# =============================================================================


def _mlx_geglu_forward(e_mlx, g_mlx, kernel_fn):
    """Pure MLX GEGLU forward."""
    import mlx.core as mx

    shape = e_mlx.shape
    n = e_mlx.size
    e_flat = e_mlx.flatten()
    g_flat = g_mlx.flatten()
    n_arr = mx.array([n], dtype=mx.uint32)
    grid_size = n
    out = kernel_fn(
        inputs=[e_flat, g_flat, n_arr],
        output_shapes=[(n,)],
        output_dtypes=[mx.float16],
        grid=(grid_size, 1, 1),
        threadgroup=(min(256, grid_size), 1, 1),
    )
    h = out[0].reshape(shape)
    return h


def _mlx_geglu_backward(dw_mlx, e_mlx, g_mlx, kernel_fn):
    """Pure MLX GEGLU backward."""
    import mlx.core as mx

    shape = e_mlx.shape
    n = e_mlx.size
    # Clip gradients to avoid inf propagation
    dw_mlx = mx.clip(dw_mlx.astype(mx.float32), -65504.0, 65504.0).astype(dw_mlx.dtype)
    
    dw_flat = dw_mlx.flatten()
    e_flat = e_mlx.flatten()
    g_flat = g_mlx.flatten()
    n_arr = mx.array([n], dtype=mx.uint32)
    grid_size = n
    outs = kernel_fn(
        inputs=[dw_flat, e_flat, g_flat, n_arr],
        output_shapes=[(n,), (n,), (n,)],
        output_dtypes=[mx.float16, mx.float16, mx.float16],
        grid=(grid_size, 1, 1),
        threadgroup=(min(256, grid_size), 1, 1),
    )
    h = outs[0].reshape(shape)
    df = outs[1].reshape(shape)
    de = outs[2].reshape(shape)
    return h, df, de


def mlx_geglu_exact_forward(e_mlx, g_mlx):
    """Fused GEGLU exact forward: GELU(e) * g using Metal kernel."""
    return _mlx_geglu_forward(e_mlx, g_mlx, _get_exact_forward())


def mlx_geglu_exact_backward(dw_mlx, e_mlx, g_mlx):
    """Fused GEGLU exact backward using Metal kernel."""
    return _mlx_geglu_backward(dw_mlx, e_mlx, g_mlx, _get_exact_backward())


def mlx_geglu_approx_forward(e_mlx, g_mlx):
    """Fused GEGLU approx forward: tanh-GELU(e) * g using Metal kernel."""
    return _mlx_geglu_forward(e_mlx, g_mlx, _get_approx_forward())


# =============================================================================
# PyTorch wrappers (for integration with main Unsloth dispatch)
# =============================================================================


def _metal_geglu_forward(
    e: "torch.Tensor", g: "torch.Tensor", mlx_fn
) -> "torch.Tensor":
    shape = e.shape
    with mlx_context():
        # CHAINING: Check inputs for cached MLX tensors
        e_mlx = getattr(e, "_mlx_cache", None)
        if e_mlx is None:
            e_mlx = torch_to_mlx(e)

        g_mlx = getattr(g, "_mlx_cache", None)
        if g_mlx is None:
            g_mlx = torch_to_mlx(g)

        out_mlx = mlx_fn(e_mlx, g_mlx)

        # CHAINING: Attach MLX output
        out_torch = mlx_to_torch(out_mlx).view(*shape)
        out_torch._mlx_cache = out_mlx
        return out_torch


def _metal_geglu_backward(
    dw: "torch.Tensor", e: "torch.Tensor", g: "torch.Tensor", mlx_fn
):
    shape = e.shape
    with mlx_context():
        # CHAINING: Check inputs for cached MLX tensors
        dw_mlx = getattr(dw, "_mlx_cache", None)
        if dw_mlx is None:
            dw_mlx = torch_to_mlx(dw)

        e_mlx = getattr(e, "_mlx_cache", None)
        if e_mlx is None:
            e_mlx = torch_to_mlx(e)

        g_mlx = getattr(g, "_mlx_cache", None)
        if g_mlx is None:
            g_mlx = torch_to_mlx(g)

        h_mlx, df_mlx, de_mlx = mlx_fn(dw_mlx, e_mlx, g_mlx)

        h = mlx_to_torch(h_mlx).view(*shape)
        df = mlx_to_torch(df_mlx).view(*shape)
        de = mlx_to_torch(de_mlx).view(*shape)

        # CHAINING: Attach outputs
        h._mlx_cache = h_mlx
        df._mlx_cache = df_mlx
        de._mlx_cache = de_mlx

        return h, df, de


class Metal_GEGLU_Exact(torch.autograd.Function):
    @staticmethod
    def forward(ctx, e, g):
        with mlx_context():
            e_mlx = getattr(e, "_mlx_cache", None)
            if e_mlx is None: e_mlx = torch_to_mlx(e)
            
            g_mlx = getattr(g, "_mlx_cache", None)
            if g_mlx is None: g_mlx = torch_to_mlx(g)
            
            h_mlx = mlx_geglu_exact_forward(e_mlx, g_mlx)
            h = mlx_to_torch(h_mlx)
            
            ctx.save_for_backward(e, g)
            h._mlx_cache = h_mlx
            return h

    @staticmethod
    def backward(ctx, dw):
        e, g = ctx.saved_tensors
        with mlx_context():
            dw_mlx = getattr(dw, "_mlx_cache", None)
            if dw_mlx is None: dw_mlx = torch_to_mlx(dw)
            
            e_mlx = getattr(e, "_mlx_cache", None)
            if e_mlx is None: e_mlx = torch_to_mlx(e)
            
            g_mlx = getattr(g, "_mlx_cache", None)
            if g_mlx is None: g_mlx = torch_to_mlx(g)
            
            h_mlx, df_mlx, de_mlx = mlx_geglu_exact_backward(dw_mlx, e_mlx, g_mlx)
            
            df = mlx_to_torch(df_mlx)
            de = mlx_to_torch(de_mlx)
            
            df._mlx_cache = df_mlx
            de._mlx_cache = de_mlx
            return de, df


class Metal_GEGLU_Approx(torch.autograd.Function):
    @staticmethod
    def forward(ctx, e, g):
        with mlx_context():
            e_mlx = getattr(e, "_mlx_cache", None)
            if e_mlx is None: e_mlx = torch_to_mlx(e)
            
            g_mlx = getattr(g, "_mlx_cache", None)
            if g_mlx is None: g_mlx = torch_to_mlx(g)
            
            h_mlx = mlx_geglu_approx_forward(e_mlx, g_mlx)
            h = mlx_to_torch(h_mlx)
            
            ctx.save_for_backward(e, g)
            h._mlx_cache = h_mlx
            return h

    @staticmethod
    def backward(ctx, dw):
        e, g = ctx.saved_tensors
        with mlx_context():
            dw_mlx = getattr(dw, "_mlx_cache", None)
            if dw_mlx is None: dw_mlx = torch_to_mlx(dw)
            
            e_mlx = getattr(e, "_mlx_cache", None)
            if e_mlx is None: e_mlx = torch_to_mlx(e)
            
            g_mlx = getattr(g, "_mlx_cache", None)
            if g_mlx is None: g_mlx = torch_to_mlx(g)
            
            h_mlx, df_mlx, de_mlx = mlx_geglu_approx_backward(dw_mlx, e_mlx, g_mlx)
            
            df = mlx_to_torch(df_mlx)
            de = mlx_to_torch(de_mlx)
            
            df._mlx_cache = df_mlx
            de._mlx_cache = de_mlx
            return de, df


def metal_geglu_exact_forward(e, g):
    if torch.is_grad_enabled():
        return Metal_GEGLU_Exact.apply(e, g)
    return _metal_geglu_forward(e, g, mlx_geglu_exact_forward)


def metal_geglu_exact_backward(dw, e, g):
    return _metal_geglu_backward(dw, e, g, mlx_geglu_exact_backward)


def metal_geglu_approx_forward(e, g):
    if torch.is_grad_enabled():
        return Metal_GEGLU_Approx.apply(e, g)
    return _metal_geglu_forward(e, g, mlx_geglu_approx_forward)


def metal_geglu_approx_backward(dw, e, g):
    return _metal_geglu_backward(dw, e, g, mlx_geglu_approx_backward)
