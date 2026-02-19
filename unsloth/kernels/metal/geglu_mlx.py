# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""
Pure MLX/Metal GEGLU kernels - no PyTorch dependencies.

Uses MLX's mx.fast.metal_kernel for maximum performance.
Supports both Exact and Approximate (Tanh) variants.
"""

from __future__ import annotations
from functools import lru_cache
from typing import TYPE_CHECKING, Tuple, Optional

__all__ = [
    "mlx_geglu_exact_forward",
    "mlx_geglu_exact_backward", 
    "mlx_geglu_approx_forward",
    "mlx_geglu_approx_backward",
    "is_mlx_geglu_available",
]


def is_mlx_geglu_available() -> bool:
    """Check if MLX GEGLU kernels are available."""
    try:
        import platform

        if platform.system() != "Darwin":
            return False

        import mlx.core as mx

        if not hasattr(mx, "fast") or not hasattr(mx.fast, "metal_kernel"):
            return False

        return True
    except Exception:
        return False


# -----------------------------------------------------------------------------
# Optimized Metal Shaders
# -----------------------------------------------------------------------------

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


# -----------------------------------------------------------------------------
# Cached Metal Kernels
# -----------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _get_exact_forward():
    """Get cached Metal kernel for exact GEGLU forward."""
    import mlx.core as mx

    return mx.fast.metal_kernel(
        name="geglu_exact_forward_mlx",
        input_names=["e", "g", "n_ptr"],
        output_names=["h"],
        source=GEGLU_EXACT_FORWARD_BODY,
    )


@lru_cache(maxsize=1)
def _get_exact_backward():
    """Get cached Metal kernel for exact GEGLU backward."""
    import mlx.core as mx

    return mx.fast.metal_kernel(
        name="geglu_exact_backward_mlx",
        input_names=["dw_in", "e_in", "g_in", "n_ptr"],
        output_names=["h_out", "df_out", "de_out"],
        source=GEGLU_EXACT_BACKWARD_BODY,
    )


@lru_cache(maxsize=1)
def _get_approx_forward():
    """Get cached Metal kernel for approximate GEGLU forward."""
    import mlx.core as mx

    return mx.fast.metal_kernel(
        name="geglu_approx_forward_mlx",
        input_names=["e", "g", "n_ptr"],
        output_names=["h"],
        source=GEGLU_APPROX_FORWARD_BODY,
    )


@lru_cache(maxsize=1)
def _get_approx_backward():
    """Get cached Metal kernel for approximate GEGLU backward."""
    import mlx.core as mx

    return mx.fast.metal_kernel(
        name="geglu_approx_backward_mlx",
        input_names=["dw_in", "e_in", "g_in", "n_ptr"],
        output_names=["h_out", "df_out", "de_out"],
        source=GEGLU_APPROX_BACKWARD_BODY,
    )


# -----------------------------------------------------------------------------
# Pure MLX API
# -----------------------------------------------------------------------------

def mlx_geglu_exact_forward(e: "mx.array", g: "mx.array") -> "mx.array":
    """
    Fused GEGLU exact forward: GELU(e) * g using Metal kernel.
    
    Args:
        e: Input tensor (e.g., gate projection)
        g: Input tensor (e.g., value projection)
        
    Returns:
        h: Output tensor = GELU(e) * g
    """
    import mlx.core as mx
    
    if not is_mlx_geglu_available():
        # Fallback to pure MLX implementation
        gelu_e = 0.5 * e * (1 + mx.erf(e / mx.sqrt(2)))
        return gelu_e * g
    
    shape = e.shape
    n = e.size
    e_flat = e.flatten()
    g_flat = g.flatten()
    n_arr = mx.array([n], dtype=mx.uint32)
    grid_size = n
    
    kernel = _get_exact_forward()
    out = kernel(
        inputs=[e_flat, g_flat, n_arr],
        output_shapes=[(n,)],
        output_dtypes=[mx.float16],
        grid=(grid_size, 1, 1),
        threadgroup=(min(256, grid_size), 1, 1),
    )
    h = out[0].reshape(shape)
    return h


def mlx_geglu_exact_backward(
    dw: "mx.array", e: "mx.array", g: "mx.array"
) -> Tuple["mx.array", "mx.array", "mx.array"]:
    """
    Fused GEGLU exact backward using Metal kernel.
    
    Args:
        dw: Gradient from upstream
        e: Forward input e
        g: Forward input g
        
    Returns:
        h: Intermediate output (for potential reuse)
        df: Gradient w.r.t. f (g input)
        de: Gradient w.r.t. e
    """
    import mlx.core as mx
    
    if not is_mlx_geglu_available():
        # Fallback to pure MLX implementation
        # GELU forward: h = 0.5 * e * (1 + erf(e / sqrt(2)))
        sqrt2 = mx.sqrt(2)
        gelu_e = 0.5 * e * (1 + mx.erf(e / sqrt2))
        
        # GELU derivative
        # d/dx[GELU(x)] = 0.5 * (1 + erf(x/sqrt(2))) + x * exp(-x^2/2) / sqrt(2*pi)
        cdf = 0.5 * (1 + mx.erf(e / sqrt2))
        pdf = mx.exp(-e * e / 2) / mx.sqrt(2 * mx.pi)
        d_gelu = cdf + e * pdf
        
        h = gelu_e * g
        df = dw * gelu_e
        de = dw * g * d_gelu
        return h, df, de
    
    shape = e.shape
    n = e.size
    
    # Clip gradients to avoid inf propagation
    dw = mx.clip(dw.astype(mx.float32), -65504.0, 65504.0).astype(dw.dtype)
    
    dw_flat = dw.flatten()
    e_flat = e.flatten()
    g_flat = g.flatten()
    n_arr = mx.array([n], dtype=mx.uint32)
    grid_size = n
    
    kernel = _get_exact_backward()
    outs = kernel(
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


def mlx_geglu_approx_forward(e: "mx.array", g: "mx.array") -> "mx.array":
    """
    Fused GEGLU approx forward: tanh-GELU(e) * g using Metal kernel.
    
    Args:
        e: Input tensor (e.g., gate projection)
        g: Input tensor (e.g., value projection)
        
    Returns:
        h: Output tensor = tanh-GELU(e) * g
    """
    import mlx.core as mx
    
    if not is_mlx_geglu_available():
        # Fallback to pure MLX implementation
        sqrt_2_over_pi = mx.sqrt(2 / mx.pi)
        inner = sqrt_2_over_pi * e * (1 + 0.044715 * e * e)
        tanh_gelu = 0.5 * e * (1 + mx.tanh(inner))
        return tanh_gelu * g
    
    shape = e.shape
    n = e.size
    e_flat = e.flatten()
    g_flat = g.flatten()
    n_arr = mx.array([n], dtype=mx.uint32)
    grid_size = n
    
    kernel = _get_approx_forward()
    out = kernel(
        inputs=[e_flat, g_flat, n_arr],
        output_shapes=[(n,)],
        output_dtypes=[mx.float16],
        grid=(grid_size, 1, 1),
        threadgroup=(min(256, grid_size), 1, 1),
    )
    h = out[0].reshape(shape)
    return h


def mlx_geglu_approx_backward(
    dw: "mx.array", e: "mx.array", g: "mx.array"
) -> Tuple["mx.array", "mx.array", "mx.array"]:
    """
    Fused GEGLU approx backward using Metal kernel.
    
    Args:
        dw: Gradient from upstream
        e: Forward input e
        g: Forward input g
        
    Returns:
        h: Intermediate output (for potential reuse)
        df: Gradient w.r.t. f (g input)
        de: Gradient w.r.t. e
    """
    import mlx.core as mx
    
    if not is_mlx_geglu_available():
        # Fallback to pure MLX implementation
        sqrt_2_over_pi = mx.sqrt(2 / mx.pi)
        inner = sqrt_2_over_pi * e * (1 + 0.044715 * e * e)
        tanh_inner = mx.tanh(inner)
        tanh_gelu = 0.5 * e * (1 + tanh_inner)
        
        # Derivative of tanh-GELU
        sech2 = 1 - tanh_inner * tanh_inner
        d_inner = sqrt_2_over_pi * (1 + 3 * 0.044715 * e * e)
        d_tanh_gelu = 0.5 * (1 + tanh_inner) + 0.5 * e * sech2 * d_inner
        
        h = tanh_gelu * g
        df = dw * tanh_gelu
        de = dw * g * d_tanh_gelu
        return h, df, de
    
    shape = e.shape
    n = e.size
    
    # Clip gradients to avoid inf propagation
    dw = mx.clip(dw.astype(mx.float32), -65504.0, 65504.0).astype(dw.dtype)
    
    dw_flat = dw.flatten()
    e_flat = e.flatten()
    g_flat = g.flatten()
    n_arr = mx.array([n], dtype=mx.uint32)
    grid_size = n
    
    kernel = _get_approx_backward()
    outs = kernel(
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
