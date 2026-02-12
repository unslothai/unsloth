# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""
Metal-accelerated SwiGLU kernels for Apple Silicon.

Uses MLX's mx.fast.metal_kernel for maximum performance with inline shader bodies.
v9: half4 SIMD vectorization (128-bit aligned access, 4 elements/thread).
"""

from __future__ import annotations
from functools import lru_cache
from typing import TYPE_CHECKING, Tuple, Optional
import torch

__all__ = [
    "metal_swiglu_forward",
    "metal_swiglu_backward",
    "mlx_swiglu_forward",
    "mlx_swiglu_backward",
    "mlx_swiglu_forward_compiled",
    "mlx_swiglu_backward_compiled",
    "is_metal_swiglu_available",
]


from unsloth.kernels.mlx.bridge import torch_to_mlx, mlx_to_torch, mlx_context


_METAL_SWIGLU_AVAILABLE: Optional[bool] = None


def is_metal_swiglu_available() -> bool:
    """Check if Metal SwiGLU kernels are available."""
    global _METAL_SWIGLU_AVAILABLE
    if _METAL_SWIGLU_AVAILABLE is not None:
        return _METAL_SWIGLU_AVAILABLE

    try:
        import platform

        if platform.system() != "Darwin":
            _METAL_SWIGLU_AVAILABLE = False
            return False

        import mlx.core as mx

        if not hasattr(mx, "fast") or not hasattr(mx.fast, "metal_kernel"):
            _METAL_SWIGLU_AVAILABLE = False
            return False

        _METAL_SWIGLU_AVAILABLE = True
        return True
    except Exception:
        _METAL_SWIGLU_AVAILABLE = False
        return False


# Optimized Metal Shaders — half4 SIMD vectorized (128-bit access)
# -----------------------------------------------------------------------------
#
# Strategy: Each thread processes a half4 (4 x fp16 = 64 bits read, but Metal
# loads are 128-bit aligned when using float4 intermediates).  Threads beyond
# n/4 handle the scalar tail (0-3 leftover elements).
#
# The grid is dispatched as (n4 + tail, 1, 1) where n4 = n >> 2.

SWIGLU_FORWARD_BODY = """
    uint gid = thread_position_in_grid.x;
    uint n = n_ptr[0];
    uint n4 = n >> 2;

    // ---- Vectorized path: 4 elements per thread via half4 ----
    if (gid < n4) {
        const device half4* e4 = (const device half4*)e;
        const device half4* g4 = (const device half4*)g;
        device half4* h4 = (device half4*)h;

        float4 ev = float4(e4[gid]);
        float4 gv = float4(g4[gid]);

        // SiLU(x) = x * sigmoid(x)
        float4 sig = 1.0f / (1.0f + exp(-ev));
        h4[gid] = half4(ev * sig * gv);
        return;
    }

    // ---- Scalar tail: remaining 0-3 elements ----
    uint tail_idx = (n4 << 2) + (gid - n4);
    if (tail_idx >= n) return;
    float ev = float(e[tail_idx]);
    float gv = float(g[tail_idx]);
    float sig = 1.0f / (1.0f + exp(-ev));
    h[tail_idx] = half(ev * sig * gv);
"""

SWIGLU_BACKWARD_BODY = """
    uint gid = thread_position_in_grid.x;
    uint n = n_ptr[0];
    uint n4 = n >> 2;

    // ---- Vectorized path ----
    if (gid < n4) {
        const device half4* dw4 = (const device half4*)dw_in;
        const device half4* e4  = (const device half4*)e_in;
        const device half4* g4  = (const device half4*)g_in;
        device half4* h4  = (device half4*)h_out;
        device half4* df4 = (device half4*)df_out;
        device half4* de4 = (device half4*)de_out;

        float4 dwv = float4(dw4[gid]);
        float4 ev  = float4(e4[gid]);
        float4 gv  = float4(g4[gid]);

        // Branchless stable SiLU derivative via select()
        float4 se  = 1.0f / (1.0f + exp(-ev));
        float4 nse = 1.0f / (1.0f + exp(ev));
        float4 df_de_normal = se * fma(ev, nse, 1.0f);

        // Clamp to {0, 1} for extreme values (|x| > 20)
        float4 df_de = select(df_de_normal, float4(1.0f), ev > 20.0f);
        df_de = select(df_de, float4(0.0f), ev < -20.0f);

        // SiLU forward value (branchless)
        float4 f = ev * se;
        f = select(f, ev, ev > 20.0f);
        f = select(f, float4(0.0f), ev < -20.0f);

        h4[gid]  = half4(f * gv);
        df4[gid] = half4(dwv * f);
        de4[gid] = half4(dwv * gv * df_de);
        return;
    }

    // ---- Scalar tail ----
    uint tail_idx = (n4 << 2) + (gid - n4);
    if (tail_idx >= n) return;

    float dwv = float(dw_in[tail_idx]);
    float ev  = float(e_in[tail_idx]);
    float gv  = float(g_in[tail_idx]);

    float df_de;
    if (ev > 20.0f) {
        df_de = 1.0f;
    } else if (ev < -20.0f) {
        df_de = 0.0f;
    } else {
        float se  = 1.0f / (1.0f + exp(-ev));
        float nse = 1.0f / (1.0f + exp(ev));
        df_de = se * fma(ev, nse, 1.0f);
    }

    float f = (ev > 20.0f) ? ev : ((ev < -20.0f) ? 0.0f : (ev / (1.0f + exp(-ev))));

    h_out[tail_idx]  = half(f * gv);
    df_out[tail_idx] = half(dwv * f);
    de_out[tail_idx] = half(dwv * gv * df_de);
"""


@lru_cache(maxsize=1)
def _get_forward_kernel():
    import mlx.core as mx

    return mx.fast.metal_kernel(
        name="swiglu_forward_v9",
        input_names=["e", "g", "n_ptr"],
        output_names=["h"],
        source=SWIGLU_FORWARD_BODY,
    )


@lru_cache(maxsize=1)
def _get_backward_kernel():
    import mlx.core as mx

    return mx.fast.metal_kernel(
        name="swiglu_backward_v9",
        input_names=["dw_in", "e_in", "g_in", "n_ptr"],
        output_names=["h_out", "df_out", "de_out"],
        source=SWIGLU_BACKWARD_BODY,
    )


# =============================================================================
# Pure MLX wrappers (recommended for performance)
# =============================================================================


def mlx_swiglu_forward(e_mlx, g_mlx):
    """Fused SwiGLU forward: silu(e) * g using vectorized Metal kernel."""
    import mlx.core as mx

    shape = e_mlx.shape
    n = e_mlx.size
    e_flat = e_mlx.flatten()
    g_flat = g_mlx.flatten()
    n_arr = mx.array([n], dtype=mx.uint32)

    # Vectorized grid: n/4 threads for half4, plus up to 3 scalar tail threads
    n4 = n >> 2
    tail = n - (n4 << 2)
    grid_size = n4 + tail

    kernel = _get_forward_kernel()
    out = kernel(
        inputs=[e_flat, g_flat, n_arr],
        output_shapes=[(n,)],
        output_dtypes=[mx.float16],
        grid=(grid_size, 1, 1),
        threadgroup=(min(256, grid_size), 1, 1),
    )
    h = out[0].reshape(shape)
    return h


def mlx_swiglu_backward(dw_mlx, e_mlx, g_mlx):
    """Fused SwiGLU backward using vectorized Metal kernel."""
    import mlx.core as mx

    shape = e_mlx.shape
    n = e_mlx.size
    # Clip gradients to avoid inf propagation
    dw_mlx = mx.clip(dw_mlx.astype(mx.float32), -65504.0, 65504.0).astype(dw_mlx.dtype)

    dw_flat = dw_mlx.flatten()
    e_flat = e_mlx.flatten()
    g_flat = g_mlx.flatten()
    n_arr = mx.array([n], dtype=mx.uint32)

    # Vectorized grid: n/4 threads for half4, plus up to 3 scalar tail threads
    n4 = n >> 2
    tail = n - (n4 << 2)
    grid_size = n4 + tail

    kernel = _get_backward_kernel()
    outputs = kernel(
        inputs=[dw_flat, e_flat, g_flat, n_arr],
        output_shapes=[(n,), (n,), (n,)],
        output_dtypes=[mx.float16, mx.float16, mx.float16],
        grid=(grid_size, 1, 1),
        threadgroup=(min(256, grid_size), 1, 1),
    )
    h = outputs[0].reshape(shape)
    df = outputs[1].reshape(shape)
    de = outputs[2].reshape(shape)
    return h, df, de


# =============================================================================
# Compiled MLX wrappers (graph-level fusion via mx.compile)
# =============================================================================

_compiled_swiglu_forward = None
_compiled_swiglu_backward = None


def mlx_swiglu_forward_compiled(e_mlx, g_mlx):
    """SwiGLU forward via mx.compile: lets MLX fuse sigmoid + mul."""
    import mlx.core as mx

    global _compiled_swiglu_forward
    if _compiled_swiglu_forward is None:
        def _fwd(e, g):
            return mx.sigmoid(e) * e * g
        _compiled_swiglu_forward = mx.compile(_fwd)

    return _compiled_swiglu_forward(e_mlx, g_mlx)


def mlx_swiglu_backward_compiled(dw_mlx, e_mlx, g_mlx):
    """SwiGLU backward via mx.compile: compiled VJP."""
    import mlx.core as mx

    global _compiled_swiglu_backward
    if _compiled_swiglu_backward is None:
        def _fwd(e, g):
            return mx.sigmoid(e) * e * g

        def _bwd(dw, e, g):
            f = mx.compile(_fwd)
            h = f(e, g)
            se = mx.sigmoid(e)
            nse = 1.0 - se
            df_de = se * (1.0 + e * nse)
            df = dw * h / (e * se + 1e-12)  # avoid div by zero
            # Simpler: recompute properly
            de = dw * g * df_de
            dg = dw * se * e
            return h, dg, de

        _compiled_swiglu_backward = mx.compile(_bwd)

    return _compiled_swiglu_backward(dw_mlx, e_mlx, g_mlx)


# =============================================================================
# PyTorch wrappers (for integration with main Unsloth dispatch)
# =============================================================================


def metal_swiglu_forward(e: "torch.Tensor", g: "torch.Tensor") -> "torch.Tensor":
    """Fused SwiGLU forward using Metal kernel (PyTorch interface)."""
    import torch
    if torch.is_grad_enabled():
        return Metal_SwiGLU.apply(e, g)
    
    import mlx.core as mx
    shape = e.shape
    with mlx_context():
        # CHAINING: Check for cached MLX tensors on inputs
        e_mlx = getattr(e, "_mlx_cache", None)
        if e_mlx is None: e_mlx = torch_to_mlx(e)

        g_mlx = getattr(g, "_mlx_cache", None)
        if g_mlx is None: g_mlx = torch_to_mlx(g)

        out_mlx = mlx_swiglu_forward(e_mlx, g_mlx)

        # CHAINING: Attach MLX output to PyTorch tensor for next layer
        out_torch = mlx_to_torch(out_mlx).view(*shape)
        out_torch._mlx_cache = out_mlx
        return out_torch


def metal_swiglu_backward(
    dw: "torch.Tensor", e: "torch.Tensor", g: "torch.Tensor"
) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
    """Fused SwiGLU backward using Metal kernel (PyTorch interface)."""
    import mlx.core as mx

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

        h_mlx, df_mlx, de_mlx = mlx_swiglu_backward(dw_mlx, e_mlx, g_mlx)

        h = mlx_to_torch(h_mlx).view(*shape)
        df = mlx_to_torch(df_mlx).view(*shape)
        de = mlx_to_torch(de_mlx).view(*shape)

        # CHAINING: Attach outputs for next backward step
        h._mlx_cache = h_mlx
        df._mlx_cache = df_mlx
        de._mlx_cache = de_mlx

        return h, df, de


class Metal_SwiGLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, e, g):
        with mlx_context():
            e_mlx = getattr(e, "_mlx_cache", None)
            if e_mlx is None: e_mlx = torch_to_mlx(e)
            
            g_mlx = getattr(g, "_mlx_cache", None)
            if g_mlx is None: g_mlx = torch_to_mlx(g)
            
            h_mlx = mlx_swiglu_forward(e_mlx, g_mlx)
            h = mlx_to_torch(h_mlx)
            
            ctx.save_for_backward(e, g)
            
            # CHAINING: Attach MLX output
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
            
            h_mlx, df_mlx, de_mlx = mlx_swiglu_backward(dw_mlx, e_mlx, g_mlx)
            
            df = mlx_to_torch(df_mlx)
            de = mlx_to_torch(de_mlx)
            
            # CHAINING: Attach MLX outputs
            df._mlx_cache = df_mlx
            de._mlx_cache = de_mlx
            
            return de, df
