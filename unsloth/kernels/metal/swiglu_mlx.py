# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""
Pure MLX SwiGLU kernels for Apple Silicon.

No PyTorch dependencies - works directly with mx.array for maximum performance.
Uses MLX's mx.fast.metal_kernel for custom Metal shaders.
"""

from __future__ import annotations
from functools import lru_cache
from typing import Tuple, Optional
import mlx.core as mx

__all__ = [
    "swiglu_forward",
    "swiglu_backward", 
    "SwiGLU",
    "is_available",
]


# Metal Shaders — half4 SIMD vectorized (128-bit aligned access)
SWIGLU_FORWARD_BODY = """
    uint gid = thread_position_in_grid.x;
    uint n = n_ptr[0];
    uint n4 = n >> 2;

    if (gid < n4) {
        const device half4* e4 = (const device half4*)e;
        const device half4* g4 = (const device half4*)g;
        device half4* h4 = (device half4*)h;

        float4 ev = float4(e4[gid]);
        float4 gv = float4(g4[gid]);

        float4 sig_ev = 1.0f / (1.0f + exp(-ev));
        float4 hv = ev * sig_ev * gv;

        h4[gid] = half4(hv);
    } else {
        uint idx = n4 * 4 + (gid - n4);
        if (idx < n) {
            float e_val = float(e[idx]);
            float g_val = float(g[idx]);
            float sig_e = 1.0f / (1.0f + exp(-e_val));
            h[idx] = half(e_val * sig_e * g_val);
        }
    }
"""


SWIGLU_BACKWARD_BODY = """
    uint gid = thread_position_in_grid.x;
    uint n = n_ptr[0];
    uint n4 = n >> 2;

    if (gid < n4) {
        const device half4* dw4 = (const device half4*)dw;
        const device half4* e4 = (const device half4*)e;
        const device half4* g4 = (const device half4*)g;
        device half4* h4 = (device half4*)h;
        device half4* df4 = (device half4*)df;
        device half4* de4 = (device half4*)de;

        float4 dwv = float4(dw4[gid]);
        float4 ev = float4(e4[gid]);
        float4 gv = float4(g4[gid]);

        float4 sig_ev = 1.0f / (1.0f + exp(-ev));
        float4 hv = ev * sig_ev * gv;

        float4 one_minus_sig = 1.0f - sig_ev;
        float4 df_de = sig_ev * (1.0f + ev * one_minus_sig);

        float4 dfv = dwv * hv / (ev * sig_ev + 1e-6f);
        float4 dev = dwv * gv * df_de;
        float4 dgv = dwv * ev * sig_ev;

        h4[gid] = half4(hv);
        df4[gid] = half4(dfv);
        de4[gid] = half4(dev);
    } else {
        uint idx = n4 * 4 + (gid - n4);
        if (idx < n) {
            float dw_val = float(dw[idx]);
            float e_val = float(e[idx]);
            float g_val = float(g[idx]);

            float sig_e = 1.0f / (1.0f + exp(-e_val));
            float h_val = e_val * sig_e * g_val;

            float one_minus_sig = 1.0f - sig_e;
            float df_de = sig_e * (1.0f + e_val * one_minus_sig);

            h[idx] = half(h_val);
            df[idx] = half(dw_val * h_val / (e_val * sig_e + 1e-6f));
            de[idx] = half(dw_val * g_val * df_de);
        }
    }
"""


@lru_cache(maxsize=1)
def _get_forward_kernel():
    """Get cached Metal forward kernel."""
    return mx.fast.metal_kernel(
        name="swiglu_forward_v10_mlx",
        input_names=["e", "g", "n_ptr"],
        output_names=["h"],
        source=SWIGLU_FORWARD_BODY,
    )


@lru_cache(maxsize=1)
def _get_backward_kernel():
    """Get cached Metal backward kernel."""
    return mx.fast.metal_kernel(
        name="swiglu_backward_v10_mlx",
        input_names=["dw", "e", "g", "n_ptr"],
        output_names=["h", "df", "de"],
        source=SWIGLU_BACKWARD_BODY,
    )


def is_available() -> bool:
    """Check if Metal SwiGLU kernels are available."""
    try:
        import platform
        if platform.system() != "Darwin":
            return False
        return hasattr(mx, "fast") and hasattr(mx.fast, "metal_kernel")
    except Exception:
        return False


def swiglu_forward(e: mx.array, g: mx.array) -> mx.array:
    """
    Fused SwiGLU forward: h = SiLU(e) * g
    
    Args:
        e: Gate tensor (any shape, will be flattened)
        g: Up tensor (same shape as e)
    
    Returns:
        h: Output tensor (same shape as input)
    """
    shape = e.shape
    n = e.size
    
    # Flatten for kernel
    e_flat = e.flatten()
    g_flat = g.flatten()
    n_arr = mx.array([n], dtype=mx.uint32)
    
    # Grid: n/4 threads for half4 + tail
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
    return out[0].reshape(shape)


def swiglu_backward(
    dh: mx.array, e: mx.array, g: mx.array
) -> Tuple[mx.array, mx.array, mx.array]:
    """
    Fused SwiGLU backward pass.
    
    Args:
        dh: Gradient w.r.t. output
        e: Saved gate tensor from forward
        g: Saved up tensor from forward
    
    Returns:
        (h, dg, de): Output, gradient w.r.t. g, gradient w.r.t. e
    """
    shape = e.shape
    n = e.size
    
    # Clip gradients to avoid inf
    dh = mx.clip(dh.astype(mx.float32), -65504.0, 65504.0).astype(dh.dtype)
    
    # Flatten
    dh_flat = dh.flatten()
    e_flat = e.flatten()
    g_flat = g.flatten()
    n_arr = mx.array([n], dtype=mx.uint32)
    
    # Grid
    n4 = n >> 2
    tail = n - (n4 << 2)
    grid_size = n4 + tail
    
    kernel = _get_backward_kernel()
    outputs = kernel(
        inputs=[dh_flat, e_flat, g_flat, n_arr],
        output_shapes=[(n,), (n,), (n,)],
        output_dtypes=[mx.float16, mx.float16, mx.float16],
        grid=(grid_size, 1, 1),
        threadgroup=(min(256, grid_size), 1, 1),
    )
    
    h = outputs[0].reshape(shape)
    dg = outputs[1].reshape(shape)
    de = outputs[2].reshape(shape)
    return h, dg, de


class SwiGLU:
    """
    SwiGLU activation function with custom backward pass.
    
    Similar to torch.autograd.Function but for MLX.
    Usage:
        h = SwiGLU.apply(e, g)
    """
    
    @staticmethod
    def apply(e: mx.array, g: mx.array) -> mx.array:
        """
        Apply SwiGLU forward with gradient tracking.
        
        MLX automatically tracks gradients when arrays are created
        with gradient tracking enabled in the computational graph.
        """
        # Forward pass
        h = swiglu_forward(e, g)
        
        # For MLX, we rely on the higher-level grad() function
        # This is just the forward operation
        return h
    
    @staticmethod
    def forward_and_backward(
        dh: mx.array, e: mx.array, g: mx.array
    ) -> Tuple[mx.array, mx.array]:
        """
        Compute both output and gradients efficiently.
        
        Returns:
            (dg, de): Gradients w.r.t. inputs
        """
        h, dg, de = swiglu_backward(dh, e, g)
        return dg, de


# Convenience function for mx.compile compatibility
def swiglu_forward_compiled(e: mx.array, g: mx.array) -> mx.array:
    """
    SwiGLU forward compatible with mx.compile.
    
    For use with mx.compile() graph fusion.
    """
    # Use native MLX operations that mx.compile can fuse
    sig_e = mx.sigmoid(e)
    return sig_e * e * g


def swiglu_backward_compiled(
    dh: mx.array, e: mx.array, g: mx.array
) -> Tuple[mx.array, mx.array, mx.array]:
    """
    SwiGLU backward compatible with mx.compile.
    """
    sig_e = mx.sigmoid(e)
    h = sig_e * e * g
    
    # Gradients
    one_minus_sig = 1.0 - sig_e
    df_de = sig_e * (1.0 + e * one_minus_sig)
    
    dg = dh * sig_e * e
    de = dh * g * df_de
    
    return h, dg, de
