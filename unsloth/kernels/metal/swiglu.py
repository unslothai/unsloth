# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""
Metal-accelerated SwiGLU kernels for Apple Silicon.

Provides high-performance fused SwiGLU forward and backward passes using
custom Metal shaders with float4/half4 vectorization.
"""

from __future__ import annotations

from pathlib import Path
from functools import lru_cache
from typing import TYPE_CHECKING, Tuple, Optional

if TYPE_CHECKING:
    import torch

__all__ = ["metal_swiglu_forward", "metal_swiglu_backward", "is_metal_swiglu_available"]


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

        from ..mlx.utils import is_mlx_available
        if not is_mlx_available():
            _METAL_SWIGLU_AVAILABLE = False
            return False

        # Verify shader exists
        shader_path = Path(__file__).parent / "swiglu.metal"
        if not shader_path.exists():
            _METAL_SWIGLU_AVAILABLE = False
            return False

        _METAL_SWIGLU_AVAILABLE = True
        return True
    except Exception:
        _METAL_SWIGLU_AVAILABLE = False
        return False


@lru_cache(maxsize=1)
def _load_swiglu_shader() -> str:
    """Load SwiGLU Metal shader source."""
    path = Path(__file__).parent / "swiglu.metal"
    if not path.exists():
        raise FileNotFoundError(f"Shader not found: {path}")
    return path.read_text(encoding="utf-8")


def metal_swiglu_forward(e: "torch.Tensor", g: "torch.Tensor") -> "torch.Tensor":
    """
    Fused SwiGLU forward pass using Metal kernel.

    Computes: h = silu(e) * g = (e * sigmoid(e)) * g

    Args:
        e: Gate input tensor [batch, seq_len, hidden_dim]
        g: Up projection tensor [batch, seq_len, hidden_dim]

    Returns:
        h: Output tensor [batch, seq_len, hidden_dim]
    """
    import torch
    import mlx.core as mx
    from ..mlx import torch_to_mlx, mlx_to_torch, synchronize_mps

    # Preserve shape for output
    shape = e.shape
    n_elements = e.numel()

    # Flatten to contiguous 1D for vectorized processing
    e_flat = e.reshape(-1).contiguous()
    g_flat = g.reshape(-1).contiguous()

    synchronize_mps()

    # Convert to MLX
    e_mlx = torch_to_mlx(e_flat)
    g_mlx = torch_to_mlx(g_flat)

    # Determine dtype and kernel
    is_half = e_mlx.dtype in (mx.float16, mx.bfloat16)
    kernel_name = "swiglu_forward_f16" if is_half else "swiglu_forward_f32"

    # Calculate vectorized dimensions
    # float4/half4 = 4 elements per vector
    n_vec = n_elements // 4
    n_remainder = n_elements % 4

    # Allocate output
    h_mlx = mx.zeros_like(e_mlx)

    try:
        if n_vec > 0:
            # Reshape for vectorized access
            e_vec = e_mlx.reshape(-1, 4)[:n_vec]
            g_vec = g_mlx.reshape(-1, 4)[:n_vec]
            h_vec = h_mlx.reshape(-1, 4)[:n_vec]

            # Launch vectorized kernel
            # Grid: one thread per float4 vector
            # Threadgroup: 256 threads for good occupancy
            threadgroup_size = min(256, n_vec)
            grid_size = n_vec

            outputs = mx.metal_kernel(
                name=kernel_name,
                source=_load_swiglu_shader(),
                input_names=["e", "g"],
                output_names=["h"],
                inputs=[e_vec, g_vec],
                outputs=[h_vec],
                constants={"n_vec": n_vec},
                grid=(grid_size, 1, 1),
                threadgroup=(threadgroup_size, 1, 1),
            )
            mx.eval(outputs[0])

            # Copy vectorized results back
            h_mlx = h_mlx.reshape(-1, 4)
            h_mlx = mx.concatenate([outputs[0], h_mlx[n_vec:]], axis=0).reshape(-1)

        # Handle remainder with scalar kernel
        if n_remainder > 0:
            scalar_kernel = "swiglu_forward_f16_scalar" if is_half else "swiglu_forward_f32_scalar"
            offset = n_vec * 4

            outputs = mx.metal_kernel(
                name=scalar_kernel,
                source=_load_swiglu_shader(),
                input_names=["e", "g"],
                output_names=["h"],
                inputs=[e_mlx, g_mlx],
                outputs=[h_mlx],
                constants={"offset": offset, "n_elements": n_elements},
                grid=(n_remainder, 1, 1),
                threadgroup=(n_remainder, 1, 1),
            )
            mx.eval(outputs[0])
            h_mlx = outputs[0]

        # Convert back to torch
        h = mlx_to_torch(h_mlx, device=e.device, dtype=e.dtype)
        return h.view(*shape)

    except Exception as ex:
        # Fallback to MPS implementation
        from ..mps.swiglu import mps_swiglu_forward
        return mps_swiglu_forward(e, g)


def metal_swiglu_backward(
    dw: "torch.Tensor", e: "torch.Tensor", g: "torch.Tensor"
) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
    """
    Fused SwiGLU backward pass using Metal kernel.

    Given upstream gradient dw = dL/dh, computes:
        h  = silu(e) * g               (forward recomputation)
        df = dw * silu(e)              (gradient for down projection)
        de = dw * g * sigmoid(e) * (1 + e * (1 - sigmoid(e)))

    In-place storage (matching Triton kernel semantics):
        DW buffer -> h
        e buffer  -> df
        g buffer  -> de

    Args:
        dw: Upstream gradient [batch_seq, hidden_dim]
        e: Gate input tensor [batch_seq, hidden_dim]
        g: Up projection tensor [batch_seq, hidden_dim]

    Returns:
        Tuple of (h, df, de) tensors
    """
    import torch
    import mlx.core as mx
    from ..mlx import torch_to_mlx, mlx_to_torch, synchronize_mps

    shape = e.shape
    n_elements = e.numel()

    # Clone inputs since backward modifies in-place
    dw_flat = dw.reshape(-1).clone().contiguous()
    e_flat = e.reshape(-1).clone().contiguous()
    g_flat = g.reshape(-1).clone().contiguous()

    synchronize_mps()

    # Convert to MLX
    dw_mlx = torch_to_mlx(dw_flat)
    e_mlx = torch_to_mlx(e_flat)
    g_mlx = torch_to_mlx(g_flat)

    is_half = e_mlx.dtype in (mx.float16, mx.bfloat16)
    kernel_name = "swiglu_backward_f16" if is_half else "swiglu_backward_f32"

    n_vec = n_elements // 4
    n_remainder = n_elements % 4

    try:
        if n_vec > 0:
            # Reshape for vectorized access
            dw_vec = dw_mlx.reshape(-1, 4)[:n_vec]
            e_vec = e_mlx.reshape(-1, 4)[:n_vec]
            g_vec = g_mlx.reshape(-1, 4)[:n_vec]

            threadgroup_size = min(256, n_vec)
            grid_size = n_vec

            outputs = mx.metal_kernel(
                name=kernel_name,
                source=_load_swiglu_shader(),
                input_names=["DW", "e", "g"],
                output_names=["DW", "e", "g"],
                inputs=[dw_vec, e_vec, g_vec],
                outputs=[dw_vec, e_vec, g_vec],
                constants={"n_vec": n_vec},
                grid=(grid_size, 1, 1),
                threadgroup=(threadgroup_size, 1, 1),
            )
            mx.eval(outputs)

            # Reassemble from vectorized outputs
            h_mlx = mx.concatenate([outputs[0].reshape(-1), dw_mlx[n_vec * 4:]])
            df_mlx = mx.concatenate([outputs[1].reshape(-1), e_mlx[n_vec * 4:]])
            de_mlx = mx.concatenate([outputs[2].reshape(-1), g_mlx[n_vec * 4:]])
        else:
            h_mlx = dw_mlx
            df_mlx = e_mlx
            de_mlx = g_mlx

        # Handle remainder
        if n_remainder > 0:
            scalar_kernel = "swiglu_backward_f16_scalar" if is_half else "swiglu_backward_f32_scalar"
            offset = n_vec * 4

            outputs = mx.metal_kernel(
                name=scalar_kernel,
                source=_load_swiglu_shader(),
                input_names=["DW", "e", "g"],
                output_names=["DW", "e", "g"],
                inputs=[h_mlx, df_mlx, de_mlx],
                outputs=[h_mlx, df_mlx, de_mlx],
                constants={"offset": offset, "n_elements": n_elements},
                grid=(n_remainder, 1, 1),
                threadgroup=(n_remainder, 1, 1),
            )
            mx.eval(outputs)
            h_mlx, df_mlx, de_mlx = outputs[0], outputs[1], outputs[2]

        # Convert back to torch
        h = mlx_to_torch(h_mlx, device=dw.device, dtype=dw.dtype).view(*shape)
        df = mlx_to_torch(df_mlx, device=e.device, dtype=e.dtype).view(*shape)
        de = mlx_to_torch(de_mlx, device=g.device, dtype=g.dtype).view(*shape)

        return h, df, de

    except Exception as ex:
        # Fallback to MPS implementation
        from ..mps.swiglu import mps_swiglu_backward
        return mps_swiglu_backward(dw, e, g)
