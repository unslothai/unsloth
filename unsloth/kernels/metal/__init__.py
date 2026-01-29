# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Metal kernel acceleration for Apple Silicon.

Uses mx.fast.rms_norm for standard RMSNorm (highly optimized by Apple).
Custom Metal kernel only for Gemma variant which uses (1+W) scaling.
"""

from __future__ import annotations

from pathlib import Path
from functools import lru_cache
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    import torch

_METAL_AVAILABLE: Optional[bool] = None


def is_metal_available() -> bool:
    """Check if Metal kernel acceleration is available."""
    global _METAL_AVAILABLE
    if _METAL_AVAILABLE is not None:
        return _METAL_AVAILABLE
    
    try:
        import platform
        if platform.system() != "Darwin":
            _METAL_AVAILABLE = False
            return False
        
        from ..mlx.utils import is_mlx_available
        if not is_mlx_available():
            _METAL_AVAILABLE = False
            return False
        
        import mlx.core as mx
        if not hasattr(mx.fast, "rms_norm"):
            _METAL_AVAILABLE = False
            return False
        
        _METAL_AVAILABLE = True
        return True
    except Exception:
        _METAL_AVAILABLE = False
        return False


USE_METAL_KERNEL = is_metal_available()


@lru_cache(maxsize=1)
def _load_gemma_shader() -> str:
    """Load Gemma variant Metal shader."""
    path = Path(__file__).parent / "layernorm.metal"
    if not path.exists():
        raise FileNotFoundError(f"Shader not found: {path}")
    return path.read_text(encoding="utf-8")


def metal_rms_layernorm(
    X: "torch.Tensor",
    W: "torch.Tensor",
    eps: float,
    gemma: bool = False,
) -> "torch.Tensor":
    """
    RMS LayerNorm using MLX acceleration.
    
    Standard variant uses mx.fast.rms_norm (Apple-optimized).
    Gemma variant uses custom Metal kernel for (1+W) scaling.
    """
    import torch
    import mlx.core as mx
    from ..mlx import torch_to_mlx, mlx_to_torch, synchronize_mps
    
    shape = X.shape
    dim = shape[-1]
    X_2d = X.reshape(-1, dim).contiguous()
    W = W.contiguous()
    n_rows, n_cols = X_2d.shape
    
    synchronize_mps()
    
    X_mlx = torch_to_mlx(X_2d)
    W_mlx = torch_to_mlx(W)
    
    is_half = X_mlx.dtype in (mx.float16, mx.bfloat16)
    
    if not gemma:
        # Use MLX's native optimized RMS norm
        Y_mlx = mx.fast.rms_norm(X_mlx, W_mlx, eps)
        mx.eval(Y_mlx)
        Y = mlx_to_torch(Y_mlx, device=X.device, dtype=X.dtype)
        return Y.view(*shape)
    
    # Gemma variant: custom kernel for (1+W) scaling
    if is_half:
        X_mlx = X_mlx.astype(mx.float32)
        W_mlx = W_mlx.astype(mx.float32)
    
    Y_mlx = mx.zeros_like(X_mlx)
    rms_inv = mx.zeros((n_rows,), dtype=mx.float32)
    
    try:
        outputs = mx.metal_kernel(
            name="rms_layernorm_gemma_forward",
            source=_load_gemma_shader(),
            input_names=["X", "W"],
            output_names=["Y", "rms_inv_out"],
            inputs=[X_mlx, W_mlx],
            outputs=[Y_mlx, rms_inv],
            constants={"n_rows": n_rows, "n_cols": n_cols, "eps": eps},
            grid=(n_rows, 1, 1),
            threadgroup=(256, 1, 1),
        )
        mx.eval(outputs[0])
        Y = mlx_to_torch(outputs[0], device=X.device, dtype=X.dtype if is_half else torch.float32)
    except Exception:
        # Fallback to MPS
        from ..mps.rms_layernorm import mps_rms_layernorm
        return mps_rms_layernorm(X, W, eps, gemma)
    
    return Y.view(*shape)


__all__ = ["is_metal_available", "USE_METAL_KERNEL", "metal_rms_layernorm"]
