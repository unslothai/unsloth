# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""MLX fast kernel wrappers for Apple Silicon.

Uses mx.fast.* operations which are highly optimized by Apple:
- mx.fast.layer_norm
- mx.fast.rope
- mx.fast.scaled_dot_product_attention
- mx.fast.rms_norm
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    import torch

_MLX_FAST_AVAILABLE: Optional[bool] = None


def is_mlx_fast_available() -> bool:
    """Check if MLX fast operations are available."""
    global _MLX_FAST_AVAILABLE
    if _MLX_FAST_AVAILABLE is not None:
        return _MLX_FAST_AVAILABLE

    try:
        import platform

        if platform.system() != "Darwin":
            _MLX_FAST_AVAILABLE = False
            return False

        from .utils import is_mlx_available

        if not is_mlx_available():
            _MLX_FAST_AVAILABLE = False
            return False

        import mlx.core as mx

        # Check all fast ops are available
        if not all(hasattr(mx.fast, op) for op in ["rms_norm", "layer_norm", "rope"]):
            _MLX_FAST_AVAILABLE = False
            return False

        _MLX_FAST_AVAILABLE = True
        return True
    except Exception:
        _MLX_FAST_AVAILABLE = False
        return False


USE_MLX_FAST = is_mlx_fast_available()


def mlx_layer_norm(
    X: "torch.Tensor",
    W: "torch.Tensor",
    b: "torch.Tensor",
    eps: float,
) -> "torch.Tensor":
    """LayerNorm using mx.fast.layer_norm."""
    import torch
    import mlx.core as mx
    from .bridge import torch_to_mlx, mlx_to_torch, synchronize_mps

    shape = X.shape
    X_2d = X.reshape(-1, shape[-1]).contiguous()

    synchronize_mps()

    X_mlx = torch_to_mlx(X_2d)
    W_mlx = torch_to_mlx(W)
    b_mlx = torch_to_mlx(b)

    Y_mlx = mx.fast.layer_norm(X_mlx, W_mlx, b_mlx, eps)
    mx.eval(Y_mlx)

    Y = mlx_to_torch(Y_mlx, device=X.device, dtype=X.dtype)
    return Y.view(*shape)


def mlx_rope(
    Q: "torch.Tensor",
    cos: "torch.Tensor",
    sin: "torch.Tensor",
    offset: int = 0,
) -> "torch.Tensor":
    """RoPE embedding using mx.fast.rope."""
    import torch
    import mlx.core as mx
    from .bridge import torch_to_mlx, mlx_to_torch, synchronize_mps

    synchronize_mps()

    Q_mlx = torch_to_mlx(Q.contiguous())

    # mx.fast.rope expects shape (batch, seq, heads, dim)
    # Unsloth uses (batch, heads, seq, dim) - need transpose
    if Q_mlx.ndim == 4:
        Q_mlx = mx.transpose(Q_mlx, (0, 2, 1, 3))

    Y_mlx = mx.fast.rope(Q_mlx, Q_mlx.shape[-1], traditional=False, offset=offset)

    # Transpose back
    if Y_mlx.ndim == 4:
        Y_mlx = mx.transpose(Y_mlx, (0, 2, 1, 3))

    mx.eval(Y_mlx)
    Y = mlx_to_torch(Y_mlx, device=Q.device, dtype=Q.dtype)
    return Y


def mlx_rope_qk(
    Q: "torch.Tensor",
    K: "torch.Tensor",
    cos: "torch.Tensor",
    sin: "torch.Tensor",
    offset: int = 0,
) -> tuple["torch.Tensor", "torch.Tensor"]:
    """RoPE embedding for Q and K using mx.fast.rope."""
    import torch
    import mlx.core as mx
    from .bridge import torch_to_mlx, mlx_to_torch, synchronize_mps

    synchronize_mps()

    Q_mlx = torch_to_mlx(Q.contiguous())
    K_mlx = torch_to_mlx(K.contiguous())

    # Transpose to (batch, seq, heads, dim) for MLX
    if Q_mlx.ndim == 4:
        Q_mlx = mx.transpose(Q_mlx, (0, 2, 1, 3))
        K_mlx = mx.transpose(K_mlx, (0, 2, 1, 3))

    dim = Q_mlx.shape[-1]
    Q_out = mx.fast.rope(Q_mlx, dim, traditional=False, offset=offset)
    K_out = mx.fast.rope(K_mlx, dim, traditional=False, offset=offset)

    # Transpose back
    if Q_out.ndim == 4:
        Q_out = mx.transpose(Q_out, (0, 2, 1, 3))
        K_out = mx.transpose(K_out, (0, 2, 1, 3))

    mx.eval(Q_out, K_out)

    Q_t = mlx_to_torch(Q_out, device=Q.device, dtype=Q.dtype)
    K_t = mlx_to_torch(K_out, device=K.device, dtype=K.dtype)
    return Q_t, K_t


def mlx_scaled_dot_product_attention(
    Q: "torch.Tensor",
    K: "torch.Tensor",
    V: "torch.Tensor",
    scale: float = None,
    mask: "torch.Tensor" = None,
) -> "torch.Tensor":
    """Scaled dot-product attention using mx.fast.scaled_dot_product_attention."""
    import torch
    import mlx.core as mx
    from .bridge import torch_to_mlx, mlx_to_torch, synchronize_mps

    synchronize_mps()

    Q_mlx = torch_to_mlx(Q.contiguous())
    K_mlx = torch_to_mlx(K.contiguous())
    V_mlx = torch_to_mlx(V.contiguous())

    if scale is None:
        scale = Q.shape[-1] ** -0.5

    mask_mlx = torch_to_mlx(mask.contiguous()) if mask is not None else None

    Y_mlx = mx.fast.scaled_dot_product_attention(
        Q_mlx, K_mlx, V_mlx, scale=scale, mask=mask_mlx
    )
    mx.eval(Y_mlx)

    return mlx_to_torch(Y_mlx, device=Q.device, dtype=Q.dtype)


__all__ = [
    "is_mlx_fast_available",
    "USE_MLX_FAST",
    "mlx_layer_norm",
    "mlx_rope",
    "mlx_rope_qk",
    "mlx_scaled_dot_product_attention",
]
