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

import mlx.core as mx
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


def mlx_rms_norm(
    X: "torch.Tensor",
    W: "torch.Tensor",
    eps: float,
) -> "torch.Tensor":
    """RMSNorm using mx.fast.rms_norm."""
    import torch
    import mlx.core as mx
    from .bridge import torch_to_mlx, mlx_to_torch, synchronize_mps

    shape = X.shape
    X_2d = X.reshape(-1, shape[-1]).contiguous()

    synchronize_mps()

    X_mlx = torch_to_mlx(X_2d)
    W_mlx = torch_to_mlx(W)

    Y_mlx = mx.fast.rms_norm(X_mlx, W_mlx, eps)
    mx.eval(Y_mlx)

    Y = mlx_to_torch(Y_mlx, device=X.device, dtype=X.dtype)
    return Y.view(*shape)


@mx.compile
def _mlx_rope_kernel(Q, cos, sin):
    # Standard Rotary: [q0, q1, ..., qn/2, qn/2+1, ..., qn]
    # rotate_half: [-qn/2, ..., -qn, q0, ..., qn/2]
    half = Q.shape[-1] // 2
    Q1 = Q[..., :half]
    Q2 = Q[..., half:]
    
    # cos/sin are [1, 1, S, D] or similar, broadcastable to Q
    # We expect cos/sin to be the full head_dim (or broadcastable)
    # Match MPS fallback: (Q * cos) + (rotate_half(Q) * sin)
    # Out1 = Q1 * cos1 - Q2 * sin1
    # Out2 = Q2 * cos2 + Q1 * sin2 (if cos1 == cos2)
    
    # If cos/sin were provided as half-dim [S, D/2], we assume they are repeated
    # In fast_ops, bridge will likely have them as full dim if we follow the MPS fallback logic
    
    cos1 = cos[..., :half]
    cos2 = cos[..., half:]
    sin1 = sin[..., :half]
    sin2 = sin[..., half:]

    out1 = Q1 * cos1 - Q2 * sin1
    out2 = Q2 * cos2 + Q1 * sin2
    
    return mx.concatenate([out1, out2], axis=-1)

def mlx_rope(
    Q: "torch.Tensor",
    cos: "torch.Tensor",
    sin: "torch.Tensor",
    position_ids: "torch.Tensor" = None,
) -> "torch.Tensor":
    """RoPE embedding using correctly composed MLX arithmetic."""
    import torch
    import mlx.core as mx
    from .bridge import torch_to_mlx, mlx_to_torch, synchronize_mps

    synchronize_mps()

    if isinstance(Q, mx.array):
        Q_mlx = Q
        return_mlx = True
    else:
        Q_mlx = torch_to_mlx(Q.contiguous())
        return_mlx = False

    cos_mlx = torch_to_mlx(cos)
    sin_mlx = torch_to_mlx(sin)

    # Broadcast cos/sin to Q_mlx shape (usually batch, heads, seq, dim)
    # We assume cos/sin are [seq, dim] or [batch, seq, dim]
    # We need to unsqueeze to match Q_mlx [B, H, S, D]
    if Q_mlx.ndim == 4:
        # Check if Q is [B, H, S, D] (Unsloth typical)
        # or [B, S, H, D]
        if Q_mlx.shape[2] == cos_mlx.shape[-2]: # [B, H, S, D]
            cos_mlx = cos_mlx.reshape(1, 1, cos_mlx.shape[-2], cos_mlx.shape[-1])
            sin_mlx = sin_mlx.reshape(1, 1, sin_mlx.shape[-2], sin_mlx.shape[-1])
        else: # [B, S, H, D]
            cos_mlx = cos_mlx.reshape(1, cos_mlx.shape[-2], 1, cos_mlx.shape[-1])
            sin_mlx = sin_mlx.reshape(1, sin_mlx.shape[-2], 1, sin_mlx.shape[-1])

    Y_mlx = _mlx_rope_kernel(Q_mlx, cos_mlx, sin_mlx)


    
    if return_mlx:
        return Y_mlx
        
    Y = mlx_to_torch(Y_mlx, device=Q.device, dtype=Q.dtype)
    return Y


def mlx_rope_qk(
    Q: "torch.Tensor",
    K: "torch.Tensor",
    cos: "torch.Tensor",
    sin: "torch.Tensor",
    position_ids: "torch.Tensor" = None,
) -> tuple["torch.Tensor", "torch.Tensor"]:
    """RoPE embedding for Q and K using correctly composed MLX arithmetic."""
    import torch
    import mlx.core as mx
    from .bridge import torch_to_mlx, mlx_to_torch, synchronize_mps

    synchronize_mps()

    if isinstance(Q, mx.array):
        Q_mlx = Q
        K_mlx = K
        return_mlx = True
    else:
        Q_mlx = torch_to_mlx(Q.contiguous())
        K_mlx = torch_to_mlx(K.contiguous())
        return_mlx = False

    if isinstance(cos, mx.array):
        cos_mlx = cos
    else:
        cos_mlx = torch_to_mlx(cos)
        
    if isinstance(sin, mx.array):
        sin_mlx = sin
    else:
        sin_mlx = torch_to_mlx(sin)

    # Broadcasting
    if Q_mlx.ndim == 4:
        if Q_mlx.shape[2] == cos_mlx.shape[-2]: # [B, H, S, D]
            cos_mlx = cos_mlx.reshape(1, 1, cos_mlx.shape[-2], cos_mlx.shape[-1])
            sin_mlx = sin_mlx.reshape(1, 1, sin_mlx.shape[-2], sin_mlx.shape[-1])
        else: # [B, S, H, D]
            cos_mlx = cos_mlx.reshape(1, cos_mlx.shape[-2], 1, cos_mlx.shape[-1])
            sin_mlx = sin_mlx.reshape(1, sin_mlx.shape[-2], 1, sin_mlx.shape[-1])

    Q_out = _mlx_rope_kernel(Q_mlx, cos_mlx, sin_mlx)
    K_out = _mlx_rope_kernel(K_mlx, cos_mlx, sin_mlx)



    if return_mlx:
        return Q_out, K_out

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
    "mlx_rms_norm",
    "mlx_rope",
    "mlx_rope_qk",
    "mlx_scaled_dot_product_attention",
]
