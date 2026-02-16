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

import torch
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    pass

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
    gemma: bool = False,
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

    if gemma:
        W_mlx = W_mlx + 1.0

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

    # Handle position_ids for sliced RoPE (generation/prefill)
    if position_ids is not None:
        if isinstance(cos, mx.array):
            cos_mlx = cos
            sin_mlx = sin
        else:
            cos_mlx = torch_to_mlx(cos)
            sin_mlx = torch_to_mlx(sin)
        
        # Index into cos/sin using position_ids
        pos_mlx = torch_to_mlx(position_ids) if not isinstance(position_ids, mx.array) else position_ids
        cos_mlx = cos_mlx[pos_mlx]  # [B, S, D/2] or [S, D/2]
        sin_mlx = sin_mlx[pos_mlx]
        
        # Add dimensions for broadcasting
        if cos_mlx.ndim == 2:
            cos_mlx = cos_mlx[None, ...]  # [1, S, D/2]
            sin_mlx = sin_mlx[None, ...]
        
        if Q_mlx.ndim == 4:
            seq_len = cos_mlx.shape[-2]
            if Q_mlx.shape[2] == seq_len:  # [B, H, S, D]
                cos_mlx = cos_mlx[:, None, :, :]  # [B, 1, S, D/2]
                sin_mlx = sin_mlx[:, None, :, :]
            else:  # [B, S, H, D]
                cos_mlx = cos_mlx[:, :, None, :]  # [B, S, 1, D/2]
                sin_mlx = sin_mlx[:, :, None, :]
    else:
        # Full sequence RoPE - cos/sin are [MaxSeq, D] or [MaxSeq, D/2]
        if isinstance(cos, mx.array):
            cos_mlx = cos
            sin_mlx = sin
        else:
            cos_mlx = torch_to_mlx(cos)
            sin_mlx = torch_to_mlx(sin)
        
        if Q_mlx.ndim == 4:
            # Determine sequence length from Q
            # Q is either [B, H, S, D] or [B, S, H, D]
            s1 = Q_mlx.shape[1]
            s2 = Q_mlx.shape[2]
            
            # Heuristic: if s2 <= cos seq dim and s1 != cos seq dim, then Q is [B, H, S, D]
            cos_seq_dim = cos_mlx.shape[0] if cos_mlx.ndim == 2 else cos_mlx.shape[-2]
            
            if s2 <= cos_seq_dim and s1 != cos_seq_dim:
                # [B, H, S, D] case - slice cos/sin to actual seq length
                cos_mlx = cos_mlx[:s2].reshape(1, 1, s2, -1)
                sin_mlx = sin_mlx[:s2].reshape(1, 1, s2, -1)
            else:
                # [B, S, H, D] case
                cos_mlx = cos_mlx[:s1].reshape(1, s1, 1, -1)
                sin_mlx = sin_mlx[:s1].reshape(1, s1, 1, -1)
        elif Q_mlx.ndim == 3:
            # [B, S, D] - slice to actual seq
            seq = Q_mlx.shape[1]
            cos_mlx = cos_mlx[:seq].reshape(1, seq, -1)
            sin_mlx = sin_mlx[:seq].reshape(1, seq, -1)

    # If head_dim in cos/sin is half of Q's head_dim, repeat it
    if cos_mlx.shape[-1] * 2 == Q_mlx.shape[-1]:
        cos_mlx = mx.concatenate([cos_mlx, cos_mlx], axis=-1)
        sin_mlx = mx.concatenate([sin_mlx, sin_mlx], axis=-1)

    Q_out = _mlx_rope_kernel(Q_mlx, cos_mlx, sin_mlx)
    K_out = _mlx_rope_kernel(K_mlx, cos_mlx, sin_mlx)

    if return_mlx:
        return Q_out, K_out

    Q_t = mlx_to_torch(Q_out, device=Q.device, dtype=Q.dtype)
    K_t = mlx_to_torch(K_out, device=K.device, dtype=K.dtype)
    return Q_t, K_t


@mx.compile
def _mlx_swiglu_kernel(e, g):
    return (e * mx.sigmoid(e)) * g


def mlx_swiglu(
    e: "torch.Tensor",
    g: "torch.Tensor",
) -> "torch.Tensor":
    """SwiGLU activation using MLX."""
    import torch
    import mlx.core as mx
    from .bridge import torch_to_mlx, mlx_to_torch, synchronize_mps

    shape = e.shape
    synchronize_mps()

    e_mlx = torch_to_mlx(e.contiguous())
    g_mlx = torch_to_mlx(g.contiguous())

    h_mlx = _mlx_swiglu_kernel(e_mlx, g_mlx)
    mx.eval(h_mlx)

    return mlx_to_torch(h_mlx, device=e.device, dtype=e.dtype)


@mx.compile
def _mlx_geglu_exact_kernel(e, g):
    return (0.5 * e * (1 + mx.erf(e / 1.4142135623730951))) * g  # 1.414... is sqrt(2)


def mlx_geglu_exact(
    e: "torch.Tensor",
    g: "torch.Tensor",
) -> "torch.Tensor":
    """Exact GeGLU activation using MLX."""
    import torch
    import mlx.core as mx
    from .bridge import torch_to_mlx, mlx_to_torch, synchronize_mps

    shape = e.shape
    synchronize_mps()

    e_mlx = torch_to_mlx(e.contiguous())
    g_mlx = torch_to_mlx(g.contiguous())

    h_mlx = _mlx_geglu_exact_kernel(e_mlx, g_mlx)
    mx.eval(h_mlx)

    return mlx_to_torch(h_mlx, device=e.device, dtype=e.dtype)


@mx.compile
def _mlx_geglu_approx_kernel(e, g):
    # f = 1/2 * e * (1 + tanh( sqrt(2/pi) * (x + 0.044715 * x^3 ) ))
    s = 0.7978845608028654  # sqrt(2 / pi)
    return (0.5 * e * (1 + mx.tanh(s * (e + 0.044715 * e * e * e)))) * g


def mlx_geglu_approx(
    e: "torch.Tensor",
    g: "torch.Tensor",
) -> "torch.Tensor":
    """Approximate GeGLU activation using MLX."""
    import torch
    import mlx.core as mx
    from .bridge import torch_to_mlx, mlx_to_torch, synchronize_mps

    shape = e.shape
    synchronize_mps()

    e_mlx = torch_to_mlx(e.contiguous())
    g_mlx = torch_to_mlx(g.contiguous())

    h_mlx = _mlx_geglu_approx_kernel(e_mlx, g_mlx)
    mx.eval(h_mlx)

    return mlx_to_torch(h_mlx, device=e.device, dtype=e.dtype)


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


class MLXRMSLayerNorm(torch.autograd.Function):
    """MLX-accelerated RMSNorm with custom autograd for training support."""

    @staticmethod
    def forward(ctx, X: "torch.Tensor", W: "torch.Tensor", eps: float, gemma: bool = False):
        import mlx.core as mx
        from .bridge import torch_to_mlx, mlx_to_torch, synchronize_mps

        synchronize_mps()

        shape = X.shape
        dim = shape[-1]
        X_2d = X.reshape(-1, dim).contiguous()

        X_mlx = torch_to_mlx(X_2d)
        W_mlx = torch_to_mlx(W)

        if gemma:
            W_mlx = W_mlx + 1.0

        Y_mlx = mx.fast.rms_norm(X_mlx, W_mlx, eps)
        mx.eval(Y_mlx)

        Y = mlx_to_torch(Y_mlx, device=X.device, dtype=X.dtype)
        Y = Y.view(*shape)

        ctx.save_for_backward(X, W)
        ctx.eps = eps
        ctx.gemma = gemma
        return Y

    @staticmethod
    def backward(ctx, dY: "torch.Tensor"):
        import torch

        shape = dY.shape
        dim = shape[-1]
        dY = dY.reshape(-1, dim)

        X, W = ctx.saved_tensors
        eps = ctx.eps
        gemma = ctx.gemma

        X_f32 = X.reshape(-1, dim).to(torch.float32)
        dY_f32 = dY.to(torch.float32)

        if gemma:
            W_effective = W.to(torch.float32) + 1.0
        else:
            W_effective = W.to(torch.float32)

        variance = X_f32.pow(2).mean(-1, keepdim=True)
        rms_inv = torch.rsqrt(variance + eps)
        X_norm = X_f32 * rms_inv

        dX_norm = dY_f32 * W_effective
        dW = (dY_f32 * X_norm).sum(dim=0)

        N = dim
        rowsum_dY_normed = (dX_norm * X_norm).sum(-1, keepdim=True)
        dX = rms_inv * (dX_norm - (X_norm / N) * rowsum_dY_normed)

        return dX.reshape(*shape).to(X.dtype), dW.to(W.dtype), None, None


class MLXLayerNorm(torch.autograd.Function):
    """MLX-accelerated LayerNorm with custom autograd for training support."""

    @staticmethod
    def forward(ctx, X: "torch.Tensor", W: "torch.Tensor", b: "torch.Tensor", eps: float):
        import mlx.core as mx
        from .bridge import torch_to_mlx, mlx_to_torch, synchronize_mps

        synchronize_mps()

        shape = X.shape
        dim = shape[-1]
        X_2d = X.reshape(-1, dim).contiguous()

        X_mlx = torch_to_mlx(X_2d)
        W_mlx = torch_to_mlx(W)
        b_mlx = torch_to_mlx(b)

        Y_mlx = mx.fast.layer_norm(X_mlx, W_mlx, b_mlx, eps)
        mx.eval(Y_mlx)

        Y = mlx_to_torch(Y_mlx, device=X.device, dtype=X.dtype)
        Y = Y.view(*shape)

        ctx.save_for_backward(X, W, b)
        ctx.eps = eps
        return Y

    @staticmethod
    def backward(ctx, dY: "torch.Tensor"):
        import torch

        shape = dY.shape
        dim = shape[-1]
        dY = dY.reshape(-1, dim)

        X, W, b = ctx.saved_tensors
        eps = ctx.eps

        X_f32 = X.reshape(-1, dim).to(torch.float32)
        dY_f32 = dY.to(torch.float32)

        mean = X_f32.mean(-1, keepdim=True)
        X_centered = X_f32 - mean
        variance = X_f32.var(-1, keepdim=True, unbiased=False)
        inv_std = torch.rsqrt(variance + eps)
        X_norm = X_centered * inv_std

        dW = (dY_f32 * X_norm).sum(0)
        db = dY_f32.sum(0)

        dX_norm = dY_f32 * W.to(torch.float32)

        N = dim
        sum_dX_norm = dX_norm.sum(-1, keepdim=True)
        sum_dX_norm_X_norm = (dX_norm * X_norm).sum(-1, keepdim=True)

        dX = inv_std * (
            dX_norm
            - (sum_dX_norm / N)
            - X_norm * (sum_dX_norm_X_norm / N)
        )

        return dX.reshape(*shape).to(X.dtype), dW.to(W.dtype), db.to(b.dtype), None


def mlx_rms_norm_autograd(
    X: "torch.Tensor",
    W: "torch.Tensor",
    eps: float,
    gemma: bool = False,
) -> "torch.Tensor":
    """RMSNorm using MLX with autograd support for training."""
    return MLXRMSLayerNorm.apply(X, W, eps, gemma)


def mlx_layer_norm_autograd(
    X: "torch.Tensor",
    W: "torch.Tensor",
    b: "torch.Tensor",
    eps: float,
) -> "torch.Tensor":
    """LayerNorm using MLX with autograd support for training."""
    return MLXLayerNorm.apply(X, W, b, eps)


class MLXRoPE(torch.autograd.Function):
    """MLX-accelerated RoPE with custom autograd for training support."""

    @staticmethod
    def forward(ctx, Q: "torch.Tensor", K: "torch.Tensor", cos: "torch.Tensor", sin: "torch.Tensor"):
        import mlx.core as mx
        from .bridge import torch_to_mlx, mlx_to_torch, synchronize_mps

        synchronize_mps()

        Q_mlx = torch_to_mlx(Q.contiguous())
        K_mlx = torch_to_mlx(K.contiguous()) if K is not None else None
        
        seq_len = Q.shape[2] if Q.ndim == 4 else Q.shape[1]
        cos_mlx = torch_to_mlx(cos[:seq_len])
        sin_mlx = torch_to_mlx(sin[:seq_len])

        half = Q_mlx.shape[-1] // 2
        cos1 = cos_mlx[..., :half]
        cos2 = cos_mlx[..., half:]
        sin1 = sin_mlx[..., :half]
        sin2 = sin_mlx[..., half:]

        if Q_mlx.ndim == 4 and Q_mlx.shape[2] == cos_mlx.shape[-2]:
            cos1 = cos1.reshape(1, 1, cos1.shape[-2], cos1.shape[-1])
            cos2 = cos2.reshape(1, 1, cos2.shape[-2], cos2.shape[-1])
            sin1 = sin1.reshape(1, 1, sin1.shape[-2], sin1.shape[-1])
            sin2 = sin2.reshape(1, 1, sin2.shape[-2], sin2.shape[-1])

        Q1 = Q_mlx[..., :half]
        Q2 = Q_mlx[..., half:]
        Q_out1 = Q1 * cos1 - Q2 * sin1
        Q_out2 = Q2 * cos2 + Q1 * sin2
        Q_out = mx.concatenate([Q_out1, Q_out2], axis=-1)

        if K_mlx is not None:
            K1 = K_mlx[..., :half]
            K2 = K_mlx[..., half:]
            K_out1 = K1 * cos1 - K2 * sin1
            K_out2 = K2 * cos2 + K1 * sin2
            K_out = mx.concatenate([K_out1, K_out2], axis=-1)
            mx.eval(Q_out, K_out)
            ctx.save_for_backward(Q, K, cos, sin)
            return mlx_to_torch(Q_out, device=Q.device, dtype=Q.dtype), mlx_to_torch(K_out, device=K.device, dtype=K.dtype)

        mx.eval(Q_out)
        ctx.save_for_backward(Q, K, cos, sin)
        return mlx_to_torch(Q_out, device=Q.device, dtype=Q.dtype), None

    @staticmethod
    def backward(ctx, dQ_out: "torch.Tensor", dK_out: "torch.Tensor"):
        import torch

        Q, K, cos, sin = ctx.saved_tensors

        half = Q.shape[-1] // 2
        cos1 = cos[..., :half]
        cos2 = cos[..., half:]
        sin1 = sin[..., :half]
        sin2 = sin[..., half:]

        if Q.ndim == 4 and Q.shape[2] == cos.shape[-2]:
            cos1 = cos1.reshape(1, 1, cos1.shape[-2], cos1.shape[-1])
            cos2 = cos2.reshape(1, 1, cos2.shape[-2], cos2.shape[-1])
            sin1 = sin1.reshape(1, 1, sin1.shape[-2], sin1.shape[-1])
            sin2 = sin2.reshape(1, 1, sin2.shape[-2], sin2.shape[-1])

        dQ = dQ_out
        dQ1 = dQ[..., :half]
        dQ2 = dQ[..., half:]
        dQ_out1 = dQ1 * cos1 + dQ2 * sin2
        dQ_out2 = dQ2 * cos2 - dQ1 * sin1

        dQ_final = torch.concatenate([dQ_out1, dQ_out2], dim=-1)

        dK_final = None
        if dK_out is not None and K is not None:
            dK = dK_out
            dK1 = dK[..., :half]
            dK2 = dK[..., half:]
            dK_out1 = dK1 * cos1 + dK2 * sin2
            dK_out2 = dK2 * cos2 - dK1 * sin1
            dK_final = torch.concatenate([dK_out1, dK_out2], dim=-1)

        return dQ_final, dK_final, None, None


class MLXSwiGLU(torch.autograd.Function):
    """MLX-accelerated SwiGLU with custom autograd for training support."""

    @staticmethod
    def forward(ctx, e: "torch.Tensor", g: "torch.Tensor"):
        import mlx.core as mx
        from .bridge import torch_to_mlx, mlx_to_torch, synchronize_mps

        synchronize_mps()

        e_mlx = torch_to_mlx(e.contiguous())
        g_mlx = torch_to_mlx(g.contiguous())

        h_mlx = (e_mlx * mx.sigmoid(e_mlx)) * g_mlx
        mx.eval(h_mlx)

        h = mlx_to_torch(h_mlx, device=e.device, dtype=e.dtype)
        ctx.save_for_backward(e, g)
        return h

    @staticmethod
    def backward(ctx, dh: "torch.Tensor"):
        import torch

        e, g = ctx.saved_tensors
        e_f32 = e.to(torch.float32)
        g_f32 = g.to(torch.float32)
        dh_f32 = dh.to(torch.float32)

        sigmoid_e = torch.sigmoid(e_f32)
        silu_e = e_f32 * sigmoid_e
        dsilu = sigmoid_e * (1.0 + e_f32 * (1.0 - sigmoid_e))

        de = dh_f32 * g_f32 * dsilu
        dg = dh_f32 * silu_e

        return de.to(e.dtype), dg.to(g.dtype)


class MLXGeGLUExact(torch.autograd.Function):
    """MLX-accelerated GeGLU exact with custom autograd for training support."""

    @staticmethod
    def forward(ctx, gate: "torch.Tensor", up: "torch.Tensor"):
        import mlx.core as mx
        from .bridge import torch_to_mlx, mlx_to_torch, synchronize_mps

        synchronize_mps()

        gate_mlx = torch_to_mlx(gate.contiguous())
        up_mlx = torch_to_mlx(up.contiguous())

        h_mlx = (0.5 * gate_mlx * (1 + mx.erf(gate_mlx / 1.4142135623730951))) * up_mlx
        mx.eval(h_mlx)

        h = mlx_to_torch(h_mlx, device=gate.device, dtype=gate.dtype)
        ctx.save_for_backward(gate, up)
        return h

    @staticmethod
    def backward(ctx, dh: "torch.Tensor"):
        import torch
        import math

        gate, up = ctx.saved_tensors
        gate_f32 = gate.to(torch.float32)
        up_f32 = up.to(torch.float32)
        dh_f32 = dh.to(torch.float32)

        sqrt_2_over_pi = math.sqrt(2.0 / math.pi)
        cdf = 0.5 * (1.0 + torch.erf(gate_f32 / math.sqrt(2.0)))
        pdf = sqrt_2_over_pi * torch.exp(-0.5 * gate_f32 * gate_f32)
        dgelu = cdf + gate_f32 * pdf

        dh_up = dh_f32 * torch.nn.functional.gelu(gate_f32)
        dh_gate = dh_f32 * up_f32 * dgelu

        return dh_gate.to(gate.dtype), dh_up.to(up.dtype)


class MLXGeGLUApprox(torch.autograd.Function):
    """MLX-accelerated GeGLU approx with custom autograd for training support."""

    @staticmethod
    def forward(ctx, gate: "torch.Tensor", up: "torch.Tensor"):
        import mlx.core as mx
        from .bridge import torch_to_mlx, mlx_to_torch, synchronize_mps

        synchronize_mps()

        gate_mlx = torch_to_mlx(gate.contiguous())
        up_mlx = torch_to_mlx(up.contiguous())

        s = 0.7978845608028654
        h_mlx = (0.5 * gate_mlx * (1 + mx.tanh(s * (gate_mlx + 0.044715 * gate_mlx * gate_mlx * gate_mlx)))) * up_mlx
        mx.eval(h_mlx)

        h = mlx_to_torch(h_mlx, device=gate.device, dtype=gate.dtype)
        ctx.save_for_backward(gate, up)
        return h

    @staticmethod
    def backward(ctx, dh: "torch.Tensor"):
        import torch
        import math

        gate, up = ctx.saved_tensors
        gate_f32 = gate.to(torch.float32)
        up_f32 = up.to(torch.float32)
        dh_f32 = dh.to(torch.float32)

        sqrt_2_over_pi = math.sqrt(2.0 / math.pi)
        inner = sqrt_2_over_pi * (gate_f32 + 0.044715 * gate_f32 * gate_f32 * gate_f32)
        tanh_val = torch.tanh(inner)
        f = 0.5 * gate_f32 * (1.0 + tanh_val)
        dtanh = 1.0 - tanh_val * tanh_val
        dinner = sqrt_2_over_pi * (1.0 + 3.0 * 0.044715 * gate_f32 * gate_f32)
        dgelu = 0.5 * (1.0 + tanh_val) + 0.5 * gate_f32 * dtanh * dinner

        dh_up = dh_f32 * f
        dh_gate = dh_f32 * up_f32 * dgelu

        return dh_gate.to(gate.dtype), dh_up.to(up.dtype)


def mlx_rope_autograd(
    Q: "torch.Tensor",
    K: "torch.Tensor",
    cos: "torch.Tensor",
    sin: "torch.Tensor",
) -> tuple["torch.Tensor", "torch.Tensor"]:
    """RoPE using MLX with autograd support for training."""
    return MLXRoPE.apply(Q, K, cos, sin)


def mlx_swiglu_autograd(
    e: "torch.Tensor",
    g: "torch.Tensor",
) -> "torch.Tensor":
    """SwiGLU using MLX with autograd support for training."""
    return MLXSwiGLU.apply(e, g)


def mlx_geglu_exact_autograd(
    gate: "torch.Tensor",
    up: "torch.Tensor",
) -> "torch.Tensor":
    """GeGLU exact using MLX with autograd support for training."""
    return MLXGeGLUExact.apply(gate, up)


def mlx_geglu_approx_autograd(
    gate: "torch.Tensor",
    up: "torch.Tensor",
) -> "torch.Tensor":
    """GeGLU approx using MLX with autograd support for training."""
    return MLXGeGLUApprox.apply(gate, up)


__all__ = [
    "is_mlx_fast_available",
    "USE_MLX_FAST",
    "mlx_layer_norm",
    "mlx_rms_norm",
    "mlx_rms_norm_autograd",
    "mlx_layer_norm_autograd",
    "mlx_rope",
    "mlx_rope_qk",
    "mlx_rope_autograd",
    "mlx_swiglu",
    "mlx_swiglu_autograd",
    "mlx_geglu_exact",
    "mlx_geglu_exact_autograd",
    "mlx_geglu_approx",
    "mlx_geglu_approx_autograd",
    "mlx_scaled_dot_product_attention",
]
