"""
Pure MLX RMS LayerNorm Kernels - No PyTorch dependency.
"""

import functools
import mlx.core as mx
from typing import Tuple


def mlx_rms_layernorm_forward(
    X: mx.array,
    W: mx.array,
    eps: float = 1e-6,
    gemma: bool = False
) -> Tuple[mx.array, mx.array]:
    """
    Forward RMS LayerNorm using pure MLX.
    
    Args:
        X: Input tensor [..., D]
        W: Weight tensor [D]
        eps: Epsilon for numerical stability
        gemma: If True, use Gemma-style (W + 1) scaling
    
    Returns:
        Y: Normalized output [..., D]
        r: Inverse RMS [..., 1]
    """
    # Cast to float32 for robust variance calculation
    X_f32 = X.astype(mx.float32)
    
    # r = 1 / sqrt(mean(X^2) + eps)
    r = mx.rsqrt(mx.mean(mx.square(X_f32), axis=-1, keepdims=True) + eps)
    
    if not gemma:
        # Standard RMSNorm
        Y = (X_f32 * r) * W.astype(mx.float32)
    else:
        # Gemma uses (W + 1)
        Y = (X_f32 * r) * (W.astype(mx.float32) + 1.0)
    
    return Y.astype(X.dtype), r


def mlx_rms_layernorm_backward(
    dY: mx.array,
    X: mx.array,
    W: mx.array,
    r: mx.array,
    gemma: bool = False
) -> Tuple[mx.array, mx.array]:
    """
    Backward RMS LayerNorm using pure MLX.
    
    Args:
        dY: Gradient from upstream [..., D]
        X: Input tensor [..., D]
        W: Weight tensor [D]
        r: Inverse RMS from forward [..., 1]
        gemma: If True, use Gemma-style (W + 1) scaling
    
    Returns:
        dX: Gradient w.r.t. input [..., D]
        dW: Gradient w.r.t. weight [D]
    """
    # Cast to float32 for high numerical precision
    X_f32 = X.astype(mx.float32)
    W_f32 = W.astype(mx.float32)
    dY_f32 = dY.astype(mx.float32)
    r_f32 = r.astype(mx.float32)
    
    # Clip gradients to the float16 range
    dY_safe = mx.clip(dY_f32, -65504.0, 65504.0)
    
    W_eff = (W_f32 + 1.0) if gemma else W_f32
    
    # 1. dW = sum(dY * X_norm)
    X_norm = X_f32 * r_f32
    dW = mx.sum(dY_safe * X_norm, axis=list(range(X.ndim - 1)))
    
    # 2. dX = r * (dy_w - X_norm * mean(dy_w * X_norm))
    dy_w = dY_safe * W_eff
    dot = dy_w * X_norm
    mean_dot = mx.mean(dot, axis=-1, keepdims=True)
    
    dX = r_f32 * (dy_w - X_norm * mean_dot)
    
    return dX.astype(X.dtype), dW.astype(W.dtype)


class RMSNormMLX:
    """
    MLX-native RMSNorm layer with automatic differentiation.
    """
    
    def __init__(self, dim: int, eps: float = 1e-6, gemma: bool = False):
        self.dim = dim
        self.eps = eps
        self.gemma = gemma
        # Initialize weights
        self.weight = mx.ones((dim,), dtype=mx.float32)
    
    def __call__(self, x: mx.array) -> mx.array:
        return mlx_rms_layernorm_forward(x, self.weight, self.eps, self.gemma)[0]
    
    def parameters(self):
        return {"weight": self.weight}
    
    def state_dict(self):
        return {"weight": self.weight}
    
    def load_state_dict(self, state_dict):
        self.weight = state_dict["weight"]


# Export the functions
__all__ = [
    "mlx_rms_layernorm_forward",
    "mlx_rms_layernorm_backward",
    "RMSNormMLX",
]
