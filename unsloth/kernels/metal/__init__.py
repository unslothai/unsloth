# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Metal kernel acceleration for Apple Silicon.

Uses MLX's mx.fast.metal_kernel for custom fused operations.
"""

from __future__ import annotations

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

        import mlx.core as mx

        if not hasattr(mx, "fast") or not hasattr(mx.fast, "metal_kernel"):
            _METAL_AVAILABLE = False
            return False

        _METAL_AVAILABLE = True
        return True
    except Exception:
        _METAL_AVAILABLE = False
        return False


USE_METAL_KERNEL = is_metal_available()


# Import SwiGLU functions
from .swiglu import (
    metal_swiglu_forward,
    metal_swiglu_backward,
    is_metal_swiglu_available,
)

# Import GEGLU functions
from .geglu import (
    metal_geglu_exact_forward,
    metal_geglu_exact_backward,
    metal_geglu_approx_forward,
    metal_geglu_approx_backward,
    is_metal_geglu_available,
)

# Import RMSNorm functions
from .rms_layernorm import metal_rms_layernorm

# Import pure MLX/Metal functions (no PyTorch dependency)
from .swiglu_mlx import (
    mlx_swiglu_forward,
    mlx_swiglu_backward,
    is_mlx_swiglu_available,
)

from .geglu_mlx import (
    mlx_geglu_exact_forward,
    mlx_geglu_exact_backward,
    mlx_geglu_approx_forward,
    mlx_geglu_approx_backward,
    is_mlx_geglu_available,
)

from .rms_layernorm_mlx import (
    mlx_rms_layernorm,
    is_mlx_rms_layernorm_available,
)

__all__ = [
    "is_metal_available",
    "USE_METAL_KERNEL",
    # PyTorch-wrapped Metal kernels
    "metal_swiglu_forward",
    "metal_swiglu_backward",
    "is_metal_swiglu_available",
    "metal_geglu_exact_forward",
    "metal_geglu_exact_backward",
    "metal_geglu_approx_forward",
    "metal_geglu_approx_backward",
    "is_metal_geglu_available",
    "metal_rms_layernorm",
    # Pure MLX/Metal kernels (no PyTorch)
    "mlx_swiglu_forward",
    "mlx_swiglu_backward",
    "is_mlx_swiglu_available",
    "mlx_geglu_exact_forward",
    "mlx_geglu_exact_backward",
    "mlx_geglu_approx_forward",
    "mlx_geglu_approx_backward",
    "is_mlx_geglu_available",
    "mlx_rms_layernorm",
    "is_mlx_rms_layernorm_available",
]
