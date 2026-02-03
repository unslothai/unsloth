# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
PyTorch ↔ MLX Tensor Bridge.

This module provides zero-copy (where possible) tensor conversion between
PyTorch and MLX using DLPack. This enables seamless interop for running
MLX-accelerated kernels on tensors from PyTorch models.

Key considerations:
1. MPS tensors must be synchronized before conversion
2. MLX uses lazy evaluation - must call mx.eval() before extracting
3. DLPack enables zero-copy sharing when memory layouts are compatible
"""

from __future__ import annotations

import functools
from contextlib import contextmanager
from typing import Callable, TypeVar, TYPE_CHECKING

from .utils import is_mlx_available, require_mlx, UnslothMLXError

if TYPE_CHECKING:
    import torch
    import mlx.core as mx

__all__ = [
    "synchronize_mps",
    "synchronize_mlx",
    "is_in_mlx_context",
    "mlx_context",
]

# Thread-local storage for context state could be used, but simple global works for now
_IN_MLX_CONTEXT = False
_MLX_HAS_FROM_DLPACK = None


def is_in_mlx_context() -> bool:
    """Check if we are currently inside an mlx_context."""
    return _IN_MLX_CONTEXT


F = TypeVar("F", bound=Callable)


def synchronize_mps() -> None:
    """
    Synchronize MPS device to ensure all pending operations complete.

    This must be called before converting MPS tensors to MLX to ensure
    the data is fully written and visible.
    """
    import torch

    if hasattr(torch, "mps") and hasattr(torch.mps, "synchronize"):
        torch.mps.synchronize()


def synchronize_mlx() -> None:
    """
    Synchronize MLX to ensure all lazy evaluations complete.

    MLX uses lazy evaluation - operations are not executed until
    their results are needed. This forces all pending operations.
    """
    if not is_mlx_available():
        return
    import mlx.core as mx

    mx.eval([])  # Evaluate empty list acts as a barrier


# Caches for lookups to save microseconds
_torch_to_dlpack = None
_mx_from_dlpack = None
_mx_array = None


@require_mlx()
def torch_to_mlx(
    tensor: "torch.Tensor",
    *,
    stream: "mx.Stream | None" = None,
) -> "mx.array":
    """
    Optimized Tensor conversion. Minimizes Python overhead.
    """
    global _torch_to_dlpack, _mx_from_dlpack, _mx_array
    if _mx_array is None:
        import torch.utils.dlpack
        import mlx.core as mx

        _torch_to_dlpack = torch.utils.dlpack.to_dlpack
        _mx_from_dlpack = getattr(mx, "from_dlpack", None)
        _mx_array = mx.array

    # 0. Check for MLX Cache (Quantized Weights)
    if hasattr(tensor, "_mlx_cache"):
        return getattr(tensor, "_mlx_cache")

    # 1. MPS Fast Path (Zero-copy)
    if tensor.device.type == "mps":
        if not _IN_MLX_CONTEXT:
            synchronize_mps()

        if not tensor.is_contiguous():
            tensor = tensor.contiguous()

        try:
            capsule = _torch_to_dlpack(tensor)
            if _mx_from_dlpack is not None:
                return _mx_from_dlpack(capsule)
            # Fallback to mx.array(capsule) if from_dlpack is missing
            return _mx_array(capsule)
        except Exception:
            # Fallback to CPU if DLPack fails
            pass

    # 2. CPU Fallback
    if not tensor.is_contiguous():
        tensor = tensor.contiguous()
    if tensor.device.type != "cpu":
        tensor = tensor.cpu()
    return _mx_array(tensor.detach().numpy())


# Caches for lookups
_torch_from_dlpack = None


@require_mlx()
def mlx_to_torch(
    array: "mx.array",
    *,
    device: str = "mps",
    dtype: "torch.dtype | None" = None,
) -> "torch.Tensor":
    """
    Optimized MLX to Torch conversion.
    """
    global _torch_from_dlpack
    if _torch_from_dlpack is None:
        import torch.utils.dlpack

        _torch_from_dlpack = torch.utils.dlpack.from_dlpack

    # Fast-path: Try zero-copy immediately
    try:
        tensor = _torch_from_dlpack(array)
        if tensor.device.type != device:
            tensor = tensor.to(device=device)
    except Exception:
        import mlx.core as mx

        mx.eval(array)
        try:
            tensor = _torch_from_dlpack(array)
            if tensor.device.type != device:
                tensor = tensor.to(device=device)
        except Exception:
            import numpy as np

            if array.dtype == mx.bfloat16:
                array = array.astype(mx.float32)
                mx.eval(array)
            try:
                tensor = torch.as_tensor(np.array(array, copy=False), device=device)
            except:
                tensor = torch.tensor(np.array(array), device=device)

    if dtype is not None and tensor.dtype != dtype:
        tensor = tensor.to(dtype=dtype)
    return tensor


@contextmanager
def mlx_context():
    """
    Context manager for safe MLX ↔ PyTorch interoperation.

    Handles synchronization on entry and exit to ensure data consistency
    when mixing PyTorch and MLX operations.

    Usage:
        with mlx_context():
            # Safe to convert tensors and run MLX ops
            arr = torch_to_mlx(tensor)
            result = mlx_operation(arr)
            tensor_out = mlx_to_torch(result)

    Raises:
        UnslothMLXError: If MLX is not available.
    """
    import mlx.core as mx

    global _IN_MLX_CONTEXT

    # Sync PyTorch MPS before entering MLX context
    if not _IN_MLX_CONTEXT:
        synchronize_mps()
    prev_state = _IN_MLX_CONTEXT
    _IN_MLX_CONTEXT = True

    try:
        yield
    finally:
        # Sync MLX before returning to PyTorch
        if not prev_state:
            mx.eval([])
            synchronize_mps()
        _IN_MLX_CONTEXT = prev_state


def with_mlx_context(func: F) -> F:
    """
    Decorator version of mlx_context.

    Wraps a function to execute within an MLX context, handling
    synchronization automatically.

    Example:
        @with_mlx_context
        def my_mlx_operation(tensor):
            arr = torch_to_mlx(tensor)
            # ... MLX operations ...
            return mlx_to_torch(result)
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with mlx_context():
            return func(*args, **kwargs)

    return wrapper  # type: ignore
