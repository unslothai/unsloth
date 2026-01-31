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
]

# Thread-local storage for context state could be used, but simple global works for now
_IN_MLX_CONTEXT = False

def is_in_mlx_context() -> bool:
    """Check if we are currently inside an mlx_context."""
    return _IN_MLX_CONTEXT

F = TypeVar("F", bound = Callable)


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


@require_mlx
def torch_to_mlx(
    tensor: "torch.Tensor",
    *,
    stream: "mx.Stream | None" = None,
) -> "mx.array":
    """
    Convert a PyTorch tensor to an MLX array.

    Both MPS and MLX use Apple's unified memory architecture, so data
    lives in the same physical memory. We use DLPack for zero-copy
    sharing when possible.

    Args:
        tensor: PyTorch tensor to convert.
        stream: Optional MLX stream for the operation.

    Returns:
        MLX array with the same data.

    Raises:
        UnslothMLXError: If MLX is not available.

    Example:
        >>> import torch
        >>> from unsloth.kernels.mlx import torch_to_mlx
        >>> t = torch.randn(4, 4, device="mps")
        >>> arr = torch_to_mlx(t)
    """
    import torch.utils.dlpack
    import mlx.core as mx

    # Ensure MPS writes are complete before accessing memory
    # We only sync if we are NOT already inside an mlx_context (which synced on entry)
    if tensor.device.type == "mps" and not _IN_MLX_CONTEXT:
        synchronize_mps()

    # Use DLPack for zero-copy sharing on same device (MPS -> MLX)
    if tensor.device.type == "mps":
        try:
            capsule = torch.utils.dlpack.to_dlpack(tensor)
            if hasattr(mx, "from_dlpack"):
                return mx.from_dlpack(capsule)
            else:
                return mx.array(capsule)
        except Exception:
            pass

    # Ensure contiguous memory layout for fallback
    if not tensor.is_contiguous():
        tensor = tensor.contiguous()

    # Move to CPU and convert via numpy
    if tensor.device.type != "cpu":
        tensor = tensor.cpu()

    return mx.array(tensor.numpy())


@require_mlx
def mlx_to_torch(
    array: "mx.array",
    *,
    device: str = "mps",
    dtype: "torch.dtype | None" = None,
) -> "torch.Tensor":
    """
    Convert an MLX array to a PyTorch tensor.

    Both MLX and MPS use Apple's unified memory, so this should be
    efficient. We evaluate any lazy MLX operations first, then use
    the buffer protocol for zero-copy transfer when possible.

    Args:
        array: MLX array to convert.
        device: Target PyTorch device ("mps", "cpu", etc.).
        dtype: Optional dtype override for the PyTorch tensor.

    Returns:
        PyTorch tensor with the same data.

    Raises:
        UnslothMLXError: If MLX is not available.

    Example:
        >>> import mlx.core as mx
        >>> from unsloth.kernels.mlx import mlx_to_torch
        >>> arr = mx.ones((4, 4))
        >>> t = mlx_to_torch(arr, device="mps")
    """
    import torch.utils.dlpack
    import mlx.core as mx

    # Force evaluation of lazy ops - critical before memory access
    mx.eval(array)

    # Use DLPack for zero-copy sharing (MLX -> MPS/CPU)
    try:
        # Check if MLX array has DLPack interface
        # Recent MLX arrays support __dlpack__ directly
        tensor = torch.utils.dlpack.from_dlpack(array)
        if tensor.device.type != device:
            tensor = tensor.to(device = device)
    except Exception:
        # Fallback Strategy: memoryview/numpy
        import numpy as np

        # Evaluate before access
        mx.eval(array)

        if array.dtype == mx.bfloat16:
            # Numpy doesn't like bfloat16, convert to float32
            array_f32 = array.astype(mx.float32)
            mx.eval(array_f32)
            array = array_f32

        try:
            # Direct buffer protocol access (zero-copy if already on CPU)
            tensor = torch.tensor(memoryview(array))
        except (TypeError, ValueError):
            try:
                # Fallback to copy via numpy
                tensor = torch.from_numpy(np.array(array, copy = False))
            except:
                tensor = torch.from_numpy(np.array(array))

        if tensor.device.type != device:
            tensor = tensor.to(device = device)

    # Apply dtype if specified
    if dtype is not None:
        tensor = tensor.to(dtype = dtype)

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
    synchronize_mps()
    prev_state = _IN_MLX_CONTEXT
    _IN_MLX_CONTEXT = True

    try:
        yield
    finally:
        # Sync MLX before returning to PyTorch
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
