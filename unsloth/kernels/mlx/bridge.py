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
    "torch_to_mlx",
    "mlx_to_torch",
    "mlx_context",
    "synchronize_mps",
    "synchronize_mlx",
]

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
    import torch
    import mlx.core as mx
    import numpy as np
    
    # Ensure MPS writes are complete before accessing memory
    if tensor.device.type == "mps":
        synchronize_mps()
    
    # Ensure contiguous memory layout
    if not tensor.is_contiguous():
        tensor = tensor.contiguous()
    
    # Try direct DLPack path (zero-copy unified memory)
    try:
        # For MPS tensors on unified memory, this should be zero-copy
        # MLX and MPS share the same Metal unified memory
        return mx.array(np.from_dlpack(tensor.detach()), copy=False)
    except (TypeError, RuntimeError):
        # Fallback: go through numpy (still fast on unified memory)
        if tensor.device.type != "cpu":
            tensor = tensor.cpu()
        return mx.array(tensor.numpy(), copy=False)


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
    DLPack for zero-copy transfer when possible.
    
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
    import torch
    import mlx.core as mx
    import numpy as np
    
    # Force evaluation of lazy ops - critical before memory access
    mx.eval(array)
    
    # Handle bfloat16 specially (numpy doesn't support it)
    if array.dtype == mx.bfloat16:
        # Convert to float32 in MLX first
        array = array.astype(mx.float32)
        mx.eval(array)
    
    # Try direct DLPack path (zero-copy unified memory)
    try:
        tensor = torch.from_numpy(np.array(array, copy=False))
    except (TypeError, RuntimeError):
        # Fallback for edge cases
        tensor = torch.from_numpy(np.array(array))
    
    # Apply dtype if specified
    if dtype is not None:
        tensor = tensor.to(dtype=dtype)
    
    # Move to target device (fast on unified memory)
    return tensor.to(device=device)


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
    if not is_mlx_available():
        raise UnslothMLXError("mlx_context requires MLX to be installed")
    
    import mlx.core as mx
    
    # Sync PyTorch MPS before entering MLX context
    synchronize_mps()
    
    try:
        yield
    finally:
        # Sync MLX before returning to PyTorch
        mx.eval([])
        synchronize_mps()


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
