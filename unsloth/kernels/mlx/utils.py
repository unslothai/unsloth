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
MLX utilities for Unsloth.

This module provides utilities for integrating Apple's MLX framework
on Apple Silicon Macs. MLX provides optimized operations that can
outperform PyTorch MPS for certain workloads.

Note: MLX is optional. If not installed, is_mlx_available() returns False
and UnslothMLXError is raised if MLX functions are called.
"""

import sys
import functools
from typing import Callable, TypeVar

__all__ = [
    "is_mlx_available",
    "get_mlx_version",
    "UnslothMLXError",
    "require_mlx",
]

# Type variable for decorator
F = TypeVar("F", bound=Callable)


class UnslothMLXError(RuntimeError):
    """
    Exception raised when MLX operations are attempted without MLX installed.
    
    This provides a clear error message guiding users to install MLX
    when they attempt to use MLX-accelerated features.
    """
    
    def __init__(self, message: str = None):
        if message is None:
            message = (
                "MLX is not available. To use MLX-accelerated features, "
                "install MLX with: pip install 'unsloth[apple]' "
                "or pip install mlx>=0.6.0. "
                "Note: MLX only works on Apple Silicon Macs."
            )
        super().__init__(message)


@functools.cache
def is_mlx_available() -> bool:
    """
    Check if MLX framework is available.
    
    This checks:
    1. Running on macOS (Darwin)
    2. MLX package can be imported
    
    Returns:
        bool: True if MLX is available for use, False otherwise.
    
    Note:
        Result is cached for performance. The first call performs the
        actual check; subsequent calls return the cached result.
    """
    # Must be on macOS
    if sys.platform != "darwin":
        return False
    
    # Try to import MLX
    try:
        import mlx.core  # noqa: F401
        return True
    except ImportError:
        return False


def get_mlx_version() -> str | None:
    """
    Get the installed MLX version.
    
    Returns:
        str | None: Version string if MLX is installed, None otherwise.
    """
    if not is_mlx_available():
        return None
    
    try:
        import mlx
        return getattr(mlx, "__version__", "unknown")
    except Exception:
        return None


def require_mlx(func: F) -> F:
    """
    Decorator that ensures MLX is available before executing a function.
    
    Raises:
        UnslothMLXError: If MLX is not available.
    
    Example:
        @require_mlx
        def mlx_accelerated_operation(tensor):
            import mlx.core as mx
            # ... MLX operations
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if not is_mlx_available():
            raise UnslothMLXError(
                f"Cannot use '{func.__name__}' - MLX is not available. "
                "Install with: pip install 'unsloth[apple]'"
            )
        return func(*args, **kwargs)
    
    return wrapper  # type: ignore
