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
Quantization Backend Abstraction for Unsloth

Provides a unified interface for different quantization backends:
- CUDA: bitsandbytes (4-bit, 8-bit NF4/FP4)
- MPS:  MLX quantization (4-bit)
- CPU:  PyTorch native (float16/32 only)
"""

from abc import ABC, abstractmethod
from typing import Optional, TYPE_CHECKING
import torch

if TYPE_CHECKING:
    from torch import nn

__all__ = [
    "QuantizationBackend",
    "BitsAndBytesBackend",
    "MLXBackend", 
    "TorchNativeBackend",
    "get_quantization_backend",
    "is_quantization_available",
]


class QuantizationBackend(ABC):
    """Abstract base class for quantization backends."""
    
    name: str = "base"
    supports_4bit: bool = False
    supports_8bit: bool = False
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if this backend is available on the current system."""
        pass
    
    @abstractmethod
    def quantize_model(self, model: "nn.Module", bits: int = 4) -> "nn.Module":
        """Quantize a model to the specified bit precision."""
        pass
    
    @abstractmethod
    def dequantize_model(self, model: "nn.Module") -> "nn.Module":
        """Dequantize a model back to full precision."""
        pass


class BitsAndBytesBackend(QuantizationBackend):
    """bitsandbytes backend for NVIDIA CUDA GPUs."""
    
    name = "bitsandbytes"
    supports_4bit = True
    supports_8bit = True
    
    _bnb = None
    _available = None
    
    def is_available(self) -> bool:
        if self._available is not None:
            return self._available
        
        try:
            import bitsandbytes as bnb
            self._bnb = bnb
            self._available = True
        except ImportError:
            self._available = False
        
        return self._available
    
    def quantize_model(self, model: "nn.Module", bits: int = 4) -> "nn.Module":
        if not self.is_available():
            raise ImportError("bitsandbytes is not available")
        # BnB quantization is typically done at load time via BitsAndBytesConfig
        # This is a placeholder for explicit quantization if needed
        raise NotImplementedError(
            "bitsandbytes quantization is done at model load time. "
            "Use BitsAndBytesConfig with transformers instead."
        )
    
    def dequantize_model(self, model: "nn.Module") -> "nn.Module":
        if not self.is_available():
            raise ImportError("bitsandbytes is not available")
        # Import the fast_dequantize from kernels
        from .kernels import fast_dequantize
        # This would need to iterate over model and dequantize Linear4bit layers
        return model


class MLXBackend(QuantizationBackend):
    """MLX backend for Apple Silicon (MPS)."""
    
    name = "mlx"
    supports_4bit = True
    supports_8bit = False
    
    _mlx = None
    _available = None
    
    def is_available(self) -> bool:
        if self._available is not None:
            return self._available
        
        try:
            import mlx.core
            self._mlx = mlx
            self._available = True
        except ImportError:
            self._available = False
        
        return self._available
    
    def quantize_model(self, model: "nn.Module", bits: int = 4) -> "nn.Module":
        if not self.is_available():
            raise ImportError("MLX is not available. Install via: pip install mlx")
        
        if bits != 4:
            raise ValueError(f"MLX backend only supports 4-bit quantization, got {bits}")
        
        from .kernels.mlx.utils import fast_quantize
        return fast_quantize(model)
    
    def dequantize_model(self, model: "nn.Module") -> "nn.Module":
        # MLX models can be dequantized by loading without quantization
        raise NotImplementedError("MLX dequantization not yet implemented")


class TorchNativeBackend(QuantizationBackend):
    """PyTorch native backend - no quantization, just dtype conversion."""
    
    name = "torch_native"
    supports_4bit = False
    supports_8bit = False
    
    def is_available(self) -> bool:
        return True  # Always available
    
    def quantize_model(self, model: "nn.Module", bits: int = 4) -> "nn.Module":
        raise ValueError(
            f"PyTorch native backend does not support {bits}-bit quantization. "
            "Use float16 or bfloat16 dtype instead."
        )
    
    def dequantize_model(self, model: "nn.Module") -> "nn.Module":
        return model  # No-op for native backend


# Backend instances (lazy initialization)
_backends = {
    "bitsandbytes": BitsAndBytesBackend(),
    "mlx": MLXBackend(),
    "torch_native": TorchNativeBackend(),
}


def get_quantization_backend(device_type: Optional[str] = None) -> QuantizationBackend:
    """
    Get the appropriate quantization backend for the given device type.
    
    Args:
        device_type: One of "cuda", "mps", "cpu", or None (auto-detect)
    
    Returns:
        The appropriate QuantizationBackend instance
    """
    if device_type is None:
        # Auto-detect from DEVICE_TYPE
        from .device_type import DEVICE_TYPE
        device_type = DEVICE_TYPE
    
    if device_type == "cuda":
        backend = _backends["bitsandbytes"]
        if backend.is_available():
            return backend
        # Fall back to native
        return _backends["torch_native"]
    
    elif device_type == "mps":
        backend = _backends["mlx"]
        if backend.is_available():
            return backend
        # Fall back to native
        return _backends["torch_native"]
    
    else:  # cpu or other
        return _backends["torch_native"]


def is_quantization_available(bits: int = 4, device_type: Optional[str] = None) -> bool:
    """
    Check if quantization is available for the given bit precision.
    
    Args:
        bits: The bit precision (4 or 8)
        device_type: The device type, or None to auto-detect
    
    Returns:
        True if quantization is available
    """
    backend = get_quantization_backend(device_type)
    
    if bits == 4:
        return backend.supports_4bit and backend.is_available()
    elif bits == 8:
        return backend.supports_8bit and backend.is_available()
    else:
        return False
