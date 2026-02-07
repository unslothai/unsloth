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
Unified Device Utilities for Unsloth

Provides a consistent API for device properties across CUDA, MPS, and XPU backends.
This replaces direct torch.cuda.get_device_properties() calls throughout the codebase.
"""

from types import SimpleNamespace
from functools import lru_cache
import torch

from unsloth_zoo.device_type import DEVICE_TYPE

__all__ = [
    "get_device_properties",
    "get_device_name",
    "get_total_memory",
    "get_available_memory",
    "get_memory_allocated",
    "get_current_memory_usage",
]


@lru_cache(maxsize=1)
def get_device_properties() -> SimpleNamespace:
    """
    Get device properties for the current backend.
    
    Returns a SimpleNamespace with consistent attributes:
        - name: str (device/chip name)
        - total_memory: int (bytes)
        - major: int (compute capability major, 0 for non-CUDA)
        - minor: int (compute capability minor, 0 for non-CUDA)
        - multi_processor_count: int (SM count or GPU core estimate)
    """
    if DEVICE_TYPE == "cuda":
        props = torch.cuda.get_device_properties(0)
        return SimpleNamespace(
            name=props.name,
            total_memory=props.total_memory,
            major=props.major,
            minor=props.minor,
            multi_processor_count=props.multi_processor_count,
        )
    
    elif DEVICE_TYPE == "mps":
        from unsloth.kernels.mps import get_apple_hardware_info
        hw = get_apple_hardware_info()
        return SimpleNamespace(
            name=hw.get("chip_name", "Apple Silicon"),
            total_memory=hw.get("total_memory_bytes", 16 * 1024**3),
            major=0,  # Not applicable for Apple Silicon
            minor=0,
            multi_processor_count=hw.get("gpu_cores", 8),
        )
    
    elif DEVICE_TYPE == "xpu":
        props = torch.xpu.get_device_properties(0)
        return SimpleNamespace(
            name=props.name,
            total_memory=props.total_memory,
            major=0,
            minor=0,
            multi_processor_count=getattr(props, "gpu_subslice_count", 1),
        )
    
    else:  # CPU fallback
        import psutil
        return SimpleNamespace(
            name="CPU",
            total_memory=psutil.virtual_memory().total,
            major=0,
            minor=0,
            multi_processor_count=1,
        )


def get_device_name() -> str:
    """Get human-readable device name."""
    return get_device_properties().name


def get_total_memory() -> int:
    """Get total device memory in bytes."""
    return get_device_properties().total_memory


def get_available_memory() -> int:
    """
    Get currently available device memory in bytes.
    
    Note: This is an approximation on some backends.
    """
    if DEVICE_TYPE == "cuda":
        return torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)
    
    elif DEVICE_TYPE == "mps":
        # MPS unified memory - use system available memory capped by usable limit
        from unsloth.kernels.mps import get_apple_hardware_info
        hw = get_apple_hardware_info()
        import psutil
        available = psutil.virtual_memory().available
        
        # We should not exceed the "usable" memory limit for ML tasks
        total_usable = hw.get("usable_memory_gb", 0.0) * (1024**3)
        if total_usable > 0:
            # Allocated by MPS
            allocated = torch.mps.current_allocated_memory()
            # Approximation of what's left within the usable budget
            usable_remaining = max(0, total_usable - allocated)
            return min(available, int(usable_remaining))
        
        return available
    
    elif DEVICE_TYPE == "xpu":
        return torch.xpu.get_device_properties(0).total_memory - torch.xpu.memory_allocated(0)
    
    else:
        import psutil
        return psutil.virtual_memory().available


def get_memory_allocated() -> int:
    """
    Get current memory allocated on the device in bytes.
    """
    if DEVICE_TYPE == "cuda":
        return torch.cuda.memory_allocated(0)
    
    elif DEVICE_TYPE == "mps":
        # MPS unified memory - return current MPS allocation
        return torch.mps.current_allocated_memory()
    
    elif DEVICE_TYPE == "xpu":
        return torch.xpu.memory_allocated(0)
    
    else:
        import psutil
        vm = psutil.virtual_memory()
        return vm.total - vm.available


def get_current_memory_usage() -> int:
    """Alias for get_memory_allocated() for easier migration."""
    return get_memory_allocated()
