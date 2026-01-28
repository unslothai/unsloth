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
Apple Silicon (MPS) specific utilities for Unsloth.

This module provides capability detection and device information
utilities for running Unsloth on Apple Silicon Macs with the
Metal Performance Shaders (MPS) backend.
"""

import torch
import functools
import platform
import subprocess

__all__ = [
    "is_mps_available",
    "get_mps_device_info",
    "get_mps_memory_info",
    "get_mps_capabilities",
]


@functools.cache
def is_mps_available() -> bool:
    """
    Check if MPS backend is available and built.
    
    Returns:
        bool: True if MPS is available for use, False otherwise.
    """
    return (
        hasattr(torch.backends, "mps") and
        torch.backends.mps.is_available() and
        torch.backends.mps.is_built()
    )


def get_mps_device_info() -> dict:
    """
    Get Apple Silicon device information.
    
    Returns:
        dict: Device information including chip, macOS version, and PyTorch version.
    """
    if not is_mps_available():
        return {"available": False}
    
    # Get chip info from sysctl
    chip = "Unknown"
    try:
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            chip = result.stdout.strip()
    except Exception:
        chip = platform.processor() or "Apple Silicon"
    
    return {
        "available": True,
        "chip": chip,
        "mac_version": platform.mac_ver()[0],
        "pytorch_version": torch.__version__,
        "mps_built": torch.backends.mps.is_built(),
        "python_version": platform.python_version(),
    }


def get_mps_memory_info() -> dict:
    """
    Get MPS memory information.
    
    Note: Apple Silicon uses unified memory - the GPU shares RAM with the CPU.
    This is different from discrete GPUs with dedicated VRAM.
    
    Returns:
        dict: Memory information including total system memory.
    """
    if not is_mps_available():
        return {"available": False}
    
    total_memory_gb = None
    try:
        result = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            total_memory = int(result.stdout.strip())
            total_memory_gb = round(total_memory / (1024**3), 1)
    except Exception:
        pass
    
    return {
        "available": True,
        "total_system_memory_gb": total_memory_gb,
        "memory_type": "unified",
        "note": "Apple Silicon uses unified memory - GPU shares RAM with CPU",
    }


def get_mps_capabilities() -> dict:
    """
    Get MPS backend capabilities for Unsloth.
    
    Returns:
        dict: Capabilities including supported dtypes and features.
    """
    if not is_mps_available():
        return {"available": False}
    
    # Test bfloat16 support
    supports_bfloat16 = False
    try:
        test_tensor = torch.tensor([1.0], dtype=torch.bfloat16, device="mps")
        _ = test_tensor + test_tensor
        supports_bfloat16 = True
        del test_tensor
    except Exception:
        pass
    
    # Test float16 support (should always work on MPS)
    supports_float16 = False
    try:
        test_tensor = torch.tensor([1.0], dtype=torch.float16, device="mps")
        _ = test_tensor + test_tensor
        supports_float16 = True
        del test_tensor
    except Exception:
        pass
    
    return {
        "available": True,
        "supports_bfloat16": supports_bfloat16,
        "supports_float16": supports_float16,
        "supports_quantization": False,  # bitsandbytes not supported
        "supports_triton": False,  # Triton doesn't support Metal
        "recommended_dtype": "bfloat16" if supports_bfloat16 else "float16",
        "notes": [
            "Use 16-bit LoRA or full finetuning (4-bit not supported)",
            "Triton kernels replaced with PyTorch operations",
            "Unified memory allows larger batch sizes than VRAM-limited GPUs",
        ],
    }


def print_mps_info():
    """Print formatted MPS device information for debugging."""
    if not is_mps_available():
        print("MPS is not available on this system.")
        return
    
    device_info = get_mps_device_info()
    memory_info = get_mps_memory_info()
    capabilities = get_mps_capabilities()
    
    print("=" * 50)
    print("Unsloth MPS Device Information")
    print("=" * 50)
    print(f"Chip: {device_info.get('chip', 'Unknown')}")
    print(f"macOS: {device_info.get('mac_version', 'Unknown')}")
    print(f"PyTorch: {device_info.get('pytorch_version', 'Unknown')}")
    print(f"Memory: {memory_info.get('total_system_memory_gb', 'Unknown')} GB (unified)")
    print("-" * 50)
    print("Capabilities:")
    print(f"  bfloat16: {'✓' if capabilities.get('supports_bfloat16') else '✗'}")
    print(f"  float16:  {'✓' if capabilities.get('supports_float16') else '✗'}")
    print(f"  Quantization: ✗ (bitsandbytes not supported)")
    print(f"  Recommended dtype: {capabilities.get('recommended_dtype', 'float16')}")
    print("=" * 50)
