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
    "get_apple_hardware_info",
    "USE_MPS_FALLBACK",
    # LoRA & Linear fallbacks
    "mps_gemv",
    "mps_linear_forward",
    "mps_matmul_lora",
    "mps_apply_lora_mlp_swiglu",
    "mps_apply_lora_mlp_geglu_exact",
    "mps_apply_lora_mlp_geglu_approx",
    "mps_apply_lora_qkv",
    "mps_apply_lora_o",
    # MoE operations
    "dispatch_grouped_gemm",
    "grouped_gemm_mps",
    # Dispatch functions
    "_is_metal_available",
    "_is_mlx_available",
    "dispatch_rms_layernorm",
    "dispatch_rope_embedding",
    "dispatch_swiglu_fg",
    "dispatch_layernorm",
    "dispatch_cross_entropy_loss",
    "dispatch_swiglu_backward",
    "dispatch_geglu_exact_forward",
    "dispatch_geglu_exact_backward",
    "dispatch_geglu_approx_forward",
    "dispatch_geglu_approx_backward",
    "dispatch_matmul_lora",
    "dispatch_gemv",
    "dispatch_lora_mlp_swiglu",
    "dispatch_lora_mlp_geglu_exact",
    "dispatch_lora_mlp_geglu_approx",
    "dispatch_lora_qkv",
    "dispatch_lora_o",
]


# Global flag to control MPS fallback usage
# Can be disabled for benchmarking or when Metal kernels are available
# NOTE: When gradient checkpointing is enabled on MPS, this should be set to False
# to avoid 'element 0 of tensors does not require grad' errors. The custom autograd
# functions in the MPS fallback have compatibility issues with gradient checkpointing.
USE_MPS_FALLBACK = True


@functools.lru_cache(maxsize=1)
def get_apple_hardware_info() -> dict:
    """
    Get detailed Apple Silicon hardware information using sysctl/system_profiler.
    
    This is the authoritative source of truth for Apple Silicon hardware detection.
    Uses native macOS utilities to get real chip info instead of mocking NVIDIA.
    
    Returns:
        dict with keys:
        - is_apple_silicon: bool - True if running on Apple Silicon
        - chip_name: str - Full chip name (e.g., "Apple M2 Pro")
        - chip_family: str - Base family ("M1", "M2", "M3")
        - chip_variant: str - Variant ("base", "Pro", "Max", "Ultra")
        - total_memory_bytes: int - Total unified memory in bytes
        - total_memory_gb: float - Total unified memory in GB
        - usable_memory_gb: float - Memory available for ML (~70-90% based on chip)
        - cpu_cores_total: int - Total CPU cores
        - cpu_cores_performance: int - P-core count (if detectable)
        - cpu_cores_efficiency: int - E-core count (if detectable)
        - gpu_cores: int - GPU core count (if detectable)
    """
    import json
    import re
    
    result = {
        "is_apple_silicon": False,
        "chip_name": "Unknown",
        "chip_family": "Unknown",
        "chip_variant": "base",
        "total_memory_bytes": 0,
        "total_memory_gb": 0.0,
        "usable_memory_gb": 0.0,
        "cpu_cores_total": 0,
        "cpu_cores_performance": 0,
        "cpu_cores_efficiency": 0,
        "gpu_cores": 0,
    }
    
    # Step 1: Check if this is Apple Silicon (ARM64)
    try:
        check = subprocess.run(
            ["sysctl", "-n", "hw.optional.arm64"],
            capture_output=True, text=True, timeout=5
        )
        result["is_apple_silicon"] = check.returncode == 0 and check.stdout.strip() == "1"
    except Exception:
        # Fallback: check platform
        result["is_apple_silicon"] = platform.processor() == "arm"
    
    if not result["is_apple_silicon"]:
        return result
    
    # Step 2: Get chip name from system_profiler (JSON format for reliable parsing)
    try:
        sp = subprocess.run(
            ["system_profiler", "SPHardwareDataType", "-json"],
            capture_output=True, text=True, timeout=10
        )
        if sp.returncode == 0:
            data = json.loads(sp.stdout)
            hw_items = data.get("SPHardwareDataType", [{}])
            if hw_items:
                hw = hw_items[0]
                result["chip_name"] = hw.get("chip_type", "Apple Silicon")
                result["cpu_cores_total"] = int(hw.get("number_processors", "0").split()[0]) if hw.get("number_processors") else 0
    except Exception:
        pass
    
    # Step 3: Parse chip family and variant from chip_name
    chip = result["chip_name"]
    family_match = re.search(r"(M[1-9])", chip)
    if family_match:
        result["chip_family"] = family_match.group(1)
    
    if "Ultra" in chip:
        result["chip_variant"] = "Ultra"
    elif "Max" in chip:
        result["chip_variant"] = "Max"
    elif "Pro" in chip:
        result["chip_variant"] = "Pro"
    else:
        result["chip_variant"] = "base"
    
    # Step 4: Get total memory
    try:
        mem = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            capture_output=True, text=True, timeout=5
        )
        if mem.returncode == 0:
            result["total_memory_bytes"] = int(mem.stdout.strip())
            result["total_memory_gb"] = round(result["total_memory_bytes"] / (1024**3), 1)
    except Exception:
        pass
    
    # Step 5: Calculate usable memory based on chip tier
    # Conservative estimates: base chips run hotter and need more headroom
    usable_percent = {
        "base": 0.70,   # 70% for base M1/M2/M3
        "Pro": 0.80,    # 80% for Pro variants
        "Max": 0.85,    # 85% for Max variants
        "Ultra": 0.88,  # 88% for Ultra variants
    }
    pct = usable_percent.get(result["chip_variant"], 0.70)
    result["usable_memory_gb"] = round(result["total_memory_gb"] * pct, 1)
    
    # Step 6: Get core counts (P-cores and E-cores)
    try:
        perf = subprocess.run(
            ["sysctl", "-n", "hw.perflevel0.logicalcpu"],
            capture_output=True, text=True, timeout=5
        )
        if perf.returncode == 0:
            result["cpu_cores_performance"] = int(perf.stdout.strip())
    except Exception:
        pass
    
    try:
        eff = subprocess.run(
            ["sysctl", "-n", "hw.perflevel1.logicalcpu"],
            capture_output=True, text=True, timeout=5
        )
        if eff.returncode == 0:
            result["cpu_cores_efficiency"] = int(eff.stdout.strip())
    except Exception:
        pass
    
    # Step 7: Estimate GPU cores based on chip variant (Apple doesn't expose this via sysctl)
    gpu_core_estimates = {
        ("M1", "base"): 8,    ("M1", "Pro"): 16,   ("M1", "Max"): 32,   ("M1", "Ultra"): 64,
        ("M2", "base"): 10,   ("M2", "Pro"): 19,   ("M2", "Max"): 38,   ("M2", "Ultra"): 76,
        ("M3", "base"): 10,   ("M3", "Pro"): 18,   ("M3", "Max"): 40,   ("M3", "Ultra"): 80,
        ("M4", "base"): 10,   ("M4", "Pro"): 20,   ("M4", "Max"): 40,   ("M4", "Ultra"): 80,
    }
    result["gpu_cores"] = gpu_core_estimates.get(
        (result["chip_family"], result["chip_variant"]), 
        8  # Conservative default
    )
    
    return result


@functools.cache
def is_mps_available() -> bool:
    """
    Check if MPS backend is available and built.

    Returns:
        bool: True if MPS is available for use, False otherwise.
    """
    return (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
        and torch.backends.mps.is_built()
    )


def get_mps_device_info() -> dict:
    """
    Get Apple Silicon device information.

    Returns:
        dict: Device information including chip, macOS version, and PyTorch version.
    """
    if not is_mps_available():
        return {"available": False}

    # Use the new comprehensive hardware info
    hw_info = get_apple_hardware_info()
    chip = hw_info.get("chip_name", "Apple Silicon")

    return {
        "available": True,
        "chip": chip,
        "chip_family": hw_info.get("chip_family", "Unknown"),
        "chip_variant": hw_info.get("chip_variant", "base"),
        "is_apple_silicon": hw_info.get("is_apple_silicon", True),
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
        dict: Memory information including total and usable memory.
    """
    if not is_mps_available():
        return {"available": False}

    # Use the comprehensive hardware info
    hw_info = get_apple_hardware_info()

    return {
        "available": True,
        "total_memory_bytes": hw_info.get("total_memory_bytes", 0),
        "total_memory_gb": hw_info.get("total_memory_gb", 0.0),
        "usable_memory_gb": hw_info.get("usable_memory_gb", 0.0),
        "total_system_memory_gb": hw_info.get("total_memory_gb", 0.0),  # Compat alias
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
    print(
        f"Memory: {memory_info.get('total_system_memory_gb', 'Unknown')} GB (unified)"
    )
    print("-" * 50)
    print("Capabilities:")
    print(f"  bfloat16: {'✓' if capabilities.get('supports_bfloat16') else '✗'}")
    print(f"  float16:  {'✓' if capabilities.get('supports_float16') else '✗'}")
    print(f"  Quantization: ✗ (bitsandbytes not supported)")
    print(f"  Recommended dtype: {capabilities.get('recommended_dtype', 'float16')}")
    print("=" * 50)


from .linear import (
    mps_gemv,
    mps_linear_forward,
)
from .fast_lora import (
    mps_matmul_lora,
    mps_apply_lora_mlp_swiglu,
    mps_apply_lora_mlp_geglu_exact,
    mps_apply_lora_mlp_geglu_approx,
    mps_apply_lora_qkv,
    mps_apply_lora_o,
)
from .moe import (
    dispatch_grouped_gemm,
    grouped_gemm_mps,
)
from .dispatch import (
    _is_metal_available,
    _is_mlx_available,
    dispatch_rms_layernorm,
    dispatch_rope_embedding,
    dispatch_swiglu_fg,
    dispatch_layernorm,
    dispatch_cross_entropy_loss,
    dispatch_swiglu_backward,
    dispatch_geglu_exact_forward,
    dispatch_geglu_exact_backward,
    dispatch_geglu_approx_forward,
    dispatch_geglu_approx_backward,
    dispatch_matmul_lora,
    dispatch_gemv,
    dispatch_lora_mlp_swiglu,
    dispatch_lora_mlp_geglu_exact,
    dispatch_lora_mlp_geglu_approx,
    dispatch_lora_qkv,
    dispatch_lora_o,
)
