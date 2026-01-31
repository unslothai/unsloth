# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""
Runtime patches for Apple Silicon (MPS) compatibility.

The `unsloth_zoo` package enforces CUDA/HIP/XPU device checks at import time,
which blocks execution on Apple Silicon. This module provides runtime patches
to enable MPS support by intercepting the device type detection.
"""

import sys
import platform
from types import ModuleType
from typing import Optional

__all__ = ["patch_unsloth_zoo_for_mps", "is_patched"]

_PATCH_APPLIED = False


def is_patched() -> bool:
    """Check if the unsloth_zoo patch has been applied."""
    return _PATCH_APPLIED


def _is_mps_available() -> bool:
    """Check if MPS backend is available."""
    if platform.system() != "Darwin":
        return False
    try:
        import torch

        return torch.backends.mps.is_available()
    except Exception:
        return False


def patch_unsloth_zoo_for_mps() -> bool:
    """
    Patch unsloth_zoo.device_type to support Apple Silicon (MPS).

    This must be called BEFORE importing unsloth_zoo.

    The patch:
    1. Pre-creates a mock device_type module in sys.modules
    2. Sets DEVICE_TYPE to "mps" for Apple Silicon compatibility
    3. Provides all expected exports from the original module

    Returns:
        True if patch was applied, False otherwise.
    """
    global _PATCH_APPLIED

    # Skip if not on macOS or MPS not available
    if not _is_mps_available():
        return False

    # Skip if already patched
    if _PATCH_APPLIED:
        return True

    # Skip if unsloth_zoo.device_type already imported (too late to patch)
    if "unsloth_zoo.device_type" in sys.modules:
        return False

    # Create mock device_type module
    mock_device_type = ModuleType("unsloth_zoo.device_type")

    # Define patched functions and constants for MPS
    def get_device_type() -> str:
        return "mps"

    def is_hip() -> bool:
        return False

    def get_device_count() -> int:
        return 1  # MPS is always single-GPU

    # Set module attributes - these match what unsloth expects
    mock_device_type.get_device_type = get_device_type
    mock_device_type.is_hip = is_hip
    mock_device_type.DEVICE_TYPE = "mps"
    mock_device_type.DEVICE_TYPE_TORCH = "mps"
    mock_device_type.DEVICE_COUNT = 1
    mock_device_type.ALLOW_PREQUANTIZED_MODELS = True  # MPS can load prequantized
    mock_device_type.ALLOW_BITSANDBYTES = False  # bitsandbytes doesn't support MPS

    # Inject into sys.modules before unsloth_zoo import
    sys.modules["unsloth_zoo.device_type"] = mock_device_type

    # --- EXTENDED MOCKING FOR TORCH.CUDA ---
    # Many parts of unsloth_zoo/trl assume CUDA exists and call memory functions.
    import torch
    
    if not torch.cuda.is_available():
        # Mock torch.cuda.memory.mem_get_info
        if not hasattr(torch.cuda, "memory"):
            class MockMemory: pass
            torch.cuda.memory = MockMemory
        
        # We MUST override even if it exists, because the original raises AssertionError
        def mock_mem_get_info(device=None):
            return (16 * 1024**3, 24 * 1024**3)
        torch.cuda.memory.mem_get_info = mock_mem_get_info

        # Mock torch.cuda.get_device_properties
        class MockProps:
            major = 8
            minor = 0
            multi_processor_count = 1
            total_global_mem = 24 * 1024**3
            name = "Apple Silicon (MPS Mock)"
        
        torch.cuda.get_device_properties = lambda device=None: MockProps()
        torch.cuda.synchronize = lambda device=None: None
        torch.cuda.empty_cache = lambda: None
        torch.cuda.set_device = lambda device: None
        torch.cuda.current_device = lambda: 0
        torch.cuda.get_device_capability = lambda device=None: (8, 0)
        torch.cuda.is_bf16_supported = lambda: True # Most modern Macs support it

    # --- MOCK TRITON ---
    # Triton is not available on macOS, but unsloth_zoo (and others) import it.
    if "triton" not in sys.modules:
        mock_triton = ModuleType("triton")
        mock_triton.__version__ = "3.0.0"
        
        # Add triton.language
        mock_triton_lang = ModuleType("triton.language")
        mock_triton.language = mock_triton_lang
        
        sys.modules["triton"] = mock_triton
        sys.modules["triton.language"] = mock_triton_lang

    _PATCH_APPLIED = True
    return True


def ensure_mps_compatibility():
    """
    Convenience function that applies MPS patch if needed.
    Call this at the very top of your script before any unsloth imports.

    Example:
        from unsloth.patches import ensure_mps_compatibility
        ensure_mps_compatibility()

        import unsloth  # Now works on Apple Silicon!
    """
    if _is_mps_available():
        patch_unsloth_zoo_for_mps()
