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

            class MockMemory:
                pass

            torch.cuda.memory = MockMemory

        # [Phase 3.1] Add truthful mocks for get_device_properties and get_device_capability
        # to satisfy external libraries and generated code without "lying" about NVIDIA.
        class AppleSiliconProps:
            def __init__(self):
                try:
                    from unsloth.kernels.mps import get_apple_hardware_info
                    hw_info = get_apple_hardware_info()
                    self.name = hw_info.get("chip_name", "Apple Silicon")
                    self.total_memory = int(hw_info.get("usable_memory_gb", 16) * 1024**3)
                except:
                    self.name = "Apple Silicon"
                    self.total_memory = 16 * 1024**3
                self.major = 0
                self.minor = 0
                self.multi_processor_count = 1
                self.is_integrated = True

        def mock_get_device_properties(device=None):
            return AppleSiliconProps()

        def mock_get_device_capability(device=None):
            return (0, 0)

        def mock_mem_get_info(device=None):
            # Use real memory info when available
            try:
                from unsloth.kernels.mps import get_apple_hardware_info
                hw_info = get_apple_hardware_info()
                total = hw_info.get("total_memory_bytes", 24 * 1024**3)
                usable = int(hw_info.get("usable_memory_gb", 16) * 1024**3)
                return (usable, total)
            except Exception:
                return (16 * 1024**3, 24 * 1024**3)

        torch.cuda.get_device_properties = mock_get_device_properties
        torch.cuda.get_device_capability = mock_get_device_capability
        torch.cuda.memory.mem_get_info = mock_mem_get_info

        pass

    # --- MOCK TRITON (THE GHOST PACKAGE) ---
    # Triton is not available on macOS, but unsloth_zoo (and others) import it deeply.
    # We use a MetaPathFinder to catch ALL triton-related imports and return fakes.
    if "triton" not in sys.modules:
        from importlib.machinery import ModuleSpec
        from importlib.abc import MetaPathFinder, Loader

        class FakeTriton(ModuleType):
            def __init__(self, name, *args, **kwargs):
                # args might be (bases, dict) if used as a metaclass
                super().__init__(str(name))
                self.__path__ = []
                self.__version__ = "3.0.0"
                # Satisfy importlib.util.find_spec
                self.__spec__ = ModuleSpec(
                    name=str(name), loader=TritonMockLoader(), origin="mocked"
                )

            def __getattr__(self, name):
                if name.startswith("__"):
                    return super().__getattribute__(name)
                # Return a sub-fake on the fly for any attribute
                full_name = f"{self.__name__}.{name}"
                if full_name not in sys.modules:
                    m = FakeTriton(full_name)
                    m.__spec__ = ModuleSpec(
                        name=full_name, loader=TritonMockLoader(), origin="mocked"
                    )
                    sys.modules[full_name] = m
                return sys.modules[full_name]

            def __call__(self, *args, **kwargs):
                return self  # Act as dummy decorator/constructor

            def __getitem__(self, key):
                return self  # Support tl.constexpr[int] or similar if needed

            def __len__(self):
                return 0

            def __iter__(self):
                return iter([])

            def __bool__(self):
                return False

            def __repr__(self):
                return f"<FakeTriton {self.__name__}>"

            # To act as a base class, we need to handle being used in a class definition
            def __class_getitem__(cls, key):
                return cls

        class TritonMockLoader(Loader):
            def create_module(self, spec):
                if spec.name not in sys.modules:
                    m = FakeTriton(spec.name)
                    m.__spec__ = spec
                    sys.modules[spec.name] = m
                return sys.modules[spec.name]

            def exec_module(self, module):
                # Specific overrides for logic checks
                if module.__name__ == "triton.backends":
                    module.backends = {}
                elif module.__name__ == "triton.backends.compiler":

                    class AttrsDescriptor:
                        def __init__(self, *args, **kwargs):
                            pass

                    module.AttrsDescriptor = AttrsDescriptor
                elif module.__name__ == "triton.language":

                    class MockTritonMeta:
                        def __repr__(self):
                            return "MockTritonMeta"

                    module.dtype = MockTritonMeta

        class TritonMockFinder(MetaPathFinder):
            def find_spec(self, fullname, path, target=None):
                if fullname == "triton" or fullname.startswith("triton."):
                    return ModuleSpec(fullname, TritonMockLoader())
                if fullname == "bitsandbytes" or fullname.startswith("bitsandbytes."):
                    return ModuleSpec(fullname, TritonMockLoader())
                return None

        # Inject the finder at the start of meta_path
        sys.meta_path.insert(0, TritonMockFinder())

        # Trigger root imports to populate sys.modules
        import triton

        try:
            import bitsandbytes
        except ImportError:
            pass

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
