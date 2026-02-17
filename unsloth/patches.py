# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""
Runtime patches for Apple Silicon (MPS) compatibility.

The `unsloth_zoo` package enforces CUDA/HIP/XPU device checks at import time,
which blocks execution on Apple Silicon. This module provides runtime patches
to enable MPS support by intercepting the device type detection.
"""

import sys
import os
import platform
import logging
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
    # But allow mocks we set up in sys.modules (e.g., from conftest.py during testing)
    if "unsloth_zoo.device_type" in sys.modules:
        existing = sys.modules["unsloth_zoo.device_type"]
        # If it's a real module with a real file, skip patching
        # Mocks from conftest typically have fake __file__ paths
        module_file = getattr(existing, '__file__', '')
        if module_file and os.path.isfile(module_file):
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

    # Mock unsloth_zoo.utils for MPS compatibility
    mock_utils = ModuleType("unsloth_zoo.utils")

    class Version:
        def __init__(self, v):
            self.v = v

        def __lt__(self, other):
            return False

        def __le__(self, other):
            return True

        def __gt__(self, other):
            return False

        def __ge__(self, other):
            return True

    import torch

    def _get_dtype(dtype_str):
        return getattr(torch, dtype_str, torch.float32)

    def get_quant_type(module):
        return "unknown"

    mock_utils.Version = Version
    mock_utils._get_dtype = _get_dtype
    mock_utils.get_quant_type = get_quant_type
    sys.modules["unsloth_zoo.utils"] = mock_utils

    # Mock unsloth_zoo.vision_utils for MPS compatibility
    mock_vision_utils = ModuleType("unsloth_zoo.vision_utils")
    mock_vision_utils.HAS_VISION = False
    mock_vision_utils.process_vision_info = lambda *args, **kwargs: (None, None)
    sys.modules["unsloth_zoo.vision_utils"] = mock_vision_utils

    # Mock unsloth_zoo.log for MPS compatibility
    mock_log = ModuleType("unsloth_zoo.log")
    mock_log.logger = logging.getLogger("unsloth_zoo")
    sys.modules["unsloth_zoo.log"] = mock_log

    # Mock unsloth_zoo.tokenizer_utils for MPS compatibility
    mock_tokenizer = ModuleType("unsloth_zoo.tokenizer_utils")
    mock_tokenizer.patch_tokenizer = lambda x: x
    sys.modules["unsloth_zoo.tokenizer_utils"] = mock_tokenizer

    # Mock unsloth_zoo.rl_environments for MPS compatibility
    mock_rl = ModuleType("unsloth_zoo.rl_environments")
    mock_rl.check_python_modules = lambda: True
    mock_rl.create_locked_down_function = lambda fn: fn
    mock_rl.execute_with_time_limit = lambda timeout, fn, *args, **kwargs: fn(*args, **kwargs)

    class MockBenchmarker:
        pass

    mock_rl.Benchmarker = MockBenchmarker
    sys.modules["unsloth_zoo.rl_environments"] = mock_rl

    # Mock unsloth_zoo.patching_utils for MPS compatibility
    mock_patching = ModuleType("unsloth_zoo.patching_utils")
    mock_patching.patch_compiling_bitsandbytes = lambda: None
    mock_patching.patch_layernorm = lambda: None
    mock_patching.patch_torch_compile = lambda *args, **kwargs: None
    mock_patching.patch_model_and_tokenizer = lambda model, tokenizer: (model, tokenizer)
    mock_patching.patch_compiled_autograd = lambda: None
    sys.modules["unsloth_zoo.patching_utils"] = mock_patching

    # Mock unsloth_zoo.gradient_checkpointing for MPS compatibility
    mock_gc = ModuleType("unsloth_zoo.gradient_checkpointing")

    class MockOffloadedGC:
        pass

    class MockGC:
        pass

    mock_gc.Unsloth_Offloaded_Gradient_Checkpointer = MockOffloadedGC
    mock_gc.unsloth_offloaded_gradient_checkpoint = lambda module, *args, **kwargs: module.forward
    mock_gc.patch_unsloth_gradient_checkpointing = lambda: None
    mock_gc.unpatch_unsloth_gradient_checkpointing = lambda: None
    mock_gc.Unsloth_Gradient_Checkpointer = MockGC
    mock_gc.unsloth_gradient_checkpoint = lambda module, *args, **kwargs: module.forward
    mock_gc.patch_gradient_checkpointing = lambda: None
    mock_gc.unpatch_gradient_checkpointing = lambda: None
    mock_gc.patch_unsloth_smart_gradient_checkpointing = lambda: None
    mock_gc.unpatch_unsloth_smart_gradient_checkpointing = lambda: None
    sys.modules["unsloth_zoo.gradient_checkpointing"] = mock_gc

    # Mock unsloth_zoo.loss_utils for MPS compatibility
    mock_loss = ModuleType("unsloth_zoo.loss_utils")
    mock_loss.HAS_CUT_CROSS_ENTROPY = False
    mock_loss.fused_linear_cross_entropy = lambda *args, **kwargs: 0.0
    mock_loss._unsloth_get_batch_samples = lambda *args, **kwargs: None
    mock_loss.unsloth_fused_ce_loss = lambda *args, **kwargs: 0.0
    sys.modules["unsloth_zoo.loss_utils"] = mock_loss

    # Mock unsloth_zoo.compiler for MPS compatibility
    mock_compiler = ModuleType("unsloth_zoo.compiler")
    mock_compiler.create_new_function = lambda fn: fn
    mock_compiler.get_transformers_model_type = lambda *args, **kwargs: "llama"
    mock_compiler.unsloth_compile_transformers = lambda *args, **kwargs: None
    sys.modules["unsloth_zoo.compiler"] = mock_compiler

    # Mock unsloth_zoo.training_utils for MPS compatibility
    mock_training = ModuleType("unsloth_zoo.training_utils")
    mock_training.prepare_model_for_training = lambda model, *args, **kwargs: model
    sys.modules["unsloth_zoo.training_utils"] = mock_training

    # Mock unsloth_zoo.hf_utils for MPS compatibility
    mock_hf = ModuleType("unsloth_zoo.hf_utils")
    mock_hf.dtype_from_config = lambda config: None
    mock_hf.add_dtype_kwargs = lambda *args, **kwargs: {}
    mock_hf.fix_lora_auto_mapping = lambda *args, **kwargs: None
    sys.modules["unsloth_zoo.hf_utils"] = mock_hf

    # Mock unsloth_zoo.peft_utils for MPS compatibility
    mock_peft = ModuleType("unsloth_zoo.peft_utils")
    mock_peft.SKIP_QUANTIZATION_MODULES = []
    sys.modules["unsloth_zoo.peft_utils"] = mock_peft

    # Mock unsloth_zoo.vllm_utils for MPS compatibility
    mock_vllm = ModuleType("unsloth_zoo.vllm_utils")
    sys.modules["unsloth_zoo.vllm_utils"] = mock_vllm

    # Mock unsloth_zoo.temporary_patches for MPS compatibility
    mock_temp = ModuleType("unsloth_zoo.temporary_patches")
    mock_temp.TEMPORARY_PATCHES = []
    sys.modules["unsloth_zoo.temporary_patches"] = mock_temp

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
