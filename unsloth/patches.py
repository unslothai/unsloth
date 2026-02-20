# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""
Unsloth Patcher for Apple Silicon (macOS/MPS) Compatibility
===========================================================

This module provides comprehensive patching for unsloth_zoo and related
dependencies to work correctly on Apple Silicon with MPS.
"""

from __future__ import annotations

import sys
import os
import platform
import logging
import warnings
from types import ModuleType
from typing import Optional, Any, Dict, List, Callable, Union, Tuple
from dataclasses import dataclass, field
from contextlib import contextmanager
from enum import Enum, auto
from functools import wraps

# Configure logging
logger = logging.getLogger("unsloth.patches")

class PatchStatus(Enum):
    """Status of a patch operation."""
    SUCCESS = auto()
    SKIPPED = auto()
    FAILED = auto()
    ALREADY_APPLIED = auto()
    NOT_NEEDED = auto()
    ERROR = auto()

@dataclass
class PatchResult:
    """Result of a patch operation."""
    name: str
    status: PatchStatus
    message: str = ""
    error: Optional[Exception] = None
    
    @property
    def success(self) -> bool:
        """Check if patch succeeded."""
        return self.status in (PatchStatus.SUCCESS, PatchStatus.ALREADY_APPLIED, PatchStatus.NOT_NEEDED)
    
    def __repr__(self) -> str:
        return f"PatchResult({self.name}: {self.status.name}{' - ' + self.message if self.message else ''})"

@dataclass
class PatchConfig:
    """Configuration for Mac compatibility patches."""
    enable_logging: bool = os.environ.get("UNSLOTH_ENABLE_LOGGING", "0") == "1"
    mock_bitsandbytes: bool = True
    mock_triton: bool = True
    mock_torch_cuda: bool = True
    patch_device_type: bool = True
    patch_fused_losses: bool = True
    patch_peft: bool = True
    patch_compilers: bool = True
    verbose: bool = os.environ.get("UNSLOTH_VERBOSE", "0") == "1"

class MacPatcher:
    """
    Comprehensive Mac/MPS compatibility patcher for unsloth.
    """
    
    def __init__(self, config: Optional[PatchConfig] = None):
        self.config = config or PatchConfig()
        self._patch_results: Dict[str, PatchResult] = {}
        self._applied = False
        self._original_modules: Dict[str, Any] = {}
        self._mock_finders: List[Any] = []
        
    @property
    def is_applied(self) -> bool:
        """Check if patches have been applied."""
        return self._applied
    
    def _log(self, level: int, message: str) -> None:
        """Log a message if logging is enabled."""
        if self.config.enable_logging:
            logger.log(level, message)
        if self.config.verbose and level >= logging.INFO:
            print(f"Unsloth Patches: {message}")
    
    @staticmethod
    def is_mac_with_mps() -> bool:
        """Check if running on macOS with MPS available."""
        if platform.system() != "Darwin":
            return False
        if os.environ.get("UNSLOTH_FORCE_MPS", "0") == "1":
            return True
        try:
            import torch
            return torch.backends.mps.is_available()
        except Exception:
            return False
    
    def _create_patch_result(
        self,
        name: str,
        status: PatchStatus,
        message: str = "",
        error: Optional[Exception] = None
    ) -> PatchResult:
        """Create and store a patch result."""
        result = PatchResult(name, status, message, error)
        self._patch_results[name] = result
        
        if status == PatchStatus.SUCCESS:
            self._log(logging.INFO, f"✓ {name}: {message or 'patched successfully'}")
        elif status == PatchStatus.FAILED:
            self._log(logging.ERROR, f"✗ {name}: {message}")
        return result

    def patch_device_type(self) -> PatchResult:
        """Patch unsloth_zoo.device_type to support MPS."""
        name = "device_type"
        if "unsloth_zoo.device_type" in sys.modules:
            return self._create_patch_result(name, PatchStatus.SKIPPED, "already imported")
        
        try:
            from unsloth.kernels.mps import get_apple_hardware_info
            hw_info = get_apple_hardware_info()
            
            mock_device_type = ModuleType("unsloth_zoo.device_type")
            mock_device_type.get_device_type = lambda: "mps"
            mock_device_type.is_hip = lambda: False
            mock_device_type.is_mps = lambda: True
            mock_device_type.get_device_count = lambda: 1
            mock_device_type.DEVICE_TYPE = "mps"
            mock_device_type.DEVICE_TYPE_TORCH = "mps"
            mock_device_type.DEVICE_COUNT = 1
            mock_device_type.ALLOW_PREQUANTIZED_MODELS = False
            mock_device_type.ALLOW_BITSANDBYTES = False
            mock_device_type.HAS_CUDA = False
            mock_device_type.HAS_MPS = True
            mock_device_type.HAS_HIP = False
            mock_device_type.HAS_XPU = False
            mock_device_type.TOTAL_MEMORY_GB = hw_info.get("total_memory_gb", 16.0)
            mock_device_type.USABLE_MEMORY_GB = hw_info.get("usable_memory_gb", 12.0)
            
            sys.modules["unsloth_zoo.device_type"] = mock_device_type
            return self._create_patch_result(name, PatchStatus.SUCCESS, "MPS device type configured")
        except Exception as e:
            return self._create_patch_result(name, PatchStatus.FAILED, str(e), e)

    def patch_torch_cuda(self) -> PatchResult:
        """Mock torch.cuda functions for MPS compatibility."""
        name = "torch_cuda"
        try:
            import torch
            if torch.cuda.is_available():
                return self._create_patch_result(name, PatchStatus.SKIPPED, "CUDA is available")
            
            from unsloth.kernels.mps import get_apple_hardware_info
            hw_info = get_apple_hardware_info()
            
            class AppleSiliconProps:
                def __init__(self):
                    self.name = hw_info.get("chip_name", "Apple Silicon")
                    self.total_memory = int(hw_info.get("usable_memory_gb", 16) * 1024**3)
                    self.major = 0
                    self.minor = 0
                    self.multi_processor_count = hw_info.get("cpu_count", 8)
                    self.is_integrated = True

            torch.cuda.get_device_properties = lambda device=None: AppleSiliconProps()
            torch.cuda.get_device_capability = lambda device=None: (0, 0)
            
            if not hasattr(torch.cuda, "memory"):
                torch.cuda.memory = ModuleType("torch.cuda.memory")
            
            torch.cuda.memory.mem_get_info = lambda device=None: (
                int(hw_info.get("usable_memory_gb", 16) * 1024**3),
                hw_info.get("total_memory_bytes", 24 * 1024**3)
            )
            torch.cuda.is_available = lambda: False
            torch.cuda.device_count = lambda: 0
            
            return self._create_patch_result(name, PatchStatus.SUCCESS, "torch.cuda mocked")
        except Exception as e:
            return self._create_patch_result(name, PatchStatus.FAILED, str(e), e)

    def _create_mock_module(self, fullname: str, loader=None):
        mod = ModuleType(fullname)
        mod.__path__ = []
        from importlib.machinery import ModuleSpec
        mod.__spec__ = ModuleSpec(fullname, loader, origin="mocked")
        
        if fullname == "unsloth_zoo":
            mod.__version__ = "999.0.0"
        elif fullname == "unsloth_zoo.device_type":
            from unsloth.kernels.mps import get_apple_hardware_info
            hw_info = get_apple_hardware_info()
            mod.get_device_type = lambda: "mps"
            mod.is_hip = lambda: False
            mod.is_mps = lambda: True
            mod.DEVICE_TYPE = "mps"
            mod.DEVICE_COUNT = 1
            mod.ALLOW_PREQUANTIZED_MODELS = False
            mod.ALLOW_BITSANDBYTES = False
            mod.TOTAL_MEMORY_GB = hw_info.get("total_memory_gb", 16.0)
        elif fullname == "unsloth_zoo.utils":
            class Version:
                def __init__(self, v): self.v = v
                def __ge__(self, other): return True
                def __le__(self, other): return True
            mod.Version = Version
            mod._get_dtype = lambda d: getattr(__import__('torch'), d, __import__('torch').float32)
            mod.get_quant_type = lambda m: "unknown"
        elif fullname == "unsloth_zoo.vision_utils":
            mod.HAS_VISION = False
            mod.process_vision_info = lambda *a, **k: (None, None)
        elif fullname == "unsloth_zoo.log":
            mod.logger = logging.getLogger("unsloth_zoo")
        
        return mod

    def patch_unsloth_zoo(self) -> PatchResult:
        """Thoroughly mock unsloth_zoo submodules."""
        name = "unsloth_zoo"
        from importlib.abc import MetaPathFinder, Loader
        from importlib.machinery import ModuleSpec
        
        patcher_self = self
        class MockLoader(Loader):
            def create_module(self, spec):
                if spec.name in sys.modules: return sys.modules[spec.name]
                return patcher_self._create_mock_module(spec.name, self)
            def exec_module(self, module): pass

        class MockFinder(MetaPathFinder):
            def find_spec(self, fullname, path, target=None):
                if fullname == "unsloth_zoo" or fullname.startswith("unsloth_zoo."):
                    if fullname in sys.modules: return None
                    return ModuleSpec(fullname, MockLoader(), origin="mocked")
                if fullname in ["triton", "bitsandbytes"] or fullname.startswith(("triton.", "bitsandbytes.")):
                    if fullname in sys.modules: return None
                    return ModuleSpec(fullname, MockLoader(), origin="mocked")
                return None

        finder = MockFinder()
        sys.meta_path.insert(0, finder)
        self._mock_finders.append(finder)
        
        # Pre-populate some critical ones
        for sub in ["device_type", "utils", "vision_utils", "log", "hf_utils", "peft_utils"]:
            full = f"unsloth_zoo.{sub}"
            if full not in sys.modules:
                sys.modules[full] = self._create_mock_module(full)
                
        return self._create_patch_result(name, PatchStatus.SUCCESS, "unsloth_zoo and dependencies mocked")

    def patch_peft(self) -> PatchResult:
        """Patch PEFT to disable bitsandbytes detection."""
        name = "peft"
        try:
            import peft.import_utils
            peft.import_utils.is_bnb_available = lambda: False
            peft.import_utils.is_bnb_4bit_available = lambda: False
            peft.import_utils.is_bnb_8bit_available = lambda: False
            return self._create_patch_result(name, PatchStatus.SUCCESS, "PEFT bnb detection disabled")
        except ImportError:
            return self._create_patch_result(name, PatchStatus.SKIPPED, "PEFT not installed")

    def apply(self) -> List[PatchResult]:
        """Apply all patches."""
        if self._applied: return list(self._patch_results.values())
        if not self.is_mac_with_mps(): return []
        
        results = []
        results.append(self.patch_device_type())
        results.append(self.patch_torch_cuda())
        results.append(self.patch_unsloth_zoo())
        results.append(self.patch_peft())
        
        self._applied = True
        return results

_PATCHER = MacPatcher()

def patch_unsloth_zoo_for_mps() -> bool:
    """Legacy entry point for compatibility with __init__.py"""
    results = _PATCHER.apply()
    return any(r.success for r in results)

def is_patched() -> bool:
    """Check if patches have been applied."""
    return _PATCHER.is_applied
