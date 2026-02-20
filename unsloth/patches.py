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
import subprocess
import json
import re
import collections
from types import ModuleType
from typing import Optional, Any, Dict, List, Callable, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum, auto

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
    patch_peft: bool = True
    verbose: bool = os.environ.get("UNSLOTH_VERBOSE", "0") == "1"

def _unsloth_dummy_fn(*args, **kwargs):
    """Dummy function for mocks that need to support inspect.getsource."""
    pass

class MacPatcher:
    """
    Comprehensive Mac/MPS compatibility patcher for unsloth.
    """
    
    def __init__(self, config: Optional[PatchConfig] = None):
        self.config = config or PatchConfig()
        self._patch_results: Dict[str, PatchResult] = {}
        self._applied = False
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

    @staticmethod
    def get_hardware_info() -> dict:
        """Get Apple Silicon hardware info WITHOUT importing other unsloth modules."""
        info = {
            "chip_name": "Apple Silicon",
            "total_memory_bytes": 16 * 1024**3,
            "usable_memory_gb": 12.0,
            "cpu_count": 8,
            "gpu_cores": 8,
        }
        try:
            # Chip Name
            res = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True)
            if res.returncode == 0: info["chip_name"] = res.stdout.strip()
            
            # Memory
            res = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True)
            if res.returncode == 0: 
                info["total_memory_bytes"] = int(res.stdout.strip())
                info["usable_memory_gb"] = (info["total_memory_bytes"] / 1024**3) * 0.75
            
            # CPUs
            res = subprocess.run(["sysctl", "-n", "hw.ncpu"], capture_output=True, text=True)
            if res.returncode == 0: info["cpu_count"] = int(res.stdout.strip())

            # GPU cores (approximate based on chip name if not directly available)
            chip = info["chip_name"]
            if "M1" in chip:
                info["gpu_cores"] = 8 if "Max" not in chip else 32
            elif "M2" in chip:
                info["gpu_cores"] = 10 if "Max" not in chip else 38
            elif "M3" in chip:
                info["gpu_cores"] = 10 if "Max" not in chip else 40
            elif "M4" in chip:
                info["gpu_cores"] = 10 if "Max" not in chip else 40
        except: pass
        return info
    
    def _create_patch_result(self, name: str, status: PatchStatus, message: str = "", error: Optional[Exception] = None) -> PatchResult:
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
            hw_info = self.get_hardware_info()
            mock = ModuleType("unsloth_zoo.device_type")
            mock.get_device_type = lambda: "mps"
            mock.is_hip = lambda: False
            mock.is_mps = lambda: True
            mock.get_device_count = lambda: 1
            mock.DEVICE_TYPE = "mps"
            mock.DEVICE_TYPE_TORCH = "mps"
            mock.DEVICE_COUNT = 1
            mock.ALLOW_PREQUANTIZED_MODELS = False
            mock.ALLOW_BITSANDBYTES = False
            mock.HAS_CUDA = False
            mock.HAS_MPS = True
            mock.TOTAL_MEMORY_GB = hw_info["total_memory_bytes"] / 1024**3
            mock.USABLE_MEMORY_GB = hw_info["usable_memory_gb"]
            sys.modules["unsloth_zoo.device_type"] = mock
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
            
            hw_info = self.get_hardware_info()
            class AppleSiliconProps:
                def __init__(self):
                    self.name = hw_info["chip_name"]
                    self.total_memory = int(hw_info["usable_memory_gb"] * 1024**3)
                    self.major = 0
                    self.minor = 0
                    self.multi_processor_count = hw_info["gpu_cores"]
                    self.is_integrated = True

            torch.cuda.get_device_properties = lambda device=None: AppleSiliconProps()
            torch.cuda.get_device_capability = lambda device=None: (0, 0)
            if not hasattr(torch.cuda, "memory"):
                torch.cuda.memory = ModuleType("torch.cuda.memory")
            torch.cuda.memory.mem_get_info = lambda device=None: (int(hw_info["usable_memory_gb"] * 1024**3), hw_info["total_memory_bytes"])
            torch.cuda.is_available = lambda: False
            torch.cuda.device_count = lambda: 0
            return self._create_patch_result(name, PatchStatus.SUCCESS, "torch.cuda mocked")
        except Exception as e:
            return self._create_patch_result(name, PatchStatus.FAILED, str(e), e)

    def _create_mock_module(self, fullname: str, loader=None):
        """Creates a robust mock module that avoids 'TypeError: module object is not callable'."""
        class MockModule(ModuleType):
            def __init__(self, name, *args, **kwargs):
                super().__init__(str(name))
                self.__path__ = []
            def __call__(self, *args, **kwargs):
                return self
            def __getattr__(self, name):
                if name.startswith("__"): return super().__getattribute__(name)
                # Ensure torch_compile and other expected callables return a dummy decorator
                if name in ("torch_compile", "unsloth_compile_transformers", "patch_torch_compile"):
                    return lambda *args, **kwargs: (lambda f: f)
                # Default to returning a mock that is itself callable
                m = MockModule(f"{self.__name__}.{name}")
                return m
            def __iter__(self):
                return iter([])
            def __contains__(self, item):
                return False
            def __len__(self):
                return 0
            def __bool__(self):
                return False
            def __mro_entries__(self, bases):
                return (object,)
            def __getitem__(self, key):
                return self
            def __setitem__(self, key, value):
                pass

        mod = MockModule(fullname)
        from importlib.machinery import ModuleSpec
        mod.__spec__ = ModuleSpec(fullname, loader, origin="mocked")
        
        # Specific attribute overrides
        if fullname == "unsloth_zoo":
            mod.__version__ = "999.0.0"
        elif "device_type" in fullname:
            hw_info = self.get_hardware_info()
            mod.get_device_type = lambda: "mps"
            mod.DEVICE_TYPE = "mps"
            mod.ALLOW_BITSANDBYTES = False
            mod.ALLOW_PREQUANTIZED_MODELS = False
        elif "vision_utils" in fullname:
            mod.HAS_VISION = False
            mod.process_vision_info = lambda *a, **k: (None, None)
        elif "utils" in fullname and not fullname.endswith("vision_utils"):
            class Version:
                def __init__(self, v): 
                    self.v = str(v)
                def __lt__(self, other): return False
                def __le__(self, other): return True
                def __gt__(self, other): return True
                def __ge__(self, other): return True
                def __eq__(self, other): return True
                def __ne__(self, other): return False
                def __str__(self): return self.v
                def __repr__(self): return f"Version('{self.v}')"
            mod.Version = Version
            mod.get_quant_type = lambda m: "unknown"
        elif fullname == "triton.backends":
            mod.backends = {}
        elif "compiler" in fullname:
            mod.get_transformers_model_type = lambda *a, **k: ["llama"]
            mod.unsloth_compile_transformers = lambda *a, **k: (["llama"], True)
        elif "rl_replacements" in fullname:
            # grpo_compute_loss_slow is appended directly as a string in rl_replacements.py
            # others are passed to inspect.getsource(), so they must be functions
            mod.RL_REPLACEMENTS = collections.defaultdict(lambda: _unsloth_dummy_fn)
            mod.RL_REPLACEMENTS["grpo_compute_loss_slow"] = "def dummy(): pass"
            mod.RL_PRE_ITEMS = collections.defaultdict(list)
        elif "patching_utils" in fullname:
            class DummyLinear: pass
            mod.Bnb_Linear4bit = DummyLinear
            mod.Peft_Linear4bit = DummyLinear
            mod.Peft_Linear = DummyLinear
            mod.patch_model_and_tokenizer = lambda m, t: (m, t)
            mod.patch_compiled_autograd = lambda: None
            mod.patch_layernorm = lambda: None
            mod.patch_torch_compile = lambda *a, **k: None
        
        return mod

    def patch_unsloth_zoo(self) -> PatchResult:
        """Thoroughly mock unsloth_zoo and CUDA-only packages."""
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
        
        # Populate sys.modules to prevent real imports
        for sub in ["device_type", "utils", "vision_utils", "log", "hf_utils", "peft_utils", "temporary_patches", "temporary_patches.common", "rl_replacements", "patching_utils"]:
            full = f"unsloth_zoo.{sub}"
            if full not in sys.modules:
                sys.modules[full] = self._create_mock_module(full)
                
        return self._create_patch_result(name, PatchStatus.SUCCESS, "unsloth_zoo, triton, and bnb mocked")

    def patch_peft(self) -> PatchResult:
        """Patch PEFT to disable bitsandbytes detection and ensure classes exist."""
        name = "peft"
        try:
            import peft.import_utils
            peft.import_utils.is_bnb_available = lambda: False
            peft.import_utils.is_bnb_4bit_available = lambda: False
            peft.import_utils.is_bnb_8bit_available = lambda: False
            
            # Ensure peft.tuners.lora has the classes Unsloth expects
            import torch.nn as nn
            try:
                import peft.tuners.lora as lora_mod
                if not hasattr(lora_mod, "Linear4bit"):
                    class Linear4bit(nn.Module): pass
                    lora_mod.Linear4bit = Linear4bit
                if not hasattr(lora_mod, "Linear"):
                    class Linear(nn.Module): pass
                    lora_mod.Linear = Linear
            except ImportError:
                pass

            return self._create_patch_result(name, PatchStatus.SUCCESS, "PEFT bnb detection disabled and classes injected")
        except:
            return self._create_patch_result(name, PatchStatus.SKIPPED, "PEFT not installed or already patched")

    def patch_checkpoint(self) -> PatchResult:
        """Patch torch.utils.checkpoint to use use_reentrant=False for MPS."""
        name = "checkpoint"
        try:
            import torch.utils.checkpoint as checkpoint_module
            if not hasattr(checkpoint_module, "_unsloth_patched"):
                original_checkpoint = checkpoint_module.checkpoint
                def _mps_checkpoint(function, *args, use_reentrant=True, **kwargs):
                    return original_checkpoint(function, *args, use_reentrant=False, **kwargs)
                checkpoint_module.checkpoint = _mps_checkpoint
                checkpoint_module._unsloth_patched = True
                return self._create_patch_result(name, PatchStatus.SUCCESS, "torch.utils.checkpoint patched for MPS")
            return self._create_patch_result(name, PatchStatus.ALREADY_APPLIED)
        except Exception as e:
            return self._create_patch_result(name, PatchStatus.FAILED, str(e), e)

    def apply(self) -> List[PatchResult]:
        """Apply all patches."""
        if self._applied: return list(self._patch_results.values())
        if not self.is_mac_with_mps(): return []
        
        results = []
        # Order matters: Mocking must come before any imports that might trigger real unsloth_zoo
        results.append(self.patch_unsloth_zoo())
        results.append(self.patch_device_type())
        results.append(self.patch_torch_cuda())
        results.append(self.patch_peft())
        results.append(self.patch_checkpoint())
        
        self._applied = True
        return results

_PATCHER = MacPatcher()

def patch_unsloth_zoo_for_mps() -> bool:
    """Legacy entry point for compatibility with __init__.py"""
    if is_patched(): return True
    results = _PATCHER.apply()
    return any(r.success for r in results)

def is_patched() -> bool:
    """Check if patches have been applied."""
    return _PATCHER.is_applied
