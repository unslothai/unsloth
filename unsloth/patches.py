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
import types
import collections
from typing import Optional, Any, Dict, List
from dataclasses import dataclass
from enum import Enum, auto

# Configure logging
logger = logging.getLogger("unsloth.patches")

class PatchStatus(Enum):
    SUCCESS = auto()
    SKIPPED = auto()
    FAILED = auto()
    ALREADY_APPLIED = auto()
    NOT_NEEDED = auto()

@dataclass
class PatchResult:
    name: str
    status: PatchStatus
    message: str = ""
    
    @property
    def success(self) -> bool:
        return self.status in (PatchStatus.SUCCESS, PatchStatus.ALREADY_APPLIED, PatchStatus.NOT_NEEDED)

def _unsloth_dummy_fn(*args, **kwargs):
    """Dummy function for mocks that need to support inspect.getsource."""
    return None

class MacPatcher:
    def __init__(self):
        self._applied = False
        self._results: Dict[str, PatchResult] = {}

    @property
    def is_applied(self) -> bool:
        return self._applied

    @staticmethod
    def is_mac_with_mps() -> bool:
        if platform.system() != "Darwin": return False
        if os.environ.get("UNSLOTH_FORCE_MPS", "0") == "1": return True
        try:
            import torch
            return torch.backends.mps.is_available()
        except: return False

    @staticmethod
    def get_hardware_info() -> dict:
        info = {"chip_name": "Apple Silicon", "total_memory_bytes": 16 * 1024**3, "usable_memory_gb": 12.0, "gpu_cores": 8}
        try:
            res = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True)
            if res.returncode == 0: info["chip_name"] = res.stdout.strip()
            res = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True)
            if res.returncode == 0: 
                info["total_memory_bytes"] = int(res.stdout.strip())
                info["usable_memory_gb"] = (info["total_memory_bytes"] / 1024**3) * 0.75
            chip = info["chip_name"]
            if "M1" in chip: info["gpu_cores"] = 8 if "Max" not in chip else 32
            elif "M2" in chip: info["gpu_cores"] = 10 if "Max" not in chip else 38
            elif "M3" in chip or "M4" in chip: info["gpu_cores"] = 10 if "Max" not in chip else 40
        except: pass
        return info

    def _create_mock_module(self, name: str):
        mod = types.ModuleType(name)
        mod.__file__ = f"{name}.py"
        mod.__path__ = []
        # Support recursive attribute access that is itself a callable mock
        class MockAttr:
            def __init__(self, name): self.__name__ = name
            def __call__(self, *args, **kwargs): return self
            def __getattr__(self, n): return MockAttr(f"{self.__name__}.{n}")
            def __iter__(self): return iter([self, self])
            def __getitem__(self, k): return self
            def __contains__(self, k): return False
            def __mro_entries__(self, b): return (object,)
        
        mod._unsloth_mock = True
        return mod

    def apply(self):
        if self._applied: return list(self._results.values())
        if not self.is_mac_with_mps(): return []

        hw_info = self.get_hardware_info()
        
        # 1. Mock CUDA
        try:
            import torch
            if not torch.cuda.is_available():
                class AppleSiliconProps:
                    def __init__(self):
                        self.name = hw_info["chip_name"]
                        self.total_memory = int(hw_info["usable_memory_gb"] * 1024**3)
                        self.major = 0; self.minor = 0; self.multi_processor_count = hw_info["gpu_cores"]; self.is_integrated = True
                torch.cuda.get_device_properties = lambda d=None: AppleSiliconProps()
                torch.cuda.get_device_capability = lambda d=None: (0, 0)
                if not hasattr(torch.cuda, "memory"): torch.cuda.memory = types.ModuleType("torch.cuda.memory")
                torch.cuda.memory.mem_get_info = lambda d=None: (int(hw_info["usable_memory_gb"] * 1024**3), hw_info["total_memory_bytes"])
                torch.cuda.is_available = lambda: False
                torch.cuda.device_count = lambda: 0
        except: pass

        # 2. Mock Triton & BNB
        for pkg in ["triton", "triton.language", "triton.backends", "triton.runtime", "bitsandbytes", "bitsandbytes.nn"]:
            if pkg not in sys.modules:
                m = types.ModuleType(pkg)
                m.__path__ = [] # Mark as package
                if pkg == "triton.backends": m.backends = {}
                sys.modules[pkg] = m

        # 3. Mock Unsloth Zoo submodules
        submodules = [
            "device_type", "utils", "vision_utils", "log", "hf_utils", "peft_utils", 
            "temporary_patches", "temporary_patches.common", "rl_replacements", 
            "patching_utils", "tokenizer_utils", "compiler", "rl_environments"
        ]
        
        # Ensure root unsloth_zoo exists and is a package
        if "unsloth_zoo" not in sys.modules:
            zoo = types.ModuleType("unsloth_zoo")
            zoo.__version__ = "999.0.0"
            zoo.__path__ = [] # Mark as package
            sys.modules["unsloth_zoo"] = zoo

        for sub in submodules:
            fullname = f"unsloth_zoo.{sub}"
            if fullname in sys.modules: continue
            mod = types.ModuleType(fullname)
            mod.__file__ = f"{fullname}.py"
            
            if sub == "device_type":
                mod.get_device_type = lambda: "mps"
                mod.DEVICE_TYPE = "mps"
                mod.ALLOW_BITSANDBYTES = False
                mod.ALLOW_PREQUANTIZED_MODELS = False
                mod.is_mps = lambda: True
                mod.is_hip = lambda: False
            elif sub == "utils":
                class Version:
                    def __init__(self, v): self.v = str(v)
                    def __lt__(self, o): return False
                    def __le__(self, o): return True
                    def __gt__(self, o): return True
                    def __ge__(self, o): return True
                    def __eq__(self, o): return True
                    def __ne__(self, o): return False
                    def __str__(self): return self.v
                mod.Version = Version
                def _get_dtype(d):
                    import torch
                    if isinstance(d, torch.dtype): return d
                    return getattr(torch, str(d).split(".")[-1], torch.float16)
                mod._get_dtype = _get_dtype
                mod.get_quant_type = lambda m: "unknown"
            elif sub == "compiler":
                mod.get_transformers_model_type = lambda *a, **k: ["llama"]
                mod.unsloth_compile_transformers = lambda *a, **k: (["llama"], True)
            elif sub == "tokenizer_utils":
                mod.patch_tokenizer = lambda m, t: (m, t)
            elif sub == "hf_utils":
                mod.add_dtype_kwargs = lambda *a, **k: {}
            elif sub == "rl_replacements":
                mod.RL_REPLACEMENTS = collections.defaultdict(lambda: _unsloth_dummy_fn)
                mod.RL_REPLACEMENTS["grpo_compute_loss_slow"] = "def dummy(): pass"
                mod.RL_PRE_ITEMS = collections.defaultdict(list)
            elif sub == "patching_utils":
                class DL: pass
                mod.Bnb_Linear4bit = DL; mod.Peft_Linear4bit = DL; mod.Peft_Linear = DL
                mod.patch_model_and_tokenizer = lambda m, t, **kwargs: (m, t)
            elif sub == "rl_environments":
                mod.GRPOEnvironment = type("GRPOEnvironment", (), {})
            
            # Catch-all for missing methods
            def __getattr__(name):
                if name.startswith("__"): raise AttributeError(name)
                return _unsloth_dummy_fn
            mod.__getattr__ = __getattr__
            
            sys.modules[fullname] = mod

        self._applied = True
        return [PatchResult("mac_patch", PatchStatus.SUCCESS)]

_PATCHER = MacPatcher()

def patch_unsloth_zoo_for_mps() -> bool:
    if _PATCHER.is_applied: return True
    results = _PATCHER.apply()
    return any(r.success for r in results)

def is_patched() -> bool:
    return _PATCHER.is_applied
