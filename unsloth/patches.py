# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""
Unsloth Automatic Patcher for Apple Silicon (macOS/MPS) Compatibility
=====================================================================

This module provides a fully automatic, dynamic patching system for unsloth_zoo
and CUDA-specific dependencies. It uses a MetaPathFinder to intercept imports
and provide robust mocks on the fly.
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
from importlib.abc import MetaPathFinder, Loader
from importlib.machinery import ModuleSpec

# Configure logging
logger = logging.getLogger("unsloth.patches")

import torch
import torch.nn as nn

def _unsloth_dummy_fn(*args, **kwargs):
    """Universal dummy function for mocks."""
    return None

class MockModule(nn.Module):
    """Extremely robust mock module that satisfies both Python imports and torch.nn.Module checks."""
    def __init__(self, name):
        super().__init__()
        object.__setattr__(self, "__name__", name)
        object.__setattr__(self, "__path__", [])
        object.__setattr__(self, "__file__", f"{name}.py")
        object.__setattr__(self, "__package__", name.rsplit(".", 1)[0] if "." in name else "")
        object.__setattr__(self, "_unsloth_mock", True)

    def __call__(self, *args, **kwargs):
        return self

    def __getattr__(self, name):
        # Handle special attributes by raising AttributeError
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
            
        # Handle nn.Module internal attributes that might be missing in early init
        if name.startswith("_") and name not in ("_unsloth_mock",):
            try: return self.__dict__[name]
            except KeyError: raise AttributeError(name)
            
        # Check for specialized mocks first
        fullname = f"{self.__name__}.{name}"
        
        # RL Specialization
        if "rl_replacements" in self.__name__ and name == "RL_REPLACEMENTS":
            return collections.defaultdict(lambda: _unsloth_dummy_fn, {"grpo_compute_loss_slow": "def dummy(): pass"})
        if "rl_replacements" in self.__name__ and name == "RL_PRE_ITEMS":
            return collections.defaultdict(list)
            
        # Versioning Specialization
        if name == "Version":
            class Version:
                def __init__(self, v): self.v = str(v)
                def __lt__(self, o): return False
                def __le__(self, o): return True
                def __gt__(self, o): return True
                def __ge__(self, o): return True
                def __eq__(self, o): return True
                def __ne__(self, o): return False
                def __str__(self): return self.v
            return Version

        # Logic Specialization
        if name == "get_transformers_model_type": return lambda *a, **k: ["llama"]
        if name == "unsloth_compile_transformers": return lambda *a, **k: (["llama"], True)
        if name == "patch_tokenizer": return lambda m, t: (m, t)
        if name == "patch_model_and_tokenizer": return lambda m, t, **k: (m, t)
        if name == "patch_layernorm": return lambda *a, **k: None
        if name == "add_dtype_kwargs": return lambda *a, **k: {}
        if name == "_get_dtype":
            def _get_dtype(d):
                if d is None: return torch.float16
                if isinstance(d, torch.dtype): return d
                try: return getattr(torch, str(d).split(".")[-1])
                except: return torch.float16
            return _get_dtype
        if name == "backends" and "triton" in self.__name__: return {}

        # Device Type Specialization
        if "device_type" in self.__name__:
            if name == "DEVICE_TYPE": return "mps"
            if name == "DEVICE_TYPE_TORCH": return "mps"
            if name == "DEVICE_COUNT": return 1
            if name in ("is_mps", "HAS_MPS"): return lambda: True
            if name in ("is_hip", "is_xpu", "HAS_CUDA", "HAS_HIP"): return lambda: False
            if name == "get_device_type": return lambda: "mps"
            if name == "ALLOW_PREQUANTIZED_MODELS": return False
            if name == "ALLOW_BITSANDBYTES": return False

        # Default: Return another callable mock
        return MockModule(fullname)

    def __iter__(self):
        return iter([self, self])

    def __getitem__(self, key):
        return self

    def __setitem__(self, key, value):
        pass

    def __contains__(self, key):
        return False

    def __len__(self):
        return 0

    def __bool__(self):
        return True

    def __mro_entries__(self, bases):
        return (object,)

class UnslothMockLoader(Loader):
    def create_module(self, spec):
        return MockModule(spec.name)
    def exec_module(self, module):
        sys.modules[module.__name__] = module

class UnslothMockFinder(MetaPathFinder):
    """Automatically catches any import for specified packages and provides mocks."""
    def __init__(self):
        self.targets = ("unsloth_zoo", "triton", "bitsandbytes", "peft.tuners.lora.bnb", "peft.tuners.lora.bnb4bit")
        self.loader = UnslothMockLoader()

    def find_spec(self, fullname, path, target=None):
        if any(fullname == t or fullname.startswith(f"{t}.") for t in self.targets):
            return ModuleSpec(fullname, self.loader, origin="unsloth_mock")
        return None

_PATCH_APPLIED = False

def patch_unsloth_zoo_for_mps() -> bool:
    global _PATCH_APPLIED
    if _PATCH_APPLIED: return True

    if platform.system() != "Darwin": return False
    
    # 1. Install the automatic finder
    sys.meta_path.insert(0, UnslothMockFinder())

    # 2. Mock CUDA properties (cannot be done via MetaPathFinder easily)
    try:
        import torch
        if not torch.cuda.is_available():
            # Minimal hardware info for mocking
            info = {"chip": "Apple Silicon", "mem": 16 * 1024**3}
            try:
                res = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True)
                if res.returncode == 0: info["chip"] = res.stdout.strip()
                res = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True)
                if res.returncode == 0: info["mem"] = int(res.stdout.strip())
            except: pass

            class AppleSiliconProps:
                def __init__(self):
                    self.name = info["chip"]
                    self.total_memory = int(info["mem"] * 0.75)
                    self.major = 0; self.minor = 0; self.multi_processor_count = 8; self.is_integrated = True
            
            torch.cuda.get_device_properties = lambda d=None: AppleSiliconProps()
            torch.cuda.get_device_capability = lambda d=None: (0, 0)
            if not hasattr(torch.cuda, "memory"): torch.cuda.memory = types.ModuleType("torch.cuda.memory")
            torch.cuda.memory.mem_get_info = lambda d=None: (int(info["mem"] * 0.75), info["mem"])
            torch.cuda.is_available = lambda: False
            torch.cuda.device_count = lambda: 0
            
            # Patch checkpointing
            import torch.utils.checkpoint as cp
            if not hasattr(cp, "_unsloth_patched"):
                orig = cp.checkpoint
                cp.checkpoint = lambda f, *a, use_reentrant=True, **k: orig(f, *a, use_reentrant=False, **k)
                cp._unsloth_patched = True
    except: pass

    # 3. Patch PEFT detection and inject dummy classes
    try:
        import torch.nn as nn
        
        # Ensure bitsandbytes.nn has Linear4bit
        import bitsandbytes.nn as bnb_nn
        if not hasattr(bnb_nn, "Linear4bit"):
            class Linear4bit(nn.Module): pass
            bnb_nn.Linear4bit = Linear4bit
            
        # Ensure peft.tuners.lora has Linear4bit and Linear
        import peft.import_utils
        peft.import_utils.is_bnb_available = lambda: False
        peft.import_utils.is_bnb_4bit_available = lambda: False
        
        try:
            import peft.tuners.lora as lora_mod
            if not hasattr(lora_mod, "Linear4bit"):
                class Linear4bit(nn.Module): pass
                lora_mod.Linear4bit = Linear4bit
            if not hasattr(lora_mod, "Linear"):
                class Linear(nn.Module): pass
                lora_mod.Linear = Linear
        except ImportError: pass
    except: pass

    _PATCH_APPLIED = True
    return True

def is_patched() -> bool:
    return _PATCH_APPLIED
