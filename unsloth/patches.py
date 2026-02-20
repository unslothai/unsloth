# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""
Unsloth Full Automatic Patcher for Apple Silicon (macOS/MPS) Compatibility
==========================================================================

This module provides a dynamic, intercepting patching system. It uses a
MetaPathFinder to catch imports and either provide mocks or wrap real
loaders to inject missing attributes (like Linear4bit) on the fly.
"""

from __future__ import annotations

import sys
import os
import platform
import logging
import subprocess
import types
import collections
import torch
import torch.nn as nn
from typing import Optional, Any, Dict, List
from dataclasses import dataclass
from importlib.abc import MetaPathFinder, Loader
from importlib.machinery import ModuleSpec

# Configure logging
logger = logging.getLogger("unsloth.patches")


def _unsloth_dummy_fn(*args, **kwargs):
    """Universal dummy function for mocks."""
    return None


class MockModule(nn.Module):
    """Robust mock module that survives both import system and torch.nn.Module checks."""

    def __init__(self, name):
        super().__init__()
        # Bypass nn.Module.__setattr__ to set module attributes
        for k, v in {
            "__name__": name,
            "__path__": [],
            "__file__": f"{name}.py",
            "__package__": name.rsplit(".", 1)[0] if "." in name else "",
            "_unsloth_mock": True,
        }.items():
            object.__setattr__(self, k, v)

    def __call__(self, *args, **kwargs):
        # If it looks like a layer constructor, return an instance that is also an nn.Module
        if any(x in self.__name__ for x in ("Linear", "Layer", "Norm", "dispatch")):
            return nn.Module()
        return self

    def __getattr__(self, name):
        # 1. Handle double-underscore names
        if name.startswith("__") and name.endswith("__"):
            # Allow common metadata attributes
            if name == "__version__":
                return "3.0.0"
            if name == "__origin__":
                return "unsloth_mock"
            raise AttributeError(name)

        # 2. Handle nn.Module internals
        if name.startswith("_") and name not in ("_unsloth_mock",):
            try:
                return self.__dict__[name]
            except KeyError:
                raise AttributeError(name)

        # 3. Specialized Mocks
        fullname = f"{self.__name__}.{name}"

        # RL Specialization
        if "rl_replacements" in self.__name__:
            if name == "RL_REPLACEMENTS":
                return collections.defaultdict(
                    lambda: _unsloth_dummy_fn,
                    {"grpo_compute_loss_slow": "def dummy(): pass"},
                )
            if name == "RL_PRE_ITEMS":
                return collections.defaultdict(list)

        # Versioning/Utility Specialization
        if name == "Version":

            class Version:
                def __init__(self, v):
                    self.v = str(v)

                def __lt__(self, o):
                    return False

                def __le__(self, o):
                    return True

                def __gt__(self, o):
                    return True

                def __ge__(self, o):
                    return True

                def __eq__(self, o):
                    return True

                def __ne__(self, o):
                    return False

                def __str__(self):
                    return self.v

            return Version

        if name == "get_transformers_model_type":
            return lambda *a, **k: ["llama"]
        if name == "unsloth_compile_transformers":
            return lambda *a, **k: (["llama"], True)
        if name == "patch_tokenizer":
            return lambda m, t: (m, t)
        if name == "patch_model_and_tokenizer":
            return lambda m, t, **k: (m, t)
        if name == "patch_layernorm":
            return lambda *a, **k: None
        if name == "add_dtype_kwargs":
            return lambda *a, **k: {}
        if name == "dtype_from_config":
            return lambda *a, **k: torch.float16
        if name == "_get_dtype":

            def _get_dtype(d):
                if d is None:
                    return torch.float16
                if isinstance(d, torch.dtype):
                    return d
                if isinstance(d, str):
                    try:
                        return getattr(torch, d)
                    except:
                        return torch.float16
                if str(type(d)) == "<class 'unsloth.patches.MockModule'>":
                    return torch.float16
                try:
                    return getattr(torch, str(d).split(".")[-1])
                except:
                    return torch.float16

            return _get_dtype

        # Device Type Specialization
        if "device_type" in self.__name__:
            if name == "DEVICE_TYPE":
                return "mps"
            if name == "DEVICE_TYPE_TORCH":
                return "mps"
            if name == "DEVICE_COUNT":
                return 1
            if name in ("is_mps", "HAS_MPS"):
                return lambda: True
            if name in ("is_hip", "is_xpu", "HAS_CUDA", "HAS_HIP"):
                return lambda: False
            if name == "get_device_type":
                return lambda: "mps"
            if name == "ALLOW_PREQUANTIZED_MODELS":
                return False
            if name == "ALLOW_BITSANDBYTES":
                return False

        # Default: Return another mock ONLY if we are in a mock target package
        if any(
            self.__name__.startswith(t)
            for t in ("unsloth_zoo", "triton", "bitsandbytes", "peft.tuners.lora.bnb")
        ):
            return MockModule(fullname)

        raise AttributeError(f"MockModule '{self.__name__}' has no attribute '{name}'")

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
        return (nn.Module,)


class PatchedLoader(Loader):
    """Wraps a real loader to inject missing attributes after execution."""

    def __init__(self, real_loader):
        self.real_loader = real_loader

    def create_module(self, spec):
        return self.real_loader.create_module(spec)

    def exec_module(self, module):
        self.real_loader.exec_module(module)
        # On-the-fly hardening for peft
        if module.__name__ == "peft.tuners.lora":
            if not hasattr(module, "Linear4bit"):
                module.Linear4bit = MockModule("peft.tuners.lora.Linear4bit")
            if not hasattr(module, "Linear"):
                module.Linear = MockModule("peft.tuners.lora.Linear")
        # Hardening for bitsandbytes.nn
        elif module.__name__ == "bitsandbytes.nn":
            if not hasattr(module, "Linear4bit"):
                module.Linear4bit = MockModule("bitsandbytes.nn.Linear4bit")


class UnslothMockFinder(MetaPathFinder):
    """Dynamic finder that provides mocks or wraps real loaders for on-the-fly patching."""

    def __init__(self):
        self.mock_targets = ("unsloth_zoo", "triton", "bitsandbytes")
        self.patch_targets = ("peft.tuners.lora", "bitsandbytes.nn")
        self._disabled = False

    def find_spec(self, fullname, path, target=None):
        if self._disabled:
            return None

        # 1. Full Mocking for Zoo, Triton, BNB
        if any(
            fullname == t or fullname.startswith(f"{t}.") for t in self.mock_targets
        ):
            # Check if it's already a real module (e.g. partially loaded before patcher)
            if fullname in sys.modules and not hasattr(
                sys.modules[fullname], "_unsloth_mock"
            ):
                return None
            return ModuleSpec(fullname, UnslothMockLoader(), origin="unsloth_mock")

        # 2. On-the-fly Patching for PEFT/BNB.nn
        if fullname in self.patch_targets:
            self._disabled = True  # Prevent recursion
            try:
                import importlib.util

                spec = importlib.util.find_spec(fullname)
                if spec and spec.loader:
                    spec.loader = PatchedLoader(spec.loader)
                    return spec
            except:
                pass
            finally:
                self._disabled = False

        return None


class UnslothMockLoader(Loader):
    def create_module(self, spec):
        if spec.name in sys.modules:
            return sys.modules[spec.name]
        return MockModule(spec.name)

    def exec_module(self, module):
        sys.modules[module.__name__] = module


_PATCH_APPLIED = False


def patch_unsloth_zoo_for_mps() -> bool:
    global _PATCH_APPLIED
    if _PATCH_APPLIED:
        return True

    if platform.system() != "Darwin":
        return False

    # 1. Install the full automatic finder
    sys.meta_path.insert(0, UnslothMockFinder())

    # 2. Mock CUDA properties
    try:
        if not torch.cuda.is_available():
            info = {"chip": "Apple Silicon", "mem": 16 * 1024**3, "gpu": 8}
            try:
                res = subprocess.run(
                    ["sysctl", "-n", "machdep.cpu.brand_string"],
                    capture_output=True,
                    text=True,
                )
                if res.returncode == 0:
                    info["chip"] = res.stdout.strip()
                res = subprocess.run(
                    ["sysctl", "-n", "hw.memsize"], capture_output=True, text=True
                )
                if res.returncode == 0:
                    info["mem"] = int(res.stdout.strip())
                chip = info["chip"]
                if "M1" in chip:
                    info["gpu"] = 8 if "Max" not in chip else 32
                elif "M2" in chip:
                    info["gpu"] = 10 if "Max" not in chip else 38
                elif "M3" in chip or "M4" in chip:
                    info["gpu"] = 10 if "Max" not in chip else 40
            except:
                pass

            class AppleSiliconProps:
                def __init__(self):
                    self.name = info["chip"]
                    self.total_memory = int(info["mem"] * 0.75)
                    self.major = 0
                    self.minor = 0
                    self.multi_processor_count = info["gpu"]
                    self.is_integrated = True

            torch.cuda.get_device_properties = lambda d=None: AppleSiliconProps()
            torch.cuda.get_device_capability = lambda d=None: (0, 0)
            if not hasattr(torch.cuda, "memory"):
                torch.cuda.memory = types.ModuleType("torch.cuda.memory")
            torch.cuda.memory.mem_get_info = lambda d=None: (
                int(info["mem"] * 0.75),
                info["mem"],
            )
            torch.cuda.is_available = lambda: False
            torch.cuda.device_count = lambda: 0

            # Patch checkpointing
            import torch.utils.checkpoint as cp

            if not hasattr(cp, "_unsloth_patched"):
                orig = cp.checkpoint
                cp.checkpoint = lambda f, *a, use_reentrant=True, **k: orig(
                    f, *a, use_reentrant=False, **k
                )
                cp._unsloth_patched = True
    except:
        pass

    # 3. Patch PEFT detection
    try:
        import peft.import_utils

        peft.import_utils.is_bnb_available = lambda: False
        peft.import_utils.is_bnb_4bit_available = lambda: False
    except:
        pass

    _PATCH_APPLIED = True
    return True


def is_patched() -> bool:
    return _PATCH_APPLIED
