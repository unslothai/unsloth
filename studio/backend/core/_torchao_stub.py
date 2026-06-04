# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Shared torchao Windows-ROCm import stub.

torchao (pulled in by transformers.quantizers) imports
torch.distributed._functional_collectives at module level, which imports
distributed_c10d.py unconditionally — that file crashes on Windows ROCm because
torch._C._distributed_c10d (the RCCL backend) is absent.
torch/distributed/__init__.py itself is guarded by `if is_available()` so
`import torch.distributed` alone is safe; the crash only comes via torchao's
import chain.  Stubbing torchao short-circuits it entirely.
_StubSubpackageFinder handles any depth of torchao.xxx.yyy imports.

This logic used to be duplicated inline inside run_export_process() and
run_training_process(); it now lives here so both worker subprocesses call the
single `install_torchao_windows_rocm_stub()` entrypoint before importing
transformers / unsloth_zoo.
"""

from __future__ import annotations

import sys
import types
import importlib.abc
import importlib.machinery

_STUB_SENTINEL = object()


# Metaclass for stub types so that isinstance(x, StubClass) returns False
# instead of raising TypeError ("arg 2 must be a type").
# peft/tuners/lora/torchao.py does:
#   from torchao.dtypes import AffineQuantizedTensor, LinearActivationQuantizedTensor
#   isinstance(weight, (AffineQuantizedTensor, LinearActivationQuantizedTensor))
# If those names resolve to stub modules rather than types, isinstance() raises.
class _StubTypeMeta(type):
    def __instancecheck__(cls, instance):
        return False

    def __subclasscheck__(cls, subclass):
        return False

    def __getattr__(cls, attr):
        if attr.startswith("__"):
            raise AttributeError(attr)
        child = _StubTypeMeta(attr, (), {})
        setattr(cls, attr, child)
        return child

    def __call__(cls, *args, **kwargs):
        return None


def _make_stub_type(name):
    """Stub class: accepted by isinstance() (always False), supports attr access."""
    return _StubTypeMeta(name, (), {})


def _make_mod_stub(mod_name):
    m = types.ModuleType(mod_name)
    m.__path__ = []
    m.__package__ = mod_name
    m._unsloth_stub = _STUB_SENTINEL
    m.__spec__ = importlib.machinery.ModuleSpec(mod_name, loader = None, is_package = True)

    def _ga(attr, _m = m, _n = mod_name):
        if attr.startswith("__"):
            raise AttributeError(attr)
        # Return a stub CLASS (not a module) so that isinstance(x, attr)
        # works and returns False instead of raising TypeError.
        child = _make_stub_type(f"{_n}.{attr}")
        setattr(_m, attr, child)
        return child

    m.__getattr__ = _ga
    return m


class _StubSubpackageLoader(importlib.abc.Loader):
    def __init__(self, mod_name):
        self._mod_name = mod_name

    def create_module(self, spec):
        return _make_mod_stub(self._mod_name)

    def exec_module(self, module):
        pass


class _StubSubpackageFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target = None):
        if "." not in fullname:
            return None
        parent = sys.modules.get(fullname.rsplit(".", 1)[0])
        if parent is None:
            return None
        if getattr(parent, "_unsloth_stub", None) is not _STUB_SENTINEL:
            return None
        return importlib.machinery.ModuleSpec(
            fullname, _StubSubpackageLoader(fullname), is_package = True
        )


def install_torchao_windows_rocm_stub() -> None:
    """Pre-stub torchao on Windows ROCm so transformers/peft imports don't crash.

    No-op on every other platform (Windows CUDA included — there torchao is real
    and shadowing it would break torchao-based quantization paths). Must run
    before any import of transformers / unsloth_zoo. Safe to call once per worker
    process.
    """
    # Gate on the active torch runtime, not env-var presence -- HIP_PATH /
    # ROCM_PATH stay set after a user installs the HIP SDK and reverts to a
    # CUDA torch wheel. AMD SDK / Radeon ROCm wheels may not set torch.version.hip
    # but still encode "rocm" in torch.__version__, so accept either.
    _is_win32_rocm = False
    if sys.platform == "win32":
        try:
            import torch as _torch_probe

            _is_win32_rocm = bool(
                getattr(getattr(_torch_probe, "version", None), "hip", None)
                or "rocm" in getattr(_torch_probe, "__version__", "").lower()
            )
            del _torch_probe
        except Exception:
            pass
    if _is_win32_rocm:
        # Register the finder only on Windows ROCm -- on other platforms there
        # are no stub modules seeded, so appending is a pure accumulation.
        sys.meta_path.append(_StubSubpackageFinder())
        # Seed torchao top-level + key submodules; the finder handles the rest.
        for _tao_name in (
            "torchao",
            "torchao.quantization",
            "torchao.dtypes",
            "torchao.float8",
            "torchao.utils",
        ):
            if _tao_name not in sys.modules:
                sys.modules[_tao_name] = _make_mod_stub(_tao_name)
