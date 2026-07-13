# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Shared torchao Windows-ROCm import stub.

torchao (pulled in by transformers.quantizers) imports distributed_c10d.py
unconditionally, which crashes on Windows ROCm because the RCCL backend
(torch._C._distributed_c10d) is absent. Stubbing torchao short-circuits its
import chain; _StubSubpackageFinder handles any depth of torchao.xxx.yyy.
Worker subprocesses call install_torchao_windows_rocm_stub() before importing
transformers / unsloth_zoo.
"""

from __future__ import annotations

import sys
import types
import importlib.abc
import importlib.machinery

_STUB_SENTINEL = object()


# Metaclass for stub types so isinstance(x, StubClass) returns False instead of
# raising TypeError -- peft's lora/torchao.py does isinstance() against torchao
# types, which fails if those names resolve to stub modules rather than types.
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

    def _ga(
        attr,
        _m = m,
        _n = mod_name,
    ):
        if attr.startswith("__"):
            raise AttributeError(attr)
        # Return a stub CLASS (not module) so isinstance() returns False, not TypeError.
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
    def find_spec(
        self,
        fullname,
        path,
        target = None,
    ):
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


def is_win32_rocm() -> bool:
    """True on Windows ROCm, where torch.distributed (and thus torchao) is unavailable.

    Gate on the runtime torch, not env vars (HIP_PATH persists after a CUDA revert). AMD SDK
    wheels lack torch.version.hip but tag "rocm" in __version__, so accept either. Shared by the
    import stub and the export gate so they can't drift.
    """
    if sys.platform != "win32":
        return False
    try:
        import torch
        return bool(
            getattr(getattr(torch, "version", None), "hip", None)
            or "rocm" in getattr(torch, "__version__", "").lower()
        )
    except Exception:
        return False


def install_torchao_windows_rocm_stub() -> None:
    """Pre-stub torchao on Windows ROCm so transformers/peft imports don't crash.

    No-op elsewhere (incl. Windows CUDA, where torchao is real). Must run before
    importing transformers / unsloth_zoo. Safe to call once per worker.
    """
    if not is_win32_rocm():
        return
    # Register the finder only on Windows ROCm, and only once (no duplicates on re-call).
    if not any(isinstance(_f, _StubSubpackageFinder) for _f in sys.meta_path):
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
