# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Shared Windows-ROCm import stubs.

torchao (pulled in by transformers.quantizers) imports distributed_c10d.py
unconditionally, which crashes on Windows ROCm because the RCCL backend
(torch._C._distributed_c10d) is absent. Stubbing torchao short-circuits its
import chain; _StubSubpackageFinder handles any depth of torchao.xxx.yyy.
Worker subprocesses call install_torchao_windows_rocm_stub() before importing
transformers / unsloth_zoo.

xformers hits the same absent backend and takes diffusers with it, so the diffusion paths
install both stubs before importing diffusers.
"""

from __future__ import annotations

import os
import re
import sys
import types
import importlib.abc
import importlib.machinery
import importlib.util
from typing import Optional

_STUB_SENTINEL = object()


# isinstance() against a stub module raises TypeError; peft's lora/torchao.py needs it to return False.
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


def _module_is_rocm(mod) -> bool:
    """Whether an already-imported torch module is a ROCm build. Some ROCm wheels lack
    torch.version.hip but still encode "rocm" in __version__."""
    return bool(
        getattr(getattr(mod, "version", None), "hip", None)
        or "rocm" in getattr(mod, "__version__", "").lower()
    )


# torch/version.py is generated, always as ``hip: Optional[str] = None`` on CUDA, ``= '6.4.5...'`` on ROCm.
_HIP_LINE_RE = re.compile(r"^hip\s*(?::[^=]*)?=\s*(.+?)\s*$", re.MULTILINE)


def _version_is_rocm_tagged() -> Optional[bool]:
    """Whether the installed wheel's version carries a rocm tag. None if unreadable."""
    # Neither on-disk signal was readable, so importing is the only way left.
    try:
        from importlib.metadata import version
        return "rocm" in version("torch").lower()
    except Exception:  # noqa: BLE001 -- no dist-info / unreadable METADATA
        return None


def _hip_field_is_set() -> Optional[bool]:
    """Whether torch/version.py's ``hip`` field names a ROCm version, None if unreadable.
    find_spec resolves the path without executing torch."""
    try:
        spec = importlib.util.find_spec("torch")
        origin = getattr(spec, "origin", None) if spec is not None else None
        if not origin:
            return None
        with open(
            os.path.join(os.path.dirname(origin), "version.py"),
            encoding = "utf-8",
            errors = "replace",
        ) as handle:
            found = _HIP_LINE_RE.search(handle.read())
        if found is None:
            return None
        return found.group(1).strip().strip("\"'") not in ("None", "")
    except Exception:  # noqa: BLE001 -- an unreadable tree is not a verdict
        return None


def _installed_torch_is_rocm() -> Optional[bool]:
    """ROCm or not, read off disk without importing torch. None when it cannot be told.

    Avoiding the import is the point: run.py calls this at import time, where torch costs seconds,
    sizes the OpenMP/BLAS pools before configure_cpu_threads() can set them, and on Windows ROCm
    can fail outright until main.py has registered the HIP DLL directories. Neither signal alone
    is enough: AMD's Windows build (torch-2.8.0a0+gitfc14c65) carries no rocm tag so only ``hip``
    answers, while a wheel without dist-info has no version to read. So a NEGATIVE needs BOTH
    signals legible and both saying no.
    """
    tagged = _version_is_rocm_tagged()
    if tagged:
        return True
    hip = _hip_field_is_set()
    if hip:
        return True
    return False if (tagged is False and hip is False) else None


def torch_is_rocm() -> bool:
    """True when the active torch is a ROCm/HIP build, using import-free checks when possible."""
    mod = sys.modules.get("torch")
    if mod is not None:
        return _module_is_rocm(mod)
    verdict = _installed_torch_is_rocm()
    if verdict is not None:
        return verdict
    try:
        import torch
    except Exception:
        return False
    return _module_is_rocm(torch)


def _is_windows_rocm() -> bool:
    """True on a Windows host whose active torch is a ROCm build."""
    return sys.platform == "win32" and torch_is_rocm()


def _ensure_finder() -> None:
    """Register the subpackage finder once, however many stubs are installed."""
    if not any(isinstance(f, _StubSubpackageFinder) for f in sys.meta_path):
        sys.meta_path.append(_StubSubpackageFinder())


def is_stubbed(package: str) -> bool:
    """True iff ``package`` resolves to one of these stubs. ``find_spec`` and even
    ``from torchao.quantization import quantize_`` succeed against a stub, so any caller
    that needs the package to WORK must ask this first."""
    return getattr(sys.modules.get(package), "_unsloth_stub", None) is _STUB_SENTINEL


def install_torchao_windows_rocm_stub() -> None:
    """Pre-stub torchao on Windows ROCm so transformers/peft imports don't crash.

    No-op elsewhere (incl. Windows CUDA, where torchao is real). Must run before
    importing transformers / unsloth_zoo. Safe to call once per worker.
    """
    if _is_windows_rocm():
        _ensure_finder()
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


def install_xformers_windows_rocm_stub() -> None:
    """Pre-stub xformers on Windows ROCm so diffusers can import at all. No-op elsewhere, and must
    precede diffusers: the Windows xformers pin is CUDA-only, so against a ROCm torch (no
    distributed backend) ``import xformers.ops`` dies in torch.distributed, and diffusers imports
    xformers on sight, taking every model import with it."""
    if _is_windows_rocm():
        _ensure_finder()
        for _xf_name in ("xformers", "xformers.ops"):
            if _xf_name not in sys.modules:
                sys.modules[_xf_name] = _make_mod_stub(_xf_name)
