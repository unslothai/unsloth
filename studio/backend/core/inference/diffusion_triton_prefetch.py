# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Skip Triton >= 3.8's sm_8x prefetch pass for 8-bit tt.dot modules (triton-lang/triton#12486: spills, ~7x slower).

The cache key ignores the skip; harmless since only load timing changes, never output.
"""

from __future__ import annotations

import functools
import importlib.abc
import importlib.util
import inspect
import os
import re
import sys
import threading
from pathlib import Path

_TRITON_PREFETCH_ENV = "UNSLOTH_TRITON_8BIT_PREFETCH_FIX"
_TRITON_PREFETCH_SENTINEL = "__unsloth_8bit_prefetch_fix__"
_TRITON_PREFETCH_FIRST = (3, 8)  # the first release whose prefetch pass splits 8-bit dots (#9806)
_TRITON_PREFETCH_PIPELINE = re.compile(
    r"if\s+capability\s*//\s*10\s*==\s*8\s*:\s*\n\s*passes\.ttgpuir\.add_prefetch\(pm\)"
)
_INSTALL_LOCK = threading.Lock()
_TRITON_8BIT_DOT = re.compile(r"\btt\.dot\b[^\n]*?:\s*tensor<[0-9x]*x(?:i8|f8E[0-9A-Za-z]+)>")


def _triton_8bit_prefetch_fix_mode():
    return (os.environ.get(_TRITON_PREFETCH_ENV) or "1").strip().lower()


def _triton_ttir_has_8bit_dot(mod):
    try:
        text = mod.str_nodebug() if hasattr(mod, "str_nodebug") else str(mod)
    except Exception:
        return False
    return "tt.dot" in text and _TRITON_8BIT_DOT.search(text) is not None


def fix_triton_sm8x_8bit_prefetch():
    """Returns True when the fix is installed (now or before), False when it does not apply."""
    if _triton_8bit_prefetch_fix_mode() in ("0", "off", "false", "no"):
        return False
    try:
        import triton
        from triton._C.libtriton import passes
        from triton.backends.nvidia import compiler as nvidia_compiler
    except Exception:
        return False
    backend = getattr(nvidia_compiler, "CUDABackend", None)
    ttgpuir = getattr(passes, "ttgpuir", None)
    stage = None if backend is None else backend.__dict__.get("make_ttgir")
    make_ttgir = getattr(stage, "__func__", stage)
    original_prefetch = getattr(ttgpuir, "add_prefetch", None)
    if not callable(make_ttgir) or not callable(original_prefetch):
        return False
    if getattr(make_ttgir, _TRITON_PREFETCH_SENTINEL, False):
        return True
    match = re.match(r"(\d+)\.(\d+)", str(getattr(triton, "__version__", "")))
    version = (int(match.group(1)), int(match.group(2))) if match else None
    if (
        version is None or version < _TRITON_PREFETCH_FIRST
    ) and _triton_8bit_prefetch_fix_mode() != "force":
        return False
    try:
        source = inspect.getsource(make_ttgir)
    except Exception:
        return False
    if _TRITON_PREFETCH_PIPELINE.search(source) is None:
        return False  # this Triton no longer adds the pass for sm_8x where we expect it

    # concurrent first calls must not pair one thread's make_ttgir wrapper with another's skip flag
    with _INSTALL_LOCK:
        stage = backend.__dict__.get("make_ttgir")
        if getattr(getattr(stage, "__func__", stage), _TRITON_PREFETCH_SENTINEL, False):
            return True
        original_prefetch = getattr(ttgpuir, "add_prefetch", None)
        if not callable(original_prefetch):
            return False
        # skip flag lives on the pass wrapper, shared by any second copy of this fix
        state = getattr(original_prefetch, _TRITON_PREFETCH_SENTINEL, None)
        if not isinstance(state, threading.local):
            state = threading.local()

            def add_prefetch(*args, **kwargs):
                if getattr(state, "active", False):
                    return None
                return original_prefetch(*args, **kwargs)

            setattr(add_prefetch, _TRITON_PREFETCH_SENTINEL, state)
            add_prefetch.__wrapped__ = original_prefetch
            ttgpuir.add_prefetch = add_prefetch

        @functools.wraps(make_ttgir)
        def patched_make_ttgir(*args, **kwargs):
            mod = args[0] if args else kwargs.get("mod")
            capability = args[3] if len(args) > 3 else kwargs.get("capability")
            skip = (
                isinstance(capability, int)
                and capability // 10 == 8
                and _triton_8bit_prefetch_fix_mode() not in ("0", "off", "false", "no")
                and _triton_ttir_has_8bit_dot(mod)
            )
            if not skip:
                return make_ttgir(*args, **kwargs)
            previous = getattr(state, "active", False)
            state.active = True
            try:
                return make_ttgir(*args, **kwargs)
            finally:
                state.active = previous

        setattr(patched_make_ttgir, _TRITON_PREFETCH_SENTINEL, True)
        backend.make_ttgir = staticmethod(patched_make_ttgir)
        return True


# Inductor's compile sidecar would cache spilling cubins; it imports the pickler by name, so ours installs the fix there.
_INDUCTOR_POOL_MODULE = "torch._inductor.compile_worker.subproc_pool"
_TRITON_COMPILER_MODULE = "triton.backends.nvidia.compiler"
_WORKER_PICKLER = "_UnslothTritonPrefetchPickler"
_WORKER_PICKLER_CLASS = []


def _worker_pickler_class():
    if not _WORKER_PICKLER_CLASS:
        from torch._inductor.compile_worker.subproc_pool import SubprocPickler
        cls = type(_WORKER_PICKLER, (SubprocPickler,), {"__module__": __name__})
        _WORKER_PICKLER_CLASS.append(cls)
    return _WORKER_PICKLER_CLASS[0]


def __getattr__(name):
    if name == _WORKER_PICKLER:
        try:
            fix_triton_sm8x_8bit_prefetch()
        except Exception:
            pass
        return _worker_pickler_class()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _workers_can_import_this_module():
    """An unimportable pickler name kills the sidecar pool, so swap only when this module is reachable."""
    if __name__ == "__main__" or not __name__.count("."):
        return False
    root = Path(__file__).resolve().parents[__name__.count(".")]
    custom = os.environ.get("TORCH_CUSTOM_PYTHONPATH")
    entries = custom.split(os.pathsep) if custom is not None else list(sys.path)
    for entry in entries:
        try:
            if entry and Path(entry).resolve() == root:
                return True
        except Exception:
            continue
    return False


def _cover_inductor_workers(module):
    pool = getattr(module, "SubprocPool", None)
    init = getattr(pool, "__init__", None)
    if init is None or getattr(init, _TRITON_PREFETCH_SENTINEL, False):
        return False

    @functools.wraps(init)
    def __init__(self, *args, **kwargs):
        if (
            kwargs.get("pickler") is None
            and len(args) < 2
            and _triton_8bit_prefetch_fix_mode() not in ("0", "off", "false", "no")
            and _workers_can_import_this_module()
        ):
            try:
                kwargs["pickler"] = _worker_pickler_class()()
            except Exception:
                pass
        return init(self, *args, **kwargs)

    setattr(__init__, _TRITON_PREFETCH_SENTINEL, True)
    pool.__init__ = __init__
    return True


_POST_IMPORT = {
    _TRITON_COMPILER_MODULE: lambda module: fix_triton_sm8x_8bit_prefetch(),
    _INDUCTOR_POOL_MODULE: _cover_inductor_workers,
}


class _PostImportLoader(importlib.abc.Loader):
    __slots__ = ("_loader", "_hook")

    def __init__(self, loader, hook):
        self._loader = loader
        self._hook = hook

    def create_module(self, spec):
        create = getattr(self._loader, "create_module", None)
        return None if create is None else create(spec)

    def exec_module(self, module):
        self._loader.exec_module(module)
        try:
            self._hook(module)
        except Exception:
            pass

    def __getattr__(self, attribute):
        return getattr(self._loader, attribute)


class _PostImportFinder(importlib.abc.MetaPathFinder):
    """Runs post-import hooks; never changes what is imported."""

    __slots__ = (_TRITON_PREFETCH_SENTINEL, "_finding")

    def __init__(self):
        setattr(self, _TRITON_PREFETCH_SENTINEL, True)
        self._finding = False

    def find_spec(
        self,
        fullname,
        path = None,
        target = None,
    ):
        hook = _POST_IMPORT.get(fullname)
        if hook is None or self._finding:
            return None
        self._finding = True
        try:
            spec = importlib.util.find_spec(fullname)
        except Exception:
            return None
        finally:
            self._finding = False
        if spec is None or spec.loader is None or not hasattr(spec.loader, "exec_module"):
            return None
        spec.loader = _PostImportLoader(spec.loader, hook)
        return spec


def triton_prefetch_fix_installed():
    """True in a process whose Triton compiles through the fix (also run inside Inductor's compile workers)."""
    module = sys.modules.get(_TRITON_COMPILER_MODULE)
    backend = getattr(module, "CUDABackend", None)
    stage = None if backend is None else backend.__dict__.get("make_ttgir")
    return bool(getattr(getattr(stage, "__func__", stage), _TRITON_PREFETCH_SENTINEL, False))


def install_triton_prefetch_fix():
    """Patch Triton and Inductor's pool now or on import (imports nothing); False when the kill switch is set."""
    if _triton_8bit_prefetch_fix_mode() in ("0", "off", "false", "no"):
        return False
    pending = False
    for name, hook in _POST_IMPORT.items():
        module = sys.modules.get(name)
        if module is None:
            pending = True
            continue
        try:
            hook(module)
        except Exception:
            pass
    if pending and not any(getattr(f, _TRITON_PREFETCH_SENTINEL, False) for f in sys.meta_path):
        sys.meta_path.insert(0, _PostImportFinder())
    return True
