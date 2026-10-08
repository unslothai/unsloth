# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Early CPU thread-pool configuration for Unsloth processes."""

import importlib.abc
import importlib.util
import os
import sys
import threading
from typing import MutableMapping, Optional


_THREAD_POOL_ENV_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def configure_cpu_threads(env: Optional[MutableMapping[str, str]] = None) -> None:
    """Apply ``UNSLOTH_CPU_THREADS`` to native CPU pools when configured, else cap OpenBLAS at one thread.

    Must run before importing libraries that initialize an OpenMP or BLAS
    pool. Library-specific vars are left untouched so users can override a
    single runtime independently.
    """
    environ = os.environ if env is None else env
    configured = environ.get("UNSLOTH_CPU_THREADS", "").strip()
    if not configured:
        # OpenBLAS reads a blank value as 0 (one thread per core), so blank counts as unset.
        if not environ.get("OPENBLAS_NUM_THREADS", "").strip():
            environ["OPENBLAS_NUM_THREADS"] = "1"
        if env is None:
            install_openblas_runtime_cap()
        return

    try:
        thread_count = int(configured)
    except ValueError as exc:
        raise ValueError("UNSLOTH_CPU_THREADS must be a positive integer") from exc
    if thread_count < 1:
        raise ValueError("UNSLOTH_CPU_THREADS must be a positive integer")

    value = str(thread_count)
    for variable in _THREAD_POOL_ENV_VARS:
        environ.setdefault(variable, value)
    if env is None:
        install_openblas_runtime_cap()


# AMD's Windows ROCm torch wheels load this OpenBLAS, which ignores OPENBLAS_NUM_THREADS and
# OMP_NUM_THREADS and starts one worker per logical CPU (measured on gfx1151: 32 with
# OPENBLAS_NUM_THREADS=1), so the cap above never reaches it; its runtime setter does (#12942).
_RUNTIME_CAPPED_OPENBLAS = ("rocm-openblas.dll",)
_OPENBLAS_CAP_SENTINEL = "_unsloth_openblas_cap_finder"


def _openblas_thread_target(environ: Optional[MutableMapping[str, str]] = None) -> Optional[int]:
    environ = os.environ if environ is None else environ
    try:
        target = int(environ.get("OPENBLAS_NUM_THREADS", "").strip())
    except ValueError:
        return None
    if target < 1:
        return None
    # OpenBLAS clamps an env value to the core count; its runtime setter only to MAX_THREADS.
    return min(target, os.cpu_count() or target)


def apply_openblas_runtime_cap() -> list:
    """Hand OPENBLAS_NUM_THREADS to an already loaded OpenBLAS that ignores it. Windows only; never loads a DLL.

    Returns the DLL names capped, empty when there was nothing to do."""
    if sys.platform != "win32":
        return []
    target = _openblas_thread_target()
    if target is None:
        return []
    capped = []
    try:
        import ctypes

        # A private handle: argtypes / restype on ctypes.windll's shared one would change it for every caller.
        get_module = ctypes.WinDLL("kernel32", use_last_error = True).GetModuleHandleW
        get_module.argtypes = [ctypes.c_wchar_p]
        get_module.restype = ctypes.c_void_p
        for name in _RUNTIME_CAPPED_OPENBLAS:
            handle = get_module(name)
            if not handle:
                continue
            setter = getattr(ctypes.CDLL(name, handle = handle), "openblas_set_num_threads", None)
            if setter is None:
                continue
            setter.argtypes = [ctypes.c_int]
            setter.restype = None
            setter(target)
            capped.append(name)
    except Exception:  # noqa: BLE001 -- a thread cap must never break startup or a torch import
        return capped
    return capped


class _TorchLoader(importlib.abc.Loader):
    """torch's real loader, then the cap once the module body (and so its DLLs) has loaded."""

    __slots__ = ("_loader",)

    def __init__(self, loader):
        self._loader = loader

    def create_module(self, spec):
        create = getattr(self._loader, "create_module", None)
        return None if create is None else create(spec)

    def exec_module(self, module):
        self._loader.exec_module(module)
        apply_openblas_runtime_cap()

    def __getattr__(self, attribute):
        if attribute == "_loader":
            raise AttributeError(attribute)
        return getattr(self._loader, attribute)


class _TorchImportFinder(importlib.abc.MetaPathFinder):
    """At the FRONT of sys.meta_path, wrapping only the top-level ``torch`` import's loader."""

    __slots__ = (_OPENBLAS_CAP_SENTINEL, "_finding")

    def __init__(self):
        setattr(self, _OPENBLAS_CAP_SENTINEL, True)
        # Per thread: find_spec below walks sys.meta_path again, and a bare find_spec("torch") on another
        # thread (Studio has several, outside any import) must not hide the warm thread's real import.
        self._finding = threading.local()

    def find_spec(
        self,
        fullname,
        path = None,
        target = None,
    ):
        if fullname != "torch" or getattr(self._finding, "active", False):
            return None
        self._finding.active = True
        try:
            spec = importlib.util.find_spec(fullname)
        except Exception:  # noqa: BLE001
            return None
        finally:
            self._finding.active = False
        if spec is None or spec.loader is None or not hasattr(spec.loader, "exec_module"):
            return None
        try:
            spec.loader = _TorchLoader(spec.loader)
        except Exception:  # noqa: BLE001
            return None
        return spec


def install_openblas_runtime_cap() -> bool:
    """Apply the cap now if torch is already imported, else right after it is. Windows only; idempotent."""
    if sys.platform != "win32":
        return False
    if "torch" in sys.modules:
        apply_openblas_runtime_cap()
        return True
    if not any(getattr(finder, _OPENBLAS_CAP_SENTINEL, False) for finder in sys.meta_path):
        sys.meta_path.insert(0, _TorchImportFinder())
    return True
