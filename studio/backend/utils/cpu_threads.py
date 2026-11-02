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
# Every OpenBLAS worker gets its own buffer and a failed allocation kills the process (#12374); past 8 threads
# numpy gains little on SMT siblings, so the defaults are capped and fitted to the free memory.
_OPENBLAS_DEFAULT_MAX = 8
_OPENBLAS_THREAD_COST = 32 << 20  # numpy's buffer
_OPENBLAS_MEMORY_SHARE = 0.1
_ROCM_OPENBLAS_THREAD_COST = 128 << 20  # rocm-openblas.dll keeps the stock 128 MB buffer


def _threads_that_fit(threads: int, cost: int) -> int:
    headroom = _openblas_memory_headroom()
    if headroom is None:
        return threads
    return max(1, min(threads, int(headroom * _OPENBLAS_MEMORY_SHARE) // cost))


def _windows_affinity_cpus() -> Optional[int]:
    """CPUs in this process's Windows affinity mask when narrower than the system's, else None.

    Python reads no affinity on Windows (process_cpu_count is cpu_count there). Zero masks mean the
    process spans processor groups, and a full group mask says nothing about the others.
    """
    try:
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error = True)
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        get_mask = kernel32.GetProcessAffinityMask
        get_mask.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.POINTER(ctypes.c_size_t),
        ]
        get_mask.restype = wintypes.BOOL
        process_mask, system_mask = ctypes.c_size_t(), ctypes.c_size_t()
        if not get_mask(
            kernel32.GetCurrentProcess(), ctypes.byref(process_mask), ctypes.byref(system_mask)
        ):
            return None
        allowed = bin(process_mask.value).count("1")
        if not allowed or allowed >= bin(system_mask.value).count("1"):
            return None
        return allowed
    except Exception:  # noqa: BLE001 -- unknown affinity means the core count
        return None


def _usable_cpus() -> int:
    """CPUs this process may run on: an affinity mask (taskset, Slurm, cpusets) can be far below the host's."""
    if sys.platform == "win32":
        allowed = _windows_affinity_cpus()
        if allowed is not None:
            return allowed
    count = getattr(os, "process_cpu_count", None)  # 3.13+
    if count is not None:
        return count() or 1
    try:
        return len(os.sched_getaffinity(0)) or 1
    except (AttributeError, OSError):
        return os.cpu_count() or 1


def default_openblas_threads() -> int:
    """About one thread per physical core, at most 8, fewer when memory is short."""
    threads = max(1, min(_OPENBLAS_DEFAULT_MAX, _usable_cpus() // 2))
    return _threads_that_fit(threads, _OPENBLAS_THREAD_COST)


def _openblas_memory_headroom() -> Optional[int]:
    """Bytes OpenBLAS's buffers can still claim, None if unbounded. Windows commits them; Linux only reserves them."""
    try:
        if sys.platform == "win32":
            return _windows_commit_headroom()
        if sys.platform.startswith("linux"):
            return _address_space_headroom()
    except Exception:  # noqa: BLE001 -- unknown headroom means the core-count default
        return None
    return None


def _address_space_headroom() -> Optional[int]:
    import resource

    soft, _ = resource.getrlimit(resource.RLIMIT_AS)
    if soft == resource.RLIM_INFINITY or soft <= 0:
        return None
    with open("/proc/self/statm", encoding = "ascii") as handle:
        used = int(handle.read().split()[0]) * os.sysconf("SC_PAGE_SIZE")
    return max(0, soft - used)


def _windows_commit_headroom() -> Optional[int]:
    import ctypes
    from ctypes import wintypes

    class MemoryStatusEx(ctypes.Structure):
        _fields_ = [("dwLength", wintypes.DWORD), ("dwMemoryLoad", wintypes.DWORD)] + [
            (name, ctypes.c_ulonglong)
            for name in (
                "ullTotalPhys",
                "ullAvailPhys",
                "ullTotalPageFile",
                "ullAvailPageFile",
                "ullTotalVirtual",
                "ullAvailVirtual",
                "ullAvailExtendedVirtual",
            )
        ]

    class BasicLimits(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", ctypes.c_int64),
            ("PerJobUserTimeLimit", ctypes.c_int64),
            ("LimitFlags", wintypes.DWORD),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", wintypes.DWORD),
            ("Affinity", ctypes.c_size_t),
            ("PriorityClass", wintypes.DWORD),
            ("SchedulingClass", wintypes.DWORD),
        ]

    class ExtendedLimits(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", BasicLimits),
            ("IoInfo", ctypes.c_ulonglong * 6),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        ]

    class ProcessMemoryCounters(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
            (name, ctypes.c_size_t)
            for name in (
                "PeakWorkingSetSize",
                "WorkingSetSize",
                "QuotaPeakPagedPoolUsage",
                "QuotaPagedPoolUsage",
                "QuotaPeakNonPagedPoolUsage",
                "QuotaNonPagedPoolUsage",
                "PagefileUsage",
                "PeakPagefileUsage",
                "PrivateUsage",
            )
        ]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error = True)
    status = MemoryStatusEx()
    status.dwLength = ctypes.sizeof(status)
    if not kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        return None
    headroom = int(status.ullAvailPageFile)
    limits = ExtendedLimits()
    # 9 = JobObjectExtendedLimitInformation of this process's job; fails outside any job.
    if kernel32.QueryInformationJobObject(
        None, 9, ctypes.byref(limits), ctypes.sizeof(limits), None
    ):
        flags = limits.BasicLimitInformation.LimitFlags
        if flags & 0x300:  # JOB_OBJECT_LIMIT_PROCESS_MEMORY | JOB_OBJECT_LIMIT_JOB_MEMORY
            counters = ProcessMemoryCounters()
            counters.cb = ctypes.sizeof(counters)
            kernel32.GetCurrentProcess.restype = wintypes.HANDLE
            kernel32.K32GetProcessMemoryInfo.argtypes = [
                wintypes.HANDLE,
                ctypes.c_void_p,
                wintypes.DWORD,
            ]
            used = 0
            if kernel32.K32GetProcessMemoryInfo(
                kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb
            ):
                used = int(counters.PrivateUsage)
            if flags & 0x100:
                headroom = min(headroom, max(0, int(limits.ProcessMemoryLimit) - used))
            if flags & 0x200:  # the whole job's commit counts, so its peak, never just this process
                job_used = max(used, int(limits.PeakJobMemoryUsed))
                headroom = min(headroom, max(0, int(limits.JobMemoryLimit) - job_used))
    return headroom


def configure_cpu_threads(env: Optional[MutableMapping[str, str]] = None) -> None:
    """Applies UNSLOTH_CPU_THREADS to CPU pools when set, else caps OpenBLAS; must run before OpenMP or BLAS imports."""
    environ = os.environ if env is None else env
    configured = environ.get("UNSLOTH_CPU_THREADS", "").strip()
    if not configured:
        # OpenBLAS reads a blank value as 0 (one thread per core), so blank counts as unset.
        if not environ.get("OPENBLAS_NUM_THREADS", "").strip():
            value = str(default_openblas_threads())
            environ["OPENBLAS_NUM_THREADS"] = value
            if env is None:
                # Lets spawned workers tell this default from a user's own value.
                environ[_OPENBLAS_DEFAULT_MARKER] = value
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


# AMD's Windows ROCm torch wheels ship this OpenBLAS, which ignores OPENBLAS_NUM_THREADS; its runtime setter works (#12942).
_RUNTIME_CAPPED_OPENBLAS = ("rocm-openblas.dll",)
_OPENBLAS_CAP_SENTINEL = "_unsloth_openblas_cap_finder"
_OPENBLAS_DEFAULT_MARKER = "UNSLOTH_OPENBLAS_DEFAULTED"


def _torch_thread_count() -> Optional[int]:
    try:
        count = int(sys.modules["torch"].get_num_threads())
    except Exception:  # noqa: BLE001
        return None
    return count if count >= 1 else None


def _openblas_thread_target(environ: Optional[MutableMapping[str, str]] = None) -> Optional[int]:
    environ = os.environ if environ is None else environ
    raw = environ.get("OPENBLAS_NUM_THREADS", "").strip()
    target = None
    if raw and raw == environ.get(_OPENBLAS_DEFAULT_MARKER):
        # Studio's default is sized for numpy; this DLL is torch's CPU BLAS, so it gets torch's thread count.
        target = _torch_thread_count()
        if target is not None:
            target = _threads_that_fit(target, _ROCM_OPENBLAS_THREAD_COST)
    if target is None:
        try:
            target = int(raw)
        except ValueError:
            return None
    if target < 1:
        return None
    # OpenBLAS clamps an env value to the core count; its runtime setter only to MAX_THREADS.
    return min(target, os.cpu_count() or target, _usable_cpus())


def runtime_capped_openblas_loaded() -> bool:
    """Whether torch's CPU BLAS is AMD's Windows ROCm OpenBLAS: about 30x slower per core than MKL (#14313)."""
    if sys.platform != "win32":
        return False
    try:
        import ctypes

        get_module = ctypes.WinDLL("kernel32", use_last_error = True).GetModuleHandleW
        get_module.argtypes = [ctypes.c_wchar_p]
        get_module.restype = ctypes.c_void_p
        return any(get_module(name) for name in _RUNTIME_CAPPED_OPENBLAS)
    except Exception:  # noqa: BLE001
        return False


def apply_openblas_runtime_cap() -> list:
    """Hand the thread target to an already loaded OpenBLAS that ignores the env; returns the DLLs capped."""
    if sys.platform != "win32":
        return []
    target = _openblas_thread_target()
    if target is None:
        return []
    capped = []
    try:
        import ctypes

        # Private handle: argtypes on ctypes.windll's shared one would leak to every caller.
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


_affinity_lock = threading.Lock()
_torch_threads_unfitted: Optional[int] = None
_fitted_affinity: Optional[int] = None


def fit_threads_to_affinity() -> None:
    """Fit torch's pool on this thread, and the ROCm OpenBLAS one, to the CPUs Windows lets this process use.

    Affinity is often narrowed after launch (Task Manager), and neither pool reads it: 16 threads on one
    allowed CPU spin against each other (#14313). Widening it again restores the launch counts. Cheap
    enough to call before every CPU batch; a no-op off Windows, before torch loads, or while unrestricted.
    """
    if sys.platform != "win32" or "torch" not in sys.modules:
        return
    global _torch_threads_unfitted, _fitted_affinity
    with _affinity_lock:
        allowed = _windows_affinity_cpus()
        if allowed is None and _torch_threads_unfitted is None:
            return
        try:
            torch = sys.modules["torch"]
            current = int(torch.get_num_threads())
            if _torch_threads_unfitted is None:
                _torch_threads_unfitted = current
            target = (
                _torch_threads_unfitted
                if allowed is None
                else min(_torch_threads_unfitted, allowed)
            )
            # Per thread under OpenMP: the setter reaches only the calling thread's pool.
            if current != target:
                torch.set_num_threads(target)
        except Exception:  # noqa: BLE001 -- a thread cap must never break a CPU batch
            return
        if allowed != _fitted_affinity:
            _fitted_affinity = allowed
            apply_openblas_runtime_cap()


class _TorchLoader(importlib.abc.Loader):
    __slots__ = ("_loader",)

    def __init__(self, loader):
        self._loader = loader

    def create_module(self, spec):
        create = getattr(self._loader, "create_module", None)
        return None if create is None else create(spec)

    def exec_module(self, module):
        try:
            self._loader.exec_module(module)
        finally:
            # pkg_resources dispatches on the loader's type, and a reload would wrap this wrapper again.
            spec = getattr(module, "__spec__", None)
            if spec is not None and spec.loader is self:
                spec.loader = self._loader
            if getattr(module, "__loader__", None) is self:
                module.__loader__ = self._loader
        apply_openblas_runtime_cap()
        fit_threads_to_affinity()

    def __getattr__(self, attribute):
        if attribute == "_loader":
            raise AttributeError(attribute)
        return getattr(self._loader, attribute)


class _TorchImportFinder(importlib.abc.MetaPathFinder):
    __slots__ = (_OPENBLAS_CAP_SENTINEL, "_finding")

    def __init__(self):
        setattr(self, _OPENBLAS_CAP_SENTINEL, True)
        # Per thread: a bare find_spec("torch") on another thread must not hide the warm thread's import.
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
            # Not caught: a later finder's error must fail the import exactly as without this one.
            spec = importlib.util.find_spec(fullname)
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
    """Cap now if torch is imported, else right after it is. Windows only; idempotent."""
    if sys.platform != "win32":
        return False
    if "torch" in sys.modules:
        apply_openblas_runtime_cap()
        fit_threads_to_affinity()
        return True
    if not any(getattr(finder, _OPENBLAS_CAP_SENTINEL, False) for finder in sys.meta_path):
        sys.meta_path.insert(0, _TorchImportFinder())
    return True
