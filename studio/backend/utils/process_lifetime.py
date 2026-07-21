# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Bind Unsloth child processes to the parent's lifetime so none survive an
abnormal parent exit (terminal-window close, Task Manager "End Task", SIGKILL,
crash) -- the cooperative shutdown path only runs on graceful exits.

Windows: one parent-owned Job Object with JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE.
The parent is assigned to it, children inherit it automatically, and the OS
reaps every process in the job when the parent's last handle closes. Mirrors the
desktop app's job in studio/src-tauri/src/windows_job.rs.

POSIX: each long-lived child sets prctl(PR_SET_PDEATHSIG) on Linux via a tiny
preexec hook (macOS has no equivalent and relies on the cooperative path +
terminate_all). Linux's signal is per-direct-child only, so multiprocessing
workers are also tracked for terminate_all.

Best-effort throughout: any failure degrades to today's behavior, never raises.
Stdlib only.
"""

from __future__ import annotations

import os
import signal
import sys
import threading
from typing import Callable, Optional

_PR_SET_PDEATHSIG = 1
_JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000
_JobObjectExtendedLimitInformation = 9

_lock = threading.Lock()
_initialized = False
_win_job_handle: Optional[int] = None  # retained for the interpreter's lifetime
_tracked_pids: "dict[int, Optional[str]]" = {}  # pid -> identity, reaped by terminate_all


def _is_linux() -> bool:
    return sys.platform.startswith("linux")


def _is_windows() -> bool:
    return sys.platform == "win32"


# ── Parent setup ──


def initialize_parent_lifetime() -> None:
    """Install the parent-death reaper once, as early as possible at startup.

    Windows builds and holds the Job Object; POSIX has nothing to install (the
    guarantee is per-child via preexec). Idempotent and never raises.
    """
    global _initialized
    with _lock:
        if _initialized:
            return
        _initialized = True
        if _is_windows():
            _install_windows_job()


def _win_signatures(kernel32) -> None:
    # Explicit HANDLE-width signatures. Without argtypes, ctypes marshals the
    # 64-bit job/process handles as c_int and truncates them on Win64, so the
    # job calls silently operate on a bogus handle and assignment fails.
    import ctypes
    from ctypes import wintypes

    H, BOOL, DWORD = wintypes.HANDLE, wintypes.BOOL, wintypes.DWORD
    kernel32.CreateJobObjectW.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p]
    kernel32.CreateJobObjectW.restype = H
    kernel32.SetInformationJobObject.argtypes = [
        H,
        ctypes.c_int,
        ctypes.c_void_p,
        DWORD,
    ]
    kernel32.SetInformationJobObject.restype = BOOL
    kernel32.AssignProcessToJobObject.argtypes = [H, H]
    kernel32.AssignProcessToJobObject.restype = BOOL
    kernel32.GetCurrentProcess.argtypes = []
    kernel32.GetCurrentProcess.restype = H
    kernel32.CloseHandle.argtypes = [H]
    kernel32.CloseHandle.restype = BOOL


def _install_windows_job() -> None:
    global _win_job_handle
    try:
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error = True)
        _win_signatures(kernel32)

        class _BASIC(ctypes.Structure):
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

        class _IO(ctypes.Structure):
            _fields_ = [
                (n, ctypes.c_uint64)
                for n in (
                    "ReadOperationCount",
                    "WriteOperationCount",
                    "OtherOperationCount",
                    "ReadTransferCount",
                    "WriteTransferCount",
                    "OtherTransferCount",
                )
            ]

        class _EXT(ctypes.Structure):
            _fields_ = [
                ("BasicLimitInformation", _BASIC),
                ("IoInfo", _IO),
                ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t),
                ("PeakJobMemoryUsed", ctypes.c_size_t),
            ]

        job = kernel32.CreateJobObjectW(None, None)
        if not job:
            return
        info = _EXT()
        info.BasicLimitInformation.LimitFlags = _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not kernel32.SetInformationJobObject(
            job,
            _JobObjectExtendedLimitInformation,
            ctypes.byref(info),
            ctypes.sizeof(info),
        ):
            kernel32.CloseHandle(job)
            return
        # AssignProcessToJobObject(parent) makes children inherit the job. May
        # fail if Unsloth already runs inside an incompatible host job (pre-Win8);
        # degrade to the cooperative path rather than blocking startup.
        if not kernel32.AssignProcessToJobObject(job, kernel32.GetCurrentProcess()):
            kernel32.CloseHandle(job)
            return
        _win_job_handle = job  # hold the handle so the job is not closed early
    except Exception:
        pass


# ── Child binding ──


def _pdeathsig_preexec() -> None:
    # Runs in the forked child before exec. prctl is Linux-only; the getppid
    # check closes the race where the parent died before this ran.
    try:
        import ctypes
        ctypes.CDLL("libc.so.6", use_errno = True).prctl(
            _PR_SET_PDEATHSIG, signal.SIGTERM
        )
        if os.getppid() == 1:
            os._exit(1)
    except Exception:
        pass


def bind_current_process_to_parent_lifetime() -> None:
    """Bind the CURRENT process to its parent's death (Linux). For multiprocessing
    children, which cannot take a preexec_fn, so the parent cannot set
    PR_SET_PDEATHSIG for them -- the child must do it itself at startup."""
    if _is_linux():
        _pdeathsig_preexec()


def compose_preexec(
    existing: Optional[Callable[[], None]],
) -> Optional[Callable[[], None]]:
    """Run the PDEATHSIG hook then any caller-supplied preexec (Linux only)."""
    if not _is_linux():
        return existing
    if existing is None:
        return _pdeathsig_preexec

    def _composed() -> None:
        _pdeathsig_preexec()
        existing()

    return _composed


def child_popen_kwargs(preexec_fn: Optional[Callable[[], None]] = None) -> dict:
    """Popen kwargs that bind a long-lived child to the parent's lifetime.

    On Linux returns a composed ``preexec_fn`` (PDEATHSIG + any existing one);
    empty elsewhere (Windows is covered by the inherited Job Object). Merge via
    ``**child_popen_kwargs()`` alongside the caller's existing kwargs.
    """
    if _is_linux():
        return {"preexec_fn": compose_preexec(preexec_fn)}
    return {}


def _pid_identity(pid: int) -> Optional[str]:
    # Linux /proc starttime (stat field 22); pins identity so a reused pid is not
    # signalled later. None (other platforms / unreadable) disables the check.
    if not _is_linux():
        return None
    try:
        with open(f"/proc/{pid}/stat", encoding = "utf-8") as fh:
            stat = fh.read()
        return stat[stat.rfind(")") + 2 :].split()[19]  # after comm: starttime
    except Exception:
        return None


def forget_pid(pid: Optional[int]) -> None:
    """Stop tracking a child the owner has reaped, so terminate_all never
    signals a recycled pid."""
    if pid:
        _tracked_pids.pop(pid, None)


def adopt_pid(pid: Optional[int]) -> None:
    """Track a child (e.g. a multiprocessing worker started after the parent job
    was set up) and, on Windows, assign it to the job as belt-and-suspenders.
    Tolerates a None or already-exited pid."""
    if not pid:
        return
    _tracked_pids[pid] = _pid_identity(pid)
    if _is_windows() and _win_job_handle:
        try:
            import ctypes
            from ctypes import wintypes

            kernel32 = ctypes.WinDLL("kernel32", use_last_error = True)
            _win_signatures(kernel32)
            kernel32.OpenProcess.argtypes = [
                wintypes.DWORD,
                wintypes.BOOL,
                wintypes.DWORD,
            ]
            kernel32.OpenProcess.restype = wintypes.HANDLE
            PROCESS_SET_QUOTA, PROCESS_TERMINATE = 0x0100, 0x0001
            handle = kernel32.OpenProcess(
                PROCESS_SET_QUOTA | PROCESS_TERMINATE, False, pid
            )
            if handle:
                kernel32.AssignProcessToJobObject(_win_job_handle, handle)
                kernel32.CloseHandle(handle)
        except Exception:
            pass


def terminate_all(timeout: float = 5.0) -> None:
    """Backstop sweep over adopted pids, after per-subsystem cleanup. SIGTERM,
    then SIGKILL the survivors after `timeout`. Skips a pid whose identity no
    longer matches (recycled). Idempotent and teardown-safe."""
    for pid, identity in list(_tracked_pids.items()):
        _tracked_pids.pop(pid, None)
        if identity is not None and _pid_identity(pid) != identity:
            continue  # pid was reused by an unrelated process
        try:
            if _is_windows():
                os.kill(pid, signal.SIGTERM)
                continue
            _posix_terminate(pid, timeout)
        except Exception:
            pass


def _posix_terminate(pid: int, timeout: float = 5.0) -> None:
    # SIGTERM, give the child up to `timeout` to exit, then SIGKILL. Reaping
    # belongs to the child's owner (or init for orphans). Prefer the group
    # (covers grandchildren) when pid leads its own group.
    import time

    killer = os.kill
    try:
        if os.getpgid(pid) == pid:
            killer = os.killpg
    except Exception:
        pass
    try:
        killer(pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    except Exception:
        return
    deadline = time.monotonic() + max(0.0, timeout)
    while time.monotonic() < deadline:
        try:
            killer(pid, 0)  # still alive?
        except ProcessLookupError:
            return
        except Exception:
            break
        time.sleep(0.05)
    try:
        killer(pid, signal.SIGKILL)
    except Exception:
        pass
