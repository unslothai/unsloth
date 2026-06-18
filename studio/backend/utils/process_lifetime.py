# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Bind Studio child processes to the parent's lifetime so none survive an
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
_tracked_pids: set[int] = set()  # adopted children, reaped by terminate_all


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


def _install_windows_job() -> None:
    global _win_job_handle
    try:
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error = True)

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
            _fields_ = [(n, ctypes.c_uint64) for n in (
                "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
                "ReadTransferCount", "WriteTransferCount", "OtherTransferCount",
            )]

        class _EXT(ctypes.Structure):
            _fields_ = [
                ("BasicLimitInformation", _BASIC),
                ("IoInfo", _IO),
                ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t),
                ("PeakJobMemoryUsed", ctypes.c_size_t),
            ]

        kernel32.CreateJobObjectW.restype = wintypes.HANDLE
        kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
        kernel32.SetInformationJobObject.restype = wintypes.BOOL

        job = kernel32.CreateJobObjectW(None, None)
        if not job:
            return
        info = _EXT()
        info.BasicLimitInformation.LimitFlags = _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not kernel32.SetInformationJobObject(
            job, _JobObjectExtendedLimitInformation, ctypes.byref(info), ctypes.sizeof(info)
        ):
            kernel32.CloseHandle(job)
            return
        # AssignProcessToJobObject(parent) makes children inherit the job. May
        # fail if Studio already runs inside an incompatible host job (pre-Win8);
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
        ctypes.CDLL("libc.so.6", use_errno = True).prctl(_PR_SET_PDEATHSIG, signal.SIGTERM)
        if os.getppid() == 1:
            os._exit(1)
    except Exception:
        pass


def compose_preexec(existing: Optional[Callable[[], None]]) -> Optional[Callable[[], None]]:
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


def adopt_pid(pid: Optional[int]) -> None:
    """Track a child (e.g. a multiprocessing worker started after the parent job
    was set up) and, on Windows, assign it to the job as belt-and-suspenders.
    Tolerates a None or already-exited pid."""
    if not pid:
        return
    _tracked_pids.add(pid)
    if _is_windows() and _win_job_handle:
        try:
            import ctypes
            from ctypes import wintypes

            kernel32 = ctypes.WinDLL("kernel32", use_last_error = True)
            kernel32.OpenProcess.restype = wintypes.HANDLE
            PROCESS_SET_QUOTA, PROCESS_TERMINATE = 0x0100, 0x0001
            handle = kernel32.OpenProcess(PROCESS_SET_QUOTA | PROCESS_TERMINATE, False, pid)
            if handle:
                kernel32.AssignProcessToJobObject(_win_job_handle, handle)
                kernel32.CloseHandle(handle)
        except Exception:
            pass


def terminate_all(timeout: float = 5.0) -> None:
    """Backstop sweep over adopted pids, after per-subsystem cleanup. POSIX
    signals the process group when the child leads one. Idempotent and
    teardown-safe (snapshots the registry, swallows everything)."""
    for pid in list(_tracked_pids):
        _tracked_pids.discard(pid)
        try:
            if _is_windows():
                os.kill(pid, signal.SIGTERM)
                continue
            _posix_terminate(pid)
        except Exception:
            pass


def _posix_terminate(pid: int) -> None:
    # Signal only; reaping belongs to the child's owner (or init for orphans).
    # Prefer the group (covers grandchildren) when pid leads its own group.
    killer = os.kill
    try:
        if os.getpgid(pid) == pid:
            killer = os.killpg
    except Exception:
        pass
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            killer(pid, sig)
        except ProcessLookupError:
            return
        except Exception:
            return
