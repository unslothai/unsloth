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
preexec hook. Linux's signal is per-direct-child only, so multiprocessing
workers are also tracked for terminate_all.

macOS has neither mechanism, so tracked children are also recorded on disk and
the next startup sweeps whatever the previous run left behind
(reap_recorded_children). That record is the only reaper macOS has after a
crash, a Force Quit or a closed terminal.

Best-effort throughout: any failure degrades to today's behavior, never raises.
Stdlib only.
"""

from __future__ import annotations

import os
import signal
import sys
import contextvars
import functools
import threading
import time
from typing import Callable, Optional

_PR_SET_PDEATHSIG = 1
_JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000
_JobObjectExtendedLimitInformation = 9

# killpg(1, sig) is kill(-1, sig): signals everything the user owns. Never record or signal < 2.
_LOWEST_SIGNALABLE_PID = 2


def is_signalable_pid(pid: object) -> bool:
    """Excludes bool, since True would read as pid 1; public so every signalling site shares this floor."""
    return isinstance(pid, int) and not isinstance(pid, bool) and pid >= _LOWEST_SIGNALABLE_PID


_signalable = is_signalable_pid


_lock = threading.Lock()
_spawner_lock = threading.Lock()
_spawner: "Optional[_Spawner]" = None
_initialized = False
_win_job_handle: Optional[int] = None
_tracked_pids: "dict[int, Optional[str]]" = {}
# The group outlives an exited leader and is the only handle on its children.
_tracked_pgids: "dict[int, int]" = {}
# Serialises edit-then-write so concurrent adopts do not drop a pid.
_record_lock = threading.Lock()


# A silent failure leaks every child on a crash, so record and log it.
_win_job_status: "tuple[bool, str]" = (False, "not attempted")


def _last_error(ctypes_module) -> int:
    getter = getattr(ctypes_module, "get_last_error", None)
    try:
        return int(getter()) if getter else 0
    except Exception:
        return 0


def _record_job_status(
    ok: bool,
    detail: str,
    last_error: int = 0,
) -> None:
    global _win_job_status
    if last_error:
        detail = f"{detail} (WinError {last_error})"
    _win_job_status = (ok, detail)
    try:
        import logging
        logger = logging.getLogger(__name__)
        if ok:
            logger.info("Child-process cleanup on abnormal exit: %s", detail)
        else:
            logger.warning(
                "Child-process cleanup on abnormal exit is NOT guaranteed: %s. Children may "
                "survive a crash or a force quit; the startup sweep reaps them on the next "
                "launch.",
                detail,
            )
    except Exception:
        pass


def windows_job_status() -> "tuple[bool, str]":
    """(in_force, detail) for the Windows kill-on-close job."""
    return _win_job_status


def _is_linux() -> bool:
    return sys.platform.startswith("linux")


def _is_windows() -> bool:
    return sys.platform == "win32"


def initialize_parent_lifetime() -> None:
    """Idempotent and never raises; on Windows it holds the Job Object, while POSIX installs nothing."""
    global _initialized
    with _lock:
        if _initialized:
            return
        _initialized = True
        if _is_windows():
            _install_windows_job()
        elif _is_linux():
            if _pdeathsig_available():
                _record_job_status(True, "PR_SET_PDEATHSIG per child")
            else:
                _record_job_status(False, "prctl is unavailable here (seccomp or container policy)")
        else:
            _record_job_status(False, "no kernel-level parent-death signal on this platform")


def _pdeathsig_available() -> bool:
    """A read-only PR_GET proves nothing under seccomp, so the probe does a no-op PR_SET instead."""
    _PR_GET_PDEATHSIG, _PR_SET_PDEATHSIG = 2, 1
    try:
        import ctypes

        libc = ctypes.CDLL("libc.so.6", use_errno = True)
        current = ctypes.c_int(0)
        if libc.prctl(_PR_GET_PDEATHSIG, ctypes.byref(current), 0, 0, 0) != 0:
            return False
        return libc.prctl(_PR_SET_PDEATHSIG, current.value, 0, 0, 0) == 0
    except Exception:
        return False


def _win_signatures(kernel32) -> None:
    # Explicit HANDLE signatures: c_int defaults truncate handles on Win64.
    import ctypes
    from ctypes import wintypes

    H, BOOL, DWORD = wintypes.HANDLE, wintypes.BOOL, wintypes.DWORD
    kernel32.CreateJobObjectW.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p]
    kernel32.CreateJobObjectW.restype = H
    kernel32.SetInformationJobObject.argtypes = [H, ctypes.c_int, ctypes.c_void_p, DWORD]
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
            _record_job_status(False, "CreateJobObjectW failed", _last_error(ctypes))
            return
        info = _EXT()
        info.BasicLimitInformation.LimitFlags = _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not kernel32.SetInformationJobObject(
            job, _JobObjectExtendedLimitInformation, ctypes.byref(info), ctypes.sizeof(info)
        ):
            _record_job_status(False, "SetInformationJobObject failed", _last_error(ctypes))
            kernel32.CloseHandle(job)
            return
        # May fail inside an incompatible host job (pre-Win8); degrade, do not block startup.
        if not kernel32.AssignProcessToJobObject(job, kernel32.GetCurrentProcess()):
            _record_job_status(False, "AssignProcessToJobObject failed", _last_error(ctypes))
            kernel32.CloseHandle(job)
            return
        _win_job_handle = job
        _record_job_status(True, "kill-on-close job installed")
    except Exception as error:
        _record_job_status(False, f"{type(error).__name__}: {error}")


def _pdeathsig_preexec(owner_pid: Optional[int] = None) -> None:
    # Runs pre-exec in the child. Compare getppid to owner_pid, not 1: a parent can legitimately
    # be pid 1 (container entrypoint), and subreapers are not pid 1.
    try:
        import ctypes

        ctypes.CDLL("libc.so.6", use_errno = True).prctl(_PR_SET_PDEATHSIG, signal.SIGTERM)
        parent_pid = os.getppid()
        orphaned = parent_pid != owner_pid if owner_pid is not None else parent_pid == 1
        if orphaned:
            os._exit(1)
    except Exception:
        pass


def bind_current_process_to_parent_lifetime() -> None:
    """Bind the CURRENT process to its parent's death (Linux). For multiprocessing
    children, which cannot take a preexec_fn, so the parent cannot set
    PR_SET_PDEATHSIG for them -- the child must do it itself at startup."""
    if not _is_linux():
        return
    parent = None
    try:
        import multiprocessing
        parent = multiprocessing.parent_process()
    except Exception:
        pass
    if parent is None:
        # Only arm PDEATHSIG: a getppid() == 1 test would kill a parent-is-pid-1 process.
        _pdeathsig_preexec(os.getppid())
        return
    # Orphan check uses the creator's sentinel, not a pid compare: under forkserver the kernel
    # parent is the fork server.
    _pdeathsig_preexec(os.getppid())
    try:
        if not parent.is_alive():
            os._exit(1)
    except Exception:
        pass


def allow_child_processes() -> None:
    """Grandchildren inherit the cleared flag and hang worker exit unless they pass daemon = True."""
    try:
        from multiprocessing import process as multiprocessing_process
        config = getattr(multiprocessing_process.current_process(), "_config", None)
        if isinstance(config, dict):
            config["daemon"] = False
    except Exception:
        pass


def compose_preexec(
    existing: Optional[Callable[[], None]], owner_pid: Optional[int] = None
) -> Optional[Callable[[], None]]:
    """Run the PDEATHSIG hook then any caller-supplied preexec (Linux only)."""
    if not _is_linux():
        return existing

    def _composed() -> None:
        _pdeathsig_preexec(owner_pid)
        if existing is not None:
            existing()

    return _composed


_fork_reset_installed = False


def _reset_after_fork() -> None:
    """A fork child inherits both locks in whatever state they were in and a
    _spawner whose thread does not exist here. Start clean instead of deadlocking."""
    global _spawner, _spawner_lock, _record_lock, _owner_identity, _shutdown_latch
    _spawner_lock = threading.Lock()
    _owner_identity = None
    _tracked_pids.clear()
    _tracked_pgids.clear()
    # Recreate after fork: a lock held by another thread at fork time is never released.
    _record_lock = threading.Lock()
    # Event has an internal lock too; rebuild it keeping its latched state.
    _was_latched = _shutdown_latch.is_set()
    _shutdown_latch = threading.Event()
    if _was_latched:
        _shutdown_latch.set()
    _spawner = None


class _UncachedReformatters(threading.local):
    """PyAV's per-thread scaler cache while a fork is in flight: reads find nothing, writes are dropped."""

    def __setattr__(self, name: str, value: object) -> None:
        if name != "reformatter":
            super().__setattr__(name, value)


_forks_in_flight = 0


def _suspend_native_caches() -> None:
    """Frees PyAV's FFmpeg scaler caches before a fork; a child would hang waiting on threads it lacks."""
    global _forks_in_flight
    _forks_in_flight += 1
    try:
        frame = sys.modules.get("av.video.frame")
        cache = getattr(frame, "_thread_local", None)
        if isinstance(cache, threading.local) and not isinstance(cache, _UncachedReformatters):
            frame._thread_local = _UncachedReformatters()
    except Exception:  # noqa: BLE001 - a fork hook must not get in the fork's way
        pass


def _resume_native_caches() -> None:
    try:
        frame = sys.modules.get("av.video.frame")
        if isinstance(getattr(frame, "_thread_local", None), _UncachedReformatters):
            frame._thread_local = threading.local()
    except Exception:  # noqa: BLE001
        pass


def _resume_native_caches_in_parent() -> None:
    global _forks_in_flight
    _forks_in_flight = max(0, _forks_in_flight - 1)
    # Another thread's fork may still be between its own hooks.
    if _forks_in_flight == 0:
        _resume_native_caches()


def _resume_native_caches_in_child() -> None:
    global _forks_in_flight
    _forks_in_flight = 0
    _resume_native_caches()


# At import, not lazily: run_server imports this module before anything spawns, and every fork needs it, including
# the ones that bring their own preexec_fn or call os.fork directly.
if hasattr(os, "register_at_fork"):
    os.register_at_fork(
        before = _suspend_native_caches,
        after_in_parent = _resume_native_caches_in_parent,
        after_in_child = _resume_native_caches_in_child,
    )


def _adopt_fork_reset() -> None:
    """Register the child-side reset once, lazily. Best-effort like the rest of
    this module: os.register_at_fork is POSIX-only and absent on Windows."""
    global _fork_reset_installed
    if _fork_reset_installed:
        return
    _fork_reset_installed = True
    try:
        os.register_at_fork(after_in_child = _reset_after_fork)
    except (AttributeError, RuntimeError):
        pass


def spawn_on_lifetime_thread(spawn: Callable[[], object]) -> object:
    """PR_SET_PDEATHSIG fires when the forking thread exits, so spawn from a process-lifetime thread."""
    if not _is_linux():
        return spawn()
    global _spawner
    _adopt_fork_reset()
    with _spawner_lock:
        if _spawner is None or not _spawner.usable():
            candidate = _Spawner()
            _spawner = candidate if candidate.usable() else None
        spawner = _spawner
    # No helper thread (tests replace threading.Thread): spawn inline.
    return spawn() if spawner is None else spawner.run(spawn)


class _Spawner:
    """One daemon thread that forks on behalf of any caller. Daemon is correct:
    the process dies at interpreter exit, which is when children should die."""

    def __init__(self) -> None:
        import queue

        self._jobs: "queue.Queue" = queue.Queue()
        self._ready = threading.Event()
        self._thread = None
        try:
            self._thread = threading.Thread(
                target = self._run, name = "unsloth-child-spawner", daemon = True
            )
            self._thread.start()
        except Exception:  # noqa: BLE001 - fall back to an inline spawn
            self._thread = None

    def usable(self) -> bool:
        """True only once the helper has actually begun running."""
        if self._thread is None:
            return False
        return self._ready.wait(5.0) and bool(getattr(self._thread, "is_alive", bool)())

    def _run(self) -> None:
        self._ready.set()
        while True:
            spawn, box, done = self._jobs.get()
            try:
                box.append((True, spawn()))
            except BaseException as exc:  # noqa: BLE001 - re-raised in the caller
                box.append((False, exc))
            finally:
                done.set()

    def run(self, spawn: Callable[[], object]) -> object:
        box: list = []
        done = threading.Event()
        self._jobs.put((functools.partial(contextvars.copy_context().run, spawn), box, done))
        done.wait()
        ok, value = box[0]
        if not ok:
            raise value
        return value


def child_popen_kwargs(preexec_fn: Optional[Callable[[], None]] = None) -> dict:
    """The returned preexec_fn sets PDEATHSIG on Linux; Windows relies on the inherited Job Object."""
    if _is_linux():
        # getpid() in the spawner lets the child tell reparenting from a pid-1 parent.
        return {"preexec_fn": compose_preexec(preexec_fn, os.getpid())}
    return {}


def _recorded_identity(value: object) -> "Optional[str]":
    """An identity read back from a record, or None when it is not one."""
    return value if isinstance(value, str) and value else None


def _same_identity(recorded: str, current: str) -> bool:
    """Old Linux records hold starttime:comm; compare only the start time there, else the whole string."""
    # A malformed record must not raise, or one bad file aborts the whole sweep.
    if not isinstance(recorded, str) or not isinstance(current, str):
        return False
    if _is_linux():
        return recorded.split(":", 1)[0] == current.split(":", 1)[0]
    if sys.platform == "darwin":
        # lstart only: a framework python re-execs into Python.app, changing comm.
        recorded_start, current_start = recorded.split()[:5], current.split()[:5]
        if len(recorded_start) == 5 and len(current_start) == 5:
            return recorded_start == current_start
    return recorded == current


def _pid_identity(pid: int) -> Optional[str]:
    if _is_linux():
        try:
            with open(f"/proc/{pid}/stat", encoding = "utf-8") as fh:
                stat = fh.read()
            # Start time only: comm is mutable, so a renamed child would be dropped unsignalled.
            return stat[stat.rfind(")") + 2 :].split()[19]
        except Exception:
            return None
    if _is_windows():
        try:
            import ctypes
            from ctypes import wintypes

            PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
            kernel32 = ctypes.WinDLL("kernel32", use_last_error = True)
            kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
            kernel32.OpenProcess.restype = wintypes.HANDLE
            kernel32.GetProcessTimes.argtypes = [wintypes.HANDLE] + [
                ctypes.POINTER(wintypes.FILETIME)
            ] * 4
            kernel32.GetProcessTimes.restype = wintypes.BOOL
            kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
            kernel32.CloseHandle.restype = wintypes.BOOL
            handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
            if not handle:
                return None
            try:
                created = wintypes.FILETIME()
                other = [wintypes.FILETIME() for _ in range(3)]
                if not kernel32.GetProcessTimes(
                    handle, ctypes.byref(created), *[ctypes.byref(x) for x in other]
                ):
                    return None
                return f"{created.dwHighDateTime}:{created.dwLowDateTime}"
            finally:
                kernel32.CloseHandle(handle)
        except Exception:
            return None
    if sys.platform == "darwin":
        try:
            import subprocess

            # TZ pinned: lstart is formatted in local time.
            out = subprocess.run(
                ["ps", "-o", "lstart=,comm=", "-p", str(pid)],
                capture_output = True,
                text = True,
                encoding = "utf-8",
                errors = "replace",
                timeout = 5,
                env = {**os.environ, "TZ": "UTC"},
            )
            line = (out.stdout or "").strip()
            return line or None
        except Exception:
            return None
    return None


def forget_pid(pid: Optional[int]) -> None:
    """Kept while its process group has members: the shim can exit before the server it started."""
    if not pid:
        return
    with _record_lock:
        if pid not in _tracked_pids and pid not in _tracked_pgids:
            return
        if _group_has_members(_tracked_pgids.get(pid)):
            return
        _tracked_pids.pop(pid, None)
        _tracked_pgids.pop(pid, None)
        _write_breadcrumb()


def _group_has_members(pgid: object) -> bool:
    """killpg(pgid, 0) also succeeds for zombie-only groups, which would then keep their record forever."""
    # killpg(1, 0) always succeeds, so without the floor a poisoned pgid retries forever.
    if not _signalable(pgid) or _is_windows() or not hasattr(os, "killpg"):
        return False
    try:
        os.killpg(pgid, 0)
    except Exception:
        return False
    # Check the leader first: enumerating a group scans every process (about 62ms at 6000).
    if _pid_alive(pgid) and not _pid_is_zombie(pgid):
        return True
    members = _group_member_pids(pgid)
    if members is None:
        return True
    return any(not _pid_is_zombie(pid) for pid in members)


def _windows_creation_time(identity: "Optional[str]") -> "Optional[int]":
    """Parses the high:low identity to one 64-bit FILETIME, so creation times can be ordered."""
    if not isinstance(identity, str):
        return None
    parts = identity.split(":")
    if len(parts) != 2:
        return None
    try:
        return (int(parts[0]) << 32) | int(parts[1])
    except ValueError:
        return None


def _windows_identity_of_handle(kernel32, handle) -> "Optional[str]":
    """Read through an open handle, which pins the process; None means not proven, never a match."""
    try:
        import ctypes
        from ctypes import wintypes

        kernel32.GetProcessTimes.argtypes = [wintypes.HANDLE] + [
            ctypes.POINTER(wintypes.FILETIME)
        ] * 4
        kernel32.GetProcessTimes.restype = wintypes.BOOL
        created = wintypes.FILETIME()
        other = [wintypes.FILETIME() for _ in range(3)]
        if not kernel32.GetProcessTimes(
            handle, ctypes.byref(created), *[ctypes.byref(x) for x in other]
        ):
            return None
        return f"{created.dwHighDateTime}:{created.dwLowDateTime}"
    except Exception:  # noqa: BLE001 -- unreadable is "not proven", handled by the caller
        return None


def _windows_terminate_through_a_handle(pid: int, identity: "Optional[str]") -> "Optional[bool]":
    """Kills via a proven handle, since taskkill /PID and os.kill re-resolve the reusable pid name."""
    if identity is None:
        return None
    try:
        import ctypes
        from ctypes import wintypes

        PROCESS_TERMINATE = 0x0001
        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        kernel32 = ctypes.WinDLL("kernel32", use_last_error = True)
        kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.TerminateProcess.argtypes = [wintypes.HANDLE, wintypes.UINT]
        kernel32.TerminateProcess.restype = wintypes.BOOL
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel32.CloseHandle.restype = wintypes.BOOL
        handle = kernel32.OpenProcess(
            PROCESS_TERMINATE | PROCESS_QUERY_LIMITED_INFORMATION, False, pid
        )
        if not handle:
            # Gone, protected, or denied: cannot answer, not a wrong process.
            return None
        try:
            opened = _windows_identity_of_handle(kernel32, handle)
            if opened is None or opened != identity:
                return False
            return bool(kernel32.TerminateProcess(handle, 1))
        finally:
            kernel32.CloseHandle(handle)
    except Exception:  # noqa: BLE001 -- no ctypes, no kernel32: the caller falls back
        return None


def _windows_filetime_now() -> "Optional[int]":
    """Wall clock as a FILETIME; a process created after a pre-snapshot reading was not in the snapshot."""
    if not _is_windows():
        return None
    try:
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error = True)
        kernel32.GetSystemTimeAsFileTime.argtypes = [ctypes.POINTER(wintypes.FILETIME)]
        kernel32.GetSystemTimeAsFileTime.restype = None
        stamp = wintypes.FILETIME()
        kernel32.GetSystemTimeAsFileTime(ctypes.byref(stamp))
        return (int(stamp.dwHighDateTime) << 32) | int(stamp.dwLowDateTime)
    except Exception:  # noqa: BLE001 -- cannot tell; the walk keeps its old behaviour
        return None


def _windows_child_pid_map() -> "Optional[dict[int, list[int]]]":
    """Uses ctypes and a Toolhelp snapshot, not psutil or a helper, since teardown must not spawn one."""
    try:
        import ctypes
        from ctypes import wintypes

        TH32CS_SNAPPROCESS = 0x0000_0002
        MAX_PATH = 260

        class PROCESSENTRY32W(ctypes.Structure):
            _fields_ = [
                ("dwSize", wintypes.DWORD),
                ("cntUsage", wintypes.DWORD),
                ("th32ProcessID", wintypes.DWORD),
                ("th32DefaultHeapID", ctypes.c_size_t),
                ("th32ModuleID", wintypes.DWORD),
                ("cntThreads", wintypes.DWORD),
                ("th32ParentProcessID", wintypes.DWORD),
                ("pcPriClassBase", ctypes.c_long),
                ("dwFlags", wintypes.DWORD),
                ("szExeFile", wintypes.WCHAR * MAX_PATH),
            ]

        kernel32 = ctypes.WinDLL("kernel32", use_last_error = True)
        kernel32.CreateToolhelp32Snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
        kernel32.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
        kernel32.Process32FirstW.argtypes = [wintypes.HANDLE, ctypes.POINTER(PROCESSENTRY32W)]
        kernel32.Process32FirstW.restype = wintypes.BOOL
        kernel32.Process32NextW.argtypes = [wintypes.HANDLE, ctypes.POINTER(PROCESSENTRY32W)]
        kernel32.Process32NextW.restype = wintypes.BOOL
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel32.CloseHandle.restype = wintypes.BOOL

        snapshot = kernel32.CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0)
        # INVALID_HANDLE_VALUE (-1) arrives as a large unsigned value, not falsy.
        if not snapshot or snapshot == ctypes.c_void_p(-1).value:
            return None
        try:
            entry = PROCESSENTRY32W()
            entry.dwSize = ctypes.sizeof(PROCESSENTRY32W)
            if not kernel32.Process32FirstW(snapshot, ctypes.byref(entry)):
                # An empty walk is a failed read, not a machine with no processes.
                return None
            ERROR_NO_MORE_FILES = 18
            table: "dict[int, list[int]]" = {}
            while True:
                child = int(entry.th32ProcessID)
                parent = int(entry.th32ParentProcessID)
                if child:
                    table.setdefault(parent, []).append(child)
                if not kernel32.Process32NextW(snapshot, ctypes.byref(entry)):
                    # FALSE also means a mid-walk failure; only ERROR_NO_MORE_FILES ends the list.
                    # Matches studio/src-tauri/src/process.rs.
                    if ctypes.get_last_error() != ERROR_NO_MORE_FILES:
                        return None
                    break
            return table
        finally:
            kernel32.CloseHandle(snapshot)
    except Exception:
        return None


def _child_pid_map() -> "Optional[dict[int, list[int]]]":
    """Parent pid -> its children, or None when the table cannot be read."""
    if _is_linux():
        try:
            table: "dict[int, list[int]]" = {}
            for entry in os.listdir("/proc"):
                if not entry.isdigit():
                    continue
                try:
                    with open(f"/proc/{entry}/stat", encoding = "utf-8") as fh:
                        stat = fh.read()
                except OSError:
                    continue
                # After the comm field: state, ppid, pgrp, ...
                tail = stat[stat.rfind(")") + 2 :].split()
                if len(tail) > 1:
                    table.setdefault(int(tail[1]), []).append(int(entry))
            return table
        except Exception:
            return None
    if sys.platform == "darwin":
        try:
            import subprocess

            out = subprocess.run(
                ["ps", "-A", "-o", "pid=,ppid="],
                capture_output = True,
                text = True,
                encoding = "utf-8",
                errors = "replace",
                timeout = 5,
            )
            if out.returncode != 0:
                return None
            table = {}
            for line in (out.stdout or "").splitlines():
                parts = line.split()
                if len(parts) != 2:
                    continue
                try:
                    child, parent = int(parts[0]), int(parts[1])
                except ValueError:
                    continue
                table.setdefault(parent, []).append(child)
            return table
        except Exception:
            return None
    if _is_windows():
        return _windows_child_pid_map()
    return None


def collect_descendants_known(
    pid: "Optional[int]",
) -> "tuple[list[tuple[int, Optional[str]]], bool]":
    """known is False when the walk failed; an unreadable walk must never be read as no children."""
    if not pid:
        return [], True
    # Windows never clears a stale creator pid, so check each candidate against its own
    # parent's creation time (a child cannot predate its parent). Unreadable candidates are skipped.
    if _is_windows():
        return _windows_collect_descendants_known(pid)
    # POSIX: the kernel rewrites parent links on reparent, so no floor is needed.
    table = _child_pid_map()
    if not table:
        return [], False
    found: "list[tuple[int, Optional[str]]]" = []
    seen = {pid}
    queue = list(table.get(pid, ()))
    while queue:
        child = queue.pop(0)
        if child in seen:
            continue
        seen.add(child)
        found.append((child, _pid_identity(child)))
        queue.extend(table.get(child, ()))
    return found, True


def collect_descendants(pid: "Optional[int]") -> "list[tuple[int, Optional[str]]]":
    """Call before signalling the parent, since its children are reparented on exit and lose the link."""
    return collect_descendants_known(pid)[0]


def _windows_collect_descendants(pid: int) -> "list[tuple[int, Optional[str]]]":
    """The list alone. See `_windows_collect_descendants_known`."""
    return _windows_collect_descendants_known(pid)[0]


def _windows_collect_descendants_known(
    pid: int, identity: "Optional[str]" = None
) -> "tuple[list[tuple[int, Optional[str]]], bool]":
    """Orders each child against its own parent's creation time, since Windows keeps stale creator pids."""
    # Use the caller's identity, not a fresh read: a reused root pid would root the walk at a stranger.
    if identity is not None and not _provably_the_same(pid, identity):
        return [], False
    root_floor = _windows_creation_time(identity if identity is not None else _pid_identity(pid))
    if root_floor is None:
        # No floor means no ancestry proof: indeterminate, not empty.
        return [], False
    # Floor read before the snapshot: anything created after it is unprovable, so the walk
    # reports incomplete and later rounds re-snapshot.
    snapshot_floor = _windows_filetime_now()
    table = _child_pid_map()
    # Ceiling read after the snapshot: a process created later cannot be one the snapshot listed.
    # Taking it first would wrongly reject genuine children born in the gap.
    snapshot_ceiling = _windows_filetime_now()
    if not table:
        return [], False
    found: "list[tuple[int, Optional[str]]]" = []
    complete = True
    seen = {pid}
    # (candidate pid, creation time of the parent that listed it)
    queue: "list[tuple[int, int]]" = [(child, root_floor) for child in table.get(pid, ())]
    while queue:
        child, parent_created = queue.pop(0)
        if child in seen:
            continue
        seen.add(child)
        identity = _pid_identity(child)
        created = _windows_creation_time(identity)
        if created is None:
            # Unreadable but alive means the tree is not fully enumerated.
            if _pid_alive(child) and not _pid_is_zombie(child):
                complete = False
            continue
        if created < parent_created:
            # Predates its listed creator: recycled number, rejected; walk stays complete.
            continue
        if snapshot_ceiling is not None and created > snapshot_ceiling:
            # Did not exist at snapshot time: the listed process exited and its pid was reused.
            continue
        if snapshot_floor is not None and created > snapshot_floor:
            # Created during the snapshot: unprovable, so skip and mark the walk incomplete.
            complete = False
            continue
        found.append((child, identity))
        queue.extend((grandchild, created) for grandchild in table.get(child, ()))
    return found, complete


# Kills are asynchronous on both platforms, so wait briefly before calling a pid a survivor;
# false survivors keep records and cause reap sweeps on every later launch.
_KILL_SETTLE_SECONDS = 0.5
_KILL_SETTLE_POLL_SECONDS = 0.01

# Two consecutive gone reads required: _pid_alive fails open on Windows.
# Doubling backoff: off Linux/Windows each probe forks ps.
_KILL_SETTLE_CONFIRMATIONS = 2
_KILL_SETTLE_POLL_CEILING_SECONDS = 0.1


def _survivors_after_settling(
    candidates: "list[tuple[int, Optional[str]]]",
    still_a_survivor: "Callable[[int, Optional[str]], bool]",
    grace: float = _KILL_SETTLE_SECONDS,
) -> "list[tuple[int, Optional[str]]]":
    """A pid drops out after _KILL_SETTLE_CONFIRMATIONS consecutive agreements that it has gone."""
    if not candidates:
        return []
    agreed: "dict[int, int]" = {}

    def _pass() -> "list[tuple[int, Optional[str]]]":
        still: "list[tuple[int, Optional[str]]]" = []
        for pid, identity in candidates:
            if agreed.get(pid, 0) >= _KILL_SETTLE_CONFIRMATIONS:
                continue
            if still_a_survivor(pid, identity):
                agreed[pid] = 0
                still.append((pid, identity))
            else:
                agreed[pid] = agreed.get(pid, 0) + 1
                if agreed[pid] < _KILL_SETTLE_CONFIRMATIONS:
                    still.append((pid, identity))
        return still

    remaining = _pass()
    deadline = time.monotonic() + max(0.0, grace)
    wait = _KILL_SETTLE_POLL_SECONDS
    while remaining and time.monotonic() < deadline:
        time.sleep(wait)
        wait = min(wait * 2, _KILL_SETTLE_POLL_CEILING_SECONDS)
        remaining = _pass()
    return remaining


def confirm_pid_exited(pid: "Optional[int]", grace: float = _KILL_SETTLE_SECONDS) -> bool:
    """Waits out the settle grace; False when unknown, since True authorises dropping the last handles."""
    if not _signalable(pid):
        return False
    return not _survivors_after_settling(
        [(int(pid), None)], lambda candidate, _identity: pid_is_running(candidate), grace
    )


def _settle_after_the_kill(
    pid: int,
    group_leader: bool,
    grace: float = _KILL_SETTLE_SECONDS,
) -> None:
    """Waits briefly after SIGKILL, since an immediate liveness read still sees the child running."""

    def _still_there(candidate: int, _identity: "Optional[str]") -> bool:
        # A leader can be gone while its session is not.
        if group_leader and _group_has_members(candidate):
            return True
        return _pid_alive(candidate) and not _pid_is_zombie(candidate)

    _survivors_after_settling([(pid, None)], _still_there, grace)


def terminate_descendants(
    collected: "list[tuple[int, Optional[str]]]", timeout: float = 5.0
) -> "list[tuple[int, Optional[str]]]":
    """Returns survivors with the collected identity; a survivor must never read as an empty list."""
    if not collected:
        return []
    if _is_windows():
        return _windows_terminate_collected(collected)
    live: "list[tuple[int, Optional[str]]]" = []
    for pid, identity in collected:
        if not _signalable(pid) or not _still_the_same(pid, identity):
            continue
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            continue
        live.append((pid, identity))
    deadline = time.monotonic() + max(0.0, timeout)
    while live:
        live = [item for item in live if _pid_alive(item[0]) and not _pid_is_zombie(item[0])]
        if not live or time.monotonic() >= deadline:
            break
        time.sleep(0.05)
    for pid, identity in live:
        if not _still_the_same(pid, identity):
            continue
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            pass
    # Re-read after a grace: SIGKILL can be outlived (D state) and dying processes linger briefly.
    return _survivors_after_settling(
        live,
        lambda pid, identity: (
            _pid_alive(pid) and not _pid_is_zombie(pid) and _still_the_same(pid, identity)
        ),
    )


# Capture-before-kill recursion depth; exhausting it returns None (unresolved), never drops.
_SUBTREE_CAPTURE_DEPTH = 8


def _windows_kill_below(
    anchor: int,
    attempted: "list[tuple[int, Optional[str]]]",
    unresolved: "list[tuple[int, Optional[str]]]",
    killed: "set[int]",
    depth: int,
    anchor_identity: "Optional[str]" = None,
) -> "Optional[bool]":
    """Reads each level's children just before signalling that level, so late starts are not missed."""
    if depth <= 0:
        # None, not False: the walk was not performed, so the record must survive.
        return None
    # Use the anchor's identity so a reused pid cannot root the snapshot at a stranger.
    found, known = _windows_collect_descendants_known(anchor, anchor_identity)
    if not known:
        return None
    progressed = False
    for child_pid, child_identity in reversed(found):
        if child_pid in killed:
            continue
        if not _signalable(child_pid) or not _pid_alive(child_pid):
            killed.add(child_pid)
            continue
        if not _provably_the_same(child_pid, child_identity):
            # Provably other is dropped; unreadable is reported so its record survives.
            if (
                _pid_alive(child_pid)
                and not _pid_is_zombie(child_pid)
                and not _provably_different(child_pid, child_identity)
            ):
                unresolved.append((child_pid, child_identity))
            killed.add(child_pid)
            continue
        # Below first: once the anchor dies its children can no longer be found.
        below = _windows_kill_below(
            child_pid, attempted, unresolved, killed, depth - 1, child_identity
        )
        if below is None:
            unresolved.append((child_pid, child_identity))
        try:
            _windows_terminate_pid(child_pid, child_identity)
        except Exception:  # noqa: BLE001 - best effort, like the rest of this
            pass
        killed.add(child_pid)
        attempted.append((child_pid, child_identity))
        progressed = True
    return progressed


# Re-walks catch children started during each taskkill's 15s ceiling.
_LATE_WALK_ROUNDS = 4


def _windows_terminate_collected(
    collected: "list[tuple[int, Optional[str]]]",
) -> "list[tuple[int, Optional[str]]]":
    """Uses taskkill /F per survivor, never /T, which follows stale parent-pid links to recycled pids."""
    attempted: "list[tuple[int, Optional[str]]]" = []
    unresolved: "list[tuple[int, Optional[str]]]" = []
    for pid, identity in reversed(collected):
        if not _signalable(pid) or not _pid_alive(pid):
            continue
        if not _provably_the_same(pid, identity):
            # Unknown identity: never signalled, always reported as unresolved.
            if (
                _pid_alive(pid)
                and not _pid_is_zombie(pid)
                and not _provably_different(pid, identity)
            ):
                unresolved.append((pid, identity))
            continue
        # Repeated, bounded rounds: late descendants can spawn during each 15s taskkill.
        killed_below: "set[int]" = set()
        still_growing = True
        for _round in range(_LATE_WALK_ROUNDS):
            progressed = _windows_kill_below(
                pid, attempted, unresolved, killed_below, _SUBTREE_CAPTURE_DEPTH, identity
            )
            if progressed is None:
                # The walk failed: still kill the survivor, but report it unresolved so the record stays.
                unresolved.append((pid, identity))
                still_growing = False
                break
            if not progressed:
                still_growing = False
                break
        if still_growing:
            # Rounds ran out with pids still appearing: report unresolved.
            unresolved.append((pid, identity))
        try:
            _windows_terminate_pid(pid, identity)
        except Exception:  # noqa: BLE001 - best effort, like the rest of this
            pass
        attempted.append((pid, identity))
    # Accounting, not signalling: alive with unreadable identity still counts as running.
    # Settle first: TerminateProcess is asynchronous.
    survivors = _survivors_after_settling(
        attempted,
        lambda pid, identity: (
            _pid_alive(pid) and not _pid_is_zombie(pid) and not _provably_different(pid, identity)
        ),
    )
    # Dedup: a pid can be both unresolved and still alive.
    ordered: "list[tuple[int, Optional[str]]]" = []
    seen: "set[int]" = set()
    for pid, identity in unresolved + survivors:
        if pid in seen:
            continue
        seen.add(pid)
        ordered.append((pid, identity))
    return ordered


def _windows_terminate_validated_tree(pid: int, identity: "Optional[str]" = None) -> bool:
    """Tree kill like taskkill /T /F, but keeps the descendant filter; False if anything survives."""
    # Caller's identity, not a fresh read: a recycled leader pid would bless a stranger's tree.
    if identity is not None and not _provably_the_same(pid, identity):
        return False
    descendants, known = _windows_collect_descendants_known(pid, identity)
    # Kill the root first, right after the snapshot, so it cannot spawn during slow taskkills.
    if _signalable(pid) and _pid_alive(pid):
        try:
            _windows_terminate_pid(pid, identity if identity is not None else _pid_identity(pid))
        except Exception:  # noqa: BLE001 - best effort, like the rest of this
            pass
    # Re-collect each survivor with creation-time validation before killing to reach late children.
    survivors = _windows_terminate_collected(descendants)
    # An unenumerable tree is not empty, so this cannot answer True.
    if not known:
        return False
    # Read back the root with the same grace: TerminateProcess returns before exit.
    if _survivors_after_settling(
        [(pid, identity)],
        lambda candidate, _identity: _pid_alive(candidate) and not _pid_is_zombie(candidate),
    ):
        return False
    return not survivors


def _still_the_same(pid: int, identity: "Optional[str]") -> bool:
    """False only when the pid is provably a different process now."""
    current = _pid_identity(pid)
    if identity is None or current is None:
        return True
    return _same_identity(identity, current)


def _provably_different(pid: int, identity: "Optional[str]") -> bool:
    """True only when the pid is provably a different process; an unreadable identity is not proof."""
    if identity is None:
        return False
    current = _pid_identity(pid)
    if current is None:
        return False
    return not _same_identity(identity, current)


def _provably_the_same(pid: int, identity: "Optional[str]") -> bool:
    """Fail-closed: an identity that cannot be read on either side means the process is left alone."""
    if identity is None:
        return False
    current = _pid_identity(pid)
    if current is None:
        return False
    return _same_identity(identity, current)


def _group_member_pids(pgid: int) -> "Optional[list[int]]":
    """Pids in a process group, or None when they cannot be enumerated."""
    if _is_linux():
        try:
            found = []
            for entry in os.listdir("/proc"):
                if not entry.isdigit():
                    continue
                try:
                    with open(f"/proc/{entry}/stat", encoding = "utf-8") as fh:
                        stat = fh.read()
                except OSError:
                    continue
                # After the comm field: state, ppid, pgrp, ...
                tail = stat[stat.rfind(")") + 2 :].split()
                if len(tail) > 2 and tail[2] == str(pgid):
                    found.append(int(entry))
            return found
        except Exception:
            return None
    if sys.platform == "darwin":
        try:
            import subprocess

            out = subprocess.run(
                ["ps", "-o", "pid=", "-g", str(pgid)],
                capture_output = True,
                text = True,
                encoding = "utf-8",
                errors = "replace",
                timeout = 5,
            )
            # ps exits nonzero on failure too; reporting empty would drop a live descendant's record.
            if out.returncode != 0:
                return None
            return [int(x) for x in (out.stdout or "").split()]
        except Exception:
            return None
    return None


# macOS has no PDEATHSIG or job objects: record children and sweep leftovers at startup.
def _breadcrumb_dir():
    from pathlib import Path

    override = os.environ.get("UNSLOTH_STUDIO_CHILD_RECORD")
    if override:
        return Path(override)
    try:
        from utils.paths.storage_roots import studio_root
        return Path(studio_root()) / "run" / "children"
    except Exception:
        return None


def _breadcrumb_file():
    # One file per owner: two instances can share a home.
    directory = _breadcrumb_dir()
    return None if directory is None else directory / f"{os.getpid()}.json"


_owner_identity: Optional[str] = None


def _own_identity() -> "Optional[str]":
    """Captured once and kept: a None would make a reused pid read as this owner still running."""
    global _owner_identity
    if _owner_identity is None:
        _owner_identity = _identity_for_record(os.getpid())
    return _owner_identity


def _refreshed_identity(pid: int, identity: "Optional[str]") -> "Optional[str]":
    """A None identity is never signalled, so fill it in while the child lives and write it back."""
    if identity is not None or not _pid_alive(pid):
        return identity
    identity = _pid_identity(pid)
    if identity is not None:
        _tracked_pids[pid] = identity
    return identity


def _write_breadcrumb() -> None:
    path = _breadcrumb_file()
    if path is None:
        return
    try:
        import json

        path.parent.mkdir(parents = True, exist_ok = True)
        payload = {
            "owner_pid": os.getpid(),
            "owner_identity": _own_identity(),
            "children": [
                {
                    "pid": pid,
                    "identity": _refreshed_identity(pid, identity),
                    "pgid": _tracked_pgids.get(pid),
                }
                for pid, identity in _tracked_pids.items()
            ],
        }
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload), encoding = "utf-8")
        tmp.replace(path)
    except Exception:
        pass


def clear_breadcrumb() -> None:
    """Keeps the record while any tracked child is still alive, or the next startup cannot find it."""
    with _record_lock:
        if _tracked_pids:
            _write_breadcrumb()
            return
        # Inside the lock, or a concurrent spawn's record could be deleted.
        path = _breadcrumb_file()
        if path is not None:
            _unlink(path)


def _identity_or_none(pid) -> "Optional[str]":
    return _pid_identity(pid) if isinstance(pid, int) and pid > 0 else None


def _pid_alive(pid: int) -> bool:
    if _is_windows():
        # NOT os.kill(pid, 0): on Windows that is TerminateProcess.
        try:
            import ctypes
            from ctypes import wintypes

            SYNCHRONIZE = 0x0010_0000
            PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
            WAIT_TIMEOUT = 0x102
            ERROR_ACCESS_DENIED = 5

            kernel32 = ctypes.WinDLL("kernel32", use_last_error = True)
            kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
            kernel32.OpenProcess.restype = wintypes.HANDLE
            kernel32.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
            kernel32.WaitForSingleObject.restype = wintypes.DWORD
            kernel32.CloseHandle.argtypes = [wintypes.HANDLE]

            handle = kernel32.OpenProcess(
                SYNCHRONIZE | PROCESS_QUERY_LIMITED_INFORMATION, False, pid
            )
            if not handle:
                # ACCESS_DENIED means alive but another user's; else it is gone.
                return _last_error(ctypes) == ERROR_ACCESS_DENIED
            try:
                return kernel32.WaitForSingleObject(handle, 0) == WAIT_TIMEOUT
            finally:
                kernel32.CloseHandle(handle)
        except Exception:
            return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def pid_is_running(pid: "Optional[int]") -> bool:
    """An exited child nobody has waited on is a zombie and does not count as running."""
    if not is_signalable_pid(pid):
        return False
    return _pid_alive(pid) and not _pid_is_zombie(pid)


def _pid_is_zombie(pid: int) -> bool:
    """An exited child nobody waited on. It answers signals like a live process,
    so a survivor check has to tell the two apart."""
    if _is_windows():
        return False
    if _is_linux():
        try:
            with open(f"/proc/{pid}/stat", encoding = "utf-8") as fh:
                return fh.read().rsplit(")", 1)[1].split()[0] == "Z"
        except Exception:
            return False
    try:
        import subprocess
        out = subprocess.run(
            ["ps", "-o", "state=", "-p", str(pid)],
            capture_output = True,
            text = True,
            encoding = "utf-8",
            errors = "replace",
            timeout = 5,
        )
        return (out.stdout or "").strip().startswith("Z")
    except Exception:
        return False


def _identity_for_record(pid: int, attempts: int = 3) -> Optional[str]:
    """Retries, since an identity recorded as None means the entry is never signalled."""
    import time

    for attempt in range(attempts):
        identity = _pid_identity(pid)
        if identity is not None:
            return identity
        if not _pid_alive(pid):
            break
        if attempt + 1 < attempts:
            time.sleep(0.05 * (attempt + 1))
    return None


def adopt_pid(
    pid: Optional[int],
    identity: "Optional[str]" = None,
    *,
    from_snapshot: bool = False,
) -> None:
    """Pids from an earlier snapshot must pass their identity, so a recycled number is never adopted."""
    # pid 1 is init: recording it turns the sweep into killing everything the user owns.
    if not _signalable(pid):
        return
    if identity is None and from_snapshot:
        return
    if identity is not None and not _provably_the_same(pid, identity):
        # Unknown adopts nobody: it may already be a stranger, and that is not recoverable.
        return
    # Without the fork handler a fork child would claim this process's children.
    _adopt_fork_reset()
    if identity is None:
        identity = _identity_for_record(pid)
    pgid = _own_process_group(pid)
    with _record_lock:
        _tracked_pids[pid] = identity
        if pgid is not None:
            _tracked_pgids[pid] = pgid
        _write_breadcrumb()
    if _is_windows() and _win_job_handle:
        try:
            import ctypes
            from ctypes import wintypes

            kernel32 = ctypes.WinDLL("kernel32", use_last_error = True)
            _win_signatures(kernel32)
            kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
            kernel32.OpenProcess.restype = wintypes.HANDLE
            PROCESS_SET_QUOTA, PROCESS_TERMINATE = 0x0100, 0x0001
            PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
            handle = kernel32.OpenProcess(
                PROCESS_SET_QUOTA | PROCESS_TERMINATE | PROCESS_QUERY_LIMITED_INFORMATION,
                False,
                pid,
            )
            if handle:
                # Verify identity through the handle being assigned: the pid may have been reused, and a
                # kill-on-close job would kill a stranger. Unreadable means skip.
                opened = _windows_identity_of_handle(kernel32, handle)
                if identity is not None and opened is not None and opened == identity:
                    kernel32.AssignProcessToJobObject(_win_job_handle, handle)
                kernel32.CloseHandle(handle)
        except Exception:
            pass


_shutdown_latch = threading.Event()


def mark_process_shutting_down() -> None:
    """Process-wide latch: spawners on other objects also refuse after terminate_all takes its snapshot."""
    _shutdown_latch.set()


def is_process_shutting_down() -> bool:
    """Whether a spawn must be refused because the app is quitting."""
    return _shutdown_latch.is_set()


_quit_signal_at: Optional[float] = None


def mark_quit_signal_received() -> None:
    global _quit_signal_at
    if _quit_signal_at is None:
        _quit_signal_at = time.monotonic()


def seconds_since_quit_signal() -> Optional[float]:
    return None if _quit_signal_at is None else time.monotonic() - _quit_signal_at


def begin_process_lifecycle() -> None:
    """Clear the latch for an embedded host that calls run_server again.

    Quitting is terminal for a CLI run, but in-process callers (studio/backend/colab.py)
    reuse the interpreter, and a latch that never cleared would refuse every spawn of
    the second session.
    """
    global _quit_signal_at
    _shutdown_latch.clear()
    _quit_signal_at = None


def terminate_all(timeout: float = 5.0) -> "list[int]":
    """Only signals pids whose identity matches; unverifiable ones are kept and returned as survivors."""
    survivors: "list[int]" = []
    # Snapshot under the write lock: adopt_pid can still run concurrently.
    with _record_lock:
        tracked = list(_tracked_pids.items())
    for pid, identity in tracked:
        with _record_lock:
            _tracked_pids.pop(pid, None)
            pgid = _tracked_pgids.pop(pid, None)
        if not _signalable(pid):
            continue
        current = _pid_identity(pid)
        if not _pid_alive(pid):
            # A leader that exited first can still have a group holding the GPU.
            if not _reap_orphaned_group(pgid, pid, timeout) and _group_has_members(pgid):
                survivors.append(pid)
                with _record_lock:
                    _tracked_pids[pid] = identity
                    if pgid is not None:
                        _tracked_pgids[pid] = pgid
            continue
        if current is not None and identity is not None and not _same_identity(identity, current):
            continue
        if identity is None or current is None:
            # Cannot prove it is ours: do not signal, but keep it recorded while alive.
            if _pid_alive(pid) and not _pid_is_zombie(pid):
                with _record_lock:
                    _tracked_pids[pid] = identity
                    # Keep the group with the pid: it is the only handle once the leader exits.
                    if pgid is not None:
                        _tracked_pgids[pid] = pgid
            continue
        tree_stands = False
        try:
            if _is_windows():
                # Kill the tree via the validated collector, never taskkill /T (stale parent links).
                tree_stands = not _windows_terminate_validated_tree(pid, identity)
            else:
                _posix_terminate(pid, timeout)
        except Exception:
            pass
        current_now = _identity_or_none(pid)
        still_ours = (
            _pid_alive(pid)
            and not _pid_is_zombie(pid)
            and current_now is not None
            and _same_identity(identity, current_now)
        )
        if still_ours or tree_stands or _group_has_members(pgid):
            survivors.append(pid)
            with _record_lock:
                _tracked_pids[pid] = identity
                if pgid is not None:
                    _tracked_pgids[pid] = pgid
    return survivors


def terminate_pid(
    pid: "Optional[int]",
    timeout: float = 5.0,
    *,
    owner_verified: bool = False,
) -> None:
    """owner_verified waives only the cannot-prove refusal; a pid provably reused is still left alone."""
    # Floor here too: _windows_terminate_tree bypasses the POSIX helper's check.
    if not _signalable(pid):
        return
    with _record_lock:
        identity = _tracked_pids.get(pid)
        pgid = _tracked_pgids.get(pid)
    # Verify identity: an announced child can exit unannounced and its pid be reused.
    current = _pid_identity(pid)
    if identity is not None and current is not None and not _same_identity(identity, current):
        forget_pid(pid)
        return
    if not owner_verified and _pid_alive(pid) and (identity is None or current is None):
        # Cannot prove it is ours: leave it and keep the record for the startup sweep.
        return
    tree_stands = False
    try:
        if _is_windows():
            # Not taskkill /T: it follows stale parent links and would kill rejected strangers.
            tree_stands = not _windows_terminate_validated_tree(pid, identity)
        else:
            _posix_terminate(pid, timeout)
            # A dead leader loses getpgid, so the recorded group is the only handle left.
            if _group_has_members(pgid):
                _reap_orphaned_group(pgid, pid, timeout)
    except Exception:  # noqa: BLE001 - best effort, like the rest of this
        pass
    if tree_stands:
        return
    forget_pid(pid)


def reap_recorded_children(timeout: float = 5.0) -> "list[int]":
    """Considers every record in the directory, so a crash while a sibling ran is still cleaned up."""
    directory = _breadcrumb_dir()
    if directory is None or not directory.is_dir():
        return []
    killed: "list[int]" = []
    # Revisit only deferred records whose owner may die later in this sweep; never signal twice.
    pending = sorted(directory.glob("*.json"))
    for _pass in range(4):
        deferred: "list" = []
        found: "list[int]" = []
        for path in pending:
            reaped, owner_alive = _reap_one_record(path, timeout)
            found.extend(reaped)
            if owner_alive:
                deferred.append(path)
        killed.extend(found)
        if not deferred or not found:
            break
        pending = deferred
    return killed


def _reap_one_record(path, timeout: float) -> "tuple[list[int], bool]":
    """``(pids signalled, whether it was left alone because its owner is alive)``."""
    import json

    killed: "list[int]" = []
    try:
        record = json.loads(path.read_text(encoding = "utf-8"))
    except Exception:
        _unlink(path)
        return killed, False
    if not isinstance(record, dict):
        _unlink(path)
        return killed, False

    owner_pid = record.get("owner_pid")
    # Anything but the string this wrote is treated as no identity.
    owner_identity = _recorded_identity(record.get("owner_identity"))
    # Identity decides, not the pid: pids recycle.
    current_owner = _identity_or_none(owner_pid)
    owner_matches = (
        owner_identity is None
        or current_owner is None
        or _same_identity(owner_identity, current_owner)
    )
    if owner_pid == os.getpid() and owner_matches:
        return killed, False
    if (
        isinstance(owner_pid, int)
        and owner_matches
        and _pid_alive(owner_pid)
        # kill(pid, 0) succeeds for a zombie, which is still a dead owner.
        and not _pid_is_zombie(owner_pid)
    ):
        return killed, True

    unresolved = False
    children = record.get("children")
    for entry in children if isinstance(children, list) else []:
        pid = entry.get("pid") if isinstance(entry, dict) else None
        if not _signalable(pid):
            # Old records can name pid 1; drop rather than defer so it is gone after one startup.
            continue
        # A zombie holds nothing; signalling it would burn the grace period.
        if not _pid_alive(pid) or _pid_is_zombie(pid):
            # The group can outlive its leader (shim crashed, server holds the GPU).
            pgid = entry.get("pgid")
            if _reap_orphaned_group(pgid, pid, timeout):
                killed.append(pid)
            elif _group_has_members(pgid):
                unresolved = True
            continue
        identity = _recorded_identity(entry.get("identity"))
        current = _identity_or_none(pid)
        if identity is None or current is None:
            unresolved = True
            continue
        if not _same_identity(identity, current):
            continue
        tree_stands = False
        if _is_windows():
            # Kill the tree via the validated collector, not taskkill /T, which follows stale parent links.
            tree_stands = not _windows_terminate_validated_tree(pid, identity)
        else:
            _posix_terminate(pid, timeout = timeout)
        killed.append(pid)
        if tree_stands:
            unresolved = True
        if _pid_alive(pid) and not _pid_is_zombie(pid):
            unresolved = True
        elif _group_has_members(entry.get("pgid")):
            unresolved = True

    if not unresolved:
        _unlink(path)
    if killed:
        try:
            import logging
            logging.getLogger(__name__).warning(
                "Reaped %d orphaned child process(es) left by a previous Unsloth: %s",
                len(killed),
                killed,
            )
        except Exception:
            pass
    return killed, False


def _own_process_group(pid: int) -> Optional[int]:
    """Returns the group only if the pid leads one; a shared group would take Unsloth down with it."""
    if _is_windows() or not hasattr(os, "getpgid"):
        return None
    try:
        pgid = os.getpgid(pid)
    except Exception:
        return None
    if not _signalable(pgid):
        return None
    return pgid if pgid == pid else None


def _reap_orphaned_group(pgid: object, pid: int, timeout: float) -> bool:
    """No identity check needed: the dead leader's pid stays reserved while the group still has members."""
    if not _signalable(pgid) or pgid != pid or _is_windows() or not hasattr(os, "killpg"):
        return False
    try:
        os.killpg(pgid, 0)
    except Exception:
        return False
    # Zombies answer the probe where pid 1 does not reap; skip empty groups.
    if not _group_has_members(pgid):
        return False
    import time

    try:
        os.killpg(pgid, signal.SIGTERM)
    except Exception:
        return False
    deadline = time.monotonic() + max(0.0, timeout)
    next_state_check = 0.0
    while time.monotonic() < deadline:
        try:
            os.killpg(pgid, 0)
        except Exception:
            return True
        now = time.monotonic()
        if now >= next_state_check:
            # Where pid 1 does not reap, exited members stay zombies; recheck state periodically.
            next_state_check = now + 0.5
            if not _group_has_members(pgid):
                return True
        time.sleep(0.1)
    try:
        os.killpg(pgid, signal.SIGKILL)
    except Exception:
        pass
    # Only gone counts: the record is the last handle on whatever holds the GPU.
    try:
        os.killpg(pgid, 0)
        return False
    except Exception:
        return True


def _windows_terminate_pid(pid: int, identity: "Optional[str]" = None) -> bool:
    """Kills one pid with taskkill /F, no /T; the fallback re-checks identity, as the pid can be reused."""
    import subprocess

    # Terminate through a handle first: taskkill /PID re-resolves the number and can hit a reused pid.
    through_handle = _windows_terminate_through_a_handle(pid, identity)
    if through_handle is not None:
        return through_handle

    # No handle: re-verify identity right before taskkill, or skip; a leak is recoverable, a wrong kill is not.
    if not _provably_the_same(pid, identity):
        return False

    try:
        completed = subprocess.run(
            ["taskkill", "/PID", str(pid), "/F"],
            capture_output = True,
            timeout = 15,
            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        # 128 is already gone.
        if completed.returncode in (0, 128):
            return True
    except (OSError, subprocess.SubprocessError):
        pass
    # Re-check: taskkill can take 15s and the pid may be freed meanwhile.
    if not _provably_the_same(pid, identity):
        return False
    try:
        os.kill(pid, signal.SIGTERM)
    except OSError:
        pass
    return False


def _windows_terminate_tree(pid: int) -> bool:
    """Runs taskkill /T /F; False means workers may still run, so the caller must keep the record."""
    import subprocess

    try:
        completed = subprocess.run(
            ["taskkill", "/PID", str(pid), "/T", "/F"],
            capture_output = True,
            timeout = 15,
            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        # check=False does not raise; 128 is already gone.
        if completed.returncode in (0, 128):
            return True
    except (OSError, subprocess.SubprocessError):
        pass
    try:
        os.kill(pid, signal.SIGTERM)
    except OSError:
        pass
    return False


def _unlink(path) -> None:
    try:
        path.unlink(missing_ok = True)
    except Exception:
        pass


def _posix_terminate(pid: int, timeout: float = 5.0) -> None:
    # SIGTERM, wait timeout, then SIGKILL; prefer the group when pid leads one.
    if not _signalable(pid):
        return
    group_leader = False
    try:
        group_leader = os.getpgid(pid) == pid
    except Exception:
        pass
    # Same-group children are unreachable by killpg; collect them now while the parent lives.
    descendants = [] if group_leader else collect_descendants(pid)
    try:
        _posix_terminate_one(pid, group_leader, timeout)
    finally:
        terminate_descendants(descendants, timeout)


def _posix_terminate_one(pid: int, group_leader: bool, timeout: float) -> None:
    # Belt and braces: killpg(1) would reach every process the user owns.
    if not _signalable(pid):
        return
    killer = os.killpg if group_leader else os.kill
    try:
        killer(pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    except Exception:
        return
    deadline = time.monotonic() + max(0.0, timeout)
    next_state_check = 0.0
    while time.monotonic() < deadline:
        try:
            killer(pid, 0)
        except ProcessLookupError:
            return
        except Exception:
            break
        now = time.monotonic()
        if now >= next_state_check:
            # A zombie answers signal 0 like a live process; check state periodically (forks off Linux).
            next_state_check = now + 0.5
            gone = (not _group_has_members(pid)) if group_leader else _pid_is_zombie(pid)
            if gone:
                return
        time.sleep(0.05)
    try:
        killer(pid, signal.SIGKILL)
    except Exception:
        pass
    _settle_after_the_kill(pid, group_leader)
