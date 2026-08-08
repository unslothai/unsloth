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
import threading
from typing import Callable, Optional

_PR_SET_PDEATHSIG = 1
_JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000
_JobObjectExtendedLimitInformation = 9

_lock = threading.Lock()
_spawner_lock = threading.Lock()
_spawner: "Optional[_Spawner]" = None
_initialized = False
_win_job_handle: Optional[int] = None  # retained for the interpreter's lifetime
_tracked_pids: "dict[int, Optional[str]]" = {}  # pid -> identity, reaped by terminate_all


# Whether cleanup-on-abnormal-exit is in force, and why not. A silent failure
# here leaks every child on a crash, so record it and log it.
_win_job_status: "tuple[bool, str]" = (False, "not attempted")


def _last_error(ctypes_module) -> int:
    # get_last_error is Windows-only; a POSIX probe must not raise here.
    getter = getattr(ctypes_module, "get_last_error", None)
    try:
        return int(getter()) if getter else 0
    except Exception:
        return 0


def _record_job_status(ok: bool, detail: str, last_error: int = 0) -> None:
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
        elif _is_linux():
            _record_job_status(True, "PR_SET_PDEATHSIG per child")
        else:
            # macOS: no pdeathsig, no job object; reap_recorded_children covers it.
            _record_job_status(False, "no kernel-level parent-death signal on this platform")


def _win_signatures(kernel32) -> None:
    # Explicit HANDLE-width signatures. Without argtypes, ctypes marshals the
    # 64-bit job/process handles as c_int and truncates them on Win64, so the
    # job calls silently operate on a bogus handle and assignment fails.
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
        # AssignProcessToJobObject(parent) makes children inherit the job. May
        # fail if Unsloth already runs inside an incompatible host job (pre-Win8);
        # degrade to the cooperative path rather than blocking startup.
        if not kernel32.AssignProcessToJobObject(job, kernel32.GetCurrentProcess()):
            _record_job_status(False, "AssignProcessToJobObject failed", _last_error(ctypes))
            kernel32.CloseHandle(job)
            return
        _win_job_handle = job  # hold the handle so the job is not closed early
        _record_job_status(True, "kill-on-close job installed")
    except Exception as error:
        _record_job_status(False, f"{type(error).__name__}: {error}")


# ── Child binding ──


def _pdeathsig_preexec(owner_pid: Optional[int] = None) -> None:
    # Runs in the forked child before exec (PR_SET_PDEATHSIG does not survive the
    # fork); the getppid check closes the race where the parent died first.
    # owner_pid is the spawner's pid, read pre-fork. A bare `getppid() == 1` also
    # matches a parent that legitimately IS pid 1, which is how Studio runs as a
    # container entrypoint: it killed every llama-server before exec (#7886). It
    # also misses reparenting to a subreaper, whose pid is not 1.
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
        # No creator to consult and nothing orphaned, so only arm PDEATHSIG: the
        # bare getppid() == 1 test would kill a parent-is-pid-1 process (#7886).
        _pdeathsig_preexec(os.getppid())
        return
    # PDEATHSIG binds to the kernel parent; "orphaned" comes from the creator's
    # sentinel, not a pid compare, since under forkserver the kernel parent is the
    # fork server and comparing would kill every healthy worker. The sentinel is
    # exact under spawn and forkserver; under fork a sibling can inherit the
    # creator's write end and read stale-alive, but PDEATHSIG covers fork. A
    # forkserver creator dying later stays uncovered, as on main.
    _pdeathsig_preexec(os.getppid())
    try:
        if not parent.is_alive():
            os._exit(1)
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
    """A fork child inherits _spawner_lock in whatever state it was in and a
    _spawner whose thread does not exist here. Start clean instead of deadlocking."""
    global _spawner, _spawner_lock
    _spawner_lock = threading.Lock()
    _spawner = None


def _adopt_fork_reset() -> None:
    """Register the child-side reset once, lazily. Best-effort like the rest of
    this module: os.register_at_fork is POSIX-only and absent on Windows."""
    global _fork_reset_installed
    if _fork_reset_installed:
        return
    _fork_reset_installed = True
    try:
        os.register_at_fork(after_in_child = _reset_after_fork)
    except (AttributeError, RuntimeError):  # no fork support; nothing to guard
        pass


def spawn_on_lifetime_thread(spawn: Callable[[], object]) -> object:
    """Run *spawn* (a Popen call) on a thread that lives as long as the process.

    PR_SET_PDEATHSIG fires when the forking THREAD exits, not the process, so a
    child spawned from a short-lived worker dies as soon as that worker returns.
    Forking from one process-lifetime thread restores "die with the parent
    process". Non-Linux arms no per-thread signal, so it spawns directly.
    """
    if not _is_linux():
        return spawn()
    global _spawner
    _adopt_fork_reset()
    with _spawner_lock:
        if _spawner is None or not _spawner.usable():
            candidate = _Spawner()
            _spawner = candidate if candidate.usable() else None
        spawner = _spawner
    # No helper thread available (only when threading.Thread is replaced, as
    # some tests do): spawn inline, the pre-existing behaviour, rather than block.
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
        self._jobs.put((spawn, box, done))
        done.wait()
        ok, value = box[0]
        if not ok:
            raise value
        return value


def child_popen_kwargs(preexec_fn: Optional[Callable[[], None]] = None) -> dict:
    """Popen kwargs that bind a long-lived child to the parent's lifetime.

    On Linux returns a composed ``preexec_fn`` (PDEATHSIG + any existing one);
    empty elsewhere (Windows is covered by the inherited Job Object). Merge via
    ``**child_popen_kwargs()`` alongside the caller's existing kwargs.
    """
    if _is_linux():
        # os.getpid() runs in the spawner, so the child tells real reparenting
        # apart from a parent that is pid 1 (see _pdeathsig_preexec).
        return {"preexec_fn": compose_preexec(preexec_fn, os.getpid())}
    return {}


def _pid_identity(pid: int) -> Optional[str]:
    # Start time pins identity so a reused pid is never signalled later. None
    # (unreadable, or a platform with no cheap source) disables the check.
    if _is_linux():
        try:
            with open(f"/proc/{pid}/stat", encoding = "utf-8") as fh:
                stat = fh.read()
            return stat[stat.rfind(")") + 2 :].split()[19]  # after comm: starttime
        except Exception:
            return None
    if _is_windows():
        # Creation time, so a recycled pid is never mistaken for the child that
        # was recorded. Same purpose as starttime on Linux.
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
        # No /proc; `ps -o lstart` is enough to spot a recycled pid.
        try:
            import subprocess

            out = subprocess.run(
                ["ps", "-o", "lstart=,comm=", "-p", str(pid)],
                capture_output = True,
                text = True,
                encoding = "utf-8",
                errors = "replace",
                timeout = 5,
            )
            line = (out.stdout or "").strip()
            return line or None
        except Exception:
            return None
    return None


def forget_pid(pid: Optional[int]) -> None:
    """Stop tracking a child the owner has reaped, so terminate_all never
    signals a recycled pid."""
    if pid:
        _tracked_pids.pop(pid, None)
        _write_breadcrumb()


# ── Crash-survivable child record ──
#
# macOS has neither PR_SET_PDEATHSIG nor job objects, so a crash leaves every
# sidecar running. Record children as they are adopted; sweep the previous
# run's leftovers at startup.


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
    # One file per owner: two Studios can share a home (different ports), and a
    # single shared file would let the second erase the first's children.
    directory = _breadcrumb_dir()
    return None if directory is None else directory / f"{os.getpid()}.json"


def _write_breadcrumb() -> None:
    path = _breadcrumb_file()
    if path is None:
        return
    try:
        import json

        path.parent.mkdir(parents = True, exist_ok = True)
        payload = {
            "owner_pid": os.getpid(),
            "owner_identity": _pid_identity(os.getpid()),
            "children": [
                {"pid": pid, "identity": identity} for pid, identity in _tracked_pids.items()
            ],
        }
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload), encoding = "utf-8")
        tmp.replace(path)
    except Exception:
        pass


def clear_breadcrumb() -> None:
    """Drop our record after a clean shutdown; everything in it is already gone."""
    path = _breadcrumb_file()
    if path is not None:
        _unlink(path)


def _identity_or_none(pid) -> "Optional[str]":
    return _pid_identity(pid) if isinstance(pid, int) and pid > 0 else None


def _pid_alive(pid: int) -> bool:
    if _is_windows():
        # NOT os.kill(pid, 0): that is TerminateProcess here, so the probe
        # would kill the process it is asking about.
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
                # Signalled means exited; still waiting means running.
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


def adopt_pid(pid: Optional[int]) -> None:
    """Track a child (e.g. a multiprocessing worker started after the parent job
    was set up) and, on Windows, assign it to the job as belt-and-suspenders.
    Tolerates a None or already-exited pid."""
    if not pid:
        return
    _tracked_pids[pid] = _pid_identity(pid)
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
            handle = kernel32.OpenProcess(PROCESS_SET_QUOTA | PROCESS_TERMINATE, False, pid)
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


def reap_recorded_children(timeout: float = 5.0) -> "list[int]":
    """Kill children recorded by a previous Studio that is no longer running.

    Runs once at startup, before anything new spawns. Every record in the
    directory is considered, so a Studio that crashed while a sibling was
    running is still cleaned up. A child is only signalled when its recorded
    start-time identity still matches, so a recycled pid is never touched.
    """
    directory = _breadcrumb_dir()
    if directory is None or not directory.is_dir():
        return []
    killed: "list[int]" = []
    for path in sorted(directory.glob("*.json")):
        killed.extend(_reap_one_record(path, timeout))
    return killed


def _reap_one_record(path, timeout: float) -> "list[int]":
    import json

    killed: "list[int]" = []
    try:
        record = json.loads(path.read_text(encoding = "utf-8"))
    except Exception:
        _unlink(path)
        return killed

    owner_pid = record.get("owner_pid")
    owner_identity = record.get("owner_identity")
    # Identity decides, not the pid: pids recycle, and this process may have
    # inherited the pid of the Studio that wrote the record.
    owner_matches = owner_identity is None or owner_identity == _identity_or_none(owner_pid)
    if owner_pid == os.getpid() and owner_matches:
        return killed  # our own record
    if isinstance(owner_pid, int) and owner_matches and _pid_alive(owner_pid):
        return killed  # that Studio is still running; its children are its own

    unresolved = False
    for entry in record.get("children") or []:
        pid = entry.get("pid")
        if not isinstance(pid, int) or not _pid_alive(pid):
            continue
        identity = entry.get("identity")
        current = _identity_or_none(pid)
        if identity is None or current is None:
            # Unverifiable: never signal a pid that might now be something else.
            # Windows has no identity source here, and the Job Object already
            # covers that case.
            unresolved = True
            continue
        if identity != current:
            continue  # recycled by an unrelated process
        if _is_windows():
            try:
                os.kill(pid, signal.SIGTERM)
            except OSError:
                pass
        else:
            _posix_terminate(pid, timeout = timeout)
        # Counted once signalled: an unwaited child lingers as a zombie, which
        # a liveness re-check would read as alive.
        killed.append(pid)

    if not unresolved:
        _unlink(path)
    if killed:
        try:
            import logging

            logging.getLogger(__name__).warning(
                "Reaped %d orphaned child process(es) left by a previous Studio: %s",
                len(killed), killed,
            )
        except Exception:
            pass
    return killed


def _unlink(path) -> None:
    try:
        path.unlink(missing_ok = True)
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
