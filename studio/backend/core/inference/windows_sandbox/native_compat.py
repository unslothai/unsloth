# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Private bootstrap ownership adapters; never selected by the production launcher.

Native structures and API bindings come from the isolated stdlib module. Private
invocations retain separate profile and process ownership from installation-wide
production containers. Lifecycle implementations originate at 85c6db4b8.
"""

from __future__ import annotations
import ctypes
from ctypes import wintypes
from dataclasses import dataclass
import os
import shutil
import subprocess
import threading
import time
from typing import Any
from . import native_bindings as _production
from .native_bindings import (
    _api,
    _winerror,
    _revoke_sid,
    _validated_private_temp,
    _INFINITE,
    _WAIT_TIMEOUT,
    _WAIT_OBJECT_0,
    SandboxUnavailableError,
)

_PROCESS_CLOSE_TIMEOUT = 5.0


def __getattr__(name):
    return getattr(_production, name)


@dataclass
class _InvocationIdentity:
    moniker: str
    sid: ctypes.c_void_p
    sid_string: str
    profile_folder: str
    private_temp: str
    manifest_path: str
    granted_roots: tuple[str, ...]
    traverse_roots: tuple[str, ...]
    owner_pid: int
    owner_created: int
    cleaned: bool = False

    def cleanup(self) -> None:
        if self.cleaned:
            return
        errors: list[str] = []
        traverse = {os.path.normcase(path) for path in self.traverse_roots}
        for path in reversed(self.granted_roots):
            try:
                _revoke_sid(path, self.sid, exact = os.path.normcase(path) in traverse)
            except Exception as exc:  # noqa: BLE001 - continue ownership cleanup
                errors.append(f"ACL {path}: {exc}")
        try:
            private_temp = _validated_private_temp(self.profile_folder, self.private_temp)
            shutil.rmtree(private_temp, ignore_errors = False)
        except FileNotFoundError:
            pass
        except Exception as exc:  # noqa: BLE001
            errors.append(f"temp {self.private_temp}: {exc}")
        if errors:
            raise OSError("; ".join(errors))
        result = _api().userenv.DeleteAppContainerProfile(self.moniker)
        unsigned_result = ctypes.c_uint32(result).value
        if unsigned_result not in (0, 0x80070002):
            raise OSError(f"DeleteAppContainerProfile: 0x{unsigned_result:08x}")
        for path in (self.manifest_path + ".tmp", self.manifest_path):
            try:
                os.unlink(path)
            except FileNotFoundError:
                pass
            except OSError as exc:
                raise OSError(f"manifest: {exc}") from exc
        if _api().advapi32.FreeSid(self.sid):
            raise OSError("FreeSid did not release the invocation SID")
        self.sid = ctypes.c_void_p()
        self.cleaned = True


class _WindowsJob:
    def __init__(self, handle: wintypes.HANDLE):
        self._handle = handle
        self._lock = threading.Lock()

    def terminate(self) -> bool:
        with self._lock:
            return bool(self._handle and _api().kernel32.TerminateJobObject(self._handle, 1))

    def close(self) -> None:
        with self._lock:
            handle = self._handle
            if handle:
                if not _api().kernel32.CloseHandle(handle):
                    raise _winerror("CloseHandle(Job)")
                self._handle = None


class WindowsLpacProcess:
    """Small Popen-compatible adapter used by Studio's existing drain lifecycle."""

    def __init__(
        self,
        argv: tuple[str, ...],
        process: wintypes.HANDLE,
        thread: wintypes.HANDLE,
        pid: int,
        stdout: Any,
        job: _WindowsJob,
    ) -> None:
        self.args = argv
        self._handle = process
        self._thread_handle = thread
        self.pid = pid
        self.stdout = stdout
        self.returncode: int | None = None
        self._reaped = False
        self._unsloth_job = job
        self._operations = threading.Condition()
        self._active_waits = 0
        self._closing = False
        self._close_lock = threading.Lock()

    def poll(self) -> int | None:
        return self._wait_result(0)

    def _wait_result(self, timeout: float | None) -> int | None:
        deadline = None if timeout is None else time.monotonic() + max(0, timeout)
        with self._operations:
            if self._closing and self.returncode is None:
                remaining = None if deadline is None else max(0, deadline - time.monotonic())
                if not self._operations.wait_for(
                    lambda: not self._closing or self.returncode is not None, remaining
                ):
                    return None
            if self._reaped and self.returncode is not None:
                return self.returncode
            if not self._handle:
                raise SandboxUnavailableError(
                    "The process handle was closed without an exit result"
                )
            handle = self._handle
            self._active_waits += 1
        try:
            milliseconds = (
                _INFINITE
                if deadline is None
                else max(0, min(int((deadline - time.monotonic()) * 1000), 0xFFFFFFFE))
            )
            # No coordination lock spans this potentially infinite native wait.
            # close() kills the Job first, then waits for these handle users.
            result = _api().kernel32.WaitForSingleObject(handle, milliseconds)
            if result == _WAIT_TIMEOUT:
                return None
            if result != _WAIT_OBJECT_0:
                raise _winerror("WaitForSingleObject")
            self._reaped = True
            return self._read_returncode()
        finally:
            with self._operations:
                self._active_waits -= 1
                self._operations.notify_all()

    def _read_returncode(self) -> int:
        code = wintypes.DWORD()
        if not _api().kernel32.GetExitCodeProcess(self._handle, ctypes.byref(code)):
            raise _winerror("GetExitCodeProcess")
        # Called only after a signalled process wait. A payload can choose 259
        # as its real exit value; it is not a liveness test at this point.
        self.returncode = ctypes.c_int32(code.value).value
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        result = self._wait_result(timeout)
        if result is None:
            raise subprocess.TimeoutExpired(self.args, timeout)
        return result

    def reap(self, timeout: float = 5) -> None:
        """Terminate the owned Job and confirm exit, including partial-close retries."""
        if self._unsloth_job._handle:
            if not self._unsloth_job.terminate():
                raise _winerror("TerminateJobObject(reap)")
        elif not self._reaped:
            raise SandboxUnavailableError("Cannot reap a process without its owned Job")
        self.wait(timeout = timeout)

    def terminate(self) -> None:
        # The leader may have exited while other processes still own the pipe.
        self._unsloth_job.terminate()

    kill = terminate

    def close(self) -> None:
        if not self._close_lock.acquire(timeout = _PROCESS_CLOSE_TIMEOUT):
            raise SandboxUnavailableError("Another process cleanup is still in progress")
        try:
            with self._operations:
                self._closing = True
            # Kill pipe writers before waiting for native waiters or stream locks.
            self._unsloth_job.close()
            deadline = time.monotonic() + _PROCESS_CLOSE_TIMEOUT
            with self._operations:
                if not self._operations.wait_for(
                    lambda: not self._active_waits, _PROCESS_CLOSE_TIMEOUT
                ):
                    raise SandboxUnavailableError("Process handle still has an active native wait")
                if self._handle and not (self._reaped and self.returncode is not None):
                    milliseconds = max(0, int((deadline - time.monotonic()) * 1000))
                    result = _api().kernel32.WaitForSingleObject(self._handle, milliseconds)
                    if result == _WAIT_TIMEOUT:
                        raise SandboxUnavailableError(
                            "Process exit was not confirmed during cleanup"
                        )
                    if result != _WAIT_OBJECT_0:
                        raise _winerror("WaitForSingleObject(close)")
                    self._reaped = True
                    self._read_returncode()
                self._operations.notify_all()
            if self.stdout is not None:
                self.stdout.close()
                self.stdout = None
            for name in ("_thread_handle", "_handle"):
                handle = getattr(self, name, None)
                if handle:
                    if not _api().kernel32.CloseHandle(handle):
                        raise _winerror(f"CloseHandle({name})")
                    setattr(self, name, None)
        finally:
            with self._operations:
                self._closing = False
                self._operations.notify_all()
            self._close_lock.release()


def _create_job(process_handle, *, active_process_limit = None):
    # Construction precedes CreateProcessW so the Job is inherited atomically.
    owned = _production._job_object_with_limits(active_process_limit = active_process_limit)
    job = _WindowsJob(owned._handle)
    owned._handle = None
    if process_handle is not None:
        if not _api().kernel32.AssignProcessToJobObject(job._handle, process_handle):
            error = _winerror("AssignProcessToJobObject")
            job.close()
            raise error
    return job
