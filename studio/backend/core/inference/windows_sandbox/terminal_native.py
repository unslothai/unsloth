# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Private raw zero-capability LPAC Terminal launch and runtime inventory.

Resource ownership derives from the preserved Terminal research slice. No
production backend selection or Python startup capabilities are imported.
"""

from __future__ import annotations
import ctypes
from contextlib import ExitStack, contextmanager
from ctypes import wintypes
from dataclasses import dataclass
import io
import json
import os
from pathlib import Path
import platform
import secrets
import shutil
import socket
import stat
import subprocess
import sys
import sysconfig
import tempfile
import threading
import time
from typing import Any
from . import native_compat as _shared

SandboxUnavailableError = _shared.SandboxUnavailableError
WindowsLpacProcess = _shared.WindowsLpacProcess
_InvocationIdentity = _shared._InvocationIdentity
_PROCESS_CLOSE_TIMEOUT = _shared._PROCESS_CLOSE_TIMEOUT
_PROCESS_INFORMATION = _shared._PROCESS_INFORMATION
_SECURITY_CAPABILITIES = _shared._SECURITY_CAPABILITIES
_STARTUPINFOEXW = _shared._STARTUPINFOEXW
_STARTUPINFOW = _shared._STARTUPINFOW
_WAIT_OBJECT_0 = _shared._WAIT_OBJECT_0
_WindowsJob = _shared._WindowsJob
_api = _shared._api
_canonical_local_directory = _shared._canonical_local_directory
_create_job = _shared._create_job
_environment_block = _shared._environment_block
_initial_appcontainer_environment = _shared._initial_appcontainer_environment
_is_within = _shared._is_within
_winerror = _shared._winerror


def __getattr__(name):
    return getattr(_shared, name)


_RUNTIME_SCAN_ENTRY_LIMIT = 1_000_000

_ERROR_INSUFFICIENT_BUFFER = 122

_PROC_THREAD_ATTRIBUTE_HANDLE_LIST = 0x00020002

_PROC_THREAD_ATTRIBUTE_SECURITY_CAPABILITIES = 0x00020009

_PROC_THREAD_ATTRIBUTE_JOB_LIST = 0x0002000D

_PROC_THREAD_ATTRIBUTE_ALL_APPLICATION_PACKAGES_POLICY = 0x0002000F

_PROCESS_CREATION_ALL_APPLICATION_PACKAGES_OPT_OUT = 0x1

_EXTENDED_STARTUPINFO_PRESENT = 0x00080000

_CREATE_SUSPENDED = 0x00000004

_CREATE_UNICODE_ENVIRONMENT = 0x00000400

_CREATE_BREAKAWAY_FROM_JOB = 0x01000000

_STARTF_USESTDHANDLES = 0x00000100


def _validate_runtime_trees(roots: tuple[str, ...]) -> None:
    """Reject runtime aliases that could receive an AppContainer ACE outside the runtime."""
    entries = 0
    link_counts: dict[tuple[int, int], int] = {}
    link_totals: dict[tuple[int, int], int] = {}
    link_paths: dict[tuple[int, int], str] = {}

    def inspect(path: str) -> None:
        nonlocal entries
        entries += 1
        if entries > _RUNTIME_SCAN_ENTRY_LIMIT:
            raise SandboxUnavailableError("the LPAC runtime exceeds its safety scan limit")
        try:
            info = os.lstat(path)
        except OSError as exc:
            raise SandboxUnavailableError(
                f"an LPAC runtime path cannot be inspected: {path}"
            ) from exc
        if getattr(info, "st_file_attributes", 0) & 0x400:
            target = os.path.realpath(path)
            target_within_runtime = os.path.normcase(target) != os.path.normcase(
                os.path.abspath(path)
            ) and any(_is_within(target, root) for root in roots)
            if not target_within_runtime or not os.path.isfile(target):
                raise SandboxUnavailableError(
                    f"an LPAC runtime contains an unsafe reparse point: {path}"
                )
            return
        if stat.S_ISREG(info.st_mode) and info.st_nlink > 1:
            key = (info.st_dev, info.st_ino)
            link_counts[key] = link_counts.get(key, 0) + 1
            link_totals[key] = max(link_totals.get(key, 0), info.st_nlink)
            link_paths.setdefault(key, path)
        elif not (stat.S_ISREG(info.st_mode) or stat.S_ISDIR(info.st_mode)):
            raise SandboxUnavailableError(f"an LPAC runtime contains a special file: {path}")

    def walk_error(exc: OSError) -> None:
        raise SandboxUnavailableError(
            f"an LPAC runtime cannot be fully inspected: {exc.filename or roots[0]}"
        ) from exc

    for root in roots:
        inspect(root)
        if os.path.isdir(root):
            for base, dirs, names in os.walk(root, followlinks = False, onerror = walk_error):
                for name in [*dirs, *names]:
                    inspect(os.path.join(base, name))
    for key, count in link_counts.items():
        if count < link_totals[key]:
            raise SandboxUnavailableError(
                f"an LPAC runtime contains a hardlink crossing its boundary: {link_paths[key]}"
            )


def _runtime_roots(
    workdir: str,
    argv: tuple[str, ...],
    execution_kind: str | None = None,
) -> tuple[str, ...]:
    if execution_kind not in (None, "python", "terminal"):
        raise SandboxUnavailableError("LPAC requires a recognized tool execution kind")
    candidates: list[str] = []
    # Terminal gets only the selected shell runtime, not Studio's Python
    # installation, packages, policy shim, or an unrelated COMSPEC runtime.
    # None preserves the legacy direct-backend contract; tools set the kind.
    if execution_kind != "terminal":
        candidates.extend(
            (sys.executable, os.path.realpath(sys.executable), sys.prefix, sys.base_prefix)
        )
        candidates.append(os.path.join(os.path.dirname(__file__), "sandbox_site"))
        candidates.extend(path for path in sysconfig.get_paths().values() if path)
    if argv and os.path.isabs(argv[0]):
        executable_dir = os.path.dirname(argv[0])
        candidates.extend((argv[0], executable_dir))
        if os.path.basename(argv[0]).lower() in {"bash", "bash.exe"}:
            shell_root = os.path.dirname(executable_dir)
            candidates.append(os.path.join(shell_root, "usr", "bin"))
    if execution_kind != "terminal":
        comspec = os.environ.get("COMSPEC") or os.path.join(
            os.environ.get("SystemRoot", r"C:\Windows"), "System32", "cmd.exe"
        )
        candidates.append(comspec)
    selected: list[str] = []
    for candidate in candidates:
        if not candidate or not os.path.isabs(candidate) or not os.path.exists(candidate):
            continue
        canonical = os.path.realpath(os.path.abspath(candidate))
        drive, tail = os.path.splitdrive(canonical)
        if not drive or tail in ("", "\\", "/") or canonical.startswith(("\\\\", "//")):
            raise SandboxUnavailableError(f"an LPAC runtime root is unsafe: {canonical}")
        _canonical_local_directory(
            canonical if os.path.isdir(canonical) else os.path.dirname(canonical)
        )
        try:
            common = os.path.commonpath((canonical, workdir))
        except ValueError:
            common = ""
        if common in (canonical, workdir):
            raise SandboxUnavailableError("the LPAC runtime and writable workdir overlap")
        if any(_is_within(canonical, existing) for existing in selected):
            continue
        selected = [existing for existing in selected if not _is_within(existing, canonical)]
        selected.append(canonical)
    return tuple(selected)


def _canonical_inner_argv(argv: tuple[str, ...], env: dict[str, str]) -> tuple[str, ...]:
    if not argv or not argv[0] or "\0" in argv[0]:
        raise SandboxUnavailableError("LPAC requires a non-empty executable path")
    executable = argv[0]
    if not os.path.isabs(executable):
        executable = shutil.which(executable, path = env.get("PATH")) or ""
    if not executable or not os.path.isfile(executable):
        raise SandboxUnavailableError("LPAC could not resolve the selected tool executable")
    canonical = os.path.realpath(os.path.abspath(executable))
    if canonical.startswith(("\\\\", "//")) or not os.path.splitdrive(canonical)[0]:
        raise SandboxUnavailableError("LPAC requires an executable on a local drive")
    return (canonical, *argv[1:])


def _needs_explicit_acl(path: str) -> bool:
    """System files already carry restricted-package ACLs and are not user-editable."""
    windows = os.path.realpath(os.environ.get("SystemRoot", r"C:\Windows"))
    try:
        return os.path.commonpath((os.path.realpath(path), windows)) != windows
    except ValueError:
        return True


_pending_native_cleanup: set[_NativeLaunchOwner] = set()

_pending_native_lock = threading.Lock()


def _retry_native_cleanup() -> None:
    with _pending_native_lock:
        pending = tuple(_pending_native_cleanup)
    for owner in pending:
        owner.cleanup()
    with _pending_native_lock:
        if _pending_native_cleanup:
            raise SandboxUnavailableError("A prior native sandbox still owns unclosed resources.")


class _NativeLaunchOwner:
    """Keep actual native resources alive across discarded launch/cleanup errors."""

    def __init__(
        self,
        identity,
        *,
        owns_identity = False,
    ):
        self.identity, self.owns_identity = identity, owns_identity
        self.handles: set[int] = set()
        self.info = _PROCESS_INFORMATION()
        self.job = self.process = self.stdout = None
        self.attempted = self.started = self.closed = False
        self._lock = threading.Lock()

    def close_handle(self, handle):
        if handle in self.handles:
            if not _api().kernel32.CloseHandle(handle):
                raise _winerror("CloseHandle(native launch)")
            self.handles.remove(handle)

    def prepare_stdio(self, kwargs):
        from .native_io import NativePipeReader

        api = _api().kernel32
        api.CreatePipe.argtypes = [
            ctypes.POINTER(wintypes.HANDLE),
            ctypes.POINTER(wintypes.HANDLE),
            ctypes.c_void_p,
            wintypes.DWORD,
        ]
        api.CreatePipe.restype = wintypes.BOOL
        read, write = wintypes.HANDLE(), wintypes.HANDLE()
        if not api.CreatePipe(ctypes.byref(read), ctypes.byref(write), None, 4096):
            raise _winerror("CreatePipe(native launch)")
        self.handles.update((read.value, write.value))
        self.stdout = NativePipeReader(read.value, self.close_handle)
        self.stdout = io.BufferedReader(self.stdout)
        self.stdout = io.TextIOWrapper(
            self.stdout,
            encoding = kwargs.get("encoding", "utf-8"),
            errors = kwargs.get("errors", "replace"),
        )
        api.CreateFileW.argtypes = [
            wintypes.LPCWSTR,
            wintypes.DWORD,
            wintypes.DWORD,
            ctypes.c_void_p,
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.HANDLE,
        ]
        api.CreateFileW.restype = wintypes.HANDLE
        stdin = api.CreateFileW("NUL", 0x80000000, 3, None, 3, 0, None)
        if stdin == ctypes.c_void_p(-1).value:
            raise _winerror("CreateFileW(native stdin)")
        self.handles.add(stdin)
        api.SetHandleInformation.argtypes = [wintypes.HANDLE, wintypes.DWORD, wintypes.DWORD]
        api.SetHandleInformation.restype = wintypes.BOOL
        for handle in (stdin, write.value):
            if not api.SetHandleInformation(handle, 1, 1):
                raise _winerror("SetHandleInformation(native stdio)")
        return stdin, write.value

    def cleanup(self):
        acquired = self._lock.acquire(timeout = _PROCESS_CLOSE_TIMEOUT)
        try:
            if not acquired:
                raise SandboxUnavailableError("Native launch cleanup is still in progress.")
            if self.closed:
                with _pending_native_lock:
                    _pending_native_cleanup.discard(self)
                return
            api = _api().kernel32
            if self.process is not None:
                self.process.close()
                self.process = None
            else:
                if self.info.hProcess:
                    if self.job is None or not self.job.terminate():
                        raise _winerror("TerminateJobObject(unreturned native launch)")
                    if api.WaitForSingleObject(self.info.hProcess, 5000) != _WAIT_OBJECT_0:
                        raise SandboxUnavailableError("Unreturned native process was not reaped.")
                for field in ("hThread", "hProcess"):
                    handle = getattr(self.info, field)
                    if handle:
                        if not api.CloseHandle(handle):
                            raise _winerror(f"CloseHandle(native {field})")
                        setattr(self.info, field, None)
            if self.job is not None:
                self.job.close()
            if self.stdout is not None:
                self.stdout.close()
            for handle in tuple(self.handles):
                self.close_handle(handle)
            if self.owns_identity:
                self.identity.cleanup()
            self.closed = True
            with _pending_native_lock:
                _pending_native_cleanup.discard(self)
        except BaseException as error:
            with _pending_native_lock:
                _pending_native_cleanup.add(self)
            failure = SandboxUnavailableError(
                "Native sandbox cleanup is incomplete; new native launches are blocked until cleanup succeeds."
            )
            failure.retained_launch = self
            raise failure from error
        finally:
            if acquired:
                self._lock.release()


def _spawn_lpac(
    prepared: PreparedSandboxLaunch,
    popen_kwargs: dict[str, Any],
    identity: _InvocationIdentity,
    *,
    command_line: str | None = None,
    before_resume = None,
) -> WindowsLpacProcess:
    if before_resume is not None and not callable(before_resume):
        raise SandboxUnavailableError("Invalid native LPAC pre-resume ownership callback")
    if command_line is not None and (
        type(command_line) is not str
        or not command_line
        or "\0" in command_line
        or len(command_line.encode("utf-16-le")) // 2 >= 32767
    ):
        raise SandboxUnavailableError("Invalid native LPAC command line")
    if (
        popen_kwargs.get("stdout") != subprocess.PIPE
        or popen_kwargs.get("stderr") != subprocess.STDOUT
        or popen_kwargs.get("stdin") != subprocess.DEVNULL
        or not popen_kwargs.get("close_fds", True)
    ):
        raise SandboxUnavailableError("LPAC accepts only Studio's closed-descriptor stdio plan")
    owner = getattr(prepared, "_lpac_native_owner", None)
    if owner is None:
        owner = _NativeLaunchOwner(identity)
        prepared._lpac_native_owner = owner
        prepared.cleanup_callbacks.append(owner.cleanup)
    if (
        type(owner) is not _NativeLaunchOwner
        or owner.identity is not identity
        or owner.attempted
        or owner.closed
    ):
        raise SandboxUnavailableError("A native launch cannot be replaced, reused or replayed.")
    _retry_native_cleanup()
    owner.attempted = True
    api = _api()
    process_info = owner.info
    attribute_buffer: ctypes.Array[Any] | None = None
    attribute_list: ctypes.c_void_p | None = None
    attributes_initialized = False
    job: _WindowsJob | None = None
    try:
        stdin, write = owner.prepare_stdio(popen_kwargs)
        child_stdin = wintypes.HANDLE(stdin)
        child_stdout = wintypes.HANDLE(write)
        handles = (wintypes.HANDLE * 2)(child_stdin, child_stdout)

        # Ownership must exist inside CreateProcessW, including while the child
        # is suspended. A later assignment leaks it if the Studio broker dies.
        owner.job = job = _create_job(None)
        jobs = (wintypes.HANDLE * 1)(job._handle)

        size = ctypes.c_size_t()
        api.kernel32.InitializeProcThreadAttributeList(None, 4, 0, ctypes.byref(size))
        if ctypes.get_last_error() != _ERROR_INSUFFICIENT_BUFFER or not size.value:
            raise _winerror("InitializeProcThreadAttributeList(size)")
        attribute_buffer = ctypes.create_string_buffer(size.value)
        attribute_list = ctypes.cast(attribute_buffer, ctypes.c_void_p)
        if not api.kernel32.InitializeProcThreadAttributeList(
            attribute_list, 4, 0, ctypes.byref(size)
        ):
            raise _winerror("InitializeProcThreadAttributeList")
        attributes_initialized = True
        capabilities = _SECURITY_CAPABILITIES(identity.sid, None, 0, 0)
        policy = wintypes.DWORD(_PROCESS_CREATION_ALL_APPLICATION_PACKAGES_OPT_OUT)
        for key, value, value_size in (
            (
                _PROC_THREAD_ATTRIBUTE_SECURITY_CAPABILITIES,
                ctypes.byref(capabilities),
                ctypes.sizeof(capabilities),
            ),
            (
                _PROC_THREAD_ATTRIBUTE_ALL_APPLICATION_PACKAGES_POLICY,
                ctypes.byref(policy),
                ctypes.sizeof(policy),
            ),
            (
                _PROC_THREAD_ATTRIBUTE_HANDLE_LIST,
                ctypes.byref(handles),
                ctypes.sizeof(handles),
            ),
            (
                _PROC_THREAD_ATTRIBUTE_JOB_LIST,
                ctypes.byref(jobs),
                ctypes.sizeof(jobs),
            ),
        ):
            if not api.kernel32.UpdateProcThreadAttribute(
                attribute_list,
                0,
                key,
                value,
                value_size,
                None,
                None,
            ):
                raise _winerror(f"UpdateProcThreadAttribute({key:#x})")

        startup = _STARTUPINFOEXW()
        startup.StartupInfo.cb = ctypes.sizeof(startup)
        startup.StartupInfo.dwFlags = _STARTF_USESTDHANDLES
        startup.StartupInfo.hStdInput = child_stdin
        startup.StartupInfo.hStdOutput = child_stdout
        startup.StartupInfo.hStdError = child_stdout
        startup.lpAttributeList = attribute_list
        native_command = ctypes.create_unicode_buffer(
            subprocess.list2cmdline(prepared.argv) if command_line is None else command_line
        )
        environment = _environment_block(_initial_appcontainer_environment(prepared.env, identity))
        flags = (
            int(popen_kwargs.get("creationflags", 0))
            | _CREATE_SUSPENDED
            | _CREATE_UNICODE_ENVIRONMENT
            | _EXTENDED_STARTUPINFO_PRESENT
        )
        if flags & _CREATE_BREAKAWAY_FROM_JOB:
            raise SandboxUnavailableError("LPAC processes may not break away from their Job Object")
        if not api.kernel32.CreateProcessW(
            prepared.argv[0],
            native_command,
            None,
            None,
            True,
            flags,
            environment,
            prepared.workdir,
            ctypes.cast(ctypes.byref(startup), ctypes.POINTER(_STARTUPINFOW)),
            ctypes.byref(process_info),
        ):
            raise _winerror("CreateProcessW(LPAC)")
        owner.close_handle(write)
        owner.close_handle(stdin)
        owner.process = WindowsLpacProcess(
            prepared.argv,
            process_info.hProcess,
            process_info.hThread,
            int(process_info.dwProcessId),
            owner.stdout,
            job,
        )
        process_info.hProcess = process_info.hThread = None
        if before_resume is not None:
            before_resume(owner.process)
        if api.kernel32.ResumeThread(owner.process._thread_handle) == 0xFFFFFFFF:
            raise _winerror("ResumeThread")
        owner.started = True
        return owner.process
    except BaseException:
        owner.cleanup()
        raise
    finally:
        if attributes_initialized and attribute_list is not None:
            api.kernel32.DeleteProcThreadAttributeList(attribute_list)
