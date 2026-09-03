# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Coordinate launches that consume the Tauri-managed Unsloth environment."""

from __future__ import annotations

import contextlib
import ctypes
import hashlib
import ntpath
import os
import sys
from ctypes import wintypes
from pathlib import Path
from typing import Iterator, Mapping


_RUNTIME_MUTEX_PREFIX = "Global\\UnslothStudioManagedEnvironment-"
_PATH_RUNTIME_MUTEX_PREFIX = "Global\\UnslothStudioManagedEnvironmentPath-"
_RUNTIME_GATE_HANDOFF_ENV = "_UNSLOTH_STUDIO_RUNTIME_GATE_HANDOFF"
_RUNTIME_GATE_ACQUIRE_ENV = "_UNSLOTH_STUDIO_RUNTIME_GATE_ACQUIRE"
_POSIX_RUNTIME_LOCK_FILE = ".studio-runtime.lock"


class StudioRuntimeGateBusy(RuntimeError):
    """The managed Unsloth environment is being installed or repaired."""


class _SidAndAttributes(ctypes.Structure):
    _fields_ = [
        ("sid", ctypes.c_void_p),
        ("attributes", wintypes.DWORD),
    ]


class _TokenUser(ctypes.Structure):
    _fields_ = [("user", _SidAndAttributes)]


def runtime_mutex_name_for_sid(sid: str) -> str:
    return f"{_RUNTIME_MUTEX_PREFIX}{sid}"


def _windows_profile_path() -> Path:
    shell32 = ctypes.WinDLL("shell32", use_last_error = True)
    get_folder_path = shell32.SHGetFolderPathW
    get_folder_path.argtypes = [
        wintypes.HWND,
        ctypes.c_int,
        wintypes.HANDLE,
        wintypes.DWORD,
        wintypes.LPWSTR,
    ]
    get_folder_path.restype = ctypes.c_long

    buffer = ctypes.create_unicode_buffer(32768)
    result = get_folder_path(None, 0x0028, None, 0, buffer)  # CSIDL_PROFILE
    if result != 0:
        raise OSError(
            f"SHGetFolderPathW(CSIDL_PROFILE) failed with HRESULT 0x{result & 0xFFFFFFFF:08x}"
        )
    return Path(buffer.value)


def _resolved_windows_path(path: Path) -> str:
    resolved = str(path.resolve(strict = False))
    if resolved.startswith("\\\\?\\UNC\\"):
        resolved = "\\\\" + resolved[8:]
    elif resolved.startswith("\\\\?\\"):
        resolved = resolved[4:]
    drive, tail = ntpath.splitdrive(resolved)
    if drive and tail and not tail.rstrip("\\/"):
        return drive + "\\"
    return resolved.rstrip("\\/")


def resolve_windows_powershell() -> str:
    r"""Resolve Windows PowerShell without relying solely on ``PATH`` (#9440).

    Public because the install path spawns PowerShell from more than this module: powershell.exe
    lives in a SUBDIRECTORY of System32, so ``CreateProcess``'s implicit system-directory lookup
    never finds it and a caller whose PATH omits that entry gets ``WinError 2`` instead.
    """
    import shutil

    on_path = shutil.which("powershell.exe")
    if on_path:
        return on_path
    system_root = os.environ.get("SystemRoot") or r"C:\Windows"
    builtin = ntpath.join(system_root, "System32", "WindowsPowerShell", "v1.0", "powershell.exe")
    if os.path.isfile(builtin):
        return builtin
    pwsh = shutil.which("pwsh.exe")
    if pwsh:
        return pwsh
    return "powershell.exe"


def _canonical_windows_path(path: Path) -> str:
    return _resolved_windows_path(path).replace("/", "\\")


def _windows_paths_equal(left: str, right: str) -> bool:
    kernel32 = ctypes.WinDLL("kernel32", use_last_error = True)
    compare = kernel32.CompareStringOrdinal
    compare.argtypes = [
        wintypes.LPCWSTR,
        ctypes.c_int,
        wintypes.LPCWSTR,
        ctypes.c_int,
        wintypes.BOOL,
    ]
    compare.restype = ctypes.c_int
    result = compare(left, -1, right, -1, True)
    if result == 0:
        raise ctypes.WinError(ctypes.get_last_error())
    return result == 2  # CSTR_EQUAL


def uses_tauri_managed_root(studio_home: Path) -> bool:
    if sys.platform != "win32":
        return False
    managed_root = _windows_profile_path() / ".unsloth" / "studio"
    return _windows_paths_equal(
        _resolved_windows_path(studio_home),
        _resolved_windows_path(managed_root),
    )


def runtime_mutex_name_for_studio_home(studio_home: Path) -> str:
    if uses_tauri_managed_root(studio_home):
        return runtime_mutex_name_for_sid(_current_windows_user_sid())
    canonical = _resolved_windows_path(studio_home)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return f"{_PATH_RUNTIME_MUTEX_PREFIX}{digest}"


def _current_windows_user_sid() -> str:
    kernel32 = ctypes.WinDLL("kernel32", use_last_error = True)
    advapi32 = ctypes.WinDLL("advapi32", use_last_error = True)

    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL
    kernel32.LocalFree.argtypes = [ctypes.c_void_p]
    kernel32.LocalFree.restype = ctypes.c_void_p

    advapi32.OpenProcessToken.argtypes = [
        wintypes.HANDLE,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.HANDLE),
    ]
    advapi32.OpenProcessToken.restype = wintypes.BOOL
    advapi32.GetTokenInformation.argtypes = [
        wintypes.HANDLE,
        ctypes.c_int,
        ctypes.c_void_p,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD),
    ]
    advapi32.GetTokenInformation.restype = wintypes.BOOL
    advapi32.ConvertSidToStringSidW.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(wintypes.LPWSTR),
    ]
    advapi32.ConvertSidToStringSidW.restype = wintypes.BOOL

    token = wintypes.HANDLE()
    if not advapi32.OpenProcessToken(kernel32.GetCurrentProcess(), 0x0008, ctypes.byref(token)):
        raise ctypes.WinError(ctypes.get_last_error())

    try:
        needed = wintypes.DWORD()
        advapi32.GetTokenInformation(token, 1, None, 0, ctypes.byref(needed))  # TokenUser
        if not needed.value:
            raise ctypes.WinError(ctypes.get_last_error())

        token_buffer = ctypes.create_string_buffer(needed.value)
        if not advapi32.GetTokenInformation(
            token,
            1,
            token_buffer,
            needed,
            ctypes.byref(needed),
        ):
            raise ctypes.WinError(ctypes.get_last_error())

        token_user = ctypes.cast(token_buffer, ctypes.POINTER(_TokenUser)).contents
        sid_text = wintypes.LPWSTR()
        if not advapi32.ConvertSidToStringSidW(token_user.user.sid, ctypes.byref(sid_text)):
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            return sid_text.value
        finally:
            kernel32.LocalFree(ctypes.cast(sid_text, ctypes.c_void_p))
    finally:
        kernel32.CloseHandle(token)


@contextlib.contextmanager
def studio_runtime_launch_guard(
    studio_home: Path,
    *,
    inherited: bool = False,
    wait: bool = False,
) -> Iterator[bool]:
    """Hold the shared launch gate through backend admission."""

    if inherited and wait:
        raise ValueError("a runtime gate cannot be inherited and acquired")
    if inherited:
        yield False
        return

    if sys.platform != "win32":
        import fcntl

        studio_home.mkdir(parents = True, exist_ok = True)
        lock_file = (studio_home / _POSIX_RUNTIME_LOCK_FILE).open("a+b")
        try:
            try:
                flags = fcntl.LOCK_EX if wait else fcntl.LOCK_EX | fcntl.LOCK_NB
                fcntl.flock(lock_file.fileno(), flags)
            except BlockingIOError as exc:
                raise StudioRuntimeGateBusy(str(lock_file.name)) from exc
            yield True
        finally:
            try:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
            finally:
                lock_file.close()
        return

    kernel32 = ctypes.WinDLL("kernel32", use_last_error = True)
    kernel32.CreateMutexW.argtypes = [
        ctypes.c_void_p,
        wintypes.BOOL,
        wintypes.LPCWSTR,
    ]
    kernel32.CreateMutexW.restype = wintypes.HANDLE
    kernel32.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel32.WaitForSingleObject.restype = wintypes.DWORD
    kernel32.ReleaseMutex.argtypes = [wintypes.HANDLE]
    kernel32.ReleaseMutex.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL

    name = runtime_mutex_name_for_studio_home(studio_home)
    handle = kernel32.CreateMutexW(None, False, name)
    if not handle:
        raise ctypes.WinError(ctypes.get_last_error())

    timeout = 0xFFFFFFFF if wait else 0
    wait_result = kernel32.WaitForSingleObject(handle, timeout)
    if wait_result not in (0x00000000, 0x00000080):  # WAIT_OBJECT_0, WAIT_ABANDONED
        kernel32.CloseHandle(handle)
        if wait_result == 0x00000102:  # WAIT_TIMEOUT
            raise StudioRuntimeGateBusy(name)
        raise ctypes.WinError(ctypes.get_last_error())

    try:
        yield True
    finally:
        kernel32.ReleaseMutex(handle)
        kernel32.CloseHandle(handle)


def _windows_path_is_within(candidate: str, root: str) -> bool:
    candidate_key = candidate.rstrip("\\/").replace("/", "\\")
    root_key = root.rstrip("\\/").replace("/", "\\")
    if _windows_paths_equal(candidate_key, root_key):
        return True
    prefix = root_key + "\\"
    if len(candidate_key) < len(prefix):
        return False
    return _windows_paths_equal(candidate_key[: len(prefix)], prefix)


def ensure_managed_environment_is_idle(studio_home: Path) -> None:
    """Reject a Windows update while a confirmed managed executable is running."""

    if sys.platform != "win32":
        return

    import json
    import subprocess

    venv = studio_home / "unsloth_studio"
    protected_root = _canonical_windows_path(venv)
    # Not gated on exists(): a shim renamed out of the way mid-update still runs.
    protected_files = {
        _canonical_windows_path(candidate)
        for candidate in (
            venv / "Scripts" / "unsloth.exe",
            studio_home / "bin" / "unsloth.exe",
        )
    }

    script = (
        "$ErrorActionPreference='Stop';"
        "[Console]::OutputEncoding=[System.Text.UTF8Encoding]::new($false);"
        "$items=@(Get-CimInstance Win32_Process -ErrorAction Stop|"
        "Select-Object ProcessId,ParentProcessId,Name,ExecutablePath);"
        "[Console]::Out.Write(($items|ConvertTo-Json -Compress))"
    )
    result = subprocess.run(
        [resolve_windows_powershell(), "-NoProfile", "-NonInteractive", "-Command", script],
        capture_output = True,
        text = True,
        encoding = "utf-8",
        errors = "replace",
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0),
        check = False,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or f"exit code {result.returncode}"
        raise RuntimeError(f"Could not inspect running processes before Unsloth update: {detail}")

    try:
        payload = json.loads(result.stdout or "[]")
    except json.JSONDecodeError as error:
        raise RuntimeError(
            f"Could not decode the running-process list before Unsloth update: {error}"
        ) from error
    processes = payload if isinstance(payload, list) else [payload]
    process_by_pid = {
        int(process.get("ProcessId") or -1): process
        for process in processes
        if int(process.get("ProcessId") or -1) > 0
    }

    # The updater may itself have been entered through the managed console shim,
    # so exempt verified launcher ancestors only: a managed backend that starts
    # an update as its child must still block replacement.
    #
    # venv\Scripts\python.exe is a redirector (bpo-34977): it starts base Python
    # as a child and waits, so a venv launch arrives as python.exe -> us. Our
    # image is then the base interpreter while sys.executable still names the
    # redirector, which identifies our direct parent as that launcher rather
    # than a live consumer. Exempt that one hop only; above it must be a shim.
    excluded_pids = {os.getpid()}
    descendant_pid = os.getpid()
    self_executable = (process_by_pid.get(os.getpid()) or {}).get("ExecutablePath")
    interpreter = _canonical_windows_path(Path(sys.executable)) if sys.executable else ""
    launcher_redirector = ""
    if (
        interpreter
        and self_executable
        and _windows_path_is_within(interpreter, protected_root)
        and not _windows_paths_equal(
            _canonical_windows_path(Path(str(self_executable))), interpreter
        )
    ):
        launcher_redirector = interpreter
    for _ in range(16):
        descendant = process_by_pid.get(descendant_pid)
        if descendant is None:
            break
        parent_pid = int(descendant.get("ParentProcessId") or -1)
        if parent_pid <= 0 or parent_pid in excluded_pids:
            break
        parent = process_by_pid.get(parent_pid)
        if parent is None:
            break
        parent_executable = parent.get("ExecutablePath")
        if not parent_executable:
            break
        parent_image = _canonical_windows_path(Path(str(parent_executable)))
        is_shim = any(
            _windows_paths_equal(parent_image, protected_file) for protected_file in protected_files
        )
        is_our_redirector = (
            descendant_pid == os.getpid()
            and bool(launcher_redirector)
            and _windows_paths_equal(parent_image, launcher_redirector)
        )
        if not (is_shim or is_our_redirector):
            break
        excluded_pids.add(parent_pid)
        descendant_pid = parent_pid

    for process_id, process in process_by_pid.items():
        if process_id in excluded_pids:
            continue
        executable = process.get("ExecutablePath")
        if not executable:
            continue
        image = _canonical_windows_path(Path(str(executable)))
        if _windows_path_is_within(image, protected_root) or any(
            _windows_paths_equal(image, protected_file) for protected_file in protected_files
        ):
            name = process.get("Name") or "process"
            raise RuntimeError(
                "The managed Unsloth environment is in use by "
                f"{name} (PID {process_id}). Stop that process, then retry the update."
            )


def consume_runtime_gate_handoff() -> bool:
    return os.environ.pop(_RUNTIME_GATE_HANDOFF_ENV, None) == "1"


def consume_runtime_gate_acquire() -> bool:
    return os.environ.pop(_RUNTIME_GATE_ACQUIRE_ENV, None) == "1"


def runtime_gate_child_environment(base: Mapping[str, str] | None = None) -> dict[str, str]:
    child_env = dict(os.environ if base is None else base)
    child_env[_RUNTIME_GATE_HANDOFF_ENV] = "1"
    return child_env
