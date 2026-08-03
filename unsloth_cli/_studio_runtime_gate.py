# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Coordinate Windows launches that consume the Tauri-managed Studio environment."""

from __future__ import annotations

import contextlib
import ctypes
import hashlib
import os
import sys
from ctypes import wintypes
from pathlib import Path
from typing import Iterator, Mapping


_RUNTIME_MUTEX_PREFIX = "Global\\UnslothStudioManagedEnvironment-"
_PATH_RUNTIME_MUTEX_PREFIX = "Global\\UnslothStudioManagedEnvironmentPath-"
_RUNTIME_GATE_HANDOFF_ENV = "_UNSLOTH_STUDIO_RUNTIME_GATE_HANDOFF"


class StudioRuntimeGateBusy(RuntimeError):
    """The managed Studio environment is being installed or repaired."""


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
    resolved = str(path.resolve(strict = False)).rstrip("\\/")
    if resolved.startswith("\\\\?\\UNC\\"):
        resolved = "\\\\" + resolved[8:]
    elif resolved.startswith("\\\\?\\"):
        resolved = resolved[4:]
    return resolved


def _canonical_windows_path(path: Path) -> str:
    return _resolved_windows_path(path).casefold()


def uses_tauri_managed_root(studio_home: Path) -> bool:
    if sys.platform != "win32":
        return False
    managed_root = _windows_profile_path() / ".unsloth" / "studio"
    return _canonical_windows_path(studio_home) == _canonical_windows_path(managed_root)


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
def studio_runtime_launch_guard(studio_home: Path, *, inherited: bool = False) -> Iterator[bool]:
    """Hold the shared Windows launch gate through backend admission."""

    if sys.platform != "win32" or inherited:
        yield False
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

    wait_result = kernel32.WaitForSingleObject(handle, 0)
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
    candidate_key = candidate.rstrip("\\/").replace("/", "\\").casefold()
    root_key = root.rstrip("\\/").replace("/", "\\").casefold()
    return candidate_key == root_key or candidate_key.startswith(root_key + "\\")


def _command_line_references_windows_path(command_line: str, path: str) -> bool:
    normalized_line = command_line.replace("/", "\\").casefold()
    normalized_path = path.rstrip("\\/").replace("/", "\\").casefold()
    search_from = 0
    before_boundaries = {" ", "\t", "\r", "\n", '"', "'", "="}
    after_boundaries = before_boundaries | {"\\"}
    while search_from < len(normalized_line):
        match_index = normalized_line.find(normalized_path, search_from)
        if match_index < 0:
            return False
        end_index = match_index + len(normalized_path)
        before_ok = match_index == 0 or normalized_line[match_index - 1] in before_boundaries
        after_ok = (
            end_index == len(normalized_line) or normalized_line[end_index] in after_boundaries
        )
        if before_ok and after_ok:
            return True
        search_from = end_index
    return False


def _windows_command_line_arguments(command_line: str) -> list[str]:
    shell32 = ctypes.WinDLL("shell32", use_last_error = True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error = True)
    shell32.CommandLineToArgvW.argtypes = [
        wintypes.LPCWSTR,
        ctypes.POINTER(ctypes.c_int),
    ]
    shell32.CommandLineToArgvW.restype = ctypes.POINTER(wintypes.LPWSTR)
    kernel32.LocalFree.argtypes = [ctypes.c_void_p]
    kernel32.LocalFree.restype = ctypes.c_void_p

    argc = ctypes.c_int()
    argv = shell32.CommandLineToArgvW(command_line, ctypes.byref(argc))
    if not argv:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        return [argv[index] for index in range(argc.value)]
    finally:
        kernel32.LocalFree(ctypes.cast(argv, ctypes.c_void_p))


def _command_line_references_resolved_windows_path(
    command_line: str,
    protected_roots: tuple[str, ...],
    protected_files: tuple[str, ...],
    working_directory: Path | None,
) -> bool:
    if any(
        _command_line_references_windows_path(command_line, path)
        for path in (*protected_roots, *protected_files)
    ):
        return True

    for argument in _windows_command_line_arguments(command_line):
        candidates = [argument]
        if "=" in argument:
            candidates.append(argument.split("=", 1)[1])
        for candidate in candidates:
            candidate = candidate.strip().strip('"').strip("'")
            if not candidate:
                continue
            candidate_path = Path(candidate)
            if candidate_path.is_absolute():
                path_candidates = (candidate_path,)
            elif working_directory is not None:
                path_candidates = (working_directory / candidate_path,)
            else:
                path_candidates = ()
            for path_candidate in path_candidates:
                if not path_candidate.exists():
                    continue
                try:
                    resolved = _resolved_windows_path(path_candidate)
                except OSError:
                    continue
                if any(_windows_path_is_within(resolved, root) for root in protected_roots):
                    return True
                if any(
                    resolved.rstrip("\\/").casefold() == path.rstrip("\\/").casefold()
                    for path in protected_files
                ):
                    return True
    return False


def ensure_managed_environment_is_idle(studio_home: Path) -> None:
    """Reject a Windows update while another process consumes managed Studio files."""

    if sys.platform != "win32":
        return

    import json
    import subprocess

    try:
        import psutil
    except ImportError as error:
        raise RuntimeError(
            "Could not inspect process working directories before Studio update: psutil is unavailable"
        ) from error

    venv = studio_home / "unsloth_studio"
    protected_root_spellings = tuple(
        dict.fromkeys(
            spelling
            for candidate in (venv,)
            for spelling in (
                str(candidate.absolute()),
                _resolved_windows_path(candidate),
            )
        )
    )
    shim_candidates = (
        venv / "Scripts" / "unsloth.exe",
        studio_home / "bin" / "unsloth.exe",
    )
    protected_file_spellings = tuple(
        dict.fromkeys(
            spelling
            for candidate in shim_candidates
            if candidate.exists()
            for spelling in (
                str(candidate.absolute()),
                _resolved_windows_path(candidate),
            )
        )
    )

    script = (
        "$ErrorActionPreference='Stop';"
        "[Console]::OutputEncoding=[System.Text.UTF8Encoding]::new($false);"
        "$items=@(Get-CimInstance Win32_Process -ErrorAction Stop|"
        "Select-Object ProcessId,ParentProcessId,Name,ExecutablePath,CommandLine);"
        "[Console]::Out.Write(($items|ConvertTo-Json -Compress))"
    )
    result = subprocess.run(
        ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script],
        capture_output = True,
        text = True,
        encoding = "utf-8",
        errors = "replace",
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0),
        check = False,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or f"exit code {result.returncode}"
        raise RuntimeError(f"Could not inspect running processes before Studio update: {detail}")

    try:
        payload = json.loads(result.stdout or "[]")
    except json.JSONDecodeError as error:
        raise RuntimeError(
            f"Could not decode the running-process list before Studio update: {error}"
        ) from error
    processes = payload if isinstance(payload, list) else [payload]
    current_pid = os.getpid()
    excluded_pids = {current_pid}

    parent_pid = -1
    parent_images: list[str] = []
    try:
        parent_process = psutil.Process(current_pid).parent()
    except (psutil.Error, OSError):
        parent_process = None
    if parent_process is not None:
        parent_pid = int(parent_process.pid)
        try:
            parent_images.append(parent_process.exe())
        except (psutil.Error, OSError):
            pass

    if parent_pid <= 0:
        current = next(
            (
                process
                for process in processes
                if int(process.get("ProcessId") or -1) == current_pid
            ),
            None,
        )
        if current is not None:
            parent_pid = int(current.get("ParentProcessId") or -1)

    if parent_pid > 0:
        parent = next(
            (process for process in processes if int(process.get("ProcessId") or -1) == parent_pid),
            None,
        )
        if parent is not None and parent.get("ExecutablePath"):
            parent_images.append(str(parent["ExecutablePath"]))

    for parent_image in parent_images:
        try:
            parent_key = _resolved_windows_path(Path(parent_image))
        except OSError:
            parent_key = parent_image
        if any(
            parent_key.rstrip("\\/").casefold() == path.rstrip("\\/").casefold()
            for path in protected_file_spellings
        ):
            excluded_pids.add(parent_pid)
            break

    for process in processes:
        process_id = int(process.get("ProcessId") or -1)
        # WMI can include synthetic rows without a usable OS process ID.
        # They cannot identify a live managed-environment consumer, and
        # psutil rejects negative PIDs before its documented Error hierarchy.
        if process_id <= 0:
            continue
        if process_id in excluded_pids:
            continue
        executable = process.get("ExecutablePath") or ""
        image_match = False
        if executable:
            image_spellings = [str(executable)]
            try:
                image_spellings.append(_resolved_windows_path(Path(executable)))
            except OSError:
                pass
            image_match = any(
                any(_windows_path_is_within(image, root) for root in protected_root_spellings)
                or any(
                    image.rstrip("\\/").casefold() == path.rstrip("\\/").casefold()
                    for path in protected_file_spellings
                )
                for image in image_spellings
            )
        command_line = process.get("CommandLine") or ""
        try:
            working_directory = Path(psutil.Process(process_id).cwd())
        except (psutil.Error, OSError):
            working_directory = None
        command_line_match = bool(command_line) and _command_line_references_resolved_windows_path(
            command_line,
            protected_root_spellings,
            protected_file_spellings,
            working_directory,
        )
        if image_match or command_line_match:
            name = process.get("Name") or "process"
            raise RuntimeError(
                "The managed Studio environment is in use by "
                f"{name} (PID {process_id}). Stop that process, then retry the update."
            )


def consume_runtime_gate_handoff() -> bool:
    return os.environ.pop(_RUNTIME_GATE_HANDOFF_ENV, None) == "1"


def runtime_gate_child_environment(base: Mapping[str, str] | None = None) -> dict[str, str]:
    child_env = dict(os.environ if base is None else base)
    child_env[_RUNTIME_GATE_HANDOFF_ENV] = "1"
    return child_env
