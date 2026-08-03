# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Coordinate Windows launches that consume the Tauri-managed Studio environment."""

from __future__ import annotations

import contextlib
import ctypes
import os
import sys
from ctypes import wintypes
from pathlib import Path
from typing import Iterator, Mapping


_RUNTIME_MUTEX_PREFIX = "Global\\UnslothStudioManagedEnvironment-"
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


def _canonical_windows_path(path: Path) -> str:
    return str(path.resolve(strict = False)).rstrip("\\/").casefold()


def uses_tauri_managed_root(studio_home: Path) -> bool:
    if sys.platform != "win32":
        return False
    managed_root = _windows_profile_path() / ".unsloth" / "studio"
    return _canonical_windows_path(studio_home) == _canonical_windows_path(managed_root)


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

    if sys.platform != "win32" or inherited or not uses_tauri_managed_root(studio_home):
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

    name = runtime_mutex_name_for_sid(_current_windows_user_sid())
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


def consume_runtime_gate_handoff() -> bool:
    return os.environ.pop(_RUNTIME_GATE_HANDOFF_ENV, None) == "1"


def runtime_gate_child_environment(base: Mapping[str, str] | None = None) -> dict[str, str]:
    child_env = dict(os.environ if base is None else base)
    child_env[_RUNTIME_GATE_HANDOFF_ENV] = "1"
    return child_env
