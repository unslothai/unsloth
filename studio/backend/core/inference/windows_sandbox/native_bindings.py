# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Stdlib-only Win32 bindings for isolated bootstrap workers.

Extracted from committed 85c6db4b8, with no backend selection, runtime probing,
production logging, or third-party imports. Job limits follow the current
production allocation policy; each experimental owner keeps its own handles.
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


class SandboxUnavailableError(RuntimeError):
    """A native bootstrap operation failed closed."""


_PROFILE_PREFIX = "unsloth.studio."

_SCAN_ENTRY_LIMIT = 100_000

_SE_FILE_OBJECT = 1

_DACL_SECURITY_INFORMATION = 0x00000004

_TRUSTEE_IS_SID = 0

_TRUSTEE_IS_UNKNOWN = 0

_NO_MULTIPLE_TRUSTEE = 0

_GRANT_ACCESS = 1

_REVOKE_ACCESS = 4

_SUB_CONTAINERS_AND_OBJECTS_INHERIT = 3

_GENERIC_READ = 0x80000000

_GENERIC_WRITE = 0x40000000

_GENERIC_EXECUTE = 0x20000000

_DELETE = 0x00010000

_FILE_TRAVERSE = 0x00000020

_WAIT_OBJECT_0 = 0

_WAIT_TIMEOUT = 258

_INFINITE = 0xFFFFFFFF

_STILL_ACTIVE = 259

_PROCESS_QUERY_LIMITED_INFORMATION = 0x1000


class _TRUSTEE_W(ctypes.Structure):
    pass


_TRUSTEE_W._fields_ = [
    ("pMultipleTrustee", ctypes.POINTER(_TRUSTEE_W)),
    ("MultipleTrusteeOperation", wintypes.DWORD),
    ("TrusteeForm", wintypes.DWORD),
    ("TrusteeType", wintypes.DWORD),
    ("ptstrName", wintypes.LPWSTR),
]


class _EXPLICIT_ACCESS_W(ctypes.Structure):
    _fields_ = [
        ("grfAccessPermissions", wintypes.DWORD),
        ("grfAccessMode", wintypes.DWORD),
        ("grfInheritance", wintypes.DWORD),
        ("Trustee", _TRUSTEE_W),
    ]


class _SECURITY_CAPABILITIES(ctypes.Structure):
    _fields_ = [
        ("AppContainerSid", ctypes.c_void_p),
        ("Capabilities", ctypes.c_void_p),
        ("CapabilityCount", wintypes.DWORD),
        ("Reserved", wintypes.DWORD),
    ]


class _SECURITY_DESCRIPTOR(ctypes.Structure):
    _fields_ = [
        ("Revision", ctypes.c_ubyte),
        ("Sbz1", ctypes.c_ubyte),
        ("Control", wintypes.WORD),
        ("Owner", ctypes.c_void_p),
        ("Group", ctypes.c_void_p),
        ("Sacl", ctypes.c_void_p),
        ("Dacl", ctypes.c_void_p),
    ]


class _ACL(ctypes.Structure):
    _fields_ = [
        ("revision", ctypes.c_ubyte),
        ("reserved", ctypes.c_ubyte),
        ("size", wintypes.WORD),
        ("count", wintypes.WORD),
        ("reserved2", wintypes.WORD),
    ]


class _ACE_HEADER(ctypes.Structure):
    _fields_ = [("kind", ctypes.c_ubyte), ("flags", ctypes.c_ubyte), ("size", wintypes.WORD)]


class _STARTUPINFOW(ctypes.Structure):
    _fields_ = [
        ("cb", wintypes.DWORD),
        ("lpReserved", wintypes.LPWSTR),
        ("lpDesktop", wintypes.LPWSTR),
        ("lpTitle", wintypes.LPWSTR),
        ("dwX", wintypes.DWORD),
        ("dwY", wintypes.DWORD),
        ("dwXSize", wintypes.DWORD),
        ("dwYSize", wintypes.DWORD),
        ("dwXCountChars", wintypes.DWORD),
        ("dwYCountChars", wintypes.DWORD),
        ("dwFillAttribute", wintypes.DWORD),
        ("dwFlags", wintypes.DWORD),
        ("wShowWindow", wintypes.WORD),
        ("cbReserved2", wintypes.WORD),
        ("lpReserved2", ctypes.POINTER(ctypes.c_ubyte)),
        ("hStdInput", wintypes.HANDLE),
        ("hStdOutput", wintypes.HANDLE),
        ("hStdError", wintypes.HANDLE),
    ]


class _STARTUPINFOEXW(ctypes.Structure):
    _fields_ = [("StartupInfo", _STARTUPINFOW), ("lpAttributeList", ctypes.c_void_p)]


class _PROCESS_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("hProcess", wintypes.HANDLE),
        ("hThread", wintypes.HANDLE),
        ("dwProcessId", wintypes.DWORD),
        ("dwThreadId", wintypes.DWORD),
    ]


class _JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
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


class _IO_COUNTERS(ctypes.Structure):
    _fields_ = [
        (name, ctypes.c_uint64)
        for name in (
            "ReadOperationCount",
            "WriteOperationCount",
            "OtherOperationCount",
            "ReadTransferCount",
            "WriteTransferCount",
            "OtherTransferCount",
        )
    ]


class _JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("BasicLimitInformation", _JOBOBJECT_BASIC_LIMIT_INFORMATION),
        ("IoInfo", _IO_COUNTERS),
        ("ProcessMemoryLimit", ctypes.c_size_t),
        ("JobMemoryLimit", ctypes.c_size_t),
        ("PeakProcessMemoryUsed", ctypes.c_size_t),
        ("PeakJobMemoryUsed", ctypes.c_size_t),
    ]


@dataclass(frozen = True)
class _WinApi:
    kernel32: Any
    advapi32: Any
    userenv: Any
    ole32: Any


_API: _WinApi | None = None


def _winerror(prefix: str, code: int | None = None) -> OSError:
    number = ctypes.get_last_error() if code is None else int(code)
    return OSError(number, f"{prefix}: {ctypes.FormatError(number).strip()}")


def _hresult_error(prefix: str, value: int) -> OSError:
    unsigned = ctypes.c_uint32(value).value
    return OSError(unsigned, f"{prefix} failed with HRESULT 0x{unsigned:08x}")


def _api() -> _WinApi:
    global _API
    if _API is not None:
        return _API
    if os.name != "nt" or not hasattr(ctypes, "WinDLL"):
        raise OSError("Windows AppContainer APIs are unavailable on this host")
    kernel32 = ctypes.WinDLL("kernel32", use_last_error = True)
    advapi32 = ctypes.WinDLL("advapi32", use_last_error = True)
    userenv = ctypes.WinDLL("userenv", use_last_error = True)
    ole32 = ctypes.WinDLL("ole32", use_last_error = True)

    userenv.CreateAppContainerProfile.argtypes = [
        wintypes.LPCWSTR,
        wintypes.LPCWSTR,
        wintypes.LPCWSTR,
        ctypes.c_void_p,
        wintypes.DWORD,
        ctypes.POINTER(ctypes.c_void_p),
    ]
    userenv.CreateAppContainerProfile.restype = ctypes.c_long
    userenv.DeleteAppContainerProfile.argtypes = [wintypes.LPCWSTR]
    userenv.DeleteAppContainerProfile.restype = ctypes.c_long
    userenv.GetAppContainerFolderPath.argtypes = [
        wintypes.LPCWSTR,
        ctypes.POINTER(wintypes.LPWSTR),
    ]
    userenv.GetAppContainerFolderPath.restype = ctypes.c_long

    advapi32.ConvertSidToStringSidW.argtypes = [ctypes.c_void_p, ctypes.POINTER(wintypes.LPWSTR)]
    advapi32.ConvertSidToStringSidW.restype = wintypes.BOOL
    advapi32.FreeSid.argtypes = [ctypes.c_void_p]
    advapi32.FreeSid.restype = ctypes.c_void_p
    advapi32.GetLengthSid.argtypes = [ctypes.c_void_p]
    advapi32.GetLengthSid.restype = wintypes.DWORD
    advapi32.GetAce.argtypes = [ctypes.c_void_p, wintypes.DWORD, ctypes.POINTER(ctypes.c_void_p)]
    advapi32.GetAce.restype = wintypes.BOOL
    advapi32.GetNamedSecurityInfoW.argtypes = [
        wintypes.LPWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_void_p),
    ]
    advapi32.GetNamedSecurityInfoW.restype = wintypes.DWORD
    advapi32.SetEntriesInAclW.argtypes = [
        wintypes.ULONG,
        ctypes.POINTER(_EXPLICIT_ACCESS_W),
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_void_p),
    ]
    advapi32.SetEntriesInAclW.restype = wintypes.DWORD
    advapi32.SetNamedSecurityInfoW.argtypes = [
        wintypes.LPWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    advapi32.SetNamedSecurityInfoW.restype = wintypes.DWORD
    advapi32.InitializeSecurityDescriptor.argtypes = [ctypes.c_void_p, wintypes.DWORD]
    advapi32.InitializeSecurityDescriptor.restype = wintypes.BOOL
    advapi32.SetSecurityDescriptorDacl.argtypes = [
        ctypes.c_void_p,
        wintypes.BOOL,
        ctypes.c_void_p,
        wintypes.BOOL,
    ]
    advapi32.SetSecurityDescriptorDacl.restype = wintypes.BOOL
    advapi32.SetFileSecurityW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, ctypes.c_void_p]
    advapi32.SetFileSecurityW.restype = wintypes.BOOL

    kernel32.InitializeProcThreadAttributeList.argtypes = [
        ctypes.c_void_p,
        wintypes.DWORD,
        wintypes.DWORD,
        ctypes.POINTER(ctypes.c_size_t),
    ]
    kernel32.InitializeProcThreadAttributeList.restype = wintypes.BOOL
    kernel32.UpdateProcThreadAttribute.argtypes = [
        ctypes.c_void_p,
        wintypes.DWORD,
        ctypes.c_size_t,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    kernel32.UpdateProcThreadAttribute.restype = wintypes.BOOL
    kernel32.DeleteProcThreadAttributeList.argtypes = [ctypes.c_void_p]
    kernel32.CreateProcessW.argtypes = [
        wintypes.LPCWSTR,
        wintypes.LPWSTR,
        ctypes.c_void_p,
        ctypes.c_void_p,
        wintypes.BOOL,
        wintypes.DWORD,
        ctypes.c_void_p,
        wintypes.LPCWSTR,
        ctypes.POINTER(_STARTUPINFOW),
        ctypes.POINTER(_PROCESS_INFORMATION),
    ]
    kernel32.CreateProcessW.restype = wintypes.BOOL
    kernel32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
    kernel32.CreateJobObjectW.restype = wintypes.HANDLE
    kernel32.SetInformationJobObject.argtypes = [
        wintypes.HANDLE,
        ctypes.c_int,
        ctypes.c_void_p,
        wintypes.DWORD,
    ]
    kernel32.SetInformationJobObject.restype = wintypes.BOOL
    kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
    kernel32.ResumeThread.argtypes = [wintypes.HANDLE]
    kernel32.ResumeThread.restype = wintypes.DWORD
    kernel32.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel32.WaitForSingleObject.restype = wintypes.DWORD
    kernel32.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    kernel32.GetExitCodeProcess.restype = wintypes.BOOL
    kernel32.TerminateProcess.argtypes = [wintypes.HANDLE, wintypes.UINT]
    kernel32.TerminateProcess.restype = wintypes.BOOL
    kernel32.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
    kernel32.TerminateJobObject.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL
    kernel32.GetCurrentProcess.argtypes = []
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    kernel32.GetCurrentProcessId.argtypes = []
    kernel32.GetCurrentProcessId.restype = wintypes.DWORD
    kernel32.GetProcessTimes.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(wintypes.FILETIME),
        ctypes.POINTER(wintypes.FILETIME),
        ctypes.POINTER(wintypes.FILETIME),
        ctypes.POINTER(wintypes.FILETIME),
    ]
    kernel32.GetProcessTimes.restype = wintypes.BOOL
    kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.LocalFree.argtypes = [ctypes.c_void_p]
    kernel32.LocalFree.restype = ctypes.c_void_p
    kernel32.GetDriveTypeW.argtypes = [wintypes.LPCWSTR]
    kernel32.GetDriveTypeW.restype = wintypes.UINT
    ole32.CoTaskMemFree.argtypes = [ctypes.c_void_p]

    _API = _WinApi(kernel32, advapi32, userenv, ole32)
    return _API


def _sid_string(api: _WinApi, sid: ctypes.c_void_p) -> str:
    value = wintypes.LPWSTR()
    if not api.advapi32.ConvertSidToStringSidW(sid, ctypes.byref(value)):
        raise _winerror("ConvertSidToStringSidW")
    try:
        return value.value
    finally:
        api.kernel32.LocalFree(value)


def _process_identity(pid: int | None = None) -> tuple[int, int] | None:
    api = _api()
    close_handle = pid is not None
    if pid is None:
        pid = int(api.kernel32.GetCurrentProcessId())
        handle = api.kernel32.GetCurrentProcess()
    else:
        handle = api.kernel32.OpenProcess(_PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not handle:
            error = ctypes.get_last_error()
            if error == 87:  # no process has this PID
                return None
            raise _winerror("OpenProcess(manifest owner)", error)
    created = wintypes.FILETIME()
    exited = wintypes.FILETIME()
    kernel = wintypes.FILETIME()
    user = wintypes.FILETIME()
    try:
        if close_handle:
            exit_code = wintypes.DWORD()
            if not api.kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
                raise _winerror("GetExitCodeProcess(manifest owner)")
            if exit_code.value != _STILL_ACTIVE:
                return None
        if not api.kernel32.GetProcessTimes(
            handle,
            ctypes.byref(created),
            ctypes.byref(exited),
            ctypes.byref(kernel),
            ctypes.byref(user),
        ):
            raise _winerror("GetProcessTimes(manifest owner)")
        ticks = (int(created.dwHighDateTime) << 32) | int(created.dwLowDateTime)
        return pid, ticks
    finally:
        if close_handle:
            api.kernel32.CloseHandle(handle)


def _manifest_root() -> str:
    local = os.environ.get("LOCALAPPDATA")
    if not local or not os.path.isabs(local):
        raise SandboxUnavailableError("LOCALAPPDATA is unavailable for LPAC ownership manifests")
    local = os.path.realpath(local)
    spelled = os.path.join(local, "Unsloth", "Studio", "lpac-manifests")
    os.makedirs(spelled, mode = 0o700, exist_ok = True)
    root = os.path.realpath(spelled)
    if not _is_within(root, local):
        raise SandboxUnavailableError("the LPAC manifest directory escapes LOCALAPPDATA")
    if getattr(os.lstat(spelled), "st_file_attributes", 0) & 0x400:
        raise SandboxUnavailableError("the LPAC manifest directory is a reparse point")
    return root


def _validated_private_temp(profile_folder: str, private_temp: str) -> str:
    expected_parent = os.path.join(os.path.realpath(profile_folder), "Temp")
    spelled = os.path.abspath(private_temp)
    name = os.path.basename(spelled)
    current = os.path.normcase(spelled) == os.path.normcase(expected_parent)
    # Older manifests owned a random child of Temp. Retain their cleanup path.
    legacy = (
        os.path.normcase(os.path.dirname(spelled)) == os.path.normcase(expected_parent)
        and len(name) == 24
        and all(character in "0123456789abcdef" for character in name.lower())
    )
    if not (current or legacy):
        raise SandboxUnavailableError("an LPAC private temp path is outside its profile")
    for root in {expected_parent, spelled}:
        if os.path.lexists(root) and getattr(os.lstat(root), "st_file_attributes", 0) & 0x400:
            raise SandboxUnavailableError("the LPAC private temp root is a reparse point")
    if os.path.isdir(spelled):
        for base, dirs, names in os.walk(spelled, followlinks = False):
            for name in [*dirs, *names]:
                path = os.path.join(base, name)
                if getattr(os.lstat(path), "st_file_attributes", 0) & 0x400:
                    raise SandboxUnavailableError(
                        f"the LPAC private temp contains a reparse point: {path}"
                    )
    return spelled


def _acl_contains_sid(acl: ctypes.c_void_p, sid: ctypes.c_void_p) -> bool:
    if not acl:
        return False
    api = _api()
    sid_bytes = ctypes.string_at(sid, api.advapi32.GetLengthSid(sid))
    header = ctypes.cast(acl, ctypes.POINTER(_ACL)).contents
    for index in range(header.count):
        entry = ctypes.c_void_p()
        if not api.advapi32.GetAce(acl, index, ctypes.byref(entry)):
            raise _winerror("GetAce(LPAC cleanup)")
        size = ctypes.cast(entry, ctypes.POINTER(_ACE_HEADER)).contents.size
        # Cover ordinary, inherited, object, and callback ACE layouts. A match
        # only requests REVOKE_ACCESS for this SID; it never deletes a whole ACE.
        if sid_bytes in ctypes.string_at(entry, size):
            return True
    return False


def _set_sid_acl(
    path: str,
    sid: ctypes.c_void_p,
    *,
    mode: int,
    access: int = 0,
    inheritance: int | None = None,
) -> None:
    api = _api()
    old_acl = ctypes.c_void_p()
    descriptor = ctypes.c_void_p()
    result = api.advapi32.GetNamedSecurityInfoW(
        path,
        _SE_FILE_OBJECT,
        _DACL_SECURITY_INFORMATION,
        None,
        None,
        ctypes.byref(old_acl),
        None,
        ctypes.byref(descriptor),
    )
    if result != 0:
        raise _winerror(f"GetNamedSecurityInfoW({path})", result)
    new_acl = ctypes.c_void_p()
    try:
        # A failed grant is still in the write-ahead manifest. Cleanup must not
        # need WRITE_DAC on a read-only host path which never received our SID.
        if mode == _REVOKE_ACCESS and not _acl_contains_sid(old_acl, sid):
            return
        trustee = _TRUSTEE_W(
            None,
            _NO_MULTIPLE_TRUSTEE,
            _TRUSTEE_IS_SID,
            _TRUSTEE_IS_UNKNOWN,
            ctypes.cast(sid, wintypes.LPWSTR),
        )
        entry = _EXPLICIT_ACCESS_W(
            access,
            mode,
            (_SUB_CONTAINERS_AND_OBJECTS_INHERIT if os.path.isdir(path) else 0)
            if inheritance is None
            else inheritance,
            trustee,
        )
        result = api.advapi32.SetEntriesInAclW(
            1,
            ctypes.byref(entry),
            old_acl,
            ctypes.byref(new_acl),
        )
        if result != 0:
            raise _winerror(f"SetEntriesInAclW({path})", result)
        if inheritance == 0:
            # SetNamedSecurityInfo propagates a directory DACL through its
            # descendants. Ancestor traversal is deliberately an exact ACE,
            # so use SetFileSecurity with an absolute descriptor instead.
            exact_descriptor = _SECURITY_DESCRIPTOR()
            if not api.advapi32.InitializeSecurityDescriptor(ctypes.byref(exact_descriptor), 1):
                raise _winerror(f"InitializeSecurityDescriptor({path})")
            if not api.advapi32.SetSecurityDescriptorDacl(
                ctypes.byref(exact_descriptor), True, new_acl, False
            ):
                raise _winerror(f"SetSecurityDescriptorDacl({path})")
            if not api.advapi32.SetFileSecurityW(
                path,
                _DACL_SECURITY_INFORMATION,
                ctypes.byref(exact_descriptor),
            ):
                raise _winerror(f"SetFileSecurityW({path})")
        else:
            result = api.advapi32.SetNamedSecurityInfoW(
                path,
                _SE_FILE_OBJECT,
                _DACL_SECURITY_INFORMATION,
                None,
                None,
                new_acl,
                None,
            )
            if result != 0:
                raise _winerror(f"SetNamedSecurityInfoW({path})", result)
    finally:
        if new_acl:
            api.kernel32.LocalFree(new_acl)
        if descriptor:
            api.kernel32.LocalFree(descriptor)


def _grant_read_execute(path: str, sid: ctypes.c_void_p) -> None:
    _set_sid_acl(path, sid, mode = _GRANT_ACCESS, access = _GENERIC_READ | _GENERIC_EXECUTE)


def _grant_modify(path: str, sid: ctypes.c_void_p) -> None:
    _set_sid_acl(
        path,
        sid,
        mode = _GRANT_ACCESS,
        access = _GENERIC_READ | _GENERIC_WRITE | _GENERIC_EXECUTE | _DELETE,
    )


def _grant_traverse(path: str, sid: ctypes.c_void_p) -> None:
    _set_sid_acl(
        path,
        sid,
        mode = _GRANT_ACCESS,
        access = _FILE_TRAVERSE,
        inheritance = 0,
    )


def _revoke_sid(
    path: str,
    sid: ctypes.c_void_p,
    *,
    exact: bool = False,
) -> None:
    if os.path.exists(path):
        _set_sid_acl(path, sid, mode = _REVOKE_ACCESS, inheritance = 0 if exact else None)


def _canonical_local_directory(path: str) -> str:
    canonical = os.path.realpath(os.path.abspath(path))
    drive, tail = os.path.splitdrive(canonical)
    if (
        not drive
        or tail in ("", "\\", "/")
        or canonical.startswith(("\\\\", "//"))
        or not os.path.isdir(canonical)
    ):
        raise SandboxUnavailableError("LPAC requires a non-root directory on a local drive")
    if _api().kernel32.GetDriveTypeW(drive + "\\") != 3:
        raise SandboxUnavailableError("LPAC does not accept network, removable, or virtual drives")
    return canonical


def _validate_workdir(workdir: str) -> str:
    root = _canonical_local_directory(workdir)
    manifest_root = _manifest_root()
    if _is_within(root, manifest_root) or _is_within(manifest_root, root):
        raise SandboxUnavailableError("the LPAC workdir overlaps backend-private ownership state")
    root_info = os.lstat(root)
    if getattr(root_info, "st_file_attributes", 0) & 0x400:
        raise SandboxUnavailableError("the LPAC workdir root is a reparse point")
    entries = 0
    link_counts: dict[tuple[int, int], int] = {}
    link_totals: dict[tuple[int, int], int] = {}
    link_paths: dict[tuple[int, int], str] = {}

    def walk_error(exc: OSError) -> None:
        raise SandboxUnavailableError(
            f"the LPAC workdir cannot be fully inspected: {exc.filename or root}"
        ) from exc

    for base, dirs, names in os.walk(root, followlinks = False, onerror = walk_error):
        for name in [*dirs, *names]:
            entries += 1
            if entries > _SCAN_ENTRY_LIMIT:
                raise SandboxUnavailableError("the LPAC workdir exceeds its safety scan limit")
            path = os.path.join(base, name)
            info = os.lstat(path)
            if getattr(info, "st_file_attributes", 0) & 0x400:
                raise SandboxUnavailableError(f"the LPAC workdir contains a reparse point: {path}")
            if not (stat.S_ISREG(info.st_mode) or stat.S_ISDIR(info.st_mode)):
                raise SandboxUnavailableError(f"the LPAC workdir contains a special file: {path}")
            if stat.S_ISREG(info.st_mode) and info.st_nlink > 1:
                key = (info.st_dev, info.st_ino)
                link_counts[key] = link_counts.get(key, 0) + 1
                link_totals[key] = max(link_totals.get(key, 0), info.st_nlink)
                link_paths.setdefault(key, path)
    for key, count in link_counts.items():
        if count < link_totals[key]:
            raise SandboxUnavailableError(
                f"the LPAC workdir contains a hardlink crossing its boundary: {link_paths[key]}"
            )
    return root


def _is_within(path: str, root: str) -> bool:
    try:
        return os.path.commonpath(
            (os.path.realpath(path), os.path.realpath(root))
        ) == os.path.realpath(root)
    except ValueError:
        return False


def _traverse_ancestors(paths: tuple[str, ...]) -> tuple[str, ...]:
    selected: list[str] = []
    roots = {os.path.normcase(os.path.realpath(path)) for path in paths}
    user_profile = os.path.normcase(
        os.path.realpath(os.environ.get("USERPROFILE", os.path.expanduser("~")))
    )
    for path in paths:
        current = os.path.dirname(os.path.realpath(path))
        while current and os.path.dirname(current) != current:
            normalized = os.path.normcase(current)
            if normalized not in roots and normalized not in {
                os.path.normcase(item) for item in selected
            }:
                selected.append(current)
            if normalized == user_profile:
                break
            current = os.path.dirname(current)
    return tuple(selected)


def _environment_block(env: dict[str, str]) -> ctypes.Array[Any]:
    entries: list[str] = []
    for key, value in sorted(env.items(), key = lambda item: item[0].upper()):
        if not key or "=" in key or "\0" in key or "\0" in value:
            raise SandboxUnavailableError("the LPAC environment contains an invalid entry")
        entries.append(f"{key}={value}")
    return ctypes.create_unicode_buffer("\0".join(entries) + "\0\0")


def _initial_appcontainer_environment(
    env: dict[str, str], identity: _InvocationIdentity
) -> dict[str, str]:
    profile = Path(identity.profile_folder)
    package = profile.parent
    packages = package.parent
    if (
        not profile.is_absolute()
        or profile.name.lower() != "ac"
        or package.name != identity.moniker
        or packages.name.lower() != "packages"
    ):
        raise SandboxUnavailableError("LPAC returned an unsupported profile directory layout")
    # CreateProcessW constructs the package environment from the host LocalAppData
    # prefix. Supplying the already redirected path duplicates Packages/<id>/AC.
    # Only its input block uses this prefix; prepared.env describes the child.
    initial = {key: value for key, value in env.items() if key.upper() != "LOCALAPPDATA"}
    initial["LOCALAPPDATA"] = str(packages.parent)
    return initial


def _safe_environment(
    env: dict[str, str], workdir: str, identity: _InvocationIdentity, argv: tuple[str, ...]
) -> dict[str, str]:
    denied = {
        "APPDATA",
        "DOCKER_HOST",
        "HOMEDRIVE",
        "HOMEPATH",
        "SSH_AUTH_SOCK",
        "USERPROFILE",
    }
    safe = {key: value for key, value in env.items() if key.upper() not in denied}
    system_root = os.environ.get("SystemRoot", r"C:\Windows")
    runtime_bin = os.path.dirname(os.path.realpath(sys.executable))
    path_entries = [runtime_bin, os.path.join(system_root, "System32")]
    executable_dir = os.path.dirname(argv[0])
    if executable_dir not in path_entries:
        path_entries.insert(0, executable_dir)
    if os.path.basename(argv[0]).lower() in {"bash", "bash.exe"}:
        posix_bin = os.path.join(os.path.dirname(executable_dir), "usr", "bin")
        if os.path.isdir(posix_bin):
            path_entries.insert(1, posix_bin)
    safe.update(
        {
            "APPDATA": identity.private_temp,
            "HOME": workdir,
            "LOCALAPPDATA": identity.profile_folder,
            "PATH": os.pathsep.join(path_entries),
            "TEMP": identity.private_temp,
            "TMP": identity.private_temp,
            "USERPROFILE": workdir,
        }
    )
    return safe


_JOB_OBJECT_LIMIT_PROCESS_TIME = 0x00000002
_JOB_OBJECT_LIMIT_ACTIVE_PROCESS = 0x00000008
_JOB_OBJECT_LIMIT_PROCESS_MEMORY = 0x00000100
_JOB_OBJECT_LIMIT_JOB_MEMORY = 0x00000200
_JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000

_JOB_OBJECT_EXTENDED_LIMIT_INFORMATION = 9


def _job_object_with_limits(*, active_process_limit: int | None = None) -> _WindowsJob:
    """A kill-on-close Job Object carrying Studio's resource limits, with no process yet.

    Split from _create_job so the job can be attached at creation through
    PROC_THREAD_ATTRIBUTE_JOB_LIST, which leaves no window in which the child
    runs outside it.
    """
    from .native_compat import _WindowsJob

    api = _api()
    handle = api.kernel32.CreateJobObjectW(None, None)
    if not handle:
        raise _winerror("CreateJobObjectW")
    try:
        info = _JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
        info.BasicLimitInformation.LimitFlags = (
            _JOB_OBJECT_LIMIT_PROCESS_TIME
            | _JOB_OBJECT_LIMIT_ACTIVE_PROCESS
            | _JOB_OBJECT_LIMIT_PROCESS_MEMORY
            | _JOB_OBJECT_LIMIT_JOB_MEMORY
            | _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        )
        try:
            # The Python bootstrap profile supplies 1 from trusted backend policy.
            # Environment settings must not enlarge that profile's process limit.
            if active_process_limit is None:
                active_process_limit = max(
                    1, int(os.environ.get("UNSLOTH_STUDIO_SANDBOX_NPROC", "10000"))
                )
            if type(active_process_limit) is not int or not 1 <= active_process_limit <= 0xFFFFFFFF:
                raise ValueError("invalid active process limit")
            info.BasicLimitInformation.ActiveProcessLimit = active_process_limit
            memory = (
                max(1, int(os.environ.get("UNSLOTH_STUDIO_SANDBOX_AS_GB", "8")))
                * 1024
                * 1024
                * 1024
            )
            info.BasicLimitInformation.PerProcessUserTimeLimit = (
                max(1, int(os.environ.get("UNSLOTH_STUDIO_SANDBOX_CPU_S", "600"))) * 10_000_000
            )
        except ValueError as exc:
            raise SandboxUnavailableError("Windows sandbox resource limits are invalid") from exc
        info.ProcessMemoryLimit = memory
        info.JobMemoryLimit = memory
        if not api.kernel32.SetInformationJobObject(
            handle,
            _JOB_OBJECT_EXTENDED_LIMIT_INFORMATION,
            ctypes.byref(info),
            ctypes.sizeof(info),
        ):
            raise _winerror("SetInformationJobObject")
        return _WindowsJob(handle)
    except Exception:
        api.kernel32.CloseHandle(handle)
        raise
