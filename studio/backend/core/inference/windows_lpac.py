# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Source-only Windows Less-Privileged AppContainer launcher for Studio tools."""

from __future__ import annotations

import ctypes
from contextlib import ExitStack, contextmanager
from ctypes import wintypes
from dataclasses import dataclass
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
from typing import Any

from .os_sandbox import (
    PreparedSandboxLaunch,
    SandboxCapability,
    SandboxUnavailableError,
    ToolLaunchPlan,
)


_PROFILE_PREFIX = "unsloth.studio."
_PROFILE_ID = "windows-lpac-preview-v1"
_PROBE_TOKEN = "UNSLOTH_WINDOWS_LPAC_PROBE_OK"
_SCAN_ENTRY_LIMIT = 100_000
_RUNTIME_SCAN_ENTRY_LIMIT = 1_000_000

_ERROR_INSUFFICIENT_BUFFER = 122
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

_PROC_THREAD_ATTRIBUTE_HANDLE_LIST = 0x00020002
_PROC_THREAD_ATTRIBUTE_SECURITY_CAPABILITIES = 0x00020009
_PROC_THREAD_ATTRIBUTE_ALL_APPLICATION_PACKAGES_POLICY = 0x0002000F
_PROCESS_CREATION_ALL_APPLICATION_PACKAGES_OPT_OUT = 0x1
_EXTENDED_STARTUPINFO_PRESENT = 0x00080000
_CREATE_SUSPENDED = 0x00000004
_CREATE_UNICODE_ENVIRONMENT = 0x00000400
_CREATE_BREAKAWAY_FROM_JOB = 0x01000000
_STARTF_USESTDHANDLES = 0x00000100

_JOB_OBJECT_LIMIT_PROCESS_TIME = 0x00000002
_JOB_OBJECT_LIMIT_ACTIVE_PROCESS = 0x00000008
_JOB_OBJECT_LIMIT_PROCESS_MEMORY = 0x00000100
_JOB_OBJECT_LIMIT_JOB_MEMORY = 0x00000200
_JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
_JOB_OBJECT_EXTENDED_LIMIT_INFORMATION = 9
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


def _profile_folder(api: _WinApi, sid_string: str) -> str:
    value = wintypes.LPWSTR()
    result = api.userenv.GetAppContainerFolderPath(sid_string, ctypes.byref(value))
    if result != 0:
        raise _hresult_error("GetAppContainerFolderPath", result)
    try:
        return os.path.realpath(value.value)
    finally:
        api.ole32.CoTaskMemFree(value)


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
            return None
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


def _runtime_roots(workdir: str, argv: tuple[str, ...]) -> tuple[str, ...]:
    candidates = [sys.executable, os.path.realpath(sys.executable), sys.prefix, sys.base_prefix]
    candidates.append(os.path.join(os.path.dirname(__file__), "sandbox_site"))
    candidates.extend(path for path in sysconfig.get_paths().values() if path)
    if argv and os.path.isabs(argv[0]):
        executable_dir = os.path.dirname(argv[0])
        candidates.extend((argv[0], executable_dir))
        if os.path.basename(argv[0]).lower() in {"bash", "bash.exe"}:
            shell_root = os.path.dirname(executable_dir)
            candidates.append(os.path.join(shell_root, "usr", "bin"))
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


def _is_within(path: str, root: str) -> bool:
    try:
        return os.path.commonpath(
            (os.path.realpath(path), os.path.realpath(root))
        ) == os.path.realpath(root)
    except ValueError:
        return False


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
        try:
            os.unlink(self.manifest_path)
        except FileNotFoundError:
            pass
        except OSError as exc:
            raise OSError(f"manifest: {exc}") from exc
        _api().advapi32.FreeSid(self.sid)
        self.sid = ctypes.c_void_p()
        self.cleaned = True


def _write_manifest(identity: _InvocationIdentity) -> None:
    payload = {
        "version": 1,
        "moniker": identity.moniker,
        "sid": identity.sid_string,
        "profile_folder": identity.profile_folder,
        "private_temp": identity.private_temp,
        "granted_roots": list(identity.granted_roots),
        "traverse_roots": list(identity.traverse_roots),
        "owner_pid": identity.owner_pid,
        "owner_created": identity.owner_created,
    }
    temporary = identity.manifest_path + ".tmp"
    with open(temporary, "x", encoding = "utf-8") as stream:
        json.dump(payload, stream, sort_keys = True)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, identity.manifest_path)


def _create_identity(granted_roots: tuple[str, ...]) -> _InvocationIdentity:
    api = _api()
    moniker = _PROFILE_PREFIX + secrets.token_hex(16)
    sid = ctypes.c_void_p()
    result = api.userenv.CreateAppContainerProfile(
        moniker,
        "Unsloth Studio tool",
        "Transient zero-capability Studio LPAC",
        None,
        0,
        ctypes.byref(sid),
    )
    if result != 0 or not sid:
        raise _hresult_error("CreateAppContainerProfile", result)
    private_temp = ""
    manifest_path = ""
    try:
        sid_text = _sid_string(api, sid)
        profile_folder = _profile_folder(api, sid_text)
        # Windows redirects TEMP/TMP to this exact directory. The profile itself
        # has a new random identity for every invocation; it is never reused.
        private_temp = os.path.join(profile_folder, "Temp")
        os.makedirs(private_temp, mode = 0o700, exist_ok = True)
        _validated_private_temp(profile_folder, private_temp)
        manifest_path = os.path.join(_manifest_root(), moniker + ".json")
        permission_roots = (*granted_roots, private_temp)
        owner = _process_identity()
        if owner is None:
            raise SandboxUnavailableError("LPAC could not record its owning process identity")
        identity = _InvocationIdentity(
            moniker,
            sid,
            sid_text,
            profile_folder,
            private_temp,
            manifest_path,
            (*permission_roots, *_traverse_ancestors(permission_roots)),
            _traverse_ancestors(permission_roots),
            owner[0],
            owner[1],
        )
        _write_manifest(identity)
        return identity
    except Exception:
        if manifest_path:
            try:
                os.unlink(manifest_path)
            except FileNotFoundError:
                pass
        if private_temp:
            shutil.rmtree(private_temp, ignore_errors = True)
        api.userenv.DeleteAppContainerProfile(moniker)
        api.advapi32.FreeSid(sid)
        raise


class _WindowsJob:
    def __init__(self, handle: wintypes.HANDLE):
        self._handle = handle

    def terminate(self) -> bool:
        return bool(self._handle and _api().kernel32.TerminateJobObject(self._handle, 1))

    def close(self) -> None:
        handle, self._handle = self._handle, None
        if handle:
            _api().kernel32.CloseHandle(handle)


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
        self._unsloth_job = job

    def poll(self) -> int | None:
        if self.returncode is not None:
            return self.returncode
        if _api().kernel32.WaitForSingleObject(self._handle, 0) == _WAIT_TIMEOUT:
            return None
        return self._read_returncode()

    def _read_returncode(self) -> int:
        code = wintypes.DWORD()
        if not _api().kernel32.GetExitCodeProcess(self._handle, ctypes.byref(code)):
            raise _winerror("GetExitCodeProcess")
        if code.value == _STILL_ACTIVE:
            return self.returncode if self.returncode is not None else _STILL_ACTIVE
        self.returncode = ctypes.c_int32(code.value).value
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        milliseconds = (
            _INFINITE if timeout is None else max(0, min(int(timeout * 1000), 0xFFFFFFFE))
        )
        result = _api().kernel32.WaitForSingleObject(self._handle, milliseconds)
        if result == _WAIT_TIMEOUT:
            raise subprocess.TimeoutExpired(self.args, timeout)
        if result != _WAIT_OBJECT_0:
            raise _winerror("WaitForSingleObject")
        return self._read_returncode()

    def terminate(self) -> None:
        # The leader may have exited while other processes still own the pipe.
        self._unsloth_job.terminate()

    kill = terminate

    def close(self) -> None:
        # Kill pipe writers before waiting for a concurrent reader's stream lock.
        self._unsloth_job.close()
        if self.stdout is not None:
            try:
                self.stdout.close()
            except OSError:
                pass
            self.stdout = None
        for name in ("_thread_handle", "_handle"):
            handle = getattr(self, name, None)
            if handle:
                _api().kernel32.CloseHandle(handle)
                setattr(self, name, None)


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


def _create_job(
    process_handle: wintypes.HANDLE, *, active_process_limit: int | None = None
) -> _WindowsJob:
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
        if not api.kernel32.AssignProcessToJobObject(handle, process_handle):
            raise _winerror("AssignProcessToJobObject")
        return _WindowsJob(handle)
    except Exception:
        api.kernel32.CloseHandle(handle)
        raise


def _spawn_lpac(
    prepared: PreparedSandboxLaunch, popen_kwargs: dict[str, Any], identity: _InvocationIdentity
) -> WindowsLpacProcess:
    if (
        popen_kwargs.get("stdout") != subprocess.PIPE
        or popen_kwargs.get("stderr") != subprocess.STDOUT
        or popen_kwargs.get("stdin") != subprocess.DEVNULL
        or not popen_kwargs.get("close_fds", True)
    ):
        raise SandboxUnavailableError("LPAC accepts only Studio's closed-descriptor stdio plan")
    import msvcrt

    api = _api()
    read_fd, write_fd = os.pipe()
    stdin_fd = os.open(os.devnull, os.O_RDONLY)
    process_info = _PROCESS_INFORMATION()
    attribute_buffer: ctypes.Array[Any] | None = None
    attribute_list: ctypes.c_void_p | None = None
    attributes_initialized = False
    job: _WindowsJob | None = None
    stdout = None
    try:
        os.set_inheritable(read_fd, False)
        os.set_inheritable(write_fd, True)
        os.set_inheritable(stdin_fd, True)
        child_stdin = wintypes.HANDLE(msvcrt.get_osfhandle(stdin_fd))
        child_stdout = wintypes.HANDLE(msvcrt.get_osfhandle(write_fd))
        handles = (wintypes.HANDLE * 2)(child_stdin, child_stdout)

        size = ctypes.c_size_t()
        api.kernel32.InitializeProcThreadAttributeList(None, 3, 0, ctypes.byref(size))
        if ctypes.get_last_error() != _ERROR_INSUFFICIENT_BUFFER or not size.value:
            raise _winerror("InitializeProcThreadAttributeList(size)")
        attribute_buffer = ctypes.create_string_buffer(size.value)
        attribute_list = ctypes.cast(attribute_buffer, ctypes.c_void_p)
        if not api.kernel32.InitializeProcThreadAttributeList(
            attribute_list, 3, 0, ctypes.byref(size)
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
        command_line = ctypes.create_unicode_buffer(subprocess.list2cmdline(prepared.argv))
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
            command_line,
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
        os.close(write_fd)
        write_fd = -1
        os.close(stdin_fd)
        stdin_fd = -1
        job = _create_job(process_info.hProcess)
        if api.kernel32.ResumeThread(process_info.hThread) == 0xFFFFFFFF:
            raise _winerror("ResumeThread")
        stdout = os.fdopen(
            read_fd,
            "r",
            encoding = popen_kwargs.get("encoding", "utf-8"),
            errors = popen_kwargs.get("errors", "replace"),
        )
        read_fd = -1
        process = WindowsLpacProcess(
            prepared.argv,
            process_info.hProcess,
            process_info.hThread,
            int(process_info.dwProcessId),
            stdout,
            job,
        )
        prepared.cleanup_callbacks.append(process.close)
        return process
    except Exception:
        if process_info.hProcess:
            api.kernel32.TerminateProcess(process_info.hProcess, 1)
        if job is not None:
            job.close()
        for handle in (process_info.hThread, process_info.hProcess):
            if handle:
                api.kernel32.CloseHandle(handle)
        if stdout is not None:
            stdout.close()
        raise
    finally:
        if attributes_initialized and attribute_list is not None:
            api.kernel32.DeleteProcThreadAttributeList(attribute_list)
        for fd in (read_fd, write_fd, stdin_fd):
            if fd >= 0:
                os.close(fd)


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


@contextmanager
def _probe_network_endpoints():
    """Prove host endpoints work, then reject any traffic from the LPAC probe."""
    with ExitStack() as stack:
        servers = []
        endpoints = []
        control = secrets.token_bytes(32)
        for family, host in ((socket.AF_INET, "127.0.0.1"), (socket.AF_INET6, "::1")):
            for kind in (socket.SOCK_STREAM, socket.SOCK_DGRAM):
                server = stack.enter_context(socket.socket(family, kind))
                server.settimeout(1)
                server.bind((host, 0))
                address = server.getsockname()
                if kind == socket.SOCK_STREAM:
                    server.listen(4)
                with socket.socket(family, kind) as client:
                    client.settimeout(1)
                    if kind == socket.SOCK_STREAM:
                        client.connect(address)
                        with server.accept()[0] as accepted:
                            client.sendall(control)
                            # TCP may split the control message into multiple reads.
                            accepted.settimeout(1)
                            received = bytearray()
                            while len(received) < len(control):
                                part = accepted.recv(len(control) - len(received))
                                if not part:
                                    break
                                received.extend(part)
                            if received != control:
                                raise SandboxUnavailableError("LPAC TCP probe control failed")
                    else:
                        client.sendto(control, address)
                        if server.recvfrom(128)[0] != control:
                            raise SandboxUnavailableError("LPAC UDP probe control failed")
                servers.append((server, kind))
                endpoints.append((int(family), int(kind), address))
        yield endpoints
        for server, kind in servers:
            server.settimeout(0.1)
            try:
                if kind == socket.SOCK_STREAM:
                    connection, _ = server.accept()
                    connection.close()
                else:
                    server.recvfrom(128)
            except socket.timeout:
                continue
            raise SandboxUnavailableError("the LPAC live probe reached a host network endpoint")


def _probe_network_payload(endpoints: list[tuple]) -> str:
    return f"""import socket
for family, kind, address in {endpoints!r}:
    sock = None
    try:
        sock = socket.socket(family, kind)
        sock.settimeout(1)
        if kind == socket.SOCK_STREAM:
            sock.connect(address)
        else:
            sock.sendto(b'UNSLOTH_LPAC_NETWORK_PROBE', address)
    except OSError as exc:
        # Refusal, timeout, and an absent address family are not enforcement.
        assert exc.winerror == 10013, ('unexpected network error', repr(exc))
    else:
        raise AssertionError('LPAC network operation was not denied')
    finally:
        if sock is not None:
            sock.close()
"""


def _probe_payload(workdir: str, external: str, expected_sid: str, endpoints: list[tuple]) -> str:
    return f"""import ctypes, os, socket, sys
from ctypes import wintypes
k = ctypes.WinDLL('kernel32', use_last_error=True)
a = ctypes.WinDLL('advapi32', use_last_error=True)
token = wintypes.HANDLE()
assert a.OpenProcessToken(k.GetCurrentProcess(), 0x0008, ctypes.byref(token))
def token_dword(kind):
    value = wintypes.DWORD(); size = wintypes.DWORD(ctypes.sizeof(value))
    assert a.GetTokenInformation(token, kind, ctypes.byref(value), size, ctypes.byref(size))
    return value.value
assert token_dword(29) == 1
assert token_dword(46) == 1
needed = wintypes.DWORD()
a.GetTokenInformation(token, 25, None, 0, ctypes.byref(needed))
integrity = ctypes.create_string_buffer(needed.value)
assert a.GetTokenInformation(token, 25, integrity, needed, ctypes.byref(needed))
integrity_sid = ctypes.cast(integrity, ctypes.POINTER(ctypes.c_void_p)).contents
a.GetSidSubAuthorityCount.argtypes = [ctypes.c_void_p]
a.GetSidSubAuthorityCount.restype = ctypes.POINTER(ctypes.c_ubyte)
a.GetSidSubAuthority.argtypes = [ctypes.c_void_p, wintypes.DWORD]
a.GetSidSubAuthority.restype = ctypes.POINTER(wintypes.DWORD)
count = a.GetSidSubAuthorityCount(integrity_sid).contents.value
integrity_rid = a.GetSidSubAuthority(integrity_sid, count - 1).contents.value
assert integrity_rid <= 0x1000, hex(integrity_rid)
needed = wintypes.DWORD()
a.GetTokenInformation(token, 30, None, 0, ctypes.byref(needed))
buf = ctypes.create_string_buffer(needed.value)
assert a.GetTokenInformation(token, 30, buf, needed, ctypes.byref(needed))
assert ctypes.cast(buf, ctypes.POINTER(wintypes.DWORD)).contents.value == 0
needed = wintypes.DWORD()
a.GetTokenInformation(token, 31, None, 0, ctypes.byref(needed))
buf2 = ctypes.create_string_buffer(needed.value)
assert a.GetTokenInformation(token, 31, buf2, needed, ctypes.byref(needed))
sid = ctypes.cast(buf2, ctypes.POINTER(ctypes.c_void_p)).contents
sid_text = wintypes.LPWSTR()
assert a.ConvertSidToStringSidW(sid, ctypes.byref(sid_text))
try:
    assert sid_text.value == {expected_sid!r}
finally:
    k.LocalFree(sid_text)
wd = {workdir!r}
assert open(os.path.join(wd, 'probe-read'), encoding='utf-8').read() == 'readable'
open(os.path.join(wd, 'probe-write'), 'w', encoding='utf-8').write('ok')
for path in ({external!r}, sys.executable):
    mode = 'r' if path == {external!r} else 'ab'
    try:
        with open(path, mode) as stream:
            if mode == 'r': stream.read()
        raise AssertionError('LPAC escaped file policy: ' + path)
    except OSError:
        pass
{_probe_network_payload(endpoints)}
assert os.path.commonpath((os.environ['TEMP'], os.environ['LOCALAPPDATA'])) == os.environ['LOCALAPPDATA']
print({_PROBE_TOKEN!r})
"""


class WindowsLpacBackend:
    identity = "windows-lpac"
    profile_id = _PROFILE_ID

    def fingerprint_data(self) -> dict[str, Any]:
        runtime = []
        for path in (sys.executable, sys.prefix, sys.base_prefix):
            try:
                info = os.stat(path)
                runtime.append((os.path.realpath(path), info.st_size, info.st_mtime_ns))
            except OSError:
                runtime.append((os.path.realpath(path), None, None))
        symbols: dict[str, bool] = {}
        try:
            api = _api()
            symbols = {
                name: hasattr(library, name)
                for library, name in (
                    (api.userenv, "CreateAppContainerProfile"),
                    (api.userenv, "DeleteAppContainerProfile"),
                    (api.kernel32, "UpdateProcThreadAttribute"),
                    (api.kernel32, "CreateProcessW"),
                )
            }
        except Exception:
            symbols = {"appcontainer_apis": False}
        return {
            "windows_build": platform.version(),
            "architecture": platform.machine().lower(),
            "runtime": runtime,
            "symbols": symbols,
        }

    def probe(self) -> SandboxCapability:
        if os.name != "nt":
            return SandboxCapability(self.identity, False, "LPAC requires Windows", available = False)
        try:
            _api()
            self.reconcile_stale_manifests()
            with (
                tempfile.TemporaryDirectory(prefix = "unsloth-lpac-probe-") as base,
                _probe_network_endpoints() as endpoints,
            ):
                workdir = os.path.join(base, "work")
                os.mkdir(workdir)
                Path(workdir, "probe-read").write_text("readable", encoding = "utf-8")
                external = os.path.join(base, "host-secret")
                Path(external).write_text("secret", encoding = "utf-8")
                prepared = self.prepare(
                    ToolLaunchPlan(
                        argv = (sys.executable, "-I", "-S", "-c", ""),
                        workdir = workdir,
                        env = {"PYTHONIOENCODING": "utf-8"},
                    )
                )
                try:
                    identity = getattr(prepared.spawn_callback, "_lpac_identity", None)
                    if identity is None:
                        raise SandboxUnavailableError("LPAC launch identity was lost")
                    prepared.argv = (
                        sys.executable,
                        "-I",
                        "-S",
                        "-c",
                        _probe_payload(workdir, external, identity.sid_string, endpoints),
                    )
                    process = prepared.spawn_callback(
                        prepared,
                        {
                            "stdout": subprocess.PIPE,
                            "stderr": subprocess.STDOUT,
                            "stdin": subprocess.DEVNULL,
                            "text": True,
                            "encoding": "utf-8",
                            "errors": "replace",
                            "cwd": prepared.workdir,
                            "env": prepared.env,
                            "close_fds": True,
                            "creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0),
                        },
                    )
                    process.wait(timeout = 8)
                    output = process.stdout.read()
                    if process.returncode != 0 or _PROBE_TOKEN not in output:
                        if process.returncode == -1073741790:
                            detail = (
                                "the selected interpreter could not initialize under zero-capability "
                                "LPAC (STATUS_ACCESS_DENIED); this runtime is incompatible with the "
                                "current profile, and capabilities were not widened"
                            )
                        else:
                            detail = output[-400:]
                        return SandboxCapability(
                            self.identity,
                            False,
                            f"the LPAC live probe failed ({process.returncode}): {detail}",
                            available = False,
                        )
                finally:
                    prepared.cleanup()
                    if prepared.cleanup_diagnostics:
                        raise SandboxUnavailableError("LPAC probe cleanup failed")
        except Exception as exc:  # noqa: BLE001 - capability failure blocks Required mode
            return SandboxCapability(
                self.identity,
                False,
                f"the LPAC live probe could not run: {exc}",
                available = False,
            )
        return SandboxCapability(
            self.identity,
            True,
            "zero-capability LPAC live enforcement probe passed",
            available = True,
            protection_state = "preview",
            profile_id = self.profile_id,
        )

    def prepare(self, spec: ToolLaunchPlan) -> PreparedSandboxLaunch:
        workdir = _validate_workdir(spec.workdir)
        argv = _canonical_inner_argv(spec.argv, spec.env)
        runtime_roots = _runtime_roots(workdir, argv)
        acl_runtime_roots = tuple(path for path in runtime_roots if _needs_explicit_acl(path))
        _validate_runtime_trees(acl_runtime_roots)
        identity = _create_identity((*acl_runtime_roots, workdir))
        try:
            for root in acl_runtime_roots:
                _grant_read_execute(root, identity.sid)
            _grant_modify(workdir, identity.sid)
            _grant_modify(identity.private_temp, identity.sid)
            for root in identity.traverse_roots:
                _grant_traverse(root, identity.sid)

            def spawn(prepared: PreparedSandboxLaunch, kwargs: dict[str, Any]) -> object:
                return _spawn_lpac(prepared, kwargs, identity)

            setattr(spawn, "_lpac_identity", identity)
            return PreparedSandboxLaunch(
                argv = argv,
                workdir = workdir,
                env = _safe_environment(spec.env, workdir, identity, argv),
                preexec_fn = None,
                backend = self.identity,
                timeout_seconds = spec.timeout_seconds,
                close_fds = spec.close_fds,
                terminate_descendants = spec.terminate_descendants,
                spawn_callback = spawn,
                cleanup_callbacks = [identity.cleanup],
            )
        except BaseException:
            identity.cleanup()
            raise

    def reconcile_stale_manifests(self) -> None:
        root = _manifest_root()
        for manifest in Path(root).glob(_PROFILE_PREFIX + "*.json"):
            try:
                payload = json.loads(manifest.read_text(encoding = "utf-8"))
                moniker = payload.get("moniker")
                sid_text = payload.get("sid")
                roots = payload.get("granted_roots")
                traverse_roots = payload.get("traverse_roots", [])
                private_temp = payload.get("private_temp")
                profile_folder = payload.get("profile_folder")
                owner_pid = payload.get("owner_pid")
                owner_created = payload.get("owner_created")
                if (
                    payload.get("version") != 1
                    or not isinstance(moniker, str)
                    or not moniker.startswith(_PROFILE_PREFIX)
                    or manifest.name != moniker + ".json"
                    or not isinstance(sid_text, str)
                    or not sid_text.startswith("S-1-15-2-")
                    or not isinstance(roots, list)
                    or not all(isinstance(path, str) and os.path.isabs(path) for path in roots)
                    or not isinstance(traverse_roots, list)
                    or not all(
                        isinstance(path, str) and os.path.isabs(path) for path in traverse_roots
                    )
                    or not isinstance(private_temp, str)
                    or not os.path.isabs(private_temp)
                    or not isinstance(profile_folder, str)
                    or not os.path.isabs(profile_folder)
                    or not isinstance(owner_pid, int)
                    or not isinstance(owner_created, int)
                ):
                    continue
                if _process_identity(owner_pid) == (owner_pid, owner_created):
                    continue
                sid = ctypes.c_void_p()
                derive = _api().userenv.DeriveAppContainerSidFromAppContainerName
                derive.argtypes = [wintypes.LPCWSTR, ctypes.POINTER(ctypes.c_void_p)]
                derive.restype = ctypes.c_long
                result = derive(moniker, ctypes.byref(sid))
                if result != 0 or not sid:
                    continue
                if _sid_string(_api(), sid) != sid_text:
                    _api().advapi32.FreeSid(sid)
                    continue
                try:
                    derived_profile = _profile_folder(_api(), sid_text)
                    if os.path.normcase(os.path.realpath(profile_folder)) != os.path.normcase(
                        derived_profile
                    ):
                        raise SandboxUnavailableError(
                            "the LPAC manifest profile folder does not match its SID"
                        )
                    _validated_private_temp(derived_profile, private_temp)
                except BaseException:
                    _api().advapi32.FreeSid(sid)
                    raise
                identity = _InvocationIdentity(
                    moniker,
                    sid,
                    sid_text,
                    derived_profile,
                    private_temp,
                    str(manifest),
                    tuple(roots),
                    tuple(traverse_roots),
                    owner_pid,
                    owner_created,
                )
                identity.cleanup()
            except Exception:
                # A stale record is retained for the next startup; never delete
                # evidence or reuse its identity after partial reconciliation.
                continue


__all__ = ["WindowsLpacBackend", "WindowsLpacProcess"]
