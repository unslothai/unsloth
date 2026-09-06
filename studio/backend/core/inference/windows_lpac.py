# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Source-only Windows Less-Privileged AppContainer launcher for Studio tools.

Identity model
--------------
One AppContainer profile is used per (user, Studio installation), named
``unsloth.studio.sandbox.<16 hex>`` where the digest covers the interpreter
path, the account and the profile root. ``DeriveAppContainerSidFromAppContainerName``
maps that name to the same package SID every time, so the SID is stable across
launches and across Studio restarts even if the profile folder is deleted and
recreated. The profile is created on the first launch of the process
(``ERROR_ALREADY_EXISTS`` means reuse) and is kept for the lifetime of the
process; it is not created and deleted per launch.

The reason is cost, measured on hosted windows-2022 and windows-latest runners
with CPython 3.12 in ``C:\\hostedtoolcache``: granting a fresh package SID
read+execute on the interpreter tree costs about 14 s of
``SetNamedSecurityInfoW`` propagation over thousands of files, revoking it and
deleting the profile costs about another 11 s, and validating the tree costs
about 2 s. That was paid on every Python or Terminal call. With a stable SID
the interpreter tree is granted once per installation, recorded in a persistent
manifest, and never revoked at launch cleanup, so only the first launch after
an install or an interpreter upgrade pays it. Later launches pay one DACL read
per runtime root plus the per-launch workdir work.

What a stable SID changes
-------------------------
* The AppContainer named-object namespace
  (``\\Sessions\\<n>\\AppContainerNamedObjects\\<SID>``) is now shared by every
  launch of the same installation, so two concurrent tool calls can see each
  other's named objects, and one can create a name the other is about to open,
  instead of each being in a namespace of its own.
* Both launches also share the container profile directory, and with it the
  container temp ``<profile>\\Temp``. Windows builds the package environment for
  an AppContainer process and points ``TEMP``/``TMP`` at that directory, so a
  per-launch subdirectory of it could be planned and granted but never used: the
  child would write to ``<profile>\\Temp`` regardless. The temp is therefore the
  container's, shared by concurrent launches of this installation, and emptied
  when the last of them finishes rather than deleted per launch.
* While two launches run at once, each one's workdir carries the shared SID, so
  a tool call in one Studio chat can reach another chat's workdir for as long as
  that other call is running. The grant is released when the last launch holding
  it finishes.
* A file a launch leaves in its workdir stays readable by the next launch in the
  same workdir. That was already true: the workdir is per chat session, not per
  launch, and it is the directory the user asked the tool to work in.
These are disclosed as ``concurrent_launches_share_the_container``.

Everything in that list is shared with the launches of *other* Studio processes
of the same installation, which this one sees only through their write-ahead
manifests. Reading a directory of manifests is a check and not a lock, so every
path that grants or revokes for an installation runs inside a ledger keyed by
its moniker (``_installation_ledger``): scan-and-revoke, prepare's
create-and-grant, reconciliation and the persistent removal never interleave
between processes. The ledger is a lock file opened with no sharing beside the
manifests, which is the scope the state actually has (per account, every
session), behind a session mutex that keeps same-session contenders off it.

A ledger that cannot be taken fails **closed** for anything destructive.
Revoking an ACE, emptying the container temp and removing a persistent profile
are skipped and left for the next reconciliation rather than performed
unsynchronised, because an unsynchronised revoke is the exact damage the ledger
exists to prevent. Granting is not destructive and proceeds with a logged
diagnostic, so a busy ledger never fails a tool call.

What it does not change
-----------------------
Everything outside the granted roots and the workdir stays denied, the network
stays denied, other processes stay unreachable, the container still carries zero
capabilities, and the token is still a Less-Privileged AppContainer (or the
plain AppContainer the live probe fell back to). The per-launch state that
enforced none of the above and only paid for it - the profile creation and
deletion - is the only thing that was removed.
"""

from __future__ import annotations

import ctypes
from contextlib import ExitStack, contextmanager
from ctypes import wintypes
from dataclasses import dataclass, field
import hashlib
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
from typing import Any, Callable

from loggers import get_logger

from .os_sandbox import (
    PreparedSandboxLaunch,
    SandboxCapability,
    SandboxUnavailableError,
    ToolLaunchPlan,
)


logger = get_logger(__name__)

_PROFILE_PREFIX = "unsloth.studio."
# One profile per (user, installation); one manifest per launch of it.
_INSTALL_PREFIX = _PROFILE_PREFIX + "sandbox."
_LAUNCH_PREFIX = _PROFILE_PREFIX + "launch."
# Everything CreateAppContainerProfile accepts in a profile name, which is also
# everything a moniker read out of a manifest is allowed to contain: the ledger
# turns one into a file name beside the manifests.
_MONIKER_CHARACTERS = frozenset(
    "-_. abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
)
_MANIFEST_KIND_LAUNCH = "lpac-launch"
_MANIFEST_KIND_PERSISTENT = "lpac-persistent"
# Manifests written before the stable identity: one random profile per launch,
# named after its own moniker and owning the profile it must delete.
_MANIFEST_KIND_SINGLE_USE = "lpac-single-use"
_ORPHAN_TEMPORARY_MANIFEST_SECONDS = 300.0
_PROFILE_ID = "windows-lpac-preview-v1"
_APPCONTAINER_PROFILE_ID = "windows-appcontainer-preview-v1"
_PROFILE_LPAC = "lpac"
_PROFILE_APPCONTAINER = "appcontainer"
_PROFILE_BY_ID = {_PROFILE_ID: _PROFILE_LPAC, _APPCONTAINER_PROFILE_ID: _PROFILE_APPCONTAINER}
_PROBE_TOKEN = "UNSLOTH_WINDOWS_LPAC_PROBE_OK"
_ALL_APPLICATION_PACKAGES_SID = "S-1-15-2-1"
_ALL_RESTRICTED_APPLICATION_PACKAGES_SID = "S-1-15-2-2"
_LIMITATION_AMBIENT_READ = "all_application_packages_ambient_read"
_LIMITATION_IPV6 = "ipv6_unavailable_on_host"
# An AppContainer token gets no access to \Device\Null or to named pipes it did
# not create with an AppContainer-aware descriptor, so inside the sandbox
# open(os.devnull) and multiprocessing.Pipe() raise PermissionError. Code that
# needs them (multiprocessing, torch through dill) runs in Limited or Full mode.
_LIMITATION_NULL_DEVICE_PIPES = "null_device_and_named_pipes_denied"
# One AppContainer identity per installation, so concurrent launches of that
# installation share the container's named-object namespace, its profile
# directory (and therefore each other's private temp), and each other's workdir
# grant for as long as both are running. See the module docstring.
_LIMITATION_SHARED_CONTAINER = "concurrent_launches_share_the_container"
_STATUS_ACCESS_DENIED = -1073741790
_STATUS_DLL_NOT_FOUND = -1073741515
_ERROR_ACCESS_DENIED = 5
_SCAN_ENTRY_LIMIT = 100_000
_RUNTIME_SCAN_ENTRY_LIMIT = 1_000_000

_ERROR_INSUFFICIENT_BUFFER = 122
_HRESULT_ALREADY_EXISTS = 0x800700B7  # HRESULT_FROM_WIN32(ERROR_ALREADY_EXISTS)
_ERROR_NOT_SUPPORTED = 50
_ERROR_INVALID_PARAMETER = 87
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
_GENERIC_ALL = 0x10000000
_DELETE = 0x00010000
_FILE_TRAVERSE = 0x00000020
_FILE_GENERIC_READ = 0x00120089
_FILE_GENERIC_EXECUTE = 0x001200A0
_ACCESS_ALLOWED_ACE_TYPE = 0
_ACCESS_DENIED_ACE_TYPE = 1
_INHERIT_ONLY_ACE = 0x08

_PROC_THREAD_ATTRIBUTE_HANDLE_LIST = 0x00020002
_PROC_THREAD_ATTRIBUTE_JOB_LIST = 0x0002000D
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
_JOB_OBJECT_BASIC_ACCOUNTING_INFORMATION = 1
_WAIT_OBJECT_0 = 0
# The previous owner of a mutex died without releasing it. Ownership is granted
# to this caller all the same, which is what a crashed Studio looks like here.
_WAIT_ABANDONED = 0x00000080
_WAIT_TIMEOUT = 258
_INFINITE = 0xFFFFFFFF
_STILL_ACTIVE = 259
_PROCESS_QUERY_LIMITED_INFORMATION = 0x1000

# TerminateJobObject and KILL_ON_JOB_CLOSE only start the kill, so the container
# temp and the workdir ACE are released only once the job has actually drained.
_JOB_DRAIN_SECONDS = 5.0
_JOB_DRAIN_FIRST_POLL_SECONDS = 0.005
_JOB_DRAIN_MAX_POLL_SECONDS = 0.1

# The cross-process ledger, one per installation moniker.
#
# What the ledger protects is per account and machine-wide, not per session: the
# manifests live under LOCALAPPDATA, the AppContainer profile is registered for
# the account, and the ACEs name a package SID derived from a moniker that
# already covers the interpreter, the account and LOCALAPPDATA. A console login
# and an RDP login of one account share every byte of that state, so a session
# namespace is the wrong scope for the lock on its own.
#
# The lock is therefore a file next to the manifests, opened with no sharing.
# CreateFileW documents dwShareMode 0 as "cannot be shared and cannot be opened
# again until the handle to the file or device is closed", and that the sharing
# options "remain in effect until that handle is closed, regardless of process
# context", so one handle excludes every process in every session. It needs no
# privilege beyond the write access to LOCALAPPDATA the manifests already
# require; and Windows closes the handle when the owning process dies, which is
# the crashed Studio WAIT_ABANDONED was there for. No unprivileged account can
# squat it either, because no unprivileged account can reach this account's
# LOCALAPPDATA. An administrator can, but an administrator can already rewrite
# the DACLs this whole ledger is about, so no defence is lost there.
#
# Global\ was the other candidate and was rejected on the squatting argument
# alone, not the privilege one: SeCreateGlobalPrivilege gates only file-mapping
# and symbolic-link objects, so a Global\ mutex is creatable by an ordinary
# user. But the global namespace is flat across every account on the machine, so
# any unprivileged local account can create this exact name first and then hold
# it forever - and against a lock that now fails closed, that is a permanent
# denial of sandbox cleanup handed to an unprivileged attacker.
#
# The Local\ mutex is kept in front of the file as a same-session fast path. It
# blocks in the kernel instead of polling, so at most one process per session
# ever polls the lock file. It is never sufficient on its own, and it is always
# taken first, so no two holders can order the two primitives differently.
_LEDGER_MUTEX_PREFIX = "Local\\unsloth.studio.ledger."
_LEDGER_LOCK_SUFFIX = ".ledger-lock"
# One budget for both primitives, so a slow mutex cannot add to a slow file.
_LEDGER_WAIT_SECONDS = 30.0
_LEDGER_LOCK_FIRST_POLL_SECONDS = 0.005
_LEDGER_LOCK_MAX_POLL_SECONDS = 0.25

_FILE_SHARE_NONE = 0
_OPEN_ALWAYS = 4
_FILE_ATTRIBUTE_NORMAL = 0x00000080
_INVALID_HANDLE_VALUE = ctypes.c_void_p(-1).value
_ERROR_SHARING_VIOLATION = 32
_ERROR_LOCK_VIOLATION = 33


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


class _JOBOBJECT_BASIC_ACCOUNTING_INFORMATION(ctypes.Structure):
    """Only ``ActiveProcesses`` is read, to tell a drained job from a draining one."""

    _fields_ = [
        ("TotalUserTime", ctypes.c_int64),
        ("TotalKernelTime", ctypes.c_int64),
        ("ThisPeriodTotalUserTime", ctypes.c_int64),
        ("ThisPeriodTotalKernelTime", ctypes.c_int64),
        ("TotalPageFaultCount", wintypes.DWORD),
        ("TotalProcesses", wintypes.DWORD),
        ("ActiveProcesses", wintypes.DWORD),
        ("TotalTerminatedProcesses", wintypes.DWORD),
    ]


@dataclass(frozen = True)
class _WinApi:
    kernel32: Any
    advapi32: Any
    userenv: Any
    ole32: Any
    # Window station and desktop security, for the restricted-token launcher.
    user32: Any = None


_API: _WinApi | None = None


def _winerror(prefix: str, code: int | None = None) -> OSError:
    number = ctypes.get_last_error() if code is None else int(code)
    return OSError(number, f"{prefix}: {ctypes.FormatError(number).strip()}")


def _hresult_error(prefix: str, value: int) -> OSError:
    unsigned = ctypes.c_uint32(value).value
    return OSError(unsigned, f"{prefix} failed with HRESULT 0x{unsigned:08x}")


def _is_windows() -> bool:
    return os.name == "nt"


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
    user32 = ctypes.WinDLL("user32", use_last_error = True)

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
    userenv.DeriveAppContainerSidFromAppContainerName.argtypes = [
        wintypes.LPCWSTR,
        ctypes.POINTER(ctypes.c_void_p),
    ]
    userenv.DeriveAppContainerSidFromAppContainerName.restype = ctypes.c_long
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
    advapi32.EqualSid.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    advapi32.EqualSid.restype = wintypes.BOOL
    advapi32.ConvertStringSidToSidW.argtypes = [wintypes.LPCWSTR, ctypes.POINTER(ctypes.c_void_p)]
    advapi32.ConvertStringSidToSidW.restype = wintypes.BOOL
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
    advapi32.GetSecurityDescriptorDacl.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(wintypes.BOOL),
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(wintypes.BOOL),
    ]
    advapi32.GetSecurityDescriptorDacl.restype = wintypes.BOOL
    # Window station and desktop DACLs. CreateProcessAsUser documents that "the
    # DACLs for the window station and desktop must grant access to the user or
    # the logon session represented by the hToken parameter"; the restricted
    # token's per-launch SID is in neither until the launcher adds it.
    #
    # Declared here only because this module owns _api(). Nothing in this file
    # calls GetUserObjectSecurity or SetUserObjectSecurity: an AppContainer is
    # launched with CreateProcessW and a security-capabilities attribute, never
    # with CreateProcessAsUserW, so the package SID is never added to a window
    # station or a desktop DACL and no window-station right is widened for it.
    # The narrowing question a reviewer raised belongs to the only caller,
    # windows_restricted_token.
    user32.GetProcessWindowStation.argtypes = []
    user32.GetProcessWindowStation.restype = wintypes.HANDLE
    user32.GetThreadDesktop.argtypes = [wintypes.DWORD]
    user32.GetThreadDesktop.restype = wintypes.HANDLE
    user32.GetUserObjectInformationW.argtypes = [
        wintypes.HANDLE,
        ctypes.c_int,
        ctypes.c_void_p,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD),
    ]
    user32.GetUserObjectInformationW.restype = wintypes.BOOL
    user32.GetUserObjectSecurity.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(wintypes.DWORD),
        ctypes.c_void_p,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD),
    ]
    user32.GetUserObjectSecurity.restype = wintypes.BOOL
    user32.SetUserObjectSecurity.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(wintypes.DWORD),
        ctypes.c_void_p,
    ]
    user32.SetUserObjectSecurity.restype = wintypes.BOOL
    # Token APIs used by the write-restricted Limited launcher (windows_restricted_token).
    advapi32.OpenProcessToken.argtypes = [
        wintypes.HANDLE,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.HANDLE),
    ]
    advapi32.OpenProcessToken.restype = wintypes.BOOL
    advapi32.CreateRestrictedToken.argtypes = [
        wintypes.HANDLE,
        wintypes.DWORD,
        wintypes.DWORD,
        ctypes.c_void_p,
        wintypes.DWORD,
        ctypes.c_void_p,
        wintypes.DWORD,
        ctypes.c_void_p,
        ctypes.POINTER(wintypes.HANDLE),
    ]
    advapi32.CreateRestrictedToken.restype = wintypes.BOOL
    advapi32.GetTokenInformation.argtypes = [
        wintypes.HANDLE,
        ctypes.c_int,
        ctypes.c_void_p,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD),
    ]
    advapi32.GetTokenInformation.restype = wintypes.BOOL
    advapi32.SetTokenInformation.argtypes = [
        wintypes.HANDLE,
        ctypes.c_int,
        ctypes.c_void_p,
        wintypes.DWORD,
    ]
    advapi32.SetTokenInformation.restype = wintypes.BOOL
    advapi32.IsTokenRestricted.argtypes = [wintypes.HANDLE]
    advapi32.IsTokenRestricted.restype = wintypes.BOOL
    advapi32.CreateProcessAsUserW.argtypes = [
        wintypes.HANDLE,
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
    advapi32.CreateProcessAsUserW.restype = wintypes.BOOL

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
    kernel32.IsProcessInJob.argtypes = [
        wintypes.HANDLE,
        wintypes.HANDLE,
        ctypes.POINTER(wintypes.BOOL),
    ]
    kernel32.IsProcessInJob.restype = wintypes.BOOL
    kernel32.ResumeThread.argtypes = [wintypes.HANDLE]
    kernel32.ResumeThread.restype = wintypes.DWORD
    kernel32.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel32.WaitForSingleObject.restype = wintypes.DWORD
    # The cross-process ledger; the manifests alone are a check, not a lock. The
    # mutex orders the Studio processes of one session, the exclusively opened
    # lock file orders every session of the account. See _LEDGER_MUTEX_PREFIX.
    kernel32.CreateMutexW.argtypes = [ctypes.c_void_p, wintypes.BOOL, wintypes.LPCWSTR]
    kernel32.CreateMutexW.restype = wintypes.HANDLE
    kernel32.ReleaseMutex.argtypes = [wintypes.HANDLE]
    kernel32.ReleaseMutex.restype = wintypes.BOOL
    kernel32.CreateFileW.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        ctypes.c_void_p,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    ]
    kernel32.CreateFileW.restype = wintypes.HANDLE
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
    kernel32.GetCurrentThreadId.argtypes = []
    kernel32.GetCurrentThreadId.restype = wintypes.DWORD
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

    _API = _WinApi(kernel32, advapi32, userenv, ole32, user32)
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


def _derive_container_sid(moniker: str) -> ctypes.c_void_p:
    """The package SID of ``moniker``, whether or not its profile exists.

    The SID is a hash of the name, so it survives deleting and recreating the
    profile. Freed with ``FreeSid``, as the profile SID is.
    """
    api = _api()
    sid = ctypes.c_void_p()
    result = api.userenv.DeriveAppContainerSidFromAppContainerName(moniker, ctypes.byref(sid))
    if result != 0 or not sid:
        raise _hresult_error("DeriveAppContainerSidFromAppContainerName", result)
    return sid


def _is_container_sid_text(text: Any) -> bool:
    """Whether ``text`` is the SID of a parent AppContainer profile.

    ``S-1-15-2`` with seven further sub-authorities (the documented count of 8
    includes the base RID) is what ``DeriveAppContainerSidFromAppContainerName``
    produces. The well-known ambient package SIDs ``S-1-15-2-1`` and
    ``S-1-15-2-2`` have one, so they can never name a manifest: nothing in this
    module may drive an ACL revoke against them.
    """
    if not isinstance(text, str) or not text.startswith("S-1-15-2-"):
        return False
    parts = text.split("-")[4:]
    return len(parts) == 7 and all(
        part.isdigit() and 0 <= int(part) <= 0xFFFFFFFF for part in parts
    )


def _install_moniker() -> str:
    """One AppContainer name per (user, Studio installation), stable across restarts.

    ``CreateAppContainerProfile`` accepts up to 64 characters of
    ``[-_. A-Za-z0-9]``; this is 39 and uses only dots, hyphens and hex.
    """
    try:
        user = os.getlogin()
    except OSError:
        user = ""
    if not user:
        user = os.environ.get("USERNAME") or os.environ.get("USER") or ""
    profile_root = os.environ.get("LOCALAPPDATA") or ""
    if profile_root:
        profile_root = os.path.realpath(profile_root)
    parts = (
        os.path.normcase(os.path.realpath(sys.executable)),
        os.path.normcase(user),
        os.path.normcase(platform.node()),
        os.path.normcase(profile_root),
    )
    digest = hashlib.sha256("\0".join(parts).encode("utf-8", "surrogatepass")).hexdigest()
    return _INSTALL_PREFIX + digest[:16]


@dataclass(frozen = True)
class _InstallProfile:
    """The AppContainer identity shared by every launch of this installation."""

    moniker: str
    sid: ctypes.c_void_p
    sid_string: str
    profile_folder: str


_INSTALL_PROFILE: _InstallProfile | None = None
_INSTALL_PROFILE_LOCK = threading.Lock()


def _install_profile() -> _InstallProfile:
    """Create the installation's profile once, then reuse it for the process lifetime.

    ``ERROR_ALREADY_EXISTS`` is the ordinary outcome from the second Studio start
    onwards and means "reuse"; the SID is then derived from the name, which
    returns the same value the first ``CreateAppContainerProfile`` returned.
    """
    global _INSTALL_PROFILE
    with _INSTALL_PROFILE_LOCK:
        if _INSTALL_PROFILE is not None:
            return _INSTALL_PROFILE
        api = _api()
        moniker = _install_moniker()
        for last_attempt in (False, True):
            sid = ctypes.c_void_p()
            result = ctypes.c_uint32(
                api.userenv.CreateAppContainerProfile(
                    moniker,
                    "Unsloth Studio tool",
                    "Zero-capability Studio tool container",
                    None,
                    0,
                    ctypes.byref(sid),
                )
            ).value
            if result == _HRESULT_ALREADY_EXISTS:
                # Learn does not say whether the out parameter is written on this
                # result, so release anything it left before deriving the SID.
                if sid:
                    api.advapi32.FreeSid(sid)
                sid = _derive_container_sid(moniker)
            elif result != 0 or not sid:
                raise _hresult_error("CreateAppContainerProfile", result)
            try:
                sid_text = _sid_string(api, sid)
                profile_folder = _profile_folder(api, sid_text)
            except BaseException:
                api.advapi32.FreeSid(sid)
                raise
            if os.path.isdir(profile_folder):
                break
            # Registered, but its storage was deleted behind Windows' back. Only
            # CreateAppContainerProfile puts the package ACL on that directory, so
            # recreating it by hand would leave the container unable to read its
            # own profile, and _container_owned keeps this module from ever
            # granting there. Drop the registration and let the next pass build
            # it; if that does not work, refuse the launch rather than run
            # against a profile directory nothing has ACLed.
            api.advapi32.FreeSid(sid)
            if last_attempt:
                raise SandboxUnavailableError(
                    "the Studio AppContainer profile directory is missing and could not be "
                    f"created: {profile_folder}"
                )
            api.userenv.DeleteAppContainerProfile(moniker)
        _INSTALL_PROFILE = _InstallProfile(moniker, sid, sid_text, profile_folder)
        return _INSTALL_PROFILE


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


def _is_reparse_point(path: str) -> bool:
    """Whether a path is a junction or a symlink, tested without following it."""
    try:
        info = os.lstat(path)
    except OSError:
        return False
    return bool(getattr(info, "st_file_attributes", 0) & 0x400)


def _remove_reparse_point(path: str) -> None:
    """Remove the link itself.

    A sandboxed child needs no privilege to create a directory junction, so one
    planted in a directory this module empties must not be able to block cleanup
    forever, or to redirect it outside the container. ``os.rmdir`` is
    ``RemoveDirectoryW``, which deletes the reparse point and never follows it;
    a file symlink needs ``os.unlink`` instead.
    """
    try:
        os.rmdir(path)
        return
    except FileNotFoundError:
        return
    except OSError:
        pass
    try:
        os.unlink(path)
    except FileNotFoundError:
        return


def _force_removable(function: Callable[[str], Any], path: str, exc: BaseException) -> None:
    """Clear FILE_ATTRIBUTE_READONLY (or drop a reparse point) and retry one removal."""
    if isinstance(exc, FileNotFoundError):
        return
    if _is_reparse_point(path):
        _remove_reparse_point(path)
        return
    os.chmod(path, stat.S_IWRITE)
    function(path)


def _rmtree_onerror(function: Callable[[str], Any], path: str, info: tuple[Any, ...]) -> None:
    """The pre-3.12 ``onerror`` shape, which reports ``sys.exc_info`` instead."""
    _force_removable(function, path, info[1])


def _rmtree_error_handler() -> dict[str, Any]:
    """``onexc`` where it exists (3.12+), ``onerror`` on the older interpreters."""
    if sys.version_info >= (3, 12):
        return {"onexc": _force_removable}
    return {"onerror": _rmtree_onerror}


def _empty_directory(path: str) -> None:
    """Delete everything inside ``path``, leaving the directory itself in place.

    The container temp is named in the package environment Windows builds for
    every AppContainer child of this installation, so it is emptied rather than
    removed: a concurrent launch would otherwise find its own TEMP gone.

    A sandboxed child is free to leave a read-only file or a junction in there,
    and neither needs a privilege to create. Without the same
    FILE_ATTRIBUTE_READONLY clearing the Limited-mode removal does, one such file
    fails this call, and with it the cleanup of every later launch that inherits
    the same container temp.
    """
    with os.scandir(path) as entries:
        children = list(entries)
    handler = _rmtree_error_handler()
    for entry in children:
        # Checked before is_dir: a junction reports as a directory, and rmtree
        # would walk through it into whatever it points at.
        if _is_reparse_point(entry.path):
            _remove_reparse_point(entry.path)
        elif entry.is_dir(follow_symlinks = False):
            shutil.rmtree(entry.path, **handler)
        else:
            try:
                os.unlink(entry.path)
            except FileNotFoundError:
                continue
            except OSError as exc:
                _force_removable(os.unlink, entry.path, exc)


def _is_container_temp(profile_folder: str, private_temp: str) -> bool:
    """Whether ``private_temp`` is the container temp itself, not a child of it."""
    return os.path.normcase(os.path.abspath(private_temp)) == os.path.normcase(
        os.path.join(os.path.realpath(profile_folder), "Temp")
    )


def _validated_private_temp(profile_folder: str, private_temp: str) -> str:
    expected_parent = os.path.join(os.path.realpath(profile_folder), "Temp")
    spelled = os.path.abspath(private_temp)
    name = os.path.basename(spelled)
    # A launch now owns the container temp itself, because that is where Windows
    # points the child's TEMP; it is emptied, never removed, and only once no
    # other live launch of this installation holds it. A random child of Temp is
    # still accepted so a manifest written by an earlier build reconciles.
    inherited = (
        os.path.normcase(os.path.dirname(spelled)) == os.path.normcase(expected_parent)
        and len(name) == 24
        and all(character in "0123456789abcdef" for character in name.lower())
    )
    container = os.path.normcase(spelled) == os.path.normcase(expected_parent)
    if not (container or inherited):
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


@contextmanager
def _well_known_sid(text: str):
    """A SID allocated from its string form, freed on exit."""
    api = _api()
    sid = ctypes.c_void_p()
    if not api.advapi32.ConvertStringSidToSidW(text, ctypes.byref(sid)) or not sid:
        raise _winerror(f"ConvertStringSidToSidW({text})")
    try:
        yield sid
    finally:
        api.kernel32.LocalFree(sid)


def _ambient_sid_text(profile: str) -> str:
    """The group every token of this container kind carries implicitly.

    A less-privileged AppContainer opts out of ALL APPLICATION PACKAGES and only
    honours ALL RESTRICTED APPLICATION PACKAGES; a plain AppContainer is the reverse.
    """
    if profile == _PROFILE_APPCONTAINER:
        return _ALL_APPLICATION_PACKAGES_SID
    return _ALL_RESTRICTED_APPLICATION_PACKAGES_SID


def _ace_mask_covers(mask: int, required: int) -> bool:
    effective = mask
    if mask & _GENERIC_ALL:
        return True
    if mask & _GENERIC_READ:
        effective |= _FILE_GENERIC_READ
    if mask & _GENERIC_EXECUTE:
        effective |= _FILE_GENERIC_EXECUTE
    return (effective & required) == required


def _acl_grants(acl: ctypes.c_void_p, sids: tuple[ctypes.c_void_p, ...], required: int) -> bool:
    """Structural walk: do the allow ACEs for any of ``sids`` cover ``required``?

    Deny ACEs for one of the SIDs win, inherit-only ACEs do not apply to the
    object itself, and rights accumulate across allow ACEs the way the access
    check does. Anything unrecognised counts as not granted.
    """
    if not acl:
        return False
    api = _api()
    header = ctypes.cast(acl, ctypes.POINTER(_ACL)).contents
    allowed = 0
    for index in range(header.count):
        entry = ctypes.c_void_p()
        if not api.advapi32.GetAce(acl, index, ctypes.byref(entry)):
            raise _winerror("GetAce(LPAC access check)")
        ace_header = ctypes.cast(entry, ctypes.POINTER(_ACE_HEADER)).contents
        if ace_header.kind not in (_ACCESS_ALLOWED_ACE_TYPE, _ACCESS_DENIED_ACE_TYPE):
            continue
        if ace_header.flags & _INHERIT_ONLY_ACE:
            continue
        base = entry.value or 0
        # ACCESS_ALLOWED_ACE / ACCESS_DENIED_ACE: header (4 bytes), Mask (4 bytes), SidStart.
        mask = ctypes.cast(base + ctypes.sizeof(_ACE_HEADER), ctypes.POINTER(ctypes.c_uint32)).contents.value
        ace_sid = ctypes.c_void_p(base + ctypes.sizeof(_ACE_HEADER) + 4)
        if not any(api.advapi32.EqualSid(ace_sid, sid) for sid in sids):
            continue
        if ace_header.kind == _ACCESS_DENIED_ACE_TYPE:
            if mask & required or mask & (_GENERIC_ALL | _GENERIC_READ | _GENERIC_EXECUTE):
                return False
            continue
        allowed |= mask
        if _ace_mask_covers(allowed, required):
            return True
    return _ace_mask_covers(allowed, required)


def _existing_access(path: str, sids: tuple[ctypes.c_void_p, ...], required: int) -> bool:
    """True only when the current DACL provably already grants ``required``.

    Any failure to read or walk the DACL means "not verified", so the caller
    proceeds to grant exactly as before.
    """
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
        return False
    try:
        return _acl_grants(old_acl, sids, required)
    except OSError:
        return False
    finally:
        if descriptor:
            api.kernel32.LocalFree(descriptor)


_ACCESS_MEMO: dict[tuple[str, int, str, str], tuple[int, float, bool]] = {}
_ACCESS_MEMO_LOCK = threading.Lock()
# Setting or removing a DACL entry does not change a directory's modification
# time, so the mtime pin cannot notice another process (or icacls, or a repair
# install) dropping the grant. A Windows last write time also only advances with
# the system clock tick (about 15.6 ms), so the pin cannot separate two changes
# inside one tick either. The answer therefore also expires on its own; one
# DACL read per runtime root per minute is nothing next to a launch.
_ACCESS_MEMO_SECONDS = 60.0
_ACCESS_MEMO_LIMIT = 512


def _root_stamp(path: str) -> int:
    """The directory modification time a memoized access answer is pinned to."""
    try:
        return int(os.stat(path).st_mtime_ns)
    except OSError:
        return -1


def _memoized_existing_access(
    path: str,
    sids: tuple[ctypes.c_void_p, ...],
    required: int,
    *,
    sid_text: str,
    ambient_text: str,
) -> bool:
    """``_existing_access`` for a grant that is never revoked while this process lives.

    Only the persistent runtime grants use this. A per-launch ACE is revoked at
    cleanup, so an answer about one may not be carried into the next launch. The
    entry is pinned to the root's modification time, so an interpreter upgrade
    that writes into the root directory forces a fresh DACL read; it expires
    after ``_ACCESS_MEMO_SECONDS``, because a DACL change on its own moves no
    timestamp; and a failed ``stat`` is never memoized.
    """
    key = (os.path.normcase(path), int(required), sid_text, ambient_text)
    stamp = _root_stamp(path)
    if stamp != -1:
        with _ACCESS_MEMO_LOCK:
            entry = _ACCESS_MEMO.get(key)
        if (
            entry is not None
            and entry[0] == stamp
            and time.monotonic() - entry[1] < _ACCESS_MEMO_SECONDS
        ):
            return entry[2]
    result = _existing_access(path, sids, required)
    _memoize_access(path, required, sid_text, ambient_text, result, stamp = stamp)
    return result


def _memoize_access(
    path: str,
    required: int,
    sid_text: str,
    ambient_text: str,
    value: bool,
    *,
    stamp: int | None = None,
) -> None:
    if stamp is None:
        stamp = _root_stamp(path)
    if stamp == -1:
        return
    with _ACCESS_MEMO_LOCK:
        if len(_ACCESS_MEMO) >= _ACCESS_MEMO_LIMIT:
            # Keyed by path, so a long-lived Studio that saw many interpreters is
            # the only way to get here. Start again rather than grow forever.
            _ACCESS_MEMO.clear()
        _ACCESS_MEMO[(os.path.normcase(path), int(required), sid_text, ambient_text)] = (
            stamp,
            time.monotonic(),
            value,
        )


# Roots Windows refused to let this process ACL, by normalised path. The refusal
# is recorded once and not retried, so every later launch reads its disclosure
# from here rather than reporting a tree it never verified as verified.
_UNVERIFIED_ROOTS: dict[str, str] = {}

_SHARED_GRANTS: dict[str, int] = {}
# Re-entrant, and held across the whole grant and revoke of a shared path, not
# only across the counter update. Releasing the count first and revoking after
# would let a launch that starts in between read the ACE as already present,
# skip its own grant, and then lose it to the revoke.
_SHARED_GRANTS_LOCK = threading.RLock()
# One thread at a time reads the persistent manifest, decides what is missing and
# writes it back, so two concurrent first launches cannot each drop the other's
# roots from the record (or each pay the same propagation).
_PERSISTENT_GRANT_LOCK = threading.RLock()

# One named-mutex handle per key, for the life of the process. A host that
# cannot give us one is recorded as None and never asked again. Reconciliation
# asks about every installation the manifest directory names, not only this one,
# so the cache is capped: past the cap the fast path is simply given up and the
# lock file, which is the actual lock, answers alone.
_LEDGER_MUTEXES: dict[str, Any] = {}
_LEDGER_MUTEX_LOCK = threading.Lock()
_LEDGER_MUTEX_LIMIT = 64


def _ledger_mutex(key: str) -> Any:
    """The named mutex ``key`` names, created once per process, or ``None``."""
    with _LEDGER_MUTEX_LOCK:
        if key in _LEDGER_MUTEXES:
            return _LEDGER_MUTEXES[key]
        if len(_LEDGER_MUTEXES) >= _LEDGER_MUTEX_LIMIT:
            return None
        handle = None
        try:
            create = getattr(_api().kernel32, "CreateMutexW", None)
            if create is not None:
                handle = create(None, False, _LEDGER_MUTEX_PREFIX + key) or None
                if handle is None:
                    raise _winerror("CreateMutexW(sandbox ledger)")
        except Exception:  # noqa: BLE001 - the manifests stay the record either way
            logger.warning(
                "Could not create the sandbox ledger mutex for %s; two Studio processes are "
                "ordered by their manifests alone",
                key,
                exc_info = True,
            )
            handle = None
        _LEDGER_MUTEXES[key] = handle
        return handle


@contextmanager
def _named_mutex(key: str, deadline: float):
    """Hold the session mutex ``key`` names, as a fast path in front of the lock file.

    The session namespace (``Local\\``) is deliberately narrower than the state
    the ledger protects, which is per account and shared by every session. This
    mutex is not the lock; ``_ledger_file_lock`` is. What it buys is a kernel
    wait instead of a poll for the common case of two Studios in one session, so
    at most one process per session ever polls the lock file.

    It is therefore best effort: a host that will not give us the mutex, and a
    holder that has not released it inside the ledger's budget, both fall
    through to the lock file, which is what decides whether the ledger is held.
    ``WAIT_ABANDONED`` is ownership too, and is exactly the crashed Studio the
    manifests are reconciled for. Win32 mutexes are owned by a thread and are
    recursive, so the nesting the call sites do is one acquire per release.
    """
    handle = _ledger_mutex(key)
    held = False
    if handle is not None:
        milliseconds = max(0, int((deadline - time.monotonic()) * 1000))
        try:
            result = _api().kernel32.WaitForSingleObject(handle, milliseconds)
        except Exception:  # noqa: BLE001 - a wait that cannot run is not a launch failure
            logger.warning("Could not wait on the sandbox ledger mutex %s", key, exc_info = True)
            result = None
        if result in (_WAIT_OBJECT_0, _WAIT_ABANDONED):
            held = True
        else:
            logger.warning(
                "The sandbox ledger mutex %s was not acquired (%s); the lock file beside the "
                "manifests is what orders this section now",
                key,
                result,
            )
    try:
        yield held
    finally:
        if held:
            try:
                _api().kernel32.ReleaseMutex(handle)
            except Exception:  # noqa: BLE001 - never mask the body's own failure
                logger.warning("Could not release the sandbox ledger mutex %s", key, exc_info = True)


# The lock-file handle, the re-entrancy depth and the refusal depth of every
# ledger a thread is inside, keyed by (thread, moniker) and read and written only
# while _SHARED_GRANTS_LOCK is held, which _installation_ledger takes before
# anything else. The thread is in the key rather than implied by that lock, so
# that narrowing its scope later cannot silently make one thread's ledger answer
# for another's.
#
# A depth above zero means this thread already owns the lock file, and a nested
# section must not try to open it a second time: no-sharing is exclusive against
# this process too. A refusal depth above zero means an enclosing section already
# waited out the whole budget and was refused; a nested section is refused at
# once rather than waiting the budget again, which is what turned one contended
# prepare into two full waits before the tool process was even spawned.
_LEDGER_FILE_LOCKS: dict[tuple[int, str], Any] = {}
_LEDGER_DEPTH: dict[tuple[int, str], int] = {}
_LEDGER_REFUSED: dict[tuple[int, str], int] = {}


def _ledger_lock_path(key: str) -> str:
    """The lock file ``key`` names, inside the manifest root and nowhere else.

    ``key`` is a moniker, and a moniker can come out of a manifest this process
    did not write. ``CreateAppContainerProfile`` accepts only
    ``[-_. A-Za-z0-9]``, so anything outside that is not a profile name at all
    and certainly not a path component: a planted ``..\\..\\`` would otherwise
    take an exclusive handle on any file this account can write.
    """
    if not key or len(key) > 64 or any(character not in _MONIKER_CHARACTERS for character in key):
        raise SandboxUnavailableError("an LPAC ledger key is not an AppContainer profile name")
    root = _manifest_root()
    path = os.path.join(root, key + _LEDGER_LOCK_SUFFIX)
    if os.path.dirname(path) != root:
        raise SandboxUnavailableError("an LPAC ledger lock escapes the manifest directory")
    return path


def _ledger_lock_last_error() -> int:
    """The Win32 error the last ``CreateFileW`` left behind.

    ``ctypes.get_last_error`` exists only on Windows. Where it does not, the
    caller is the Linux test host driving a fake ``kernel32``, whose only way to
    report contention is the invalid handle, so read that as the contended case
    rather than as an unrecoverable one.
    """
    getter = getattr(ctypes, "get_last_error", None)
    if getter is None:
        return _ERROR_SHARING_VIOLATION
    return int(getter())


def _open_ledger_lock(key: str, deadline: float) -> Any:
    """Open the ledger lock file for ``key`` exclusively, or ``None`` by ``deadline``.

    ``dwShareMode`` 0 is the exclusion: the file cannot be opened again, in any
    session and by any process, until this handle is closed, and Windows closes
    it when this process dies. There is no blocking form of that, so contention
    is polled with a backoff; a failure that is not contention is not retried,
    because spinning for the whole budget would only delay a refusal.
    """
    try:
        path = _ledger_lock_path(key)
    except (SandboxUnavailableError, OSError):
        logger.warning("Could not place the sandbox ledger lock file for %s", key, exc_info = True)
        return None
    create = getattr(_api().kernel32, "CreateFileW", None)
    if create is None:
        logger.warning("This host offers no CreateFileW for the sandbox ledger lock %s", path)
        return None
    delay = _LEDGER_LOCK_FIRST_POLL_SECONDS
    started = time.monotonic()
    while True:
        try:
            handle = create(
                path,
                _GENERIC_WRITE,
                _FILE_SHARE_NONE,
                None,
                _OPEN_ALWAYS,
                _FILE_ATTRIBUTE_NORMAL,
                None,
            )
        except Exception:  # noqa: BLE001 - reported by the caller as "not held"
            logger.warning("Could not open the sandbox ledger lock %s", path, exc_info = True)
            return None
        if handle and handle != _INVALID_HANDLE_VALUE:
            return handle
        error = _ledger_lock_last_error()
        if error not in (_ERROR_SHARING_VIOLATION, _ERROR_LOCK_VIOLATION):
            # A sharing violation is the documented answer to "another handle
            # holds it", and the only one worth waiting on. Anything else, an
            # access denial included, refuses now: the caller defers the work and
            # the next tool call retries it, which beats stalling this one for
            # thirty seconds on a condition that will not clear.
            logger.warning("The sandbox ledger lock %s could not be opened (%s)", path, error)
            return None
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            logger.warning(
                "The sandbox ledger lock %s is still held by another Studio process after %.2fs",
                path,
                time.monotonic() - started,
            )
            return None
        time.sleep(min(delay, remaining))
        delay = min(delay * 2, _LEDGER_LOCK_MAX_POLL_SECONDS)


@contextmanager
def _nested_ledger_depth(counts: dict[tuple[int, str], int], entry: tuple[int, str], answer: bool):
    """Count one more section on ``entry``, and drop the key when the last leaves."""
    counts[entry] = counts.get(entry, 0) + 1
    try:
        yield answer
    finally:
        counts[entry] -= 1
        if not counts[entry]:
            del counts[entry]


@contextmanager
def _ledger_file_lock(key: str, deadline: float):
    """Hold the machine-wide, per-account ledger lock file for ``key``."""
    entry = (threading.get_ident(), key)
    if _LEDGER_DEPTH.get(entry, 0):
        # Already ours; a second exclusive open would fail against our own handle.
        with _nested_ledger_depth(_LEDGER_DEPTH, entry, True) as held:
            yield held
        return
    if _LEDGER_REFUSED.get(entry, 0):
        # An enclosing section already spent the budget on this key and lost.
        with _nested_ledger_depth(_LEDGER_REFUSED, entry, False) as held:
            yield held
        return
    handle = _open_ledger_lock(key, deadline)
    if handle is None:
        with _nested_ledger_depth(_LEDGER_REFUSED, entry, False) as held:
            yield held
        return
    _LEDGER_FILE_LOCKS[entry] = handle
    try:
        with _nested_ledger_depth(_LEDGER_DEPTH, entry, True) as held:
            yield held
    finally:
        _LEDGER_FILE_LOCKS.pop(entry, None)
        try:
            _api().kernel32.CloseHandle(handle)
        except Exception:  # noqa: BLE001 - never mask the body's own failure
            logger.warning("Could not release the sandbox ledger lock %s", key, exc_info = True)


def _ledger_is_held(key: str) -> bool:
    """Whether this thread is inside a ledger section that actually took the lock.

    For a destructive step nested inside a section that is allowed to run
    unlocked: prepare grants without the ledger, but the temp it empties when a
    launch fails to prepare is not something it may do unsynchronised.
    """
    with _SHARED_GRANTS_LOCK:
        return _LEDGER_DEPTH.get((threading.get_ident(), key), 0) > 0


@contextmanager
def _installation_ledger(moniker: str = "", *, wait_seconds: float | None = None):
    """Hold this installation's ownership ledger against every thread and process.

    ``_SHARED_GRANTS_LOCK`` orders the launches of one Studio and nothing else.
    The write-ahead manifests are what a second Studio of the same installation
    is seen through, and reading them is a check, not a lock: it can write its
    manifest in the window between this process's scan and its revoke, and then
    lose the ACE it was just granted, under a container that is already running.
    A ledger keyed by the installation moniker closes that window. It is held
    across scan-and-revoke, across prepare's create-and-grant, across
    reconciliation and across the persistent removal, so no two of those
    interleave between processes.

    ``held`` is the whole contract: it is true only when the lock file was taken,
    and every destructive step must refuse to run when it is false. Use
    ``_destructive_ledger`` rather than reading it by hand.

    ``_SHARED_GRANTS_LOCK`` is always taken first and the mutex before the lock
    file, so no two holders can acquire the three in different orders.

    ``wait_seconds`` of 0 asks for the ledger without waiting for it, which is
    what background housekeeping wants: a reconciliation pass covers every
    installation on the account, and waiting the full budget on each busy one in
    turn would stall a Studio start by that many multiples of it for work the
    next pass does anyway.
    """
    key = moniker or _install_moniker()
    budget = _LEDGER_WAIT_SECONDS if wait_seconds is None else wait_seconds
    with _SHARED_GRANTS_LOCK:
        # Started after the process-local lock, not before it. The budget is for
        # the wait on other processes; spending it queued behind this process's
        # own 14 s first-launch propagation would fail a cleanup closed over a
        # contention that was never cross-process at all.
        deadline = time.monotonic() + budget
        with _named_mutex(key, deadline):
            with _ledger_file_lock(key, deadline) as held:
                yield held


class _LedgerUnavailableError(SandboxUnavailableError):
    """The ledger could not be taken, so a destructive step was refused."""


@contextmanager
def _destructive_ledger(moniker: str, operation: str):
    """The ledger for a block that revokes, empties or deletes, failing closed.

    Revoking an ACE under another process's running container, emptying the
    container temp under it, or deleting the profile it is using are the exact
    damage the ledger exists to prevent, so a ledger that could not be taken
    refuses the block instead of running it unsynchronised. Every one of these
    is recoverable: the write-ahead manifest that named the state is still on
    disk, and the next reconciliation that does hold the ledger releases it.
    """
    with _installation_ledger(moniker) as held:
        if not held:
            raise _LedgerUnavailableError(
                f"the Studio sandbox ledger for {moniker} could not be taken, so {operation} was "
                "left for the next reconciliation"
            )
        yield


# Launch records whose destructive cleanup the ledger refused, by moniker. The
# identity is kept whole - nothing of its claim is given back - and the whole
# cleanup is retried under the next ledger of that installation this process
# does hold, so a lock that was busy for one cleanup does not strand an ACE
# until the process exits.
_DEFERRED_CLEANUPS: dict[str, list[Any]] = {}
# A host whose ledger is never available would otherwise queue one identity per
# tool call, forever. The oldest is dropped instead: its manifest is still on
# disk and still names every ACE it owns, so the next Studio that does hold the
# ledger reconciles it from there.
_DEFERRED_CLEANUP_LIMIT = 256


def _defer_cleanup(identity: Any) -> None:
    with _SHARED_GRANTS_LOCK:
        pending = _DEFERRED_CLEANUPS.setdefault(identity.moniker, [])
        pending.append(identity)
        while len(pending) > _DEFERRED_CLEANUP_LIMIT:
            dropped = pending.pop(0)
            # Its count is given back even though its ACE is not revoked. The
            # count is this process's own bookkeeping and the launch really has
            # finished; leaving it would pin those paths as "in use" for the rest
            # of the process, which refuses every later container release. The
            # manifest still names the ACE, and is what releases it in the end.
            if dropped.shared_roots and dropped._released is None:
                dropped._released = _release_shared_grants(dropped.shared_roots)
            logger.warning(
                "The sandbox ledger for %s has been unavailable for %d cleanups; the grants of "
                "launch %s are now left entirely to a reconciliation after this process exits",
                identity.moniker,
                _DEFERRED_CLEANUP_LIMIT,
                dropped.launch_id,
            )


def _drain_deferred_cleanups(moniker: str) -> None:
    """Retry the cleanups the ledger refused. Only ever called while holding it."""
    with _SHARED_GRANTS_LOCK:
        pending = _DEFERRED_CLEANUPS.pop(moniker, [])
    for identity in pending:
        try:
            identity.cleanup()
        except Exception:  # noqa: BLE001 - queued again rather than dropped
            logger.warning(
                "Could not release the deferred LPAC launch %s", identity.launch_id, exc_info = True
            )
            # Back on the queue, at the tail, so a record whose revoke keeps
            # failing does not simply vanish from this process's retry list. The
            # list was popped before the loop, so this is not retried until the
            # next drain.
            _defer_cleanup(identity)


def _hold_shared_grants(paths: tuple[str, ...]) -> None:
    """Record that one more live launch of this installation needs these ACEs.

    Every launch of an installation now carries the same SID, so a per-launch
    revoke would remove an ACE a concurrent launch is still using. The count is
    per process; a launch left behind by a crashed Studio is reconciled from its
    manifest instead.
    """
    with _SHARED_GRANTS_LOCK:
        for path in paths:
            key = os.path.normcase(path)
            _SHARED_GRANTS[key] = _SHARED_GRANTS.get(key, 0) + 1


def _held_shared_grants() -> frozenset[str]:
    """The paths a live launch of this process is relying on right now."""
    with _SHARED_GRANTS_LOCK:
        return frozenset(key for key, count in _SHARED_GRANTS.items() if count > 0)


def _release_shared_grants(paths: tuple[str, ...]) -> set[str]:
    """Drop this launch's hold and return the paths no live launch needs any more."""
    released: set[str] = set()
    with _SHARED_GRANTS_LOCK:
        for path in paths:
            key = os.path.normcase(path)
            remaining = _SHARED_GRANTS.get(key, 1) - 1
            if remaining > 0:
                _SHARED_GRANTS[key] = remaining
                continue
            _SHARED_GRANTS.pop(key, None)
            released.add(key)
    return released


def _live_launch_holds(
    exclude_launch_id: str = "", *, moniker: str | None = None
) -> frozenset[str]:
    """Normalised paths a launch of any live Studio process still needs.

    ``_SHARED_GRANTS`` counts this process's launches only, and every launch of
    an installation now carries the same SID, so a second Studio's ACE on a
    shared path is indistinguishable from this one's: revoking it would leave
    that container unable to reach its own workdir. The write-ahead manifests
    are the cross-process ledger. One is written before its launch grants
    anything and removed only after it has revoked, so a manifest whose owner
    process is still alive names paths this cleanup must leave alone.

    ``moniker`` narrows that to one installation, and every caller that is
    revoking passes its own. Two installations (two interpreters, two accounts)
    derive different package SIDs from their different names, so nothing one
    revokes can reach the other's ACE, and a foreign manifest answering here
    would only suppress this installation's revoke on a path both happen to use
    - permanently, for as long as that manifest exists, since nothing in this
    process ever removes another installation's record. ``None`` means every
    installation, which is what the all-installations reset asks about.

    Best effort by design. A manifest root that cannot be read yields nothing,
    which is where this was before the ledger existed. The instant between
    another process writing its manifest and this one reading the directory is
    closed by the named mutex ``_installation_ledger`` holds across the scan and
    the revoke it gates, not by the scan itself.
    """
    try:
        root = _manifest_root()
    except (SandboxUnavailableError, OSError):
        logger.warning("Could not read the LPAC manifest root for the launch ledger", exc_info = True)
        return frozenset()
    held: set[str] = set()
    for manifest in Path(root).glob(_LAUNCH_PREFIX + "*.json"):
        payload = _parse_manifest(manifest)
        if payload is None or payload["kind"] != _MANIFEST_KIND_LAUNCH:
            continue
        if moniker is not None and payload["moniker"] != moniker:
            continue
        if exclude_launch_id and payload["launch_id"] == exclude_launch_id:
            continue
        owner = (payload["owner_pid"], payload["owner_created"])
        try:
            live = _process_identity(owner[0]) == owner
        except OSError:
            # An owner this process cannot query is assumed live: a revoke under
            # a running container is worse than an ACE reconciled a little late.
            live = True
        if not live:
            continue
        held.update(os.path.normcase(path) for path in payload["granted_roots"])
        held.add(os.path.normcase(payload["private_temp"]))
    return frozenset(held)


def _release_private_temp(
    profile_folder: str,
    private_temp: str,
    *,
    released: set[str] | None,
    elsewhere: frozenset[str],
) -> None:
    """Give up one launch's claim on its temp without disturbing a live one.

    The container temp is the directory Windows points every AppContainer child
    of this installation at, so it belongs to the container and not to a launch:
    it is emptied, never removed, and only once nothing else holds it. Removing
    it would take the TEMP out from under every concurrent launch, and the
    shared-temp change is what made that possible - before it, a launch owned a
    per-launch subdirectory that was its alone.

    ``released`` is the set of paths this launch was the last holder of in this
    process, or ``None`` when there is no per-process count to consult (a
    manifest reconciled for a dead owner). ``elsewhere`` is what a launch of
    another Studio process still holds. A per-launch child of the temp, which
    only a manifest written by an earlier build can name, is removed as before.
    """
    validated = _validated_private_temp(profile_folder, private_temp)
    key = os.path.normcase(validated)
    if key in elsewhere:
        return
    if _is_container_temp(profile_folder, validated):
        if released is None or key in released:
            _empty_directory(validated)
        return
    shutil.rmtree(validated, ignore_errors = False)


def _container_owned(path: str, profile_folder: str) -> bool:
    """Whether Windows, and not this module, put the package SID on ``path``.

    ``CreateAppContainerProfile`` builds ``...\\Packages\\<moniker>`` with an ACE
    for the package SID, and everything under it inherits that ACE. Granting
    there is unnecessary, and revoking there would strip the container's access
    to its own storage, so those paths never enter a revoke list. The profile
    used to be deleted at every launch, which hid this.
    """
    package = os.path.dirname(os.path.realpath(profile_folder))
    drive, tail = os.path.splitdrive(package)
    if not package or tail in ("", "\\", "/"):
        return False
    return _is_within(path, package)


def _machine_wide(path: str) -> bool:
    """Trees Windows owns and already ACLs for application packages."""
    roots = []
    for name in ("ProgramFiles", "ProgramFiles(x86)", "ProgramW6432", "SystemRoot"):
        value = os.environ.get(name)
        if value and os.path.isabs(value):
            roots.append(os.path.realpath(value))
    if not roots:
        roots = [r"C:\Program Files", r"C:\Windows"]
    return any(_is_within(path, root) for root in roots)


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
    profile: str = _PROFILE_LPAC
    unverified_access: tuple[str, ...] = ()
    workdir: str = ""
    launch_id: str = ""
    # Paths whose ACE another live launch of this installation may still need.
    shared_roots: tuple[str, ...] = ()
    # Only a single-use profile, the shape that predates the stable identity,
    # is deleted and has its SID freed by whoever reconciles its manifest.
    delete_profile: bool = False
    free_sid: bool = False
    _released: set[str] | None = field(default = None, repr = False, compare = False)

    def cleanup(self, *, defer_when_locked: bool = True) -> None:
        """Revoke this launch's grants and give its share of the temp back.

        Every step here is destructive, so all of it is inside the ledger and
        none of it runs without it. ``defer_when_locked`` keeps the whole record
        - the manifest, the in-process hold, the SID - and queues the identity
        for the next ledger this process does hold. A reconciliation passes
        ``False``: it already holds the ledger of the installation it is
        processing, and its identity owns a SID that its own caller frees.
        """
        if self.cleaned:
            return
        errors: list[str] = []
        # The ledger spans the release and the revokes, in this process and in
        # every other Studio of this installation: a launch starting in between
        # would otherwise see the ACE, skip its own grant, and lose it here.
        with _installation_ledger(self.moniker) as held:
            if not held:
                # Nothing of the claim is given back: the manifest, the hold and
                # the SID stay exactly as they are, so the retry has the same
                # record to work from that a reconciliation would.
                if not defer_when_locked:
                    raise _LedgerUnavailableError(
                        f"the Studio sandbox ledger for {self.moniker} could not be taken, so "
                        "this launch's grants were left for the next reconciliation"
                    )
                _defer_cleanup(self)
                logger.warning(
                    "The sandbox ledger for %s could not be taken, so the grants of launch %s were "
                    "left in place rather than revoked unsynchronised; they are released by the "
                    "next reconciliation of this installation",
                    self.moniker,
                    self.launch_id,
                )
                return
            targets = self.granted_roots
            if self.shared_roots:
                if self._released is None:
                    self._released = _release_shared_grants(self.shared_roots)
                shared = {os.path.normcase(path) for path in self.shared_roots}
                targets = tuple(
                    path
                    for path in self.granted_roots
                    if os.path.normcase(path) not in shared
                    or os.path.normcase(path) in self._released
                )
            # The in-process count answers for this Studio only, and the SID is
            # the installation's: another live Studio's launch needs the same
            # ACEs and would lose them here.
            elsewhere = _live_launch_holds(self.launch_id, moniker = self.moniker)
            targets = tuple(path for path in targets if os.path.normcase(path) not in elsewhere)
            traverse = {os.path.normcase(path) for path in self.traverse_roots}
            for path in reversed(targets):
                try:
                    _revoke_sid(path, self.sid, exact = os.path.normcase(path) in traverse)
                except Exception as exc:  # noqa: BLE001 - continue ownership cleanup
                    errors.append(f"ACL {path}: {exc}")
            # Inside the ledger: a launch starting between the release above and
            # the removal below would otherwise have its temp emptied under it.
            try:
                _release_private_temp(
                    self.profile_folder,
                    self.private_temp,
                    released = self._released,
                    elsewhere = elsewhere,
                )
            except FileNotFoundError:
                pass
            except Exception as exc:  # noqa: BLE001
                errors.append(f"temp {self.private_temp}: {exc}")
            if errors:
                raise OSError("; ".join(errors))
            # Still inside the ledger. Deleting the profile a single-use manifest
            # names, and removing the record that says which ACEs this launch
            # owns, are as destructive as the revokes above: a reconciliation
            # that read the record between the revoke and the unlink would revoke
            # the same paths a second time, and the profile is the directory
            # another launch's temp lives in.
            if self.delete_profile:
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
            if self.free_sid:
                _api().advapi32.FreeSid(self.sid)
                self.sid = ctypes.c_void_p()
            self.cleaned = True


def _atomic_write_manifest(path: str, payload: dict[str, Any]) -> None:
    """Replace ``path`` with ``payload``, leaving either the old file or the new one.

    The temporary name carries its own random suffix. The persistent manifest has
    a fixed name, and an interrupted write must not make every later write fail
    with ``FileExistsError``; the leftover is swept by the next reconciliation.
    """
    temporary = f"{path}.{secrets.token_hex(8)}.tmp"
    try:
        with open(temporary, "x", encoding = "utf-8") as stream:
            json.dump(payload, stream, sort_keys = True)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def _write_manifest(identity: _InvocationIdentity) -> None:
    """Record what one launch owns, before it is granted anything."""
    payload = {
        "version": 1,
        "kind": _MANIFEST_KIND_LAUNCH,
        "moniker": identity.moniker,
        "launch_id": identity.launch_id,
        "sid": identity.sid_string,
        "profile_folder": identity.profile_folder,
        "private_temp": identity.private_temp,
        "workdir": identity.workdir,
        "granted_roots": list(identity.granted_roots),
        "traverse_roots": list(identity.traverse_roots),
        "owner_pid": identity.owner_pid,
        "owner_created": identity.owner_created,
    }
    _atomic_write_manifest(identity.manifest_path, payload)


def _persistent_manifest_path(moniker: str) -> str:
    return os.path.join(_manifest_root(), moniker + ".json")


def _write_persistent_manifest(
    install: _InstallProfile,
    granted_roots: tuple[str, ...],
    traverse_roots: tuple[str, ...],
) -> None:
    """Record the installation-wide grants, before they are made.

    No owning process is recorded on purpose: the grant outlives the process that
    made it and is released by ``remove_persistent_grants`` or by a reconciliation
    that finds the recorded interpreter gone.
    """
    payload = {
        "version": 1,
        "kind": _MANIFEST_KIND_PERSISTENT,
        "moniker": install.moniker,
        "sid": install.sid_string,
        "profile_folder": install.profile_folder,
        "interpreter": os.path.realpath(sys.executable),
        "granted_roots": list(granted_roots),
        "traverse_roots": list(traverse_roots),
    }
    _atomic_write_manifest(_persistent_manifest_path(install.moniker), payload)


def _is_hex(value: Any, length: int) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
        and all(character in "0123456789abcdef" for character in value)
    )


def _absolute_string_list(value: Any) -> bool:
    return isinstance(value, list) and all(
        isinstance(path, str) and os.path.isabs(path) for path in value
    )


def _is_ancestor_of_any(path: str, targets: tuple[str, ...]) -> bool:
    normalized = os.path.normcase(os.path.abspath(path))
    for target in targets:
        current = os.path.normcase(os.path.abspath(target))
        parent = os.path.dirname(current)
        while parent != current:
            current = parent
            if current == normalized:
                return True
            parent = os.path.dirname(current)
    return False


def _parse_manifest(manifest: Path) -> dict[str, Any] | None:
    """The validated payload of a manifest, or ``None`` when it is not one of ours.

    Reconciliation revokes an ACE on every path a manifest names, so a manifest
    this process did not write must not be able to name an arbitrary path or an
    arbitrary principal. Two invariants bound that. The SID must be a parent
    AppContainer SID, and the caller checks it against the SID derived from the
    moniker, which is constrained to the ``unsloth.studio.`` namespace: no real
    application package, and no account, can be named that way. And a launch
    manifest may only name its workdir, its private temp, and their ancestors.

    A persistent manifest cannot be constrained that way: the roots it records
    are whichever runtime trees the tools on this host live in. What bounds it is
    the SID: a revoke asks only for that one derived package SID, and
    ``_set_sid_acl`` does nothing at all when the DACL does not already carry it,
    so a planted path this installation never granted is a no-op.
    """
    try:
        payload = json.loads(manifest.read_text(encoding = "utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(payload, dict) or payload.get("version") != 1:
        return None
    moniker = payload.get("moniker")
    roots = payload.get("granted_roots")
    traverse = payload.get("traverse_roots", [])
    private_temp = payload.get("private_temp")
    profile_folder = payload.get("profile_folder")
    kind = payload.get("kind", _MANIFEST_KIND_SINGLE_USE)
    if (
        not isinstance(moniker, str)
        or not moniker.startswith(_PROFILE_PREFIX)
        or len(moniker) > 64
        or any(character not in _MONIKER_CHARACTERS for character in moniker)
        # A moniker names the ledger lock file as well as the profile, so it is
        # held to what CreateAppContainerProfile itself accepts. Without this a
        # planted record could put a path separator in it and make this process
        # take an exclusive handle anywhere the account can write.
        or not _is_container_sid_text(payload.get("sid"))
        or not _absolute_string_list(roots)
        or not _absolute_string_list(traverse)
        or not isinstance(profile_folder, str)
        or not os.path.isabs(profile_folder)
    ):
        return None
    if kind == _MANIFEST_KIND_PERSISTENT:
        interpreter = payload.get("interpreter")
        if (
            not moniker.startswith(_INSTALL_PREFIX)
            or manifest.name != moniker + ".json"
            or not isinstance(interpreter, str)
            or not os.path.isabs(interpreter)
        ):
            return None
        return {**payload, "kind": kind, "traverse_roots": traverse}
    if (
        not isinstance(private_temp, str)
        or not os.path.isabs(private_temp)
        or not isinstance(payload.get("owner_pid"), int)
        or not isinstance(payload.get("owner_created"), int)
    ):
        return None
    if kind == _MANIFEST_KIND_LAUNCH:
        workdir = payload.get("workdir")
        if (
            not moniker.startswith(_INSTALL_PREFIX)
            # The only manifest kind whose file name does not repeat its moniker,
            # so the moniker is pinned to the shape _install_moniker produces
            # instead: the installation prefix and the 16 hex of its digest.
            or not _is_hex(moniker[len(_INSTALL_PREFIX) :], 16)
            or not _is_hex(payload.get("launch_id"), 32)
            or manifest.name != _LAUNCH_PREFIX + payload["launch_id"] + ".json"
            or not isinstance(workdir, str)
            or not os.path.isabs(workdir)
            # The container temp, which reconciliation empties rather than
            # removes, or the random child of it an earlier build created there.
            or not (
                _is_container_temp(profile_folder, private_temp)
                or (
                    _is_hex(os.path.basename(private_temp), 24)
                    and os.path.normcase(os.path.dirname(private_temp))
                    == os.path.normcase(os.path.join(profile_folder, "Temp"))
                )
            )
        ):
            return None
        # A launch grants, and so revokes, its workdir and the ancestors it had
        # to make traversable. Its private temp is deleted rather than revoked,
        # and is pinned to the profile the SID resolves to by the caller.
        if {os.path.normcase(path) for path in roots} != {os.path.normcase(workdir)} | {
            os.path.normcase(path) for path in traverse
        }:
            return None
        if any(not _is_ancestor_of_any(path, (workdir, private_temp)) for path in traverse):
            return None
        if any(_container_owned(path, profile_folder) for path in roots):
            return None
        return {**payload, "kind": kind, "traverse_roots": traverse}
    if kind == _MANIFEST_KIND_SINGLE_USE:
        # Written before the stable identity: one random profile per launch, its
        # manifest named after that moniker, and the profile is its to delete.
        if (
            "kind" in payload
            or moniker.startswith(_INSTALL_PREFIX)
            or moniker.startswith(_LAUNCH_PREFIX)
            or manifest.name != moniker + ".json"
        ):
            return None
        return {**payload, "kind": kind, "traverse_roots": traverse}
    return None


def _remove_orphan_temporary_manifests(root: str) -> None:
    """Delete what a crash between ``open`` and ``os.replace`` left behind."""
    for temporary in Path(root).glob(_PROFILE_PREFIX + "*.tmp"):
        try:
            if time.time() - temporary.stat().st_mtime <= _ORPHAN_TEMPORARY_MANIFEST_SECONDS:
                continue
            temporary.unlink()
        except OSError:
            logger.warning(
                "Could not remove the orphaned LPAC manifest %s", temporary, exc_info = True
            )


@dataclass(frozen = True)
class _PersistentGrants:
    """What the installation already holds, so a launch does not repeat it."""

    paths: frozenset[str]
    unverified: tuple[str, ...]


def _read_persistent_manifest(moniker: str) -> dict[str, Any] | None:
    manifest = Path(_persistent_manifest_path(moniker))
    if not manifest.is_file():
        return None
    payload = _parse_manifest(manifest)
    if payload is None or payload["kind"] != _MANIFEST_KIND_PERSISTENT:
        return None
    return payload if payload["moniker"] == moniker else None


def _ensure_persistent_grants(
    install: _InstallProfile, roots: tuple[str, ...], profile: str
) -> _PersistentGrants:
    """Grant read+execute on the runtime roots once per installation, and keep it.

    This is the 14 s that used to be paid on every tool call: propagating a fresh
    package SID over the interpreter tree. The SID is now stable, so the grant is
    made only when the DACL does not already carry it, is recorded in a manifest
    no launch cleanup touches, and is released only by ``remove_persistent_grants``
    or by a reconciliation that finds the recorded interpreter gone. A launch that
    needs a runtime root the installation has not seen (a Terminal call reaching
    Git bash after a Python call) pays for that root once and appends it.
    """
    temp_root = os.path.join(install.profile_folder, "Temp")
    traverse = tuple(
        path
        for path in _traverse_ancestors((*roots, temp_root))
        if not _container_owned(path, install.profile_folder)
    )
    ambient_text = _ambient_sid_text(profile)
    read_execute = _FILE_GENERIC_READ | _FILE_GENERIC_EXECUTE
    unverified: list[str] = []
    # The ledger covers the read, the decision and the write-back across
    # processes too: two Studios of one installation starting for the first time
    # would otherwise each record only the roots it found missing, and the second
    # write would drop the first's from the manifest that releases them.
    with (
        _installation_ledger(install.moniker),
        _PERSISTENT_GRANT_LOCK,
        _well_known_sid(ambient_text) as ambient,
    ):
        # _UNVERIFIED_ROOTS is read and written under this lock only.
        recorded = _read_persistent_manifest(install.moniker) or {}
        sids = (install.sid, ambient)

        def resolved(path: str, required: int) -> bool:
            return _memoized_existing_access(
                path,
                sids,
                required,
                sid_text = install.sid_string,
                ambient_text = ambient_text,
            )

        missing_roots = tuple(path for path in roots if not resolved(path, read_execute))
        missing_traverse = tuple(path for path in traverse if not resolved(path, _FILE_TRAVERSE))
        granted_all = tuple(dict.fromkeys((*recorded.get("granted_roots", ()), *missing_roots)))
        traverse_all = tuple(
            dict.fromkeys((*recorded.get("traverse_roots", ()), *missing_traverse))
        )
        if missing_roots or missing_traverse:
            if missing_roots:
                # Only a tree that is about to receive a propagating ACE is
                # walked. A tree this installation already granted is not walked
                # again: nothing is written to it, so a reparse point planted
                # afterwards cannot redirect an ACE that is not being applied.
                # A traverse ACE is exact and applies to the directory alone.
                _validate_runtime_trees(missing_roots)
            _write_persistent_manifest(install, granted_all, traverse_all)
            for path, grant, required in (
                *((path, _grant_read_execute, read_execute) for path in missing_roots),
                *((path, _grant_traverse, _FILE_TRAVERSE) for path in missing_traverse),
            ):
                try:
                    grant(path, install.sid)
                except OSError as exc:
                    if exc.errno == _ERROR_ACCESS_DENIED and _machine_wide(path):
                        # Windows owns this tree and already ACLs it for
                        # application packages. Record the refusal once instead of
                        # re-walking and re-attempting it on every later launch.
                        _UNVERIFIED_ROOTS[os.path.normcase(path)] = path
                        _memoize_access(path, required, install.sid_string, ambient_text, True)
                        continue
                    raise
                _UNVERIFIED_ROOTS.pop(os.path.normcase(path), None)
                _memoize_access(path, required, install.sid_string, ambient_text, True)
        unverified = [
            path
            for path in (*roots, *traverse)
            if os.path.normcase(path) in _UNVERIFIED_ROOTS
        ]
    return _PersistentGrants(
        frozenset(os.path.normcase(path) for path in (*granted_all, *traverse_all)),
        tuple(unverified),
    )


def _revoke_persistent_manifest(manifest: Path, payload: dict[str, Any]) -> None:
    """Revoke every grant a persistent manifest records, then delete the manifest."""
    sid = _derive_container_sid(payload["moniker"])
    try:
        if _sid_string(_api(), sid) != payload["sid"]:
            raise SandboxUnavailableError(
                "the LPAC persistent manifest SID does not match its profile name"
            )
        traverse = {os.path.normcase(path) for path in payload["traverse_roots"]}
        errors: list[str] = []
        for path in (*payload["granted_roots"], *payload["traverse_roots"]):
            try:
                _revoke_sid(path, sid, exact = os.path.normcase(path) in traverse)
            except Exception as exc:  # noqa: BLE001 - keep revoking the rest
                errors.append(f"ACL {path}: {exc}")
        if errors:
            raise OSError("; ".join(errors))
    finally:
        _api().advapi32.FreeSid(sid)
    try:
        manifest.unlink()
    except FileNotFoundError:
        pass


def _recorded_monikers(root: str, prefix: str, kind: str | None = None) -> set[str]:
    """Which installations the manifests under ``root`` belong to.

    Every manifest carries the moniker of the installation that wrote it, which
    is what says whose ledger orders the process that owns it. Reading it needs
    no lock: it decides only which locks to take, and a record that appears
    afterwards is skipped by the caller rather than acted on unlocked.
    """
    found: set[str] = set()
    for manifest in Path(root).glob(prefix + "*.json"):
        payload = _parse_manifest(manifest)
        if payload is not None and (kind is None or payload["kind"] == kind):
            found.add(payload["moniker"])
    return found


def _remove_persistent_grants(*, all_installations: bool = False) -> tuple[str, ...]:
    """Revoke the installation-wide grants and delete its container profile.

    For uninstall or a deliberate reset. The next launch recreates the profile and
    pays the interpreter grant once more; the SID is unchanged, because it is
    derived from the profile name. Refused while any launch is live: it would
    revoke the interpreter grant under a running container and delete the profile
    directory that launch's temp lives in.

    The check and the removal are one critical section, under the ledger the
    launch path holds across its own manifest and grants. Checking outside it let
    a launch start in between and lose the profile it had just been prepared
    with. A launch of another Studio process of this installation is just as
    live, is seen through its manifest, and is ordered against by the same ledger
    rather than only looked for.

    ``all_installations`` removes the containers of installations this process is
    not, and one installation's ledger orders nothing against the processes that
    own another's. Every moniker about to be touched is therefore locked, in
    sorted order so two concurrent resets cannot take two of them in opposite
    orders, and the whole reset is refused if any one of them cannot be taken.
    """
    global _INSTALL_PROFILE
    removed: list[str] = []
    moniker = _install_moniker()
    root = _manifest_root()
    locked = {moniker}
    if all_installations:
        locked |= _recorded_monikers(root, _INSTALL_PREFIX, _MANIFEST_KIND_PERSISTENT)
    with ExitStack() as stack:
        for name in sorted(locked):
            stack.enter_context(_destructive_ledger(name, f"releasing the container of {name}"))
        # Before the liveness check, not after it. A cleanup the ledger refused
        # keeps its hold and its manifest, so an undrained one looks exactly like
        # a running tool call and would refuse this release for good.
        for name in sorted(locked):
            _drain_deferred_cleanups(name)
        if _held_shared_grants() or _live_launch_holds(
            moniker = None if all_installations else moniker
        ):
            raise SandboxUnavailableError(
                "a sandboxed tool call is still running; its container cannot be released"
            )
        with _PERSISTENT_GRANT_LOCK:
            for manifest in sorted(Path(root).glob(_INSTALL_PREFIX + "*.json")):
                payload = _parse_manifest(manifest)
                if payload is None or payload["kind"] != _MANIFEST_KIND_PERSISTENT:
                    continue
                if not all_installations and payload["moniker"] != moniker:
                    continue
                if payload["moniker"] not in locked:
                    # Recorded after the monikers were enumerated, so no ledger of
                    # ours covers it. The next reset sees it under its own lock.
                    logger.warning(
                        "The container of %s appeared after this reset took its ledgers and was "
                        "left alone",
                        payload["moniker"],
                    )
                    continue
                _revoke_persistent_manifest(manifest, payload)
                result = ctypes.c_uint32(
                    _api().userenv.DeleteAppContainerProfile(payload["moniker"])
                ).value
                if result not in (0, 0x80070002):
                    raise OSError(f"DeleteAppContainerProfile: 0x{result:08x}")
                removed.append(payload["moniker"])
    with _INSTALL_PROFILE_LOCK:
        if _INSTALL_PROFILE is not None and (
            all_installations or _INSTALL_PROFILE.moniker == moniker
        ):
            # The SID is deliberately not freed: a launch of this process may
            # still hold it. Deriving the name again returns the same value.
            _INSTALL_PROFILE = None
    with _ACCESS_MEMO_LOCK:
        _ACCESS_MEMO.clear()
    with _PERSISTENT_GRANT_LOCK:
        _UNVERIFIED_ROOTS.clear()
    return tuple(removed)


def _create_identity(
    install: _InstallProfile,
    workdir: str,
    *,
    already_granted: frozenset[str],
    profile: str = _PROFILE_LPAC,
) -> _InvocationIdentity:
    """The per-launch state inside the installation's container.

    The container temp, a write-ahead manifest, and a hold on every ACE (and on
    the temp) this launch shares with concurrent launches of the same
    installation.
    """
    launch_id = secrets.token_hex(16)
    private_temp = ""
    manifest_path = ""
    granted_roots: tuple[str, ...] = ()
    held = False
    try:
        temp_root = os.path.join(install.profile_folder, "Temp")
        os.makedirs(temp_root, mode = 0o700, exist_ok = True)
        # The container temp itself. A per-launch child of it cannot be made to
        # take effect: Windows builds the package environment for an AppContainer
        # process and overwrites TEMP/TMP with <profile>\Temp, so the child would
        # write here whatever the launch plan said. Shared with concurrent
        # launches of this installation (concurrent_launches_share_the_container)
        # and emptied when the last of them finishes.
        private_temp = temp_root
        _validated_private_temp(install.profile_folder, private_temp)
        manifest_path = os.path.join(_manifest_root(), _LAUNCH_PREFIX + launch_id + ".json")
        traverse_roots = tuple(
            path
            for path in _traverse_ancestors((workdir, private_temp))
            if os.path.normcase(path) not in already_granted
            and not _container_owned(path, install.profile_folder)
        )
        # The private temp is granted but not listed: it is deleted at cleanup,
        # and it is inside the package directory Windows ACLs for the container.
        granted_roots = (workdir, *traverse_roots)
        # The private temp is held too, so that a manifest naming it cannot make
        # a reconciliation delete a running launch's directory.
        shared_roots = (*granted_roots, private_temp)
        owner = _process_identity()
        if owner is None:
            raise SandboxUnavailableError("LPAC could not record its owning process identity")
        _hold_shared_grants(shared_roots)
        held = True
        identity = _InvocationIdentity(
            install.moniker,
            install.sid,
            install.sid_string,
            install.profile_folder,
            private_temp,
            manifest_path,
            granted_roots,
            traverse_roots,
            owner[0],
            owner[1],
            profile = profile,
            workdir = workdir,
            launch_id = launch_id,
            shared_roots = shared_roots,
        )
        _write_manifest(identity)
        return identity
    except Exception:
        with _SHARED_GRANTS_LOCK:
            # Empty when nothing else holds it, never remove: private_temp is the
            # container temp now, shared by every concurrent launch of this
            # installation, and the rmtree this used to do would delete the TEMP
            # Windows names for all of them. Nothing of this launch ever ran, so
            # the set of paths it was the last holder of is the whole claim it
            # has to give back; a launch that never took the hold gives back
            # nothing and leaves the directory exactly as it found it.
            released: set[str] = (
                _release_shared_grants((*granted_roots, private_temp)) if held else set()
            )
            if manifest_path:
                try:
                    os.unlink(manifest_path)
                except FileNotFoundError:
                    pass
            if private_temp and not _ledger_is_held(install.moniker):
                # Emptying the container temp is destructive, and prepare is
                # allowed to run without the ledger. Leave it to the next
                # reconciliation rather than clear it under another Studio.
                logger.warning(
                    "The sandbox ledger for %s is not held, so the container temp of a failed "
                    "launch was left for the next reconciliation",
                    install.moniker,
                )
            elif private_temp:
                try:
                    _release_private_temp(
                        install.profile_folder,
                        private_temp,
                        released = released,
                        elsewhere = _live_launch_holds(launch_id, moniker = install.moniker)
                        | _held_shared_grants(),
                    )
                except Exception:  # noqa: BLE001 - never replace the original failure
                    logger.warning(
                        "Could not release the LPAC private temp of a failed launch",
                        exc_info = True,
                    )
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


def _job_query_function(api: Any) -> Any:
    """``QueryInformationJobObject``, declared on first use, or ``None`` when absent."""
    query = getattr(getattr(api, "kernel32", None), "QueryInformationJobObject", None)
    if query is None:
        return None
    if getattr(query, "argtypes", None) is None:
        try:
            query.argtypes = [
                wintypes.HANDLE,
                ctypes.c_int,
                ctypes.c_void_p,
                wintypes.DWORD,
                ctypes.POINTER(wintypes.DWORD),
            ]
            query.restype = wintypes.BOOL
        except (AttributeError, TypeError):
            pass
    return query


def _job_active_processes(api: Any, handle: Any) -> int | None:
    """How many processes the job still holds, or ``None`` when that cannot be read."""
    query = _job_query_function(api)
    if query is None:
        return None
    info = _JOBOBJECT_BASIC_ACCOUNTING_INFORMATION()
    returned = wintypes.DWORD()
    if not query(
        handle,
        _JOB_OBJECT_BASIC_ACCOUNTING_INFORMATION,
        ctypes.byref(info),
        ctypes.sizeof(info),
        ctypes.byref(returned),
    ):
        return None
    return int(info.ActiveProcesses)


def _wait_for_job_drain(job: Any, timeout: float = _JOB_DRAIN_SECONDS) -> bool | None:
    """Wait, bounded, until the job holds no process.

    ``TerminateJobObject`` and ``KILL_ON_JOB_CLOSE`` only initiate termination.
    Revoking the workdir ACE and emptying the container temp before the children
    are actually gone races the kernel closing their handles.

    ``True`` when the job is empty, ``False`` when it still held a process at the
    deadline, and ``None`` when this host cannot be asked (no
    ``QueryInformationJobObject``). Only ``False`` is evidence of a child that
    outlived its job; the caller must not turn a host it cannot observe into a
    failure.
    """
    handle = getattr(job, "_handle", None)
    if not handle:
        return True
    api = _api()
    deadline = time.monotonic() + timeout
    delay = _JOB_DRAIN_FIRST_POLL_SECONDS
    while True:
        active = _job_active_processes(api, handle)
        if active == 0:
            return True
        if active is None:
            logger.warning("Could not observe the sandbox job draining; removal retries instead")
            return None
        if time.monotonic() >= deadline:
            logger.warning(
                "The sandbox job still held %d process(es) after %.1f s", active, timeout
            )
            return False
        time.sleep(delay)
        delay = min(delay * 2, _JOB_DRAIN_MAX_POLL_SECONDS)


def _close_after_drain(process: Any, job: Any) -> None:
    """Kill the job, wait for it to drain, then release the process and job handles.

    Both launchers need this, and for the same reason. Cleanup is LIFO, so this
    callback runs before the ACL revoke and the temp removal, and closing a
    kill-on-close job only starts the kill: without the wait, Required mode
    revokes the workdir ACE and empties the container temp while a child of the
    launch is still holding both, which is the asymmetry the Limited path did not
    have.

    A drain that provably did not finish is raised, so ``cleanup`` records it as
    a diagnostic instead of continuing in silence. The handles are released
    either way - holding them would leak the job and the process for the rest of
    Studio's life - and the grants are still revoked afterwards, because a child
    that outlived ``TerminateJobObject`` is a runaway that must not keep its
    access. A host that cannot be asked (``None``) is not a failure.
    """
    job.terminate()
    drained = _wait_for_job_drain(job)
    process.close()
    if drained is False:
        raise OSError(
            f"the sandbox job still held a process {_JOB_DRAIN_SECONDS:.0f} s after it was "
            "terminated; the ACL revoke and the private temp removal below ran while a child "
            "of this launch was still alive"
        )


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


def _job_object_with_limits(*, active_process_limit: int | None = None) -> _WindowsJob:
    """A kill-on-close Job Object carrying Studio's resource limits, with no process yet.

    Split from _create_job so the job can be attached at creation through
    PROC_THREAD_ATTRIBUTE_JOB_LIST, which leaves no window in which the child
    runs outside it.
    """
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


def _create_job(
    process_handle: wintypes.HANDLE, *, active_process_limit: int | None = None
) -> _WindowsJob:
    job = _job_object_with_limits(active_process_limit = active_process_limit)
    if not _api().kernel32.AssignProcessToJobObject(job._handle, process_handle):
        error = _winerror("AssignProcessToJobObject")
        job.close()
        raise error
    return job


def _spawn_lpac(
    prepared: PreparedSandboxLaunch, popen_kwargs: dict[str, Any], identity: _InvocationIdentity
) -> WindowsLpacProcess:
    """Create the child inside ``identity``'s container.

    ``identity.profile`` decides whether the ALL_APPLICATION_PACKAGES opt-out
    attribute is applied (less-privileged AppContainer) or omitted (plain
    AppContainer). Both carry zero capabilities; the profile is chosen once, by
    the live probe, and a user payload is never re-run under the weaker one.
    """
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

        less_privileged = identity.profile != _PROFILE_APPCONTAINER
        # The Job Object is created first and attached at process creation through
        # PROC_THREAD_ATTRIBUTE_JOB_LIST, so no window exists in which a child is
        # alive (even suspended) without a kill-on-close owner. Hosts without the
        # attribute fall back to AssignProcessToJobObject before the resume.
        job = _job_object_with_limits()
        job_handles = (wintypes.HANDLE * 1)(job._handle)
        attribute_count = (3 if less_privileged else 2) + 1
        size = ctypes.c_size_t()
        api.kernel32.InitializeProcThreadAttributeList(None, attribute_count, 0, ctypes.byref(size))
        if ctypes.get_last_error() != _ERROR_INSUFFICIENT_BUFFER or not size.value:
            raise _winerror("InitializeProcThreadAttributeList(size)")
        attribute_buffer = ctypes.create_string_buffer(size.value)
        attribute_list = ctypes.cast(attribute_buffer, ctypes.c_void_p)
        if not api.kernel32.InitializeProcThreadAttributeList(
            attribute_list, attribute_count, 0, ctypes.byref(size)
        ):
            raise _winerror("InitializeProcThreadAttributeList")
        attributes_initialized = True
        capabilities = _SECURITY_CAPABILITIES(identity.sid, None, 0, 0)
        policy = wintypes.DWORD(_PROCESS_CREATION_ALL_APPLICATION_PACKAGES_OPT_OUT)
        attributes = [
            (
                _PROC_THREAD_ATTRIBUTE_SECURITY_CAPABILITIES,
                ctypes.byref(capabilities),
                ctypes.sizeof(capabilities),
            ),
        ]
        if less_privileged:
            attributes.append(
                (
                    _PROC_THREAD_ATTRIBUTE_ALL_APPLICATION_PACKAGES_POLICY,
                    ctypes.byref(policy),
                    ctypes.sizeof(policy),
                )
            )
        attributes.append(
            (
                _PROC_THREAD_ATTRIBUTE_HANDLE_LIST,
                ctypes.byref(handles),
                ctypes.sizeof(handles),
            )
        )
        for key, value, value_size in attributes:
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
        job_attached_at_creation = bool(
            api.kernel32.UpdateProcThreadAttribute(
                attribute_list,
                0,
                _PROC_THREAD_ATTRIBUTE_JOB_LIST,
                ctypes.byref(job_handles),
                ctypes.sizeof(job_handles),
                None,
                None,
            )
        )
        if not job_attached_at_creation and ctypes.get_last_error() not in (
            _ERROR_NOT_SUPPORTED,
            _ERROR_INVALID_PARAMETER,
        ):
            raise _winerror("UpdateProcThreadAttribute(job list)")

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
        if not job_attached_at_creation and not api.kernel32.AssignProcessToJobObject(
            job._handle, process_info.hProcess
        ):
            raise _winerror("AssignProcessToJobObject")
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
        # Not process.close: closing the kill-on-close job only starts the kill,
        # and identity.cleanup below this in the LIFO order revokes the workdir
        # ACE and empties the container temp a surviving child is still using.
        prepared.cleanup_callbacks.append(lambda: _close_after_drain(process, job))
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
    git_dir = _trusted_git_directory()
    if git_dir and git_dir not in path_entries:
        path_entries.append(git_dir)
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


def _trusted_git_directory() -> str:
    """The trusted Git-for-Windows directory the host-side PATH builder keeps (#7317)."""
    try:
        from .tools import _resolve_trusted_windows_git
    except Exception:  # noqa: BLE001 - optional convenience, never a launch failure
        return ""
    try:
        directory, _extension = _resolve_trusted_windows_git()
    except Exception:  # noqa: BLE001
        return ""
    return directory if directory and os.path.isdir(directory) else ""


class _ProbeEndpoints(list):
    """Host endpoints the probe must not reach, plus host-side limitations."""

    limitations: tuple[str, ...] = ()


@contextmanager
def _probe_network_endpoints():
    """Prove host endpoints work, then reject any traffic from the LPAC probe.

    IPv4 loopback is mandatory. IPv6 is probed only when the host can bind
    ``::1``; a host without IPv6 records ``ipv6_unavailable_on_host`` instead of
    reporting the sandbox unavailable.
    """
    with ExitStack() as stack:
        servers = []
        endpoints = _ProbeEndpoints()
        control = secrets.token_bytes(32)
        for family, host in ((socket.AF_INET, "127.0.0.1"), (socket.AF_INET6, "::1")):
            if family == socket.AF_INET6:
                try:
                    with socket.socket(family, socket.SOCK_STREAM) as check:
                        check.bind((host, 0))
                except OSError:
                    endpoints.limitations = (_LIMITATION_IPV6,)
                    continue
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
        # WSAEACCES (10013): the operation was denied on a socket that exists.
        # WSAEPROVIDERFAILEDINIT (10106): the AppContainer could not even
        # initialize the Winsock provider, so no socket can be created at all.
        assert exc.winerror in (10013, 10106), ('unexpected network error', repr(exc))
    else:
        raise AssertionError('LPAC network operation was not denied')
    finally:
        if sock is not None:
            sock.close()
"""


def _probe_payload(
    workdir: str,
    external: str,
    expected_sid: str,
    endpoints: list[tuple],
    *,
    less_privileged: bool = True,
) -> str:
    return f"""import ctypes, os, socket, sys
from ctypes import wintypes
k = ctypes.WinDLL('kernel32', use_last_error=True)
a = ctypes.WinDLL('advapi32', use_last_error=True)
# Explicit signatures: the default c_int return truncates the 64-bit pseudo handle
# from GetCurrentProcess to 0xFFFFFFFF, which under an AppContainer's strict handle
# checks raises STATUS_INVALID_HANDLE (0xC0000008) instead of a clean failure.
k.GetCurrentProcess.restype = wintypes.HANDLE
a.OpenProcessToken.argtypes = [wintypes.HANDLE, wintypes.DWORD, ctypes.POINTER(wintypes.HANDLE)]
a.OpenProcessToken.restype = wintypes.BOOL
a.GetTokenInformation.argtypes = [wintypes.HANDLE, wintypes.DWORD, ctypes.c_void_p, wintypes.DWORD, ctypes.POINTER(wintypes.DWORD)]
a.GetTokenInformation.restype = wintypes.BOOL
token = wintypes.HANDLE()
assert a.OpenProcessToken(k.GetCurrentProcess(), 0x0008, ctypes.byref(token)), ctypes.get_last_error()
def token_dword(kind):
    value = wintypes.DWORD(); size = wintypes.DWORD(ctypes.sizeof(value))
    assert a.GetTokenInformation(token, kind, ctypes.byref(value), size, ctypes.byref(size)), (kind, ctypes.get_last_error())
    return value.value
assert token_dword(29) == 1
def token_dword_or_zero(kind):
    # TokenIsLessPrivilegedAppContainer is only answerable for LPAC tokens; a plain
    # AppContainer token reports ERROR_INVALID_PARAMETER (87), which means "not LPAC".
    value = wintypes.DWORD(); size = wintypes.DWORD(ctypes.sizeof(value))
    if a.GetTokenInformation(token, kind, ctypes.byref(value), size, ctypes.byref(size)):
        return value.value
    assert ctypes.get_last_error() == 87, (kind, ctypes.get_last_error())
    return 0
if {less_privileged!r}:
    assert token_dword(46) == 1
else:
    assert token_dword_or_zero(46) == 0
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

    def __init__(self) -> None:
        self._profile = _PROFILE_LPAC
        self._last_probe_limitations: tuple[str, ...] = ()

    @property
    def active_profile(self) -> str:
        """``lpac`` or ``appcontainer``: the container kind the last probe qualified."""
        return self._profile

    def active_profile_id(self) -> str:
        return _APPCONTAINER_PROFILE_ID if self._profile == _PROFILE_APPCONTAINER else _PROFILE_ID

    def probe(self) -> SandboxCapability:
        if not _is_windows():
            return SandboxCapability(self.identity, False, "LPAC requires Windows", available = False)
        # Every probe starts from the strongest profile; a previous fallback never sticks
        # once the environment fingerprint moved. ``profile_id`` shadows the class default
        # on the instance so callers reading ``backend.profile_id`` see the active kind.
        self._profile = _PROFILE_LPAC
        self.profile_id = _PROFILE_ID
        try:
            _api()
            self.reconcile_stale_manifests()
            outcome = self._probe_profile(_PROFILE_LPAC)
            if outcome is None:
                return SandboxCapability(
                    self.identity,
                    True,
                    "zero-capability LPAC live enforcement probe passed",
                    available = True,
                    protection_state = "preview",
                    profile_id = _PROFILE_ID,
                    limitations = (
                        _LIMITATION_NULL_DEVICE_PIPES,
                        _LIMITATION_SHARED_CONTAINER,
                        *self._last_probe_limitations,
                    ),
                )
            returncode, reason = outcome
            if returncode in (_STATUS_ACCESS_DENIED, _STATUS_DLL_NOT_FOUND):
                fallback = self._probe_profile(_PROFILE_APPCONTAINER)
                if fallback is None:
                    self._profile = _PROFILE_APPCONTAINER
                    self.profile_id = _APPCONTAINER_PROFILE_ID
                    return SandboxCapability(
                        self.identity,
                        True,
                        "zero-capability LPAC could not start the selected interpreter "
                        f"({returncode}); the zero-capability AppContainer fallback passed its live "
                        "enforcement probe. Files shared with all application packages (Program "
                        "Files, Windows) are readable inside it; the user profile, the network and "
                        "every path outside the workdir stay denied",
                        available = True,
                        protection_state = "preview",
                        profile_id = _APPCONTAINER_PROFILE_ID,
                        limitations = (
                            _LIMITATION_AMBIENT_READ,
                            _LIMITATION_NULL_DEVICE_PIPES,
                            _LIMITATION_SHARED_CONTAINER,
                            *self._last_probe_limitations,
                        ),
                    )
                reason = f"{reason}; the AppContainer fallback probe also failed: {fallback[1]}"
            return SandboxCapability(self.identity, False, reason, available = False)
        except Exception as exc:  # noqa: BLE001 - capability failure blocks Required mode
            return SandboxCapability(
                self.identity,
                False,
                f"the LPAC live probe could not run: {exc}",
                available = False,
            )

    def _probe_profile(self, profile: str) -> tuple[int | None, str] | None:
        """Run the live enforcement probe under ``profile``.

        Returns ``None`` on success, else ``(returncode, reason)``. Infrastructure
        failures raise and are reported by the caller.
        """
        less_privileged = profile != _PROFILE_APPCONTAINER
        label = "LPAC" if less_privileged else "AppContainer"
        with (
            tempfile.TemporaryDirectory(prefix = "unsloth-lpac-probe-") as base,
            _probe_network_endpoints() as endpoints,
        ):
            self._last_probe_limitations = tuple(getattr(endpoints, "limitations", ()))
            workdir = os.path.join(base, "work")
            os.mkdir(workdir)
            Path(workdir, "probe-read").write_text("readable", encoding = "utf-8")
            external = os.path.join(base, "host-secret")
            Path(external).write_text("secret", encoding = "utf-8")
            prepared = self._prepare(
                ToolLaunchPlan(
                    argv = (sys.executable, "-I", "-S", "-c", ""),
                    workdir = workdir,
                    env = {"PYTHONIOENCODING": "utf-8"},
                ),
                profile,
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
                    _probe_payload(
                        workdir,
                        external,
                        identity.sid_string,
                        list(endpoints),
                        less_privileged = less_privileged,
                    ),
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
                    if process.returncode == _STATUS_ACCESS_DENIED:
                        detail = (
                            "the selected interpreter could not initialize under zero-capability "
                            f"{label} (STATUS_ACCESS_DENIED); capabilities were not widened"
                        )
                    elif process.returncode == _STATUS_DLL_NOT_FOUND:
                        detail = (
                            "the selected interpreter could not load its runtime libraries under "
                            f"zero-capability {label} (STATUS_DLL_NOT_FOUND)"
                        )
                    else:
                        detail = output[-400:]
                    if identity.unverified_access:
                        detail += (
                            "; access to these paths could not be granted and was left to the "
                            "existing ACLs: " + ", ".join(identity.unverified_access)
                        )
                    return (
                        process.returncode,
                        f"the {label} live probe failed ({process.returncode}): {detail}",
                    )
            finally:
                prepared.cleanup()
                if prepared.cleanup_diagnostics:
                    raise SandboxUnavailableError(f"{label} probe cleanup failed")
        return None

    def prepare(self, spec: ToolLaunchPlan) -> PreparedSandboxLaunch:
        return self._prepare(spec, self._profile)

    def prepare_for_profile(self, spec: ToolLaunchPlan, profile_id: str) -> PreparedSandboxLaunch:
        """Prepare the profile a capability was recorded with, not whatever a later probe chose."""
        try:
            profile = _PROFILE_BY_ID[profile_id]
        except KeyError as exc:
            raise SandboxUnavailableError(
                f"the recorded Windows isolation profile is not available: {profile_id}"
            ) from exc
        return self._prepare(spec, profile)

    def _prepare(self, spec: ToolLaunchPlan, profile: str) -> PreparedSandboxLaunch:
        workdir = _validate_workdir(spec.workdir)
        argv = _canonical_inner_argv(spec.argv, spec.env)
        runtime_roots = _runtime_roots(workdir, argv)
        acl_runtime_roots = tuple(path for path in runtime_roots if _needs_explicit_acl(path))
        install = _install_profile()
        # Held from the moment this launch takes its share of the workdir and its
        # ancestors until the last of them is granted, so a concurrent cleanup -
        # in this process or in another Studio of this installation - cannot
        # revoke an ACE between the check that found it and its use.
        with _installation_ledger(install.moniker) as held:
            if held:
                # The one place a deferred cleanup is cheap to retry: this
                # section already holds the ledger it was refused, and it runs on
                # every tool call. Before the grants below, so the launch that
                # follows re-grants anything the retry revoked.
                _drain_deferred_cleanups(install.moniker)
            else:
                # Granting is not destructive: an ACE this launch adds under a
                # concurrent cleanup is re-added by the check-then-grant below,
                # and refusing here would fail a tool call over housekeeping.
                logger.warning(
                    "The sandbox ledger for %s could not be taken; this launch grants without it "
                    "and its cleanup is deferred rather than run unsynchronised",
                    install.moniker,
                )
            # The runtime grant belongs to the installation and survives this
            # launch; only the workdir, the private temp and their remaining
            # ancestors are granted and revoked here.
            persistent = _ensure_persistent_grants(install, acl_runtime_roots, profile)
            identity = _create_identity(
                install, workdir, already_granted = persistent.paths, profile = profile
            )
            try:
                unverified: list[str] = list(persistent.unverified)
                with _well_known_sid(_ambient_sid_text(profile)) as ambient:
                    sids = (identity.sid, ambient)
                    _grant_modify(workdir, identity.sid)
                    _grant_modify(identity.private_temp, identity.sid)
                    for root in identity.traverse_roots:
                        # Deliberately not memoized: this ACE is revoked when the
                        # last launch holding it finishes, so an answer from an
                        # earlier launch would be stale.
                        if _existing_access(root, sids, _FILE_TRAVERSE):
                            continue
                        try:
                            _grant_traverse(root, identity.sid)
                        except OSError as exc:
                            if exc.errno == _ERROR_ACCESS_DENIED and _machine_wide(root):
                                unverified.append(root)
                                continue
                            raise
                identity.unverified_access = tuple(unverified)

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

    def remove_persistent_grants(self, *, all_installations: bool = False) -> tuple[str, ...]:
        """Release everything this installation holds outside a single launch.

        The interpreter tree grant and the container profile itself. Intended for
        uninstall or a deliberate reset; a launch pays the grant again afterwards.
        Returns the profile names that were released.
        """
        return _remove_persistent_grants(all_installations = all_installations)

    def reconcile_stale_manifests(self) -> None:
        """Release what a crashed Studio left behind, and nothing that is still in use.

        Per-launch grants of a dead owner are revoked and the container temp is
        emptied when no live launch is using it. The installation-wide grant has
        no owning process by design: it
        is kept while the interpreter it was made for still exists, and revoked
        when it does not, which is what an uninstalled or relocated runtime looks
        like from here.
        """
        root = _manifest_root()
        _remove_orphan_temporary_manifests(root)
        self._reconcile_pass(root)

    def _reconcile_pass(self, root: str) -> None:
        """One reconciliation sweep, each installation under its own ledger.

        A pass reads the manifests of every installation on this account, not
        only this one's, and the ledger of *this* installation orders nothing
        against the Studio processes that own another's: it would revoke their
        ACEs and empty their container temp with no lock between us at all. The
        moniker is in every manifest, so each record is handled under the ledger
        of the installation it belongs to.

        One critical section per installation, in this process and in every other
        Studio of it: a launch that starts between the scan that found a record
        dead and the revoke it gates must not lose an ACE, and a second
        reconciliation must not revoke the same paths twice. An installation
        whose ledger is busy keeps its records for the next pass.
        """
        # One parse of the directory for the whole pass, grouped by the
        # installation each record names, rather than one re-parse of every
        # manifest per installation found.
        grouped: dict[str, list[Path]] = {}
        for manifest in sorted(Path(root).glob(_PROFILE_PREFIX + "*.json")):
            payload = _parse_manifest(manifest)
            if payload is not None:
                grouped.setdefault(payload["moniker"], []).append(manifest)
        for moniker in sorted(grouped):
            # Asked for, never waited on: this runs at every probe, over every
            # installation on the account, and a busy one is handled next pass.
            with _installation_ledger(moniker, wait_seconds = 0.0) as held:
                if not held:
                    logger.warning(
                        "The sandbox ledger for %s could not be taken; its stale records are kept "
                        "for the next reconciliation rather than released unsynchronised",
                        moniker,
                    )
                    continue
                _drain_deferred_cleanups(moniker)
                self._reconcile_installation(grouped[moniker], moniker)

    def _reconcile_installation(self, manifests: list[Path], moniker: str) -> None:
        """The records of one installation, under the ledger its caller holds.

        Re-read here rather than carried in from the grouping pass: the grouping
        ran before this installation's ledger was taken, so only a payload read
        inside it is one no other Studio of this installation could have replaced.
        """
        for manifest in manifests:
            try:
                payload = _parse_manifest(manifest)
                if payload is None or payload["moniker"] != moniker:
                    continue
                if payload["kind"] == _MANIFEST_KIND_PERSISTENT:
                    # Another installation's grant is not ours to release; a
                    # record of *this* interpreter under a moniker that is no
                    # longer ours is this installation renamed, and only this
                    # code will ever release it.
                    superseded = (
                        payload["moniker"] != _install_moniker()
                        and os.path.normcase(payload["interpreter"])
                        == os.path.normcase(os.path.realpath(sys.executable))
                    )
                    if os.path.exists(payload["interpreter"]) and not superseded:
                        continue
                    _revoke_persistent_manifest(manifest, payload)
                    continue
                owner = (payload["owner_pid"], payload["owner_created"])
                if _process_identity(owner[0]) == owner:
                    continue
                self._reconcile_launch_manifest(manifest, payload)
            except Exception:
                # A stale record is retained for the next startup; never delete
                # evidence or reuse its identity after partial reconciliation.
                logger.warning(
                    "Could not reconcile the LPAC manifest %s", manifest, exc_info = True
                )
                continue

    def _reconcile_launch_manifest(self, manifest: Path, payload: dict[str, Any]) -> None:
        """Revoke one dead launch's grants and empty the container temp it named.

        A path a live launch of this process is holding is left alone: launches
        of one installation share a SID, and a workdir is per chat session, so a
        crashed Studio's manifest can name the directory a running call is using.
        That launch revokes it when it finishes, and a manifest that names a
        running launch's private temp is skipped entirely. A live launch of
        another Studio process of this installation is seen through its own
        manifest, and the ledger the caller holds keeps one from appearing
        between this scan and the revokes it gates.
        """
        single_use = payload["kind"] == _MANIFEST_KIND_SINGLE_USE
        held = _held_shared_grants() | _live_launch_holds(
            payload.get("launch_id", ""), moniker = payload["moniker"]
        )
        if os.path.normcase(payload["private_temp"]) in held:
            return  # a running launch of this process owns it
        owned: set[str] = set()
        if not single_use:
            # A launch never grants a runtime root, so it never releases one. A
            # planted manifest that names the interpreter tree would otherwise
            # revoke the installation's own grant out from under every container.
            record = _read_persistent_manifest(payload["moniker"]) or {}
            owned = {
                os.path.normcase(path)
                for path in (
                    *record.get("granted_roots", ()),
                    *record.get("traverse_roots", ()),
                )
            }
        granted_roots = tuple(
            path
            for path in payload["granted_roots"]
            if os.path.normcase(path) not in held and os.path.normcase(path) not in owned
        )
        sid = _derive_container_sid(payload["moniker"])
        try:
            if _sid_string(_api(), sid) != payload["sid"]:
                raise SandboxUnavailableError("the LPAC manifest SID does not match its moniker")
            derived_profile = _profile_folder(_api(), payload["sid"])
            if os.path.normcase(os.path.realpath(payload["profile_folder"])) != os.path.normcase(
                derived_profile
            ):
                raise SandboxUnavailableError(
                    "the LPAC manifest profile folder does not match its SID"
                )
            _validated_private_temp(derived_profile, payload["private_temp"])
            identity = _InvocationIdentity(
                payload["moniker"],
                sid,
                payload["sid"],
                derived_profile,
                payload["private_temp"],
                str(manifest),
                granted_roots,
                tuple(payload["traverse_roots"]),
                payload["owner_pid"],
                payload["owner_created"],
                workdir = payload.get("workdir", ""),
                launch_id = payload.get("launch_id", ""),
                # Only a single-use manifest owns the profile it names. The
                # shared one outlives every launch of the installation, and the
                # SID here was derived locally, so it is this call's to free.
                delete_profile = single_use,
                free_sid = True,
            )
        except BaseException:
            _api().advapi32.FreeSid(sid)
            raise
        try:
            # Never deferred: the identity here owns a SID this call frees below,
            # so queueing it would leave a freed pointer in the retry list. The
            # caller already holds this installation's ledger anyway.
            identity.cleanup(defer_when_locked = False)
        finally:
            # cleanup frees the SID only on the path that completes. A revoke or
            # an unremovable temp raises before that, and this manifest is retried
            # at every probe, so the allocation would leak once per probe.
            if not identity.cleaned and identity.sid:
                _api().advapi32.FreeSid(identity.sid)
                identity.sid = ctypes.c_void_p()


__all__ = ["WindowsLpacBackend", "WindowsLpacProcess"]
