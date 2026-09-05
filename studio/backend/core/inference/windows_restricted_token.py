# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Write-restricted token launcher for Studio's Limited mode on Windows.

The zero-capability AppContainer (``windows_lpac``) is the Required-mode
sandbox. It cannot read anything it was not granted, which is exactly what
Required promises, and exactly why Limited mode cannot use it: a Python
installed under Program Files or the user profile only starts after tens of
seconds of ACL grants, ``NUL`` and named pipes are denied, and Git bash cannot
initialise. Limited mode instead follows the Codex and DeepSeek harnesses and
runs the tool under a *restricted token* of the Studio user:

* ``DISABLE_MAX_PRIVILEGE`` drops every privilege but ``SeChangeNotify``;
* ``LUA_TOKEN`` removes administrative membership when Studio runs elevated;
* ``WRITE_RESTRICTED`` adds a second access check for writes only. A write
  succeeds only when the object also grants one of the *restricting SIDs*:
  a per-launch random SID (granted on the workdir and a private temp
  directory), the logon SID and Everyone. Both of the latter are required for
  process initialisation (window station, ``\\BaseNamedObjects``); they also mean
  objects that already grant Everyone or the session stay writable, which the
  execution record discloses.

Reads and execution keep the user's normal access, so the selected interpreter
runs from wherever it is installed, ``open(os.devnull)`` works, named pipes and
``multiprocessing`` work, and the network stays reachable. None of that is OS
isolation; the record says ``os_isolation = False`` and lists the limitations.

Every launch is bound to a kill-on-close Job Object carrying Studio's resource
limits, attached at creation through ``PROC_THREAD_ATTRIBUTE_JOB_LIST`` (or
before the first instruction runs on hosts without it). Ownership of the ACL
grants and the private temp is recorded in a write-ahead manifest under
``%LOCALAPPDATA%\\Unsloth\\Studio\\limited-manifests`` so a crashed Studio leaves
nothing behind that the next start does not reconcile.
"""

from __future__ import annotations

import ctypes
from ctypes import wintypes
from dataclasses import dataclass
import json
import os
from pathlib import Path
import secrets
import shutil
import subprocess
import sys
import threading
from typing import Any

from loggers import get_logger

from .os_sandbox import (
    PreparedSandboxLaunch,
    SandboxCapability,
    SandboxUnavailableError,
    ToolLaunchPlan,
)
from . import windows_lpac as _lpac

logger = get_logger(__name__)

_BACKEND_IDENTITY = "windows-restricted-token"
_PROFILE_ID = "windows-restricted-token-write-isolation-v1"
_MANIFEST_PREFIX = "unsloth.studio.limited."

# Disclosed on every record produced by this launcher.
_LIMITATION_USER_PROFILE_READABLE = "user_profile_readable"
_LIMITATION_NETWORK_UNRESTRICTED = "network_unrestricted"
_LIMITATION_EVERYONE_WRITABLE = "everyone_writable_objects_writable"
# Disclosed by os_sandbox when a Limited launch on Windows could not use this
# launcher for that one call (for example a workdir too large to ACL-scan) and
# ran under the process guard alone instead.
_LIMITATION_TOKEN_UNAVAILABLE = "restricted_token_unavailable"
_LIMITATIONS = (
    _LIMITATION_USER_PROFILE_READABLE,
    _LIMITATION_NETWORK_UNRESTRICTED,
    _LIMITATION_EVERYONE_WRITABLE,
)

_TOKEN_ASSIGN_PRIMARY = 0x0001
_TOKEN_DUPLICATE = 0x0002
_TOKEN_QUERY = 0x0008
_TOKEN_ADJUST_DEFAULT = 0x0080
_TOKEN_ACCESS = _TOKEN_ASSIGN_PRIMARY | _TOKEN_DUPLICATE | _TOKEN_QUERY | _TOKEN_ADJUST_DEFAULT

_DISABLE_MAX_PRIVILEGE = 0x1
_LUA_TOKEN = 0x4
_WRITE_RESTRICTED = 0x8
_RESTRICTED_TOKEN_FLAGS = _DISABLE_MAX_PRIVILEGE | _LUA_TOKEN | _WRITE_RESTRICTED

_TOKEN_GROUPS = 2
_TOKEN_PRIVILEGES = 3
_TOKEN_DEFAULT_DACL = 6
_TOKEN_RESTRICTED_SIDS = 11
_TOKEN_LOGON_SID = 28

_EVERYONE_SID = "S-1-1-0"
_ADMINISTRATORS_SID = "S-1-5-32-544"
_NO_INHERITANCE = 0
_CREATE_NO_WINDOW = 0x08000000
_ERROR_INSUFFICIENT_BUFFER = 122
_ERROR_NOT_SUPPORTED = 50
_ERROR_INVALID_PARAMETER = 87
_MAX_SIBLING_SCAN = 100_000


class _SID_AND_ATTRIBUTES(ctypes.Structure):
    _fields_ = [("Sid", ctypes.c_void_p), ("Attributes", wintypes.DWORD)]


class _TOKEN_GROUPS_HEADER(ctypes.Structure):
    _fields_ = [("GroupCount", wintypes.DWORD), ("Groups", _SID_AND_ATTRIBUTES * 1)]


class _TOKEN_DEFAULT_DACL(ctypes.Structure):
    _fields_ = [("DefaultDacl", ctypes.c_void_p)]


def _is_windows() -> bool:
    return os.name == "nt"


def _random_domain_sid_text() -> str:
    """A never-assigned ``S-1-5-21`` SID that names exactly one launch (Codex style)."""
    return "S-1-5-21-" + "-".join(str(secrets.randbelow(1 << 32)) for _ in range(4))


def _is_launch_sid_text(text: object) -> bool:
    if not isinstance(text, str) or not text.startswith("S-1-5-21-"):
        return False
    parts = text.split("-")
    return len(parts) == 8 and all(part.isdigit() for part in parts[3:])


def _private_root(name: str) -> str:
    """``%LOCALAPPDATA%\\Unsloth\\Studio\\<name>``, created 0700 and checked for reparse points."""
    local = os.environ.get("LOCALAPPDATA")
    if not local or not os.path.isabs(local):
        raise SandboxUnavailableError(
            "LOCALAPPDATA is unavailable for the Limited mode private directories"
        )
    local = os.path.realpath(local)
    spelled = os.path.join(local, "Unsloth", "Studio", name)
    os.makedirs(spelled, mode = 0o700, exist_ok = True)
    root = os.path.realpath(spelled)
    if not _lpac._is_within(root, local):
        raise SandboxUnavailableError("a Limited mode private directory escapes LOCALAPPDATA")
    if getattr(os.lstat(spelled), "st_file_attributes", 0) & 0x400:
        raise SandboxUnavailableError("a Limited mode private directory is a reparse point")
    return root


def _manifest_root() -> str:
    return _private_root("limited-manifests")


def _temp_root() -> str:
    return _private_root("limited-temp")


def _validated_private_temp(private_temp: str) -> str:
    """The private temp of one launch, refused unless it is a plain child of the temp root."""
    spelled = os.path.abspath(private_temp)
    root = _temp_root()
    name = os.path.basename(spelled)
    if (
        os.path.normcase(os.path.dirname(spelled)) != os.path.normcase(root)
        or len(name) != 24
        or not all(character in "0123456789abcdef" for character in name.lower())
    ):
        raise SandboxUnavailableError("a Limited mode private temp path is outside its root")
    if os.path.lexists(spelled) and getattr(os.lstat(spelled), "st_file_attributes", 0) & 0x400:
        raise SandboxUnavailableError("the Limited mode private temp is a reparse point")
    if os.path.isdir(spelled):
        entries = 0
        for base, dirs, names in os.walk(spelled, followlinks = False):
            for entry in [*dirs, *names]:
                entries += 1
                if entries > _MAX_SIBLING_SCAN:
                    raise SandboxUnavailableError("the Limited mode private temp is too large")
                path = os.path.join(base, entry)
                if getattr(os.lstat(path), "st_file_attributes", 0) & 0x400:
                    raise SandboxUnavailableError(
                        f"the Limited mode private temp contains a reparse point: {path}"
                    )
    return spelled


@dataclass
class _LaunchIdentity:
    """One launch's random SID plus everything granted to it (write-ahead recorded)."""

    sid: ctypes.c_void_p
    sid_string: str
    workdir: str
    private_temp: str
    manifest_path: str
    granted_roots: tuple[str, ...]
    owner_pid: int
    owner_created: int
    cleaned: bool = False

    def cleanup(self) -> None:
        if self.cleaned:
            return
        errors: list[str] = []
        for path in reversed(self.granted_roots):
            try:
                _lpac._revoke_sid(path, self.sid)
            except Exception as exc:  # noqa: BLE001 - continue ownership cleanup
                errors.append(f"ACL {path}: {exc}")
        try:
            shutil.rmtree(_validated_private_temp(self.private_temp), ignore_errors = False)
        except FileNotFoundError:
            pass
        except Exception as exc:  # noqa: BLE001
            errors.append(f"temp {self.private_temp}: {exc}")
        if errors:
            raise OSError("; ".join(errors))
        try:
            os.unlink(self.manifest_path)
        except FileNotFoundError:
            pass
        except OSError as exc:
            raise OSError(f"manifest: {exc}") from exc
        if self.sid:
            _lpac._api().kernel32.LocalFree(self.sid)
        self.sid = ctypes.c_void_p()
        self.cleaned = True


def _write_manifest(identity: _LaunchIdentity) -> None:
    payload = {
        "version": 1,
        "kind": "restricted-token",
        "sid": identity.sid_string,
        "workdir": identity.workdir,
        "private_temp": identity.private_temp,
        "granted_roots": list(identity.granted_roots),
        "owner_pid": identity.owner_pid,
        "owner_created": identity.owner_created,
    }
    temporary = identity.manifest_path + ".tmp"
    with open(temporary, "x", encoding = "utf-8") as stream:
        json.dump(payload, stream, sort_keys = True)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, identity.manifest_path)


def _parse_manifest(manifest: Path) -> dict[str, Any] | None:
    """The validated payload of a manifest, or ``None`` when it is not one of ours."""
    try:
        payload = json.loads(manifest.read_text(encoding = "utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    sid_text = payload.get("sid")
    roots = payload.get("granted_roots")
    private_temp = payload.get("private_temp")
    workdir = payload.get("workdir")
    if (
        payload.get("version") != 1
        or payload.get("kind") != "restricted-token"
        or not manifest.name.startswith(_MANIFEST_PREFIX)
        or not _is_launch_sid_text(sid_text)
        or not isinstance(roots, list)
        or not all(isinstance(path, str) and os.path.isabs(path) for path in roots)
        or not isinstance(private_temp, str)
        or not os.path.isabs(private_temp)
        or not isinstance(workdir, str)
        or not os.path.isabs(workdir)
        or not isinstance(payload.get("owner_pid"), int)
        or not isinstance(payload.get("owner_created"), int)
    ):
        return None
    return payload


def _sid_from_text(text: str) -> ctypes.c_void_p:
    api = _lpac._api()
    sid = ctypes.c_void_p()
    if not api.advapi32.ConvertStringSidToSidW(text, ctypes.byref(sid)) or not sid:
        raise _lpac._winerror(f"ConvertStringSidToSidW({text})")
    return sid


def _create_identity(workdir: str) -> _LaunchIdentity:
    """Allocate a launch SID, its private temp and manifest, then grant the SID its writes."""
    api = _lpac._api()
    sid_text = _random_domain_sid_text()
    sid = _sid_from_text(sid_text)
    private_temp = ""
    manifest_path = ""
    identity: _LaunchIdentity | None = None
    try:
        private_temp = os.path.join(_temp_root(), secrets.token_hex(12))
        os.makedirs(private_temp, mode = 0o700)
        _validated_private_temp(private_temp)
        manifest_path = os.path.join(_manifest_root(), _MANIFEST_PREFIX + sid_text + ".json")
        owner = _lpac._process_identity()
        if owner is None:
            raise SandboxUnavailableError(
                "the Limited launcher could not record its owning process identity"
            )
        identity = _LaunchIdentity(
            sid,
            sid_text,
            workdir,
            private_temp,
            manifest_path,
            (workdir, private_temp),
            owner[0],
            owner[1],
        )
        _write_manifest(identity)
        for root in identity.granted_roots:
            _lpac._grant_modify(root, sid)
        return identity
    except BaseException:
        if identity is not None:
            try:
                identity.cleanup()
            except Exception:  # noqa: BLE001 - the manifest keeps the record for reconciliation
                pass
        else:
            if manifest_path:
                try:
                    os.unlink(manifest_path)
                except FileNotFoundError:
                    pass
            if private_temp:
                shutil.rmtree(private_temp, ignore_errors = True)
            api.kernel32.LocalFree(sid)
        raise


def _token_information(api: Any, token: wintypes.HANDLE, kind: int) -> ctypes.Array[Any]:
    size = wintypes.DWORD()
    api.advapi32.GetTokenInformation(token, kind, None, 0, ctypes.byref(size))
    if ctypes.get_last_error() != _ERROR_INSUFFICIENT_BUFFER or not size.value:
        raise _lpac._winerror(f"GetTokenInformation({kind}, size)")
    buffer = ctypes.create_string_buffer(size.value)
    if not api.advapi32.GetTokenInformation(token, kind, buffer, size.value, ctypes.byref(size)):
        raise _lpac._winerror(f"GetTokenInformation({kind})")
    return buffer


def _token_group_sids(api: Any, token: wintypes.HANDLE, kind: int) -> list[str]:
    """The SID strings of a ``TOKEN_GROUPS`` token class (logon SID, restricted SIDs)."""
    try:
        buffer = _token_information(api, token, kind)
    except OSError:
        return []
    count = ctypes.cast(buffer, ctypes.POINTER(wintypes.DWORD))[0]
    base = ctypes.addressof(buffer) + _TOKEN_GROUPS_HEADER.Groups.offset
    stride = ctypes.sizeof(_SID_AND_ATTRIBUTES)
    texts: list[str] = []
    for index in range(int(count)):
        entry = _SID_AND_ATTRIBUTES.from_address(base + index * stride)
        if entry.Sid:
            texts.append(_lpac._sid_string(api, ctypes.c_void_p(entry.Sid)))
    return texts


def _set_default_dacl(api: Any, token: wintypes.HANDLE, sid: ctypes.c_void_p) -> None:
    """Add the launch SID to the token's default DACL.

    Objects the child creates without an explicit descriptor (anonymous pipes,
    events, sections) get this DACL. The write-restricted check needs a
    restricting SID on them, otherwise the child could not even write to a pipe
    it just created.
    """
    buffer = _token_information(api, token, _TOKEN_DEFAULT_DACL)
    current = ctypes.cast(buffer, ctypes.POINTER(_TOKEN_DEFAULT_DACL))[0].DefaultDacl
    trustee = _lpac._TRUSTEE_W(
        None,
        _lpac._NO_MULTIPLE_TRUSTEE,
        _lpac._TRUSTEE_IS_SID,
        _lpac._TRUSTEE_IS_UNKNOWN,
        ctypes.cast(sid, wintypes.LPWSTR),
    )
    entry = _lpac._EXPLICIT_ACCESS_W(_lpac._GENERIC_ALL, _lpac._GRANT_ACCESS, _NO_INHERITANCE, trustee)
    new_acl = ctypes.c_void_p()
    result = api.advapi32.SetEntriesInAclW(
        1,
        ctypes.byref(entry),
        ctypes.c_void_p(current) if current else None,
        ctypes.byref(new_acl),
    )
    if result != 0:
        raise _lpac._winerror("SetEntriesInAclW(default DACL)", result)
    try:
        default_dacl = _TOKEN_DEFAULT_DACL(new_acl)
        if not api.advapi32.SetTokenInformation(
            token,
            _TOKEN_DEFAULT_DACL,
            ctypes.byref(default_dacl),
            ctypes.sizeof(default_dacl),
        ):
            raise _lpac._winerror("SetTokenInformation(TokenDefaultDacl)")
    finally:
        api.kernel32.LocalFree(new_acl)


def _create_restricted_token(identity: _LaunchIdentity) -> wintypes.HANDLE:
    """A write-restricted, LUA, privilege-stripped copy of Studio's own primary token."""
    api = _lpac._api()
    source = wintypes.HANDLE()
    if not api.advapi32.OpenProcessToken(
        api.kernel32.GetCurrentProcess(), _TOKEN_ACCESS, ctypes.byref(source)
    ):
        raise _lpac._winerror("OpenProcessToken")
    restricted = wintypes.HANDLE()
    owned: list[ctypes.c_void_p] = []
    try:
        logon_sids = _token_group_sids(api, source, _TOKEN_LOGON_SID)
        restrict_texts = [identity.sid_string, *logon_sids[:1], _EVERYONE_SID]
        restrict = (_SID_AND_ATTRIBUTES * len(restrict_texts))()
        for index, text in enumerate(restrict_texts):
            sid = _sid_from_text(text)
            owned.append(sid)
            restrict[index].Sid = sid.value
            restrict[index].Attributes = 0
        # LUA_TOKEN already turns Administrators into a deny-only group; disabling
        # it explicitly (only when the token carries it) keeps that true on hosts
        # where the LUA filtering is a no-op.
        disable = (_SID_AND_ATTRIBUTES * 1)()
        disable_count = 0
        if _ADMINISTRATORS_SID in _token_group_sids(api, source, _TOKEN_GROUPS):
            administrators = _sid_from_text(_ADMINISTRATORS_SID)
            owned.append(administrators)
            disable[0].Sid = administrators.value
            disable[0].Attributes = 0
            disable_count = 1
        if not api.advapi32.CreateRestrictedToken(
            source,
            _RESTRICTED_TOKEN_FLAGS,
            disable_count,
            disable if disable_count else None,
            0,
            None,
            len(restrict_texts),
            restrict,
            ctypes.byref(restricted),
        ):
            raise _lpac._winerror("CreateRestrictedToken")
        _set_default_dacl(api, restricted, identity.sid)
        return restricted
    except BaseException:
        if restricted:
            api.kernel32.CloseHandle(restricted)
        raise
    finally:
        for sid in owned:
            api.kernel32.LocalFree(sid)
        api.kernel32.CloseHandle(source)


def _launch_environment(env: dict[str, str], identity: _LaunchIdentity) -> dict[str, str]:
    """The caller's (already sanitised) environment with temp redirected to the private temp."""
    launch = {key: value for key, value in env.items() if key.upper() not in {"TEMP", "TMP"}}
    if not any(key.upper() == "SYSTEMROOT" for key in launch):
        launch["SystemRoot"] = os.environ.get("SystemRoot", r"C:\Windows")
    launch["TEMP"] = identity.private_temp
    launch["TMP"] = identity.private_temp
    return launch


def _spawn_restricted(
    prepared: PreparedSandboxLaunch,
    popen_kwargs: dict[str, Any],
    identity: _LaunchIdentity,
) -> _lpac.WindowsLpacProcess:
    """Create the child under the write-restricted token, inside its resource-limited job."""
    if (
        popen_kwargs.get("stdout") != subprocess.PIPE
        or popen_kwargs.get("stderr") != subprocess.STDOUT
        or popen_kwargs.get("stdin") != subprocess.DEVNULL
        or not popen_kwargs.get("close_fds", True)
    ):
        raise SandboxUnavailableError(
            "the Limited launcher accepts only Studio's closed-descriptor stdio plan"
        )
    import msvcrt

    api = _lpac._api()
    flags = (
        int(popen_kwargs.get("creationflags", 0))
        | _lpac._CREATE_SUSPENDED
        | _lpac._CREATE_UNICODE_ENVIRONMENT
        | _lpac._EXTENDED_STARTUPINFO_PRESENT
        | _CREATE_NO_WINDOW
    )
    if flags & _lpac._CREATE_BREAKAWAY_FROM_JOB:
        raise SandboxUnavailableError("Limited processes may not break away from their Job Object")
    read_fd, write_fd = os.pipe()
    stdin_fd = -1
    token = wintypes.HANDLE()
    process_info = _lpac._PROCESS_INFORMATION()
    attribute_buffer: ctypes.Array[Any] | None = None
    attribute_list: ctypes.c_void_p | None = None
    attributes_initialized = False
    job: _lpac._WindowsJob | None = None
    stdout = None
    try:
        stdin_fd = os.open(os.devnull, os.O_RDONLY)
        # The job exists before the process so the child is never outside it,
        # not even suspended: JOB_LIST attaches at creation, the fallback
        # assigns before ResumeThread.
        job = _lpac._job_object_with_limits()
        os.set_inheritable(read_fd, False)
        os.set_inheritable(write_fd, True)
        os.set_inheritable(stdin_fd, True)
        child_stdin = wintypes.HANDLE(msvcrt.get_osfhandle(stdin_fd))
        child_stdout = wintypes.HANDLE(msvcrt.get_osfhandle(write_fd))
        handles = (wintypes.HANDLE * 2)(child_stdin, child_stdout)
        job_handles = (wintypes.HANDLE * 1)(job._handle)

        size = ctypes.c_size_t()
        api.kernel32.InitializeProcThreadAttributeList(None, 2, 0, ctypes.byref(size))
        if ctypes.get_last_error() != _ERROR_INSUFFICIENT_BUFFER or not size.value:
            raise _lpac._winerror("InitializeProcThreadAttributeList(size)")
        attribute_buffer = ctypes.create_string_buffer(size.value)
        attribute_list = ctypes.cast(attribute_buffer, ctypes.c_void_p)
        if not api.kernel32.InitializeProcThreadAttributeList(
            attribute_list, 2, 0, ctypes.byref(size)
        ):
            raise _lpac._winerror("InitializeProcThreadAttributeList")
        attributes_initialized = True
        if not api.kernel32.UpdateProcThreadAttribute(
            attribute_list,
            0,
            _lpac._PROC_THREAD_ATTRIBUTE_HANDLE_LIST,
            ctypes.byref(handles),
            ctypes.sizeof(handles),
            None,
            None,
        ):
            raise _lpac._winerror("UpdateProcThreadAttribute(handle list)")
        job_attached_at_creation = bool(
            api.kernel32.UpdateProcThreadAttribute(
                attribute_list,
                0,
                _lpac._PROC_THREAD_ATTRIBUTE_JOB_LIST,
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
            raise _lpac._winerror("UpdateProcThreadAttribute(job list)")

        startup = _lpac._STARTUPINFOEXW()
        startup.StartupInfo.cb = ctypes.sizeof(startup)
        startup.StartupInfo.dwFlags = _lpac._STARTF_USESTDHANDLES
        startup.StartupInfo.hStdInput = child_stdin
        startup.StartupInfo.hStdOutput = child_stdout
        startup.StartupInfo.hStdError = child_stdout
        startup.lpAttributeList = attribute_list
        command_line = ctypes.create_unicode_buffer(subprocess.list2cmdline(prepared.argv))
        environment = _lpac._environment_block(prepared.env)
        token = _create_restricted_token(identity)
        if not api.advapi32.CreateProcessAsUserW(
            token,
            prepared.argv[0],
            command_line,
            None,
            None,
            True,
            flags,
            environment,
            prepared.workdir,
            ctypes.cast(ctypes.byref(startup), ctypes.POINTER(_lpac._STARTUPINFOW)),
            ctypes.byref(process_info),
        ):
            raise _lpac._winerror("CreateProcessAsUserW(restricted token)")
        os.close(write_fd)
        write_fd = -1
        os.close(stdin_fd)
        stdin_fd = -1
        if not job_attached_at_creation and not api.kernel32.AssignProcessToJobObject(
            job._handle, process_info.hProcess
        ):
            raise _lpac._winerror("AssignProcessToJobObject")
        if api.kernel32.ResumeThread(process_info.hThread) == 0xFFFFFFFF:
            raise _lpac._winerror("ResumeThread")
        stdout = os.fdopen(
            read_fd,
            "r",
            encoding = popen_kwargs.get("encoding", "utf-8"),
            errors = popen_kwargs.get("errors", "replace"),
        )
        read_fd = -1
        process = _lpac.WindowsLpacProcess(
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
        if token:
            api.kernel32.CloseHandle(token)
        if attributes_initialized and attribute_list is not None:
            api.kernel32.DeleteProcThreadAttributeList(attribute_list)
        for fd in (read_fd, write_fd, stdin_fd):
            if fd >= 0:
                os.close(fd)


# The live probe runs under the token it verifies. Everything it reports is
# checked on the host (_evaluate_probe); the child only observes.
_PROBE_PAYLOAD = r'''
import ctypes, json, os, sys
from ctypes import wintypes
advapi32 = ctypes.WinDLL("advapi32", use_last_error = True)
kernel32 = ctypes.WinDLL("kernel32", use_last_error = True)
advapi32.OpenProcessToken.argtypes = [wintypes.HANDLE, wintypes.DWORD, ctypes.POINTER(wintypes.HANDLE)]
advapi32.OpenProcessToken.restype = wintypes.BOOL
advapi32.GetTokenInformation.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD, ctypes.POINTER(wintypes.DWORD)]
advapi32.GetTokenInformation.restype = wintypes.BOOL
advapi32.IsTokenRestricted.argtypes = [wintypes.HANDLE]
advapi32.IsTokenRestricted.restype = wintypes.BOOL
advapi32.ConvertSidToStringSidW.argtypes = [ctypes.c_void_p, ctypes.POINTER(wintypes.LPWSTR)]
advapi32.ConvertSidToStringSidW.restype = wintypes.BOOL
kernel32.GetCurrentProcess.restype = wintypes.HANDLE
kernel32.IsProcessInJob.argtypes = [wintypes.HANDLE, wintypes.HANDLE, ctypes.POINTER(wintypes.BOOL)]
kernel32.IsProcessInJob.restype = wintypes.BOOL
kernel32.LocalFree.argtypes = [ctypes.c_void_p]

class SA(ctypes.Structure):
    _fields_ = [("Sid", ctypes.c_void_p), ("Attributes", wintypes.DWORD)]
class TG(ctypes.Structure):
    _fields_ = [("GroupCount", wintypes.DWORD), ("Groups", SA * 1)]

token = wintypes.HANDLE()
if not advapi32.OpenProcessToken(kernel32.GetCurrentProcess(), 0x8, ctypes.byref(token)):
    raise SystemExit("OpenProcessToken failed: %d" % ctypes.get_last_error())

def info(kind):
    size = wintypes.DWORD()
    advapi32.GetTokenInformation(token, kind, None, 0, ctypes.byref(size))
    buf = ctypes.create_string_buffer(max(size.value, 4))
    if not advapi32.GetTokenInformation(token, kind, buf, size.value, ctypes.byref(size)):
        raise SystemExit("GetTokenInformation(%d) failed: %d" % (kind, ctypes.get_last_error()))
    return buf

def sids(kind):
    buf = info(kind)
    count = ctypes.cast(buf, ctypes.POINTER(wintypes.DWORD))[0]
    base = ctypes.addressof(buf) + TG.Groups.offset
    out = []
    for i in range(count):
        entry = SA.from_address(base + i * ctypes.sizeof(SA))
        text = wintypes.LPWSTR()
        if entry.Sid and advapi32.ConvertSidToStringSidW(ctypes.c_void_p(entry.Sid), ctypes.byref(text)):
            out.append(text.value)
            kernel32.LocalFree(text)
    return out

def writable(path):
    try:
        with open(path, "a", encoding = "utf-8") as stream:
            stream.write("x")
        return True
    except OSError:
        return False

def readable(path):
    try:
        with open(path, "r", encoding = "utf-8") as stream:
            return stream.read() == "secret"
    except OSError:
        return False

in_job = wintypes.BOOL()
kernel32.IsProcessInJob(kernel32.GetCurrentProcess(), None, ctypes.byref(in_job))
secret, sibling = sys.argv[1], sys.argv[2]
findings = {
    "restricted": bool(advapi32.IsTokenRestricted(token)),
    "restricted_sids": sids(11),
    "privileges": int(ctypes.cast(info(3), ctypes.POINTER(wintypes.DWORD))[0]),
    "in_job": bool(in_job.value),
    "secret_readable": readable(secret),
    "secret_writable": writable(secret),
    "sibling_writable": writable(os.path.join(sibling, "probe.txt")),
    "workdir_writable": writable(os.path.join(os.getcwd(), "probe.txt")),
    "temp_writable": writable(os.path.join(os.environ["TEMP"], "probe.txt")),
    "temp_is_private": os.path.normcase(os.environ["TEMP"]) == os.path.normcase(sys.argv[3]),
}
try:
    with open(os.devnull, "r+b"):
        findings["devnull"] = True
except OSError as exc:
    findings["devnull"] = "%s" % exc
try:
    import multiprocessing.connection as connection
    a, b = connection.Pipe()
    a.send("ping")
    findings["pipe"] = b.recv() == "ping"
    a.close(); b.close()
except Exception as exc:
    findings["pipe"] = "%s" % exc
print(json.dumps(findings))
'''


def _evaluate_probe(findings: dict[str, Any], *, sid_text: str) -> str | None:
    """The reason the probe failed, or ``None`` when every observation matches the model."""
    checks = (
        (findings.get("restricted") is True, "the token is not restricted"),
        (sid_text in findings.get("restricted_sids", ()), "the launch SID is not a restricting SID"),
        (_EVERYONE_SID in findings.get("restricted_sids", ()), "Everyone is not a restricting SID"),
        (
            isinstance(findings.get("privileges"), int) and findings["privileges"] <= 1,
            "the token kept privileges beyond SeChangeNotifyPrivilege",
        ),
        (findings.get("in_job") is True, "the child is not inside its Job Object"),
        (
            findings.get("secret_readable") is True,
            "the user's own files were not readable (the token is stronger than modelled)",
        ),
        (findings.get("secret_writable") is False, "a user-profile file outside the workdir was writable"),
        (findings.get("sibling_writable") is False, "another launch's temp directory was writable"),
        (findings.get("workdir_writable") is True, "the workdir was not writable"),
        (findings.get("temp_writable") is True, "the private temp was not writable"),
        (findings.get("temp_is_private") is True, "TEMP was not redirected to the private temp"),
        (findings.get("devnull") is True, f"the NUL device was unavailable: {findings.get('devnull')}"),
        (findings.get("pipe") is True, f"named pipes were unavailable: {findings.get('pipe')}"),
    )
    for passed, reason in checks:
        if not passed:
            return reason
    return None


class WindowsRestrictedTokenBackend:
    """Limited-mode launcher: a write-restricted token plus a kill-on-close Job Object."""

    identity = _BACKEND_IDENTITY
    profile_id = _PROFILE_ID
    limitations = _LIMITATIONS

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._capability: SandboxCapability | None = None

    def probe(self, *, force: bool = False) -> SandboxCapability:
        """Run (once per process unless forced) the live probe under a real restricted token."""
        if not _is_windows():
            return SandboxCapability(
                self.identity, False, "restricted tokens require Windows", available = False
            )
        with self._lock:
            if self._capability is not None and not force:
                return self._capability
            try:
                _lpac._api()
                self.reconcile_stale_manifests()
                failure = self._live_probe()
            except Exception as exc:  # noqa: BLE001 - an unprobeable launcher is unavailable
                failure = f"the restricted-token live probe could not run: {exc}"
            if failure is None:
                capability = SandboxCapability(
                    self.identity,
                    True,
                    "write-restricted token live probe passed: the workdir and a private temp are "
                    "the only writable user paths, the interpreter runs from its installed "
                    "location, privileges are stripped and the job kills detached children",
                    available = True,
                    protection_state = "preview",
                    profile_id = self.profile_id,
                    limitations = self.limitations,
                )
            else:
                capability = SandboxCapability(
                    self.identity,
                    False,
                    f"the restricted-token live probe failed: {failure}",
                    available = False,
                )
            self._capability = capability
            return capability

    def _live_probe(self) -> str | None:
        base = os.path.join(_private_root("limited-probe"), secrets.token_hex(8))
        os.makedirs(base, mode = 0o700)
        try:
            workdir = os.path.join(base, "work")
            sibling = os.path.join(base, "sibling")
            os.mkdir(workdir)
            os.mkdir(sibling)
            secret = os.path.join(base, "secret.txt")
            Path(secret).write_text("secret", encoding = "utf-8")
            prepared = self.prepare(
                ToolLaunchPlan(
                    argv = (sys.executable, "-I", "-S", "-c", _PROBE_PAYLOAD, secret, sibling, ""),
                    workdir = workdir,
                    env = {
                        "PYTHONIOENCODING": "utf-8",
                        "PATH": os.pathsep.join(
                            (
                                os.path.dirname(os.path.realpath(sys.executable)),
                                os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32"),
                            )
                        ),
                    },
                )
            )
            try:
                identity = getattr(prepared.spawn_callback, "_launch_identity", None)
                if identity is None:
                    raise SandboxUnavailableError("the Limited launch identity was lost")
                prepared.argv = (*prepared.argv[:-1], identity.private_temp)
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
                    },
                )
                try:
                    process.wait(timeout = 20)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout = 10)
                    return "the probe child did not finish within 20 s"
                output = process.stdout.read()
                if process.returncode != 0:
                    return f"the probe child exited with {process.returncode}: {output[-400:]}"
                try:
                    findings = json.loads(output.strip().splitlines()[-1])
                except (ValueError, IndexError):
                    return f"the probe child produced no report: {output[-400:]}"
                failure = _evaluate_probe(findings, sid_text = identity.sid_string)
                if failure is not None:
                    return failure
            finally:
                prepared.cleanup()
                if prepared.cleanup_diagnostics:
                    raise SandboxUnavailableError(
                        "probe cleanup failed: " + "; ".join(prepared.cleanup_diagnostics)
                    )
            if os.path.exists(identity.private_temp) or os.path.exists(identity.manifest_path):
                return "the probe launch left its private temp or manifest behind"
        finally:
            shutil.rmtree(base, ignore_errors = True)
        return None

    def prepare(self, spec: ToolLaunchPlan) -> PreparedSandboxLaunch:
        if not _is_windows():
            raise SandboxUnavailableError("restricted tokens require Windows")
        workdir = _lpac._validate_workdir(spec.workdir)
        for root in (_manifest_root(), _temp_root()):
            if _lpac._is_within(workdir, root) or _lpac._is_within(root, workdir):
                raise SandboxUnavailableError(
                    "the Limited workdir overlaps the launcher's private ownership state"
                )
        argv = _lpac._canonical_inner_argv(spec.argv, spec.env)
        identity = _create_identity(workdir)
        try:
            def spawn(prepared: PreparedSandboxLaunch, kwargs: dict[str, Any]) -> object:
                return _spawn_restricted(prepared, kwargs, identity)

            setattr(spawn, "_launch_identity", identity)
            return PreparedSandboxLaunch(
                argv = argv,
                workdir = workdir,
                env = _launch_environment(spec.env, identity),
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
        """Revoke grants and remove private temps whose owning Studio process is gone."""
        root = _manifest_root()
        for manifest in Path(root).glob(_MANIFEST_PREFIX + "*.json"):
            payload = _parse_manifest(manifest)
            if payload is None:
                continue
            try:
                if manifest.name != _MANIFEST_PREFIX + payload["sid"] + ".json":
                    continue
                owner = (payload["owner_pid"], payload["owner_created"])
                if _lpac._process_identity(owner[0]) == owner:
                    continue
                _validated_private_temp(payload["private_temp"])
                identity = _LaunchIdentity(
                    _sid_from_text(payload["sid"]),
                    payload["sid"],
                    payload["workdir"],
                    payload["private_temp"],
                    str(manifest),
                    tuple(payload["granted_roots"]),
                    owner[0],
                    owner[1],
                )
                identity.cleanup()
            except Exception:  # noqa: BLE001 - keep the record for the next startup
                logger.warning("Could not reconcile Limited launch manifest %s", manifest, exc_info = True)
                continue


__all__ = ["WindowsRestrictedTokenBackend"]
