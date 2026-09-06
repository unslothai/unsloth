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

Starting the child is its own problem, and a failure there is silent: the
creation call succeeds and the process is terminated during initialisation, so
the only evidence is an exit status. Three things the documentation asks for are
therefore done before the child runs, and named in the probe's failure text when
it still does not start:

* the launch SID is added to the DACL of the window station and the desktop.
  ``CreateProcessAsUser`` requires that "the DACLs for the window station and
  desktop must grant access to the user or the logon session represented by the
  hToken parameter", and a per-launch random SID is on no DACL anywhere;
* ``STARTUPINFO.lpDesktop`` names this process's own window station and desktop
  instead of being left NULL;
* the child is created ``DETACHED_PROCESS`` (Chromium's broker flags), so no
  console is allocated for it while it starts under the restricted token.

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

from contextlib import contextmanager
import ctypes
from ctypes import wintypes
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import secrets
import shutil
import stat
import subprocess
import sys
import threading
import time
from typing import Any, Callable, Iterator

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
_TOKEN_DEFAULT_DACL = 6  # TOKEN_INFORMATION_CLASS; the struct is _TOKEN_DEFAULT_DACL_INFO
_TOKEN_RESTRICTED_SIDS = 11
_TOKEN_LOGON_SID = 28

_EVERYONE_SID = "S-1-1-0"
_ADMINISTRATORS_SID = "S-1-5-32-544"
_NO_INHERITANCE = 0
# The window station ACE is the object's own, never inherited by desktops that
# do not exist yet (the interactive sample adds a second, inherit-only ACE for
# that; this launcher grants the one desktop it uses directly instead).
_NO_PROPAGATE_INHERIT_ACE = 0x04
_CREATE_NEW_CONSOLE = 0x00000010
# Chromium's broker creates every sandboxed target with exactly
# CREATE_SUSPENDED | CREATE_UNICODE_ENVIRONMENT | DETACHED_PROCESS
# (sandbox/win/src/broker_services.cc). A detached child allocates no console at
# all, so process start-up never has to create one under the restricted token;
# the caller's CREATE_NO_WINDOW is documented to be ignored beside it.
_DETACHED_PROCESS = 0x00000008
# "To enable user interaction with the new process, you must specify the name of
# the default interactive window station and desktop, winsta0\default, in the
# lpDesktop member of the STARTUPINFO structure" (CreateProcessAsUser). Codex's
# windows-sandbox-rs records the same requirement from the other side: "Some
# processes (e.g., PowerShell) can fail with STATUS_DLL_INIT_FAILED if lpDesktop
# is not set when launching with a restricted token."
_INTERACTIVE_DESKTOP = "WinSta0\\Default"

_UOI_NAME = 2
_DACL_SECURITY_INFORMATION = 0x00000004
_READ_CONTROL = 0x00020000
# Every WINSTA_* and DESKTOP_* right, and READ_CONTROL so the child can read the
# DACL it is being judged by. WRITE_DAC, WRITE_OWNER and DELETE are deliberately
# left out: a sandboxed child that could rewrite the window station DACL could
# grant itself anything. Codex's windows-sandbox-rs draws the same line for its
# desktop participants.
_WINSTA_ALL_ACCESS = 0x37F
_DESKTOP_ALL_ACCESS = 0x01FF
_WINSTA_GRANT = _WINSTA_ALL_ACCESS | _READ_CONTROL
_DESKTOP_GRANT = _DESKTOP_ALL_ACCESS | _READ_CONTROL

# An exit code the loader produced, not the payload: the child never ran.
_NTSTATUS_NAMES = {
    0xC0000005: "STATUS_ACCESS_VIOLATION",
    0xC0000017: "STATUS_NO_MEMORY",
    0xC0000022: "STATUS_ACCESS_DENIED",
    0xC000007B: "STATUS_INVALID_IMAGE_FORMAT",
    0xC0000135: "STATUS_DLL_NOT_FOUND",
    0xC0000139: "STATUS_ENTRYPOINT_NOT_FOUND",
    0xC0000142: "STATUS_DLL_INIT_FAILED",
    0xC000013A: "STATUS_CONTROL_C_EXIT",
    0xC0000409: "STATUS_STACK_BUFFER_OVERRUN",
}
_ERROR_INSUFFICIENT_BUFFER = 122
_ERROR_NOT_SUPPORTED = 50
_ERROR_INVALID_PARAMETER = 87
# LookupAccountSidW could not name the SID, which is what a launch SID must be.
_ERROR_NONE_MAPPED = 1332
_MAX_SIBLING_SCAN = 100_000
_FILE_ATTRIBUTE_REPARSE_POINT = 0x400

# TerminateJobObject and KILL_ON_JOB_CLOSE only start the kill, so the private
# temp and the workdir ACE are released only once the job has actually drained.
_JOB_OBJECT_BASIC_ACCOUNTING_INFORMATION = 1
_JOB_DRAIN_SECONDS = 5.0
_JOB_DRAIN_FIRST_POLL_SECONDS = 0.005
_JOB_DRAIN_MAX_POLL_SECONDS = 0.1
# A handle the kernel has not finished closing turns a removal into a sharing
# violation, so removal is retried before it counts as a leak.
_TEMP_REMOVAL_ATTEMPTS = 6
_TEMP_REMOVAL_BACKOFF_SECONDS = 0.05
# An interrupted manifest write leaves "<manifest>.json.tmp"; it is litter once
# its writer is gone, and always once it is this old.
_ORPHAN_TEMPORARY_MANIFEST_SECONDS = 300.0


class _SID_AND_ATTRIBUTES(ctypes.Structure):
    _fields_ = [("Sid", ctypes.c_void_p), ("Attributes", wintypes.DWORD)]


class _JOBOBJECT_BASIC_ACCOUNTING_INFORMATION(ctypes.Structure):
    """Declared here because ``windows_lpac._api()`` never needed to query a job."""

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


class _TOKEN_GROUPS_HEADER(ctypes.Structure):
    _fields_ = [("GroupCount", wintypes.DWORD), ("Groups", _SID_AND_ATTRIBUTES * 1)]


class _TOKEN_DEFAULT_DACL_INFO(ctypes.Structure):
    _fields_ = [("DefaultDacl", ctypes.c_void_p)]


def _is_windows() -> bool:
    return os.name == "nt"


def _last_error() -> int:
    """``GetLastError``, behind a seam the off-Windows behavioural tests drive."""
    return ctypes.get_last_error()


def _limited_wording(text: str) -> str:
    """Restate a shared AppContainer validation message for the Limited path it ran in."""
    return re.sub(r"\bLPAC\b", "Limited mode", text)


@contextmanager
def _limited_mode_wording() -> Iterator[None]:
    """Report helpers shared with the AppContainer backend as Limited mode failures."""
    try:
        yield
    except SandboxUnavailableError as exc:
        message = _limited_wording(str(exc))
        if message == str(exc):
            raise
        raise SandboxUnavailableError(
            message, transient = getattr(exc, "transient", False)
        ) from exc


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
    """The private temp of one launch, refused unless it is a plain child of the temp root.

    This is the only bound on the one destructive operation performed on manifest
    input, so it stays a pure path check: a 24 hex character direct child of the
    launcher's own temp root.
    """
    spelled = os.path.abspath(private_temp)
    root = _temp_root()
    name = os.path.basename(spelled)
    if (
        os.path.normcase(os.path.dirname(spelled)) != os.path.normcase(root)
        or len(name) != 24
        or not all(character in "0123456789abcdef" for character in name.lower())
    ):
        raise SandboxUnavailableError("a Limited mode private temp path is outside its root")
    return spelled


def _is_reparse_point(path: str) -> bool:
    """Whether a path is a junction or a symlink, tested without following it."""
    try:
        info = os.lstat(path)
    except OSError:
        return False
    return bool(getattr(info, "st_file_attributes", 0) & _FILE_ATTRIBUTE_REPARSE_POINT)


def _remove_reparse_point(path: str) -> None:
    """Remove the link itself.

    A sandboxed child needs no privilege to create a directory junction, so one
    planted in its private temp must not be able to block cleanup forever.
    ``os.rmdir`` is ``RemoveDirectoryW``, which deletes the reparse point and
    never follows it; a file symlink needs ``os.unlink`` instead.
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


def _prune_reparse_points(root: str) -> None:
    """Delete every reparse point under a private temp, never recursing into one."""
    entries = 0
    pending = [root]
    while pending:
        base = pending.pop()
        try:
            with os.scandir(base) as scan:
                children = list(scan)
        except FileNotFoundError:
            continue
        for child in children:
            entries += 1
            if entries > _MAX_SIBLING_SCAN:
                raise SandboxUnavailableError("the Limited mode private temp is too large")
            if _is_reparse_point(child.path):
                _remove_reparse_point(child.path)
                continue
            try:
                is_directory = child.is_dir(follow_symlinks = False)
            except OSError:
                is_directory = False
            if is_directory:
                pending.append(child.path)


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


def _remove_private_temp(private_temp: str) -> None:
    """Remove one launch's private temp, retrying past handles the kernel is still closing."""
    target = _validated_private_temp(private_temp)
    if _is_reparse_point(target):
        _remove_reparse_point(target)
        return
    handler = _rmtree_error_handler()
    for attempt in range(_TEMP_REMOVAL_ATTEMPTS):
        try:
            _prune_reparse_points(target)
            shutil.rmtree(target, **handler)
            return
        except FileNotFoundError:
            return
        except OSError:
            if attempt == _TEMP_REMOVAL_ATTEMPTS - 1:
                raise
            time.sleep(_TEMP_REMOVAL_BACKOFF_SECONDS * (attempt + 1))


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
    Revoking the workdir ACE and removing the private temp before the children
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
    api = _lpac._api()
    deadline = time.monotonic() + timeout
    delay = _JOB_DRAIN_FIRST_POLL_SECONDS
    while True:
        active = _job_active_processes(api, handle)
        if active == 0:
            return True
        if active is None:
            logger.warning("Could not observe the Limited job draining; removal retries instead")
            return None
        if time.monotonic() >= deadline:
            logger.warning(
                "The Limited job still held %d process(es) after %.1f s", active, timeout
            )
            return False
        time.sleep(delay)
        delay = min(delay * 2, _JOB_DRAIN_MAX_POLL_SECONDS)


def _close_after_drain(process: Any, job: Any) -> None:
    """Kill the job, wait for it to drain, then release the process and job handles.

    A drain that provably did not finish is raised, so ``cleanup`` records it as
    a diagnostic instead of continuing in silence: this callback runs before the
    ACL revoke and the private temp removal (cleanup is LIFO), and both of those
    then run against a child that has not exited. The handles are released
    either way - holding them would leak the job and the process for the rest of
    Studio's life - and the launch SID is still revoked afterwards, because a
    child that outlived ``TerminateJobObject`` is a runaway that must not keep
    its write access. A host that cannot be asked (``None``) is not a failure.
    """
    job.terminate()
    drained = _wait_for_job_drain(job)
    process.close()
    if drained is False:
        raise OSError(
            f"the Limited job still held a process {_JOB_DRAIN_SECONDS:.0f} s after it was "
            "terminated; the ACL revoke and the private temp removal below ran while a child "
            "of this launch was still alive"
        )


@dataclass
class _LaunchResources:
    """The token and Job Object built by ``prepare`` so their failures fall back.

    Both are owned by the prepared launch until a process takes them over; the
    cleanup callback closes whatever is left.
    """

    token: Any = None
    job: Any = None

    def take_token(self) -> Any:
        token, self.token = self.token, None
        return token

    def take_job(self) -> Any:
        job, self.job = self.job, None
        return job

    def close(self) -> None:
        token, self.token = self.token, None
        job, self.job = self.job, None
        if token:
            _lpac._api().kernel32.CloseHandle(token)
        if job is not None:
            job.terminate()
            _wait_for_job_drain(job)
            job.close()


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
    # "window station" and "desktop" once their DACLs carry this launch's SID,
    # and why they do not when a host refuses the edit.
    user_objects: tuple[str, ...] = ()
    user_object_reason: str = ""
    # "<window station>\<desktop>" for STARTUPINFO.lpDesktop.
    desktop: str = ""

    def cleanup(self) -> None:
        if self.cleaned:
            return
        if not self.sid:
            # A previous attempt failed and freed its allocation. The SID text is
            # the record, so a retry converts it again rather than reusing memory
            # that is already back with the allocator.
            self.sid = _sid_from_text(self.sid_string)
        try:
            errors: list[str] = []
            for path in reversed(self.granted_roots):
                try:
                    _lpac._revoke_sid(path, self.sid)
                except Exception as exc:  # noqa: BLE001 - continue ownership cleanup
                    errors.append(f"ACL {path}: {exc}")
            try:
                _revoke_user_objects(self)
            except Exception as exc:  # noqa: BLE001 - a session object, see the docstring
                errors.append(f"user objects: {exc}")
            try:
                _remove_private_temp(self.private_temp)
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
            self.cleaned = True
        finally:
            # Freed on every exit, not only the successful one: the failing paths
            # are the ones that repeat.
            if self.sid:
                _lpac._api().kernel32.LocalFree(self.sid)
            self.sid = ctypes.c_void_p()


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
    # The reconciler revokes an ACE on every granted root, so a planted manifest
    # must not be able to name one. _create_identity grants exactly two roots.
    if len(roots) != 2 or {os.path.normcase(path) for path in roots} != {
        os.path.normcase(workdir),
        os.path.normcase(private_temp),
    }:
        return None
    return payload


def _sid_from_text(text: str) -> ctypes.c_void_p:
    api = _lpac._api()
    sid = ctypes.c_void_p()
    if not api.advapi32.ConvertStringSidToSidW(text, ctypes.byref(sid)) or not sid:
        raise _lpac._winerror(f"ConvertStringSidToSidW({text})")
    return sid


def _sid_names_a_principal(api: Any, sid: ctypes.c_void_p) -> bool:
    """Whether ``LookupAccountSidW`` resolves the SID to a real account.

    A launch SID is a random, never-assigned ``S-1-5-21`` value, which is also
    the exact shape of a local or domain account SID. Only a SID that names
    nobody may drive an ACL revoke out of a manifest this process did not write.
    """
    lookup = getattr(getattr(api, "advapi32", None), "LookupAccountSidW", None)
    if lookup is None:
        raise SandboxUnavailableError(
            "the Limited launcher cannot check whether a manifest SID names an account"
        )
    if getattr(lookup, "argtypes", None) is None:
        try:
            lookup.argtypes = [
                wintypes.LPCWSTR,
                ctypes.c_void_p,
                wintypes.LPWSTR,
                ctypes.POINTER(wintypes.DWORD),
                wintypes.LPWSTR,
                ctypes.POINTER(wintypes.DWORD),
                ctypes.POINTER(ctypes.c_int),
            ]
            lookup.restype = wintypes.BOOL
        except (AttributeError, TypeError):
            pass
    name_length = wintypes.DWORD(0)
    domain_length = wintypes.DWORD(0)
    use = ctypes.c_int(0)
    if lookup(
        None,
        sid,
        None,
        ctypes.byref(name_length),
        None,
        ctypes.byref(domain_length),
        ctypes.byref(use),
    ):
        return True
    error = _last_error()
    if error == _ERROR_NONE_MAPPED:
        return False
    if error == _ERROR_INSUFFICIENT_BUFFER:
        # The name did not fit the zero-length buffers, so the SID resolved.
        return True
    raise _lpac._winerror("LookupAccountSidW(manifest SID)", error)


def _temporary_manifest_is_orphaned(temporary: Path) -> bool:
    """Whether an interrupted ``<manifest>.json.tmp`` can no longer become a manifest."""
    try:
        age = time.time() - temporary.stat().st_mtime
    except OSError:
        return False
    if age > _ORPHAN_TEMPORARY_MANIFEST_SECONDS:
        return True
    try:
        payload = json.loads(temporary.read_text(encoding = "utf-8"))
    except (OSError, ValueError):
        return False  # a partial write, possibly still in progress
    if not isinstance(payload, dict):
        return False
    owner = (payload.get("owner_pid"), payload.get("owner_created"))
    if not isinstance(owner[0], int) or not isinstance(owner[1], int):
        return False
    return _lpac._process_identity(owner[0]) != owner


def _remove_orphan_temporary_manifests(root: str) -> None:
    """Delete the ``.json.tmp`` files a crash between ``open`` and ``os.replace`` leaves."""
    for temporary in Path(root).glob(_MANIFEST_PREFIX + "*.json.tmp"):
        try:
            if _temporary_manifest_is_orphaned(temporary):
                temporary.unlink()
        except OSError:
            logger.warning(
                "Could not remove the orphaned Limited manifest %s", temporary, exc_info = True
            )


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
        if _is_reparse_point(private_temp):
            raise SandboxUnavailableError("the Limited mode private temp is a reparse point")
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
    if _last_error() != _ERROR_INSUFFICIENT_BUFFER or not size.value:
        raise _lpac._winerror(f"GetTokenInformation({kind}, size)")
    buffer = ctypes.create_string_buffer(size.value)
    if not api.advapi32.GetTokenInformation(token, kind, buffer, size.value, ctypes.byref(size)):
        raise _lpac._winerror(f"GetTokenInformation({kind})")
    return buffer


def _token_group_sids(api: Any, token: wintypes.HANDLE, kind: int) -> list[str]:
    """The SID strings of a ``TOKEN_GROUPS`` token class (logon SID, restricted SIDs).

    A failure is raised, never swallowed: a missing logon SID would silently
    weaken the restricting set instead of declining the launch.
    """
    buffer = _token_information(api, token, kind)
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
    current = ctypes.cast(buffer, ctypes.POINTER(_TOKEN_DEFAULT_DACL_INFO))[0].DefaultDacl
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
        default_dacl = _TOKEN_DEFAULT_DACL_INFO(new_acl)
        if not api.advapi32.SetTokenInformation(
            token,
            _TOKEN_DEFAULT_DACL,
            ctypes.byref(default_dacl),
            ctypes.sizeof(default_dacl),
        ):
            raise _lpac._winerror("SetTokenInformation(TokenDefaultDacl)")
    finally:
        api.kernel32.LocalFree(new_acl)


def _user_object_name(api: Any, handle: wintypes.HANDLE) -> str:
    """The name of a window station or desktop handle, for the launch record."""
    buffer = ctypes.create_unicode_buffer(256)
    size = wintypes.DWORD()
    if not api.user32.GetUserObjectInformationW(
        handle, _UOI_NAME, buffer, ctypes.sizeof(buffer), ctypes.byref(size)
    ):
        raise _lpac._winerror("GetUserObjectInformationW(name)")
    return buffer.value


def _process_user_objects(api: Any) -> tuple[tuple[str, Any], ...]:
    """The window station and desktop a child of this process connects to.

    Neither handle is owned here. ``GetProcessWindowStation`` and
    ``GetThreadDesktop`` hand out the process's own, which must not be closed.
    """
    winsta = api.user32.GetProcessWindowStation()
    desktop = api.user32.GetThreadDesktop(api.kernel32.GetCurrentThreadId())
    if not winsta or not desktop:
        raise _lpac._winerror("GetProcessWindowStation/GetThreadDesktop")
    return (("window station", winsta), ("desktop", desktop))


def _launch_desktop(api: Any) -> str:
    """``<window station>\\<desktop>`` for the objects this process is connected to.

    A launcher running in a session whose window station is not ``WinSta0`` (a
    service) would otherwise name a station it is not on. The documented default
    is used only when the names cannot be read at all.
    """
    try:
        names = [_user_object_name(api, handle) for _kind, handle in _process_user_objects(api)]
    except OSError:
        logger.warning("Could not name the Limited launcher's window station", exc_info = True)
        return _INTERACTIVE_DESKTOP
    return "\\".join(names)


def _edit_user_object_dacl(
    api: Any,
    handle: Any,
    sid: ctypes.c_void_p,
    *,
    mode: int,
    access: int,
    inheritance: int,
) -> bool:
    """Add or remove one SID's ACE on a window station or desktop.

    The two-call size pattern of ``GetUserObjectSecurity``, then the object's
    own DACL with this one entry added (or every entry for this trustee
    removed), written back as an absolute descriptor. The trustee is always the
    per-launch SID, so a revoke can never take another principal's ACE with it.

    An object with no DACL is left alone, and ``False`` says so. A NULL DACL
    allows everyone everything, so the SID needs nothing added; writing a DACL
    that holds only this ACE would take the interactive window station away from
    the user's own session. Chromium's ``window.cc`` guards the same trap from
    the other direction, seeding an allow-everyone entry before it denies.
    """
    information = wintypes.DWORD(_DACL_SECURITY_INFORMATION)
    needed = wintypes.DWORD()
    api.user32.GetUserObjectSecurity(
        handle, ctypes.byref(information), None, 0, ctypes.byref(needed)
    )
    if _last_error() != _ERROR_INSUFFICIENT_BUFFER or not needed.value:
        raise _lpac._winerror("GetUserObjectSecurity(size)")
    current = ctypes.create_string_buffer(needed.value)
    if not api.user32.GetUserObjectSecurity(
        handle, ctypes.byref(information), current, needed.value, ctypes.byref(needed)
    ):
        raise _lpac._winerror("GetUserObjectSecurity")
    present = wintypes.BOOL()
    defaulted = wintypes.BOOL()
    dacl = ctypes.c_void_p()
    if not api.advapi32.GetSecurityDescriptorDacl(
        current, ctypes.byref(present), ctypes.byref(dacl), ctypes.byref(defaulted)
    ):
        raise _lpac._winerror("GetSecurityDescriptorDacl(user object)")
    if not present.value or not dacl:
        return False
    trustee = _lpac._TRUSTEE_W(
        None,
        _lpac._NO_MULTIPLE_TRUSTEE,
        _lpac._TRUSTEE_IS_SID,
        _lpac._TRUSTEE_IS_UNKNOWN,
        ctypes.cast(sid, wintypes.LPWSTR),
    )
    entry = _lpac._EXPLICIT_ACCESS_W(access, mode, inheritance, trustee)
    new_acl = ctypes.c_void_p()
    result = api.advapi32.SetEntriesInAclW(1, ctypes.byref(entry), dacl, ctypes.byref(new_acl))
    if result != 0:
        raise _lpac._winerror("SetEntriesInAclW(user object)", result)
    try:
        descriptor = _lpac._SECURITY_DESCRIPTOR()
        if not api.advapi32.InitializeSecurityDescriptor(ctypes.byref(descriptor), 1):
            raise _lpac._winerror("InitializeSecurityDescriptor(user object)")
        if not api.advapi32.SetSecurityDescriptorDacl(
            ctypes.byref(descriptor), True, new_acl, False
        ):
            raise _lpac._winerror("SetSecurityDescriptorDacl(user object)")
        if not api.user32.SetUserObjectSecurity(
            handle, ctypes.byref(information), ctypes.byref(descriptor)
        ):
            raise _lpac._winerror("SetUserObjectSecurity")
    finally:
        api.kernel32.LocalFree(new_acl)
    return True


def _grant_user_objects(identity: _LaunchIdentity) -> str:
    """Let the launch SID reach the window station and desktop the child needs.

    ``CreateProcessAsUser`` documents this as a precondition: "before calling
    CreateProcessAsUser, you must change the discretionary access control list
    (DACL) of both the default interactive window station and the default
    desktop. The DACLs for the window station and desktop must grant access to
    the user or the logon session represented by the hToken parameter." The
    launch SID is freshly generated, so it appears in no DACL on the host, and
    under ``WRITE_RESTRICTED`` every write the child makes against those objects
    is checked a second time against the restricting SIDs alone. Without this
    the child depends on the logon SID and Everyone ACEs a particular host
    happens to carry, and dies in ``LdrpInitializeProcess`` where it does not.

    Best effort, and deliberately so: editing those DACLs needs ``WRITE_DAC`` on
    the handles this process already holds, and a host that refuses is left
    exactly where it was before this grant existed, with the child depending on
    the logon SID and Everyone ACEs the session happens to carry. The reason is
    recorded on the identity and reaches the live probe's failure text, so a
    refusal is named rather than guessed at.

    The ACEs are recorded on the identity and removed by ``cleanup``.
    """
    api = _lpac._api()
    granted: list[str] = []
    reason = ""
    kind = "window station"
    identity.desktop = _launch_desktop(api)
    try:
        for kind, handle in _process_user_objects(api):
            access = _WINSTA_GRANT if kind == "window station" else _DESKTOP_GRANT
            inheritance = _NO_PROPAGATE_INHERIT_ACE if kind == "window station" else _NO_INHERITANCE
            if _edit_user_object_dacl(
                api,
                handle,
                identity.sid,
                mode = _lpac._GRANT_ACCESS,
                access = access,
                inheritance = inheritance,
            ):
                granted.append(kind)
            else:
                reason = f"the {kind} has no DACL, so it already grants every SID"
    except OSError as exc:
        reason = f"the launch SID could not be granted the {kind} DACL: {exc}"
        logger.warning("Could not grant the Limited launch SID a user object", exc_info = True)
    identity.user_objects = tuple(granted)
    identity.user_object_reason = reason
    return reason


def _revoke_user_objects(identity: _LaunchIdentity) -> None:
    """Remove the launch SID's window station and desktop ACEs.

    Best effort by design: these are session objects, and a Studio reconciling
    another Studio's manifest from a different session cannot reach the ones the
    ACE was written on. The SID names no account and is never reused, so an ACE
    that outlives its launch grants nobody anything and disappears with the
    session.
    """
    if not identity.user_objects:
        return
    api = _lpac._api()
    for kind, handle in _process_user_objects(api):
        if kind not in identity.user_objects:
            continue
        _edit_user_object_dacl(
            api,
            handle,
            identity.sid,
            mode = _lpac._REVOKE_ACCESS,
            access = 0,
            inheritance = _NO_INHERITANCE,
        )


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
        # The window station and desktop DACLs are written around the logon SID,
        # so a write-restricted child without it dies during initialisation.
        # Declining here falls back to the process guard instead.
        logon_sids = _token_group_sids(api, source, _TOKEN_LOGON_SID)
        if not logon_sids:
            raise SandboxUnavailableError(
                "the Limited launcher could not read the logon SID of Studio's token"
            )
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
        try:
            groups = _token_group_sids(api, source, _TOKEN_GROUPS)
        except OSError:  # LUA_TOKEN already covers this gap, so it stays a swallow
            logger.warning("Could not read Studio's token groups", exc_info = True)
            groups = []
        if _ADMINISTRATORS_SID in groups:
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
    resources: _LaunchResources,
) -> _lpac.WindowsLpacProcess:
    """Create the child under the token and job ``prepare`` already built.

    Only the process creation, its job attachment and the resume happen here.
    Everything that can fail for a reason the caller should fall back from was
    done in ``prepare``.
    """
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
    # Chromium's broker uses exactly these three flags for its own
    # CreateProcessAsUserW call (sandbox/win/src/broker_services.cc). DETACHED_PROCESS
    # replaces the CREATE_NO_WINDOW this launcher used to add: every stdio handle is
    # redirected here, so the child needs no console, and a detached child never has
    # one allocated for it while it is starting under the restricted token.
    # CREATE_NO_WINDOW from the caller is documented to be ignored beside it.
    flags = (
        int(popen_kwargs.get("creationflags", 0))
        | _lpac._CREATE_SUSPENDED
        | _lpac._CREATE_UNICODE_ENVIRONMENT
        | _lpac._EXTENDED_STARTUPINFO_PRESENT
        | _DETACHED_PROCESS
    )
    if flags & _lpac._CREATE_BREAKAWAY_FROM_JOB:
        raise SandboxUnavailableError("Limited processes may not break away from their Job Object")
    if flags & _CREATE_NEW_CONSOLE:
        # DETACHED_PROCESS and CREATE_NEW_CONSOLE cannot both be set; the pair
        # would fail the creation call rather than the sandbox check.
        raise SandboxUnavailableError("Limited processes may not be given their own console")
    token = resources.token
    # The job exists before the process so the child is never outside it, not
    # even suspended: JOB_LIST attaches at creation, the fallback assigns
    # before ResumeThread.
    job = resources.job
    if not token or job is None:
        raise SandboxUnavailableError("the Limited launch already released its token and job")
    read_fd, write_fd = os.pipe()
    stdin_fd = -1
    token_consumed = False
    process_info = _lpac._PROCESS_INFORMATION()
    attribute_buffer: ctypes.Array[Any] | None = None
    attribute_list: ctypes.c_void_p | None = None
    attributes_initialized = False
    stdout = None
    try:
        stdin_fd = os.open(os.devnull, os.O_RDONLY)
        os.set_inheritable(read_fd, False)
        os.set_inheritable(write_fd, True)
        os.set_inheritable(stdin_fd, True)
        child_stdin = wintypes.HANDLE(msvcrt.get_osfhandle(stdin_fd))
        child_stdout = wintypes.HANDLE(msvcrt.get_osfhandle(write_fd))
        handles = (wintypes.HANDLE * 2)(child_stdin, child_stdout)
        job_handles = (wintypes.HANDLE * 1)(job._handle)

        size = ctypes.c_size_t()
        api.kernel32.InitializeProcThreadAttributeList(None, 2, 0, ctypes.byref(size))
        if _last_error() != _ERROR_INSUFFICIENT_BUFFER or not size.value:
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
        if not job_attached_at_creation and _last_error() not in (
            _ERROR_NOT_SUPPORTED,
            _ERROR_INVALID_PARAMETER,
        ):
            raise _lpac._winerror("UpdateProcThreadAttribute(job list)")

        startup = _lpac._STARTUPINFOEXW()
        startup.StartupInfo.cb = ctypes.sizeof(startup)
        # Named, never left NULL: CreateProcessAsUser puts a child whose desktop
        # is unspecified on a noninteractive window station, and a restricted
        # token that cannot connect to the one it is given dies in DLL
        # initialisation before any payload runs. The name is this process's own
        # window station and desktop, which are the objects the launch SID was
        # just granted, so the two can never disagree.
        startup.StartupInfo.lpDesktop = identity.desktop or _INTERACTIVE_DESKTOP
        startup.StartupInfo.dwFlags = _lpac._STARTF_USESTDHANDLES
        startup.StartupInfo.hStdInput = child_stdin
        startup.StartupInfo.hStdOutput = child_stdout
        startup.StartupInfo.hStdError = child_stdout
        startup.lpAttributeList = attribute_list
        command_line = ctypes.create_unicode_buffer(subprocess.list2cmdline(prepared.argv))
        with _limited_mode_wording():
            environment = _lpac._environment_block(prepared.env)
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
        # The child holds its own reference to the token now, so the launch owns
        # this handle only until the finally below.
        resources.take_token()
        token_consumed = True
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
        # The job is the process's from here; anything left with the launch is
        # released by _LaunchResources.close.
        prepared.cleanup_callbacks.append(lambda: _close_after_drain(process, job))
        resources.take_job()
        return process
    except Exception:
        if process_info.hProcess:
            api.kernel32.TerminateProcess(process_info.hProcess, 1)
        for handle in (process_info.hThread, process_info.hProcess):
            if handle:
                api.kernel32.CloseHandle(handle)
        if stdout is not None:
            stdout.close()
        raise
    finally:
        if token_consumed and token:
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


def _status_text(code: int) -> str:
    """An exit code as the loader status it is, when it is one."""
    unsigned = ctypes.c_uint32(code).value
    name = _NTSTATUS_NAMES.get(unsigned)
    return f"{name} (0x{unsigned:08x})" if name else f"0x{unsigned:08x}"


def _is_start_failure(code: int) -> bool:
    """Whether an exit code is a loader status, so the payload never ran."""
    return ctypes.c_uint32(code).value & 0xC0000000 == 0xC0000000


def _control_child(argv: tuple[str, ...], workdir: str, env: dict[str, str], desktop: str) -> str:
    """Start one trivial child under a token restricted only in its privileges.

    The control the failure text needs, and the one the staging probes lacked:
    it separates "this host will not start a ``CreateProcessAsUserW`` child of
    Studio at all" from "the restriction is what the child cannot initialise
    under". ``DISABLE_MAX_PRIVILEGE`` alone is still a restricted version of the
    caller's primary token, so no privilege is required for it, and it carries
    none of ``LUA_TOKEN``, ``WRITE_RESTRICTED`` or the restricting SIDs.

    Nothing of a caller's payload runs: the child is the probe's own interpreter
    with an immediate exit. The job object, the inherited handles and the
    suspension are left out, so this answers for the token and the desktop only.
    """
    api = _lpac._api()
    source = wintypes.HANDLE()
    if not api.advapi32.OpenProcessToken(
        api.kernel32.GetCurrentProcess(), _TOKEN_ACCESS, ctypes.byref(source)
    ):
        return f"the control launch could not open Studio's token ({_last_error()})"
    token = wintypes.HANDLE()
    process_info = _lpac._PROCESS_INFORMATION()
    try:
        if not api.advapi32.CreateRestrictedToken(
            source, _DISABLE_MAX_PRIVILEGE, 0, None, 0, None, 0, None, ctypes.byref(token)
        ):
            return f"the control token could not be built ({_last_error()})"
        startup = _lpac._STARTUPINFOW()
        startup.cb = ctypes.sizeof(startup)
        startup.lpDesktop = desktop or _INTERACTIVE_DESKTOP
        command_line = ctypes.create_unicode_buffer(
            subprocess.list2cmdline((argv[0], "-I", "-S", "-c", "raise SystemExit(0)"))
        )
        with _limited_mode_wording():
            environment = _lpac._environment_block(env)
        if not api.advapi32.CreateProcessAsUserW(
            token,
            argv[0],
            command_line,
            None,
            None,
            False,
            _DETACHED_PROCESS | _lpac._CREATE_UNICODE_ENVIRONMENT,
            environment,
            workdir,
            ctypes.byref(startup),
            ctypes.byref(process_info),
        ):
            return f"CreateProcessAsUserW refused the control launch ({_last_error()})"
        api.kernel32.WaitForSingleObject(process_info.hProcess, 20000)
        code = wintypes.DWORD()
        if not api.kernel32.GetExitCodeProcess(process_info.hProcess, ctypes.byref(code)):
            return f"the control child's exit code could not be read ({_last_error()})"
        if code.value == _lpac._STILL_ACTIVE:
            api.kernel32.TerminateProcess(process_info.hProcess, 1)
            return "the control child did not finish within 20 s"
        if code.value == 0:
            return (
                "the same launch started under a token restricted only in its privileges, so "
                "LUA_TOKEN, WRITE_RESTRICTED or the restricting SIDs are what the child cannot "
                "initialise under"
            )
        return (
            "a control child under a token restricted only in its privileges failed the same way "
            f"({_status_text(code.value)}), so this host does not start CreateProcessAsUserW "
            "children of Studio at all"
        )
    except OSError as exc:
        return f"the control launch could not run ({exc})"
    finally:
        for handle in (process_info.hThread, process_info.hProcess):
            if handle:
                api.kernel32.CloseHandle(handle)
        if token:
            api.kernel32.CloseHandle(token)
        api.kernel32.CloseHandle(source)


def _probe_start_failure(
    returncode: int, output: str, identity: _LaunchIdentity, env: dict[str, str], argv: tuple[str, ...]
) -> str:
    """Why the probe child never reported, named as concretely as the host allows.

    A loader status means the payload never ran, so the useful facts are the
    status itself, the desktop the child was given, which of the window station
    and desktop DACLs carry the launch SID, and how the same launch fares
    without the restriction. Anything else is the payload's own failure and is
    reported as the exit code and its output.
    """
    if not _is_start_failure(returncode):
        return f"the probe child exited with {returncode}: {output[-400:]}"
    granted = ", ".join(identity.user_objects) or "neither"
    detail = (
        f"the probe child died in Windows process start-up with {_status_text(returncode)}, "
        f"before running: desktop {identity.desktop or _INTERACTIVE_DESKTOP}, the launch SID "
        f"is on the DACL of {granted}"
    )
    if identity.user_object_reason:
        detail += f"; {identity.user_object_reason}"
    try:
        detail += "; " + _control_child(argv, identity.workdir, env, identity.desktop)
    except Exception as exc:  # noqa: BLE001 - a diagnostic must not replace the diagnosis
        detail += f"; the control launch raised {type(exc).__name__}: {exc}"
    if output.strip():
        detail += f"; child output: {output[-200:]}"
    return detail


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
                    return _probe_start_failure(
                        process.returncode, output, identity, prepared.env, prepared.argv
                    )
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
        with _limited_mode_wording():
            workdir = _lpac._validate_workdir(spec.workdir)
        for root in (_manifest_root(), _temp_root()):
            if _lpac._is_within(workdir, root) or _lpac._is_within(root, workdir):
                raise SandboxUnavailableError(
                    "the Limited workdir overlaps the launcher's private ownership state"
                )
        with _limited_mode_wording():
            argv = _lpac._canonical_inner_argv(spec.argv, spec.env)
        identity = _create_identity(workdir)
        resources = _LaunchResources()
        try:
            # The token, its default DACL and the job are built here, not in the
            # spawn callback: prepare is the boundary os_sandbox falls back from,
            # so a failure past it would surface as a failed tool call instead of
            # a Limited call running under the process guard.
            try:
                # Before the token, because the token is what the window station
                # and desktop DACLs are then checked against.
                _grant_user_objects(identity)
                resources.token = _create_restricted_token(identity)
                resources.job = _lpac._job_object_with_limits()
            except OSError as exc:
                raise SandboxUnavailableError(
                    f"the Limited launcher could not build its restricted token and job: {exc}"
                ) from exc

            def spawn(prepared: PreparedSandboxLaunch, kwargs: dict[str, Any]) -> object:
                return _spawn_restricted(prepared, kwargs, identity, resources)

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
                cleanup_callbacks = [identity.cleanup, resources.close],
            )
        except BaseException:
            for release in (resources.close, identity.cleanup):
                try:
                    release()
                except Exception:  # noqa: BLE001 - the manifest keeps the record
                    logger.warning(
                        "Could not release a declined Limited launch", exc_info = True
                    )
            raise

    def reconcile_stale_manifests(self) -> None:
        """Revoke grants and remove private temps whose owning Studio process is gone."""
        root = _manifest_root()
        _remove_orphan_temporary_manifests(root)
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
                sid = _sid_from_text(payload["sid"])
                try:
                    names_a_principal = _sid_names_a_principal(_lpac._api(), sid)
                except BaseException:
                    _lpac._api().kernel32.LocalFree(sid)
                    raise
                if names_a_principal:
                    _lpac._api().kernel32.LocalFree(sid)
                    logger.warning(
                        "Refusing to reconcile Limited launch manifest %s: its SID names an "
                        "existing account",
                        manifest,
                    )
                    continue
                identity = _LaunchIdentity(
                    sid,
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
