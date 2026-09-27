# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
"""One-time AppContainer read grants on Studio's runtime folders for MXC's AppContainer + DACL tier.

On that tier wxc-exec grants every readonly path per launch with an inheritable ACE, which Windows
propagates to every file below it and reverts on exit, so a Python environment costs 10+ s per
command (microsoft/mxc#572). wxc-exec skips a path whose DACL already grants ALL APPLICATION PACKAGES
read and execute, as Windows ships C:\\Windows and Program Files. Granting that once on the runtime
folders Studio owns lets every later launch skip the walk. Read and execute only, never write.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import subprocess
import threading

logger = logging.getLogger(__name__)

PERSISTENT_GRANTS_ENV = "UNSLOTH_MXC_PERSISTENT_READ_GRANTS"
ALL_APPLICATION_PACKAGES = "S-1-15-2-1"
# FILE_GENERIC_READ | FILE_GENERIC_EXECUTE: wxc-exec's readonly mask and icacls (RX).
READ_EXECUTE_MASK = 0x1200A9
# A folder holding one of these is never opened to every AppContainer on the machine.
CREDENTIAL_FILES = frozenset(
    {"pip.ini", "pip.conf", "uv.toml", ".pypirc", ".netrc", "_netrc", ".env"}
)
GRANT_TIMEOUT_SECONDS = 900

_OBJECT_INHERIT = 0x1
_CONTAINER_INHERIT = 0x2
_INHERIT_ONLY = 0x8
_lock = threading.Lock()


def _on_windows() -> bool:
    return os.name == "nt"


def enabled() -> bool:
    return os.environ.get(PERSISTENT_GRANTS_ENV, "").strip() != "0"


def record_path() -> Path:
    from .mxc_runtime import _studio_root
    return Path(os.path.abspath(_studio_root() / "mxc-runtime" / "persistent-read-grants.json"))


def _load_record() -> dict[str, str]:
    try:
        data = json.loads(record_path().read_text(encoding = "utf-8"))
    except (OSError, ValueError):
        return {}
    grants = data.get("grants") if isinstance(data, dict) else None
    if not isinstance(grants, dict):
        return {}
    return {str(k): str(v) for k, v in grants.items() if v in {"pending", "complete"}}


def _save_record(grants: dict[str, str]) -> None:
    path = record_path()
    path.parent.mkdir(parents = True, exist_ok = True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps({"grants": grants}, indent = 2, sort_keys = True), encoding = "utf-8")
    os.replace(temporary, path)


def _within(path: str, root: str) -> bool:
    folded, base = os.path.normcase(path), os.path.normcase(root).rstrip("\\/")
    return folded == base or folded.startswith(base + os.sep)


def _protected_paths() -> list[str]:
    from . import os_sandbox
    from .mxc_runtime import dacl_state_path

    paths = [str(dacl_state_path()), str(record_path()), os.path.expanduser("~")]
    paths.extend(os_sandbox.studio_state_roots())
    return [os.path.realpath(p) for p in paths if p]


def ineligible_reason(root: str) -> str | None:
    """Why ``root`` must keep MXC's per-launch grant instead of a persistent one, or None."""
    _drive, tail = os.path.splitdrive(root)
    if not tail.strip("\\/"):
        return "a volume root"
    system_root = os.environ.get("SystemRoot") or os.environ.get("WINDIR")
    if system_root and _within(root, system_root):
        return "inside the Windows directory"
    for protected in _protected_paths():
        # Only a grant ABOVE protected state exposes it; a runtime folder below the Studio home is fine.
        if _within(protected, root):
            return f"it contains {protected}"
    try:
        names = {entry.name.casefold() for entry in os.scandir(root)}
    except OSError as exc:
        return f"it cannot be listed ({exc})"
    found = sorted(names & CREDENTIAL_FILES)
    if found:
        return f"it holds a credential file ({', '.join(found)})"
    return None


def _dacl_grants_read(path: str) -> bool:
    """True when the folder's own DACL gives ALL APPLICATION PACKAGES inheritable read and execute."""
    import ctypes
    from ctypes import wintypes

    advapi32 = ctypes.WinDLL("advapi32", use_last_error = True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error = True)

    class AceHeader(ctypes.Structure):
        _fields_ = [
            ("AceType", ctypes.c_ubyte),
            ("AceFlags", ctypes.c_ubyte),
            ("AceSize", ctypes.c_ushort),
        ]

    class AccessAllowedAce(ctypes.Structure):
        _fields_ = [("Header", AceHeader), ("Mask", wintypes.DWORD), ("SidStart", wintypes.DWORD)]

    class Acl(ctypes.Structure):
        _fields_ = [
            ("AclRevision", ctypes.c_ubyte),
            ("Sbz1", ctypes.c_ubyte),
            ("AclSize", ctypes.c_ushort),
            ("AceCount", ctypes.c_ushort),
            ("Sbz2", ctypes.c_ushort),
        ]

    get_info = advapi32.GetNamedSecurityInfoW
    get_info.argtypes = [
        wintypes.LPCWSTR,
        ctypes.c_int,
        wintypes.DWORD,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.POINTER(Acl)),
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_void_p),
    ]
    get_info.restype = wintypes.DWORD
    get_ace = advapi32.GetAce
    get_ace.argtypes = [ctypes.POINTER(Acl), wintypes.DWORD, ctypes.POINTER(ctypes.c_void_p)]
    get_ace.restype = wintypes.BOOL
    sid_to_string = advapi32.ConvertSidToStringSidW
    sid_to_string.argtypes = [ctypes.c_void_p, ctypes.POINTER(wintypes.LPWSTR)]
    sid_to_string.restype = wintypes.BOOL
    local_free = kernel32.LocalFree
    local_free.argtypes = [ctypes.c_void_p]
    local_free.restype = ctypes.c_void_p

    dacl = ctypes.POINTER(Acl)()
    descriptor = ctypes.c_void_p()
    # SE_FILE_OBJECT = 1, DACL_SECURITY_INFORMATION = 4
    if get_info(path, 1, 4, None, None, ctypes.byref(dacl), None, ctypes.byref(descriptor)) != 0:
        return False
    try:
        if not dacl:
            return False
        allowed = 0
        for index in range(dacl.contents.AceCount):
            raw = ctypes.c_void_p()
            if not get_ace(dacl, index, ctypes.byref(raw)):
                continue
            ace = ctypes.cast(raw, ctypes.POINTER(AccessAllowedAce)).contents
            # ACCESS_ALLOWED_ACE_TYPE = 0, ACCESS_DENIED_ACE_TYPE = 1; object and callback ACEs are ignored.
            if ace.Header.AceType not in (0, 1):
                continue
            sid_text = wintypes.LPWSTR()
            sid = ctypes.c_void_p(raw.value + AccessAllowedAce.SidStart.offset)
            if not sid_to_string(sid, ctypes.byref(sid_text)):
                continue
            try:
                matches = sid_text.value == ALL_APPLICATION_PACKAGES
            finally:
                local_free(ctypes.cast(sid_text, ctypes.c_void_p))
            if not matches:
                continue
            if ace.Header.AceType == 1:
                return False
            flags = ace.Header.AceFlags
            if flags & _INHERIT_ONLY or (flags & (_OBJECT_INHERIT | _CONTAINER_INHERIT)) != (
                _OBJECT_INHERIT | _CONTAINER_INHERIT
            ):
                continue
            allowed |= ace.Mask
        return (allowed & READ_EXECUTE_MASK) == READ_EXECUTE_MASK
    finally:
        local_free(descriptor)


def _icacls(root: str, *args: str) -> tuple[bool, str]:
    system_root = os.environ.get("SystemRoot") or os.environ.get("WINDIR") or "C:\\Windows"
    icacls = os.path.join(system_root, "System32", "icacls.exe")
    try:
        done = subprocess.run(
            [icacls, root, *args, "/Q"],
            capture_output = True,
            text = True,
            errors = "replace",
            timeout = GRANT_TIMEOUT_SECONDS,
            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, str(exc)
    return done.returncode == 0, (done.stdout + done.stderr).strip()[-300:]


def _grant(root: str) -> tuple[bool, str]:
    return _icacls(root, "/grant", f"*{ALL_APPLICATION_PACKAGES}:(OI)(CI)(RX)")


def _revoke(root: str) -> tuple[bool, str]:
    return _icacls(root, "/remove:g", f"*{ALL_APPLICATION_PACKAGES}")


def ensure(roots: list[str]) -> tuple[str, ...]:
    """Give ``roots`` a persistent read grant where eligible; returns the roots that have one.

    Never raises: a root that cannot be granted keeps MXC's per-launch grant, which is only slower.
    """
    if not _on_windows():
        return ()
    if not enabled():
        revoke_recorded()
        return ()
    covered: list[str] = []
    with _lock:
        record = _load_record()
        for root in roots:
            key = os.path.normcase(root)
            pending = record.get(key) == "pending"
            try:
                if not pending and _dacl_grants_read(root):
                    covered.append(root)
                    continue
            except Exception as exc:  # noqa: BLE001 - the check only decides whether to grant
                logger.info("Could not read the DACL of %s: %s", root, exc)
            reason = ineligible_reason(root)
            if reason is not None:
                logger.info("Keeping the per-launch MXC grant for %s: %s", root, reason)
                continue
            record[key] = "pending"
            _save_record(record)
            ok, output = _grant(root)
            if ok:
                record[key] = "complete"
                covered.append(root)
                logger.info("Granted ALL APPLICATION PACKAGES read access to %s once", root)
            else:
                # A partly propagated grant would let wxc-exec skip files the container cannot read.
                _revoke(root)
                record.pop(key, None)
                logger.warning(
                    "Could not add the persistent MXC read grant to %s: %s", root, output
                )
            _save_record(record)
    return tuple(covered)


def revoke_recorded() -> tuple[str, ...]:
    """Remove every grant Studio recorded; returns the roots that were restored."""
    if not _on_windows():
        return ()
    restored: list[str] = []
    with _lock:
        record = _load_record()
        if not record:
            return ()
        for key in list(record):
            if not os.path.isdir(key):
                record.pop(key)
                continue
            ok, output = _revoke(key)
            if ok:
                record.pop(key)
                restored.append(key)
            else:
                logger.warning(
                    "Could not remove the persistent MXC read grant from %s: %s", key, output
                )
        _save_record(record)
    return tuple(restored)
