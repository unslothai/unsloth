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

import contextlib
from contextlib import contextmanager
import json
import logging
import os
from pathlib import Path
import subprocess
import threading
import uuid

logger = logging.getLogger(__name__)

PERSISTENT_GRANTS_ENV = "UNSLOTH_MXC_PERSISTENT_READ_GRANTS"
ALL_APPLICATION_PACKAGES = "S-1-15-2-1"
# FILE_GENERIC_READ | FILE_GENERIC_EXECUTE: wxc-exec's readonly mask and icacls (RX).
READ_EXECUTE_MASK = 0x1200A9
# A tree holding one of these anywhere is never opened to every AppContainer on the machine.
CREDENTIAL_FILES = frozenset(
    {"pip.ini", "pip.conf", "uv.toml", ".pypirc", ".netrc", "_netrc", ".env"}
)
GRANT_TIMEOUT_SECONDS = 900
LOCK_TIMEOUT_SECONDS = GRANT_TIMEOUT_SECONDS + 60

_OBJECT_INHERIT = 0x1
_CONTAINER_INHERIT = 0x2
_INHERIT_ONLY = 0x8
_INHERITED = 0x10
_lock = threading.Lock()
# Root -> folder identity whose whole tree passed the credential scan in this process; a folder
# replaced at the same path, or a new process, scans again.
_scanned: dict[str, dict[str, int] | None] = {}


class ReadGrantError(RuntimeError):
    """A persistent grant is in an unknown state, so this launch must not rely on it."""


def _on_windows() -> bool:
    return os.name == "nt"


def enabled() -> bool:
    """The environment variable decides whenever it is set; otherwise the Settings > Sandbox choice."""
    if PERSISTENT_GRANTS_ENV in os.environ:
        return os.environ[PERSISTENT_GRANTS_ENV].strip() != "0"
    try:
        from utils.mxc_isolation_settings import persistent_grants_setting
        return persistent_grants_setting()
    except Exception:  # noqa: BLE001 - outside the backend the shipped default applies
        return True


def record_path() -> Path:
    from .mxc_runtime import _studio_root
    return Path(os.path.abspath(_studio_root() / "mxc-runtime" / "persistent-read-grants.json"))


@contextmanager
def _transaction():
    """Serialize the record across threads and Studio processes; yields None when it cannot be locked."""
    from filelock import FileLock, Timeout

    path = record_path()
    with _lock:
        try:
            path.parent.mkdir(parents = True, exist_ok = True)
            lock = FileLock(str(path) + ".lock", timeout = LOCK_TIMEOUT_SECONDS)
            lock.acquire()
        except Timeout as exc:
            # Another Studio process is still granting: its tree may be half propagated.
            raise ReadGrantError(
                f"another Studio process holds the MXC read-grant record: {exc}"
            ) from exc
        except OSError as exc:
            # No lock file means no process can be granting through this record either.
            logger.warning("Could not lock the MXC read-grant record: %s", exc)
            yield None
            return
        try:
            yield _load_record()
        finally:
            lock.release()


def _load_record() -> dict[str, dict]:
    try:
        data = json.loads(record_path().read_text(encoding = "utf-8"))
    except (OSError, ValueError):
        return {}
    grants = data.get("grants") if isinstance(data, dict) else None
    if not isinstance(grants, dict):
        return {}
    return {
        str(key): value
        for key, value in grants.items()
        if isinstance(value, dict) and value.get("state") in {"pending", "complete"}
    }


def _save_record(grants: dict[str, dict]) -> None:
    path = record_path()
    path.parent.mkdir(parents = True, exist_ok = True)
    temporary = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps({"grants": grants}, indent = 2, sort_keys = True), encoding = "utf-8"
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _within(path: str, root: str) -> bool:
    folded, base = os.path.normcase(path), os.path.normcase(root).rstrip("\\/")
    return folded == base or folded.startswith(base + os.sep)


def _protected_paths() -> list[str]:
    from . import os_sandbox
    from .mxc_runtime import dacl_state_path

    paths = [str(dacl_state_path()), str(record_path()), os.path.expanduser("~")]
    paths.extend(os_sandbox.studio_state_roots())
    return [os.path.realpath(p) for p in paths if p]


def _is_reparse(entry: os.DirEntry) -> bool:
    try:
        attributes = getattr(entry.stat(follow_symlinks = False), "st_file_attributes", 0)
    except OSError:
        return True
    return entry.is_symlink() or bool(attributes & 0x400)


def _links_within(path: str, real_root: str) -> bool:
    """A symlink or junction whose final target stays inside the granted tree."""
    try:
        os.readlink(path)  # only symlinks and junctions; other reparse tags raise
        return _within(os.path.realpath(path), real_root)
    except (OSError, ValueError):
        return False


def _tree_problem(root: str, *, deep: bool) -> str | None:
    """A credential file or reparse point in the tree the inheritable grant would reach."""
    real_root = os.path.realpath(root)
    pending = [root]
    while pending:
        directory = pending.pop()
        try:
            with os.scandir(directory) as entries:
                for entry in entries:
                    if entry.name.casefold() in CREDENTIAL_FILES:
                        return f"it holds a credential file ({entry.path})"
                    if _is_reparse(entry):
                        if _links_within(entry.path, real_root):
                            continue  # e.g. setup-python's python3.exe -> python.exe; the target is scanned anyway
                        return f"it contains a reparse point ({entry.path})"
                    if deep and entry.is_dir(follow_symlinks = False):
                        pending.append(entry.path)
        except OSError as exc:
            return f"it cannot be inspected ({exc})"
    return None


def ineligible_reason(root: str, *, deep: bool = True) -> str | None:
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
    return _tree_problem(root, deep = deep)


def _identity(root: str) -> dict[str, int] | None:
    from .mxc_policy import MxcPolicyError, _object_identity, _safe_canonical_path
    try:
        if os.path.normcase(_safe_canonical_path(root, directory = True)) != os.path.normcase(root):
            return None
        return _object_identity(root, directory = True)
    except (MxcPolicyError, OSError):
        return None


def _read_execute_covered(aces: list[tuple[int, int]]) -> bool:
    """Check this folder and inheritable file/directory access separately.

    Windows' standard Program Files ACL splits (RX) on the folder from
    inherit-only (OI)(CI)(GR,GE) on children. Generic rights in the latter
    must be mapped to file rights before comparing them with (RX).
    """
    folder = files = directories = 0
    for mask, flags in aces:
        if mask & 0x10000000:  # GENERIC_ALL
            mask |= 0x1F01FF
        if mask & 0x80000000:  # GENERIC_READ
            mask |= 0x120089
        if mask & 0x40000000:  # GENERIC_WRITE
            mask |= 0x120116
        if mask & 0x20000000:  # GENERIC_EXECUTE
            mask |= 0x1200A0
        if not flags & _INHERIT_ONLY:
            folder |= mask
        if flags & 0x4:  # NO_PROPAGATE_INHERIT cannot cover the whole tree.
            continue
        if flags & _OBJECT_INHERIT:
            files |= mask
        if flags & _CONTAINER_INHERIT:
            directories |= mask
    return all(
        (mask & READ_EXECUTE_MASK) == READ_EXECUTE_MASK for mask in (folder, files, directories)
    )


def _package_aces(path: str) -> tuple[bool, bool]:
    """(covers, explicit) for ALL APPLICATION PACKAGES on the folder's own DACL.

    covers: inheritable read and execute is granted. explicit: the folder carries its own ACE for
    that SID. Raises OSError when the DACL cannot be read, so an unknown ACL is never modified.
    """
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
    status = get_info(path, 1, 4, None, None, ctypes.byref(dacl), None, ctypes.byref(descriptor))
    if status != 0:
        raise OSError(status, f"GetNamedSecurityInfoW failed for {path}")
    try:
        if not dacl:
            # A NULL DACL already grants everyone, AppContainers included: nothing to add.
            return True, False
        allowed: list[tuple[int, int]] = []
        explicit = False
        for index in range(dacl.contents.AceCount):
            raw = ctypes.c_void_p()
            if not get_ace(dacl, index, ctypes.byref(raw)):
                raise OSError(ctypes.get_last_error(), f"GetAce failed for {path}")
            ace = ctypes.cast(raw, ctypes.POINTER(AccessAllowedAce)).contents
            # ACCESS_ALLOWED_ACE_TYPE = 0 and ACCESS_DENIED_ACE_TYPE = 1 share this layout.
            if ace.Header.AceType not in (0, 1):
                continue
            sid_text = wintypes.LPWSTR()
            sid = ctypes.c_void_p(raw.value + AccessAllowedAce.SidStart.offset)
            if not sid_to_string(sid, ctypes.byref(sid_text)):
                raise OSError(ctypes.get_last_error(), f"ConvertSidToStringSidW failed for {path}")
            try:
                matches = sid_text.value == ALL_APPLICATION_PACKAGES
            finally:
                local_free(ctypes.cast(sid_text, ctypes.c_void_p))
            if not matches:
                continue
            flags = ace.Header.AceFlags
            if not flags & _INHERITED:
                explicit = True
            if ace.Header.AceType == 1:
                return False, explicit
            allowed.append((ace.Mask, flags))
        return _read_execute_covered(allowed), explicit
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
            # icacls writes the console (OEM) codepage to a pipe; the codec exists only on Windows.
            encoding = "oem" if os.name == "nt" else "utf-8",
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


def _save_quietly(record: dict) -> None:
    """Persist a record change after the ACL already matches it; a failed write only costs a retry."""
    try:
        _save_record(record)
    except OSError as exc:
        logger.warning("Could not update the MXC read-grant record: %s", exc)


def _pending_grant_has_no_explicit_aces(root: str, identity: dict) -> bool:
    """Prove a failed attempt left no explicit package ACE anywhere in its tree.

    Checking only the root would miss an interrupted propagation or rollback.
    Unknown ACLs and reparse points keep the recovery record intact.
    """
    pending = [root]
    try:
        while pending:
            directory = pending.pop()
            with os.scandir(directory) as entries:
                for entry in entries:
                    if _is_reparse(entry) or _package_aces(entry.path)[1]:
                        return False
                    if entry.is_dir(follow_symlinks = False):
                        pending.append(entry.path)
        covers, explicit = _package_aces(root)
        return covers and not explicit and _identity(root) == identity
    except OSError:
        return False


def _revoke_recorded_root(record: dict, key: str) -> str:
    """Take back one recorded grant: "revoked", "dropped" (nothing of Studio's left there), or "failed"."""
    if not os.path.isdir(key):
        record.pop(key)
        return "dropped"
    current = _identity(key)
    if current is None:
        logger.warning(
            "Not revoking the MXC read grant on %s yet: its identity could not be read", key
        )
        return "failed"
    if current != record[key].get("identity"):
        # Moved, replaced, or reached through a new junction: revoking would edit a folder Studio never touched.
        logger.warning(
            "Not revoking the MXC read grant on %s: it is no longer the folder Studio granted", key
        )
        record.pop(key)
        return "dropped"
    if record[key].get("state") == "pending" and _pending_grant_has_no_explicit_aces(key, current):
        # Only inherited Windows permissions remain; none belong to Studio.
        record.pop(key)
        return "dropped"
    ok, output = _revoke(key)
    if ok:
        record.pop(key)
        _scanned.pop(key, None)
        return "revoked"
    logger.warning("Could not remove the persistent MXC read grant from %s: %s", key, output)
    return "failed"


def _ensure_root(record: dict, root: str) -> bool:
    """Grant one root if it is eligible; True when wxc-exec will skip it. Saves the record as it goes.

    Raises ReadGrantError when a grant Studio started may be half propagated and cannot be settled.
    """
    key = os.path.normcase(root)
    entry = record.get(key)
    pending = entry is not None and entry.get("state") == "pending"
    current = _identity(root)
    if entry is not None:
        if current is None:
            if pending:
                raise ReadGrantError(f"the unfinished MXC read grant on {root} cannot be checked")
            logger.info(
                "Keeping the per-launch MXC grant for %s: its identity could not be read", root
            )
            return False
        if current != entry.get("identity"):
            # Another folder now sits at this path; Studio's claim is void and its ACL is not Studio's.
            record.pop(key)
            _save_quietly(record)
            entry, pending = None, False
    deep = key not in _scanned or _scanned[key] != current
    reason = ineligible_reason(root, deep = deep)
    if reason is not None:
        _scanned.pop(key, None)
        if entry is not None:
            # A folder Studio granted gained a credential file or a link: take the grant back.
            outcome = _revoke_recorded_root(record, key)
            _save_quietly(record)
            if outcome == "failed":
                raise ReadGrantError(f"could not revoke the MXC read grant on {root} ({reason})")
        logger.info("Keeping the per-launch MXC grant for %s: %s", root, reason)
        return False
    _scanned[key] = current
    try:
        covers, explicit = _package_aces(root)
    except OSError as exc:
        if pending:
            raise ReadGrantError(
                f"the unfinished MXC read grant on {root} cannot be checked: {exc}"
            ) from exc
        logger.info("Keeping the per-launch MXC grant for %s: %s", root, exc)
        return False
    if pending and covers and not explicit and _pending_grant_has_no_explicit_aces(root, current):
        record.pop(key)
        _save_quietly(record)
        return True
    if covers and not pending:
        return True
    if explicit and entry is None:
        # Someone else set an entry for that group here; /remove:g on opt-out would take theirs too.
        logger.info(
            "Keeping the per-launch MXC grant for %s: it already has its own entry for that group",
            root,
        )
        return False
    identity = _identity(root)
    if identity is None:
        if pending:
            raise ReadGrantError(f"the unfinished MXC read grant on {root} cannot be checked")
        logger.info("Keeping the per-launch MXC grant for %s: its identity could not be read", root)
        return False
    record[key] = {"state": "pending", "identity": identity}
    try:
        _save_record(record)
    except OSError as exc:
        if pending:
            raise ReadGrantError(
                f"could not record the retried MXC read grant on {root}: {exc}"
            ) from exc
        record.pop(key, None)
        logger.warning(
            "Keeping the per-launch MXC grant for %s: the grant record is not writable (%s)",
            root,
            exc,
        )
        return False
    ok, output = _grant(root)
    if ok:
        record[key] = {"state": "complete", "identity": identity}
        # On a failed write the pending entry stays on disk, so the next launch redoes the grant.
        _save_quietly(record)
        logger.info("Granted ALL APPLICATION PACKAGES read access to %s once", root)
        return True
    # A partly propagated grant would let wxc-exec skip files the container cannot read. The pending
    # entry stays on disk until the rollback is known to have hit the folder Studio granted.
    if _identity(root) != identity:
        raise ReadGrantError(
            f"the MXC read grant on {root} failed ({output}) and the folder changed"
        )
    restored, restore_output = _revoke(root)
    if not restored:
        raise ReadGrantError(
            f"the MXC read grant on {root} failed ({output}) and could not be rolled back ({restore_output})"
        )
    record.pop(key, None)
    _save_quietly(record)
    logger.warning("Could not add the persistent MXC read grant to %s: %s", root, output)
    return False


def ensure(roots: list[str]) -> tuple[str, ...]:
    """Give ``roots`` a persistent read grant where eligible; returns the roots wxc-exec will skip.

    A root that cannot be granted keeps MXC's per-launch grant, which is only slower. Raises
    ReadGrantError only when a grant is left in an unknown state.
    """
    if not _on_windows():
        return ()
    if not enabled():
        revoke_recorded()
        return ()
    covered: list[str] = []
    with _transaction() as record:
        if record is None:
            return ()
        for root in roots:
            if _ensure_root(record, root):
                covered.append(root)
    return tuple(covered)


def _leases_dir() -> Path:
    return record_path().parent / "read-grant-users"


def _lease_is_live(path: Path) -> bool:
    """Windows refuses to delete a file a process still holds open, so a crashed holder's lease goes."""
    try:
        path.unlink()
    except FileNotFoundError:
        return False
    except OSError:
        return True
    return False


def _live_leases() -> int:
    try:
        entries = list(_leases_dir().iterdir())
    except OSError:
        return 0
    return sum(_lease_is_live(entry) for entry in entries)


class WorkloadLease:
    """Held open for one MXC workload's lifetime: the ACEs it may have skipped stay until release."""

    def __init__(self, path: Path, handle):
        self._path, self._handle = path, handle

    def release(self) -> None:
        if self._handle is None:
            return
        self._handle.close()
        self._handle = None
        with contextlib.suppress(OSError):
            self._path.unlink()
        _revoke_if_turned_off()


def hold() -> WorkloadLease | None:
    """Mark a launch as relying on the grants from spawn to exit, across Studio processes.

    Raises ReadGrantError when the lease cannot be recorded: a launch that may skip wxc-exec's own
    grant must not run with nothing stopping a revocation under it.
    """
    if not _on_windows():
        return None
    path = _leases_dir() / f"{os.getpid()}-{uuid.uuid4().hex}"
    try:
        # Under the record lock: a revocation either finished before this or sees the lease.
        with _transaction():
            path.parent.mkdir(parents = True, exist_ok = True)
            handle = open(path, "w", encoding = "utf-8")
    except OSError as exc:
        raise ReadGrantError(
            f"could not record this MXC workload for the read grants: {exc}"
        ) from exc
    return WorkloadLease(path, handle)


def hold_if_needed() -> WorkloadLease | None:
    """A lease for every MXC launch or probe, so no revocation runs under it.

    Taken while a switch is off too: a revocation deferred for a running workload leaves the entries
    in place (wxc-exec then skips its own grant), and a switch turned on before the request is built
    installs them. Only a launch that needs the grants refuses when the lease cannot be recorded.
    """
    from . import mxc_policy

    refresh_saved_switches()
    needed = mxc_policy.dacl_fallback_enabled() and enabled()
    try:
        return hold()
    except ReadGrantError:
        if needed:
            raise
        return None


def refresh_saved_switches() -> None:
    """Drop this process's 1 s settings cache: another Studio process may have just turned a switch off."""
    try:
        from utils.mxc_isolation_settings import forget_cached_setting
        forget_cached_setting()
    except Exception:  # noqa: BLE001 - outside the backend there is no saved setting to cache
        pass


def _revoke_if_turned_off() -> None:
    """Finish a revocation the settings switch or an earlier launch deferred for running workloads."""
    try:
        from . import mxc_policy
        refresh_saved_switches()
        if not (mxc_policy.dacl_fallback_enabled() and enabled()):
            revoke_recorded()
    except Exception as exc:  # noqa: BLE001 - cleanup, the next launch or check retries
        logger.warning("Deferred MXC read-grant cleanup failed: %s", exc)


def revoke_recorded() -> tuple[str, ...]:
    """Remove every grant Studio recorded; returns the roots that were restored.

    Waits while an MXC workload holds a lease: its container may read through these ACEs, so the
    last lease to release finishes the job.
    """
    if not _on_windows() or not record_path().exists():
        return ()
    restored: list[str] = []
    try:
        transaction = _transaction()
        record = transaction.__enter__()
    except ReadGrantError as exc:
        # Cleanup never blocks a launch; the entries stay and the next check retries.
        logger.warning("Deferring the MXC read-grant cleanup: %s", exc)
        return ()
    with contextlib.ExitStack() as stack:
        stack.push(transaction)
        if not record:
            return ()
        running = _live_leases()
        if running:
            logger.info(
                "Deferring the MXC read-grant cleanup until %d running call(s) exit", running
            )
            return ()
        for key in list(record):
            if _revoke_recorded_root(record, key) == "revoked":
                restored.append(key)
        # A failed revoke keeps its entry, so the next launch or capability check retries it.
        _save_quietly(record)
    return tuple(restored)
