# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Studio tool launch contract and platform OS-isolation backends."""

from __future__ import annotations
import errno
import functools
import hashlib
import os
import platform
import re
import shutil
import stat
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from dataclasses import replace
from typing import Any, BinaryIO, Callable, Literal
from loggers import get_logger

logger = get_logger(__name__)

ToolExecutionMode = Literal["auto", "required", "full"]
TOOL_EXECUTION_MODES = ("auto", "required", "full")
PUBLIC_TOOL_EXECUTION_MODES = ("auto", "required")

PROFILE_VERSION = "unsloth-sandbox-v1"

# tools.py re-adds this only to an UNISOLATED launch, and only if it already exists.
SESSION_PACKAGES_RELPATH = ".unsloth-packages"


def with_session_packages(env: dict, workdir: str) -> dict:
    """Reuse existing session packages without letting their binaries shadow PATH."""
    packages = os.path.join(workdir, SESSION_PACKAGES_RELPATH)
    if not os.path.isdir(packages):
        return env
    updated = dict(env)
    # Block a planted usercustomize.py; in safe mode the trusted sitecustomize shim stays first on PYTHONPATH.
    updated["PYTHONNOUSERSITE"] = "1"
    # pip --target puts console scripts in Scripts on Windows and bin elsewhere.
    scripts = "Scripts" if os.name == "nt" else "bin"
    for key, value in (
        ("PYTHONPATH", packages),
        ("PATH", os.path.join(packages, scripts)),
    ):
        updated[key] = os.pathsep.join(part for part in (updated.get(key, ""), value) if part)
    return updated


_SOFTWARE_SAFEGUARDS = (
    "process_guard",
    "command_and_code_analysis",
    "sanitized_environment",
    "resource_limits",
    "descriptor_closure",
    "workdir_policy",
    "streaming",
    "timeout",
    "cancellation",
    "reaping",
    "cleanup",
)
_OS_ISOLATION_SAFEGUARDS = _SOFTWARE_SAFEGUARDS + ("filesystem_isolation", "process_isolation")
_FULL_SAFEGUARDS = ("timeout", "cancellation", "reaping", "cleanup")


class SandboxUnavailableError(RuntimeError):
    """OS isolation is unavailable or the requested launch was refused."""

    def __init__(
        self,
        message: str = "",
        *,
        remediation: str = "",
    ) -> None:
        super().__init__(message)
        self.remediation = remediation


class WorkdirUnsafeError(SandboxUnavailableError):
    """Distinguished by type: a transient probe failure must not re-open the channel the scan found."""


class SandboxBuildError(SandboxUnavailableError):
    """Refused, never fallen back: the errno is reachable from inside the jail."""


@dataclass(frozen = True)
class SandboxCapability:
    """``available`` is never inferred from the binary existing: user namespaces may still be denied."""

    backend: str
    available: bool
    reason: str
    environment: str = "unknown"
    protection_state: str = "unavailable"
    profile_id: str = "none"
    limitations: tuple[str, ...] = ()
    probe_generation: str = ""
    environment_fingerprint: str = ""
    remediation: str = ""


@dataclass(frozen = True)
class ToolExecutionRecord:
    """Record the requested policy and the isolation and safeguards actually applied."""

    requested_mode: ToolExecutionMode
    effective_mode: str
    environment: str
    backend: str
    profile_id: str
    probe_generation: str
    os_isolation: bool
    retained_safeguards: tuple[str, ...]
    limitations: tuple[str, ...] = ()
    network_policy: str = "unrestricted"
    backend_tier: str = "unknown"
    runtime_revision: str = ""
    runtime_artifact_digest: str = ""
    schema_version: str = ""
    policy_hash: str = ""
    execution_status: str = "planned"
    completion_status: str = "pending"
    cleanup_status: str = "pending"

    def as_dict(self) -> dict[str, object]:
        return {
            "requested_mode": self.requested_mode,
            "effective_mode": self.effective_mode,
            "environment": self.environment,
            "backend": self.backend,
            "profile_id": self.profile_id,
            "probe_generation": self.probe_generation,
            "os_isolation": self.os_isolation,
            "retained_safeguards": list(self.retained_safeguards),
            "limitations": list(self.limitations),
            "network_policy": self.network_policy,
            "backend_tier": self.backend_tier,
            "runtime_revision": self.runtime_revision,
            "runtime_artifact_digest": self.runtime_artifact_digest,
            "schema_version": self.schema_version,
            "policy_hash": self.policy_hash,
            "execution_status": self.execution_status,
            "completion_status": self.completion_status,
            "cleanup_status": self.cleanup_status,
        }


@dataclass(frozen = True)
class ToolLaunchPlan:
    argv: tuple[str, ...]
    workdir: str
    env: dict[str, str]
    preexec_fn: Callable[[], None] | None = None
    requested_mode: ToolExecutionMode = "auto"
    timeout_seconds: int | None = None
    close_fds: bool = True
    terminate_descendants: bool = True
    # Set by the trusted tool owner, never inferred from model args.
    execution_kind: Literal["python", "terminal"] | None = None
    cancel_event: Any = None


@dataclass
class PreparedSandboxLaunch:
    argv: tuple[str, ...]
    workdir: str
    env: dict[str, str]
    preexec_fn: Callable[[], None] | None
    backend: str
    execution_record: ToolExecutionRecord | None = None
    pass_fds: tuple[int, ...] = ()
    owned_files: list[BinaryIO] = field(default_factory = list)
    cleanup_paths: list[str] = field(default_factory = list)
    timeout_seconds: int | None = None
    close_fds: bool = True
    terminate_descendants: bool = True
    cleanup_callbacks: list[Callable[[], None]] = field(default_factory = list)
    cleanup_diagnostics: list[str] = field(default_factory = list)
    spawn_callback: Callable | None = None
    launch_limitations: tuple[str, ...] = ()

    def cleanup(self) -> None:
        while self.cleanup_callbacks:
            callback = self.cleanup_callbacks.pop()
            try:
                callback()
            except Exception as exc:  # noqa: BLE001 - cleanup continues in LIFO order
                diagnostic = f"{type(exc).__name__}: {exc}"
                self.cleanup_diagnostics.append(diagnostic)
                logger.warning("Sandbox cleanup failed: %s", diagnostic, exc_info = True)
        while self.owned_files:
            try:
                self.owned_files.pop().close()
            except Exception as exc:  # noqa: BLE001 - cleanup must continue
                diagnostic = f"{type(exc).__name__}: {exc}"
                self.cleanup_diagnostics.append(diagnostic)
                logger.warning("Could not close sandbox-owned file: %s", diagnostic, exc_info = True)
        while self.cleanup_paths:
            path = self.cleanup_paths.pop()
            try:
                shutil.rmtree(path)
            except OSError:
                self.cleanup_diagnostics.append(f"could not remove private sandbox path: {path}")
                logger.warning("Could not remove private sandbox path %s", path, exc_info = True)


def spawn_prepared_launch(prepared: PreparedSandboxLaunch, **popen_kwargs: Any) -> object:
    if prepared.spawn_callback is not None:
        return prepared.spawn_callback(prepared, popen_kwargs)
    proc = subprocess.Popen(prepared.argv, **popen_kwargs)
    if prepared.execution_record is not None:
        prepared.execution_record = replace(prepared.execution_record, execution_status = "started")
    return proc


# `auto` launches on this, `required` refuses; one name so both readers agree.
WORKDIR_SCAN_INCOMPLETE = "workdir_scan_incomplete"
WORKDIR_SCAN_ENTRIES = 50_000
WORKDIR_SCAN_SECONDS = 5.0
CACHE_SCAN_ENTRIES = 50_000
CACHE_SCAN_SECONDS = 3.0
_SCAN_JOIN_GRACE_SECONDS = 0.5

TOOL_TEMP_DIRNAME = "unsloth-tmp"
TOOL_TEMP_MODE = 0o700


class _ScanBudgetExceeded(Exception):
    """The walk ran out of budget: not a hazard, so it must not refuse an `auto` launch."""


def directory_signature(path: str) -> tuple:
    """Identity plus mtime for one directory, the unit a cached verdict is re-checked in."""
    try:
        info = os.stat(path)
    except OSError:
        return (path, None)
    return (path, info.st_dev, info.st_ino, info.st_mtime_ns)


def directory_witness_matches(witness: "list[tuple]") -> bool:
    """Whether every directory a finished scan visited is still as it left it."""
    return all(directory_signature(entry[0]) == entry for entry in witness)


def _host_channel_hazard(
    root: str,
    max_entries: int,
    seconds: float,
    witness: "list[tuple] | None" = None,
) -> str | None:
    """Return a host-access hazard under *root*, or None."""
    deadline = time.monotonic() + seconds
    entries = 0
    # Only an unaccounted hard link leads outside; cp -al, git clone --local and pip make nlink > 1.
    links: dict[tuple[int, int], list] = {}
    unreadable: list[str] = []

    for base, dirs, names in os.walk(
        root, followlinks = False, onerror = lambda exc: unreadable.append(exc.filename or root)
    ):
        if witness is not None:
            witness.append(directory_signature(base))
        if unreadable:
            return f"{unreadable[0]} cannot be fully inspected"
        for name in (*dirs, *names):
            entries += 1
            if entries > max_entries or time.monotonic() > deadline:
                raise _ScanBudgetExceeded(
                    f"could not be fully checked for host channels "
                    f"(over {max_entries} entries or {seconds:.0f}s)"
                )
            path = os.path.join(base, name)
            try:
                info = os.lstat(path)
            except OSError:
                return f"changed during its safety scan: {path}"
            # Windows only: MXC grants the workdir by path, so a junction or symlink inside it widens the grant.
            if getattr(info, "st_file_attributes", 0) & 0x400:
                return f"contains a reparse point: {path}"
            if stat.S_ISLNK(info.st_mode):
                continue
            if stat.S_ISDIR(info.st_mode):
                # Misses a same-filesystem bind mount; Linux also asks the mount table.
                if os.path.ismount(path):
                    return f"contains a nested host mount: {path}"
                continue
            if not stat.S_ISREG(info.st_mode):
                return f"contains a device or IPC node: {path}"
            if info.st_nlink > 1:
                found = links.setdefault((info.st_dev, info.st_ino), [0, info.st_nlink, path])
                found[0] += 1
    if unreadable:
        return f"{unreadable[0]} cannot be fully inspected"
    for found, total, path in links.values():
        if found < total:
            return f"contains a file hard-linked from outside it: {path}"
    return None


_scan_lock = threading.Lock()
# root -> (worker, answer, budget, deadline) of the walk in flight, so a concurrent launch shares it.
_scan_pending: "dict[str, tuple[threading.Thread, list, list, float]]" = {}


def _hazard_within_wall_clock(
    root: str,
    max_entries: int,
    seconds: float,
    prepare: "Callable[[float], None] | None" = None,
) -> str | None:
    """Scan on a worker so the budget is wall-clock: scandir/lstat on a stalled mount cannot be interrupted."""
    with _scan_lock:
        pending = _scan_pending.get(root)
        if pending is not None and not pending[0].is_alive():
            del _scan_pending[root]
            pending = None
        if pending is None:
            answer: list[str | None] = []
            budget: list[str] = []
            deadline = time.monotonic() + seconds

            def inspect() -> None:
                try:
                    if prepare is not None:
                        prepare(deadline)
                    answer.append(
                        _host_channel_hazard(
                            root, max_entries, max(0.1, deadline - time.monotonic())
                        )
                    )
                except _ScanBudgetExceeded as exc:
                    budget.append(str(exc))
                except Exception as exc:  # noqa: BLE001 - reported, never raised at the caller
                    budget.append(f"could not be checked for host channels: {exc}")

            worker = threading.Thread(target = inspect, name = "unsloth-workdir-scan", daemon = True)
            # Register under the lock, or a concurrent caller starts a second walk.
            pending = (worker, answer, budget, deadline)
            _scan_pending[root] = pending
            worker.start()
    worker, answer, budget, deadline = pending

    # Waits only out the walk's own budget: a walk a previous launch gave up on is past it already.
    worker.join(max(0.0, deadline + _SCAN_JOIN_GRACE_SECONDS - time.monotonic()))
    with _scan_lock:
        # By identity: another caller may already have replaced the entry.
        if not worker.is_alive() and _scan_pending.get(root) is pending:
            del _scan_pending[root]
    if budget:
        raise _ScanBudgetExceeded(budget[0])
    if not answer:
        raise _ScanBudgetExceeded(
            f"did not finish its host-channel check within {seconds:.0f}s (a wedged mount?)"
        )
    return answer[0]


def _looks_like_our_scratch_dir(path: str) -> bool:
    """Whether this directory was created the way tools.py creates one."""
    try:
        info = os.stat(path)
    except OSError:
        return False
    if hasattr(os, "getuid") and info.st_uid != os.getuid():
        return False
    if sys.platform == "win32":
        return True
    return stat.S_IMODE(info.st_mode) == TOOL_TEMP_MODE


def _ipc_endpoint_is_dead(path: str) -> bool:
    """Whether nothing is listening on this socket; False for anything the probe cannot settle."""
    verdict = _unix_socket_is_refused(path)
    if verdict is None:
        # sun_path is 108 bytes: reach long paths through a temporary symlink alias.
        import tempfile
        alias_dir = None
        try:
            alias_dir = tempfile.mkdtemp(prefix = "unsloth-ipc-")
            alias = os.path.join(alias_dir, "d")
            os.symlink(os.path.dirname(path), alias)
            verdict = _unix_socket_is_refused(os.path.join(alias, os.path.basename(path)))
        except OSError:
            verdict = None
        finally:
            if alias_dir is not None:
                shutil.rmtree(alias_dir, ignore_errors = True)
    return bool(verdict)


def _unix_socket_is_refused(path: str) -> bool | None:
    """True if nothing is listening, False if something is, None if undecided."""
    import socket

    probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        probe.settimeout(0.2)
        probe.connect(path)
    except OSError as exc:
        if exc.errno == errno.ECONNREFUSED:
            return True
        # errno None is CPython's own path-too-long check, not a live endpoint.
        if exc.errno is None or exc.errno in (errno.ENAMETOOLONG, errno.ENOENT):
            return None
        return False
    finally:
        probe.close()
    return False


def clear_stale_tool_ipc(workdir: str, deadline: "float | None" = None) -> tuple[str, ...]:
    """Remove dead sockets from Studio's own 0700 scratch dir; FIFOs are never swept (idle is their normal state)."""
    scratch = os.path.join(workdir, TOOL_TEMP_DIRNAME)
    removed: list[str] = []
    try:
        if not os.path.isdir(scratch) or os.path.islink(scratch):
            return ()
        if not _looks_like_our_scratch_dir(scratch):
            return ()
        for base, dirs, names in os.walk(scratch, followlinks = False):
            dirs[:] = [name for name in dirs if not os.path.ismount(os.path.join(base, name))]
            for name in names:
                if deadline is not None and time.monotonic() > deadline:
                    logger.info("Stopped sweeping the tool scratch directory at its deadline")
                    return tuple(removed)
                path = os.path.join(base, name)
                try:
                    entry = os.lstat(path)
                except OSError:
                    continue
                mode = entry.st_mode
                if not stat.S_ISSOCK(mode):
                    continue
                if entry.st_nlink > 1:
                    # Another link exists: removing this name would hide the walk's hard-link finding.
                    continue
                if not _ipc_endpoint_is_dead(path):
                    logger.info(
                        "Leaving a live IPC endpoint in the tool scratch directory: %s", path
                    )
                    continue
                try:
                    os.unlink(path)
                except OSError:
                    continue
                removed.append(path)
    except OSError:
        return tuple(removed)
    if removed:
        logger.info(
            "Removed %d stale IPC endpoint(s) left in the tool scratch directory: %s",
            len(removed),
            ", ".join(removed[:5]),
        )
    return tuple(removed)


def scan_workdir_for_host_channels(workdir: str) -> tuple[str, ...]:
    """Scan the writable workdir: a hazard fails the call, an overrun budget is recorded, not refused."""
    try:
        hazard = _hazard_within_wall_clock(
            workdir,
            WORKDIR_SCAN_ENTRIES,
            WORKDIR_SCAN_SECONDS,
            prepare = lambda deadline: clear_stale_tool_ipc(workdir, deadline),
        )
    except _ScanBudgetExceeded as exc:
        logger.warning("The session workdir %s: %s", workdir, exc)
        return (WORKDIR_SCAN_INCOMPLETE,)
    if hazard is not None:
        raise WorkdirUnsafeError(f"the session workdir {hazard}")
    return ()


def cache_share_hazard(path: str, witness: "list[tuple] | None" = None) -> str | None:
    """Return a reason not to share this writable cache component, or None."""
    try:
        return _host_channel_hazard(path, CACHE_SCAN_ENTRIES, CACHE_SCAN_SECONDS, witness)
    except _ScanBudgetExceeded as exc:
        return str(exc)


_LINUX_REQUIRED_BINARIES = ("bwrap",)
_FALLBACK_NOTE = "Python and Terminal still run, with software safeguards only and no OS isolation."


@functools.lru_cache(maxsize = 1)
def _linux_userns_blocked_by_apparmor() -> bool:
    """Whether Ubuntu's apparmor_restrict_unprivileged_userns denies bwrap's user namespace (cached)."""
    try:
        with open("/proc/sys/kernel/apparmor_restrict_unprivileged_userns", encoding = "utf-8") as f:
            if f.read().strip() != "1":
                return False
    except OSError:
        return False
    # The sysctl alone is not proof: a profile may permit bwrap.
    try:
        probe = subprocess.run(
            ["unshare", "--user", "--map-root-user", "true"],
            stdin = subprocess.DEVNULL,
            stdout = subprocess.DEVNULL,
            stderr = subprocess.DEVNULL,
            timeout = 10,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return probe.returncode != 0


@functools.lru_cache(maxsize = 1)
def editable_source_roots() -> tuple[str, ...]:
    """Editable code outside site-packages, from PEP 610 records shared by .pth and PEP 660 installs."""
    roots: list[str] = []
    try:
        from importlib import metadata
        import json
        from urllib.parse import unquote, urlparse
    except Exception:  # noqa: BLE001 - never fail a launch over this
        return ()
    try:
        distributions = list(metadata.distributions())
    except Exception:  # noqa: BLE001
        return ()
    for dist in distributions:
        try:
            raw = dist.read_text("direct_url.json")
            if not raw:
                continue
            record = json.loads(raw)
            if not record.get("dir_info", {}).get("editable"):
                continue
            parsed = urlparse(record.get("url", ""))
            if parsed.scheme != "file":
                continue
            path = os.path.abspath(unquote(parsed.path))
        except Exception:  # noqa: BLE001 - a malformed record is not a launch failure
            continue
        if path in ("/", "/usr") or not os.path.isdir(path):
            continue
        for importable in _importable_entries(path, _declared_names(dist)):
            if importable not in roots:
                roots.append(importable)
    return tuple(roots)


def model_library_roots() -> tuple[str, ...]:
    """Registered model folders the approval gate reads silently; NOT the whole read-silent set, which has /etc."""
    try:
        from . import tool_path_approval
        from utils.paths.storage_roots import well_known_model_dirs
        candidates = (
            *tool_path_approval._scan_folder_roots(),
            *well_known_model_dirs(),
        )
    except Exception:  # noqa: BLE001 - never fail a launch over the approval gate
        return ()
    state = studio_state_roots()
    forbidden = _never_a_model_library()
    kept: list[str] = []
    for path in candidates:
        if not path or not os.path.isabs(path):
            continue
        real = os.path.realpath(path)
        # Refuse a registered folder that is a root, a home, the homes parent, or a system directory.
        if _is_filesystem_root(real) or not os.path.isdir(real):
            continue
        if os.path.normcase(real) in forbidden:
            continue
        # The approval gate still asks for a credential inside a model folder; a whole-folder grant would not.
        if tool_path_approval._references_sensitive_path(real + os.sep):
            continue
        if any(_paths_overlap(real, root) for root in state):
            continue
        if real not in kept:
            kept.append(real)
    return tuple(kept)


def _is_filesystem_root(path: str, pathmod: Any = None) -> bool:
    """Whether *path* is a filesystem root; not ``path == os.sep``, which misses Windows drive roots."""
    return (pathmod or os.path).dirname(path) == path


_NEVER_A_MODEL_LIBRARY = (
    "/",
    "/etc",
    "/usr",
    "/var",
    "/opt",
    "/bin",
    "/lib",
    "/home",
    "/Users",
    "/root",
    "/tmp",
)

_WINDOWS_SYSTEM_VARS = (
    "SystemRoot",
    "windir",
    "ProgramFiles",
    "ProgramFiles(x86)",
    "ProgramW6432",
    "ProgramData",
    "PUBLIC",
)


def _never_a_model_library() -> frozenset[str]:
    """Normcased real paths no registered folder may grant, root included."""
    paths = list(_NEVER_A_MODEL_LIBRARY)
    try:
        home = os.path.realpath(os.path.expanduser("~"))
    except OSError:
        home = ""
    if home:
        paths.extend((home, os.path.dirname(home)))
    for name in _WINDOWS_SYSTEM_VARS:
        value = (os.environ.get(name) or "").strip()
        if value and os.path.isabs(value):
            paths.append(os.path.realpath(value))
    return frozenset(os.path.normcase(path) for path in paths if path)


def _paths_overlap(first: str, second: str) -> bool:
    """Whether either path is the other, or contains it."""
    try:
        common = os.path.commonpath([first, second])
    except ValueError:
        return False
    return common in (first, second)


def studio_state_roots() -> tuple[str, ...]:
    """Studio's persisted-state home (auth.db); denied because it may sit under a granted root such as Docker's /opt."""
    roots: list[str] = []
    try:
        from utils.paths.storage_roots import studio_root
        roots.append(os.path.realpath(str(studio_root())))
    except Exception:  # noqa: BLE001 - a launch never fails over this
        pass
    for name in ("UNSLOTH_STUDIO_HOME", "STUDIO_HOME"):
        value = (os.environ.get(name) or "").strip()
        if value:
            roots.append(os.path.realpath(os.path.expanduser(value)))
    return tuple(dict.fromkeys(path for path in roots if path and path != os.sep))


@functools.lru_cache(maxsize = 1)
def editable_import_roots() -> tuple[str, ...]:
    """Import roots need listing: bwrap creates empty parents, Seatbelt needs literal grants."""
    return tuple(dict.fromkeys(os.path.dirname(path) for path in editable_source_roots()))


def _declared_names(dist) -> frozenset[str]:
    """Declared names also identify PEP 420 namespaces, which have no __init__.py."""
    names: set[str] = set()
    try:
        raw = dist.read_text("top_level.txt") or ""
        names.update(line.strip() for line in raw.splitlines() if line.strip())
    except Exception:  # noqa: BLE001 - a missing or unreadable record is not fatal
        pass
    try:
        project = (dist.metadata["Name"] or "").strip()
    except Exception:  # noqa: BLE001
        project = ""
    if project:
        names.add(re.sub(r"[-_.]+", "_", project).lower())
    return frozenset(name for name in names if name and "/" not in name and name != "..")


def _within_root(path: str, root: str) -> bool:
    """Whether *path* stays under *root* once resolved, by whole components."""
    resolved, base = os.path.realpath(path), os.path.realpath(root)
    return resolved == base or resolved.startswith(base.rstrip(os.sep) + os.sep)


def _importable_entries(
    project_root: str, declared: frozenset[str] = frozenset()
) -> tuple[str, ...]:
    """Grant package entries, never the checkout's .env, .git or unrelated fixtures."""
    import_roots = [
        entry
        for entry in sys.path
        if entry
        and (
            os.path.abspath(entry) == project_root
            or os.path.abspath(entry).startswith(project_root + os.sep)
        )
    ]
    # A PEP 660 finder puts nothing on sys.path, so fall back to the common layouts.
    guessed = [
        fallback
        for fallback in (project_root, os.path.join(project_root, "src"))
        if fallback not in import_roots and os.path.isdir(fallback)
    ]
    found: list[str] = []
    for import_root in (*import_roots, *guessed):
        try:
            names = sorted(os.listdir(import_root))
        except OSError:
            continue
        for name in names:
            if name.startswith(".") or name.endswith((".egg-info", ".dist-info")):
                continue
            entry = os.path.join(import_root, name)
            declared_here = name in declared or name.removesuffix(".py") in declared
            if import_root in guessed and not declared_here:
                continue
            # Both backends grant resolved targets, so a declared symlink must stay in the checkout.
            if not _within_root(entry, project_root):
                continue
            package = os.path.isdir(entry) and (
                declared_here or os.path.exists(os.path.join(entry, "__init__.py"))
            )
            module = name.endswith(".py") and (declared_here or os.path.isfile(entry))
            if (package or module) and entry not in found:
                found.append(entry)
    return tuple(found)


# Measured on Ubuntu 24.04: apparmor-profiles ships this profile disabled, under extra-profiles.
_BWRAP_APPARMOR_FIX = (
    "sudo apt-get install -y apparmor-profiles && sudo install -m 644 "
    "/usr/share/apparmor/extra-profiles/bwrap-userns-restrict /etc/apparmor.d/ && "
    "sudo apparmor_parser -r /etc/apparmor.d/bwrap-userns-restrict"
)

# First match wins: an apt host with dnf also on PATH is still apt-managed.
_BWRAP_INSTALL_COMMANDS = (
    ("apt-get", "sudo apt-get install -y bubblewrap"),
    ("dnf", "sudo dnf install -y bubblewrap"),
    ("pacman", "sudo pacman -S --needed bubblewrap"),
    ("zypper", "sudo zypper install -y bubblewrap"),
    ("apk", "sudo apk add bubblewrap"),
)


def bwrap_install_command() -> str | None:
    """The one command that installs bubblewrap here, or None on an unknown package manager."""
    for manager, command in _BWRAP_INSTALL_COMMANDS:
        if shutil.which(manager):
            return command
    return None


def linux_unavailable_remediation() -> str:
    missing = [name for name in _LINUX_REQUIRED_BINARIES if shutil.which(name) is None]
    if missing:
        command = bwrap_install_command()
        # Installing bwrap alone leaves it blocked on Ubuntu 23.10+; keep it one copy-paste.
        if command and "apt-get" in command and _linux_userns_blocked_by_apparmor():
            command = f"{command} && {_BWRAP_APPARMOR_FIX}"
        how = (
            f"run `{command}`"
            if command
            else "install bubblewrap with your distribution's package manager"
        )
        return (
            f"bubblewrap (bwrap) is not installed. To isolate tool calls, {how}; Studio picks "
            f"it up within a minute, no restart needed. {_FALLBACK_NOTE}"
        )
    if _linux_userns_blocked_by_apparmor():
        return (
            "This host denies unprivileged user namespaces "
            "(kernel.apparmor_restrict_unprivileged_userns=1, the default on Ubuntu 23.10 and "
            "newer), so bubblewrap cannot build a sandbox even though it is installed. Load "
            f"Ubuntu's own bwrap profile, which covers only /usr/bin/bwrap: `{_BWRAP_APPARMOR_FIX}`. "
            f"{_FALLBACK_NOTE}"
        )
    return f"This host cannot start an OS sandbox. {_FALLBACK_NOTE}"


def _runtime_identity() -> str:
    """Changes with the interpreter or this module, so a swapped venv re-probes."""
    digest = hashlib.sha256()
    for path in (sys.executable, __file__):
        resolved = os.path.realpath(path)
        digest.update(resolved.encode())
        try:
            info = os.stat(resolved)
            digest.update(str((info.st_size, info.st_mtime_ns)).encode())
        except OSError:
            digest.update(b"missing")
    digest.update((PROFILE_VERSION + sys.platform + platform.release()).encode())
    digest.update((os.path.abspath(sys.executable) + sys.prefix).encode())
    if sys.platform == "linux":
        try:
            from . import sandbox_linux
            digest.update(sandbox_linux.bwrap_identity().encode())
        except Exception as exc:  # noqa: BLE001 - an unavailable backend still needs a cache key
            digest.update(f"untrusted-bwrap:{exc}".encode())
    return digest.hexdigest()


def _unavailable(reason: str, remediation: str, identity: str) -> SandboxCapability:
    return SandboxCapability(
        backend = "none",
        available = False,
        reason = reason,
        environment = sys.platform,
        protection_state = "unavailable",
        limitations = ("no_os_isolation",),
        probe_generation = hashlib.sha256((identity + "unavailable").encode()).hexdigest(),
        environment_fingerprint = identity,
        remediation = remediation,
    )


def capability_snapshot(
    *,
    force: bool = False,
    execution_kind = None,
    selected_executable = None,
    cancel_event = None,
) -> SandboxCapability:
    if sys.platform == "win32":
        from .sandbox_windows_mxc import capability_snapshot as windows_capability
        return windows_capability(
            force = force,
            execution_kind = execution_kind,
            selected_executable = selected_executable,
            cancel_event = cancel_event,
        )
    identity = _runtime_identity()
    if sys.platform == "linux":
        from . import sandbox_linux
        backend = sandbox_linux
    elif sys.platform == "darwin":
        from . import sandbox_macos
        backend = sandbox_macos
    else:
        return _unavailable(
            "OS isolation for Studio tools is available on Linux and macOS only.",
            f"No sandbox backend exists for this platform. {_FALLBACK_NOTE}",
            identity,
        )

    from .sandbox_probe import probe

    available, reason = probe(backend, force = force)
    if not available:
        remediation = (
            linux_unavailable_remediation()
            if sys.platform == "linux"
            else f"This host cannot start an OS sandbox. {_FALLBACK_NOTE}"
        )
        return _unavailable(reason, remediation, identity)
    return SandboxCapability(
        backend = backend.BACKEND_NAME,
        available = True,
        reason = reason,
        environment = sys.platform,
        protection_state = "preview",
        profile_id = backend.PROFILE_ID,
        limitations = backend.LIMITATIONS,
        probe_generation = hashlib.sha256((identity + "available").encode()).hexdigest(),
        environment_fingerprint = identity,
        remediation = (
            "Python and Terminal run with OS isolation. The network is not confined; a tool "
            "call can still reach the internet."
        ),
    )


def _record(
    plan: ToolLaunchPlan,
    capability: SandboxCapability,
    *,
    effective_mode: str,
    os_isolation: bool,
    backend: str,
    profile_id: str,
    safeguards: tuple[str, ...],
    limitations: tuple[str, ...],
) -> ToolExecutionRecord:
    return ToolExecutionRecord(
        requested_mode = plan.requested_mode,
        effective_mode = effective_mode,
        environment = capability.environment,
        backend = backend,
        profile_id = profile_id,
        probe_generation = capability.probe_generation,
        os_isolation = os_isolation,
        retained_safeguards = tuple(
            item for item in safeguards if item != "timeout" or plan.timeout_seconds is not None
        ),
        limitations = limitations,
    )


def _software_only_limitations() -> tuple[str, ...]:
    limitations = ["no_os_isolation", "host_files_readable", "unrestricted_network"]
    if sys.platform != "win32":
        limitations.append("detached_descendant_cleanup_unverified")
    return tuple(limitations)


def prepare_tool_launch(plan: ToolLaunchPlan) -> PreparedSandboxLaunch:
    """Unavailable hosts fall back in auto; unsafe workdirs and build failures refuse."""
    if plan.cancel_event is not None and plan.cancel_event.is_set():
        raise SandboxBuildError("execution cancelled before sandbox preparation")
    if plan.requested_mode not in TOOL_EXECUTION_MODES:
        raise SandboxUnavailableError(f"unknown tool execution mode: {plan.requested_mode!r}")

    if plan.requested_mode == "full":
        return PreparedSandboxLaunch(
            argv = plan.argv,
            workdir = plan.workdir,
            env = plan.env,
            preexec_fn = plan.preexec_fn,
            backend = "none",
            timeout_seconds = plan.timeout_seconds,
            close_fds = plan.close_fds,
            terminate_descendants = plan.terminate_descendants,
            execution_record = ToolExecutionRecord(
                requested_mode = plan.requested_mode,
                effective_mode = "full",
                environment = sys.platform,
                backend = "none",
                profile_id = "full-access",
                probe_generation = "",
                os_isolation = False,
                retained_safeguards = tuple(
                    item
                    for item in _FULL_SAFEGUARDS
                    if item != "timeout" or plan.timeout_seconds is not None
                ),
                limitations = ("security_restrictions_disabled",),
            ),
        )

    capability = capability_snapshot(
        execution_kind = plan.execution_kind,
        selected_executable = plan.argv[0] if plan.argv else None,
        cancel_event = plan.cancel_event,
    )

    if plan.cancel_event is not None and plan.cancel_event.is_set():
        raise SandboxBuildError("execution cancelled before sandbox launch")

    if not capability.available:
        if plan.requested_mode == "required":
            raise SandboxUnavailableError(
                f"OS_ISOLATION_UNAVAILABLE: {capability.reason}",
                remediation = capability.remediation,
            )
        return PreparedSandboxLaunch(
            argv = plan.argv,
            workdir = plan.workdir,
            env = plan.env,
            preexec_fn = plan.preexec_fn,
            backend = "software-safeguards",
            timeout_seconds = plan.timeout_seconds,
            close_fds = plan.close_fds,
            terminate_descendants = plan.terminate_descendants,
            execution_record = _record(
                plan,
                capability,
                effective_mode = "software_safeguards",
                os_isolation = False,
                backend = "software-safeguards",
                profile_id = "software-safeguards-v1",
                safeguards = _SOFTWARE_SAFEGUARDS,
                limitations = _software_only_limitations(),
            ),
        )

    if sys.platform == "win32":
        from .sandbox_windows_mxc import prepare
        return prepare(plan, capability)
    if sys.platform == "linux":
        from . import sandbox_linux as backend
    else:
        from . import sandbox_macos as backend

    try:
        prepared = backend.prepare(plan)
        if plan.requested_mode == "required" and WORKDIR_SCAN_INCOMPLETE in (
            prepared.launch_limitations
        ):
            # `auto` degrades on an overrun scan; `required` fails closed, since the unvisited part may hold a hazard.
            prepared.cleanup()
            raise WorkdirUnsafeError(
                "the session workdir is too large to check for host channels, "
                "and `required` cannot promise a boundary it did not verify. "
                "Start a new chat, or use `auto` to run with software safeguards."
            )
    except OSError as exc:
        # Must be typed: raw, tools.py's general except would fall back to software safeguards.
        raise SandboxBuildError(f"the sandbox could not be built on this host: {exc}") from exc
    prepared.execution_record = _record(
        plan,
        capability,
        effective_mode = "os_isolated",
        os_isolation = True,
        backend = capability.backend,
        profile_id = capability.profile_id,
        safeguards = _OS_ISOLATION_SAFEGUARDS,
        limitations = capability.limitations + prepared.launch_limitations,
    )
    return prepared


def verify_prepared_completion(prepared: PreparedSandboxLaunch, proc) -> dict | None:
    if prepared.backend == "mxc-processcontainer":
        from .sandbox_windows_mxc import verify_success
        return verify_success(prepared, proc)
    reason = getattr(proc, "_unsloth_completion_reason", None)
    if prepared.execution_record is None or reason not in {"finished", "timed_out", "cancelled"}:
        return None
    prepared.execution_record = replace(
        prepared.execution_record,
        execution_status = "completed",
        completion_status = reason,
    )
    return {"timedOut": reason == "timed_out", "cancelled": reason == "cancelled"}


def finalize_prepared_cleanup(prepared: PreparedSandboxLaunch) -> None:
    if prepared.backend == "mxc-processcontainer" or prepared.execution_record is None:
        return
    prepared.execution_record = replace(
        prepared.execution_record,
        cleanup_status = "uncertain" if prepared.cleanup_diagnostics else "complete",
    )
