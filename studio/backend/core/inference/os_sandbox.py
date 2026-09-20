# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Studio tool launch contract and the OS-isolation backends (bwrap, Seatbelt)."""

from __future__ import annotations
import functools
import hashlib
import os
import platform
import re
import shutil
import stat
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Any, BinaryIO, Callable, Literal
from loggers import get_logger

logger = get_logger(__name__)

ToolExecutionMode = Literal["auto", "required", "full"]
TOOL_EXECUTION_MODES = ("auto", "required", "full")

PROFILE_VERSION = "unsloth-sandbox-v1"

# tools.py re-adds this to an UNISOLATED launch only if it already exists, which
# keeps a host that never isolates byte-identical to main.
SESSION_PACKAGES_RELPATH = ".unsloth-packages"

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
    """Distinguished by TYPE, not by re-probing: a transient probe failure must
    not re-open the very channel the workdir scan just found."""


class SandboxBuildError(SandboxUnavailableError):
    """The probe passed but this launch could not be built. Refused, not fallen
    back: the errno is reachable from inside the jail, so a tool call that fills
    the disk could otherwise buy itself an unisolated launch."""


@dataclass(frozen = True)
class SandboxCapability:
    """``available`` is never inferred from a binary being on disk: an installed
    bwrap on a host that denies user namespaces looks identical until you try."""

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
    # Always "unrestricted": this confines the filesystem, not the network.
    network_policy: str = "unrestricted"

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
    # The subset of the above that means HOST state this launch changed may not
    # have been changed back: on Windows Tier 3 that is DENY and ALLOW ACEs on
    # the user's own workdir. Kept apart from the rest because a private mount
    # that could not be unlinked is the sandbox's own litter and nobody outside
    # needs telling, while a permission change left on a user's files does.
    unreverted_host_state: list[str] = field(default_factory = list)
    # Earned by THIS launch, on top of the backend's static set: what was true
    # of this call and may not be true of the next one.
    launch_limitations: tuple[str, ...] = ()

    def cleanup(self) -> None:
        while self.cleanup_callbacks:
            callback = self.cleanup_callbacks.pop()
            try:
                callback()
            except Exception as exc:  # noqa: BLE001 - cleanup continues in LIFO order
                diagnostic = f"{type(exc).__name__}: {exc}"
                self.cleanup_diagnostics.append(diagnostic)
                self.unreverted_host_state.append(diagnostic)
                logger.error("Sandbox cleanup failed: %s", diagnostic, exc_info = True)
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
    return subprocess.Popen(prepared.argv, **popen_kwargs)


# Recorded when the walk ran out of budget before it could reach a verdict.
# `auto` carries it and launches; `required` refuses on it. Named rather than
# spelled twice, because the two readers must agree.
WORKDIR_SCAN_INCOMPLETE = "workdir_scan_incomplete"
WORKDIR_SCAN_ENTRIES = 50_000
WORKDIR_SCAN_SECONDS = 5.0
# The shared cache is walked per launch, so its budget is tighter than the
# workdir's; a cache too big to check in it is simply not shared.
CACHE_SCAN_ENTRIES = 50_000
CACHE_SCAN_SECONDS = 3.0


class _ScanBudgetExceeded(Exception):
    """The walk ran out of entries or time before it could reach a verdict.

    Distinct from a hazard, because the two call for opposite answers. A hazard
    is a finding about the tree; this is only a statement about the scan's cost,
    and the tree may be perfectly safe. Treating the two alike is what let an
    ordinary `pip install` end a chat: past the entry cap every later Python and
    Terminal call was refused, and since the scan is also what must run before a
    tool can delete anything, the chat could never clean itself up.
    """


def _host_channel_hazard(root: str, max_entries: int, seconds: float) -> str | None:
    """Return a host-access hazard under *root*, or None.

    Reject sockets, devices, FIFOs, external hard links and nested mounts.
    Tool-created entries cannot be distinguished from host entries. The root
    itself may be a mount point. Raises ``_ScanBudgetExceeded`` if the walk
    cannot finish inside its budget; callers decide what an unfinished scan means
    for them, because it is not a finding.
    """
    deadline = time.monotonic() + seconds
    entries = 0
    # Refusing every st_nlink > 1 would refuse any tree built by `cp -al`,
    # `git clone --local` or pip; only an unaccounted link leads outside.
    links: dict[tuple[int, int], list] = {}
    unreadable: list[str] = []

    for base, dirs, names in os.walk(
        root, followlinks = False, onerror = lambda exc: unreadable.append(exc.filename or root)
    ):
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


def scan_workdir_for_host_channels(workdir: str) -> tuple[str, ...]:
    """The writable session workdir. A hazard here fails the call.

    Returns the per-launch limitations the scan earned, so an unfinished scan is
    reported rather than silently treated as a clean one.

    A scan that overruns its budget does NOT fail the call. Refusing looked
    conservative and was not: the call it refused was a confined one, and the
    user was left with no working tool rather than with a weaker boundary. The
    sandbox is still built, and the record says the workdir was not fully
    inspected. Genuine findings above remain fatal, so a planted socket or an
    external hard link still stops the launch.
    """
    try:
        hazard = _host_channel_hazard(workdir, WORKDIR_SCAN_ENTRIES, WORKDIR_SCAN_SECONDS)
    except _ScanBudgetExceeded as exc:
        logger.warning("The session workdir %s: %s", workdir, exc)
        return (WORKDIR_SCAN_INCOMPLETE,)
    if hazard is not None:
        raise WorkdirUnsafeError(f"the session workdir {hazard}")
    return ()


def cache_share_hazard(path: str) -> str | None:
    """Return a reason not to share this writable cache component, or None.

    Apply the workdir's host-access checks: sockets and hard links can expose
    host resources. Unsafe components are omitted, not launch failures, so a
    planted socket cannot disable later calls. Missing caches are re-downloaded.

    Unlike the workdir, a cache that overruns its budget IS a reason not to share
    it: the component is simply omitted and re-downloaded inside, which costs
    bandwidth and nothing else.
    """
    try:
        return _host_channel_hazard(path, CACHE_SCAN_ENTRIES, CACHE_SCAN_SECONDS)
    except _ScanBudgetExceeded as exc:
        return str(exc)


_LINUX_REQUIRED_BINARIES = ("bwrap",)
_FALLBACK_NOTE = "Python and Terminal still run, with software safeguards only and no OS isolation."


@functools.lru_cache(maxsize = 1)
def _linux_userns_blocked_by_apparmor() -> bool:
    """Whether Ubuntu 23.10+'s ``apparmor_restrict_unprivileged_userns`` denies the
    user namespace bwrap needs. Cached: it forks, and every snapshot reaches it."""
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


def _path_from_file_url(parsed, is_windows: bool | None = None) -> str:
    """Decode a PEP 610 ``file:`` URL into a path this platform can open.

    ``abspath(unquote(parsed.path))`` is right on POSIX and wrong on Windows.
    The standard form ``file:///C:/Users/me/project`` has a path component of
    ``/C:/Users/me/project``; that leading slash is URL syntax, and Windows
    ``abspath`` reads it as drive-root-relative and yields
    ``\\C:\\Users\\me\\project``. The following ``isdir`` then drops the
    editable source root, so the sandbox policy never grants the tree the
    PEP 660 finder needs and editable imports fail inside an isolated call.

    The authority is handled separately because on Windows it is a UNC host
    rather than part of the path at all. ``is_windows`` is a parameter so the
    Windows decode is testable from any host; it defaults to the real one.
    """
    from urllib.parse import unquote

    windows = (os.name == "nt") if is_windows is None else is_windows
    host = (parsed.netloc or "").strip()
    path = unquote(parsed.path or "")
    if not windows:
        if host and host.lower() != "localhost":
            return ""  # a remote host is not a local editable source root
        return os.path.abspath(path)
    path = path.replace("/", "\\")
    if host and host.lower() != "localhost":
        return "\\\\" + host + path
    if len(path) >= 3 and path[0] == "\\" and path[2] == ":":
        path = path[1:]
    return path


@functools.lru_cache(maxsize = 1)
def editable_source_roots() -> tuple[str, ...]:
    """Editable code outside site-packages, from PEP 610 records shared by .pth and PEP 660 installs."""
    roots: list[str] = []
    try:
        from importlib import metadata
        import json
        from urllib.parse import urlparse
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
            path = _path_from_file_url(parsed)
        except Exception:  # noqa: BLE001 - a malformed record is not a launch failure
            continue
        if not path or path in ("/", "/usr") or not os.path.isdir(path):
            continue
        for importable in _importable_entries(path, _declared_names(dist)):
            if importable not in roots:
                roots.append(importable)
    return tuple(roots)


def model_library_roots() -> tuple[str, ...]:
    """The model folders a tool is already allowed to read without asking.

    tool_path_approval treats the folders the user registered with Studio, and
    the well-known LM Studio and Ollama locations, as read-silent: "these hold
    weights, not documents, and reading them is the point of the app". Without
    granting them the two halves of the product disagree, because a read from
    a registered folder passes the approval gate silently, as designed, and
    then fails inside the sandbox on every `auto` launch.

    Deliberately NOT the whole read-silent set. That also carries /etc and
    other system directories, where staying silent at the approval gate is
    reasonable and binding the directory into the jail is not: it would hand
    over /etc/ssl/private, which the bind list excludes one file at a time.

    Read only; the write-silent set is not consulted, since the sandbox keeps
    writes to the session workdir. Degraded to nothing rather than raising,
    because a launch must not fail over this.
    """
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
    kept: list[str] = []
    for path in candidates:
        if not path or not os.path.isabs(path):
            continue
        real = os.path.realpath(path)
        if real == os.sep or not os.path.isdir(real):
            continue
        home = os.path.realpath(os.path.expanduser("~"))
        # A registered folder that IS a home, a system directory, or that holds
        # Studio's own state is a misconfiguration, and binding it would undo
        # the rest of the profile.
        if real == home or real in _NEVER_A_MODEL_LIBRARY:
            continue
        if any(_paths_overlap(real, root) for root in state):
            continue
        if real not in kept:
            kept.append(real)
    return tuple(kept)


_NEVER_A_MODEL_LIBRARY = frozenset(
    ("/", "/etc", "/usr", "/var", "/opt", "/bin", "/lib", "/home", "/Users", "/root", "/tmp")
)


def _paths_overlap(first: str, second: str) -> bool:
    """Whether either path is the other, or contains it."""
    try:
        common = os.path.commonpath([first, second])
    except ValueError:
        return False
    return common in (first, second)


def studio_state_roots() -> tuple[str, ...]:
    """Where Studio keeps ``auth/auth.db`` and the rest of its persisted state.

    The default lives under ``$HOME``, which no backend grants, but a custom
    home can sit anywhere, including inside a directory a backend DOES grant:
    the shipped Docker layout puts it at ``/opt/unsloth-studio`` (docker/run.sh
    mounts the volume there, studio_launch.sh exports it) and ``/opt`` is a
    Linux read root, while a macOS install under a Homebrew prefix lands inside
    an optional read root. That database holds the HS256 jwt_secret, and
    tools.py guards the literal path, so a path built at run time walks past the
    guard and the sandbox must not be the thing that hands the file over.

    Not cached: the resolution reads the environment, and a test or a restarted
    Studio with a different home must not inherit an earlier answer.
    """
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
    """Grant package entries, never the checkout's .env, .git or unrelated fixtures.

    Use the installer's sys.path root when present; an unconfirmed import gets no grant.
    """
    import_roots = [
        entry
        for entry in sys.path
        if entry
        and (
            os.path.abspath(entry) == project_root
            or os.path.abspath(entry).startswith(project_root + os.sep)
        )
    ]
    # A PEP 660 finder puts nothing on sys.path, so fall back to the two layouts
    # that cover almost everything published.
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
                # A guessed layout may include unrelated deploy.py or tests/ packages.
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


def linux_unavailable_remediation() -> str:
    missing = [name for name in _LINUX_REQUIRED_BINARIES if shutil.which(name) is None]
    if missing:
        return (
            f"Install the missing Linux prerequisites ({', '.join(missing)}) with your "
            f"distribution's package manager, or re-run Studio setup. {_FALLBACK_NOTE}"
        )
    if _linux_userns_blocked_by_apparmor():
        return (
            "This host denies unprivileged user namespaces "
            "(kernel.apparmor_restrict_unprivileged_userns=1, the default on Ubuntu 23.10 and "
            "newer), so bubblewrap cannot build a sandbox even though it is installed. Grant "
            "bwrap the userns permission with an AppArmor profile "
            "(/etc/apparmor.d/bwrap-userns-restrict from the apparmor-profiles package), or "
            f"re-run Studio setup, which offers to install it. {_FALLBACK_NOTE}"
        )
    return f"This host cannot start an OS sandbox. {_FALLBACK_NOTE}"


def _runtime_identity() -> str:
    """Changes when the interpreter or this module changes, so swapping the venv
    under a running Studio re-probes instead of reusing a stale verdict."""
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
    return digest.hexdigest()


# Windows isolation is opt-in while MXC is an early preview upstream. This is
# eligibility for the BACKEND, deliberately a separate axis from the execution
# mode: a model-authored tool argument must never be able to select a backend or
# relax a policy, so it is read from the environment Studio was started with and
# never from a request.
WINDOWS_PREVIEW_ENV = "UNSLOTH_WINDOWS_SANDBOX_PREVIEW"


def windows_preview_enabled() -> bool:
    return os.environ.get(WINDOWS_PREVIEW_ENV, "").strip().lower() in ("1", "true", "yes", "on")


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


def capability_snapshot(*, force: bool = False) -> SandboxCapability:
    identity = _runtime_identity()
    if sys.platform == "linux":
        from . import sandbox_linux
        backend = sandbox_linux
    elif sys.platform == "darwin":
        from . import sandbox_macos
        backend = sandbox_macos
    elif sys.platform == "win32" and windows_preview_enabled():
        from . import sandbox_windows
        backend = sandbox_windows
    elif sys.platform == "win32":
        return _unavailable(
            "OS isolation for Studio tools on Windows is an opt-in preview.",
            (
                "Windows isolation uses Microsoft's MXC ProcessContainer, which upstream "
                f"still labels an early preview and not a security boundary. Set "
                f"{WINDOWS_PREVIEW_ENV}=1 to enable it. {_FALLBACK_NOTE}"
            ),
            identity,
        )
    else:
        return _unavailable(
            "OS isolation for Studio tools is available on Linux, macOS and Windows only.",
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
        # A backend may narrow its static set for the host it is actually on, so
        # a limitation that is not true here is not claimed here.
        limitations = getattr(backend, "host_limitations", lambda: backend.LIMITATIONS)(),
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
    # Teardown is killpg on the captured group, which a tool that calls setsid
    # and closes stdout survives.
    limitations = ["no_os_isolation", "host_files_readable", "unrestricted_network"]
    if sys.platform != "win32":
        limitations.append("detached_descendant_cleanup_unverified")
    return tuple(limitations)


def prepare_tool_launch(plan: ToolLaunchPlan) -> PreparedSandboxLaunch:
    """Unavailable hosts fall back in auto; unsafe workdirs and build failures refuse."""
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

    capability = capability_snapshot()

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

    if sys.platform == "linux":
        from . import sandbox_linux as backend
    elif sys.platform == "win32":
        from . import sandbox_windows as backend
    else:
        from . import sandbox_macos as backend

    try:
        prepared = backend.prepare(plan)
        if plan.requested_mode == "required" and WORKDIR_SCAN_INCOMPLETE in (
            prepared.launch_limitations
        ):
            # `auto` degrades here on purpose: a big session workdir must not
            # end a chat, and the alternative was a permanent refusal that no
            # retry recovered from. `required` is a different promise. The
            # unvisited part of the walk could hold an external hard link or a
            # host socket, and saying so in the record does not keep a boundary
            # the caller asked to be guaranteed, so this one fails closed.
            prepared.cleanup()
            raise WorkdirUnsafeError(
                "the session workdir is too large to check for host channels, "
                "and `required` cannot promise a boundary it did not verify. "
                "Start a new chat, or use `auto` to run with software safeguards."
            )
    except OSError as exc:
        # Must be typed: raw, this reaches tools.py's general `except Exception`,
        # which answers `auto` by running with software safeguards.
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
