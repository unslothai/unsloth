# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Studio tool launch contract and the OS-isolation backends (bwrap, Seatbelt)."""

from __future__ import annotations
import functools
import hashlib
import os
import platform
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
    """``required`` on a host that cannot provide it. Never raised in ``auto``."""

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
    """``requested_mode`` and ``effective_mode`` differ exactly when ``auto`` fell back."""

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
    return subprocess.Popen(prepared.argv, **popen_kwargs)


WORKDIR_SCAN_ENTRIES = 50_000
WORKDIR_SCAN_SECONDS = 5.0
# The shared cache is walked per launch, so its budget is tighter than the
# workdir's; a cache too big to check in it is simply not shared.
CACHE_SCAN_ENTRIES = 50_000
CACHE_SCAN_SECONDS = 3.0


def _host_channel_hazard(root: str, max_entries: int, seconds: float) -> str | None:
    """Why *root* carries a way out of itself, or None. Never raises.

    A socket or device node under it is a channel no path rule closes, a hard link
    to an inode also named outside is a writable path out, and a nested mount is
    storage both backends grant writes across. *root* being a mount point itself
    is fine. Sockets, FIFOs and exceeding the budget count even though a tool call
    can create them, since the scan cannot tell those apart from the host's.
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
                return f"too large to check for host channels (over {max_entries} entries or {seconds:.0f}s)"
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


def scan_workdir_for_host_channels(workdir: str) -> None:
    """The writable session workdir. A hazard here fails the call."""
    hazard = _host_channel_hazard(workdir, WORKDIR_SCAN_ENTRIES, WORKDIR_SCAN_SECONDS)
    if hazard is not None:
        raise WorkdirUnsafeError(f"the session workdir {hazard}")


def cache_share_hazard(path: str) -> str | None:
    """Why this host cache directory must not be shared into the jail, or None.

    Same hazards as the workdir and for the same reason: the model cache is bound
    WRITABLE, so a pathname socket under it is connectable from inside (a
    read-only bind does not stop connect(), and the network namespace is shared),
    and a file hard-linked to one outside the cache is writable through the cache
    name. Both measured before this existed.

    A hazard DROPS the component from the binds rather than failing the launch.
    The cache is an optimisation: without it the call re-downloads, which is what
    every call did before the cache was shared at all. Refusing instead would let
    anything able to write one socket into the cache end every later tool call.
    """
    return _host_channel_hazard(path, CACHE_SCAN_ENTRIES, CACHE_SCAN_SECONDS)


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


@functools.lru_cache(maxsize = 1)
def editable_source_roots() -> tuple[str, ...]:
    """Source directories of editable installs, so `import unsloth` still works.

    An editable install leaves the package's code OUTSIDE site-packages, and the
    interpreter paths the backends bind do not reach it, so a sandboxed tool call
    could not import a package the same environment imported a moment earlier.

    Read from PEP 610's direct_url.json rather than by parsing .pth files,
    because that record is written whichever mechanism the installer used: the
    classic path-in-a-.pth and the PEP 660 finder with its MAPPING both appear
    here, and only one of them is on sys.path.

    Filesystem root and /usr are refused: an editable install rooted there would
    hand back most of the host, which is the same guard the runtime paths apply.
    """
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
    for distribution in distributions:
        try:
            raw = distribution.read_text("direct_url.json")
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
        if path not in roots:
            roots.append(path)
    return tuple(roots)


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
    # Teardown is killpg on the captured group, which a tool that calls setsid
    # and closes stdout survives.
    limitations = ["no_os_isolation", "host_files_readable", "unrestricted_network"]
    if sys.platform != "win32":
        limitations.append("detached_descendant_cleanup_unverified")
    return tuple(limitations)


def prepare_tool_launch(plan: ToolLaunchPlan) -> PreparedSandboxLaunch:
    """Only ``required`` on a host without a working sandbox can refuse."""
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
    else:
        from . import sandbox_macos as backend

    try:
        prepared = backend.prepare(plan)
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
        limitations = capability.limitations,
    )
    return prepared
