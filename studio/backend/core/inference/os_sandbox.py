# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Studio tool launch contract and the OS-isolation backends.

Python and Terminal tool calls have always run on the host behind software
safeguards only -- a setsid/rlimit pre-exec, an environment whitelist, and the
static analysis in ``tools.py``. This module adds a real OS boundary on the two
platforms that hand you one: bubblewrap on Linux, Seatbelt on macOS.

The default mode is ``auto``: isolate when the host can, and otherwise run
exactly as before with the tool result labelled honestly. Nothing a user could
run yesterday stops working because this landed. ``required`` is the opt-in for
someone who would rather be refused than run unisolated, and ``full`` is the
existing bypass, unchanged.

What the boundary covers is deliberately narrow, so the label can be true: no
writes outside the session workdir, and no reads of the user's home beyond the
runtime paths the interpreter itself needs. The network is NOT confined -- tool
calls still pip-install and download models -- so a script that reaches a secret
can still send it. Keeping that gap honest is why the record says
``network_policy = "unrestricted"`` rather than staying quiet about it.
"""

from __future__ import annotations
import errno
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

# "auto" is the default and never refuses. "required" refuses instead of running
# unisolated. "full" is the pre-existing bypass and keeps its old meaning.
ToolExecutionMode = Literal["auto", "required", "full"]
TOOL_EXECUTION_MODES = ("auto", "required", "full")

PROFILE_VERSION = "unsloth-sandbox-v1"

# Where a session's pip installs live, relative to the workdir. Both backends
# point PIP_TARGET at it and tools.py keeps it on the path of a launch that
# fell back, so a package survives a call that could not be isolated.
SESSION_PACKAGES_RELPATH = ".unsloth-packages"

# What a launch keeps when the OS boundary is NOT in force. This is exactly the
# set main already applies, named so the record can state it rather than imply it.
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
# The OS boundary is added to the software set, never a replacement for it.
_OS_ISOLATION_SAFEGUARDS = _SOFTWARE_SAFEGUARDS + ("filesystem_isolation", "process_isolation")
_FULL_SAFEGUARDS = ("timeout", "cancellation", "reaping", "cleanup")


class SandboxUnavailableError(RuntimeError):
    """``required`` was asked for on a host that cannot provide it.

    Never raised in ``auto``: that mode's whole contract is that it falls back
    rather than refusing.
    """

    def __init__(
        self,
        message: str = "",
        *,
        remediation: str = "",
    ) -> None:
        super().__init__(message)
        self.remediation = remediation


class WorkdirUnsafeError(SandboxUnavailableError):
    """The session workdir itself carries a way out, so this launch is refused.

    Told apart from every other refusal by TYPE rather than by asking the probe
    again: the workdir is the one thing a tool call can write to, so this is the
    error that must never be answered by running unisolated, and deciding that
    from a second probe's verdict means a transient probe failure re-opens the
    very channel the scan just found.
    """


@dataclass(frozen = True)
class SandboxCapability:
    """What this host can actually enforce, proven by a live probe.

    ``available`` is never inferred from a binary being present on disk. The
    probe launches a real sandbox and checks that its negative controls fail,
    because an installed bubblewrap on a host that denies user namespaces looks
    identical to a working one until you try it.
    """

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
    """What one launch actually got. Built by the backend, never by the model.

    ``requested_mode`` and ``effective_mode`` differ exactly when ``auto`` fell
    back, which is the case the UI has to show plainly.
    """

    requested_mode: ToolExecutionMode
    effective_mode: str
    environment: str
    backend: str
    profile_id: str
    probe_generation: str
    os_isolation: bool
    retained_safeguards: tuple[str, ...]
    limitations: tuple[str, ...] = ()
    # Always "unrestricted": this sandbox confines the filesystem, not the network.
    # Stated rather than omitted so nobody reads the badge as more than it is.
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
    """Complete policy inputs for one Python or Terminal process launch."""

    argv: tuple[str, ...]
    workdir: str
    env: dict[str, str]
    preexec_fn: Callable[[], None] | None = None
    requested_mode: ToolExecutionMode = "auto"
    timeout_seconds: int | None = None
    close_fds: bool = True
    terminate_descendants: bool = True
    # Set by the trusted tool owner, not inferred from a shell command or model
    # args. None keeps older direct callers working.
    execution_kind: Literal["python", "terminal"] | None = None


@dataclass
class PreparedSandboxLaunch:
    """A ready argv plus every resource owned until the process exits."""

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
        """Release everything in LIFO order, and never stop at the first failure.

        A callback that raises must not strand the file handles and private
        directories queued behind it, so each failure is recorded and the sweep
        continues.
        """
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
    """Spawn exactly one prepared launch."""
    return subprocess.Popen(prepared.argv, **popen_kwargs)


# ── the session workdir ──────────────────────────────────────────────

# The scan runs before every launch, so it is bounded in both directions. A
# checkpoint tree under the workdir must fail the launch honestly rather than
# stall it, or be waved through unchecked.
WORKDIR_SCAN_ENTRIES = 50_000
WORKDIR_SCAN_SECONDS = 5.0


def scan_workdir_for_host_channels(workdir: str) -> None:
    """Refuse a session workdir that carries a way out of itself.

    Both backends make this one directory the whole writable set, so a socket or
    device node under it is a channel neither a mount namespace nor a Seatbelt
    path rule closes, a hard link whose inode also has a name outside the workdir
    is a writable path out of it, and a nested mount is somebody else's storage
    wearing a path inside it -- bubblewrap's workdir bind is recursive and takes
    it along, and a Seatbelt subpath rule grants writes across it. The workdir
    itself being a mount point is fine and stays allowed; what is refused is a
    mount UNDER it. Backend-agnostic on purpose: the invariant is the boundary
    both profiles claim, not a bubblewrap detail.

    Nothing it raises on can be created from inside the jail. A socket or a FIFO
    is not refused: a tool call can make one and one inside the workdir addresses
    nothing outside it. A device node is, because the kernel refuses mknod of one
    in a user namespace. Running out of budget is not a refusal either, since a
    tool call can write 50,000 files; the scan stops, having accounted for what it
    did reach.

    Raises ``SandboxUnavailableError``, which fails the call.
    """
    deadline = time.monotonic() + WORKDIR_SCAN_SECONDS
    entries = 0
    # (device, inode) -> [names found in here, st_nlink, first name]. Counting is
    # the whole point: refusing every st_nlink > 1 would refuse the workdir of any
    # session that ran `cp -al`, `git clone --local` or a pip install, all of
    # which hard-link within a tree, and would then keep refusing for the rest of
    # the session. Only a link the workdir cannot account for leads outside it.
    links: dict[tuple[int, int], list] = {}

    def stop(exc: OSError) -> None:
        if exc.errno in (errno.EACCES, errno.EPERM):
            # A directory a tool call chmodded to 000: unreadable to the scan and
            # equally unreadable to whatever the bind carries it into, so not a
            # channel, and not something to refuse a launch over.
            logger.info("Skipped an unreadable session workdir entry: %s", exc.filename)
            return
        raise WorkdirUnsafeError(
            f"the session workdir cannot be fully inspected: {exc.filename or workdir}"
        ) from exc

    for base, dirs, names in os.walk(workdir, followlinks = False, onerror = stop):
        for name in (*dirs, *names):
            entries += 1
            if entries > WORKDIR_SCAN_ENTRIES or time.monotonic() > deadline:
                raise WorkdirUnsafeError(
                    "the session workdir is too large to check for host channels before a "
                    f"launch (over {WORKDIR_SCAN_ENTRIES} entries or "
                    f"{WORKDIR_SCAN_SECONDS:.0f}s)"
                )
            path = os.path.join(base, name)
            try:
                info = os.lstat(path)
            except OSError as exc:
                raise WorkdirUnsafeError(
                    f"the session workdir changed during its safety scan: {path}"
                ) from exc
            if stat.S_ISLNK(info.st_mode):
                continue
            if stat.S_ISDIR(info.st_mode):
                # Two stats on a directory the walk already reached, and the same
                # answer on both platforms. Linux asks the mount table as well,
                # since this misses a same-filesystem bind mount.
                if os.path.ismount(path):
                    raise WorkdirUnsafeError(
                        f"the session workdir contains a nested host mount: {path}"
                    )
                continue
            if not stat.S_ISREG(info.st_mode):
                raise WorkdirUnsafeError(
                    f"the session workdir contains a device or IPC node: {path}"
                )
            if info.st_nlink > 1:
                found = links.setdefault((info.st_dev, info.st_ino), [0, info.st_nlink, path])
                found[0] += 1
    for found, total, path in links.values():
        if found < total:
            raise WorkdirUnsafeError(
                f"the session workdir contains a file hard-linked from outside it: {path}"
            )


# ── host diagnosis ───────────────────────────────────────────────────

_LINUX_REQUIRED_BINARIES = ("bwrap",)
_FALLBACK_NOTE = "Python and Terminal still run, with software safeguards only and no OS isolation."


@functools.lru_cache(maxsize = 1)
def _linux_userns_blocked_by_apparmor() -> bool:
    """Whether this host has Ubuntu's AppArmor restriction on unprivileged user namespaces.

    Cached for the process. The diagnosis forks ``unshare``, and it is reached
    from the remediation string every capability snapshot builds, so without the
    cache a host that cannot isolate pays an extra fork and exec on every single
    tool call, forever. The answer cannot change without an operator editing an
    AppArmor profile or a sysctl, at which point Studio is restarted anyway.

    Ubuntu 23.10+ ships ``kernel.apparmor_restrict_unprivileged_userns=1``, which
    denies ``unshare(CLONE_NEWUSER)`` to any binary without a permitting profile.
    bwrap needs that namespace, so an installed bubblewrap still cannot build a
    sandbox and the probe fails with nothing a user could act on. Read-only:
    Studio reports the condition and never changes host security policy.
    """
    try:
        with open("/proc/sys/kernel/apparmor_restrict_unprivileged_userns", encoding = "utf-8") as f:
            if f.read().strip() != "1":
                return False
    except OSError:
        return False
    # The sysctl alone is not proof: a profile may permit bwrap. Ask the kernel.
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


def linux_unavailable_remediation() -> str:
    """Name what this host is actually missing, rather than only stating a refusal."""
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
    """Changes when the selected interpreter or this module changes.

    The probe result is cached against it, so swapping the venv under a running
    Studio re-probes instead of reusing a verdict about a different runtime.
    """
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


# ── capability ───────────────────────────────────────────────────────


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
    """Describe what this host can enforce, via a cached live probe."""
    identity = _runtime_identity()
    if sys.platform == "linux":
        from . import sandbox_linux
        backend = sandbox_linux
    elif sys.platform == "darwin":
        from . import sandbox_macos
        backend = sandbox_macos
    else:
        # Windows and everything else keep main's behaviour exactly.
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


# ── launch preparation ───────────────────────────────────────────────


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
    # Unconditional off Windows. descendant_sweep_supported() says only that this
    # kernel COULD host the marker-and-pidfd sweep it describes; nothing stamps
    # the marker and nothing performs the sweep, so teardown is still killpg on
    # the captured group. A tool that calls setsid (an accepted Terminal wrapper)
    # and closes stdout survives that, which is exactly what the limitation is
    # for. It comes off when the sweep is implemented, not when /proc exists.
    limitations = ["no_os_isolation", "host_files_readable", "unrestricted_network"]
    if sys.platform != "win32":
        limitations.append("detached_descendant_cleanup_unverified")
    return tuple(limitations)


def prepare_tool_launch(plan: ToolLaunchPlan) -> PreparedSandboxLaunch:
    """Turn a launch plan into the argv that will actually run.

    Three outcomes, and only one of them can refuse: ``required`` on a host
    without a working sandbox. ``auto`` always returns a runnable launch.
    """
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
        # auto: run exactly as main does, and say so in the record.
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

    prepared = backend.prepare(plan)
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
