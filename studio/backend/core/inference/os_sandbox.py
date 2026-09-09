# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Studio tool launch contracts and the managed SRT adapter."""

from __future__ import annotations
import ctypes
import hashlib
import json
import os
import platform
import shutil
import signal
import subprocess
import sys
from . import srt_adapter
from .srt_diagnostics import ProbeReason, boundary_identity, environment_context, limited_disclosure
from dataclasses import dataclass, field, replace
from typing import Any, BinaryIO, Callable, Literal
from loggers import get_logger
from .network_proxy import NetworkAudit
from .network_proxy import AllowlistProxy, NetworkAllowlist

logger = get_logger(__name__)
ToolExecutionMode = Literal["os_isolation_required", "container_isolation", "limited", "full"]
NETWORK_POLICIES = ("deny", "allowlist")
_NR_PIDFD_SEND_SIGNAL = 424
_NR_PIDFD_OPEN = 434
_pidfd_support = None

SRT_VERSION = "0.0.75"
SRT_PROFILE = "srt-0.0.75-strict-v1"
_LIMITED_SAFEGUARDS = (
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
_FULL_SAFEGUARDS = ("timeout", "cancellation", "reaping", "cleanup")


class SandboxUnavailableError(RuntimeError):
    """The required native sandbox cannot safely launch this tool call.

    ``transient`` marks conditions that may clear on their own (a scan that ran
    out of time on a cold disk cache); the capability layer reports those as
    retryable instead of caching them as a permanent unavailability.
    """

    def __init__(
        self,
        message: str = "",
        *,
        transient: bool = False,
    ) -> None:
        super().__init__(message)
        self.transient = transient


@dataclass(frozen = True)
class SandboxCapability:
    backend: str
    qualified: bool
    reason: str
    available: bool | None = None
    transient: bool = False
    environment: str = "unknown"
    protection_state: str = "unavailable"
    profile_id: str = "none"
    limitations: tuple[str, ...] = ()
    probe_generation: str = ""
    environment_fingerprint: str = ""
    remediation: str = "Use Limited mode only for a trusted task, or install a qualified backend."
    retryable: bool = False
    # Limited retains Process Guard without claiming an OS security boundary.
    limited_backend: str = "process-guard"
    limited_profile_id: str = "limited-software-safeguards-v1"
    limited_limitations: tuple[str, ...] = ()
    limited_reason: str = ""
    # Network policies the backend can enforce for an OS-isolated launch, and the
    # hosts the "allowlist" policy would admit. "deny" is always present.
    network_policies: tuple[str, ...] = ("deny",)
    network_allowlist: tuple[str, ...] = ()
    # Internal runtime proof, distinct from the UI consent generation.
    qualification_generation: str = ""
    reason_code: str | None = None
    diagnostic: dict[str, Any] | None = None
    limited_disclosure: str = ""
    nested_eligible: bool = False
    nested_profile_id: str | None = None
    nested_disclosure: str = ""


@dataclass(frozen = True)
class ToolExecutionRecord:
    requested_mode: ToolExecutionMode
    effective_mode: ToolExecutionMode
    environment: str
    backend: str
    profile_id: str
    probe_generation: str
    os_isolation: bool
    retained_safeguards: tuple[str, ...]
    limitations: tuple[str, ...] = ()
    # "deny": no network path out of the sandbox. "allowlist": CONNECT tunnels to
    # network_allowlist hosts through the per-launch loopback proxy.
    # "unrestricted": the launch has the host's network (Limited and Full).
    network_policy: str = "deny"
    network_allowlist: tuple[str, ...] = ()
    authority_disclosure: str = ""

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
            "network_allowlist": list(self.network_allowlist),
            "authority_disclosure": self.authority_disclosure,
        }


@dataclass(frozen = True)
class ToolLaunchPlan:
    """Complete policy inputs for one final Python or Terminal process launch."""

    argv: tuple[str, ...]
    workdir: str
    env: dict[str, str]
    preexec_fn: Callable[[], None] | None = None
    launcher_preexec_fn: Callable[[], None] | None = None
    requested_mode: ToolExecutionMode = "os_isolation_required"
    current_subject: str | None = None
    tool_ui_session_id: str | None = None
    limited_grant: str | None = None
    nested_grant: str | None = None
    timeout_seconds: int | None = None
    close_fds: bool = True
    terminate_descendants: bool = True
    # "deny" (default) or "allowlist". Only honored for os_isolation_required;
    # Full has the host network anyway and Limited cannot enforce a proxy.
    network_policy: str = "deny"
    # Set by the trusted tool owner, not inferred from a shell command or model args.
    # None preserves older direct backend callers until they adopt an explicit kind.
    execution_kind: Literal["python", "terminal"] | None = None
    cancel_event: Any = None


@dataclass
class PreparedSandboxLaunch:
    """A native sandbox argv plus resources owned until the process exits."""

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
    spawn_callback: Callable[["PreparedSandboxLaunch", dict[str, Any]], object] | None = None
    cleanup_callbacks: list[Callable[[], None]] = field(default_factory = list)
    cleanup_diagnostics: list[str] = field(default_factory = list)
    # Set when the launch runs behind the allowlist proxy; tools.py reads the
    # refused hosts from it for the result trailer.
    network_audit: NetworkAudit | None = None

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
    """Spawn exactly one prepared launch, using its backend-owned launcher when set."""
    if prepared.spawn_callback is not None:
        return prepared.spawn_callback(prepared, popen_kwargs)
    return subprocess.Popen(prepared.argv, **popen_kwargs)


def _pidfd_open(pid: int) -> int:
    """A file descriptor pinned to exactly this process (raises OSError)."""
    if hasattr(os, "pidfd_open"):
        return os.pidfd_open(pid, 0)
    libc = ctypes.CDLL(None, use_errno = True)
    fd = libc.syscall(_NR_PIDFD_OPEN, ctypes.c_int(pid), ctypes.c_uint(0))
    if fd < 0:
        errno_value = ctypes.get_errno()
        raise OSError(errno_value, os.strerror(errno_value))
    return int(fd)


def _pidfd_send_signal(pidfd: int, signum: int) -> None:
    if hasattr(signal, "pidfd_send_signal"):
        signal.pidfd_send_signal(pidfd, signum)
        return
    libc = ctypes.CDLL(None, use_errno = True)
    result = libc.syscall(
        _NR_PIDFD_SEND_SIGNAL, ctypes.c_int(pidfd), ctypes.c_int(signum), None, ctypes.c_uint(0)
    )
    if result < 0:
        errno_value = ctypes.get_errno()
        raise OSError(errno_value, os.strerror(errno_value))


def descendant_sweep_supported() -> bool:
    """Whether Limited launches can reap detached descendants after the leader exits.

    The sweep (tools._sweep_marked_descendants) matches processes by the per-call
    marker in ``/proc/<pid>/environ`` and signals them through a pidfd taken
    before the match, so a pid recycled between the match and the signal is
    never hit. Without ``/proc`` or pidfds (macOS, Linux before 5.3) there is no
    safe sweep and the Limited record discloses
    ``detached_descendant_cleanup_unverified`` instead.
    """
    global _pidfd_support
    if sys.platform != "linux" or not os.path.isdir("/proc"):
        return False
    if _pidfd_support is None:
        try:
            os.close(_pidfd_open(os.getpid()))
            _pidfd_support = True
        except (OSError, AttributeError, TypeError):
            _pidfd_support = False
    return _pidfd_support


SandboxLaunchSpec = ToolLaunchPlan

_LINUX_REQUIRED_BINARIES = ("bwrap", "socat", "rg")
_BLOCKED_REMEDIATION = (
    "Required remains blocked. Use Limited only after reviewing its session warning, "
    "or separately confirm Full access."
)


def _linux_userns_blocked_by_apparmor() -> bool:
    """Whether this host has Ubuntu's AppArmor restriction on unprivileged user namespaces.

    Ubuntu 23.10+ ships ``kernel.apparmor_restrict_unprivileged_userns=1``, which
    denies ``unshare(CLONE_NEWUSER)`` to any binary without a permitting profile.
    bwrap needs that namespace, so an installed bubblewrap still cannot build a
    sandbox and the probe fails with a bare "helper closed its control channel".
    Read-only: Studio reports the condition and never changes host security policy.
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


def _linux_unavailable_remediation() -> str:
    """Name what this host is actually missing, instead of only offering Limited.

    SRT_TOOL_ISOLATION.md tells users to install prerequisites "if the capability
    message requests them"; without this the message never requested anything, so
    a host that installed the helper successfully had no next step to take.
    """
    missing = [name for name in _LINUX_REQUIRED_BINARIES if shutil.which(name) is None]
    if missing:
        return (
            f"Install the missing Linux prerequisites ({', '.join(missing)}) with your "
            "distribution's package manager, then retry. " + _BLOCKED_REMEDIATION
        )
    if _linux_userns_blocked_by_apparmor():
        return (
            "This host denies unprivileged user namespaces "
            "(kernel.apparmor_restrict_unprivileged_userns=1, the default on Ubuntu 23.10 and "
            "newer), so bubblewrap cannot build a sandbox even though it is installed. Grant "
            "bwrap the userns permission with an AppArmor profile "
            "(/etc/apparmor.d/bwrap-userns-restrict from the apparmor-profiles package), then "
            "retry. Studio does not change host security policy. " + _BLOCKED_REMEDIATION
        )
    return _BLOCKED_REMEDIATION


def _runtime_identity() -> str:
    """Consent changes when the selected interpreter or shipped adapter changes."""
    from . import srt_probe

    digest = hashlib.sha256()
    digest.update(str((srt_probe._cache_epoch, srt_probe.runtime_inputs())).encode())
    shell_candidates = ()
    if sys.platform == "linux":
        # Match the directories used by the safe Terminal PATH. Bind absent
        # candidates too: installing a venv-local bash changes shell selection.
        directories = [os.path.dirname(sys.executable)]
        venv = os.environ.get("VIRTUAL_ENV")
        if venv:
            directories.append(os.path.join(venv, "bin"))
        directories.extend(("/usr/local/bin", "/usr/bin", "/bin"))
        shell_candidates = tuple(os.path.join(directory, "bash") for directory in directories)
    for path in (sys.executable, __file__, *shell_candidates):
        resolved = os.path.realpath(path)
        digest.update(resolved.encode())
        try:
            info = os.stat(resolved)
            digest.update(str((info.st_size, info.st_mtime_ns)).encode())
        except OSError:
            digest.update(b"missing")
    digest.update((SRT_PROFILE + sys.platform + platform.release()).encode())
    digest.update((os.path.abspath(sys.executable) + sys.prefix).encode())
    digest.update(srt_adapter.installation_identity().encode())
    digest.update(boundary_identity().encode())
    digest.update(os.environ.get("UNSLOTH_STUDIO_TOOL_NETWORK_ALLOWLIST", "").encode())
    return digest.hexdigest()


def capability_snapshot(
    *,
    force: bool = False,
    execution_kind = None,
    selected_executable = None,
) -> SandboxCapability:
    snapshot = _capability_snapshot(
        force = force, execution_kind = execution_kind, selected_executable = selected_executable
    )
    environment = environment_context()
    changes = {"environment": environment, "limited_disclosure": limited_disclosure(environment)}
    if not snapshot.available:
        reason = (
            snapshot.reason
            if isinstance(snapshot.reason, ProbeReason)
            else ProbeReason("probe_failed")
        )
        changes.update(reason.fields(), reason = str(reason))
        from .srt_nested import eligibility, NESTED_PROFILE, NESTED_DISCLOSURE

        eligible = eligibility((False, reason))
        changes.update(
            nested_eligible = eligible,
            nested_profile_id = NESTED_PROFILE if eligible else None,
            nested_disclosure = NESTED_DISCLOSURE if eligible else "",
        )
        changes["probe_generation"] = hashlib.sha256(
            (
                snapshot.probe_generation + reason.code + str(reason.dependency) + str(eligible)
            ).encode()
        ).hexdigest()
    return replace(snapshot, **changes)


def _capability_snapshot(
    *,
    force: bool = False,
    execution_kind = None,
    selected_executable = None,
) -> SandboxCapability:
    """Describe the shipped backend without treating installed binaries as proof."""
    identity = _runtime_identity()
    if sys.platform in ("win32", "darwin"):
        from .srt_probe import probe

        available, reason = probe(
            force = force, execution_kind = execution_kind, selected_executable = selected_executable
        )
        limitations = (
            ("srt_windows_system_dns_unfenced", "srt_windows_shared_account_grants")
            if sys.platform == "win32"
            else ("srt_macos_system_dns_unfenced", "host_files_readable")
        )
        return SandboxCapability(
            backend = "srt",
            qualified = False,
            available = available,
            reason = reason,
            environment = sys.platform,
            protection_state = "preview" if available else "unavailable",
            profile_id = "srt-0.0.75-native-v1",
            limitations = limitations,
            probe_generation = hashlib.sha256((identity + str(available)).encode()).hexdigest(),
            environment_fingerprint = identity,
            remediation = (
                "SRT uses the platform's supported filesystem and network restrictions; system DNS remains available."
                if available
                else "Run studio/install_srt_runtime.py with Studio's Python. On Windows, also run it with --windows-install for SRT's one-time setup, then retry."
            ),
            limited_limitations = ("unrestricted_network", "host_files_readable"),
        )
    elif sys.platform == "linux":
        from .srt_probe import probe

        available, reason = probe(
            force = force, execution_kind = execution_kind, selected_executable = selected_executable
        )
        limitations = ("srt_platform_qualification_incomplete",)
        if available:
            from .srt_probe import probe_network

            network_policies = ("deny",)
            hosts = ()
            try:
                hosts = NetworkAllowlist.from_env().hosts
                if probe_network(force = force):
                    network_policies = ("deny", "allowlist")
            except ValueError:
                hosts = ()
            return SandboxCapability(
                backend = "srt",
                qualified = False,
                available = True,
                reason = reason,
                environment = "linux",
                protection_state = "preview",
                profile_id = SRT_PROFILE,
                limitations = limitations,
                probe_generation = hashlib.sha256((identity + "available").encode()).hexdigest(),
                environment_fingerprint = identity,
                network_policies = network_policies,
                network_allowlist = hosts if "allowlist" in network_policies else (),
                remediation = "SRT Preview supports private Unix IPC after its live probe passes. HTTPS allowlists require the private transport probe to pass; CUDA remains unqualified.",
            )
    else:
        limitations = ("srt_platform_unqualified",)
        reason = "The selected SRT runtime has not passed Studio's strict isolation probes."
    return SandboxCapability(
        backend = "srt",
        qualified = False,
        available = False,
        reason = reason,
        environment = sys.platform,
        profile_id = SRT_PROFILE,
        limitations = limitations,
        probe_generation = hashlib.sha256((identity + "unavailable").encode()).hexdigest(),
        environment_fingerprint = identity,
        remediation = (
            _linux_unavailable_remediation() if sys.platform == "linux" else _BLOCKED_REMEDIATION
        ),
        limited_limitations = ("unrestricted_network", "host_files_readable")
        + (() if sys.platform == "win32" else ("detached_descendant_cleanup_unverified",)),
    )


def sandbox_capability() -> SandboxCapability:
    return capability_snapshot()


def _nested_launch_capability(spec, selected):
    from . import srt_probe
    from .srt_nested import NESTED_GRANTS, NESTED_PROFILE
    from .tool_isolation import LimitedGrantError

    def snapshot():
        return capability_snapshot(
            force = True,
            execution_kind = spec.execution_kind,
            selected_executable = selected,
        )

    def authorize(capability):
        if (
            capability.available
            or not capability.nested_eligible
            or capability.nested_profile_id != NESTED_PROFILE
        ):
            raise SandboxUnavailableError("Container-compatible isolation is not eligible")
        if not spec.current_subject or not spec.tool_ui_session_id:
            raise SandboxUnavailableError(
                "Container-compatible isolation requires a Studio UI session"
            )
        try:
            NESTED_GRANTS.validate(
                spec.nested_grant,
                current_subject = spec.current_subject,
                tool_ui_session_id = spec.tool_ui_session_id,
                probe_generation = capability.probe_generation,
            )
        except LimitedGrantError as exc:
            raise SandboxUnavailableError(
                "Container-compatible isolation consent is invalid or expired"
            ) from exc

    if spec.network_policy != "deny":
        raise SandboxUnavailableError(
            "Container-compatible isolation currently requires network deny"
        )
    before = snapshot()
    authorize(before)
    available, reason = srt_probe.probe(
        force = True,
        execution_kind = spec.execution_kind,
        selected_executable = selected,
        isolation_variant = "nested",
    )
    if not available:
        raise SandboxUnavailableError(f"Container-compatible isolation probe failed: {reason}")
    after = snapshot()
    if after.probe_generation != before.probe_generation:
        raise SandboxUnavailableError("Container-compatible isolation changed during its probe")
    authorize(after)
    if spec.cancel_event is not None and spec.cancel_event.is_set():
        raise SandboxUnavailableError("Execution cancelled before launch")
    return after


def prepare_tool_launch(spec: ToolLaunchPlan) -> PreparedSandboxLaunch:
    """Authorize one launch; a Required failure never becomes a host execution."""
    if spec.cancel_event is not None and spec.cancel_event.is_set():
        raise SandboxUnavailableError("Execution cancelled before launch")
    if not spec.argv or any(not isinstance(arg, str) or "\0" in arg for arg in spec.argv):
        raise SandboxUnavailableError("tool launch argv is invalid")
    if spec.requested_mode not in (
        "os_isolation_required",
        "container_isolation",
        "limited",
        "full",
    ):
        raise SandboxUnavailableError("unknown tool execution mode")
    if not spec.close_fds or not spec.terminate_descendants:
        raise SandboxUnavailableError("tool launches must close descriptors and own cleanup")
    if spec.network_policy not in NETWORK_POLICIES:
        raise SandboxUnavailableError("unknown network policy")
    workdir = os.path.realpath(spec.workdir)
    if not os.path.isdir(workdir):
        raise SandboxUnavailableError("tool workdir is unavailable")
    canonical = replace(spec, workdir = workdir, env = dict(spec.env))
    if canonical.requested_mode == "full":
        record = ToolExecutionRecord(
            requested_mode = "full",
            effective_mode = "full",
            environment = sys.platform,
            backend = "none",
            profile_id = "full-access-v1",
            probe_generation = _runtime_identity(),
            os_isolation = False,
            retained_safeguards = tuple(
                item
                for item in _FULL_SAFEGUARDS
                if item != "timeout" or canonical.timeout_seconds is not None
            ),
            network_policy = "unrestricted",
        )
    else:
        selected = (
            canonical.argv[0]
            if os.path.isabs(canonical.argv[0])
            else shutil.which(canonical.argv[0], path = canonical.env.get("PATH", ""))
        )
        capability = capability_snapshot(
            execution_kind = canonical.execution_kind, selected_executable = selected
        )
        nested = canonical.requested_mode == "container_isolation"
        if nested:
            capability = _nested_launch_capability(canonical, selected)
        if canonical.requested_mode in ("os_isolation_required", "container_isolation"):
            if not nested and not capability.available:
                raise SandboxUnavailableError(
                    f"OS_ISOLATION_UNAVAILABLE: {capability.reason} {capability.remediation}"
                )
            transport = None
            trust_store = None
            allowlist_hosts = ()
            launch_env = dict(canonical.env)
            extra_reads = ()
            try:
                if canonical.network_policy == "allowlist":
                    if "allowlist" not in capability.network_policies:
                        raise SandboxUnavailableError(
                            "SRT HTTPS allowlist transport is not qualified"
                        )
                    from .srt_network import SrtNetworkTransport, TlsTrustSnapshot

                    allowlist = NetworkAllowlist.from_env()
                    trust_store = TlsTrustSnapshot().start()
                    launch_env.update(trust_store.environment)
                    extra_reads = trust_store.read_roots
                    transport = SrtNetworkTransport(
                        AllowlistProxy(allowlist),
                        lifetime_seconds = (
                            None
                            if canonical.timeout_seconds is None
                            else canonical.timeout_seconds + 60
                        ),
                    ).start()
                    allowlist_hosts = allowlist.hosts
                    launch_env.update(transport.environment)
                request = srt_adapter.request_for(
                    canonical.argv,
                    canonical.workdir,
                    launch_env,
                    canonical.timeout_seconds,
                    additional_read_roots = extra_reads,
                    **({"isolation_variant": "nested"} if nested else {}),
                )
                if transport is not None:
                    request["network"] = {
                        "httpSocketPath": transport.http_socket_path,
                        "socksSocketPath": transport.socks_socket_path,
                        "socatPath": srt_adapter.socat_executable(),
                    }
            except Exception as exc:
                try:
                    if transport is not None:
                        transport.close()
                finally:
                    if trust_store is not None:
                        trust_store.close()
                raise SandboxUnavailableError(str(exc)) from exc
            record = ToolExecutionRecord(
                requested_mode = canonical.requested_mode,
                effective_mode = canonical.requested_mode,
                environment = capability.environment,
                backend = "srt",
                profile_id = capability.nested_profile_id if nested else capability.profile_id,
                probe_generation = capability.probe_generation,
                os_isolation = True,
                retained_safeguards = tuple(
                    item
                    for item in (*_LIMITED_SAFEGUARDS, "os_isolation")
                    if item != "timeout" or canonical.timeout_seconds is not None
                    if item != "resource_limits" or sys.platform != "win32"
                ),
                limitations = ("shared_container_proc",) if nested else capability.limitations,
                authority_disclosure = capability.nested_disclosure if nested else "",
                network_policy = canonical.network_policy,
                network_allowlist = allowlist_hosts,
            )
            prepared = PreparedSandboxLaunch(
                argv = canonical.argv,
                workdir = canonical.workdir,
                env = canonical.env,
                preexec_fn = canonical.launcher_preexec_fn,
                backend = "srt",
                execution_record = record,
                timeout_seconds = canonical.timeout_seconds,
            )
            if transport is not None:
                prepared.cleanup_callbacks.append(transport.close)
                prepared.network_audit = transport.proxy.audit
            if trust_store is not None:
                prepared.cleanup_callbacks.append(trust_store.close)

            def launch(_prepared, kwargs):
                if nested:
                    current = _nested_launch_capability(canonical, selected)
                    if current.probe_generation != capability.probe_generation:
                        raise SandboxUnavailableError(
                            "Container-compatible isolation changed before launch"
                        )
                try:
                    proc = srt_adapter.spawn(request, cancel_event = canonical.cancel_event, **kwargs)
                except Exception as exc:
                    raise SandboxUnavailableError(f"SRT launch failed: {exc}") from exc
                prepared.cleanup_callbacks.append(lambda: srt_adapter.release_control(proc))
                return proc

            prepared.spawn_callback = launch
            return prepared
        if canonical.network_policy != "deny":
            raise SandboxUnavailableError("the network allowlist requires OS isolation")
        if capability.available:
            raise SandboxUnavailableError(
                "Limited mode is not authorized while OS isolation is available"
            )
        if not canonical.current_subject or not canonical.tool_ui_session_id:
            raise SandboxUnavailableError(
                "Limited mode requires an authenticated Studio UI session"
            )
        from .tool_isolation import LimitedGrantError, validate_limited_grant

        try:
            validate_limited_grant(
                canonical.limited_grant,
                current_subject = canonical.current_subject,
                tool_ui_session_id = canonical.tool_ui_session_id,
                probe_generation = capability.probe_generation,
                requested_mode = "limited",
            )
        except LimitedGrantError as exc:
            raise SandboxUnavailableError(f"Limited mode authorization failed: {exc}") from exc
        record = ToolExecutionRecord(
            requested_mode = "limited",
            effective_mode = "limited",
            environment = capability.environment,
            backend = "process-guard",
            profile_id = "limited-software-safeguards-v1",
            probe_generation = capability.probe_generation,
            os_isolation = False,
            retained_safeguards = tuple(
                item
                for item in _LIMITED_SAFEGUARDS
                if item != "timeout" or canonical.timeout_seconds is not None
            ),
            limitations = capability.limited_limitations,
            authority_disclosure = capability.limited_disclosure,
            network_policy = "unrestricted",
        )
    return PreparedSandboxLaunch(
        argv = canonical.argv,
        workdir = canonical.workdir,
        env = canonical.env,
        preexec_fn = canonical.preexec_fn,
        backend = record.backend,
        execution_record = record,
        timeout_seconds = canonical.timeout_seconds,
        close_fds = True,
        terminate_descendants = True,
    )
