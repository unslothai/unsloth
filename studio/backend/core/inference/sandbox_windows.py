# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Windows tool isolation through Microsoft's MXC ProcessContainer.

MXC (https://github.com/microsoft/mxc, MIT) ships `wxc-exec.exe`, a native Rust
executor that takes a JSON policy and runs one workload inside an AppContainer.
It is driven here with `subprocess` and nothing else: no Node runtime, no npm at
run time, no vendored JavaScript.

What this actually buys, stated plainly because the backend name oversells it.
Per MXC's own os-version-support.md, the kernel-native containment tier (T1,
PSEC) needs build 26600+, and the brokered tier (T2, BFS) is compiled out
because it can deadlock the host. So on 23H2, 24H2 and 25H2 -- every Windows
that ships today -- the enforcing tier is T3: an AppContainer SID plus DENY and
ALLOW ACEs written onto the real host paths for the life of the call and removed
afterwards. That is a genuine boundary and it is not a kernel sandbox, and
MXC's README says in bold that no MXC profile should be treated as a security
boundary yet. Both facts are carried in LIMITATIONS so the execution record and
the UI repeat them rather than showing an unqualified "isolated" badge.

Two consequences of T3 are engineered around here rather than hoped about:
a container id is unique per launch, because MXC documents that two concurrent
runs sharing one id revoke each other's ACEs; and the ACEs are reconciled on
cleanup, because a hard kill skips the executor's own revert.
"""

from __future__ import annotations
import base64
import json
import os
import subprocess
import sys
import uuid

from loggers import get_logger

from .os_sandbox import (
    PROFILE_VERSION,
    SESSION_PACKAGES_RELPATH,
    PreparedSandboxLaunch,
    SandboxBuildError,
    SandboxUnavailableError,
    ToolLaunchPlan,
    editable_import_roots,
    editable_source_roots,
    scan_workdir_for_host_channels,
)

logger = get_logger(__name__)

BACKEND_NAME = "mxc-processcontainer"
PROFILE_ID = f"windows-mxc-{PROFILE_VERSION}"

# Pinned exactly. MXC documents stable schemas as immutable exact-match
# contracts, though the shipped 0.8.0 executor in fact accepts any 0.8.x it is
# handed (measured: 0.8.1-alpha ran, 9.9.9-alpha was rejected), so pinning is
# our guarantee rather than theirs.
SCHEMA_VERSION = "0.8.0-alpha"

# The executor is fetched at Studio setup time, never during a tool call.
_EXECUTABLE_ENV = "UNSLOTH_MXC_EXEC"
_EXECUTABLE_NAME = "wxc-exec.exe"

LIMITATIONS = (
    "unrestricted_network",
    # MXC's own words, surfaced rather than paraphrased away.
    "mxc_preview_not_a_security_boundary",
    # Only true below build 26600; capability_snapshot() drops it above that.
    "windows_tier3_dacl",
    "system_paths_readable",
    "model_cache_writable",
    "gpu_devices_hidden",
    "shared_kernel",
)

# Build 26600 is where MXC reports the PSEC contract that makes T1 reachable.
_TIER1_MIN_BUILD = 26600


def host_limitations() -> tuple[str, ...]:
    """LIMITATIONS narrowed to this host.

    On build 26600+ MXC can reach the kernel-native tier, so the DACL caveat
    would be a claim about a mechanism this host is not using. Everything else
    holds on every build.
    """
    if _windows_build() >= _TIER1_MIN_BUILD:
        return tuple(item for item in LIMITATIONS if item != "windows_tier3_dacl")
    return LIMITATIONS


def _windows_build() -> int:
    """The host's build number, or 0 when it cannot be read."""
    try:
        return int(sys.getwindowsversion().build)  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001 - an unreadable build is simply not T1
        return 0


def executable_path() -> str | None:
    """The pinned executor, or None when setup has not installed it.

    Resolved from an explicit location only. Never from PATH: PATH is
    attacker-influenced on a host where tool calls run, and this is the one
    binary whose identity the whole boundary rests on.
    """
    configured = os.environ.get(_EXECUTABLE_ENV)
    if configured:
        return configured if os.path.isfile(configured) else None
    from utils.studio_paths import studio_home  # local: Windows-only import path

    candidate = os.path.join(str(studio_home()), "mxc", _EXECUTABLE_NAME)
    return candidate if os.path.isfile(candidate) else None


def available() -> tuple[bool, str]:
    """Whether this host could isolate. Availability, never assurance.

    A real launch through prepare() is what decides it; sandbox_probe drives
    that, exactly as it does for bubblewrap and Seatbelt. This only rules out
    hosts that cannot get as far as trying.
    """
    if sys.platform != "win32":
        return False, "the MXC backend is Windows-only"
    executor = executable_path()
    if executor is None:
        return False, (
            "the MXC sandbox executor is not installed "
            "(Studio setup installs it; re-run setup to repair it)"
        )
    try:
        probe = subprocess.run(
            [executor, "--probe"],
            capture_output = True,
            timeout = 30,
            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return False, f"the MXC executor could not be run: {exc}"
    if probe.returncode != 0:
        detail = probe.stderr.decode(errors = "replace").strip()[:300]
        return False, f"the MXC host probe failed (exit {probe.returncode}): {detail}"
    return True, "MXC ProcessContainer is available on this host"


def _system_roots() -> list[str]:
    """The Windows directories any process needs simply to start.

    MXC grants nothing implicitly: "Omitted = no filesystem access beyond the
    default sandbox root". On POSIX the loader's needs are covered by binding
    /usr and /lib read-only, and this is the same grant. Without it python.exe
    cannot resolve ntdll, kernel32 or the CRT and the launch fails before the
    payload runs, which is what the first Windows CI run showed.

    Read-only, so this widens what a tool call can READ to the system
    directories, exactly as `system_paths_readable` already says for Linux. It
    does not widen what it can write.
    """
    roots: list[str] = []
    system_root = os.environ.get("SystemRoot") or os.environ.get("SYSTEMROOT") or "C:\\Windows"
    for path in (system_root, os.path.join(system_root, "System32")):
        if os.path.isdir(path) and path not in roots:
            roots.append(path)
    return roots


def _readonly_roots(workdir: str) -> list[str]:
    """What the interpreter needs to start, and nothing else.

    Reuses the same resolvers the Linux and macOS backends use, so an editable
    install that works there works here.
    """
    roots: list[str] = _system_roots() if sys.platform == "win32" else []
    for path in (*editable_source_roots(), *editable_import_roots()):
        # An editable install living inside the workdir is already writable
        # there; granting it again read-only would be contradictory.
        if not path or not os.path.isdir(path) or path in roots:
            continue
        if os.path.normcase(os.path.commonpath([path, workdir])) == os.path.normcase(workdir):
            continue
        roots.append(path)
    interpreter = os.path.dirname(os.path.realpath(sys.executable))
    if interpreter not in roots:
        roots.append(interpreter)
    return roots


# Windows will not start a process without these. They name the system
# directories and the local machine rather than carrying user data, and the
# POSIX backends pass their equivalents through for the same reason.
_REQUIRED_WINDOWS_ENV = (
    "SystemRoot",
    "SystemDrive",
    "windir",
    "COMSPEC",
    "NUMBER_OF_PROCESSORS",
    "PROCESSOR_ARCHITECTURE",
)


def _policy_environment(plan_env: dict[str, str]) -> list[str]:
    """The sanitized env, plus the few variables Windows needs to start at all.

    MXC replaces the environment rather than layering onto it
    (`inheritDefaultEnv` is 0.9 only), so whatever is handed over is the whole
    environment the child gets. A plan env that omits SystemRoot leaves
    python.exe unable to initialise, which is not a confinement result but
    looks exactly like one.

    Only filled in when the caller did not set them, so a deliberately
    overridden value still wins.
    """
    env = dict(plan_env)
    for name in _REQUIRED_WINDOWS_ENV:
        if name in env or name.upper() in {key.upper() for key in env}:
            continue
        value = os.environ.get(name)
        if value:
            env[name] = value
    return [f"{key}={value}" for key, value in sorted(env.items())]


def build_policy(plan: ToolLaunchPlan, workdir: str, container_id: str) -> dict:
    """The MXC config for one launch.

    `process.commandLine` is deliberately left off. It is a single STRING, and
    rendering model-authored argv into a Windows command line in Python is a
    quoting-injection hazard with a long history. MXC renders the argv it is
    given after `--` for the selected backend itself, which is why the argv is
    passed that way and why prepare() verifies the executor accepts it.
    """
    packages = os.path.join(workdir, SESSION_PACKAGES_RELPATH)
    policy: dict = {
        "version": SCHEMA_VERSION,
        "containerId": container_id,
        "containment": "processcontainer",
        "process": {
            "env": _policy_environment(plan.env),
            "cwd": workdir,
        },
        "filesystem": {
            "readwritePaths": [workdir, packages],
            "readonlyPaths": _readonly_roots(workdir),
        },
        # Job-object UI restrictions; the workload is never interactive.
        "ui": {"disable": True},
        # Egress stays open, the same contract Linux and macOS carry: a tool call
        # installs packages and downloads models. Explicit CIDR and port rules
        # exist in schema 0.8 but are PSEC-only, so they are unavailable on every
        # build this backend actually runs on, and claiming them would be false.
        "processContainer": {"capabilities": ["internetClient"]},
    }
    if plan.timeout_seconds is not None:
        policy["process"]["timeout"] = int(plan.timeout_seconds) * 1000
    return policy


def prepare(plan: ToolLaunchPlan) -> PreparedSandboxLaunch:
    ok, reason = available()
    if not ok:
        raise SandboxUnavailableError(reason)
    if not plan.argv:
        raise SandboxUnavailableError("a sandboxed launch needs a command to run")

    executor = executable_path()
    if executor is None:  # pragma: no cover - available() already checked
        raise SandboxUnavailableError("the MXC sandbox executor disappeared")

    workdir = os.path.abspath(plan.workdir)
    if not os.path.isdir(workdir):
        from .os_sandbox import WorkdirUnsafeError
        raise WorkdirUnsafeError(f"the session workdir does not exist: {workdir}")
    workdir_limitations = scan_workdir_for_host_channels(workdir)

    # Unique per LAUNCH, not per session. MXC keys its ACEs on the SID derived
    # from this id, and documents that two concurrent runs sharing one id revoke
    # each other's ACEs -- which for us would be two tool calls in two chats.
    container_id = f"unsloth-{uuid.uuid4().hex}"

    try:
        policy = build_policy(plan, workdir, container_id)
        encoded = base64.b64encode(json.dumps(policy).encode()).decode()
    except Exception as exc:  # noqa: BLE001 - a build failure must be typed
        raise SandboxBuildError(f"the MXC policy could not be built: {exc}") from exc

    limitations = workdir_limitations
    if _windows_build() < _TIER1_MIN_BUILD:
        # Already in LIMITATIONS; on a T1-capable host capability_snapshot drops
        # it, so nothing is claimed here that the host cannot back.
        logger.debug("MXC tier 1 is unavailable on build %s", _windows_build())

    prepared = PreparedSandboxLaunch(
        # `--` hands the argv to MXC to render. See build_policy().
        argv = (executor, "--config-base64", encoded, "--", *plan.argv),
        workdir = workdir,
        env = dict(plan.env),
        # Windows has neither, and tools.py already guards both for win32.
        preexec_fn = None,
        backend = BACKEND_NAME,
        timeout_seconds = plan.timeout_seconds,
        close_fds = plan.close_fds,
        terminate_descendants = plan.terminate_descendants,
        launch_limitations = limitations,
    )
    # A hard kill skips the executor's own ACE revert, so reconcile on the way
    # out whether or not it exited cleanly.
    prepared.cleanup_callbacks.append(lambda: _reconcile_container(executor, container_id))
    return prepared


def _reconcile_container(executor: str, container_id: str) -> None:
    """Remove any ACE and container state this launch left behind.

    `--delete` is MXC's own teardown. It is safe to call when the executor
    already cleaned up, which is the common case; this covers the kill path.
    """
    try:
        subprocess.run(
            [executor, "--delete", "--containername", container_id],
            capture_output = True,
            timeout = 60,
            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except (OSError, subprocess.SubprocessError) as exc:
        # Surfaced through cleanup_diagnostics by the caller's cleanup().
        raise RuntimeError(f"could not reconcile MXC container {container_id}: {exc}") from exc


def probe_argv(
    workdir: str,
    payload_argv: tuple[str, ...],
    env: dict[str, str] | None = None,
) -> PreparedSandboxLaunch:
    """Built through the same prepare() a real tool call uses, so the probe
    cannot qualify a sandbox nothing ever runs."""
    return prepare(
        ToolLaunchPlan(
            argv = tuple(payload_argv),
            workdir = workdir,
            env = dict(env or {}),
            requested_mode = "required",
        )
    )
