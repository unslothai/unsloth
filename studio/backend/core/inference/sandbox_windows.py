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
import site
import subprocess
import sys
import sysconfig
import tempfile
import uuid

from loggers import get_logger

from . import mxc_pins

from .os_sandbox import (
    PROFILE_VERSION,
    SESSION_PACKAGES_RELPATH,
    PreparedSandboxLaunch,
    SandboxBuildError,
    SandboxUnavailableError,
    ToolLaunchPlan,
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
# install_mxc_runtime.py's --dest override, read here too so the two sides
# cannot disagree about where the executor was installed.
_DIRECTORY_ENV = "UNSLOTH_MXC_DIR"
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
    candidate = configured or os.path.join(managed_mxc_dir(), _EXECUTABLE_NAME)
    if not os.path.isfile(candidate):
        return None
    # Checked HERE, not only at install time. The managed directory is
    # user-writable, so a same-user process, including a software-safeguarded
    # tool call made before isolation became available, can replace the file
    # afterwards. Everything this backend claims rests on this binary being
    # the one that was pinned, so the check belongs at the trust boundary.
    if not mxc_pins.matches_pin(candidate, mxc_pins.EXECUTOR_SHA256):
        logger.warning(
            "Ignoring %s: it is not the pinned MXC %s executor. Re-run "
            "Studio setup to reinstall it.",
            candidate,
            mxc_pins.MXC_VERSION,
        )
        return None
    return candidate


def managed_mxc_dir() -> str:
    """Where Studio setup installs the executor.

    Follows ``utils.node_runtime.managed_node_dir``, which solves exactly this
    problem for the Node runtime: the studio root in custom mode, the legacy
    ``~/.unsloth`` otherwise. Lazily imported and degraded the same way, so a
    host where ``utils.paths`` cannot be loaded still gets a path back rather
    than raising into the launch planner.
    """
    # The same override the installer takes, so a setup run that reported
    # success into a custom directory is not followed by every tool call
    # saying the executor is not installed.
    override = (os.environ.get(_DIRECTORY_ENV) or "").strip()
    if override:
        return os.path.expanduser(override)
    legacy = os.path.join(os.path.expanduser("~"), ".unsloth", "mxc")
    try:
        from utils.paths.storage_roots import studio_root

        resolved = str(studio_root())
        legacy_studio = os.path.join(os.path.expanduser("~"), ".unsloth", "studio")
        if os.path.normcase(resolved) == os.path.normcase(legacy_studio):
            return legacy
        return os.path.join(resolved, "mxc")
    except (ImportError, OSError, ValueError):
        override = (
            os.environ.get("UNSLOTH_STUDIO_HOME") or os.environ.get("STUDIO_HOME") or ""
        ).strip()
        if override:
            return os.path.join(os.path.expanduser(override), "mxc")
        return legacy


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
            "the MXC sandbox executor is not installed, or the installed one "
            "is not the pinned build. Studio setup installs "
            "it when the Windows isolation preview is enabled, so re-run setup "
            "with UNSLOTH_WINDOWS_SANDBOX_PREVIEW=1, or install it directly "
            "with: python studio/install_mxc_runtime.py"
        )
    # Held and re-checked across the probe too. executable_path() hashes the
    # file and then this reopens the PATHNAME, and a same-user process can
    # swap it in between, at which point the replacement runs --probe directly
    # on the host, outside MXC. prepare()'s hold starts after this returns and
    # does nothing for this invocation.
    try:
        hold = _hold_executor(executor)
    except SandboxUnavailableError as exc:
        return False, str(exc)
    try:
        if not mxc_pins.matches_pin(executor, mxc_pins.EXECUTOR_SHA256):
            return False, (
                "the MXC sandbox executor changed after it was verified; refusing to run it"
            )
        probe = subprocess.run(
            [executor, "--probe"],
            capture_output = True,
            timeout = 30,
            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return False, f"the MXC executor could not be run: {exc}"
    finally:
        if hold is not None:
            hold.close()
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


def _within(path: str, root: str) -> bool:
    """Whether ``path`` is inside ``root``, without raising on a Windows host.

    ``os.path.commonpath`` raises ValueError when its arguments sit on
    different drives, which is an ordinary Windows layout: the interpreter on
    C: and the sandbox home or the project on D:. Left to propagate it fails
    every policy build, and ``prepare`` turns that into SandboxBuildError, so
    even `auto` would refuse the tool call instead of falling back. Different
    drives simply means not inside.
    """
    try:
        common = os.path.commonpath([path, root])
    except ValueError:
        return False
    return os.path.normcase(common) == os.path.normcase(root)


def _runtime_roots() -> list[str]:
    """The interpreter's own trees.

    ``sys.executable`` under the managed Windows virtualenv is
    ``unsloth_studio\\Scripts\\python.exe``, while the packages are in the
    sibling ``Lib\\site-packages`` and the standard library comes from
    ``sys.base_prefix\\Lib``. Granting only the executable's directory leaves
    both outside the policy, so an isolated call cannot import its own runtime.
    The Windows CI job does not catch this because it runs from an
    actions/setup-python installation rather than from Studio's virtualenv.
    """
    roots = [os.path.dirname(os.path.realpath(sys.executable)), sys.prefix, sys.base_prefix]
    for name in ("stdlib", "platstdlib", "purelib", "platlib"):
        try:
            roots.append(sysconfig.get_path(name))
        except (KeyError, OSError):
            continue
    try:
        roots.extend(site.getsitepackages())
    except AttributeError:  # pragma: no cover - only absent in odd embeddings
        pass
    user_site = getattr(site, "getusersitepackages", None)
    if user_site is not None:
        try:
            roots.append(user_site())
        except Exception:  # noqa: BLE001 - never fail a launch over this
            pass
    return [path for path in roots if path]


def _launch_program_roots(plan: ToolLaunchPlan) -> list[str]:
    """Where the program this launch actually runs lives.

    The Terminal tool does not run cmd on a normal Windows host: _get_shell_cmd
    picks ``C:\\Program Files\\Git\\bin\\bash.exe`` whenever the host has a
    trusted Git for Windows, and bash needs its own ``usr\\bin`` userland to do
    anything. Both are taken from what tools.py already resolved and trust
    checked, rather than re-derived here, so the policy cannot drift away from
    the launch it is supposed to describe. Neither argv[0] nor this PATH is
    model authored: _build_safe_env constructs the PATH and _get_shell_cmd
    picks the shell.
    """
    roots: list[str] = []
    program = plan.argv[0] if plan.argv else ""
    if program and os.path.isabs(program):
        resolved = os.path.realpath(program)
        roots.append(os.path.dirname(resolved))
        # bash.exe lives in <git>\bin and its userland in <git>\usr\bin, so the
        # install root covers both without guessing at either layout.
        roots.append(os.path.dirname(os.path.dirname(resolved)))
    for entry in (plan.env.get("PATH") or "").split(os.pathsep):
        entry = entry.strip()
        if entry:
            roots.append(entry)
    return roots


def _readonly_roots(plan: ToolLaunchPlan, workdir: str) -> list[str]:
    """What the launch needs to start, and nothing else.

    Reuses the same resolvers the Linux and macOS backends use, so an editable
    install that works there works here.
    """
    roots: list[str] = _system_roots() if sys.platform == "win32" else []
    # editable_import_roots() is deliberately ABSENT. On a flat-layout editable
    # install it returns the checkout's parent, and MXC's readonlyPaths are
    # recursive, so granting it would expose the rest of the checkout -- .env,
    # credentials, fixtures, .git -- to model-authored code that also has
    # unrestricted network access. macOS grants those parents as literals, for
    # listing only, and Linux merely creates them, so neither platform hands
    # over the tree either. The source roots themselves are granted, which is
    # what the installed finder actually reads.
    candidates = (
        *editable_source_roots(),
        *_runtime_roots(),
        *_launch_program_roots(plan),
    )
    # Case-insensitively, because PATH entries and sysconfig paths routinely
    # spell the same Windows directory differently and MXC would be handed the
    # same grant twice.
    seen = {os.path.normcase(path) for path in roots}
    for path in candidates:
        # An editable install living inside the workdir is already writable
        # there; granting it again read-only would be contradictory.
        if not path or os.path.normcase(path) in seen or not os.path.isdir(path):
            continue
        if _within(path, workdir):
            continue
        seen.add(os.path.normcase(path))
        roots.append(path)
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


def _session_package_environment(env: dict[str, str], workdir: str) -> dict[str, str]:
    """Point pip at the writable package directory and make it importable.

    Granting `.unsloth-packages` write access does not by itself send an
    install there, nor make what is already there importable: plan.env comes
    from _build_safe_env, which carries only the trusted shim on PYTHONPATH and
    no PIP_TARGET. Without this a pip install targets the read-only Studio
    virtualenv and fails, and an explicit install into the directory is
    invisible to every later call. The Linux and macOS backends do exactly this
    and the Windows one must not be the odd one out.
    """
    packages = os.path.join(workdir, SESSION_PACKAGES_RELPATH)
    updated = dict(env)
    updated["PIP_TARGET"] = packages
    # Appended in both cases, so a package installed by a tool call cannot
    # shadow the sandbox_site shim or a bare command the approval logic treats
    # as safe. <target>\Scripts is pip's console entry point directory on
    # Windows, where POSIX uses bin.
    updated["PATH"] = os.pathsep.join(
        part for part in (env.get("PATH") or "", os.path.join(packages, "Scripts")) if part
    )
    updated["PYTHONPATH"] = os.pathsep.join(
        part for part in (env.get("PYTHONPATH") or "", packages) if part
    )
    return updated


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
            "env": _policy_environment(_session_package_environment(plan.env, workdir)),
            "cwd": workdir,
        },
        "filesystem": {
            "readwritePaths": [workdir, packages],
            "readonlyPaths": _readonly_roots(plan, workdir),
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

    # Held from here until after the cleanup that runs it again, so the bytes
    # available() verified cannot be swapped before Popen or before --delete.
    hold = _hold_executor(executor)
    try:
        return _prepare_held(plan, executor, hold)
    except Exception:
        if hold is not None:
            hold.close()
        raise


def _prepare_held(
    plan: ToolLaunchPlan, executor: str, hold: object | None
) -> PreparedSandboxLaunch:
    # Re-checked WHILE held: only now is the verdict about a file that cannot
    # change underneath the launch.
    if not mxc_pins.matches_pin(executor, mxc_pins.EXECUTOR_SHA256):
        raise SandboxUnavailableError(
            "the MXC sandbox executor changed after it was verified; "
            "refusing to launch through it"
        )

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

    # MXC runs in "STDIO mode: passthrough", forwarding its own handles to the
    # child, so its warnings land on the SAME stdout as the payload. The probe
    # requires its token alone on stdout, and a tool call must not have MXC's
    # banner spliced into the user's output either. --log-file diverts MXC's
    # diagnostics to a file this launch owns and removes.
    log_handle, log_path = tempfile.mkstemp(prefix = "unsloth-mxc-", suffix = ".log")
    os.close(log_handle)

    prepared = PreparedSandboxLaunch(
        # `--` hands the argv to MXC to render. See build_policy().
        argv = (executor, "--log-file", log_path, "--config-base64", encoded, "--", *plan.argv),
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
    # FIRST, so it pops LAST: the callbacks run in LIFO order and the
    # reconciliation below executes the same file, so the hold outlives it.
    if hold is not None:
        prepared.cleanup_callbacks.append(hold.close)
    # A file, so not cleanup_paths, which rmtree's whatever it is given.
    prepared.cleanup_callbacks.append(lambda: _remove_quietly(log_path))
    # A hard kill skips the executor's own ACE revert, so reconcile on the way
    # out whether or not it exited cleanly.
    prepared.cleanup_callbacks.append(lambda: _reconcile_container(executor, container_id))
    return prepared


# The hold itself lives in mxc_pins, because install_mxc_runtime.py needs the
# same protection for the elevated wxc-host-prep.exe and cannot import this
# module.
def _hold_executor(path: str) -> object | None:
    try:
        return mxc_pins.hold_file(path)
    except OSError as exc:
        # Fails CLOSED: somebody else already holds it in a way that would let
        # them write to it, which is the condition this is meant to exclude.
        raise SandboxUnavailableError(
            f"the MXC executor could not be opened for exclusive use ({exc}); "
            "refusing to launch through it"
        ) from exc


def _remove_quietly(path: str) -> None:
    try:
        os.remove(path)
    except OSError:
        pass


def _reconcile_container(executor: str, container_id: str) -> None:
    """Remove any ACE and container state this launch left behind.

    `--delete` is MXC's own teardown. It is safe to call when the executor
    already cleaned up, which is the common case; this covers the kill path.
    """
    try:
        completed = subprocess.run(
            [executor, "--delete", "--containername", container_id],
            capture_output = True,
            timeout = 60,
            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        if completed.returncode != 0:
            # An executor that started and then refused the delete is the case
            # that leaves DENY and ALLOW ACEs on the user's own workdir after a
            # hard kill, and it exits non-zero rather than raising, so it would
            # otherwise be recorded as a successful teardown.
            detail = (completed.stderr or b"").decode(errors = "replace").strip()[:300]
            raise RuntimeError(
                f"MXC could not reconcile container {container_id} "
                f"(exit {completed.returncode}): {detail or 'no output'}. "
                "Temporary file permission changes may still be in place."
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
