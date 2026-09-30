# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
"""One-click OS sandbox setup from Unsloth: install bubblewrap on Linux, install and prepare MXC on Windows.

Nothing from a request reaches the command line: the operation name picks a constant plan.
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, field
import logging
import shlex
import subprocess
import sys
import threading
import time
from typing import Callable
import uuid

from . import sandbox_setup_plan
from .mxc_host_prep_job import HOST_CHANGE_LOCK, _DECLINED_MARKER, _steps as _prepared_steps

logger = logging.getLogger(__name__)

OUTPUT_TAIL_LINES = 20
PKEXEC_DISMISSED = 126
PKEXEC_NOT_AUTHORIZED = 127
_SUDO_NEEDS_PASSWORD = "a password is required"

_lock = threading.Lock()
_current: "SetupJob | None" = None
_on_finish: list[Callable[[], None]] = []


class SetupUnavailable(ValueError):
    """The requested operation does not apply to this host right now."""


@dataclass
class SetupJob:
    id: str
    operation: str
    state: str = "running"  # running | succeeded | declined | failed
    started_at: float = field(default_factory = time.time)
    finished_at: float | None = None
    exit_code: int | None = None
    output_tail: list[str] = field(default_factory = list)
    steps: list[str] = field(default_factory = list)
    manual_command: str = ""
    note: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


def current() -> SetupJob | None:
    with _lock:
        return _current


def add_finish_hook(hook: Callable[[], None]) -> None:
    if hook not in _on_finish:
        _on_finish.append(hook)


def running() -> bool:
    job = current()
    return job is not None and job.state == "running"


def _spawn(argv: list[str], env: dict | None = None) -> subprocess.Popen:
    from utils.process_lifetime import child_popen_kwargs, spawn_on_lifetime_thread

    kwargs = dict(
        env = env,
        stdin = subprocess.DEVNULL,
        stdout = subprocess.PIPE,
        stderr = subprocess.STDOUT,
        text = True,
        encoding = "utf-8",
        errors = "replace",
        **child_popen_kwargs(),
    )
    if sys.platform == "win32":
        kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
    return spawn_on_lifetime_thread(lambda: subprocess.Popen(argv, **kwargs))


def _invalidate() -> None:
    from . import mxc_probe, os_sandbox, sandbox_probe, tools

    resets = (
        sandbox_probe.reset_probe_cache,
        os_sandbox._linux_userns_blocked_by_apparmor.cache_clear,
        mxc_probe.invalidate_cache,
        tools.reset_terminal_profile_cache,
        sandbox_setup_plan.invalidate,
        *_on_finish,
    )
    for reset in resets:
        try:
            reset()
        except Exception as exc:  # noqa: BLE001 - a stale cache only delays the new verdict
            logger.warning("Sandbox setup: cache reset failed: %s", exc)
    try:
        if not os_sandbox._background_probes_disabled():
            os_sandbox.warm_tool_isolation()
    except Exception as exc:  # noqa: BLE001 - the next read checks on its own
        logger.warning("Sandbox setup: re-check after setup failed: %s", exc)


def pkexec_script(steps) -> str:
    return "set -e\n" + "\n".join(shlex.join(list(step)) for step in steps) + "\n"


def _commands(plan: sandbox_setup_plan.SetupPlan) -> tuple[list[list[str]], dict | None]:
    """(argv list, env). Linux steps run as root: programs pinned to system binaries, never PATH."""
    if plan.action != sandbox_setup_plan.LINUX_INSTALL:
        return [list(step) for step in plan.steps], None
    kind, path = sandbox_setup_plan.linux_elevation(force = True)
    if kind is None:
        raise SetupUnavailable(
            "Neither passwordless sudo nor a desktop password prompt is available."
        )
    try:
        steps = sandbox_setup_plan.elevated_steps(plan.steps)
        shell = sandbox_setup_plan.trusted_system_binary("sh") if kind == "pkexec" else None
    except LookupError as exc:
        raise SetupUnavailable(f"{exc}; run the command in a terminal instead.") from exc
    env = dict(sandbox_setup_plan.ELEVATED_ENV)
    if kind == "root":
        return steps, env
    if kind == "sudo":
        return [[path, "-n", *step] for step in steps], env
    if shell is None:
        raise SetupUnavailable(
            "No trusted /bin/sh was found; run the command in a terminal instead."
        )
    return [[path, shell, "-c", pkexec_script(steps)]], env


def _drain(job: SetupJob, proc: subprocess.Popen, tail: deque) -> int | None:
    try:
        for line in proc.stdout:
            tail.append(line.rstrip("\r\n"))
            job.output_tail = list(tail)
        # No timeout: a package manager or ACL helper stopped mid-run leaves the host half changed.
        return proc.wait()
    except Exception as exc:  # noqa: BLE001 - reported on the job, never raised into a thread
        tail.append(f"Unsloth lost track of the setup run: {exc}")
        return None


def _outcome(job: SetupJob, argv: list[str], code: int | None, lines: list[str]) -> str:
    if code == 0:
        return "succeeded"
    if job.operation == sandbox_setup_plan.LINUX_INSTALL:
        if argv and argv[0].endswith("pkexec"):
            if code == PKEXEC_DISMISSED:
                job.note = "The password prompt was dismissed."
                return "declined"
            if code == PKEXEC_NOT_AUTHORIZED:
                job.note = "Not authorized, or no authentication agent is running on this desktop."
                return "failed"
        if any(_SUDO_NEEDS_PASSWORD in line for line in lines):
            job.note = "sudo needs a password here; run the command in a terminal instead."
            return "failed"
        return "failed"
    if any(_DECLINED_MARKER in line for line in lines):
        job.note = "The administrator prompt was declined."
        return "declined"
    return "failed"


def _run(
    job: SetupJob,
    commands: list[list[str]],
    env: dict | None = None,
) -> None:
    tail: deque[str] = deque(maxlen = OUTPUT_TAIL_LINES)
    state, code = "succeeded", 0
    for argv in commands:
        try:
            proc = _spawn(argv, env)
        except Exception as exc:  # noqa: BLE001 - surfaced as a failed job
            tail.append(f"Could not start the setup step: {exc}")
            state, code = "failed", None
            break
        code = _drain(job, proc, tail)
        lines = list(tail)
        if "--prepare-host" in argv:
            job.steps = _prepared_steps(lines)
        state = _outcome(job, argv, code, lines)
        if state != "succeeded":
            break
    job.output_tail = list(tail)
    job.exit_code = code
    job.finished_at = time.time()
    if state != "succeeded" and not job.note and code is not None:
        job.note = f"The setup step exited with code {code}."
    logger.info("Sandbox setup finished: operation=%s state=%s exit=%s", job.operation, state, code)
    _invalidate()
    job.state = state


def _joined(running: SetupJob, operation: str) -> SetupJob:
    """The run already in progress, but only for the same request: another one never stands in."""
    if running.operation != operation:
        raise SetupUnavailable(
            f"Another sandbox setup ({running.operation}) is still running; "
            "start this one once it finishes."
        )
    return running


def start(operation: str) -> SetupJob:
    """Start the setup for `operation`, or return the run of the same operation already in progress."""
    global _current
    from . import mxc_host_prep_job

    if operation not in sandbox_setup_plan.OPERATIONS:
        raise SetupUnavailable(f"Unknown setup operation: {operation}")
    in_progress = current()
    if in_progress is not None and in_progress.state == "running":
        return _joined(in_progress, operation)
    if operation == sandbox_setup_plan.WINDOWS_RUNTIME:
        plan = sandbox_setup_plan.windows_runtime_plan()
    else:
        plan = sandbox_setup_plan.detect(force = True)
    if plan.action != operation:
        raise SetupUnavailable(plan.reason or "There is nothing to set up on this computer.")
    commands, env = _commands(plan)
    with HOST_CHANGE_LOCK:
        with _lock:
            if _current is not None and _current.state == "running":
                return _joined(_current, operation)
        prep = mxc_host_prep_job.current()
        if prep is not None and prep.state == "running":
            return _joined(
                SetupJob(
                    id = prep.id,
                    operation = sandbox_setup_plan.WINDOWS_SETUP,
                    started_at = prep.started_at,
                ),
                operation,
            )
        job = SetupJob(id = uuid.uuid4().hex, operation = operation, manual_command = plan.manual_command)
        with _lock:
            _current = job
    threading.Thread(
        target = _run, args = (job, commands, env), name = "sandbox-setup", daemon = True
    ).start()
    return job
