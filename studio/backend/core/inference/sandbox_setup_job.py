# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
"""Install the Windows MXC runtime from Settings > Sandbox, one run at a time.

Nothing from a request reaches the command line: the operation name picks a constant plan. The
runtime install is not elevated; it is the same `install_mxc_prebuilt.py` call setup.ps1 makes.
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, field
import logging
import subprocess
import sys
import threading
import time
from typing import Callable
import uuid

from . import sandbox_setup_plan

logger = logging.getLogger(__name__)

OUTPUT_TAIL_LINES = 20
# install_mxc_prebuilt.py: a running MXC process holds the runtime files.
_RUNTIME_IN_USE = 3

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


def _spawn(argv: list[str]) -> subprocess.Popen:
    from utils.process_lifetime import child_popen_kwargs, spawn_on_lifetime_thread

    kwargs = dict(
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
    from . import mxc_probe, sandbox_probe, tools
    resets = (
        sandbox_probe.reset_probe_cache,
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


def _drain(job: SetupJob, proc: subprocess.Popen, tail: deque) -> int | None:
    try:
        for line in proc.stdout:
            tail.append(line.rstrip("\r\n"))
            job.output_tail = list(tail)
        # No timeout: an install stopped mid-run leaves a half-written runtime folder.
        return proc.wait()
    except Exception as exc:  # noqa: BLE001 - reported on the job, never raised into a thread
        tail.append(f"Unsloth lost track of the setup run: {exc}")
        return None


def _run(job: SetupJob, commands: list[list[str]]) -> None:
    tail: deque[str] = deque(maxlen = OUTPUT_TAIL_LINES)
    state, code = "succeeded", 0
    for argv in commands:
        try:
            proc = _spawn(argv)
        except Exception as exc:  # noqa: BLE001 - surfaced as a failed job
            tail.append(f"Could not start the setup step: {exc}")
            state, code = "failed", None
            break
        code = _drain(job, proc, tail)
        if code != 0:
            state = "failed"
            break
    job.output_tail = list(tail)
    job.exit_code = code
    job.finished_at = time.time()
    if state != "succeeded" and code == _RUNTIME_IN_USE:
        job.note = "A running Python or Terminal call is using the runtime; try again once it ends."
    elif state != "succeeded" and code is not None:
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
    plan = sandbox_setup_plan.windows_runtime_plan()
    if plan.action != operation:
        raise SetupUnavailable(plan.reason or "There is nothing to set up on this computer.")
    prep = mxc_host_prep_job.current()
    if prep is not None and prep.state == "running":
        raise SetupUnavailable("Prepare this PC is still running; try again once it finishes.")
    with _lock:
        if _current is not None and _current.state == "running":
            return _joined(_current, operation)
        job = SetupJob(id = uuid.uuid4().hex, operation = operation, manual_command = plan.manual_command)
        _current = job
    threading.Thread(
        target = _run,
        args = (job, [list(step) for step in plan.steps]),
        name = "sandbox-setup",
        daemon = True,
    ).start()
    return job
