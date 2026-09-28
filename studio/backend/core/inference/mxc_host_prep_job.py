# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
"""One-click MXC host preparation from Settings: the elevated `--prepare-host` run as a background job."""

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

logger = logging.getLogger(__name__)

OUTPUT_TAIL_LINES = 20
_DECLINED_MARKER = "administrator prompt was declined"
_PREPARED_PREFIX = "[mxc-prebuilt] host prepared:"
_ALREADY_PREPARED = "[mxc-prebuilt] host already prepared"

_lock = threading.Lock()
_current: "HostPrepJob | None" = None
# Extra resets run once a job ends (the settings route drops its status cache through this).
_on_finish: list[Callable[[], None]] = []


@dataclass
class HostPrepJob:
    id: str
    state: str = "running"  # running | succeeded | declined | failed
    started_at: float = field(default_factory = time.time)
    finished_at: float | None = None
    exit_code: int | None = None
    output_tail: list[str] = field(default_factory = list)
    steps: list[str] = field(default_factory = list)

    def as_dict(self) -> dict:
        return asdict(self)


def current() -> HostPrepJob | None:
    with _lock:
        return _current


def add_finish_hook(hook: Callable[[], None]) -> None:
    if hook not in _on_finish:
        _on_finish.append(hook)


def _steps(lines: list[str]) -> list[str]:
    for line in reversed(lines):
        text = line.strip()
        if text.startswith(_PREPARED_PREFIX):
            return [s.strip() for s in text[len(_PREPARED_PREFIX) :].split(",") if s.strip()]
        if text.startswith(_ALREADY_PREPARED):
            return []
    return []


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
    from . import mxc_probe, tools
    for reset in (mxc_probe.invalidate_cache, tools.reset_terminal_profile_cache, *_on_finish):
        try:
            reset()
        except Exception as exc:  # noqa: BLE001 - a stale cache only delays the new verdict
            logger.warning("MXC host preparation: cache reset failed: %s", exc)


def _run(job: HostPrepJob, proc: subprocess.Popen) -> None:
    tail: deque[str] = deque(maxlen = OUTPUT_TAIL_LINES)
    try:
        for line in proc.stdout:
            tail.append(line.rstrip("\r\n"))
            job.output_tail = list(tail)
        # No timeout: killing the helper mid ACL propagation would leave the host half prepared.
        code = proc.wait()
    except Exception as exc:  # noqa: BLE001 - reported on the job, never raised into a thread
        tail.append(f"Studio lost track of the preparation run: {exc}")
        code = None
    lines = list(tail)
    job.output_tail = lines
    job.exit_code = code
    job.steps = _steps(lines)
    job.finished_at = time.time()
    if code == 0:
        job.state = "succeeded"
    elif any(_DECLINED_MARKER in line for line in lines):
        job.state = "declined"
    else:
        job.state = "failed"
    logger.info("MXC host preparation finished: state=%s exit=%s", job.state, code)
    _invalidate()


def start() -> HostPrepJob:
    """Start the elevated host preparation, or return the run already in progress."""
    global _current
    from . import mxc_probe

    with _lock:
        if _current is not None and _current.state == "running":
            return _current
        job = HostPrepJob(id = uuid.uuid4().hex)
        try:
            proc = _spawn(mxc_probe.host_prep_command())
        except Exception as exc:  # noqa: BLE001 - surfaced as a failed job
            job.state, job.finished_at = "failed", time.time()
            job.output_tail = [f"Could not start the host preparation: {exc}"]
            _current = job
            return job
        _current = job
    threading.Thread(target = _run, args = (job, proc), name = "mxc-host-prep", daemon = True).start()
    return job
