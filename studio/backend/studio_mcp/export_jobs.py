# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Export jobs started over MCP. Loading a checkpoint and exporting it can take many minutes, longer than an agent waits on one tool call, so export_model runs them as a background task and get_job reports on it. Jobs are keyed by account, and an id from another account reads as unknown."""

from __future__ import annotations

import asyncio
import re
import secrets
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional

from fastmcp.exceptions import ToolError

from hub.utils.host_paths import redact_paths_in_text
from studio_mcp.tools import leaf_name

CAPACITY = 64


@dataclass
class ExportJob:
    job_id: str
    account_id: str
    format: str
    status: str = "running"
    phase: str = "starting"
    output: Optional[str] = None
    error: Optional[str] = None
    finished_at: Optional[float] = None
    # Held here so the task outlives the tool call that started it.
    task: Optional[asyncio.Task] = field(default = None, repr = False)

    @property
    def finished(self) -> bool:
        return self.status != "running"


_jobs: "OrderedDict[str, ExportJob]" = OrderedDict()


def _key(account_id: str, job_id: str) -> str:
    return f"{account_id}:{job_id}"


def _evict() -> None:
    while len(_jobs) > CAPACITY:
        oldest = next((key for key, job in _jobs.items() if job.finished), None)
        if oldest is None:
            return
        _jobs.pop(oldest)


def output_name(path: Any) -> Optional[str]:
    """An export's output as the agent may see it: relative to Unsloth Studio's exports folder, or just its name."""
    if not isinstance(path, str) or not path:
        return None
    if not (path.startswith(("/", "\\")) or re.match(r"^[A-Za-z]:[\\/]", path)):
        return path
    try:
        from utils.paths import exports_root
        return Path(path).resolve().relative_to(Path(exports_root()).resolve()).as_posix()
    except Exception:
        return f"{leaf_name(path)} (outside exports folder)"


def finish(
    job: ExportJob,
    status: str,
    *,
    error: Optional[str] = None,
) -> None:
    if job.finished:
        return
    job.status = status
    job.phase = "done"
    job.error = redact_paths_in_text(error) if error else job.error
    job.finished_at = time.monotonic()


async def settle(job: ExportJob, timeout: float) -> None:
    """Wait up to ``timeout`` seconds for the job's task to finish on its own, without cancelling it."""
    if job.task is not None and not job.task.done():
        await asyncio.wait({job.task}, timeout = timeout)


async def _drive(job: ExportJob, run: Callable[[ExportJob], Awaitable[None]]) -> None:
    try:
        await run(job)
    except asyncio.CancelledError:
        finish(job, "cancelled")
        raise
    except ToolError as exc:
        finish(job, "failed", error = str(exc))
    except Exception:
        finish(job, "failed", error = "The export stopped unexpectedly.")
    else:
        finish(job, "completed")


def start(account_id: str, format: str, run: Callable[[ExportJob], Awaitable[None]]) -> ExportJob:
    job = ExportJob(job_id = secrets.token_urlsafe(12), account_id = account_id, format = format)
    _jobs[_key(account_id, job.job_id)] = job
    _evict()
    job.task = asyncio.get_running_loop().create_task(_drive(job, run))
    return job


def lookup(account_id: str, job_id: str) -> ExportJob:
    job = _jobs.get(_key(account_id, job_id))
    if job is None:
        raise ToolError("No such export job")
    return job


def any_running() -> bool:
    return any(not job.finished for job in _jobs.values())


def jobs_of(account_id: str) -> list[ExportJob]:
    return [job for job in _jobs.values() if job.account_id == account_id]


def mark_cancelled(job: ExportJob) -> None:
    if job.task is not None and not job.task.done():
        job.task.cancel()
    finish(job, "cancelled")


def _reset() -> None:
    """Test hook."""
    _jobs.clear()
