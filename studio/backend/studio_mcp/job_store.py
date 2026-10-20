# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Background jobs started over MCP. Some work (an export, a model load) outlasts an agent's tool call, so it runs as a task held here and get_job reports on it; a client that gives up on the call never stops the work. Jobs are keyed by account, and an id from another account reads as unknown."""

from __future__ import annotations

import asyncio
import secrets
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional

from fastmcp.exceptions import ToolError

from hub.utils.host_paths import redact_paths_in_text


@dataclass
class Job:
    job_id: str
    account_id: str
    kind: str = ""
    status: str = "running"
    phase: str = "starting"
    error: Optional[str] = None
    result: Any = None
    finished_at: Optional[float] = None
    # Held here so the task outlives the tool call that started it.
    task: Optional[asyncio.Task] = field(default = None, repr = False)

    @property
    def finished(self) -> bool:
        return self.status != "running"


def new_id() -> str:
    return secrets.token_urlsafe(12)


class JobStore:
    """Jobs of one kind, oldest first. Past ``capacity`` the oldest finished job is dropped; a running one never is."""

    def __init__(self, *, capacity: int, unknown: str, failure: str) -> None:
        self.capacity, self.unknown, self.failure = capacity, unknown, failure
        self.jobs: "OrderedDict[str, Job]" = OrderedDict()

    def add(self, job: Job) -> Job:
        self.jobs[f"{job.account_id}:{job.job_id}"] = job
        self.evict()
        return job

    def evict(self) -> None:
        while len(self.jobs) > self.capacity:
            oldest = next((key for key, job in self.jobs.items() if job.finished), None)
            if oldest is None:
                return
            self.jobs.pop(oldest)

    def start(self, job: Job, run: Callable[[Job], Awaitable[Any]]) -> Job:
        """Run ``run(job)`` as a task the store holds; its return value becomes ``job.result``."""
        self.add(job)
        job.task = asyncio.get_running_loop().create_task(self._drive(job, run))
        return job

    async def _drive(self, job: Job, run: Callable[[Job], Awaitable[Any]]) -> None:
        try:
            result = await run(job)
        except asyncio.CancelledError:
            finish(job, "cancelled")
            raise
        except ToolError as exc:
            finish(job, "failed", error = str(exc))
        except Exception:
            finish(job, "failed", error = self.failure)
        else:
            job.result = result if result is not None else job.result
            finish(job, "completed")

    def lookup(self, account_id: str, job_id: str) -> Job:
        job = self.jobs.get(f"{account_id}:{job_id}")
        if job is None:
            raise ToolError(self.unknown)
        return job

    def jobs_of(self, account_id: str) -> list[Job]:
        return [job for job in self.jobs.values() if job.account_id == account_id]

    def running(self, account_id: Optional[str] = None) -> list[Job]:
        return [
            job
            for job in self.jobs.values()
            if not job.finished and (account_id is None or job.account_id == account_id)
        ]

    def reset(self) -> None:
        """Test hook."""
        self.jobs.clear()


def finish(
    job: Job,
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


async def settle(job: Job, timeout: float) -> None:
    """Wait up to ``timeout`` seconds for the job's task to finish on its own, without cancelling it."""
    if job.task is not None and not job.task.done() and timeout > 0:
        await asyncio.wait({job.task}, timeout = timeout)


def mark_cancelled(job: Job) -> None:
    if job.task is not None and not job.task.done():
        job.task.cancel()
    finish(job, "cancelled")
