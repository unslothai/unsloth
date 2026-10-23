# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Image generation queue: full generate requests wait here and run one at a time through the normal generate path.

In-process only, so it survives a page reload but not a server restart. Every call happens on the event loop, so
the job list needs no lock. Each account sees and edits only its own jobs."""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional

from loggers import get_logger

logger = get_logger(__name__)

QUEUED = "queued"
RUNNING = "running"
DONE = "done"
FAILED = "failed"
CANCELLED = "cancelled"
FINISHED = (DONE, FAILED, CANCELLED)

# Requests can carry base64 source images, so neither list grows without bound.
MAX_QUEUED_PER_ACCOUNT = 100
MAX_FINISHED_PER_ACCOUNT = 20


class JobCancelled(Exception):
    """The running job was stopped."""


class QueueFull(Exception):
    pass


@dataclass
class ImageJob:
    id: str
    account_id: str
    account: Any
    request: Any
    # The resident model when the job was queued; a job never runs on another one.
    load: Any = None
    status: str = QUEUED
    created_at: float = field(default_factory = time.time)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    error: Optional[str] = None
    image_ids: list[str] = field(default_factory = list)
    cancel_requested: bool = False


class ImageGenerationQueue:
    def __init__(
        self,
        runner: Callable[[ImageJob], Awaitable[list[str]]],
        wait_idle: Optional[Callable[[], Awaitable[None]]] = None,
    ) -> None:
        self._runner = runner
        self._wait_idle = wait_idle
        self._jobs: list[ImageJob] = []
        self._worker: Optional[asyncio.Task] = None

    def jobs(self, account_id: str) -> list[ImageJob]:
        return [job for job in self._jobs if job.account_id == account_id]

    def get(self, account_id: str, job_id: str) -> Optional[ImageJob]:
        return next((job for job in self.jobs(account_id) if job.id == job_id), None)

    def enqueue(
        self,
        account,
        request,
        load = None,
    ) -> ImageJob:
        account_id = account.account_id
        if (
            sum(1 for job in self.jobs(account_id) if job.status == QUEUED)
            >= MAX_QUEUED_PER_ACCOUNT
        ):
            raise QueueFull(f"The image queue is full ({MAX_QUEUED_PER_ACCOUNT} jobs).")
        job = ImageJob(
            id = uuid.uuid4().hex[:12],
            account_id = account_id,
            account = account,
            request = request,
            load = load,
        )
        self._jobs.append(job)
        self._ensure_worker()
        return job

    def cancel(self, account_id: str, job_id: str) -> Optional[ImageJob]:
        """A queued job is cancelled here; a running one is only flagged, the caller stops the engine."""
        job = self.get(account_id, job_id)
        if job is None:
            return None
        if job.status == QUEUED:
            self._finish(job, CANCELLED)
        elif job.status == RUNNING:
            job.cancel_requested = True
        return job

    def move(self, account_id: str, job_id: str, index: int) -> Optional[ImageJob]:
        """Move a queued job to ``index`` among this account's queued jobs, clamped to the ends."""
        mine = [job for job in self.jobs(account_id) if job.status == QUEUED]
        job = next((j for j in mine if j.id == job_id), None)
        if job is None:
            return None
        mine.remove(job)
        index = max(0, min(index, len(mine)))
        self._jobs.remove(job)
        if index < len(mine):
            self._jobs.insert(self._jobs.index(mine[index]), job)
        elif mine:
            self._jobs.insert(self._jobs.index(mine[-1]) + 1, job)
        else:
            self._jobs.append(job)
        return job

    def clear(self, account_id: str) -> int:
        """Drop this account's queued and finished jobs; a running job is left to finish."""
        removed = 0
        for job in self.jobs(account_id):
            if job.status == RUNNING:
                continue
            self._jobs.remove(job)
            removed += 1
        return removed

    def position(self, job: ImageJob) -> Optional[int]:
        if job.status != QUEUED:
            return None
        queued = [j for j in self.jobs(job.account_id) if j.status == QUEUED]
        return queued.index(job)

    def _finish(
        self,
        job: ImageJob,
        status: str,
        error: Optional[str] = None,
    ) -> None:
        job.status = status
        job.error = error
        job.finished_at = time.time()
        # Finished jobs move behind everything still pending, newest last, and the oldest fall off.
        self._jobs.remove(job)
        self._jobs.append(job)
        finished = [j for j in self.jobs(job.account_id) if j.status in FINISHED]
        for stale in finished[:-MAX_FINISHED_PER_ACCOUNT]:
            self._jobs.remove(stale)

    def _next(self) -> Optional[ImageJob]:
        return next((job for job in self._jobs if job.status == QUEUED), None)

    def _ensure_worker(self) -> None:
        if self._worker is None or self._worker.done():
            self._worker = asyncio.get_running_loop().create_task(self._drain())

    async def _drain(self) -> None:
        while True:
            if self._next() is None:
                return
            if self._wait_idle is not None:
                await self._wait_idle()
            # Re-read: the head may have been cancelled or reordered while a direct run finished.
            job = self._next()
            if job is None:
                return
            job.status = RUNNING
            job.started_at = time.time()
            try:
                job.image_ids = list(await self._runner(job))
            except JobCancelled:
                self._finish(job, CANCELLED)
            except asyncio.CancelledError:
                self._finish(job, CANCELLED)
                raise
            except Exception as exc:  # noqa: BLE001 -- one bad job must not stop the queue
                logger.warning("image_queue.job_failed: %s", exc)
                self._finish(
                    job,
                    CANCELLED if job.cancel_requested else FAILED,
                    str(exc) or type(exc).__name__,
                )
            else:
                self._finish(job, DONE)
