# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""whisper.cpp prebuilt status endpoint.

GET  /api/whisper/update-status  -> is a newer prebuilt available + job state

Detection reuses utils.whisper_cpp_freshness and fails open so the UI never
blocks on a missing marker / offline GitHub. There is no whisper-only update
trigger here: whisper updates piggyback on the single main update item
(POST /api/llama/update chains a whisper phase when whisper is behind), and
utils.whisper_cpp_update.start_update stays available for CLI/tests.
"""

from __future__ import annotations

import asyncio
import threading
from typing import Optional

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field

from auth.authentication import get_current_subject
from loggers import get_logger
from utils.whisper_cpp_update import get_update_status

logger = get_logger(__name__)
router = APIRouter()


class WhisperUpdateJob(BaseModel):
    state: str = Field("idle", description = "idle | running | success | error")
    message: str = ""
    from_tag: Optional[str] = None
    to_tag: Optional[str] = None
    reload_required: Optional[bool] = None
    error: Optional[str] = None
    progress: Optional[float] = Field(None, description = "0..1 while running, 1 on success.")
    started_at: Optional[str] = None
    finished_at: Optional[str] = None


class WhisperUpdateStatusResponse(BaseModel):
    supported: bool = Field(
        False,
        description = "True when the install came from an Unsloth prebuilt (has a marker).",
    )
    update_available: bool = Field(
        False, description = "True when the latest release is genuinely newer than the install."
    )
    stale: bool = Field(
        False, description = "Update available AND install older than the staleness threshold."
    )
    installed_tag: Optional[str] = None
    latest_tag: Optional[str] = None
    published_repo: Optional[str] = None
    installed_at_utc: Optional[str] = None
    age_days: Optional[int] = None
    source_build: bool = Field(
        False, description = "True when there is no marker (source build) but a prebuilt is offered."
    )
    update_size_bytes: Optional[int] = Field(
        None, description = "Download size of the prebuilt an update would fetch, in bytes."
    )
    job: WhisperUpdateJob = Field(default_factory = WhisperUpdateJob)


_whisper_update_lock = threading.Lock()
_last_whisper_update_step = -1


def _log_whisper_update_progress(job: WhisperUpdateJob) -> None:
    """One whisper_update_progress line per 10% step so a prebuilt update reports
    progress without a line per poll. Resyncs when a new update starts."""
    global _last_whisper_update_step
    if job.state != "running" or job.progress is None:
        return
    step = int(max(0.0, min(float(job.progress), 1.0)) * 10)
    with _whisper_update_lock:
        prev = _last_whisper_update_step
        if step == prev:
            return
        _last_whisper_update_step = step
        if step < prev:
            return  # new update; resync without logging
    logger.info("whisper_update_progress", to_tag = job.to_tag or "", percent = step * 10)


@router.get("/update-status", response_model = WhisperUpdateStatusResponse)
async def whisper_update_status(
    force_refresh: bool = Query(
        False, description = "Bypass the 24h release cache for an explicit check."
    ),
    current_subject: str = Depends(get_current_subject),
) -> WhisperUpdateStatusResponse:
    # Off the event loop: detection may probe the host and read GitHub.
    status = await asyncio.to_thread(get_update_status, force_refresh = force_refresh)
    resp = WhisperUpdateStatusResponse(**status)
    _log_whisper_update_progress(resp.job)
    return resp
