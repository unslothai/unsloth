# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""whisper.cpp prebuilt status endpoint, kept for old clients.

GET  /api/whisper/update-status  -> always ``supported: False`` with no update

whisper.cpp is retired: Unsloth Studio no longer installs, updates or re-pairs it, so this answers without
probing the install or GitHub. The response shape is unchanged so older frontends still parse it.
"""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field

from auth.authentication import get_current_subject

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


@router.get("/update-status", response_model = WhisperUpdateStatusResponse)
async def whisper_update_status(
    force_refresh: bool = Query(
        False, description = "Bypass the 24h release cache for an explicit check."
    ),
    current_subject: str = Depends(get_current_subject),
) -> WhisperUpdateStatusResponse:
    return WhisperUpdateStatusResponse()
