# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""API route backing the composer's "attach transcript" prompt for pasted YouTube links."""

from __future__ import annotations

from typing import Annotated

import httpx
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from auth.authentication import get_current_subject
from core.youtube_transcript import (
    TranscriptUnavailable,
    extract_video_id,
    fetch_transcript,
    watch_url,
)
from loggers import get_logger

logger = get_logger(__name__)

router = APIRouter()


class TranscriptRequest(BaseModel):
    url: str = Field(max_length = 2048)
    # Caption languages to prefer, best first. The UI forwards navigator.languages, so the
    # per-tag cap has to clear a fully extended BCP 47 tag rather than reject the request.
    languages: list[Annotated[str, Field(max_length = 64)]] = Field(
        default_factory = list,
        max_length = 8,
    )


class TranscriptResponse(BaseModel):
    videoId: str
    url: str
    title: str
    author: str
    lengthSeconds: int
    language: str
    languageCode: str
    isGenerated: bool
    text: str


@router.post("/transcript", response_model = TranscriptResponse)
async def get_transcript(
    request: TranscriptRequest,
    current_subject: str = Depends(get_current_subject),
) -> TranscriptResponse:
    video_id = extract_video_id(request.url)
    if video_id is None:
        raise HTTPException(status_code = 400, detail = "That is not a YouTube video link.")

    try:
        transcript = await fetch_transcript(video_id, request.languages)
    except TranscriptUnavailable as error:
        raise HTTPException(status_code = 422, detail = str(error)) from error
    except httpx.HTTPError as error:
        logger.warning(f"YouTube transcript fetch failed for {video_id}: {error}")
        raise HTTPException(
            status_code = 502,
            detail = "Could not reach YouTube.",
        ) from error

    return TranscriptResponse(
        videoId = transcript.video_id,
        url = watch_url(transcript.video_id),
        title = transcript.title,
        author = transcript.author,
        lengthSeconds = transcript.length_seconds,
        language = transcript.language,
        languageCode = transcript.language_code,
        isGenerated = transcript.is_generated,
        text = transcript.text,
    )
