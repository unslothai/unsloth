# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Usage numbers for the Profile settings tab. Read-only aggregation over the local studio.db (see
``storage.profile_stats_db``). Nothing is uploaded.
"""

import asyncio
from datetime import date
from typing import Any

from fastapi import APIRouter, Depends, Query
from fastapi.responses import Response

from auth.authentication import get_current_subject
from loggers import get_logger
from storage.profile_stats_db import (
    MAX_DAILY_DAYS,
    MAX_TZ_OFFSET_MINUTES,
    compute_profile_stats,
    export_usage_csv,
)
from utils.utils import log_and_http_error

router = APIRouter()

logger = get_logger(__name__)


@router.get("/stats")
async def get_profile_stats(
    days: int = Query(MAX_DAILY_DAYS, ge = 1, le = MAX_DAILY_DAYS),
    tz_offset_minutes: int = Query(0, ge = -MAX_TZ_OFFSET_MINUTES, le = MAX_TZ_OFFSET_MINUTES),
    tz: str = Query("", max_length = 64),
    current_subject: str = Depends(get_current_subject),
) -> dict[str, Any]:
    """Usage stats from the caller's database, bucketed in the caller's timezone (``tz``, or the ``tz_offset_minutes`` fallback)."""
    try:
        # Cold pass can take ~1 s on large histories; keep it off the event loop.
        return await asyncio.to_thread(
            compute_profile_stats,
            days = days,
            tz_offset_minutes = tz_offset_minutes,
            tz_name = tz,
            subject = current_subject,
        )
    except Exception as exc:
        raise log_and_http_error(
            exc, 500, "Failed to compute profile statistics", log = logger
        ) from exc


@router.get("/stats/export")
async def export_profile_usage(
    tz_offset_minutes: int = Query(0, ge = -MAX_TZ_OFFSET_MINUTES, le = MAX_TZ_OFFSET_MINUTES),
    tz: str = Query("", max_length = 64),
    current_subject: str = Depends(get_current_subject),
) -> Response:
    """The token usage behind the stats as CSV, one row per chat response or API request."""
    try:
        body = await asyncio.to_thread(
            export_usage_csv,
            tz_offset_minutes = tz_offset_minutes,
            tz_name = tz,
            subject = current_subject,
        )
    except Exception as exc:
        raise log_and_http_error(exc, 500, "Failed to export token usage", log = logger) from exc
    filename = f"unsloth-token-usage-{date.today().isoformat()}.csv"
    return Response(
        content = body,
        media_type = "text/csv; charset=utf-8",
        headers = {"Content-Disposition": f'attachment; filename="{filename}"'},
    )
