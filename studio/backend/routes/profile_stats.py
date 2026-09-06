# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
Usage numbers for the Profile settings tab.

Read-only aggregation over the local studio.db (see
``storage.profile_stats_db``). Nothing is uploaded.
"""

import asyncio
import threading
from typing import Any

from fastapi import APIRouter, Depends, Query

from auth.authentication import get_current_subject
from auth import policy
from loggers import get_logger
from storage.profile_stats_db import (
    MAX_DAILY_DAYS,
    MAX_TZ_OFFSET_MINUTES,
    compute_profile_stats,
    invalidate_profile_stats_cache,
)
from utils.utils import log_and_http_error
from utils.account_context import current_account_id
from utils.paths import studio_db_path

router = APIRouter()

logger = get_logger(__name__)
_stats_lock = threading.Lock()
_stats_account = None


def _compute_account_profile_stats(**kwargs) -> dict[str, Any]:
    """Fence the storage cache, including when a username is deleted and reused."""
    global _stats_account
    if not policy.installation_is_multi_user():
        _stats_account = None
        return compute_profile_stats(**kwargs)
    scope = (current_account_id(), str(studio_db_path()))
    with _stats_lock:
        if _stats_account != scope:
            invalidate_profile_stats_cache()
            _stats_account = scope
        return compute_profile_stats(**kwargs)


@router.get("/stats")
async def get_profile_stats(
    days: int = Query(MAX_DAILY_DAYS, ge = 1, le = MAX_DAILY_DAYS),
    tz_offset_minutes: int = Query(0, ge = -MAX_TZ_OFFSET_MINUTES, le = MAX_TZ_OFFSET_MINUTES),
    tz: str = Query("", max_length = 64),
    current_subject: str = Depends(get_current_subject),
) -> dict[str, Any]:
    """Usage stats from the caller's database, including private chats and training.

    API receipts additionally retain their existing subject filter.

    Days and hours are bucketed in the caller's timezone so a remote browser
    does not read the server's calendar. ``tz`` is an IANA name, which carries
    each date's own daylight-saving offset; ``tz_offset_minutes`` is the
    ``Date.getTimezoneOffset()`` fallback for hosts with no tzdata.
    """
    try:
        # A cold pass parses every message's metadata JSON: ~90 ms at 10k
        # messages, ~1.2 s at 260k. Off the event loop so it cannot stall token
        # streaming when Settings is opened mid-generation.
        return await asyncio.to_thread(
            _compute_account_profile_stats,
            days = days,
            tz_offset_minutes = tz_offset_minutes,
            tz_name = tz,
            subject = current_subject,
        )
    except Exception as exc:
        raise log_and_http_error(
            exc, 500, "Failed to compute profile statistics", log = logger
        ) from exc
