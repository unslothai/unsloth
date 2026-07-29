# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
Usage numbers for the Profile settings tab.

Read-only aggregation over the local studio.db (see
``storage.profile_stats_db``). Nothing is uploaded.
"""

import asyncio
from typing import Any

from fastapi import APIRouter, Depends, Query

from auth.authentication import get_current_subject
from loggers import get_logger
from storage.profile_stats_db import MAX_DAILY_DAYS, compute_profile_stats
from utils.utils import log_and_http_error

router = APIRouter()

logger = get_logger(__name__)


@router.get("/stats")
async def get_profile_stats(
    days: int = Query(MAX_DAILY_DAYS, ge = 1, le = MAX_DAILY_DAYS),
    current_subject: str = Depends(get_current_subject),
) -> dict[str, Any]:
    """Usage stats for the signed-in user's local history."""
    try:
        # A cold pass parses every message's metadata JSON: ~90 ms at 10k
        # messages, ~1.2 s at 260k. Off the event loop so it cannot stall token
        # streaming when Settings is opened mid-generation.
        return await asyncio.to_thread(compute_profile_stats, days = days)
    except Exception as exc:
        raise log_and_http_error(
            exc, 500, "Failed to compute profile statistics", log = logger
        ) from exc
