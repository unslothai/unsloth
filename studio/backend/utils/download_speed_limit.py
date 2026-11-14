# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Download speed limit in Mbit/s saved per install; nothing stored means no limit."""

from __future__ import annotations

import math
from typing import Any, Optional

from loggers import get_logger

logger = get_logger(__name__)

DOWNLOAD_SPEED_LIMIT_SETTING_KEY = "download_speed_limit_mbps"
DOWNLOAD_SPEED_LIMIT_ENV = "UNSLOTH_DOWNLOAD_SPEED_LIMIT_MBPS"


class DownloadSpeedLimitUnavailableError(RuntimeError):
    """The saved policy could not be read, so a download must not start."""


def normalize_speed_limit_mbps(value: Any) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    mbps = float(value)
    if not math.isfinite(mbps) or mbps <= 0:
        return None
    return mbps


def get_download_speed_limit_mbps() -> Optional[float]:
    try:
        from storage.studio_db import get_app_setting
        from utils.account_context import OWNER, run_as

        stored = run_as(OWNER, get_app_setting, DOWNLOAD_SPEED_LIMIT_SETTING_KEY, None)
        return normalize_speed_limit_mbps(stored)
    except Exception as exc:  # noqa: BLE001 - an unreadable policy must not remove the limit
        logger.warning("download speed limit read failed (%s)", exc)
        raise DownloadSpeedLimitUnavailableError(
            "Cannot read the saved download speed limit. "
            "Retry when the settings database is available."
        ) from None


def set_download_speed_limit_mbps(value: Any) -> Optional[float]:
    if value is None or value == 0:
        mbps = None
    else:
        mbps = normalize_speed_limit_mbps(value)
        if mbps is None:
            raise ValueError(
                "Download speed limit must be a number of Mbit/s above zero, or empty."
            )

    from storage.studio_db import upsert_app_settings
    from utils.account_context import OWNER, run_as

    run_as(OWNER, upsert_app_settings, {DOWNLOAD_SPEED_LIMIT_SETTING_KEY: mbps})
    return mbps


def speed_limit_bytes_per_second(raw: Optional[str]) -> Optional[float]:
    try:
        mbps = normalize_speed_limit_mbps(float(raw)) if raw else None
    except ValueError:
        return None
    return None if mbps is None else mbps * 125_000
