# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Whether the "enlarge the integrated GPU's memory" notice has been dismissed.

Server-side rather than localStorage, for the reason xet_notice_settings.py gives:
an Unsloth origin is not stable, so a per-origin store hands out a fresh notice
every time the port moves.

Dismissal records the allocation it was dismissed AT, not a bare boolean: someone who
acts on the advice, raises 32 GB to 64 GB and still runs short is in a new situation
worth one more mention, which a flag would silence forever.
"""

from __future__ import annotations

import math
from typing import Any, Optional

IGPU_CARVEOUT_NOTICE_KEY = "igpu_carveout_notice_dismissed_at_gb"


def _coerce_gb(value: Any) -> Optional[float]:
    """Unparsable reads as never dismissed; a corrupt row must not hide the notice permanently."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value) if _is_plausible_gb(value) else None
    if isinstance(value, str):
        try:
            parsed = float(value.strip())
        except ValueError:
            return None
        return parsed if _is_plausible_gb(parsed) else None
    return None


def _is_plausible_gb(value: float) -> bool:
    """Rejects non-finite values: Python's json accepts Infinity, which would silence the notice forever."""
    return math.isfinite(value) and 0 < value < 1024 * 1024


def get_dismissed_at_gb() -> Optional[float]:
    """The GPU allocation the notice was last dismissed at, or None."""
    try:
        from storage.studio_db import get_app_setting
        stored = get_app_setting(IGPU_CARVEOUT_NOTICE_KEY, None)
    except Exception:
        return None
    return _coerce_gb(stored)


def notice_already_dismissed(current_gb: Optional[float]) -> bool:
    """Silent only while the allocation is unchanged or smaller; an unknown size counts as dismissed."""
    dismissed_at = get_dismissed_at_gb()
    if dismissed_at is None:
        return False
    if current_gb is None:
        return True
    # Tenth-of-GB slack for driver jitter; compare in integer tenths to avoid float error.
    return round(float(current_gb) * 10) <= round(dismissed_at * 10) + 1


def dismiss_notice(current_gb: Optional[float]) -> Optional[float]:
    """Only raises the stored value, so a stale, smaller report cannot re-arm a dismissed notice."""
    if current_gb is None:
        return get_dismissed_at_gb()
    try:
        value = float(current_gb)
    except (TypeError, ValueError):
        return get_dismissed_at_gb()
    if not _is_plausible_gb(value):
        return get_dismissed_at_gb()

    existing = get_dismissed_at_gb()
    if existing is not None and existing >= value:
        return existing
    try:
        from storage.studio_db import upsert_app_settings
        upsert_app_settings({IGPU_CARVEOUT_NOTICE_KEY: value})
    except Exception:
        return existing
    return value
