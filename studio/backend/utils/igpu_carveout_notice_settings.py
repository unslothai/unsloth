# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Whether the "enlarge the integrated GPU's memory" notice has been dismissed.

Server-side rather than localStorage, for the reason xet_notice_settings.py gives:
an Unsloth origin is not stable, so a per-origin store hands out a fresh notice
every time the port moves and the advice never stops.

Dismissal records the allocation it was dismissed AT, not a bare boolean. Someone
who dismisses this at a 32 GB allocation and later raises it to 64 GB has acted on
the advice; if they then load a model too big for 64 GB, that is new information and
worth saying once more. A bare flag would silence the second case forever.
"""

from __future__ import annotations

import math
from typing import Any, Optional

IGPU_CARVEOUT_NOTICE_KEY = "igpu_carveout_notice_dismissed_at_gb"


def _coerce_gb(value: Any) -> Optional[float]:
    """Anything unparseable reads as "never dismissed", matching a fresh install.

    A corrupt row must not be able to wedge the notice off permanently, so this
    fails toward showing it rather than hiding it.
    """
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
    """Whether a number could be a GPU allocation someone actually has.

    Infinity is the one that matters. JSON as Python parses it accepts
    ``Infinity``, so without this a client could dismiss the notice at an
    allocation no machine will ever exceed and silence it permanently -- the
    opposite of the "fail toward showing it" rule above.
    """
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
    """Whether the notice should stay silent for an allocation of ``current_gb``.

    Silent only while the allocation is unchanged or smaller. Unknown current
    allocation is treated as "already dismissed" once anything has been dismissed,
    because re-showing on a reading we cannot compare would be the nagging this is
    meant to avoid.
    """
    dismissed_at = get_dismissed_at_gb()
    if dismissed_at is None:
        return False
    if current_gb is None:
        return True
    # A tenth of a GB of slack: the allocation is read from a driver-reported byte
    # count that need not be identical across boots (95.83 against a 96.00 setting
    # on the development machine), and a redisplay caused by rounding would look
    # like the notice ignoring the dismissal.
    #
    # Compared in TENTHS, because both sides arrive already rounded to one decimal
    # and binary floats do not land on that grid: 95.8 + 0.1 is 95.89999999999999,
    # so a reading of 95.9 -- one tenth away, the case this slack exists for -- read
    # as not dismissed and the notice came back on an unchanged allocation.
    return round(float(current_gb) * 10) <= round(dismissed_at * 10) + 1


def dismiss_notice(current_gb: Optional[float]) -> Optional[float]:
    """Record dismissal at ``current_gb``. Returns what was stored, or None.

    Only ever raises the stored value, so a stale client reporting an old, smaller
    allocation cannot re-arm a notice the user already dismissed at a larger one.
    """
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
