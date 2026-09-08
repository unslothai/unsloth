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
    """Anything unparseable reads as "never dismissed", matching a fresh install: a
    corrupt row must not wedge the notice off permanently, so this fails toward
    showing it rather than hiding it.
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

    Infinity is the one that matters: Python's json accepts ``Infinity``, so a client
    could otherwise dismiss at a value no machine will ever exceed and silence the
    notice permanently, the opposite of the rule above.
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
    """Whether the notice should stay silent at an allocation of ``current_gb``.

    Silent only while the allocation is unchanged or smaller. An unknown allocation
    counts as dismissed once anything has been, since re-showing on a reading we
    cannot compare is the nagging this exists to avoid.
    """
    dismissed_at = get_dismissed_at_gb()
    if dismissed_at is None:
        return False
    if current_gb is None:
        return True
    # A tenth of a GB of slack: the driver-reported byte count need not be identical
    # across boots (95.83 against a 96.00 setting here), and a rounding redisplay would
    # look like the notice ignoring the dismissal. Compared in TENTHS because binary
    # floats miss that grid: 95.8 + 0.1 is 95.89999999999999, so a 95.9 reading -- one
    # tenth away, the case this slack exists for -- read as not dismissed.
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
