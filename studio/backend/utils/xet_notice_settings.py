# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""How many times the Xet "download is running" notice has been shown, for the life of the install.

The count used to live in the browser, in localStorage under
``unsloth.studio.xetNoticeCount``. localStorage is scoped to an ORIGIN, and a Studio
origin is not stable: ``run.py`` defaults to port 8888 and falls back to the next free
port when that is taken, and 8888 is also Jupyter's default. So starting Studio while a
notebook server is up moves it to 8889, which is a different origin with an empty store,
and the notice starts its three all over again. Colab and the Cloudflare tunnel are
different origins again, as is a second browser or a cleared profile. The user-visible
effect was a notice that reappeared indefinitely instead of a handful of times ever.

The install has exactly one database, so the count belongs here.
"""

from __future__ import annotations

import json
from typing import Any

XET_NOTICE_COUNT_KEY = "xet_notice_shown_count"

# Three sightings is enough to learn what a Xet transfer looks like. The cap lives on
# this side because the browser is no longer trusted to remember, and a limit enforced
# where the count is stored cannot disagree with it.
XET_NOTICE_LIMIT = 3


def _coerce_count(value: Any) -> int:
    """Anything unparseable reads as zero, matching a fresh install."""
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return max(0, value)
    if isinstance(value, str):
        try:
            return max(0, int(value.strip()))
        except ValueError:
            return 0
    return 0


def get_xet_notice_count() -> int:
    """Read the count. A missing or unreadable row means none have been shown."""
    try:
        from storage.studio_db import get_app_setting
        stored = get_app_setting(XET_NOTICE_COUNT_KEY, 0)
    except Exception:
        return 0
    return _coerce_count(stored)


def reserve_xet_notice(seen_hint: int = 0) -> dict[str, Any]:
    """Take one of the remaining notices, or report that none are left.

    Read-modify-write in ONE transaction. ``get_app_setting`` and
    ``upsert_app_settings`` each open their own connection, so doing this across the
    two of them lets two tabs read the same count and both be granted, which is how
    the browser implementation had to reach for Web Locks. ``BEGIN IMMEDIATE`` takes
    the write lock up front, so the second caller waits and then reads the first
    caller's value.

    ``seen_hint`` carries a legacy localStorage count from a client that has not
    reported one before. It can only raise the stored count, never lower it: a user
    who already spent their three before this moved server-side must not get three
    more, and a client cannot use the hint to talk its own way back under the limit.
    """
    from storage.studio_db import get_connection

    hint = _coerce_count(seen_hint)
    conn = get_connection()
    try:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            "SELECT value_json FROM app_settings WHERE key = ?",
            (XET_NOTICE_COUNT_KEY,),
        ).fetchone()
        stored = 0
        if row is not None:
            try:
                stored = _coerce_count(json.loads(row["value_json"]))
            except (ValueError, TypeError):
                stored = 0
        shown = max(stored, hint)

        granted = shown < XET_NOTICE_LIMIT
        if granted:
            shown += 1
        if shown != stored:
            from datetime import datetime, timezone
            now = datetime.now(timezone.utc).isoformat()
            conn.execute(
                """
                INSERT INTO app_settings (key, value_json, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    value_json = excluded.value_json,
                    updated_at = excluded.updated_at
                """,
                (XET_NOTICE_COUNT_KEY, json.dumps(shown), now),
            )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

    return {"granted": granted, "shown": shown, "limit": XET_NOTICE_LIMIT}
