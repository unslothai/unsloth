# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Lifetime count of the Xet "download is running" notice.

It lived in localStorage, which is per-origin, and a Studio origin is not stable:
run.py falls back past port 8888 when Jupyter has it, and Colab and the tunnel differ
again. Each new origin handed out a fresh three, so the notice never stopped.
"""

from __future__ import annotations

import json
from typing import Any

XET_NOTICE_COUNT_KEY = "xet_notice_shown_count"

# Enforced where the count is stored, so the two cannot disagree.
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

    One transaction: get_app_setting and upsert_app_settings open a connection each,
    so splitting the read and write lets two tabs both be granted. BEGIN IMMEDIATE
    takes the write lock up front.

    seen_hint is a legacy localStorage count. It can only raise the stored value, so
    a client cannot talk its way back under the limit.
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
