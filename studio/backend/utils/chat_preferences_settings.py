# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Installation-wide chat preferences."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Optional


MODEL_DISCLAIMER_SETTING_KEY = "chat_show_model_disclaimer"
DEFAULT_SHOW_MODEL_DISCLAIMER = False


def _stored_bool(value: object) -> Optional[bool]:
    return value if isinstance(value, bool) else None


def get_show_model_disclaimer() -> bool:
    from storage.studio_db import get_app_setting
    stored = _stored_bool(get_app_setting(MODEL_DISCLAIMER_SETTING_KEY, None))
    return DEFAULT_SHOW_MODEL_DISCLAIMER if stored is None else stored


def set_show_model_disclaimer(enabled: bool) -> bool:
    if not isinstance(enabled, bool):
        raise ValueError("Model disclaimer setting must be a boolean.")
    from storage.studio_db import upsert_app_settings

    upsert_app_settings({MODEL_DISCLAIMER_SETTING_KEY: enabled})
    return enabled


def migrate_show_model_disclaimer(legacy: Optional[bool]) -> bool:
    """Import an enabled browser value only when the server has none."""
    if legacy is not None and not isinstance(legacy, bool):
        raise ValueError("Legacy model disclaimer setting must be a boolean.")

    from storage.studio_db import get_connection

    conn = get_connection()
    try:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            "SELECT value_json FROM app_settings WHERE key = ?",
            (MODEL_DISCLAIMER_SETTING_KEY,),
        ).fetchone()
        stored = None
        if row is not None:
            try:
                stored = _stored_bool(json.loads(row["value_json"]))
            except (TypeError, ValueError):
                stored = None

        if stored is not None:
            conn.commit()
            return stored
        if legacy is not True:
            conn.commit()
            return DEFAULT_SHOW_MODEL_DISCLAIMER

        now = datetime.now(timezone.utc).isoformat()
        conn.execute(
            """
            INSERT INTO app_settings (key, value_json, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET
                value_json = excluded.value_json,
                updated_at = excluded.updated_at
            """,
            (MODEL_DISCLAIMER_SETTING_KEY, json.dumps(legacy), now),
        )
        conn.commit()
        return legacy
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
