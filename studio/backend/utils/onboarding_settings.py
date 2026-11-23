# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
from __future__ import annotations

import time

ONBOARDING_KEY = "onboarding"


def _now_ms() -> int:
    return int(time.time() * 1000)


def get_onboarding() -> dict:
    from storage.studio_db import get_app_setting
    stored = get_app_setting(ONBOARDING_KEY, None)
    return stored if isinstance(stored, dict) else {}


def account_has_chat_history() -> bool:
    from storage.studio_db import get_connection

    conn = get_connection()
    try:
        return conn.execute("SELECT EXISTS(SELECT 1 FROM chat_threads)").fetchone()[0] == 1
    finally:
        conn.close()


def update_onboarding(pages: list[str], completed: bool) -> dict:
    from storage.studio_db import compare_and_set_app_setting, get_app_setting
    while True:
        previous = get_app_setting(ONBOARDING_KEY, None)
        stored = previous if isinstance(previous, dict) else {}
        seen = [p for p in stored.get("pages", []) if isinstance(p, str)]
        merged_pages = seen + [p for p in dict.fromkeys(pages) if p not in seen]
        completed_at = stored.get("completedAt")
        if completed and not isinstance(completed_at, int):
            completed_at = _now_ms()
        merged = {
            "pages": merged_pages,
            "completedAt": completed_at,
            "updatedAt": _now_ms(),
        }
        if compare_and_set_app_setting(ONBOARDING_KEY, previous, merged):
            return merged
