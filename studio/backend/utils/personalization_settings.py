# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Server-side personalization (profile + appearance).

Profile name/avatar and appearance (theme, language) were stored only in the
browser's localStorage, so every browser or device that connected started from
defaults. Studio is single-account, so these are persisted as one JSON blob in
the ``app_settings`` table; the value follows the account across browsers and
devices. Validation of the blob shape lives in the route's pydantic model.
"""

from __future__ import annotations

PERSONALIZATION_SETTING_KEY = "personalization"
PERSONALIZATION_VERSION = 1
# Avatar data URLs are stored inline in studio.db; cap to keep the row sane.
MAX_AVATAR_DATA_URL_BYTES = 512 * 1024  # 512 KB


def get_personalization() -> dict:
    """The stored personalization blob, or an empty dict when none is saved."""
    try:
        from storage.studio_db import get_app_setting
        stored = get_app_setting(PERSONALIZATION_SETTING_KEY, None)
    except Exception:
        stored = None
    return stored if isinstance(stored, dict) else {}


def set_personalization(data: dict) -> dict:
    """Persist the (already validated) personalization blob."""
    from storage.studio_db import upsert_app_settings

    upsert_app_settings({PERSONALIZATION_SETTING_KEY: data})
    return data
