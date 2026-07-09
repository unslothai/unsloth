# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

PERSONALIZATION_SETTING_KEY = "personalization"
PERSONALIZATION_VERSION = 1
MAX_AVATAR_DATA_URL_BYTES = 512 * 1024


def get_personalization() -> dict:
    from storage.studio_db import get_app_setting
    stored = get_app_setting(PERSONALIZATION_SETTING_KEY, None)
    return stored if isinstance(stored, dict) else {}


def set_personalization(data: dict) -> dict:
    from storage.studio_db import upsert_app_settings
    upsert_app_settings({PERSONALIZATION_SETTING_KEY: data})
    return data
