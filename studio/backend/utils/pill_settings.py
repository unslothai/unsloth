# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

from typing import Any

PILL_ENABLED_KEY = "pill_enabled"
PILL_DEFAULT_MODEL_KEY = "pill_default_model"
PILL_DEFAULT_GGUF_VARIANT_KEY = "pill_default_gguf_variant"
PILL_AUTO_LOAD_KEY = "pill_auto_load"
PILL_EXCLUDED_APPS_KEY = "pill_excluded_apps"


def _get_setting(key: str, default: Any = None) -> Any:
    try:
        from storage.studio_db import get_app_setting

        return get_app_setting(key, default)
    except Exception:
        return default


def get_pill_settings() -> dict:
    excluded = _get_setting(PILL_EXCLUDED_APPS_KEY, [])
    if not isinstance(excluded, list):
        excluded = []
    return {
        "enabled": _get_setting(PILL_ENABLED_KEY) is True,
        "defaultModel": _get_setting(PILL_DEFAULT_MODEL_KEY) or None,
        "defaultGgufVariant": _get_setting(PILL_DEFAULT_GGUF_VARIANT_KEY) or None,
        "autoLoad": _get_setting(PILL_AUTO_LOAD_KEY) is not False,
        "excludedApps": [str(item) for item in excluded],
    }


def update_pill_settings(
    enabled: bool | None = None,
    default_model: str | None = None,
    default_gguf_variant: str | None = None,
    auto_load: bool | None = None,
    excluded_apps: list[str] | None = None,
) -> dict:
    """Partial update: None leaves a field untouched."""
    from storage.studio_db import upsert_app_settings

    updates: dict[str, Any] = {}
    if enabled is not None:
        updates[PILL_ENABLED_KEY] = enabled
    if default_model is not None:
        updates[PILL_DEFAULT_MODEL_KEY] = default_model or None
    if default_gguf_variant is not None:
        updates[PILL_DEFAULT_GGUF_VARIANT_KEY] = default_gguf_variant or None
    if auto_load is not None:
        updates[PILL_AUTO_LOAD_KEY] = auto_load
    if excluded_apps is not None:
        updates[PILL_EXCLUDED_APPS_KEY] = [str(item) for item in excluded_apps]
    if updates:
        upsert_app_settings(updates)
    return get_pill_settings()
