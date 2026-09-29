# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

from typing import Any

PILL_ENABLED_KEY = "pill_enabled"
PILL_DEFAULT_MODEL_KEY = "pill_default_model"
PILL_DEFAULT_GGUF_VARIANT_KEY = "pill_default_gguf_variant"
PILL_AUTO_LOAD_KEY = "pill_auto_load"
_KEYS = [
    PILL_ENABLED_KEY,
    PILL_DEFAULT_MODEL_KEY,
    PILL_DEFAULT_GGUF_VARIANT_KEY,
    PILL_AUTO_LOAD_KEY,
]


def get_pill_settings() -> dict:
    # Not guarded: returning defaults on a read error made startup sync unregister a working shortcut.
    from storage.studio_db import get_app_settings
    stored = get_app_settings(_KEYS)
    return {
        "enabled": stored.get(PILL_ENABLED_KEY) is True,
        "defaultModel": stored.get(PILL_DEFAULT_MODEL_KEY) or None,
        "defaultGgufVariant": stored.get(PILL_DEFAULT_GGUF_VARIANT_KEY) or None,
        "autoLoad": stored.get(PILL_AUTO_LOAD_KEY) is not False,
    }


UNSET: Any = object()


def update_pill_settings(
    enabled: Any = UNSET,
    default_model: Any = UNSET,
    default_gguf_variant: Any = UNSET,
    auto_load: Any = UNSET,
) -> dict:
    """Partial update: an omitted field is left untouched, None clears it."""
    from storage.studio_db import upsert_app_settings

    updates: dict[str, Any] = {}
    if enabled is not UNSET:
        updates[PILL_ENABLED_KEY] = enabled
    if default_model is not UNSET:
        updates[PILL_DEFAULT_MODEL_KEY] = default_model or None
    if default_gguf_variant is not UNSET:
        updates[PILL_DEFAULT_GGUF_VARIANT_KEY] = default_gguf_variant or None
    if auto_load is not UNSET:
        updates[PILL_AUTO_LOAD_KEY] = auto_load
    if updates:
        upsert_app_settings(updates)
    return get_pill_settings()
