# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from threading import RLock
from typing import Literal


MediaGenerationKind = Literal["image", "video"]
_settings_lock = RLock()
_MAX_PRESETS = 100


def _setting_key(kind: MediaGenerationKind) -> str:
    return f"{kind}_generation_presets"


def _stored_settings(kind: MediaGenerationKind) -> dict:
    from storage.studio_db import get_app_setting
    stored = get_app_setting(_setting_key(kind), {})
    return dict(stored) if isinstance(stored, dict) else {}


def _with_unknown_preserved(stored, updated):
    """Carry forward what the writing build does not model.

    A GET drops fields the reading build cannot validate, so the state it sends back is a lossy
    view of the store. Without this, opening a store once with an older build would erase every
    field a newer one had written. Only a background state write goes through here; replacing a
    named preset is a deliberate whole-value overwrite by the user.
    """
    if not isinstance(stored, dict) or not isinstance(updated, dict):
        return updated
    merged = dict(updated)
    for key, value in stored.items():
        merged[key] = value if key not in merged else _with_unknown_preserved(value, merged[key])
    return merged


def _custom_presets(stored: dict) -> list:
    presets = stored.get("customPresets", [])
    return [item for item in presets if isinstance(item, dict)] if isinstance(presets, list) else []


def get_media_generation_preset_settings(kind: MediaGenerationKind) -> dict:
    with _settings_lock:
        return _stored_settings(kind)


def set_media_generation_preset_settings(kind: MediaGenerationKind, settings: dict) -> None:
    """Write the page's current recipe and selection, never the named presets.

    Those have their own endpoints so a debounced state write can never race a save or a delete
    into clobbering the list.
    """
    from storage.studio_db import upsert_app_settings
    with _settings_lock:
        stored = _stored_settings(kind)
        updated = _with_unknown_preserved(
            stored,
            {**settings, "customPresets": stored.get("customPresets", [])},
        )
        upsert_app_settings({_setting_key(kind): updated})


def upsert_media_generation_preset(kind: MediaGenerationKind, preset: dict) -> None:
    from storage.studio_db import upsert_app_settings
    with _settings_lock:
        stored = _stored_settings(kind)
        presets = _custom_presets(stored)
        replacing = any(item.get("name") == preset["name"] for item in presets)
        if not replacing and len(presets) >= _MAX_PRESETS:
            raise ValueError("Delete a preset before saving another one")
        stored["customPresets"] = [
            item for item in presets if item.get("name") != preset["name"]
        ] + [preset]
        upsert_app_settings({_setting_key(kind): stored})


def delete_media_generation_preset(kind: MediaGenerationKind, name: str) -> None:
    from storage.studio_db import upsert_app_settings
    with _settings_lock:
        stored = _stored_settings(kind)
        stored["customPresets"] = [
            item for item in _custom_presets(stored) if item.get("name") != name
        ]
        upsert_app_settings({_setting_key(kind): stored})
