# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from threading import RLock
from typing import Literal


MediaGenerationKind = Literal["image", "video"]
PresetWriteOrder = tuple[int | None, str | None]
_settings_lock = RLock()
_WRITE_VERSIONS = "_writeVersions"
_MAX_WRITE_VERSIONS = 1024


def _setting_key(kind: MediaGenerationKind) -> str:
    return f"{kind}_generation_presets"


def _version(write: PresetWriteOrder) -> tuple[int, str] | None:
    timestamp, writer = write
    if timestamp is None or writer is None:
        return None
    return timestamp, writer


def _stored_settings(kind: MediaGenerationKind) -> dict:
    from storage.studio_db import get_app_setting
    stored = get_app_setting(_setting_key(kind), {})
    return dict(stored) if isinstance(stored, dict) else {}


def _public_settings(stored: dict) -> dict:
    public = dict(stored)
    public.pop(_WRITE_VERSIONS, None)
    public.pop("activePresetSource", None)
    return public


def _is_newer(stored: dict, scope: str, write: PresetWriteOrder) -> bool:
    version = _version(write)
    if version is None:
        return True
    versions = stored.get(_WRITE_VERSIONS, {})
    current = versions.get(scope) if isinstance(versions, dict) else None
    if not isinstance(current, list) or len(current) != 2:
        return True
    try:
        current_version = int(current[0]), str(current[1])
    except (TypeError, ValueError):
        return True
    return version > current_version


def _record_write(stored: dict, scope: str, write: PresetWriteOrder) -> None:
    version = _version(write)
    if version is None:
        return
    raw = stored.get(_WRITE_VERSIONS, {})
    versions = dict(raw) if isinstance(raw, dict) else {}
    versions.pop(scope, None)
    versions[scope] = list(version)
    while len(versions) > _MAX_WRITE_VERSIONS:
        versions.pop(next(iter(versions)))
    stored[_WRITE_VERSIONS] = versions


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
        merged[key] = (
            value if key not in merged else _with_unknown_preserved(value, merged[key])
        )
    return merged


def get_media_generation_preset_settings(kind: MediaGenerationKind) -> dict:
    with _settings_lock:
        return _public_settings(_stored_settings(kind))


def set_media_generation_preset_settings(
    kind: MediaGenerationKind,
    settings: dict,
    write: PresetWriteOrder = (None, None),
) -> bool:
    from storage.studio_db import upsert_app_settings
    with _settings_lock:
        stored = _stored_settings(kind)
        if not _is_newer(stored, "settings", write):
            return False
        updated = _with_unknown_preserved(
            stored,
            {**settings, "customPresets": stored.get("customPresets", [])},
        )
        _record_write(updated, "settings", write)
        upsert_app_settings({_setting_key(kind): updated})
        return True


def upsert_media_generation_preset(
    kind: MediaGenerationKind,
    preset: dict,
    write: PresetWriteOrder = (None, None),
) -> bool:
    from storage.studio_db import upsert_app_settings
    with _settings_lock:
        stored = _stored_settings(kind)
        scope = f"custom:{preset['name']}"
        if not _is_newer(stored, scope, write):
            return False
        presets = stored.get("customPresets", [])
        presets = presets if isinstance(presets, list) else []
        replacing = any(item.get("name") == preset["name"] for item in presets)
        if not replacing and len(presets) >= 100:
            raise ValueError("Delete a preset before saving another one")
        stored["customPresets"] = [
            item for item in presets if item.get("name") != preset["name"]
        ] + [preset]
        _record_write(stored, scope, write)
        upsert_app_settings({_setting_key(kind): stored})
        return True


def delete_media_generation_preset(
    kind: MediaGenerationKind,
    name: str,
    write: PresetWriteOrder = (None, None),
) -> bool:
    from storage.studio_db import upsert_app_settings
    with _settings_lock:
        stored = _stored_settings(kind)
        scope = f"custom:{name}"
        if not _is_newer(stored, scope, write):
            return False
        presets = stored.get("customPresets", [])
        presets = presets if isinstance(presets, list) else []
        stored["customPresets"] = [item for item in presets if item.get("name") != name]
        _record_write(stored, scope, write)
        upsert_app_settings({_setting_key(kind): stored})
        return True
