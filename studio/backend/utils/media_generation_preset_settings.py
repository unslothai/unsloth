# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from collections import OrderedDict
from threading import RLock
from typing import Literal


MediaGenerationKind = Literal["image", "video"]
PresetWriteOrder = tuple[str | None, int | None]
_settings_lock = RLock()
_latest_writes: OrderedDict[tuple[MediaGenerationKind, str, str], int] = OrderedDict()


def _setting_key(kind: MediaGenerationKind) -> str:
    return f"{kind}_generation_presets"


def _accept_write(
    kind: MediaGenerationKind,
    scope: str,
    write: PresetWriteOrder,
) -> bool:
    writer, sequence = write
    if writer is None or sequence is None:
        return True
    key = (kind, scope, writer)
    if sequence <= _latest_writes.get(key, -1):
        return False
    _latest_writes[key] = sequence
    _latest_writes.move_to_end(key)
    if len(_latest_writes) > 1024:
        _latest_writes.popitem(last = False)
    return True


def get_media_generation_preset_settings(kind: MediaGenerationKind) -> dict:
    from storage.studio_db import get_app_setting

    with _settings_lock:
        stored = get_app_setting(_setting_key(kind), {})
        return stored if isinstance(stored, dict) else {}


def set_media_generation_preset_settings(
    kind: MediaGenerationKind,
    settings: dict,
    write: PresetWriteOrder = (None, None),
) -> dict:
    from storage.studio_db import upsert_app_settings

    with _settings_lock:
        stored = get_media_generation_preset_settings(kind)
        if not _accept_write(kind, "settings", write):
            return stored
        settings["customPresets"] = stored.get("customPresets", [])
        upsert_app_settings({_setting_key(kind): settings})
        return settings


def upsert_media_generation_preset(
    kind: MediaGenerationKind,
    preset: dict,
    write: PresetWriteOrder = (None, None),
) -> None:
    from storage.studio_db import upsert_app_settings

    with _settings_lock:
        if not _accept_write(kind, "custom", write):
            return
        stored = get_media_generation_preset_settings(kind)
        presets = stored.get("customPresets", [])
        replacing = any(item.get("name") == preset["name"] for item in presets)
        if not replacing and len(presets) >= 100:
            raise ValueError("Delete a preset before saving another one")
        stored["customPresets"] = [
            item for item in presets if item.get("name") != preset["name"]
        ] + [preset]
        upsert_app_settings({_setting_key(kind): stored})


def delete_media_generation_preset(
    kind: MediaGenerationKind,
    name: str,
    write: PresetWriteOrder = (None, None),
) -> None:
    from storage.studio_db import upsert_app_settings

    with _settings_lock:
        if not _accept_write(kind, "custom", write):
            return
        stored = get_media_generation_preset_settings(kind)
        presets = stored.get("customPresets", [])
        stored["customPresets"] = [item for item in presets if item.get("name") != name]
        upsert_app_settings({_setting_key(kind): stored})
