# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
from __future__ import annotations

import hashlib
import os
import tempfile
import time
from pathlib import Path
from typing import Optional

from filelock import FileLock

CLIENT_PREFERENCES_KEY = "client_preferences"
CHAT_BACKGROUND_KEY = "chat_background"
MAX_BACKGROUND_IMAGE_BYTES = 20 * 1024 * 1024
BACKGROUND_IMAGE_TYPES = ("image/png", "image/jpeg", "image/webp")


def _now_ms() -> int:
    return int(time.time() * 1000)


def _read(key: str) -> dict:
    from storage.studio_db import get_app_setting
    stored = get_app_setting(key, None)
    return stored if isinstance(stored, dict) else {}


def _write(key: str, value: dict) -> dict:
    from storage.studio_db import upsert_app_settings
    upsert_app_settings({key: value}, read_back = False)
    return value


def get_client_preferences() -> dict:
    return _read(CLIENT_PREFERENCES_KEY)


def _merge_sidebar(stored: dict, base: dict, edited: dict) -> dict:
    """Apply only this browser's edits to the latest account snapshot."""
    merged = dict(stored)
    for key, value in edited.items():
        before = base.get(key)
        if value == before:
            continue
        current = stored.get(key)
        if key == "customSections":
            old = {s["id"]: s for s in before or []}
            new = {s["id"]: s for s in value}
            live = {s["id"]: s for s in current or []}
            for removed in old.keys() - new.keys():
                live.pop(removed, None)
            for section_id, section in new.items():
                if section_id not in old:
                    live[section_id] = section
                elif section_id in live and section != old[section_id]:
                    previous = old[section_id]
                    updated = dict(live[section_id])
                    for removed in previous.keys() - section.keys():
                        updated.pop(removed, None)
                    updated.update({k: v for k, v in section.items() if previous.get(k) != v})
                    live[section_id] = updated
            merged[key] = list(live.values())
        elif isinstance(value, dict):
            mapping = dict(current or {})
            previous = before or {}
            for removed in previous.keys() - value.keys():
                mapping.pop(removed, None)
            mapping.update({k: v for k, v in value.items() if previous.get(k) != v})
            merged[key] = mapping
        elif isinstance(value, list):
            # Preserve remote additions, but do not resurrect remotely removed identifiers.
            prior, live = set(before or []), set(current or [])
            retained = [item for item in value if item not in prior or item in live]
            known = set(retained)
            merged[key] = retained + [item for item in current or [] if item not in prior and item not in known]
        else:
            merged[key] = value

    # A different browser may file a chat or create a child while its parent is
    # being deleted. Never leave that assignment pointing at a missing section.
    sections = merged.get("customSections", [])
    live_ids = {section["id"] for section in sections}
    if sections:
        merged["customSections"] = [
            {k: v for k, v in section.items() if k != "parentId"}
            if "parentId" in section and section["parentId"] not in live_ids else section
            for section in sections
        ]
    for field in ("sectionByChatId", "sectionByProjectId", "sectionByPageId"):
        if field in merged:
            merged[field] = {k: v for k, v in merged[field].items() if v in live_ids}
    return merged


def update_client_preferences(changes: dict) -> dict:
    from storage.studio_db import compare_and_set_app_setting, get_app_setting
    changes = dict(changes)
    base = changes.pop("sidebarBase", {})
    while True:
        previous = get_app_setting(CLIENT_PREFERENCES_KEY, None)
        stored = previous if isinstance(previous, dict) else {}
        merged = {
            **stored,
            **{k: v for k, v in changes.items() if v is not None},
            "updatedAt": _now_ms(),
        }
        if "sidebar" in changes:
            merged["sidebar"] = _merge_sidebar(stored.get("sidebar", {}), base, changes["sidebar"])
        if compare_and_set_app_setting(CLIENT_PREFERENCES_KEY, previous, merged):
            return merged


def get_chat_background() -> dict:
    return _read(CHAT_BACKGROUND_KEY)


def background_image_path() -> Path:
    from utils.paths.storage_roots import account_path
    return account_path("appearance/chat-background")


def background_lock() -> FileLock:
    path = background_image_path()
    path.parent.mkdir(parents = True, exist_ok = True)
    return FileLock(str(path.with_name(".chat-background.lock")))


def sniff_image_type(data: bytes) -> Optional[str]:
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return None


def save_background_image(data: bytes) -> dict:
    if not data or len(data) > MAX_BACKGROUND_IMAGE_BYTES:
        raise ValueError("Background image is empty or too large.")
    media_type = sniff_image_type(data)
    if media_type is None:
        raise ValueError("Background image must be PNG, JPEG or WebP.")
    with background_lock():
        path = background_image_path()
        fd, tmp = tempfile.mkstemp(dir = path.parent, prefix = ".chat-background-")
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(data)
            os.replace(tmp, path)
        except BaseException:
            Path(tmp).unlink(missing_ok = True)
            raise
        image = {"sha256": hashlib.sha256(data).hexdigest(), "type": media_type, "size": len(data)}
        stored = get_chat_background()
        return _write(
            CHAT_BACKGROUND_KEY,
            {"adjustments": stored.get("adjustments"), "image": image, "updatedAt": _now_ms()},
        )


def save_background_adjustments(adjustments: dict) -> dict:
    with background_lock():
        stored = get_chat_background()
        if not stored.get("image"):
            raise LookupError("No background image is saved.")
        return _write(
            CHAT_BACKGROUND_KEY, {**stored, "adjustments": adjustments, "updatedAt": _now_ms()}
        )


def clear_background() -> dict:
    with background_lock():
        background_image_path().unlink(missing_ok = True)
        return _write(
            CHAT_BACKGROUND_KEY, {"adjustments": None, "image": None, "updatedAt": _now_ms()}
        )
