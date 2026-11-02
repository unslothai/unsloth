# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Per-account copies of settings the browser used to keep only in its own storage."""

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


def update_client_preferences(changes: dict) -> dict:
    from storage.studio_db import compare_and_set_app_setting, get_app_setting
    while True:
        previous = get_app_setting(CLIENT_PREFERENCES_KEY, None)
        stored = previous if isinstance(previous, dict) else {}
        merged = {
            **stored,
            **{k: v for k, v in changes.items() if v is not None},
            "updatedAt": _now_ms(),
        }
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
