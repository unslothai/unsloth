# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Voice Settings "Load into" for speech-to-text, persisted for API loads (#15280).

The UI keeps ``sttDevice`` in browser localStorage, which ``/v1/audio/transcriptions`` never
sees. Mirrored into ``app_settings`` (as ``rag_embedding_device`` is), ``stt_device_default``
reads it when ``UNSLOTH_AUDIO_DEVICE`` is unset.
"""

from __future__ import annotations

import threading
import time
from typing import Optional

STT_DEVICE_SETTING_KEY = "stt_device"
STT_DEVICE_CHOICES = ("auto", "cpu", "gpu")

_CACHE_TTL_S = 2.0
_device_cached: list = []  # [(monotonic time, value)]
_device_generation = 0
_lock = threading.Lock()


def _stored_stt_device() -> Optional[str]:
    """The saved choice, or None when nothing valid is saved (cached briefly)."""
    now = time.monotonic()
    with _lock:
        if _device_cached and now - _device_cached[0][0] < _CACHE_TTL_S:
            return _device_cached[0][1]
        gen = _device_generation
    from utils.account_context import OWNER, bind_account, reset_account

    token = bind_account(OWNER)
    try:
        from storage.studio_db import get_app_setting
        value = get_app_setting(STT_DEVICE_SETTING_KEY, None)
    except Exception:  # noqa: BLE001 - store unavailable: hardware default
        value = None
    finally:
        reset_account(token)
    device = value if value in STT_DEVICE_CHOICES else None
    with _lock:
        if _device_generation == gen:
            _device_cached[:] = [(now, device)]
    return device


def get_stored_stt_device() -> str:
    """Voice Settings placement: ``auto`` when unset, unreadable or unrecognized."""
    return _stored_stt_device() or "auto"


def stt_device_is_saved() -> bool:
    """Whether a choice was ever saved, so a browser knows to adopt it rather than migrate its own."""
    return _stored_stt_device() is not None


def forget_cached_stt_device() -> None:
    global _device_generation
    with _lock:
        _device_cached.clear()
        _device_generation += 1


def set_stt_device(device: str) -> bool:
    """Persist placement; returns whether it changed."""
    if device not in STT_DEVICE_CHOICES:
        raise ValueError("Device must be auto, cpu or gpu.")
    forget_cached_stt_device()
    if _stored_stt_device() == device:
        return False
    from storage.studio_db import upsert_app_settings
    from utils.account_context import OWNER, bind_account, reset_account

    token = bind_account(OWNER)
    try:
        upsert_app_settings({STT_DEVICE_SETTING_KEY: device}, read_back = False)
    finally:
        reset_account(token)
        forget_cached_stt_device()
    return True
