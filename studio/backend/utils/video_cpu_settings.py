# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Developer opt-in (off by default): video runs on a CPU-only host, experimental, only when
no_accelerator is the sole blocker; other refusals stay. UNSLOTH_VIDEO_ALLOW_CPU, if a recognised
boolean, overrides the saved setting."""

from __future__ import annotations

import os
from typing import Any, Optional

VIDEO_CPU_SETTING_KEY = "developer_mode_video_cpu"
VIDEO_CPU_ENV = "UNSLOTH_VIDEO_ALLOW_CPU"
DEFAULT_VIDEO_CPU_ENABLED = False

# Host RAM below which the Settings dialog warns before enabling; the smallest video family,
# Wan2.1-T2V-1.3B, peaked at 24.5 GiB server RSS on CPU (float32, 832x480, 5 frames), so 32 GiB
# leaves room for the OS.
VIDEO_CPU_MIN_RAM_GB = 32

_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off"}


def _coerce_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in _TRUE:
            return True
        if normalized in _FALSE or normalized == "":
            return False
    return None


def video_cpu_env_override() -> Optional[bool]:
    """The environment's answer, or None when ``UNSLOTH_VIDEO_ALLOW_CPU`` is unset or unrecognised."""
    raw = os.environ.get(VIDEO_CPU_ENV)
    if raw is None or not raw.strip():
        return None
    normalized = raw.strip().lower()
    if normalized in _TRUE:
        return True
    if normalized in _FALSE:
        return False
    return None


def get_video_cpu_setting() -> bool:
    """The persisted owner setting. Read from the owner's store whoever asks, since it is installation
    wide; a missing or unreadable setting is the default (off)."""
    try:
        from storage.studio_db import get_app_setting
        from utils.account_context import OWNER, run_as
        stored = run_as(OWNER, get_app_setting, VIDEO_CPU_SETTING_KEY, None)
    except Exception:
        stored = None
    parsed = _coerce_bool(stored)
    return parsed if parsed is not None else DEFAULT_VIDEO_CPU_ENABLED


def set_video_cpu_setting(value: Any) -> bool:
    """Persist the owner setting. Callers are the owner-only settings route."""
    parsed = _coerce_bool(value)
    if parsed is None:
        raise ValueError("Developer mode CPU video must be true or false.")
    from storage.studio_db import upsert_app_settings

    upsert_app_settings({VIDEO_CPU_SETTING_KEY: parsed})
    return parsed


def video_cpu_enabled() -> bool:
    """Whether CPU video is opted in: the env override when set, else the persisted setting. Never raises."""
    forced = video_cpu_env_override()
    if forced is not None:
        return forced
    return get_video_cpu_setting()


def host_ram_bytes() -> Optional[int]:
    """Total physical RAM, or None when it cannot be read. Never raises."""
    try:
        import psutil
        return int(psutil.virtual_memory().total)
    except Exception:  # noqa: BLE001 -- fall back to sysconf
        pass
    try:
        return int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
    except Exception:  # noqa: BLE001
        return None
