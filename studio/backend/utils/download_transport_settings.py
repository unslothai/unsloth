# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The install's download transport preference: HTTPS, Xet, or Auto.

Stored so the choice follows the install rather than one browser's localStorage. Nothing picked
means Auto, which is what downloads ran on before this setting existed.
"""

from __future__ import annotations

from typing import Any, Optional

from hub.utils.hf_cache_state import TRANSPORT_AUTO, VALID_TRANSPORT_MODES
from loggers import get_logger

logger = get_logger(__name__)

DOWNLOAD_TRANSPORT_SETTING_KEY = "download_transport_mode"
# Unchanged by this setting: an install nobody has touched still lets the backend pick.
DEFAULT_DOWNLOAD_TRANSPORT = TRANSPORT_AUTO


def normalize_transport_mode(value: Any) -> Optional[str]:
    """``value`` as one of http/xet/auto, or None when it is not a transport mode."""
    if not isinstance(value, str):
        return None
    mode = value.strip().lower()
    return mode if mode in VALID_TRANSPORT_MODES else None


def get_download_transport_mode() -> str:
    """The install's stored transport, or Auto when nobody has picked one."""
    try:
        from storage.studio_db import get_app_setting
        stored = normalize_transport_mode(get_app_setting(DOWNLOAD_TRANSPORT_SETTING_KEY, None))
    except Exception as exc:  # noqa: BLE001 - an unreadable db reports the default, not a 500
        logger.debug("download transport read failed (%s)", exc)
        stored = None
    return DEFAULT_DOWNLOAD_TRANSPORT if stored is None else stored


def set_download_transport_mode(value: Any) -> str:
    """Persist the install's download transport. Raises ValueError on anything else."""
    mode = normalize_transport_mode(value)
    if mode is None:
        raise ValueError("Download transport must be one of: http, xet, auto.")

    from storage.studio_db import upsert_app_settings

    upsert_app_settings({DOWNLOAD_TRANSPORT_SETTING_KEY: mode})
    return mode
