# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Installation-wide API monitor recording preference."""

from __future__ import annotations

import logging
import threading

from core.inference.api_monitor import _api_monitor_disabled, api_monitor
from utils.account_context import OWNER, run_as

_SETTING_KEY = "api_monitor_recording_enabled"
_setting_lock = threading.Lock()
logger = logging.getLogger(__name__)


def configured_api_monitor_enabled() -> bool:
    """Read the owner's preference; a database error fails closed for privacy."""
    try:
        from storage.studio_db import get_app_setting
        value = run_as(OWNER, get_app_setting, _SETTING_KEY, True)
    except Exception:
        logger.warning("api_monitor.settings_read_failed", exc_info = True)
        return False
    return value if isinstance(value, bool) else False


def apply_api_monitor_setting() -> None:
    """Apply the saved preference at startup, honoring the environment kill switch."""
    with _setting_lock:
        api_monitor.set_enabled(configured_api_monitor_enabled() and not _api_monitor_disabled())


def set_api_monitor_enabled(enabled: bool) -> None:
    """Persist the owner preference and update admission for new monitored requests."""
    from storage.studio_db import upsert_app_settings
    with _setting_lock:
        run_as(OWNER, upsert_app_settings, {_SETTING_KEY: enabled})
        api_monitor.set_enabled(enabled and not _api_monitor_disabled())
