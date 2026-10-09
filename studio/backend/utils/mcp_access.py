# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Installation-wide switch for agent access to Studio over MCP (``/mcp``). Off by default; the owner turns it on from Settings, and ``UNSLOTH_STUDIO_ENABLE_MCP=1`` forces it on. The value is read as the owner, so a managed account's API key sees the owner's switch and not its own settings database, and an unreadable setting counts as off."""

from __future__ import annotations

import os
import threading
import time
from typing import Any, Optional

from utils.account_context import OWNER, is_owner_context, run_as

MCP_ENABLED_SETTING_KEY = "studio_mcp_enabled"
ENV_FORCE = "UNSLOTH_STUDIO_ENABLE_MCP"

# The gate asks on every /mcp request; each read opens its own sqlite connection.
_CACHE_TTL_S = 1.0
_cached: Optional[tuple[float, bool]] = None
# Bumped by every write, so a read that started before it is never published.
_generation = 0
_cache_lock = threading.Lock()


def _reset_cache() -> None:
    """Test hook: forget a value cached before the DB was written directly."""
    global _cached
    with _cache_lock:
        _cached = None


def forced_by_env() -> bool:
    return os.environ.get(ENV_FORCE) == "1"


def _read_from_db() -> bool:
    from storage.studio_db import get_app_setting
    return run_as(OWNER, get_app_setting, MCP_ENABLED_SETTING_KEY, None) is True


def get_mcp_enabled() -> bool:
    """The owner's stored switch, ignoring the environment. Fails closed."""
    global _cached
    with _cache_lock:
        cached = _cached
        if cached is not None and time.monotonic() - cached[0] < _CACHE_TTL_S:
            return cached[1]
        generation = _generation
    try:
        enabled = _read_from_db()
    except Exception:
        return False
    with _cache_lock:
        if generation == _generation:
            _cached = (time.monotonic(), enabled)
        else:
            enabled = False
    return enabled


def is_mcp_enabled() -> bool:
    return forced_by_env() or get_mcp_enabled()


def set_mcp_enabled(enabled: Any) -> bool:
    global _cached, _generation
    if not is_owner_context():
        raise ValueError("Only the installation owner can change agent access over MCP.")
    if not isinstance(enabled, bool):
        raise ValueError("Agent access over MCP must be true or false.")
    from storage.studio_db import upsert_app_settings

    try:
        upsert_app_settings({MCP_ENABLED_SETTING_KEY: enabled}, read_back = False)
    finally:
        # Invalidate even on failure: the write may have committed.
        with _cache_lock:
            _generation += 1
            _cached = None
    return enabled
