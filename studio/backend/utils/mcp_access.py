# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Installation-wide switch for agent access to Unsloth Studio over MCP (``/mcp``), and the tools the owner turned off. Off by default; the owner turns it on from Settings, and ``UNSLOTH_STUDIO_ENABLE_MCP=1`` forces it on. Both values are read as the owner, so a managed account's API key sees the owner's choices and not its own settings database. An unreadable switch counts as off and an unreadable tool list as every tool off."""

from __future__ import annotations

import os
import threading
import time
from typing import Any, Iterable, Optional

from loggers import get_logger
from utils.account_context import OWNER, is_owner_context, run_as

logger = get_logger(__name__)

MCP_ENABLED_SETTING_KEY = "studio_mcp_enabled"
MCP_DISABLED_TOOLS_SETTING_KEY = "studio_mcp_disabled_tools"
ENV_FORCE = "UNSLOTH_STUDIO_ENABLE_MCP"

# (enabled, disabled tool names); None for the names means the stored value is unreadable.
_Record = tuple[bool, Optional[frozenset[str]]]

# The gate asks on every /mcp request; each read opens its own sqlite connection.
_CACHE_TTL_S = 1.0
_RACE_RETRIES = 3
_cached: Optional[tuple[float, _Record]] = None
# Bumped by every write, so a read that started before it is never published.
_generation = 0
_cache_lock = threading.Lock()


class UnknownMcpToolError(ValueError):
    pass


def _reset_cache() -> None:
    """Test hook: forget a value cached before the DB was written directly."""
    global _cached
    with _cache_lock:
        _cached = None


def forced_by_env() -> bool:
    return os.environ.get(ENV_FORCE) == "1"


def _parse_disabled(value: Any) -> Optional[frozenset[str]]:
    if not isinstance(value, list):
        logger.warning(
            f"Unreadable {MCP_DISABLED_TOOLS_SETTING_KEY} ({type(value).__name__}); every MCP tool counts as off."
        )
        return None
    return frozenset(name for name in value if isinstance(name, str))


def _read_from_db() -> _Record:
    from storage.studio_db import get_app_settings
    values = run_as(
        OWNER, get_app_settings, [MCP_ENABLED_SETTING_KEY, MCP_DISABLED_TOOLS_SETTING_KEY]
    )
    return (
        values.get(MCP_ENABLED_SETTING_KEY) is True,
        _parse_disabled(values.get(MCP_DISABLED_TOOLS_SETTING_KEY, [])),
    )


def _record() -> Optional[_Record]:
    """The cached owner record, or None when the read failed or kept racing writes."""
    global _cached
    # A read that overlaps a write is retried rather than reported as off: the owner
    # saving one switch must not hide every tool from agents for that moment.
    for _ in range(_RACE_RETRIES):
        with _cache_lock:
            cached = _cached
            if cached is not None and time.monotonic() - cached[0] < _CACHE_TTL_S:
                return cached[1]
            generation = _generation
        try:
            record = _read_from_db()
        except Exception:
            return None
        with _cache_lock:
            if generation == _generation:
                _cached = (time.monotonic(), record)
                return record
    return None


def owner_record() -> Optional[tuple[bool, Optional[frozenset[str]]]]:
    """One snapshot of (stored switch, turned-off tools), or None when it could not be read."""
    return _record()


def get_mcp_enabled() -> bool:
    """The owner's stored switch, ignoring the environment. Fails closed."""
    record = _record()
    return record is not None and record[0]


def is_mcp_enabled() -> bool:
    return forced_by_env() or get_mcp_enabled()


def disabled_tools() -> Optional[frozenset[str]]:
    """Names the owner turned off, or None when every tool counts as off."""
    record = _record()
    return None if record is None else record[1]


def tool_enabled(name: str) -> bool:
    disabled = disabled_tools()
    return disabled is not None and name not in disabled


def set_mcp_settings(
    *, enabled: Optional[bool] = None, disabled_tools: Optional[Iterable[str]] = None
) -> None:
    """Write the switch, the turned-off tools, or both in one upsert."""
    global _cached, _generation
    if not is_owner_context():
        raise ValueError("Only the installation owner can change agent access over MCP.")
    settings: dict[str, Any] = {}
    if enabled is not None:
        if not isinstance(enabled, bool):
            raise ValueError("Agent access over MCP must be true or false.")
        settings[MCP_ENABLED_SETTING_KEY] = enabled
    if disabled_tools is not None:
        if isinstance(disabled_tools, (str, bytes)):
            raise ValueError("Turned-off MCP tools must be a list of tool names.")
        names = list(disabled_tools)
        if not all(isinstance(name, str) for name in names):
            raise ValueError("Turned-off MCP tools must be a list of tool names.")
        from mcp_server import tool_names

        known = set(tool_names())
        for name in names:
            if name not in known:
                raise UnknownMcpToolError(f"Unknown Unsloth Studio MCP tool: {name}")
        settings[MCP_DISABLED_TOOLS_SETTING_KEY] = sorted(set(names))
    if not settings:
        raise ValueError("Nothing to change.")
    from storage.studio_db import upsert_app_settings

    try:
        upsert_app_settings(settings, read_back = False)
    finally:
        # Invalidate even on failure: the write may have committed.
        with _cache_lock:
            _generation += 1
            _cached = None


def set_mcp_enabled(enabled: Any) -> bool:
    if enabled is None:
        raise ValueError("Agent access over MCP must be true or false.")
    set_mcp_settings(enabled = enabled)
    return enabled
