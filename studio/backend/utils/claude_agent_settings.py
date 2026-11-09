# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Claude Agent opt-in (off by default); a boolean UNSLOTH_CLAUDE_AGENT overrides the saved setting."""

from __future__ import annotations

import os
from typing import Any, Optional

CLAUDE_AGENT_SETTING_KEY = "developer_mode_claude_agent"
CLAUDE_AGENT_ENV = "UNSLOTH_CLAUDE_AGENT"
DEFAULT_CLAUDE_AGENT_ENABLED = False

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


def claude_agent_env_override() -> Optional[bool]:
    raw = os.environ.get(CLAUDE_AGENT_ENV)
    if raw is None or not raw.strip():
        return None
    normalized = raw.strip().lower()
    if normalized in _TRUE:
        return True
    if normalized in _FALSE:
        return False
    return None


def get_claude_agent_setting() -> bool:
    # Installation wide, so always read from the owner's store.
    try:
        from storage.studio_db import get_app_setting
        from utils.account_context import OWNER, run_as
        stored = run_as(OWNER, get_app_setting, CLAUDE_AGENT_SETTING_KEY, None)
    except Exception:
        stored = None
    parsed = _coerce_bool(stored)
    return parsed if parsed is not None else DEFAULT_CLAUDE_AGENT_ENABLED


def set_claude_agent_setting(value: Any) -> bool:
    parsed = _coerce_bool(value)
    if parsed is None:
        raise ValueError("Developer mode Claude Agent must be true or false.")
    from storage.studio_db import upsert_app_settings

    upsert_app_settings({CLAUDE_AGENT_SETTING_KEY: parsed})
    return parsed


def claude_agent_setting_enabled() -> bool:
    forced = claude_agent_env_override()
    if forced is not None:
        return forced
    return get_claude_agent_setting()
