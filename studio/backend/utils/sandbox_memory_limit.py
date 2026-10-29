# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Settings > Sandbox memory limit (RLIMIT_AS) for sandboxed Python and Terminal runs; a present env var always wins."""

from __future__ import annotations

import os
from typing import Any

MEMORY_LIMIT_ENV = "UNSLOTH_STUDIO_SANDBOX_AS_GB"
MEMORY_LIMIT_SETTING_KEY = "sandbox_memory_limit_gb"
DEFAULT_MEMORY_LIMIT_GB = 8
MIN_MEMORY_LIMIT_GB = 1
MAX_MEMORY_LIMIT_GB = 4096
_BYTES_PER_GB = 1024 * 1024 * 1024


def _coerce_gb(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    if parsed < MIN_MEMORY_LIMIT_GB or parsed > MAX_MEMORY_LIMIT_GB:
        return None
    return parsed


def locked_by_environment() -> bool:
    return MEMORY_LIMIT_ENV in os.environ


def saved_memory_limit_gb() -> int:
    try:
        from storage.studio_db import get_app_setting
        from utils.account_context import OWNER, run_as

        # Installation-wide: the owner's store, whichever account's tool call is asking.
        stored = run_as(OWNER, get_app_setting, MEMORY_LIMIT_SETTING_KEY, None)
    except Exception:  # noqa: BLE001 - an unreadable store keeps the shipped default
        stored = None
    return _coerce_gb(stored) or DEFAULT_MEMORY_LIMIT_GB


def effective_memory_limit_gb() -> int | None:
    """GiB the next sandboxed run is capped at; None = no cap (an env value int() rejects, as before)."""
    if locked_by_environment():
        try:
            return int(os.environ[MEMORY_LIMIT_ENV])
        except ValueError:
            return None
    return saved_memory_limit_gb()


def memory_limit_bytes() -> int | None:
    gb = effective_memory_limit_gb()
    return None if gb is None else gb * _BYTES_PER_GB


def set_memory_limit_gb(value: Any) -> int:
    parsed = _coerce_gb(value)
    if parsed is None:
        raise ValueError(
            f"Memory limit must be a whole number from {MIN_MEMORY_LIMIT_GB} to {MAX_MEMORY_LIMIT_GB} GB."
        )
    from storage.studio_db import upsert_app_settings
    from utils.account_context import OWNER, run_as

    run_as(OWNER, upsert_app_settings, {MEMORY_LIMIT_SETTING_KEY: parsed})
    return parsed
