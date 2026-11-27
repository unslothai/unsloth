# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Sandbox file size limit (RLIMIT_FSIZE, macOS and Linux): a present env var wins, then the saved value, then 4 GB."""

from __future__ import annotations

import os
from typing import Any

FILE_SIZE_LIMIT_ENV = "UNSLOTH_STUDIO_SANDBOX_FSIZE_MB"
FILE_SIZE_LIMIT_SETTING_KEY = "sandbox_file_size_limit_mb"
# 100 MB until #19853: debug builds (swift build, cargo, C++) link binaries past it.
DEFAULT_FILE_SIZE_LIMIT_MB = 4096
MIN_FILE_SIZE_LIMIT_MB = 1
MAX_FILE_SIZE_LIMIT_MB = 1024 * 1024
_BYTES_PER_MB = 1024 * 1024


def _coerce_mb(value: Any) -> int | None:
    if isinstance(value, bool) or (isinstance(value, float) and not value.is_integer()):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    if parsed < MIN_FILE_SIZE_LIMIT_MB or parsed > MAX_FILE_SIZE_LIMIT_MB:
        return None
    return parsed


def locked_by_environment() -> bool:
    return FILE_SIZE_LIMIT_ENV in os.environ


def _stored_file_size_limit_mb() -> int | None:
    try:
        from storage.studio_db import get_app_setting
        from utils.account_context import OWNER, run_as

        # Installation-wide: the owner's store, whichever account's tool call is asking.
        stored = run_as(OWNER, get_app_setting, FILE_SIZE_LIMIT_SETTING_KEY, None)
    except Exception:  # noqa: BLE001 - an unreadable store keeps the shipped default
        stored = None
    return _coerce_mb(stored)


def saved_file_size_limit_mb() -> int:
    return _stored_file_size_limit_mb() or DEFAULT_FILE_SIZE_LIMIT_MB


def effective_file_size_limit_mb() -> int:
    """MB one file a sandboxed run writes can reach. An env value outside the range keeps the default."""
    if locked_by_environment():
        return _coerce_mb(os.environ[FILE_SIZE_LIMIT_ENV]) or DEFAULT_FILE_SIZE_LIMIT_MB
    return saved_file_size_limit_mb()


def file_size_limit_bytes() -> int:
    return effective_file_size_limit_mb() * _BYTES_PER_MB


def set_file_size_limit_mb(value: Any) -> int:
    parsed = _coerce_mb(value)
    if parsed is None:
        raise ValueError(
            f"File size limit must be a whole number from {MIN_FILE_SIZE_LIMIT_MB} to {MAX_FILE_SIZE_LIMIT_MB} MB."
        )
    from storage.studio_db import upsert_app_settings
    from utils.account_context import OWNER, run_as

    run_as(OWNER, upsert_app_settings, {FILE_SIZE_LIMIT_SETTING_KEY: parsed})
    return parsed
