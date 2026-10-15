# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Persisted kill switch for public ``/p`` preview link sharing."""

from __future__ import annotations

from typing import Any

PREVIEW_SHARING_SETTING_KEY = "preview_public_sharing_enabled"
# Off takes the public /p surface offline: links 404 even with a valid token.
DEFAULT_PREVIEW_SHARING_ENABLED = True


def _coerce_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off", ""}:
            return False
    return None


def get_preview_sharing_enabled() -> bool:
    """Missing setting means enabled; a read failure fails closed, since this is a kill switch."""
    try:
        from storage.studio_db import get_app_setting
        from utils.account_context import OWNER, run_as
        stored = run_as(OWNER, get_app_setting, PREVIEW_SHARING_SETTING_KEY, None)
    except Exception:
        return False
    parsed = _coerce_bool(stored)
    return parsed if parsed is not None else DEFAULT_PREVIEW_SHARING_ENABLED


def set_preview_sharing_enabled(value: Any) -> bool:
    """Persist whether public ``/p`` preview links are accepted."""
    parsed = _coerce_bool(value)
    if parsed is None:
        raise ValueError("Public preview sharing must be true or false.")

    from storage.studio_db import upsert_app_settings

    upsert_app_settings({PREVIEW_SHARING_SETTING_KEY: parsed})
    return parsed
