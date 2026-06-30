# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Persisted opt-in config for training-completion webhook notifications."""

from __future__ import annotations

from typing import Any
from urllib.parse import urlparse

TRAINING_WEBHOOK_SETTING_KEY = "training_webhook_notification"

DEFAULT_TRAINING_WEBHOOK: dict[str, Any] = {
    "enabled": False,
    "url": "",
}


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return False


def _validate_url(value: Any) -> str:
    text = value.strip() if isinstance(value, str) else ""
    if not text:
        return ""
    parsed = urlparse(text)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("Webhook URL must be a valid http(s) URL.")
    return text


def get_training_webhook() -> dict[str, Any]:
    try:
        from storage.studio_db import get_app_setting
        stored = get_app_setting(TRAINING_WEBHOOK_SETTING_KEY, None)
    except Exception:
        stored = None

    if not isinstance(stored, dict):
        return dict(DEFAULT_TRAINING_WEBHOOK)

    url = stored.get("url")
    return {
        "enabled": _coerce_bool(stored.get("enabled")),
        "url": url if isinstance(url, str) else "",
    }


def set_training_webhook(enabled: Any, url: Any) -> dict[str, Any]:
    enabled_bool = _coerce_bool(enabled)
    # Validate only when enabling; disabling must succeed even with a partial draft.
    url_text = _validate_url(url) if enabled_bool else (url.strip() if isinstance(url, str) else "")
    if enabled_bool and not url_text:
        raise ValueError("A webhook URL is required to enable training notifications.")
    config = {"enabled": enabled_bool, "url": url_text}

    from storage.studio_db import upsert_app_settings

    upsert_app_settings({TRAINING_WEBHOOK_SETTING_KEY: config})
    return config
