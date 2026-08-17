# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Opt-in that serves the API without an API key.

Off by default. When an admin turns it on, a request that sends no usable
credential authenticates as the local admin, so ``curl``, the OpenAI SDKs and
``unsloth start`` work against this server the way LM Studio and Ollama do.

Two scopes, so opening up chat does not also open up training:

``inference``
    The OpenAI-compatible endpoints only, named one by one in
    ``_INFERENCE_PATHS``. Everything else keeps needing a key.
``full``
    Every route, training and settings included.

Turning it on is the admin's call and nothing here second-guesses the bind
address: ``access_exposure`` only reports how far this server currently reaches
so the UI can say who that choice lets in. Signing in to Unsloth is unaffected.
"""

from __future__ import annotations

import time
from typing import Any, Optional

KEYLESS_API_ACCESS_SETTING_KEY = "keyless_api_access_scope"
KEYLESS_SCOPE_OFF = "off"
KEYLESS_SCOPE_INFERENCE = "inference"
KEYLESS_SCOPE_FULL = "full"
KEYLESS_SCOPES = (KEYLESS_SCOPE_OFF, KEYLESS_SCOPE_INFERENCE, KEYLESS_SCOPE_FULL)
DEFAULT_KEYLESS_API_ACCESS_SCOPE = KEYLESS_SCOPE_OFF

# named one by one, not by prefix: /v1 also aliases model loading, audio and sandbox routes
_INFERENCE_PATHS = frozenset(
    {
        "/v1/chat/completions",
        "/v1/chat/count_tokens",
        "/v1/completions",
        "/v1/embeddings",
        "/v1/messages",
        "/v1/messages/count_tokens",
        "/v1/models",
        "/v1/responses",
    }
)
# GET /v1/models/<id>, and nothing else nested
_INFERENCE_PREFIXES = ("/v1/models/",)


def _coerce_scope(value: Any) -> Optional[str]:
    if isinstance(value, str) and value.strip().lower() in KEYLESS_SCOPES:
        return value.strip().lower()
    return None


# each read opens its own sqlite connection (~0.5ms), so hold the answer for a moment
_SCOPE_CACHE_TTL_S = 1.0
_cached_scope: Optional[tuple[float, str]] = None


def _reset_scope_cache() -> None:
    """Test hook: forget a scope cached before the settings DB was written directly."""
    global _cached_scope
    _cached_scope = None


def _read_scope() -> str:
    try:
        from storage.studio_db import get_app_setting
        stored = get_app_setting(KEYLESS_API_ACCESS_SETTING_KEY, None)
    except Exception:
        return KEYLESS_SCOPE_OFF
    return _coerce_scope(stored) or DEFAULT_KEYLESS_API_ACCESS_SCOPE


def get_keyless_api_access_scope() -> str:
    """Read the persisted scope; anything unreadable or unknown counts as off.

    Unlike a normal setting this one removes an authentication requirement, so a
    damaged settings DB must never resolve to an open scope.
    """
    global _cached_scope
    now = time.monotonic()
    cached = _cached_scope
    if cached is not None and now - cached[0] < _SCOPE_CACHE_TTL_S:
        return cached[1]
    scope = _read_scope()
    _cached_scope = (now, scope)
    return scope


def set_keyless_api_access_scope(value: Any) -> str:
    """Persist which routes are served without a key."""
    global _cached_scope
    scope = _coerce_scope(value)
    if scope is None:
        raise ValueError(f"Keyless API access scope must be one of: {', '.join(KEYLESS_SCOPES)}.")

    from storage.studio_db import upsert_app_settings

    upsert_app_settings({KEYLESS_API_ACCESS_SETTING_KEY: scope})
    _cached_scope = (time.monotonic(), scope)
    return scope


def access_exposure(app_state: Any) -> Optional[str]:
    """How far this server reaches beyond the machine, or None for localhost only.

    Advisory: it decides how bluntly the UI words the warning, never whether the
    setting may be used. An unknown bind host is reported as network-reachable.
    """
    from utils.host_policy import is_external_host, remote_connector_active

    if bool(getattr(app_state, "remote_access_is_colab", False)):
        return "colab"
    if bool(getattr(app_state, "secure", False)):
        return "public_url"
    if getattr(app_state, "cloudflare_url", None) or remote_connector_active():
        return "public_url"
    bind_host = getattr(app_state, "bind_host", None)
    if not isinstance(bind_host, str) or is_external_host(bind_host):
        return "network"
    return None


def scope_covers(scope: str, path: str) -> bool:
    """Whether ``scope`` serves ``path`` without a key."""
    if scope == KEYLESS_SCOPE_FULL:
        return True
    if scope != KEYLESS_SCOPE_INFERENCE:
        return False
    normalized = path.rstrip("/") or path
    return normalized in _INFERENCE_PATHS or path.startswith(_INFERENCE_PREFIXES)


def keyless_request_allowed(request: Any) -> bool:
    """Whether this request may authenticate without a usable credential."""
    scope = get_keyless_api_access_scope()
    if scope == KEYLESS_SCOPE_OFF:
        return False
    path = getattr(getattr(request, "url", None), "path", None)
    return scope_covers(scope, path if isinstance(path, str) else "")
