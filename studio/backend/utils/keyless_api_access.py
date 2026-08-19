# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Opt-in that serves the API without an API key.

Off by default. When an admin turns it on, a request that sends no usable
credential authenticates as the local admin, so ``curl`` and the OpenAI SDKs reach
this server the way they reach LM Studio and Ollama.

Two scopes, so opening up chat does not also open up training:

``inference``
    The OpenAI-compatible endpoints only, named one by one in
    ``_INFERENCE_PATHS``. Everything else keeps needing a key.
``full``
    Every route, training and settings included.

Server-side tools (python, terminal, web search) stay off for a keyless caller
whatever the scope, until the admin ticks them on separately: ``/v1/chat/completions``
runs that tool loop on this machine, so it is a bigger grant than chat itself.

Turning it on is the admin's call and nothing here second-guesses the bind
address: ``access_exposure`` only reports how far this server currently reaches
so the UI can say who that choice lets in. Signing in to Unsloth is unaffected.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Optional

KEYLESS_API_ACCESS_SETTING_KEY = "keyless_api_access_scope"
KEYLESS_API_TOOLS_SETTING_KEY = "keyless_api_access_tools"
DEFAULT_KEYLESS_API_TOOLS_ENABLED = False
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


def _coerce_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off", ""}:
            return False
    return None


# each read opens its own sqlite connection (~0.5ms), so hold the answer for a moment
_SETTINGS_CACHE_TTL_S = 1.0
_cached_settings: Optional[tuple[float, str, bool]] = None
# bumped by every write, so a refresh can tell whether its read still describes the db
_settings_generation = 0
_cache_lock = threading.Lock()


def _reset_scope_cache() -> None:
    """Test hook: forget settings cached before the DB was written directly."""
    global _cached_settings
    with _cache_lock:
        _cached_settings = None


def _read_settings() -> tuple[str, bool]:
    try:
        from storage.studio_db import get_app_setting
        scope = _coerce_scope(get_app_setting(KEYLESS_API_ACCESS_SETTING_KEY, None))
        tools = _coerce_bool(get_app_setting(KEYLESS_API_TOOLS_SETTING_KEY, None))
    except Exception:
        return KEYLESS_SCOPE_OFF, False
    return (
        scope or DEFAULT_KEYLESS_API_ACCESS_SCOPE,
        DEFAULT_KEYLESS_API_TOOLS_ENABLED if tools is None else tools,
    )


def _settings() -> tuple[str, bool]:
    """Read the persisted scope and tool grant; anything unreadable counts as off.

    Unlike a normal setting these remove an authentication requirement, so a damaged
    settings DB must never resolve to an open scope, and neither may a refresh that
    read the DB before a write closed it: sqlite reads block, so a request can be
    holding the old answer when the setting is turned off, and publishing it would
    keep the server open for the rest of the TTL. The generation counter dates each
    read against the writes, so only a read that still describes the DB is published.
    """
    global _cached_settings
    now = time.monotonic()
    cached = _cached_settings
    if cached is not None and now - cached[0] < _SETTINGS_CACHE_TTL_S:
        return cached[1], cached[2]
    with _cache_lock:
        generation = _settings_generation
    scope, tools = _read_settings()
    with _cache_lock:
        if generation != _settings_generation:
            published = _cached_settings
            if published is not None:
                return published[1], published[2]
        else:
            _cached_settings = (now, scope, tools)
    return scope, tools


def get_keyless_api_access_scope() -> str:
    return _settings()[0]


def get_keyless_api_tools_enabled() -> bool:
    """Whether a keyless caller may drive the server-side tool loop."""
    return _settings()[1]


def set_keyless_api_access(value: Any, *, tools: Any = None) -> tuple[str, bool]:
    """Persist which routes are served without a key, and whether tools come with them."""
    global _cached_settings, _settings_generation
    scope = _coerce_scope(value)
    if scope is None:
        raise ValueError(f"Keyless API access scope must be one of: {', '.join(KEYLESS_SCOPES)}.")
    allow_tools = get_keyless_api_tools_enabled() if tools is None else _coerce_bool(tools)
    if allow_tools is None:
        raise ValueError("Keyless tool access must be true or false.")
    # tools are meaningless without a scope, and leaving them ticked would surprise
    # whoever turns keyless back on later
    allow_tools = allow_tools and scope != KEYLESS_SCOPE_OFF

    from storage.studio_db import upsert_app_settings

    upsert_app_settings(
        {
            KEYLESS_API_ACCESS_SETTING_KEY: scope,
            KEYLESS_API_TOOLS_SETTING_KEY: allow_tools,
        }
    )
    with _cache_lock:
        _settings_generation += 1
        _cached_settings = (time.monotonic(), scope, allow_tools)
    return scope, allow_tools


def access_exposure(app_state: Any) -> Optional[str]:
    """How far this server reaches beyond the machine, or None for localhost only.

    Advisory: it decides how bluntly the UI words the warning, never whether the
    setting may be used. An unknown bind host is reported as network-reachable.
    """
    from utils.host_policy import (
        is_external_host,
        lan_connector_active,
        tunnel_connector_active,
    )

    if bool(getattr(app_state, "remote_access_is_colab", False)):
        return "colab"
    if bool(getattr(app_state, "secure", False)):
        return "public_url"
    if getattr(app_state, "cloudflare_url", None) or tunnel_connector_active():
        return "public_url"
    bind_host = getattr(app_state, "bind_host", None)
    if not isinstance(bind_host, str) or is_external_host(bind_host):
        return "network"
    # a LAN listener reaches the local network only, so it must not read as a public url
    if lan_connector_active():
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


class KeylessToolPolicyMiddleware:
    """Hard-disable server-side tools for a keyless caller that was not granted them.

    ``/v1/chat/completions`` runs python and terminal on this machine through the
    tool loop, and ``unsloth studio run`` turns tools on by default, so serving that
    route without a key would otherwise hand the loop to anyone who can reach it.
    Mirrors what routes/preview.py does for the public ``/p`` surface.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, asgi_scope, receive, send):
        if asgi_scope.get("type") != "http" or get_keyless_api_tools_enabled():
            await self.app(asgi_scope, receive, send)
            return
        if not asgi_request_is_keyless(asgi_scope):
            await self.app(asgi_scope, receive, send)
            return
        from state.tool_policy import tools_force_disabled

        with tools_force_disabled():
            await self.app(asgi_scope, receive, send)


def asgi_request_is_keyless(asgi_scope) -> bool:
    """Whether this ASGI request is admitted by the setting rather than by a credential.

    Middleware-side twin of ``auth.authentication.admitted_without_credential``, reading
    the raw scope because it runs before the request object exists. A Studio session and
    a working API key both authenticate as themselves, so neither is keyless: applying
    the tool restriction to an existing API client would take away tools it already had.
    """
    access = get_keyless_api_access_scope()
    if access == KEYLESS_SCOPE_OFF:
        return False
    if not scope_covers(access, asgi_scope.get("path") or ""):
        return False
    for name, value in asgi_scope.get("headers") or ():
        if bytes(name).lower() != b"authorization":
            continue
        parts = bytes(value).split(b" ", 1)
        if len(parts) != 2 or parts[0].lower() != b"bearer" or not parts[1].strip():
            return True
        from auth.authentication import bearer_is_valid_api_key, bearer_names_a_session

        token = parts[1].strip().decode("utf-8", "replace")
        return not (bearer_names_a_session(token) or bearer_is_valid_api_key(token))
    return True
