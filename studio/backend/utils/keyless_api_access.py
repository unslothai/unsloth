# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Fail-closed admission policy for serving Unsloth without an API key.

Off by default. When an admin turns it on, a request that sends no usable
credential authenticates as the local admin, so ``curl`` and the OpenAI SDKs reach
this server the way they reach LM Studio and Ollama.

Two scopes, so opening up chat does not also open up training:

``inference``
    The OpenAI-compatible endpoints only, named one by one in
    ``_INFERENCE_ROUTES``. Everything else keeps needing a key.
``full``
    Every route, but only for callers arriving and connecting over loopback.

Server-side tools (python, terminal, web search) stay off for a keyless caller
whatever the scope, until the admin ticks them on separately: ``/v1/chat/completions``
runs that tool loop on this machine, so it is a bigger grant than chat itself.

Public tunnels and Colab never receive keyless access. Private-LAN inference is
accepted only through a live settings listener or the launch-managed bind that
matches the ASGI accepting address and port. Signing in to Unsloth is unaffected.
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
APPROVED_DUMMY_BEARERS = frozenset({"not-needed", "lm-studio", "ollama"})
KEYLESS_ADMISSION_STATE_KEY = "keyless_api_admitted"

# Named by method and normalized path: /v1 also aliases model loading, media,
# sandbox, validation, and streaming side-effect routes.
_INFERENCE_ROUTES = frozenset(
    {
        ("POST", "/v1/chat/completions"),
        ("POST", "/v1/chat/count_tokens"),
        ("POST", "/v1/completions"),
        ("POST", "/v1/embeddings"),
        ("POST", "/v1/messages"),
        ("POST", "/v1/messages/count_tokens"),
        ("GET", "/v1/models"),
        ("POST", "/v1/responses"),
    }
)


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
_write_lock = threading.Lock()


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
    with _cache_lock:
        cached = _cached_settings
        if cached is not None and now - cached[0] < _SETTINGS_CACHE_TTL_S:
            return cached[1], cached[2]
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


def get_keyless_api_access_settings() -> tuple[str, bool]:
    """Return the canonical scope and tool grant from one settings generation."""
    return _settings()


def get_keyless_api_tools_enabled() -> bool:
    """Whether a keyless caller may drive the server-side tool loop."""
    return _settings()[1]


def set_keyless_api_access(value: Any, *, tools: Any = None) -> tuple[str, bool]:
    """Persist which routes are served without a key, and whether tools come with them."""
    global _cached_settings, _settings_generation
    scope = _coerce_scope(value)
    if scope is None:
        raise ValueError(f"Keyless API access scope must be one of: {', '.join(KEYLESS_SCOPES)}.")
    with _write_lock:
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
    if lan_connector_active():
        from lan_access import lan_listener_status
        from utils.lan_access_settings import _normalized_ip, _private_non_loopback

        try:
            addresses = tuple(_normalized_ip(value) for value in lan_listener_status()["addresses"])
        except Exception:
            return "network"
        return (
            "private_lan"
            if addresses
            and all(address is not None and _private_non_loopback(address) for address in addresses)
            else "network"
        )
    if bool(getattr(app_state, "lan_access_launch_managed", False)):
        from utils.lan_access_settings import _normalized_ip, _private_non_loopback
        addresses = tuple(
            _normalized_ip(value)
            for value in (getattr(app_state, "lan_access_launch_addresses", ()) or ())
        )
        if addresses and all(
            address is not None and _private_non_loopback(address) for address in addresses
        ):
            return "private_lan"
    bind_host = getattr(app_state, "bind_host", None)
    if not isinstance(bind_host, str) or is_external_host(bind_host):
        return "network"
    return None


def normalize_request_path(path: str, root_path: str = "") -> str:
    """Normalize trailing slash and an ASGI/FastAPI mount root."""
    if not isinstance(path, str) or not path.startswith("/"):
        return ""
    root = root_path.rstrip("/") if isinstance(root_path, str) else ""
    if root and root.startswith("/"):
        if path == root:
            path = "/"
        elif path.startswith(f"{root}/"):
            path = path[len(root) :]
    return path.rstrip("/") or "/"


def scope_covers(
    scope: str,
    method: str,
    path: str,
    root_path: str = "",
) -> bool:
    """Whether ``scope`` includes this exact method and normalized route."""
    normalized = normalize_request_path(path, root_path)
    normalized_method = method.upper() if isinstance(method, str) else ""
    if scope == KEYLESS_SCOPE_FULL:
        return bool(normalized and normalized_method)
    if scope != KEYLESS_SCOPE_INFERENCE:
        return False
    if (normalized_method, normalized) in _INFERENCE_ROUTES:
        return True
    # The router intentionally exposes one dynamic retrieval template. Its method
    # is still explicit; an empty id and every non-GET alias remain denied.
    return normalized_method == "GET" and normalized.startswith("/v1/models/")


def _request_app_state(request: Any):
    try:
        return request.app.state
    except Exception:
        return None


def _hosted_mode_forbidden(app_state: Any) -> bool:
    """Whether the whole launch mode forbids keyless admission."""
    if app_state is None:
        return True
    if bool(getattr(app_state, "remote_access_is_colab", False)) or bool(
        getattr(app_state, "lan_access_is_colab", False)
    ):
        return True
    if bool(getattr(app_state, "secure", False)) or bool(
        getattr(app_state, "lan_access_secure_launch", False)
    ):
        return True
    return False


def _public_tunnel_active(app_state: Any) -> bool:
    """Whether loopback transport may actually be carrying a public tunnel request."""
    if getattr(app_state, "cloudflare_url", None):
        return True
    try:
        from utils.host_policy import tunnel_connector_active
        return tunnel_connector_active()
    except Exception:
        return True


def _full_scope_transport_allowed(request: Any, app_state: Any) -> bool:
    from utils.lan_access_settings import _all_addresses_are, request_is_loopback

    if not request_is_loopback(request):
        return False
    bind_host = getattr(app_state, "bind_host", None)
    scope = getattr(request, "scope", {})
    server = scope.get("server")
    if not isinstance(bind_host, str) or bind_host in ("0.0.0.0", "::"):
        return False
    if not isinstance(server, (tuple, list)) or len(server) < 2:
        return False
    port = server[1]
    return isinstance(port, int) and _all_addresses_are(
        bind_host, port, lambda address: address.is_loopback
    )


def keyless_transport_allowed(request: Any, scope: str) -> bool:
    """Enforce the loopback/private-LAN boundary from authoritative ASGI state."""
    try:
        if request.headers.get("origin") is not None:
            return False
    except Exception:
        return False
    app_state = _request_app_state(request)
    if _hosted_mode_forbidden(app_state):
        return False
    if scope == KEYLESS_SCOPE_FULL:
        if _public_tunnel_active(app_state):
            return False
        return _full_scope_transport_allowed(request, app_state)
    if scope != KEYLESS_SCOPE_INFERENCE:
        return False
    from utils.lan_access_settings import request_is_loopback, request_on_lan_access

    if request_on_lan_access(request):
        return True
    if _public_tunnel_active(app_state):
        return False
    return request_is_loopback(request)


def keyless_request_allowed(request: Any) -> bool:
    """Whether the route and transport are eligible for keyless authentication."""
    scope = get_keyless_api_access_scope()
    if scope == KEYLESS_SCOPE_OFF:
        return False
    asgi_scope = getattr(request, "scope", {})
    method = asgi_scope.get("method", "")
    path = asgi_scope.get("path", "")
    root_path = asgi_scope.get("root_path", "")
    if not scope_covers(scope, method, path, root_path):
        return False
    return keyless_transport_allowed(request, scope)


def mark_keyless_admission(request: Any, admitted: bool) -> None:
    """Publish the authoritative admission result for downstream policy decisions."""
    try:
        setattr(request.state, KEYLESS_ADMISSION_STATE_KEY, bool(admitted))
    except Exception:
        pass


def request_was_admitted_keyless(request: Any) -> Optional[bool]:
    """Return a recorded admission decision, or None before auth has classified it."""
    try:
        value = getattr(request.state, KEYLESS_ADMISSION_STATE_KEY)
    except Exception:
        return None
    return value if isinstance(value, bool) else None


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
        if asgi_scope.get("type") != "http":
            await self.app(asgi_scope, receive, send)
            return
        from starlette.concurrency import run_in_threadpool

        admitted = await run_in_threadpool(asgi_request_is_keyless, asgi_scope)
        asgi_scope.setdefault("state", {})[KEYLESS_ADMISSION_STATE_KEY] = admitted
        if not admitted:
            await self.app(asgi_scope, receive, send)
            return
        if await run_in_threadpool(get_keyless_api_tools_enabled):
            await self.app(asgi_scope, receive, send)
            return
        from state.tool_policy import tools_force_disabled

        with tools_force_disabled():
            await self.app(asgi_scope, receive, send)


def asgi_request_is_keyless(asgi_scope) -> bool:
    """Whether this ASGI request is admitted by the setting rather than by a credential.

    Middleware-side twin of ``auth.authentication.admitted_without_credential``, reading
    the raw scope because it runs before the request object exists. An Unsloth session and
    a working API key both authenticate as themselves, so neither is keyless: applying
    the tool restriction to an existing API client would take away tools it already had.
    """
    try:
        from starlette.requests import Request
        request = Request(asgi_scope)
    except Exception:
        return False
    if not keyless_request_allowed(request):
        return False
    authorization = [
        bytes(value).decode("latin-1")
        for name, value in asgi_scope.get("headers") or ()
        if bytes(name).lower() == b"authorization"
    ]
    if not authorization:
        return True
    if len(authorization) != 1:
        return False
    scheme, separator, token = authorization[0].partition(" ")
    return bool(separator and scheme.lower() == "bearer" and token in APPROVED_DUMMY_BEARERS)
