# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Rewrites client and scheme only for the Studio Cloudflare tunnel and Colab's notebook proxy, since any
loopback peer could spoof X-Forwarded-For; only UNSLOTH_STUDIO_TRUST_FORWARDED keeps the default"""

from __future__ import annotations

from typing import Any

from utils.client_ip import (
    _is_loopback_ip,
    _normalize_addr,
    request_from_managed_tunnel,
    trust_forwarded_headers,
)


def uvicorn_forwarded_kwargs(*, is_colab: bool) -> dict[str, Any]:
    """The ``proxy_headers`` / ``forwarded_allow_ips`` arguments for ``uvicorn.Config``."""
    if trust_forwarded_headers() and not is_colab:
        # uvicorn's own behaviour, including FORWARDED_ALLOW_IPS
        return {"proxy_headers": True}
    return {"proxy_headers": False}


class ManagedTunnelProxyHeaders:
    def __init__(
        self,
        app,
        app_state,
        is_colab = False,
    ):
        self.app = app
        self.app_state = app_state
        self.is_colab = is_colab

    async def __call__(self, scope, receive, send):
        if scope.get("type") in ("http", "websocket"):
            client = scope.get("client")
            peer = client[0] if client else None
            if _is_loopback_ip(peer) and request_from_managed_tunnel(scope, self.app_state):
                scope = _rewrite_client_and_scheme(scope, b"cf-connecting-ip")
            elif self.is_colab:
                # Colab's proxy appends the peer it saw; earlier entries are caller-supplied.
                scope = _rewrite_client_and_scheme(scope, b"x-forwarded-for")
        await self.app(scope, receive, send)


def _rewrite_client_and_scheme(scope, client_header):
    proto = None
    addrs = []
    for name, value in scope.get("headers") or ():
        if name == b"x-forwarded-proto":
            proto = value.decode("latin-1").strip()
        elif name == client_header:
            addrs.append(value.decode("latin-1"))
    scope = dict(scope)
    if proto in ("http", "https", "ws", "wss"):
        if scope["type"] == "websocket":
            scope["scheme"] = proto.replace("http", "ws")
        else:
            scope["scheme"] = proto
    visitor = _normalize_addr(",".join(addrs).rsplit(",", 1)[-1]) if addrs else None
    if visitor:
        scope["client"] = (visitor, 0)
    return scope


def wrap_direct_server(
    app,
    app_state,
    is_colab = False,
):
    """The ASGI app to hand uvicorn when its own proxy-header rewriting is off."""
    return ManagedTunnelProxyHeaders(app, app_state, is_colab)
