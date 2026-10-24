# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Which forwarded headers rewrite the ASGI client and scheme.

uvicorn defaults to ``proxy_headers=True`` with ``forwarded_allow_ips="127.0.0.1"``, so any loopback peer
could set ``request.client`` from ``X-Forwarded-For`` and the scheme from ``X-Forwarded-Proto``. A same-host
proxy that relays a client-supplied ``X-Forwarded-For`` unchanged then lets the caller pick the address every
per-IP limiter keys on. Studio therefore runs uvicorn as a direct server unless forwarded headers are trusted:

* ``UNSLOTH_STUDIO_TRUST_FORWARDED`` (the operator runs Studio behind their own proxy) and Colab (its notebook
  proxy) keep uvicorn's rewriting, exactly as before.
* Otherwise uvicorn's rewriting is off, and only a request relayed by the Cloudflare tunnel Studio itself
  manages is rewritten: the client from ``CF-Connecting-IP`` (set by Cloudflare's edge) and the scheme from
  ``X-Forwarded-Proto``, matching what uvicorn did for that tunnel before.
"""

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
    if is_colab:
        # Colab's single-user sandbox: trust its reverse proxy so the app sees the real https origin.
        return {"proxy_headers": True, "forwarded_allow_ips": "*"}
    if trust_forwarded_headers():
        # Operator opt-in: uvicorn's own behaviour, including FORWARDED_ALLOW_IPS.
        return {"proxy_headers": True}
    return {"proxy_headers": False}


class ManagedTunnelProxyHeaders:
    """Rewrite client and scheme for requests relayed by Studio's managed Cloudflare tunnel only."""

    def __init__(self, app, app_state):
        self.app = app
        self.app_state = app_state

    async def __call__(self, scope, receive, send):
        if scope.get("type") in ("http", "websocket"):
            client = scope.get("client")
            peer = client[0] if client else None
            if _is_loopback_ip(peer) and request_from_managed_tunnel(scope, self.app_state):
                headers = scope.get("headers") or ()
                proto = None
                cf_ip = None
                for name, value in headers:
                    if name == b"x-forwarded-proto":
                        proto = value.decode("latin-1").strip()
                    elif name == b"cf-connecting-ip":
                        cf_ip = value.decode("latin-1")
                scope = dict(scope)
                if proto in ("http", "https", "ws", "wss"):
                    if scope["type"] == "websocket":
                        scope["scheme"] = proto.replace("http", "ws")
                    else:
                        scope["scheme"] = proto
                visitor = _normalize_addr(cf_ip)
                if visitor:
                    scope["client"] = (visitor, 0)
        await self.app(scope, receive, send)


def wrap_direct_server(app, app_state):
    """The ASGI app to hand uvicorn when its own proxy-header rewriting is off."""
    return ManagedTunnelProxyHeaders(app, app_state)
