# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Origin comparison shared by main.py and the /mcp gate."""

from __future__ import annotations

import ipaddress
import os
from typing import Any, Optional
from urllib.parse import urlparse

DEFAULT_PORTS = {"http": 80, "https": 443, "ws": 80, "wss": 443}


def canonical_origin(scheme: str, netloc: str) -> Optional[tuple[str, str, int]]:
    """Canonicalise an Origin to ``(scheme, host, port)`` for equality. Browsers strip default ports (RFC 6454
    sec 6.1) and scheme/host are case-insensitive (RFC 3986), so a bare string compare misclassifies
    same-origin requests as cross-origin. Returns ``None`` on unparseable input so callers fall to the safer
    cross-origin default."""
    scheme = (scheme or "").strip().lower()
    if not scheme or not netloc:
        return None
    # Strip userinfo (RFC 3986); Origin never carries credentials.
    if "@" in netloc:
        netloc = netloc.rsplit("@", 1)[1]
    # IPv6 hosts use brackets (RFC 3986 3.2.2): bare partition(":") breaks `-H ::1`.
    if netloc.startswith("["):
        close = netloc.find("]")
        if close == -1:
            return None
        host = netloc[1:close]
        rest = netloc[close + 1 :]
        if rest.startswith(":"):
            port_str = rest[1:]
        elif rest == "":
            port_str = ""
        else:
            return None
    else:
        host, _, port_str = netloc.partition(":")
    host = host.strip().lower()
    if not host:
        return None
    if port_str:
        try:
            port = int(port_str)
        except ValueError:
            return None
    else:
        port = DEFAULT_PORTS.get(scheme, 0)
    return (scheme, host, port)


def origin_of(url: Optional[str]) -> Optional[tuple[str, str, int]]:
    """Canonical origin of a URL or of an Origin header value, or ``None`` when it is neither."""
    if not url:
        return None
    try:
        parsed = urlparse(url)
    except ValueError:
        return None
    return canonical_origin(parsed.scheme, parsed.netloc)


# Never the localhost dev-server origins from host_policy: any local page could claim those.
TAURI_ORIGINS = ("tauri://localhost", "http://tauri.localhost")
_LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})


def _is_literal_ip(host: str) -> bool:
    try:
        ipaddress.ip_address(host)
    except ValueError:
        return False
    return True


def mcp_origin_allowed(
    origin: str, *, request_scheme: str, request_netloc: str, app_state: Any
) -> bool:
    """Whether a browser page at ``origin`` may call /mcp. "Origin equals Host" is not enough: a DNS-rebound page has both set to its own name, so a named host is trusted only when Studio itself published it (the tunnel, Tauri, UNSLOTH_CORS_ORIGINS)."""
    if not origin or origin == "null":
        return False
    canon = origin_of(origin)
    if canon is None:
        return False
    named = [*TAURI_ORIGINS, getattr(app_state, "cloudflare_url", None)]
    named += [
        entry.strip()
        for entry in os.environ.get("UNSLOTH_CORS_ORIGINS", "").split(",")
        if entry.strip() and entry.strip() != "*"
    ]
    if canon in {origin_of(value) for value in named if value}:
        return True
    try:
        own = canonical_origin(request_scheme, request_netloc)
    except ValueError:
        own = None
    scheme, host, port = canon
    if host in _LOOPBACK_HOSTS and scheme in ("http", "https"):
        ports = {getattr(app_state, "server_port", None), own[2] if own else None}
        if port in ports:
            return True
    # A rebound name cannot pose as a literal IP, so the page's own LAN address is safe to trust.
    return own is not None and canon == own and _is_literal_ip(own[1])
