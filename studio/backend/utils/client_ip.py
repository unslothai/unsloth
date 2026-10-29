# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Trust order: rightmost X-Forwarded-For only if UNSLOTH_STUDIO_TRUST_FORWARDED (leftmost is
spoofable; enable only behind an appending proxy); CF-Connecting-IP only via loopback tunnel;
relayed:<peer>; socket peer"""

from __future__ import annotations

import ipaddress
import os
from urllib.parse import urlparse

_TRUST_FORWARDED_ENV = "UNSLOTH_STUDIO_TRUST_FORWARDED"


def _trust_forwarded_for() -> bool:
    return os.environ.get(_TRUST_FORWARDED_ENV, "").strip().lower() in {"1", "true", "yes"}


def trust_forwarded_headers() -> bool:
    return _trust_forwarded_for()


def _is_loopback(host: str | None) -> bool:
    try:
        return bool(host) and ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _normalize_addr(value: str | None) -> str | None:
    """Parse an ``X-Forwarded-For`` entry into a bare, validated IP (strip port/brackets)."""
    raw = (value or "").strip().strip('"')
    if not raw:
        return None
    if raw.startswith("["):
        raw = raw[1:].split("]", 1)[0]
    elif raw.count(":") == 1:
        raw = raw.split(":", 1)[0]
    try:
        return ipaddress.ip_address(raw).compressed
    except ValueError:
        return None


def header_fields(headers, name: str) -> str:
    """All fields of a repeated header, comma-joined (``get`` returns only the first; HAProxy appends a field)."""
    getlist = getattr(headers, "getlist", None)
    if getlist is not None:
        return ",".join(getlist(name))
    return headers.get(name, "") or ""


def client_ip(request) -> str:
    """Best-effort client IP, or ``"_unknown"`` when it can't be determined."""
    if request is None:
        return "_unknown"
    peer = request.client.host if request.client else None
    if _trust_forwarded_for():
        # Rightmost hop = what the trusted proxy saw; the leftmost is spoofable.
        xff = header_fields(request.headers, "x-forwarded-for")
        if xff:
            normalized = _normalize_addr(xff.rsplit(",", 1)[-1])
            if normalized:
                return normalized
    if _is_loopback(peer):
        if request_from_managed_tunnel(
            getattr(request, "scope", None), _request_app_state(request)
        ):
            cf = _normalize_addr(request.headers.get("cf-connecting-ip"))
            if cf:
                return cf
        if any(request.headers.get(h) is not None for h in _PROXIED_CLIENT_HEADERS):
            return f"relayed:{peer}"
    return peer or "_unknown"


def _request_app_state(request):
    try:
        return request.app.state
    except Exception:
        return None


def _managed_tunnel_hostname(app_state) -> str | None:
    url = getattr(app_state, "cloudflare_url", None) if app_state is not None else None
    if not isinstance(url, str) or not url:
        return None
    try:
        host = urlparse(url).hostname
    except ValueError:
        return None
    return host.lower().rstrip(".") if host else None


def request_from_managed_tunnel(scope, app_state) -> bool:
    """Live published tunnel, one valid ``CF-Connecting-IP``, Host naming it. Caller checks the peer is loopback."""
    expected = _managed_tunnel_hostname(app_state)
    if not expected or not isinstance(scope, dict):
        return False
    cf_values = []
    host_values = []
    for name, value in scope.get("headers") or ():
        if name == b"cf-connecting-ip":
            cf_values.append(value)
        elif name == b"host":
            host_values.append(value)
    if len(cf_values) != 1 or len(host_values) != 1:
        return False
    try:
        if _normalize_addr(cf_values[0].decode("latin-1")) is None:
            return False
        request_host = urlparse("//" + host_values[0].decode("latin-1")).hostname
    except (UnicodeDecodeError, ValueError):
        return False
    return bool(request_host) and request_host.lower().rstrip(".") == expected


def _is_loopback_ip(host: str | None) -> bool:
    if not host or "%" in host:  # a scope id (::1%eth0) is never a plain loopback
        return False
    try:
        ip = ipaddress.ip_address(host)
    except (TypeError, ValueError):
        return False
    mapped = getattr(ip, "ipv4_mapped", None)
    return ip.is_loopback or (mapped is not None and mapped.is_loopback)


# A loopback peer with these headers is a proxy relaying a remote client.
_PROXIED_CLIENT_HEADERS = (
    "cf-connecting-ip",
    "forwarded",
    "x-forwarded-for",
    "x-forwarded-host",
    "x-real-ip",
)


def _host_header_is_loopback(host_header: str | None) -> bool:
    """Raw Host header, so a bad Host cannot fall back to the (loopback) ASGI server address."""
    if not host_header:
        return False
    host = host_header.strip()
    if host.startswith("["):  # [IPv6] or [IPv6]:port
        end = host.find("]")
        if end == -1 or (host[end + 1 :] and not host[end + 1 :].startswith(":")):
            return False  # unclosed bracket or junk after ] (e.g. [::1]evil)
        host = host[1:end]
    elif host.count(":") == 1:  # host:port
        host = host.split(":", 1)[0]
    host = host.lower().rstrip(".")
    return host == "localhost" or _is_loopback_ip(host)


# Hop headers a proxy adds without naming the client (Apache's X-Forwarded-Server, an nginx block that
# only sets X-Forwarded-Proto). Kept apart because Colab's tolerated-header check walks the tuple above.
_PROXY_HOP_HEADERS = ("via", "x-forwarded-port", "x-forwarded-proto", "x-forwarded-server")


def names_a_forwarded_client(request) -> bool:
    """Whether a proxy or tunnel marked this request as relayed."""
    return any(
        request.headers.get(h) is not None for h in _PROXIED_CLIENT_HEADERS + _PROXY_HOP_HEADERS
    )


def is_direct_local_request(request) -> bool:
    """A direct loopback connection to a loopback Host: not relayed by a proxy or tunnel, not DNS rebinding."""
    client = request.client
    if client is None or not _is_loopback_ip(client.host):
        return False
    if any(request.headers.get(h) is not None for h in _PROXIED_CLIENT_HEADERS):
        return False
    return _host_header_is_loopback(request.headers.get("host"))
