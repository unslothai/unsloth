# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Resolve the caller's IP for rate limiting.

Trust model, in order:
  1. If the operator opts in via ``UNSLOTH_STUDIO_TRUST_FORWARDED`` (Unsloth behind
     their own reverse proxy), honor the *rightmost* ``X-Forwarded-For`` hop -- the
     one the trusted proxy appended. The leftmost entry is client-controlled and
     spoofable, so this assumes a proxy that appends (or overwrites) the header;
     only enable the env var behind such a proxy.
  2. If the socket peer is loopback, honor ``CF-Connecting-IP``. Unsloth's managed
     Cloudflare tunnel terminates at 127.0.0.1, so every tunneled visitor would
     otherwise collapse onto the same socket peer (the local cloudflared process)
     and share one rate-limit bucket. ``CF-Connecting-IP`` is set by Cloudflare's
     edge and can't be forged by a tunneled client.
  3. Otherwise the socket peer, so a direct LAN caller can't spoof a header to
     dodge a per-IP limit.
"""

from __future__ import annotations

import ipaddress
import os

_TRUST_FORWARDED_ENV = "UNSLOTH_STUDIO_TRUST_FORWARDED"


def _trust_forwarded_for() -> bool:
    return os.environ.get(_TRUST_FORWARDED_ENV, "").strip().lower() in {"1", "true", "yes"}


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


def client_ip(request) -> str:
    """Best-effort client IP, or ``"_unknown"`` when it can't be determined."""
    if request is None:
        return "_unknown"
    peer = request.client.host if request.client else None
    if _trust_forwarded_for():
        # Rightmost hop = what the trusted proxy saw; the leftmost is spoofable.
        xff = request.headers.get("x-forwarded-for", "")
        if xff:
            normalized = _normalize_addr(xff.rsplit(",", 1)[-1])
            if normalized:
                return normalized
    if _is_loopback(peer):
        cf = _normalize_addr(request.headers.get("cf-connecting-ip"))
        if cf:
            return cf
    return peer or "_unknown"


def _is_loopback_ip(host: str | None) -> bool:
    if not host or "%" in host:  # a scope id (::1%eth0) is never a plain loopback
        return False
    try:
        ip = ipaddress.ip_address(host)
    except (TypeError, ValueError):
        return False
    mapped = getattr(ip, "ipv4_mapped", None)
    return ip.is_loopback or (mapped is not None and mapped.is_loopback)


# A loopback peer carrying any of these is a proxy/tunnel relaying a remote client, so the peer is the
# proxy, not the caller: cloudflared sets cf-connecting-ip, reverse proxies set the rest.
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


def is_direct_local_request(request) -> bool:
    """A direct loopback connection to a loopback Host: not relayed by a proxy or tunnel, not DNS rebinding."""
    client = request.client
    if client is None or not _is_loopback_ip(client.host):
        return False
    if any(request.headers.get(h) is not None for h in _PROXIED_CLIENT_HEADERS):
        return False
    return _host_header_is_loopback(request.headers.get("host"))
