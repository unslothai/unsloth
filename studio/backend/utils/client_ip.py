# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Resolve the caller's IP for rate limiting.

Trust model, in order:
  1. If the operator opts in via ``UNSLOTH_STUDIO_TRUST_FORWARDED`` (Studio behind
     their own reverse proxy), honor the *rightmost* ``X-Forwarded-For`` hop -- the
     one the trusted proxy appended. The leftmost entry is client-controlled and
     spoofable, so this assumes a proxy that appends (or overwrites) the header;
     only enable the env var behind such a proxy.
  2. If an operator explicitly opts in, or Studio's managed Cloudflare tunnel is
     active and the request carries the notebook iframe token cookie, honor
     ``CF-Connecting-IP``. The cookie condition stops raw localhost callers from
     forging fresh buckets while preserving per-client buckets for notebook
     iframe traffic that arrived through the public tunnel.
  3. Otherwise the socket peer, so a direct LAN caller can't spoof a header to
     dodge a per-IP limit.
"""

from __future__ import annotations

import ipaddress
import os

from utils.notebook_frame_auth import notebook_frame_cookie_matches

_TRUST_FORWARDED_ENV = "UNSLOTH_STUDIO_TRUST_FORWARDED"
_TRUST_CF_CONNECTING_IP_ENV = "UNSLOTH_STUDIO_TRUST_CF_CONNECTING_IP"


def _trust_forwarded_for() -> bool:
    return os.environ.get(_TRUST_FORWARDED_ENV, "").strip().lower() in {"1", "true", "yes"}


def _trust_cf_connecting_ip(request) -> bool:
    if os.environ.get(_TRUST_CF_CONNECTING_IP_ENV, "").strip().lower() in {"1", "true", "yes"}:
        return True
    try:
        if not bool(
            getattr(
                getattr(getattr(request, "app", None), "state", None),
                "trust_cloudflare_client_ip",
                False,
            )
        ):
            return False
        return notebook_frame_cookie_matches(request.headers)
    except Exception:
        return False


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
    if raw.startswith("["):  # [ipv6]:port
        raw = raw[1:].split("]", 1)[0]
    elif raw.count(":") == 1:  # ipv4:port
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
    if _is_loopback(peer) and _trust_cf_connecting_ip(request):
        cf = _normalize_addr(request.headers.get("cf-connecting-ip"))
        if cf:
            return cf
    return peer or "_unknown"
