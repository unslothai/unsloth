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
     active, honor ``CF-Connecting-IP`` from its loopback tunnel process.
  3. Otherwise the socket peer, so a direct LAN caller can't spoof a header to
     dodge a per-IP limit.
"""

from __future__ import annotations

import ipaddress
import os

_TRUST_FORWARDED_ENV = "UNSLOTH_STUDIO_TRUST_FORWARDED"
_TRUST_CF_CONNECTING_IP_ENV = "UNSLOTH_STUDIO_TRUST_CF_CONNECTING_IP"
_TRUE_ENV_VALUES = {"1", "true", "yes"}


def _env_enabled(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in _TRUE_ENV_VALUES


def _is_loopback(host: str | None) -> bool:
    try:
        return bool(host) and ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _trusted_cloudflare_client_ip(request, normalize):
    """Return a validated CF client IP only for an explicitly trusted loopback tunnel."""
    try:
        peer = request.client.host if request.client else None
        trusted = _env_enabled(_TRUST_CF_CONNECTING_IP_ENV) or bool(
            request.app.state.trust_cloudflare_client_ip
        )
        if not trusted or not _is_loopback(peer):
            return None
        return normalize(request.headers.get("cf-connecting-ip"))
    except Exception:
        return None


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
    if _env_enabled(_TRUST_FORWARDED_ENV):
        # Rightmost hop = what the trusted proxy saw; the leftmost is spoofable.
        xff = request.headers.get("x-forwarded-for", "")
        if xff:
            normalized = _normalize_addr(xff.rsplit(",", 1)[-1])
            if normalized:
                return normalized
    return _trusted_cloudflare_client_ip(request, _normalize_addr) or peer or "_unknown"
