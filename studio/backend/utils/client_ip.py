# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Resolve the caller's IP for rate limiting.

Mirrors the trust model used by the login limiter in ``routes/auth.py``: the
socket peer is used by default, and ``X-Forwarded-For`` is only honored when the
operator opts in (Studio behind a known reverse proxy / Cloudflare tunnel), so a
direct caller can't spoof the header to dodge a per-IP limit.
"""

from __future__ import annotations

import ipaddress
import os

_TRUST_FORWARDED_ENV = "UNSLOTH_STUDIO_TRUST_FORWARDED"


def _trust_forwarded_for() -> bool:
    return os.environ.get(_TRUST_FORWARDED_ENV, "").strip().lower() in {"1", "true", "yes"}


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
    if _trust_forwarded_for():
        xff = request.headers.get("x-forwarded-for", "")
        if xff:
            normalized = _normalize_addr(xff.split(",", 1)[0])
            if normalized:
                return normalized
    return (request.client.host if request.client else None) or "_unknown"
