# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""auth.authentication.is_host_session: the same-machine gate for the update
endpoints.

The desktop JWT claim is the primary signal; a loopback socket peer is the
secondary one, but only when the request carries no forwarding header (behind
the managed Cloudflare tunnel / a reverse proxy every remote visitor's peer is
loopback).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

pytest.importorskip("fastapi")
pytest.importorskip("jwt")

from auth import authentication as auth  # noqa: E402


class _Headers:
    """Case-insensitive header lookup, like starlette's Headers."""

    def __init__(self, values = None):
        self._values = {k.lower(): v for k, v in (values or {}).items()}

    def get(self, key, default = None):
        return self._values.get(key.lower(), default)


class _Client:
    def __init__(self, host):
        self.host = host


class _Request:
    def __init__(self, *, headers = None, client_host = None):
        self.headers = _Headers(headers)
        self.client = _Client(client_host) if client_host is not None else None


def test_loopback_ipv4_is_host():
    assert auth.is_host_session(_Request(client_host = "127.0.0.1")) is True


def test_loopback_ipv6_is_host():
    assert auth.is_host_session(_Request(client_host = "::1")) is True


def test_lan_peer_is_not_host():
    assert auth.is_host_session(_Request(client_host = "192.168.1.42")) is False


def test_no_client_is_not_host():
    assert auth.is_host_session(_Request(client_host = None)) is False


def test_loopback_behind_cloudflare_tunnel_is_not_host():
    # Managed tunnel terminates at 127.0.0.1; CF-Connecting-IP means the real
    # client is remote, so loopback no longer proves locality.
    req = _Request(
        client_host = "127.0.0.1",
        headers = {"CF-Connecting-IP": "203.0.113.7"},
    )
    assert auth.is_host_session(req) is False


def test_loopback_behind_reverse_proxy_is_not_host():
    req = _Request(
        client_host = "127.0.0.1",
        headers = {"X-Forwarded-For": "203.0.113.7"},
    )
    assert auth.is_host_session(req) is False


def test_loopback_with_empty_forwarding_header_is_not_host():
    # A present-but-empty forwarding header still means the request was proxied
    # (e.g. uvicorn with forwarded_allow_ips="*"); presence, not value, disqualifies.
    req = _Request(client_host = "127.0.0.1", headers = {"X-Forwarded-For": ""})
    assert auth.is_host_session(req) is False


def test_desktop_claim_beats_remote_peer(monkeypatch):
    # The desktop JWT claim identifies the host session even from a non-loopback
    # peer (e.g. the desktop app reaching its backend on a LAN interface).
    monkeypatch.setattr(auth, "is_desktop_access_token", lambda token: token == "good")
    req = _Request(
        client_host = "192.168.1.42",
        headers = {"Authorization": "Bearer good"},
    )
    assert auth.is_host_session(req) is True


def test_non_desktop_token_from_remote_peer_is_not_host(monkeypatch):
    monkeypatch.setattr(auth, "is_desktop_access_token", lambda token: False)
    req = _Request(
        client_host = "192.168.1.42",
        headers = {"Authorization": "Bearer normal-session"},
    )
    assert auth.is_host_session(req) is False


def test_desktop_claim_not_consulted_for_non_bearer(monkeypatch):
    called = {"n": 0}

    def spy(_token):
        called["n"] += 1
        return True

    monkeypatch.setattr(auth, "is_desktop_access_token", spy)
    # Basic auth header (not Bearer) must not be treated as a desktop token, and
    # with a LAN peer the request is not host-local.
    req = _Request(client_host = "192.168.1.42", headers = {"Authorization": "Basic abc"})
    assert auth.is_host_session(req) is False
    assert called["n"] == 0
