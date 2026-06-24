# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Regression coverage for the bootstrap-pw cross-origin leak (PR 5739).

``_is_same_origin_request`` gates ``_inject_bootstrap`` so the seeded admin
password only ships to same-origin callers.
"""

import os
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


def _build_request(
    host: str,
    origin: str | None,
    scheme: str = "http",
    client_host: str | None = "127.0.0.1",
    extra_headers: dict | None = None,
) -> MagicMock:
    request = MagicMock()
    request.url.scheme = scheme
    request.url.netloc = host
    headers = {"origin": origin} if origin is not None else {}
    if extra_headers:
        headers.update(extra_headers)
    request.headers = headers
    request.client = SimpleNamespace(host = client_host) if client_host is not None else None
    return request


def test_is_same_origin_request_missing_origin_is_same_origin(monkeypatch):
    from main import _is_same_origin_request
    req = _build_request("127.0.0.1:8888", origin = None)
    assert _is_same_origin_request(req) is True


def test_is_same_origin_request_matching_origin_is_same_origin():
    from main import _is_same_origin_request
    req = _build_request("127.0.0.1:8888", origin = "http://127.0.0.1:8888")
    assert _is_same_origin_request(req) is True


def test_is_same_origin_request_evil_origin_is_cross_origin():
    from main import _is_same_origin_request
    req = _build_request("127.0.0.1:8888", origin = "https://evil.example")
    assert _is_same_origin_request(req) is False


def test_is_same_origin_request_scheme_mismatch_is_cross_origin():
    # https origin against an http listener is not same-origin.
    from main import _is_same_origin_request
    req = _build_request("127.0.0.1:8888", origin = "https://127.0.0.1:8888")
    assert _is_same_origin_request(req) is False


def test_is_same_origin_request_port_mismatch_is_cross_origin():
    # Same host different port is not same-origin per the web platform.
    from main import _is_same_origin_request
    req = _build_request("127.0.0.1:8888", origin = "http://127.0.0.1:5173")
    assert _is_same_origin_request(req) is False


# ── Canonicalisation: default-port stripping + case folding ─────────


def test_is_same_origin_request_https_default_port_stripped_on_origin():
    """RFC 6454 strips default ports on Origin; canonicalise both sides so this stays same-origin."""
    from main import _is_same_origin_request

    req = _build_request("example.com:443", origin = "https://example.com", scheme = "https")
    assert _is_same_origin_request(req) is True


def test_is_same_origin_request_http_default_port_stripped_on_origin():
    from main import _is_same_origin_request
    req = _build_request("example.com:80", origin = "http://example.com")
    assert _is_same_origin_request(req) is True


def test_is_same_origin_request_default_port_present_on_origin():
    """Mirror case: Origin carries the default port, netloc doesn't. Same-origin."""
    from main import _is_same_origin_request

    req = _build_request("example.com", origin = "https://example.com:443", scheme = "https")
    assert _is_same_origin_request(req) is True


def test_is_same_origin_request_host_case_insensitive():
    """Host portion is case-insensitive per RFC 3986."""
    from main import _is_same_origin_request

    req = _build_request("example.com", origin = "http://EXAMPLE.com")
    assert _is_same_origin_request(req) is True


def test_is_same_origin_request_scheme_case_insensitive():
    """Scheme portion is case-insensitive per RFC 3986."""
    from main import _is_same_origin_request

    req = _build_request("example.com", origin = "HTTP://example.com")
    assert _is_same_origin_request(req) is True


def test_is_same_origin_request_null_origin_is_cross_origin():
    """Sandboxed iframes / file:// pages send ``Origin: null``; cross-origin."""
    from main import _is_same_origin_request

    req = _build_request("example.com", origin = "null")
    assert _is_same_origin_request(req) is False


def test_is_same_origin_request_unparseable_origin_is_cross_origin():
    """Hostless garbage falls to cross-origin so a malformed header can't leak the bootstrap."""
    from main import _is_same_origin_request

    req = _build_request("example.com", origin = "not-a-url")
    assert _is_same_origin_request(req) is False


def test_is_same_origin_request_userinfo_in_netloc_ignored():
    """``user:pass@host:port`` netlocs (RFC 3986) must compare equal to the credentials-less Origin."""
    from main import _is_same_origin_request

    req = _build_request("user:pass@example.com:80", origin = "http://example.com")
    assert _is_same_origin_request(req) is True


def test_is_same_origin_request_explicit_non_default_port_still_mismatch():
    """Canonicalisation does NOT collapse non-default ports to default."""
    from main import _is_same_origin_request

    req = _build_request("example.com", origin = "https://example.com:9999", scheme = "https")
    assert _is_same_origin_request(req) is False


# ── Local-direct backstop: bootstrap pw never crosses a network boundary ──
#
# The injection gate is ``_is_same_origin_request AND _is_local_direct_request``.
# These cover the second predicate: a direct loopback hit gets the password, but
# a public Cloudflare tunnel (proxy headers) or a raw 0.0.0.0/LAN bind
# (non-loopback peer) does not -- even with a missing/same Origin.


def test_local_direct_loopback_peer_no_headers():
    from main import _is_local_direct_request
    req = _build_request("127.0.0.1:8888", origin = None, client_host = "127.0.0.1")
    assert _is_local_direct_request(req) is True


def test_local_direct_ipv6_loopback_peer():
    from main import _is_local_direct_request
    req = _build_request("[::1]:8888", origin = None, client_host = "::1")
    assert _is_local_direct_request(req) is True


def test_local_direct_ipv4_mapped_loopback_peer():
    """``::ffff:127.0.0.1`` is loopback once the mapped IPv4 is unwrapped."""
    from main import _is_local_direct_request

    req = _build_request("127.0.0.1:8888", origin = None, client_host = "::ffff:127.0.0.1")
    assert _is_local_direct_request(req) is True


def test_local_direct_127_8_range_peer():
    """The whole 127.0.0.0/8 block is loopback, not just 127.0.0.1."""
    from main import _is_local_direct_request

    req = _build_request("127.0.0.1:8888", origin = None, client_host = "127.0.0.2")
    assert _is_local_direct_request(req) is True


def test_local_direct_localhost_literal_peer():
    from main import _is_local_direct_request
    req = _build_request("127.0.0.1:8888", origin = None, client_host = "localhost")
    assert _is_local_direct_request(req) is True


def test_local_direct_lan_peer_denied():
    from main import _is_local_direct_request
    req = _build_request("0.0.0.0:8888", origin = None, client_host = "192.168.1.50")
    assert _is_local_direct_request(req) is False


def test_local_direct_public_peer_denied():
    from main import _is_local_direct_request
    req = _build_request("0.0.0.0:8888", origin = None, client_host = "203.0.113.7")
    assert _is_local_direct_request(req) is False


def test_local_direct_missing_client_denied():
    from main import _is_local_direct_request
    req = _build_request("0.0.0.0:8888", origin = None, client_host = None)
    assert _is_local_direct_request(req) is False


def test_local_direct_garbage_peer_denied():
    from main import _is_local_direct_request
    req = _build_request("0.0.0.0:8888", origin = None, client_host = "not-an-ip")
    assert _is_local_direct_request(req) is False


@pytest.mark.parametrize(
    "header",
    [
        "cf-ray",
        "cf-connecting-ip",
        "x-forwarded-for",
        "x-forwarded-host",
        "x-real-ip",
        "forwarded",
    ],
)
def test_local_direct_forwarding_headers_denied(header):
    """A loopback peer + any proxy/tunnel marker (cloudflared adds these) is denied."""
    from main import _is_local_direct_request

    req = _build_request(
        "127.0.0.1:8888",
        origin = None,
        client_host = "127.0.0.1",
        extra_headers = {header: "1.2.3.4"},
    )
    assert _is_local_direct_request(req) is False


def test_local_direct_spoofed_xff_loopback_still_denied():
    """A spoofed ``X-Forwarded-For: 127.0.0.1`` must not unlock injection."""
    from main import _is_local_direct_request

    req = _build_request(
        "127.0.0.1:8888",
        origin = None,
        client_host = "127.0.0.1",
        extra_headers = {"x-forwarded-for": "127.0.0.1"},
    )
    assert _is_local_direct_request(req) is False


def test_local_direct_colab_exempt(monkeypatch):
    """Colab is owner-auth-gated and never tunnels, so injection stays allowed."""
    import main

    monkeypatch.setattr(main, "_IS_COLAB", True)
    req = _build_request(
        "studio.colab.dev",
        origin = None,
        client_host = "34.86.10.2",
        extra_headers = {"x-forwarded-for": "203.0.113.9"},
    )
    assert main._is_local_direct_request(req) is True
