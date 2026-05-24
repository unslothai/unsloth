# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Extra edge-case coverage for the bootstrap-pw cross-origin gate.

Companion to ``test_index_bootstrap_origin.py``. Pinned down by the audit
on PR 5739: IPv6 netlocs, ``data:`` opaque origins, comma-joined multi
``Origin`` headers, whitespace around the header value, and the
``localhost`` vs ``127.0.0.1`` distinction (which the web platform
treats as distinct origins).
"""

from unittest.mock import MagicMock


def _build_request(host: str, origin, scheme: str = "http") -> MagicMock:
    request = MagicMock()
    request.url.scheme = scheme
    request.url.netloc = host
    request.headers = {"origin": origin} if origin is not None else {}
    return request


# ── IPv6 ────────────────────────────────────────────────────────────


def test_is_same_origin_request_ipv6_loopback_same_origin():
    """Studio supports ``-H ::1`` binds; Starlette's ``request.url.netloc``
    for an IPv6 server is ``[::1]:8902`` and browsers send
    ``Origin: http://[::1]:8902``. Bare ``partition(":")`` mis-parses the
    bracketed form (host=``[``, port-str=``:1]:8902``) and refuses to
    inject the bootstrap on a legitimate same-origin nav.
    """
    from main import _is_same_origin_request

    req = _build_request("[::1]:8902", origin = "http://[::1]:8902")
    assert _is_same_origin_request(req) is True


def test_is_same_origin_request_ipv6_full_address_same_origin():
    from main import _is_same_origin_request

    req = _build_request(
        "[2001:db8::1]:8443",
        origin = "https://[2001:db8::1]:8443",
        scheme = "https",
    )
    assert _is_same_origin_request(req) is True


def test_is_same_origin_request_ipv6_default_port_stripped():
    """Browser drops :80 on ``http://[::1]``."""
    from main import _is_same_origin_request

    req = _build_request("[::1]:80", origin = "http://[::1]")
    assert _is_same_origin_request(req) is True


def test_is_same_origin_request_ipv6_case_insensitive():
    """Hex digits in IPv6 are case-insensitive per RFC 5952."""
    from main import _is_same_origin_request

    req = _build_request(
        "[2001:DB8::1]:8443",
        origin = "https://[2001:db8::1]:8443",
        scheme = "https",
    )
    assert _is_same_origin_request(req) is True


def test_is_same_origin_request_ipv6_different_host_cross_origin():
    from main import _is_same_origin_request

    req = _build_request("[::1]:8902", origin = "http://[2001:db8::1]:8902")
    assert _is_same_origin_request(req) is False


def test_is_same_origin_request_ipv6_port_mismatch_cross_origin():
    from main import _is_same_origin_request

    req = _build_request("[::1]:8902", origin = "http://[::1]:9999")
    assert _is_same_origin_request(req) is False


def test_is_same_origin_request_ipv6_userinfo_stripped():
    from main import _is_same_origin_request

    req = _build_request("user:pass@[::1]:8902", origin = "http://[::1]:8902")
    assert _is_same_origin_request(req) is True


# ── Opaque origins (data:, blob:) ───────────────────────────────────


def test_is_same_origin_request_data_url_origin_is_cross_origin():
    """``data:`` URLs are opaque origins per HTML living standard.
    They carry no host and must never be treated as same-origin.
    """
    from main import _is_same_origin_request

    req = _build_request(
        "127.0.0.1:8902", origin = "data:text/html,<script>alert(1)</script>"
    )
    assert _is_same_origin_request(req) is False


def test_is_same_origin_request_blob_url_origin_is_cross_origin():
    """``blob:`` URLs carry the inner origin in the path, but only as
    a non-canonical form. The canonical comparison rejects them.
    """
    from main import _is_same_origin_request

    req = _build_request("127.0.0.1:8902", origin = "blob:http://127.0.0.1:8902/uuid")
    assert _is_same_origin_request(req) is False


def test_is_same_origin_request_file_url_origin_is_cross_origin():
    """``file://`` pages typically send ``Origin: null`` but a few
    historical engines sent ``Origin: file://``. Either way, not
    same-origin against an http listener.
    """
    from main import _is_same_origin_request

    req = _build_request("127.0.0.1:8902", origin = "file://")
    assert _is_same_origin_request(req) is False


# ── Multi-Origin header (comma-joined by Starlette) ────────────────


def test_is_same_origin_request_comma_joined_origins_cross_origin():
    """HTTP allows repeated headers; Starlette concatenates with
    ``, ``. The canonical parser can't split this safely, so it
    must fall to cross-origin (refusing to inject) rather than
    pick the first token.
    """
    from main import _is_same_origin_request

    req = _build_request(
        "127.0.0.1:8902",
        origin = "http://127.0.0.1:8902, http://evil.example",
    )
    assert _is_same_origin_request(req) is False


# ── localhost vs 127.0.0.1 (distinct origins per web platform) ──────


def test_is_same_origin_request_localhost_vs_127_is_cross_origin():
    """The browser treats ``localhost`` and ``127.0.0.1`` as distinct
    origins even though they resolve to the same socket. The
    canonical comparison must mirror that (no DNS-style collapsing).
    """
    from main import _is_same_origin_request

    req = _build_request("127.0.0.1:8902", origin = "http://localhost:8902")
    assert _is_same_origin_request(req) is False


def test_is_same_origin_request_127_vs_localhost_is_cross_origin():
    from main import _is_same_origin_request

    req = _build_request("localhost:8902", origin = "http://127.0.0.1:8902")
    assert _is_same_origin_request(req) is False
