# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Extra edge-case coverage for the bootstrap-pw cross-origin gate.
Companion to ``test_index_bootstrap_origin.py``: IPv6 netlocs, opaque origins
(``data:``, ``blob:``), comma-joined multi-Origin headers, and the
``localhost`` vs ``127.0.0.1`` distinct-origin rule.
"""

from unittest.mock import MagicMock


def _build_request(
    host: str,
    origin,
    scheme: str = "http",
) -> MagicMock:
    request = MagicMock()
    request.url.scheme = scheme
    request.url.netloc = host
    request.headers = {"origin": origin} if origin is not None else {}
    return request


# ── IPv6 ────────────────────────────────────────────────────────────


def test_is_same_origin_request_ipv6_loopback_same_origin():
    """Unsloth supports ``-H ::1`` binds; netloc is ``[::1]:8902``. Bare
    ``partition(":")`` mis-parses the bracketed form and would refuse the
    bootstrap on legitimate same-origin navigation.
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
    """``data:`` URLs are opaque origins (HTML living standard); no host, never same-origin."""
    from main import _is_same_origin_request

    req = _build_request("127.0.0.1:8902", origin = "data:text/html,<script>alert(1)</script>")
    assert _is_same_origin_request(req) is False


def test_is_same_origin_request_blob_url_origin_is_cross_origin():
    """``blob:`` URLs carry the inner origin only in non-canonical form; the canonical comparison rejects them."""
    from main import _is_same_origin_request

    req = _build_request("127.0.0.1:8902", origin = "blob:http://127.0.0.1:8902/uuid")
    assert _is_same_origin_request(req) is False


def test_is_same_origin_request_file_url_origin_is_cross_origin():
    """``file://`` pages usually send ``Origin: null``; older engines sent
    ``Origin: file://``. Neither is same-origin vs an http listener.
    """
    from main import _is_same_origin_request

    req = _build_request("127.0.0.1:8902", origin = "file://")
    assert _is_same_origin_request(req) is False


# ── Multi-Origin header (comma-joined by Starlette) ────────────────


def test_is_same_origin_request_comma_joined_origins_cross_origin():
    """Starlette joins repeated headers with ``, ``; the canonical parser can't
    safely split this, so it falls to cross-origin.
    """
    from main import _is_same_origin_request

    req = _build_request(
        "127.0.0.1:8902",
        origin = "http://127.0.0.1:8902, http://evil.example",
    )
    assert _is_same_origin_request(req) is False


# ── localhost vs 127.0.0.1 (distinct origins per web platform) ──────


def test_is_same_origin_request_localhost_vs_127_is_cross_origin():
    """Browsers treat ``localhost`` and ``127.0.0.1`` as distinct origins; the
    canonical comparison must not DNS-collapse them.
    """
    from main import _is_same_origin_request

    req = _build_request("127.0.0.1:8902", origin = "http://localhost:8902")
    assert _is_same_origin_request(req) is False


def test_is_same_origin_request_127_vs_localhost_is_cross_origin():
    from main import _is_same_origin_request
    req = _build_request("localhost:8902", origin = "http://127.0.0.1:8902")
    assert _is_same_origin_request(req) is False


# ── urlparse ValueError robustness ─────────────────────────────────


def test_is_same_origin_request_malformed_ipv6_bracket_is_cross_origin():
    """``urlparse`` raises ``ValueError('Invalid IPv6 URL')`` on unclosed
    brackets (CVE-2024-11168 hardening). The gate must swallow it and fall to
    cross-origin rather than 500 the SPA handler.
    """
    from main import _is_same_origin_request

    req = _build_request("127.0.0.1:8902", origin = "http://[malformed")
    assert _is_same_origin_request(req) is False


def test_is_same_origin_request_invalid_ipv6_address_is_cross_origin():
    """Bracketed but invalid IPv6 (e.g. ``[::g]``) also raises
    ``ValueError`` inside ``urlparse``."""
    from main import _is_same_origin_request

    req = _build_request("127.0.0.1:8902", origin = "http://[::g]:8902")
    assert _is_same_origin_request(req) is False


def test_is_same_origin_request_bracket_with_trailing_garbage_is_cross_origin():
    """Text after the closing bracket also raises inside ``urlparse``."""
    from main import _is_same_origin_request

    req = _build_request("127.0.0.1:8902", origin = "http://[2001:db8::1]extra:8902")
    assert _is_same_origin_request(req) is False


def test_is_same_origin_request_empty_origin_header_is_cross_origin():
    """Explicit empty ``Origin:`` is not a valid serialised origin and must not
    be conflated with a missing header; cross-origin, bootstrap withheld.
    """
    from main import _is_same_origin_request

    req = _build_request("127.0.0.1:8902", origin = "")
    assert _is_same_origin_request(req) is False
