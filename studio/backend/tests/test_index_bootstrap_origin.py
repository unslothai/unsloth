# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Regression coverage for the bootstrap-pw cross-origin leak (PR 5739).

``_is_same_origin_request`` gates ``_inject_bootstrap`` so the seeded
admin password only ships to same-origin callers. See the ``CORS: GET /``
audit in tests/studio/studio_api_smoke.py.
"""

import os
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


def _build_request(host: str, origin: str | None, scheme: str = "http") -> MagicMock:
    request = MagicMock()
    request.url.scheme = scheme
    request.url.netloc = host
    request.headers = {"origin": origin} if origin is not None else {}
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
    """RFC 6454 strips default ports on Origin; Starlette's netloc may
    still carry ``:443``. Bare string-equality misreads this as cross-
    origin and refuses to inject the bootstrap on legitimate browser
    navigation. Canonicalise both sides before comparing.
    """
    from main import _is_same_origin_request

    req = _build_request(
        "example.com:443", origin = "https://example.com", scheme = "https"
    )
    assert _is_same_origin_request(req) is True


def test_is_same_origin_request_http_default_port_stripped_on_origin():
    from main import _is_same_origin_request

    req = _build_request("example.com:80", origin = "http://example.com")
    assert _is_same_origin_request(req) is True


def test_is_same_origin_request_default_port_present_on_origin():
    """Mirror of the above: Origin carries the default port but the
    listener's netloc doesn't. Same-origin.
    """
    from main import _is_same_origin_request

    req = _build_request(
        "example.com", origin = "https://example.com:443", scheme = "https"
    )
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
    """Sandboxed iframes / file:// pages send ``Origin: null``; treat
    as cross-origin so the bootstrap is never injected for them.
    """
    from main import _is_same_origin_request

    req = _build_request("example.com", origin = "null")
    assert _is_same_origin_request(req) is False


def test_is_same_origin_request_unparseable_origin_is_cross_origin():
    """Garbage values that don't carry a host are treated as cross-
    origin (safer than throwing) so a malformed header can't leak the
    bootstrap.
    """
    from main import _is_same_origin_request

    req = _build_request("example.com", origin = "not-a-url")
    assert _is_same_origin_request(req) is False


def test_is_same_origin_request_userinfo_in_netloc_ignored():
    """``user:pass@host:port`` netlocs (rare but valid per RFC 3986)
    must compare equal to the credentials-less Origin.
    """
    from main import _is_same_origin_request

    req = _build_request("user:pass@example.com:80", origin = "http://example.com")
    assert _is_same_origin_request(req) is True


def test_is_same_origin_request_explicit_non_default_port_still_mismatch():
    """Canonicalisation does NOT collapse non-default ports to default."""
    from main import _is_same_origin_request

    req = _build_request(
        "example.com", origin = "https://example.com:9999", scheme = "https"
    )
    assert _is_same_origin_request(req) is False
