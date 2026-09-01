# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Unit and regression tests for CORS origin policies in Unsloth Studio.

Covers Issue #9880: CORS support for external loopback clients in desktop mode,
UNSLOTH_CORS_ORIGINS environment overrides, and origin isolation boundaries.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from starlette.datastructures import Headers
from starlette.middleware.cors import CORSMiddleware

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from utils.host_policy import (
    _LOOPBACK_ORIGIN_REGEX,
    _TAURI_CORS_ORIGINS,
    cors_origin_regex_for_mode,
    cors_origins_for_mode,
)


class DummyRemoteAccessCORSMiddleware(CORSMiddleware):
    """Mirror RemoteAccessCORSMiddleware in main.py for isolated testing."""

    def __init__(self, cors_app, *, remote_access_state, **kwargs):
        self.remote_access_state = remote_access_state
        super().__init__(cors_app, **kwargs)

    def is_allowed_origin(self, origin: str) -> bool:
        return bool(
            getattr(self.remote_access_state, "cloudflare_url", None)
        ) or super().is_allowed_origin(origin)


@pytest.mark.parametrize(
    "api_only,secure,expected",
    [
        (False, False, ["*"]),
        (False, True, ["*"]),
        (True, True, ["*"]),
    ],
)
def test_cors_origins_for_mode_wildcard(api_only, secure, expected):
    origins = cors_origins_for_mode(api_only = api_only, secure = secure)
    assert origins == expected


def test_cors_origins_for_mode_desktop_default():
    origins = cors_origins_for_mode(api_only = True, secure = False)
    assert origins == list(_TAURI_CORS_ORIGINS)


def test_cors_origins_for_mode_env_override(monkeypatch):
    monkeypatch.setenv("UNSLOTH_CORS_ORIGINS", "https://foo.example, http://localhost:9999")
    origins = cors_origins_for_mode(api_only = True, secure = False)
    assert origins == list(_TAURI_CORS_ORIGINS) + ["https://foo.example", "http://localhost:9999"]

    origins_wildcard = cors_origins_for_mode(api_only = False, secure = False)
    assert origins_wildcard == ["https://foo.example", "http://localhost:9999"]


@pytest.mark.parametrize(
    "api_only,secure",
    [
        (True, False),
        (True, True),
        (False, False),
        (False, True),
    ],
)
def test_cors_origin_regex_for_mode_default_none(api_only, secure):
    # Regex matching is disabled by default across all modes to prevent unauthorized
    # credentialed requests from arbitrary local ports in desktop api-only mode.
    regex = cors_origin_regex_for_mode(api_only = api_only, secure = secure)
    assert regex is None


def test_cors_origin_regex_for_mode_opt_in(monkeypatch):
    monkeypatch.setenv("UNSLOTH_CORS_ALLOW_LOOPBACK", "1")
    assert cors_origin_regex_for_mode(api_only = True, secure = False) == _LOOPBACK_ORIGIN_REGEX

    monkeypatch.setenv("UNSLOTH_CORS_ORIGIN_REGEX", r"^https?://specific\.local$")
    assert cors_origin_regex_for_mode(api_only = True, secure = False) == r"^https?://specific\.local$"


@pytest.mark.parametrize(
    "origin,should_allow",
    [
        # Internal Tauri schemes (allowed by default)
        ("tauri://localhost", True),
        ("http://tauri.localhost", True),
        ("http://localhost:5173", True),
        # External loopback origins (must be blocked by default without opt-in)
        ("http://localhost:3000", False),
        ("http://localhost:8080", False),
        ("http://127.0.0.1:3000", False),
        ("http://127.0.0.1:8888", False),
        ("http://[::1]:3000", False),
        ("https://localhost:8443", False),
        ("https://127.0.0.1:8443", False),
        # Remote origins (blocked)
        ("http://malicious-site.com", False),
        ("https://evil.org", False),
    ],
)
def test_desktop_cors_default_locked_down(origin, should_allow):
    state = SimpleNamespace(cloudflare_url = None)
    middleware = DummyRemoteAccessCORSMiddleware(
        lambda *_: None,
        remote_access_state = state,
        allow_origins = cors_origins_for_mode(api_only = True, secure = False),
        allow_origin_regex = cors_origin_regex_for_mode(api_only = True, secure = False),
        allow_credentials = True,
        allow_methods = ["*"],
        allow_headers = ["*"],
    )

    request = Headers(
        {
            "origin": origin,
            "access-control-request-method": "POST",
            "access-control-request-headers": "authorization,content-type",
        }
    )
    response = middleware.preflight_response(request)

    if should_allow:
        assert response.status_code == 200
        assert response.headers.get("access-control-allow-origin") == origin
    else:
        assert response.status_code == 400
        assert "access-control-allow-origin" not in response.headers


def test_desktop_cors_opt_in_origins(monkeypatch):
    monkeypatch.setenv("UNSLOTH_CORS_ORIGINS", "http://localhost:3000, http://127.0.0.1:8080")
    state = SimpleNamespace(cloudflare_url = None)
    middleware = DummyRemoteAccessCORSMiddleware(
        lambda *_: None,
        remote_access_state = state,
        allow_origins = cors_origins_for_mode(api_only = True, secure = False),
        allow_origin_regex = cors_origin_regex_for_mode(api_only = True, secure = False),
        allow_credentials = True,
        allow_methods = ["*"],
        allow_headers = ["*"],
    )

    # Allowed opt-in origins
    for origin in ("http://localhost:3000", "http://127.0.0.1:8080", "tauri://localhost"):
        req = Headers(
            {
                "origin": origin,
                "access-control-request-method": "POST",
                "access-control-request-headers": "authorization,content-type",
            }
        )
        resp = middleware.preflight_response(req)
        assert resp.status_code == 200
        assert resp.headers.get("access-control-allow-origin") == origin

    # Still rejected origins
    for origin in ("http://localhost:9000", "http://evil.com"):
        req = Headers(
            {
                "origin": origin,
                "access-control-request-method": "POST",
                "access-control-request-headers": "authorization,content-type",
            }
        )
        resp = middleware.preflight_response(req)
        assert resp.status_code == 400
        assert "access-control-allow-origin" not in resp.headers


@pytest.mark.parametrize(
    "origin,should_allow",
    [
        ("tauri://localhost", True),
        ("http://tauri.localhost", True),
        ("http://localhost:5173", True),
        ("http://localhost:3000", True),
        ("http://localhost:8080", True),
        ("http://127.0.0.1:3000", True),
        ("http://127.0.0.1:8888", True),
        ("http://[::1]:3000", True),
        ("https://localhost:8443", True),
        ("https://127.0.0.1:8443", True),
        ("http://malicious-site.com", False),
        ("https://evil.org", False),
        ("http://localhost.attacker.com", False),
        ("http://attackerlocalhost.com", False),
        ("http://127.0.0.1.attacker.com", False),
    ],
)
def test_desktop_cors_loopback_flag_opt_in(monkeypatch, origin, should_allow):
    monkeypatch.setenv("UNSLOTH_CORS_ALLOW_LOOPBACK", "1")
    state = SimpleNamespace(cloudflare_url = None)
    middleware = DummyRemoteAccessCORSMiddleware(
        lambda *_: None,
        remote_access_state = state,
        allow_origins = cors_origins_for_mode(api_only = True, secure = False),
        allow_origin_regex = cors_origin_regex_for_mode(api_only = True, secure = False),
        allow_credentials = True,
        allow_methods = ["*"],
        allow_headers = ["*"],
    )

    request = Headers(
        {
            "origin": origin,
            "access-control-request-method": "POST",
            "access-control-request-headers": "authorization,content-type",
        }
    )
    response = middleware.preflight_response(request)

    if should_allow:
        assert response.status_code == 200
        assert response.headers.get("access-control-allow-origin") == origin
    else:
        assert response.status_code == 400
        assert "access-control-allow-origin" not in response.headers
