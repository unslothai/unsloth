# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Unit and regression tests for CORS origin policies in Unsloth Studio.

Covers Issue #9880: CORS support for external loopback clients in desktop mode,
UNSLOTH_CORS_ORIGINS environment overrides, and origin isolation boundaries.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from starlette.datastructures import Headers

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from main import RemoteAccessCORSMiddleware
from utils.host_policy import (
    _LOOPBACK_ORIGIN_REGEX,
    _TAURI_CORS_ORIGINS,
    cors_origin_regex_for_mode,
    cors_origins_for_mode,
)


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
    "api_only,secure,expected",
    [
        (True, False, _LOOPBACK_ORIGIN_REGEX),
        (True, True, None),
        (False, False, None),
        (False, True, None),
    ],
)
def test_cors_origin_regex_for_mode(api_only, secure, expected):
    regex = cors_origin_regex_for_mode(api_only = api_only, secure = secure)
    assert regex == expected


@pytest.mark.parametrize(
    "origin,should_allow",
    [
        # Internal Tauri schemes
        ("tauri://localhost", True),
        ("http://tauri.localhost", True),
        ("http://localhost:5173", True),
        # External loopback browser origins (Issue #9880)
        ("http://localhost:3000", True),
        ("http://localhost:8080", True),
        ("http://127.0.0.1:3000", True),
        ("http://127.0.0.1:8888", True),
        ("http://[::1]:3000", True),
        ("https://localhost:8443", True),
        ("https://127.0.0.1:8443", True),
        # Adversarial / remote origins (must stay blocked in local desktop mode)
        ("http://malicious-site.com", False),
        ("https://evil.org", False),
        ("http://localhost.attacker.com", False),
        ("http://attackerlocalhost.com", False),
        ("http://127.0.0.1.attacker.com", False),
    ],
)
def test_desktop_cors_preflight_origins(origin, should_allow):
    state = SimpleNamespace(cloudflare_url = None)
    middleware = RemoteAccessCORSMiddleware(
        lambda *_: None,
        remote_access_state = state,
        allow_origins = list(_TAURI_CORS_ORIGINS),
        allow_origin_regex = _LOOPBACK_ORIGIN_REGEX,
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
