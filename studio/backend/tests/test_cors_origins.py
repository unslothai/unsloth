# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""CORS origin policy: the desktop lockdown, its two opt-ins, and issue #9880."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from starlette.datastructures import Headers

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from utils.host_policy import (
    _LOOPBACK_ORIGIN_REGEX,
    _TAURI_CORS_ORIGINS,
    cors_origin_regex_for_mode,
    cors_origins_for_mode,
)


def _middleware(
    *,
    api_only = True,
    secure = False,
    cloudflare_url = None,
):
    """The class main.py mounts, not a stand-in: a copy keeps passing once the two drift."""
    from main import RemoteAccessCORSMiddleware
    return RemoteAccessCORSMiddleware(
        lambda *_: None,
        remote_access_state = SimpleNamespace(cloudflare_url = cloudflare_url),
        allow_origins = cors_origins_for_mode(api_only = api_only, secure = secure),
        allow_origin_regex = cors_origin_regex_for_mode(api_only = api_only, secure = secure),
        allow_credentials = True,
        allow_methods = ["*"],
        allow_headers = ["*"],
    )


def _preflight(middleware, origin):
    return middleware.preflight_response(
        Headers(
            {
                "origin": origin,
                "access-control-request-method": "POST",
                "access-control-request-headers": "authorization,content-type",
            }
        )
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


@pytest.mark.parametrize("api_only,secure", [(False, False), (False, True), (True, True)])
def test_cors_origins_env_never_narrows_any_origin_modes(monkeypatch, api_only, secure):
    # is_allowed_origin only waves origins through while a Cloudflare URL is published, so
    # an env list replacing ["*"] would 400 tauri://localhost the moment the tunnel drops.
    monkeypatch.setenv("UNSLOTH_CORS_ORIGINS", "http://localhost:8080")
    assert cors_origins_for_mode(api_only = api_only, secure = secure) == ["*"]

    middleware = _middleware(api_only = api_only, secure = secure)
    assert _preflight(middleware, "tauri://localhost").status_code == 200


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
    regex = cors_origin_regex_for_mode(api_only = api_only, secure = secure)
    assert regex is None


def test_cors_origin_regex_for_mode_opt_in(monkeypatch):
    monkeypatch.setenv("UNSLOTH_CORS_ALLOW_LOOPBACK", "1")
    assert cors_origin_regex_for_mode(api_only = True, secure = False) == _LOOPBACK_ORIGIN_REGEX

    monkeypatch.setenv("UNSLOTH_CORS_ORIGIN_REGEX", r"^https?://specific\.local$")
    assert cors_origin_regex_for_mode(api_only = True, secure = False) == r"^https?://specific\.local$"


@pytest.mark.parametrize("api_only,secure", [(False, False), (False, True), (True, True)])
def test_cors_origin_regex_only_applies_to_the_desktop_lockdown(monkeypatch, api_only, secure):
    monkeypatch.setenv("UNSLOTH_CORS_ALLOW_LOOPBACK", "1")
    monkeypatch.setenv("UNSLOTH_CORS_ORIGIN_REGEX", r"^https?://specific\.local$")
    assert cors_origin_regex_for_mode(api_only = api_only, secure = secure) is None


def test_main_passes_the_origin_regex_to_the_mounted_middleware():
    # The policy helpers are only worth anything if main.py hands them to add_middleware.
    src = (_BACKEND / "main.py").read_text(encoding = "utf-8")
    assert "allow_origin_regex = _cors_origin_regex" in src


@pytest.mark.parametrize(
    "origin,should_allow",
    [
        ("tauri://localhost", True),
        ("http://tauri.localhost", True),
        ("http://localhost:5173", True),
        ("http://localhost:3000", False),
        ("http://localhost:8080", False),
        ("http://127.0.0.1:3000", False),
        ("http://127.0.0.1:8888", False),
        ("http://[::1]:3000", False),
        ("https://localhost:8443", False),
        ("https://127.0.0.1:8443", False),
        ("http://malicious-site.com", False),
        ("https://evil.org", False),
    ],
)
def test_desktop_cors_default_locked_down(origin, should_allow):
    response = _preflight(_middleware(), origin)

    if should_allow:
        assert response.status_code == 200
        assert response.headers.get("access-control-allow-origin") == origin
    else:
        assert response.status_code == 400
        assert "access-control-allow-origin" not in response.headers


def test_desktop_cors_opt_in_origins(monkeypatch):
    monkeypatch.setenv("UNSLOTH_CORS_ORIGINS", "http://localhost:3000, http://127.0.0.1:8080")
    middleware = _middleware()

    for origin in ("http://localhost:3000", "http://127.0.0.1:8080", "tauri://localhost"):
        resp = _preflight(middleware, origin)
        assert resp.status_code == 200
        assert resp.headers.get("access-control-allow-origin") == origin

    # The list is an allowlist, not a loopback pass: a port not on it stays blocked.
    for origin in ("http://localhost:9000", "http://evil.com"):
        resp = _preflight(middleware, origin)
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
    response = _preflight(_middleware(), origin)

    if should_allow:
        assert response.status_code == 200
        assert response.headers.get("access-control-allow-origin") == origin
    else:
        assert response.status_code == 400
        assert "access-control-allow-origin" not in response.headers
