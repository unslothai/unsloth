# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from types import SimpleNamespace

import pytest

from utils.origin_policy import canonical_origin, mcp_origin_allowed, origin_of

TUNNEL = "https://abc-def.trycloudflare.com"


def test_main_reexports_the_moved_helpers():
    import main
    assert main._canonical_origin is canonical_origin
    assert main._origin_of is origin_of


@pytest.mark.parametrize(
    "value,expected",
    [
        ("http://localhost", ("http", "localhost", 80)),
        ("https://Example.COM:443", ("https", "example.com", 443)),
        ("http://[::1]:8888", ("http", "::1", 8888)),
        ("http://[::1]", ("http", "::1", 80)),
        ("http://user:pw@127.0.0.1:8888", ("http", "127.0.0.1", 8888)),
        ("tauri://localhost", ("tauri", "localhost", 0)),
        ("null", None),
        ("", None),
        ("http://[::1", None),
        ("http://host:port", None),
    ],
)
def test_origin_canonicalisation(value, expected):
    assert origin_of(value) == expected


def _allowed(
    origin,
    *,
    host = "127.0.0.1:8888",
    scheme = "http",
    **state,
):
    app_state = SimpleNamespace(**{"server_port": 8888, "cloudflare_url": None, **state})
    return mcp_origin_allowed(
        origin, request_scheme = scheme, request_netloc = host, app_state = app_state
    )


@pytest.mark.parametrize(
    "origin,allowed",
    [
        *(
            (o, True)
            for o in ["http://localhost:8888", "http://127.0.0.1:8888", "http://[::1]:8888"]
        ),
        *(
            (o, True)
            for o in ["HTTP://LOCALHOST:8888", "tauri://localhost", "http://tauri.localhost"]
        ),
        *(
            (o, False)
            for o in ["https://evil.example", "http://localhost:3000", "http://localhost:5173"]
        ),
        *((o, False) for o in ["http://localhost", "null", "", "not a url", "http://[::1", TUNNEL]),
    ],
)
def test_origins_against_the_default_loopback_host(origin, allowed):
    assert _allowed(origin) is allowed


@pytest.mark.parametrize(
    "origin,kwargs,allowed",
    [
        # Loopback at the request's own port passes without a published port.
        ("http://localhost:9000", {"host": "127.0.0.1:9000", "server_port": None}, True),
        ("http://localhost:9001", {"host": "127.0.0.1:9000", "server_port": None}, False),
        ("http://[::1]:8888", {"host": "[::1]:8888"}, True),
        ("http://localhost:8888", {"host": "[::1]:8888"}, True),
        ("http://[::1]:8889", {"host": "[::1]:8888"}, False),
        # A rebound name matching the Host header is still refused.
        ("http://evil.example:8888", {"host": "evil.example:8888"}, False),
        ("http://192.168.1.20:8888", {"host": "192.168.1.20:8888"}, True),
        ("http://[fe80::1]:8888", {"host": "[fe80::1]:8888"}, True),
        ("http://192.168.1.21:8888", {"host": "192.168.1.20:8888"}, False),
        ("http://192.168.1.20", {"host": "192.168.1.20:80"}, True),
        ("https://192.168.1.20:443", {"host": "192.168.1.20", "scheme": "https"}, True),
        (TUNNEL, {"cloudflare_url": TUNNEL + "/"}, True),
        (TUNNEL, {"host": "abc-def.trycloudflare.com", "cloudflare_url": TUNNEL}, True),
    ],
)
def test_origins_against_other_hosts(origin, kwargs, allowed):
    assert _allowed(origin, **kwargs) is allowed


def test_cors_origins_env_is_honoured_but_never_a_wildcard(monkeypatch):
    monkeypatch.setenv("UNSLOTH_CORS_ORIGINS", "*, https://studio.example.org")
    assert _allowed("https://studio.example.org")
    assert not _allowed("https://evil.example")
