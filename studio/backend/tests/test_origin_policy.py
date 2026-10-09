# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from types import SimpleNamespace

import pytest

from utils.origin_policy import canonical_origin, mcp_origin_allowed, origin_of


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
    "origin",
    [
        "http://localhost:8888",
        "http://127.0.0.1:8888",
        "http://[::1]:8888",
        "HTTP://LOCALHOST:8888",
        "tauri://localhost",
        "http://tauri.localhost",
    ],
)
def test_first_party_origins_pass(origin):
    assert _allowed(origin)


@pytest.mark.parametrize(
    "origin",
    [
        "https://evil.example",
        "http://localhost:3000",
        "http://localhost:5173",
        "http://localhost",
        "null",
        "",
        "not a url",
        "http://[::1",
    ],
)
def test_foreign_origins_are_refused(origin):
    assert not _allowed(origin)


def test_loopback_at_the_requests_own_port_passes_without_a_published_port():
    assert _allowed("http://localhost:9000", host = "127.0.0.1:9000", server_port = None)
    assert not _allowed("http://localhost:9001", host = "127.0.0.1:9000", server_port = None)


def test_ipv6_loopback_hosts_and_origins():
    assert _allowed("http://[::1]:8888", host = "[::1]:8888")
    assert _allowed("http://localhost:8888", host = "[::1]:8888")
    assert not _allowed("http://[::1]:8889", host = "[::1]:8888")


def test_a_rebound_name_is_refused_even_though_it_matches_the_host():
    assert not _allowed("http://evil.example:8888", host = "evil.example:8888")


def test_a_literal_ip_page_on_the_lan_passes():
    assert _allowed("http://192.168.1.20:8888", host = "192.168.1.20:8888")
    assert _allowed("http://[fe80::1]:8888", host = "[fe80::1]:8888")
    assert not _allowed("http://192.168.1.21:8888", host = "192.168.1.20:8888")


def test_default_ports_canonicalise():
    assert _allowed("http://192.168.1.20", host = "192.168.1.20:80")
    assert _allowed("https://192.168.1.20:443", host = "192.168.1.20", scheme = "https")


def test_the_tunnel_origin_comes_from_app_state():
    tunnel = "https://abc-def.trycloudflare.com"
    assert not _allowed(tunnel)
    assert _allowed(tunnel, cloudflare_url = tunnel + "/")
    assert _allowed(tunnel, host = "abc-def.trycloudflare.com", cloudflare_url = tunnel)


def test_cors_origins_env_is_honoured_but_never_a_wildcard(monkeypatch):
    monkeypatch.setenv("UNSLOTH_CORS_ORIGINS", "*, https://studio.example.org")
    assert _allowed("https://studio.example.org")
    assert not _allowed("https://evil.example")
