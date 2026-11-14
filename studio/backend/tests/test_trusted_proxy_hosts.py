# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""UNSLOTH_STUDIO_PROXY_HOSTS: the desktop web UI behind the user's own reverse proxy (#17420)."""

import asyncio
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI

from utils.client_ip import request_via_trusted_proxy, trusted_proxy_hosts

_ENV = "UNSLOTH_STUDIO_PROXY_HOSTS"


def _scope(
    headers,
    client = "127.0.0.1",
    server = "127.0.0.1",
):
    return {
        "type": "http",
        "client": (client, 50000) if client else None,
        "server": (server, 8888) if server else None,
        "headers": [(k.lower().encode("latin-1"), v.encode("latin-1")) for k, v in headers],
    }


@pytest.fixture
def allow_studio(monkeypatch):
    monkeypatch.setenv(_ENV, "studio.example.com")


def test_the_setting_parses_names_and_drops_everything_else(monkeypatch):
    monkeypatch.setenv(
        _ENV,
        "Studio.Example.com, https://ai.example.org/ other.example.net:443 * localhost 127.0.0.1 [::1] bad_name",
    )
    assert trusted_proxy_hosts() == {"studio.example.com", "ai.example.org", "other.example.net"}


def test_nothing_is_admitted_without_the_setting(monkeypatch):
    monkeypatch.delenv(_ENV, raising = False)
    assert not request_via_trusted_proxy(_scope([("host", "studio.example.com")]))
    assert not request_via_trusted_proxy(
        _scope([("host", "127.0.0.1:8888"), ("x-forwarded-host", "studio.example.com")])
    )


@pytest.mark.parametrize(
    "headers",
    [
        # cloudflared's default: the public hostname as Host
        [
            ("host", "studio.example.com"),
            ("cf-connecting-ip", "203.0.113.7"),
            ("x-forwarded-proto", "https"),
        ],
        # a proxy that keeps the origin Host and names the public one in X-Forwarded-Host
        [
            ("host", "127.0.0.1:8888"),
            ("x-forwarded-host", "studio.example.com"),
            ("x-forwarded-proto", "https"),
        ],
        [("host", "localhost:8888"), ("x-forwarded-host", "Studio.Example.com:443")],
        [("host", "studio.example.com"), ("x-forwarded-host", "studio.example.com")],
    ],
)
def test_a_local_proxy_for_an_allowlisted_name_is_admitted(allow_studio, headers):
    assert request_via_trusted_proxy(_scope(headers))


@pytest.mark.parametrize(
    "headers",
    [
        [("host", "evil.example")],
        # DNS rebinding: the page's own name is the Host, a header it sets cannot borrow the allowlist
        [("host", "evil.example:8888"), ("x-forwarded-host", "studio.example.com")],
        [("host", "studio.example.com"), ("x-forwarded-host", "evil.example")],
        [("host", "127.0.0.1:8888"), ("x-forwarded-host", "studio.example.com, evil.example")],
        [("host", "studio.example.com.evil.example")],
        [("host", "studio.example.com"), ("host", "studio.example.com")],
        [],
        # a plain loopback request is the existing direct-browser gate's to judge
        [("host", "127.0.0.1:8888")],
    ],
)
def test_other_names_are_refused(allow_studio, headers):
    assert not request_via_trusted_proxy(_scope(headers))


@pytest.mark.parametrize(
    "client, server",
    [
        ("192.168.1.20", "127.0.0.1"),
        ("127.0.0.1", "192.168.1.5"),
        (None, "127.0.0.1"),
        ("127.0.0.1", None),
    ],
)
def test_only_the_loopback_listener_honours_the_allowlist(allow_studio, client, server):
    assert not request_via_trusted_proxy(_scope([("host", "studio.example.com")], client, server))


def test_trust_forwarded_keeps_the_allowlist_working(allow_studio, monkeypatch):
    # uvicorn's proxy_headers rewrites the loopback peer to the X-Forwarded-For visitor
    monkeypatch.delenv("UNSLOTH_STUDIO_TRUST_FORWARDED", raising = False)
    proxied = _scope([("host", "studio.example.com")], client = "203.0.113.7")
    assert not request_via_trusted_proxy(proxied)
    monkeypatch.setenv("UNSLOTH_STUDIO_TRUST_FORWARDED", "1")
    assert request_via_trusted_proxy(proxied)
    assert not request_via_trusted_proxy(_scope([("host", "evil.example")], client = "203.0.113.7"))
    assert not request_via_trusted_proxy(
        _scope([("host", "studio.example.com")], client = "203.0.113.7", server = "192.168.1.5")
    )


def _desktop_frontend(tmp_path):
    import main

    dist = tmp_path / "dist"
    (dist / "assets").mkdir(parents = True)
    (dist / "index.html").write_text(
        "<html><head></head><body>studio</body></html>", encoding = "utf-8"
    )
    (dist / "assets" / "app.js").write_text("console.log(1)", encoding = "utf-8")
    app = FastAPI()
    assert main.setup_frontend(app, dist, tunnel_only = True)
    return app


def _get(app, path, headers):
    async def run():
        transport = httpx.ASGITransport(app = app, client = ("127.0.0.1", 50000))
        async with httpx.AsyncClient(
            transport = transport, base_url = "http://127.0.0.1:8888"
        ) as client:
            return await client.get(path, headers = headers)

    return asyncio.run(run())


@pytest.mark.parametrize("path", ["/", "/chat", "/assets/app.js"])
def test_the_desktop_web_ui_is_served_through_an_allowlisted_proxy(tmp_path, monkeypatch, path):
    app = _desktop_frontend(tmp_path)
    proxied = {
        "host": "studio.example.com",
        "x-forwarded-proto": "https",
        "cf-connecting-ip": "203.0.113.7",
    }

    monkeypatch.delenv(_ENV, raising = False)
    assert _get(app, path, proxied).status_code == 404

    monkeypatch.setenv(_ENV, "studio.example.com")
    assert _get(app, path, proxied).status_code == 200
    assert _get(app, path, {**proxied, "host": "evil.example"}).status_code == 404
    # a direct local browser keeps working either way
    assert _get(app, path, {}).status_code == 200


def test_a_proxied_request_still_never_counts_as_local_for_the_bootstrap_password(monkeypatch):
    import main
    monkeypatch.setenv(_ENV, "studio.example.com")
    for headers in (
        {"host": "studio.example.com"},
        {"host": "127.0.0.1", "x-forwarded-host": "studio.example.com"},
    ):
        request = SimpleNamespace(client = SimpleNamespace(host = "127.0.0.1"), headers = headers)
        assert main._is_local_bootstrap_request(request) is False, headers
