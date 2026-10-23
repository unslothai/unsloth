# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Fake-IP DNS (Clash/mihomo, sing-box, Surge TUN) answers every name from 198.18.0.0/15 (#13985)."""

from __future__ import annotations

import http.server
import socket
import sys
import threading
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from core.inference import tools  # noqa: E402

_ANSWERS = {
    "github.com": ["198.18.0.24"],
    "docs.unsloth.ai": ["198.19.255.254"],
    "mixed.example.com": ["198.18.0.7", "192.168.1.1"],
    "printer.lan": ["198.18.0.9"],
    "router": ["198.18.0.10"],
    "lan.example.com": ["192.168.1.1"],
    # The resolver reads an all-numeric host as an IPv4 literal (198.18.0.24 here).
    "3323068440": ["198.18.0.24"],
}


def _dns(monkeypatch, unknown):
    """Answers from ``_ANSWERS``; any other name (the resolver probe) gets ``unknown``, or NXDOMAIN when None."""
    monkeypatch.setattr(tools, "_fake_ip_probe", [None, False])
    real = socket.getaddrinfo

    def getaddrinfo(host, port, *args, **kwargs):
        if host == "127.0.0.1":
            return real(host, port, *args, **kwargs)
        answers = _ANSWERS.get(host, unknown)
        if answers is None:
            raise socket.gaierror(socket.EAI_NONAME, "Name or service not known")
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, port)) for ip in answers]

    monkeypatch.setattr(socket, "getaddrinfo", getaddrinfo)


@pytest.fixture
def fake_ip_dns(monkeypatch):
    _dns(monkeypatch, ["198.18.0.99"])


@pytest.fixture
def real_dns(monkeypatch):
    _dns(monkeypatch, None)


@pytest.mark.parametrize(
    "host, pinned",
    [("github.com", ["198.18.0.24"]), ("docs.unsloth.ai", ["198.19.255.254"])],
)
def test_public_name_with_a_fake_ip_answer_is_allowed(fake_ip_dns, host, pinned):
    assert tools._validate_and_resolve_host(host, 443) == (True, "", pinned)


@pytest.mark.parametrize("unknown", [None, ["93.184.215.14"]])
def test_fake_ip_answers_need_a_fake_ip_resolver(monkeypatch, unknown):
    # Real DNS (NXDOMAIN, or an ISP's NXDOMAIN redirect) on the probe: a public name pointing into the range is refused.
    _dns(monkeypatch, unknown)
    ok, reason, _ = tools._validate_and_resolve_host("github.com", 443)
    assert not ok and reason == "Blocked: refusing to fetch non-public address 198.18.0.24."


@pytest.mark.parametrize(
    "host",
    [
        "198.18.0.24",
        "198.19.0.1",
        "3323068440",
        "printer.lan",
        "router",
        "mixed.example.com",
        "lan.example.com",
    ],
)
def test_literals_local_names_and_private_answers_stay_refused(fake_ip_dns, host):
    ok, reason, pinned = tools._validate_and_resolve_host(host, 443)
    assert (ok, pinned) == (False, [])
    assert reason.startswith("Blocked: refusing to fetch non-public address"), reason


def test_fetch_reaches_the_tunnel_with_the_name_in_host(fake_ip_dns, monkeypatch):
    for name in (
        "http_proxy",
        "https_proxy",
        "all_proxy",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
    ):
        monkeypatch.delenv(name, raising = False)
    seen = {}

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            seen["host"] = self.headers["Host"]
            body = b"<html><body><p>through the tunnel</p></body></html>"
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    port = server.server_address[1]
    threading.Thread(target = server.serve_forever, daemon = True).start()
    real_connect = socket.create_connection

    # The TUN device: a connection to the fake address lands on the real server.
    def tunnel(address, *args, **kwargs):
        if address == ("198.18.0.24", port):
            address = ("127.0.0.1", port)
        return real_connect(address, *args, **kwargs)

    monkeypatch.setattr(socket, "create_connection", tunnel)
    try:
        err, text, _ = tools._fetch_url_raw(f"http://github.com:{port}/")
    finally:
        server.shutdown()
        server.server_close()
    assert err is None or err == "", err
    assert "through the tunnel" in text
    assert seen["host"] == f"github.com:{port}"
