# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import socket
import sys
import threading
from pathlib import Path

import pytest

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from core.inference import tools


def _serve_403(monkeypatch, headers):
    server = socket.create_server(("127.0.0.1", 0))
    server.settimeout(10)
    body = b"<html>Just a moment...</html>"
    head = "".join(f"{name}: {value}\r\n" for name, value in headers.items())
    response = (
        f"HTTP/1.1 403 Forbidden\r\nContent-Type: text/html\r\nContent-Length: {len(body)}\r\n{head}\r\n"
    ).encode() + body

    def respond():
        with server:
            while True:
                try:
                    conn, _ = server.accept()
                except OSError:
                    return
                with conn:
                    conn.recv(65536)
                    conn.sendall(response)

    threading.Thread(target = respond, daemon = True).start()
    monkeypatch.setattr(
        tools, "_validate_and_resolve_host", lambda host, port: (True, "", ["127.0.0.1"])
    )
    return f"http://example.com:{server.getsockname()[1]}/page"


@pytest.mark.parametrize(
    "headers",
    [
        {"Server": "cloudflare", "cf-mitigated": "challenge"},
        {"Server": "cloudflare"},
        {"Server": "DataDome", "X-DataDome": "protected"},
        {"Server": "AkamaiGHost"},
    ],
)
def test_the_model_is_told_a_bot_check_blocked_the_fetch(monkeypatch, headers):
    url = _serve_403(monkeypatch, headers)
    result = tools.execute_tool("web_search", {"url": url}, timeout = 10)
    assert result.startswith("Failed to fetch URL: HTTP 403 Forbidden")
    assert "bot protection" in result
    assert "fetching this site again will not help" in result


@pytest.mark.parametrize(
    "headers",
    [
        {"Server": "nginx"},
        # an origin's own 403 passed through Cloudflare carries a cache status
        {"Server": "cloudflare", "CF-Cache-Status": "DYNAMIC"},
    ],
)
def test_a_plain_403_keeps_its_message(monkeypatch, headers):
    url = _serve_403(monkeypatch, headers)
    result = tools.execute_tool("web_search", {"url": url}, timeout = 10)
    assert result == "Failed to fetch URL: HTTP 403 Forbidden"
