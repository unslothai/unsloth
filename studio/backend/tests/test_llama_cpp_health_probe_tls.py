# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Health probes reuse one TLS context instead of loading the CA bundle per probe."""

from __future__ import annotations

import http.server
import ssl
import sys
import threading
from pathlib import Path

import pytest

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

httpx = pytest.importorskip("httpx")
if not hasattr(httpx, "Client"):
    pytest.skip(
        reason = "test_compute_buffer.py and others replace sys.modules['httpx'] with a stub; this test needs the real client",
        allow_module_level = True,
    )

from core.inference.llama_cpp import LlamaCppBackend  # noqa: E402


class _Server(http.server.ThreadingHTTPServer):
    daemon_threads = True
    requests = 0
    connections = 0
    cookies_seen: list = []


class _Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def setup(self):
        self.server.connections += 1
        super().setup()

    def do_GET(self):
        srv = self.server
        srv.requests += 1
        srv.cookies_seen.append(self.headers.get("Cookie"))
        ok = srv.requests > 4
        body = b"ok" if ok else b"loading"
        self.send_response(200 if ok else 503)
        self.send_header("Set-Cookie", f"probe={srv.requests}")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


@pytest.fixture
def server():
    srv = _Server(("127.0.0.1", 0), _Handler)
    srv.cookies_seen = []
    threading.Thread(target = srv.serve_forever, daemon = True).start()
    yield srv
    srv.shutdown()
    srv.server_close()


def test_probes_share_one_tls_context(server, monkeypatch):
    built = []
    real_new = ssl.SSLContext.__new__

    def counting_new(cls, *a, **kw):
        built.append(cls)
        return real_new(cls, *a, **kw)

    monkeypatch.setattr(ssl.SSLContext, "__new__", counting_new)
    monkeypatch.setattr(LlamaCppBackend, "_health_listener_is_ours", lambda self, p, port: True)
    b = LlamaCppBackend()
    b._port = server.server_address[1]
    b._process = type("_Proc", (), {"returncode": None, "poll": lambda self: None})()
    b._stdout_thread = None
    b._stdout_lines = []

    assert b._wait_for_health(timeout = 10.0, interval = 0.01) is True
    assert server.requests == 5
    assert len(built) <= 1, f"{len(built)} TLS contexts for {server.requests} probes"
    # Unchanged from a bare httpx.get: a fresh connection and cookie jar per probe.
    assert server.connections == 5
    assert server.cookies_seen == [None] * 5
