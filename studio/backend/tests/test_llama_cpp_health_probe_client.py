# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""One health wait probes through one client, one fresh connection per probe, settings as httpx.get."""

from __future__ import annotations

import http.server
import sys
import threading
from pathlib import Path

import pytest

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

httpx = pytest.importorskip("httpx")
if not hasattr(httpx, "Limits"):
    pytest.skip("a sibling test installed an httpx stub", allow_module_level = True)

from core.inference import llama_cpp  # noqa: E402
from core.inference.llama_cpp import LlamaCppBackend  # noqa: E402


class _Server(http.server.ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, ready_after):
        self.ready_after = ready_after
        self.requests = 0
        self.connections = 0
        self.cookies_seen = []
        self.open_seen = []
        self._open = 0
        self._lock = threading.Lock()
        super().__init__(("127.0.0.1", 0), _Handler)

    def process_request(self, request, client_address):
        with self._lock:
            self.connections += 1
            self._open += 1
        super().process_request(request, client_address)

    def shutdown_request(self, request):
        # Runs once the handler sees the client close its end.
        super().shutdown_request(request)
        with self._lock:
            self._open -= 1


class _Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"  # keep-alive is the server's offer; the client must decline it

    def do_GET(self):
        srv = self.server
        srv.requests += 1
        srv.cookies_seen.append(self.headers.get("Cookie"))
        with srv._lock:
            srv.open_seen.append(srv._open)
        ok = srv.requests > srv.ready_after
        body = b'{"status":"ok"}' if ok else b'{"status":"loading"}'
        self.send_response(200 if ok else 503)
        self.send_header("Set-Cookie", f"probe={srv.requests}")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


class _Proc:
    returncode = None

    def poll(self):
        return None


@pytest.fixture
def server():
    srv = _Server(ready_after = 4)
    t = threading.Thread(target = srv.serve_forever, daemon = True)
    t.start()
    yield srv
    srv.shutdown()
    srv.server_close()


def _backend(port):
    b = LlamaCppBackend()
    b._port = port
    b._process = _Proc()
    b._stdout_thread = None
    b._stdout_lines = []
    return b


def test_one_client_per_wait_and_one_connection_per_probe(server, monkeypatch):
    built = []
    real_init = httpx.Client.__init__

    def counting_init(self, *a, **kw):
        built.append(kw)
        real_init(self, *a, **kw)

    # httpx.get builds its own Client per call, so count at the class.
    monkeypatch.setattr(httpx.Client, "__init__", counting_init)
    monkeypatch.setattr(LlamaCppBackend, "_health_listener_is_ours", lambda self, p, port: True)
    b = _backend(server.server_address[1])

    assert b._wait_for_health(timeout = 10.0, interval = 0.01) is True
    assert server.requests == 5
    # No keep-alive across probes: the server sees a new connection for every probe.
    assert server.connections == server.requests
    # And the previous probe's connection is already closed when the next one arrives,
    # so a probe never holds one of llama-server's HTTP threads between probes.
    assert server.open_seen == [1] * 5
    # A Set-Cookie on one probe is never replayed on the next, as with httpx.get.
    assert server.cookies_seen == [None] * 5
    assert len(built) == 1, f"expected one client per wait, built {len(built)}"


def test_probe_client_settings_match_a_top_level_get():
    """Same timeout, proxy handling, redirects, headers and TLS as httpx.get(..., timeout = 2.0, trust_env = False)."""
    probe = llama_cpp._health_probe_client()
    ref = httpx.Client(timeout = 2.0, trust_env = False)
    try:
        assert probe.timeout == ref.timeout
        assert probe.trust_env is ref.trust_env is False
        assert probe.follow_redirects == ref.follow_redirects
        assert dict(probe.headers) == dict(ref.headers)
        assert probe._mounts == ref._mounts == {}
        pool = probe._transport._pool
        ref_pool = ref._transport._pool
        assert pool._max_keepalive_connections == 0
        assert pool._max_connections == ref_pool._max_connections
        assert pool._keepalive_expiry == ref_pool._keepalive_expiry
        assert pool._http1 == ref_pool._http1 and pool._http2 == ref_pool._http2
        assert pool._ssl_context.verify_mode == ref_pool._ssl_context.verify_mode
    finally:
        probe.close()
        ref.close()


def test_the_client_is_closed_on_every_exit(server, monkeypatch):
    closed = []
    real_factory = llama_cpp._health_probe_client

    def tracking_factory():
        c = real_factory()
        real_close = c.close
        c.close = lambda: (closed.append(True), real_close())
        return c

    monkeypatch.setattr(llama_cpp, "_health_probe_client", tracking_factory)
    monkeypatch.setattr(LlamaCppBackend, "_health_listener_is_ours", lambda self, p, port: False)
    b = _backend(server.server_address[1])
    assert b._wait_for_health(timeout = 10.0, interval = 0.01) is False
    assert closed == [True]

    # A wait that ends before its first probe builds no client at all.
    b2 = _backend(server.server_address[1])
    b2._shutting_down = True
    before = server.requests
    assert b2._wait_for_health(timeout = 10.0, interval = 0.01) is False
    assert server.requests == before and closed == [True]
