# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import http.server
import socket
import subprocess
import sys
import threading

import httpx
import pytest

pytest.importorskip("psutil")

from core.inference.stt_mtmd_sidecar import MtmdSttSidecar
from core.rag import config
from core.rag.embed_llama_server import LlamaServerBackend, _embed_host
from core.systemone.native_worker import NativeClefAgent, NativeError

_CHILD_SERVER = r"""
import http.server, sys
class H(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200); self.end_headers(); self.wfile.write(b'{"status":"ok"}')
    def log_message(self, *a): pass
s = http.server.HTTPServer((sys.argv[1] if len(sys.argv) > 1 else "127.0.0.1", 0), H)
print(s.server_address[1], flush=True)
s.serve_forever()
"""


class _Health(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b'{"status":"ok"}')

    def log_message(self, *a):
        pass


@pytest.fixture
def foreign_port():
    srv = http.server.HTTPServer(("127.0.0.1", 0), _Health)
    threading.Thread(target = srv.serve_forever, daemon = True).start()
    yield srv.server_address[1]
    srv.shutdown()
    srv.server_close()


@pytest.fixture
def idle_child():
    p = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    yield p
    p.kill()
    p.wait()


@pytest.fixture
def serving_child():
    p = subprocess.Popen([sys.executable, "-c", _CHILD_SERVER], stdout = subprocess.PIPE, text = True)
    port = int(p.stdout.readline())
    yield p, port
    p.kill()
    p.wait()


def _embedder(
    process,
    port,
    monkeypatch,
    host = "127.0.0.1",
):
    monkeypatch.setattr(config, "EMBED_HOST", host)
    b = LlamaServerBackend.__new__(LlamaServerBackend)
    b._process, b._port = process, port
    return b


def _decision(process, port):
    a = NativeClefAgent.__new__(NativeClefAgent)
    a._process, a.port, a.alias, a._log_path = process, port, "clef", None
    a._client = httpx.Client(trust_env = False)
    return a


def test_embedder_refuses_a_port_another_process_answers(foreign_port, idle_child, monkeypatch):
    b = _embedder(idle_child, foreign_port, monkeypatch)
    assert b._wait_for_health(timeout = 3.0, interval = 0.05) is False


def test_embedder_accepts_its_own_child(serving_child, monkeypatch):
    b = _embedder(*serving_child, monkeypatch)
    assert b._wait_for_health(timeout = 10.0, interval = 0.05) is True


def test_embedder_accepts_its_own_child_on_a_configured_host(monkeypatch):
    p = subprocess.Popen(
        [sys.executable, "-c", _CHILD_SERVER, "127.0.0.2"], stdout = subprocess.PIPE, text = True
    )
    try:
        line = p.stdout.readline()
        if not line:
            pytest.skip("127.0.0.2 is not bindable here")
        b = _embedder(p, int(line), monkeypatch, host = "127.0.0.2")
        assert b._wait_for_health(timeout = 10.0, interval = 0.05) is True
    finally:
        p.kill()
        p.wait()


def test_embedder_dials_the_address_its_child_bound(monkeypatch):
    host = "localhost.localdomain"
    try:
        families = {a[0] for a in socket.getaddrinfo(host, None, type = socket.SOCK_STREAM)}
    except OSError:
        families = set()
    if families != {socket.AF_INET, socket.AF_INET6}:
        pytest.skip(f"{host} does not resolve to both ::1 and 127.0.0.1 here")

    class V6(http.server.HTTPServer):
        address_family = socket.AF_INET6

    hits = []

    class Foreign(_Health):
        def do_GET(self):
            hits.append(self.path)
            super().do_GET()

    foreign = V6(("::1", 0), Foreign)
    threading.Thread(target = foreign.serve_forever, daemon = True).start()
    port = foreign.server_address[1]
    p = subprocess.Popen(
        [
            sys.executable,
            "-c",
            _CHILD_SERVER.replace(", 0)", ", int(sys.argv[2]))"),
            _embed_host(host),
            str(port),
        ],
        stdout = subprocess.PIPE,
        text = True,
    )
    try:
        assert int(p.stdout.readline()) == port
        b = _embedder(p, port, monkeypatch, host = host)
        assert b._wait_for_health(timeout = 10.0, interval = 0.05) is True
        assert b._base_url == f"http://127.0.0.1:{port}"
        assert hits == []
    finally:
        p.kill()
        p.wait()
        foreign.shutdown()
        foreign.server_close()


def test_dictation_refuses_a_port_another_process_answers(foreign_port, idle_child):
    assert MtmdSttSidecar._wait_for_server(idle_child, foreign_port) is False


def test_dictation_accepts_its_own_child(serving_child):
    assert MtmdSttSidecar._wait_for_server(*serving_child) is True


def test_decision_worker_refuses_a_port_another_process_answers(foreign_port, idle_child):
    with pytest.raises(NativeError, match = "port"):
        _decision(idle_child, foreign_port)._wait_ready(None)


def test_decision_worker_accepts_its_own_child(serving_child):
    _decision(*serving_child)._wait_ready(None)
