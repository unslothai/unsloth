# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import http.server
import subprocess
import sys
import threading

import httpx
import pytest

pytest.importorskip("psutil")

from core.inference.stt_mtmd_sidecar import MtmdSttSidecar
from core.rag import config
from core.rag.embed_llama_server import LlamaServerBackend
from core.systemone.native_worker import NativeClefAgent, NativeError

_CHILD_SERVER = r"""
import http.server
class H(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200); self.end_headers(); self.wfile.write(b'{"status":"ok"}')
    def log_message(self, *a): pass
s = http.server.HTTPServer(("127.0.0.1", 0), H)
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


def _embedder(process, port, monkeypatch):
    monkeypatch.setattr(config, "EMBED_HOST", "127.0.0.1")
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


def test_dictation_refuses_a_port_another_process_answers(foreign_port, idle_child):
    assert MtmdSttSidecar._wait_for_server(idle_child, foreign_port) is False


def test_dictation_accepts_its_own_child(serving_child):
    assert MtmdSttSidecar._wait_for_server(*serving_child) is True


def test_decision_worker_refuses_a_port_another_process_answers(foreign_port, idle_child):
    with pytest.raises(NativeError, match = "port"):
        _decision(idle_child, foreign_port)._wait_ready(None)


def test_decision_worker_accepts_its_own_child(serving_child):
    _decision(*serving_child)._wait_ready(None)
