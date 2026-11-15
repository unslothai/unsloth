# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""httpx parses the CA bundle per client even for http:// loopback; every llama-server call shares one context."""

import ast
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import httpx
import pytest

_backend = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, _backend)

from core.inference import llama_cpp  # noqa: E402
from core.inference.llama_cpp import LlamaCppBackend, _local_ssl_context  # noqa: E402

_ONE_SHOT = {"get", "post", "put", "delete", "request", "stream", "Client"}


def _httpx_calls_without_verify(path: Path) -> list[int]:
    tree = ast.parse(path.read_text(encoding = "utf-8"))
    missing = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (
            isinstance(func, ast.Attribute)
            and isinstance(func.value, ast.Name)
            and func.value.id == "httpx"
            and func.attr in _ONE_SHOT
        ):
            continue
        if not any(kw.arg == "verify" for kw in node.keywords):
            missing.append(node.lineno)
    return missing


def test_loopback_llama_calls_pass_verify():
    missing = _httpx_calls_without_verify(Path(llama_cpp.__file__))
    assert missing == [], f"llama_cpp.py: httpx calls without verify= at lines {missing}"


def _stub(port: int) -> LlamaCppBackend:
    backend = LlamaCppBackend.__new__(LlamaCppBackend)
    backend._process = object()
    backend._healthy = True
    backend._port = port
    backend._api_key = None
    backend._slot_save_dir = "/tmp/slots"
    return backend


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    seen: list = []

    def log_message(self, *_args):
        pass

    def _send(
        self,
        obj,
        code = 200,
    ):
        data = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        type(self).seen.append(("GET", self.path, self.headers.get("Authorization")))
        if self.path == "/props":
            return self._send({"default_generation_settings": {"n_ctx": 2048}, "modalities": {}})
        self._send({"error": "not found"}, 404)

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        self.rfile.read(length)
        type(self).seen.append(("POST", self.path, self.headers.get("Authorization")))
        if self.path.startswith("/slots/"):
            slot = int(self.path.split("/")[2].split("?")[0])
            return self._send({"id_slot": slot, "n_erased": 3})
        self._send({"error": "not found"}, 404)


@pytest.fixture
def server():
    _Handler.seen = []
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target = httpd.serve_forever, daemon = True)
    thread.start()
    try:
        yield httpd.server_address[1]
    finally:
        httpd.shutdown()
        httpd.server_close()


@pytest.fixture
def verify_args(monkeypatch):
    seen = []
    real_init = httpx.Client.__init__

    def init(self, *args, **kwargs):
        seen.append(kwargs.get("verify", "<default>"))
        real_init(self, *args, **kwargs)

    monkeypatch.setattr(httpx.Client, "__init__", init)
    return seen


def test_query_server_props_shares_context_and_answers_unchanged(server, verify_args):
    backend = _stub(server)
    props = backend._query_server_props()
    assert props == {"default_generation_settings": {"n_ctx": 2048}, "modalities": {}}
    assert backend._query_server_n_ctx() == 2048
    assert verify_args and all(v is _local_ssl_context() for v in verify_args)


def test_query_server_props_keeps_auth_header(server, verify_args):
    backend = _stub(server)
    backend._api_key = "k"
    assert backend._query_server_props() is not None
    assert _Handler.seen[-1] == ("GET", "/props", "Bearer k")


def test_release_idle_chat_slot_shares_context(server, verify_args):
    backend = _stub(server)
    assert backend.release_idle_chat_slot(backend.base_url, 1) is True
    assert _Handler.seen[-1][:2] == ("POST", "/slots/1?action=erase")
    assert verify_args and all(v is _local_ssl_context() for v in verify_args)


def test_shared_context_is_one_object():
    assert _local_ssl_context() is _local_ssl_context()
    assert llama_cpp._local_ssl_context.cache_info().maxsize == 1
