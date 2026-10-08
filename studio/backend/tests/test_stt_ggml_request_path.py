# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""whisper-server runs under a random per-launch path, so a web page that finds its loopback port cannot reach it."""

import http.server
import json
import os
import re
import stat
import sys
import threading

import numpy as np
import pytest

import core.inference.stt_ggml_sidecar as ggml_module
from core.inference.stt_ggml_sidecar import GgmlSttSidecar


@pytest.fixture(autouse = True)
def _isolate(monkeypatch, tmp_path):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "studio"))
    monkeypatch.delenv("WHISPER_SERVER_PATH", raising = False)
    monkeypatch.setattr(ggml_module, "find_whisper_server_binary", lambda: "/bin/echo")
    monkeypatch.setattr(ggml_module, "_cached_model_path", lambda model_id: "/tmp/ggml.bin")
    monkeypatch.setattr(
        ggml_module,
        "_decode_audio_bounded",
        lambda audio, cancel_event = None: np.zeros(16000, dtype = np.float32),
    )
    ggml_module._REQUEST_PATH_SUPPORT.clear()


def _fake_popen(monkeypatch, commands):
    class FakeProcess:
        pid = 4321

        def __init__(self, command, *args, **kwargs):
            commands.append(command)

        def poll(self):
            return None

        def terminate(self):
            pass

        def wait(self, timeout = None):
            return 0

    monkeypatch.setattr(ggml_module.subprocess, "Popen", FakeProcess)
    monkeypatch.setattr(ggml_module, "adopt_pid", lambda pid: None)
    monkeypatch.setattr(ggml_module, "forget_pid", lambda pid: None)
    monkeypatch.setattr(ggml_module, "_training_active", lambda: False)


def test_each_launch_gets_a_random_request_path(monkeypatch):
    commands, routes = [], []
    _fake_popen(monkeypatch, commands)
    monkeypatch.setattr(ggml_module, "_supports_request_path", lambda binary: True)
    monkeypatch.setattr(
        GgmlSttSidecar,
        "_wait_for_server",
        staticmethod(lambda process, port, cancel_event = None, route = "": routes.append(route)),
    )
    for _ in range(2):
        sidecar = GgmlSttSidecar()
        sidecar.load("small")
        flag = commands[-1].index("--request-path")
        route = commands[-1][flag + 1]
        assert re.fullmatch(r"/[0-9a-f]{32}", route)
        assert sidecar._route == route == routes[-1]
        sidecar.unload()
        assert sidecar._route == ""
    assert commands[0][commands[0].index("--request-path") + 1] != route


def test_binary_without_the_flag_still_launches(monkeypatch):
    commands, routes = [], []
    _fake_popen(monkeypatch, commands)
    monkeypatch.setattr(ggml_module, "_supports_request_path", lambda binary: False)
    monkeypatch.setattr(
        GgmlSttSidecar,
        "_wait_for_server",
        staticmethod(lambda process, port, cancel_event = None, route = "": routes.append(route)),
    )
    sidecar = GgmlSttSidecar()
    sidecar.load("small")
    assert "--request-path" not in commands[0]
    assert routes == [""] and sidecar._route == ""


def _script(tmp_path, name, help_text):
    if sys.platform == "win32":
        pytest.skip("shell script stand-in for whisper-server")
    path = tmp_path / name
    path.write_text(f"#!/bin/sh\necho '{help_text}' >&2\necho run >> '{tmp_path}/{name}.runs'\n")
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return str(path)


def test_help_probe_detects_the_flag_and_caches(tmp_path):
    new = _script(tmp_path, "new-server", "  --request-path PATH, Request path for all requests")
    old = _script(tmp_path, "old-server", "  --port N, Port number")
    assert ggml_module._supports_request_path(new) is True
    assert ggml_module._supports_request_path(new) is True
    assert ggml_module._supports_request_path(old) is False
    assert ggml_module._supports_request_path(str(tmp_path / "missing")) is False
    assert (tmp_path / "new-server.runs").read_text().count("run") == 1


class _RoutedHandler(http.server.BaseHTTPRequestHandler):
    """Serves whisper-server's index and /inference only under ROUTE, 404 elsewhere, like --request-path."""

    route = "/" + "ab" * 16
    posts: list = []

    def do_GET(self):
        if self.path != f"{self.route}/":
            self.send_error(404)
            return
        body = b"<html>whisper.cpp server</html>"
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        self.rfile.read(length)
        if self.path != f"{self.route}/inference":
            self.send_error(404)
            return
        type(self).posts.append(self.path)
        body = json.dumps({"text": "routed"}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


@pytest.fixture()
def routed_server():
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _RoutedHandler)
    thread = threading.Thread(target = server.serve_forever, daemon = True)
    thread.start()
    yield server.server_address[1]
    server.shutdown()


class _Alive:
    def poll(self):
        return None


def test_probe_and_inference_use_the_route(monkeypatch, routed_server):
    route = _RoutedHandler.route
    assert GgmlSttSidecar._probe_is_whisper_server(_Alive(), routed_server, route) is True
    assert GgmlSttSidecar._probe_is_whisper_server(_Alive(), routed_server) is False

    sidecar = GgmlSttSidecar()

    def fake_load(model = None, request_cancel_event = None):
        sidecar._port = routed_server
        sidecar._route = route
        sidecar._model_id = ggml_module.resolve_ggml_model_id(model)

    monkeypatch.setattr(sidecar, "load", fake_load)
    assert sidecar.transcribe(b"RIFF", model = "small")["text"] == "routed"
    assert _RoutedHandler.posts[-1] == f"{route}/inference"

    # A caller without the route (a web page that found the port) gets 404 from every endpoint.
    def unrouted_load(model = None, request_cancel_event = None):
        fake_load(model)
        sidecar._route = ""

    monkeypatch.setattr(sidecar, "load", unrouted_load)
    with pytest.raises(ggml_module.SttEngineUnavailableError, match = "HTTP 404"):
        sidecar.transcribe(b"RIFF", model = "small")
