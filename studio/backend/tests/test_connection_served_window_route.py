# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import json
import sys
from http.server import BaseHTTPRequestHandler
from pathlib import Path

import pytest
from fastapi import HTTPException

from routes import providers as providers_route

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_external_provider_sampling_over_the_wire import _run, _Server  # noqa: E402


class _Ollama(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    loaded: list[dict] = []
    window = 4_096
    requests: list[str] = []

    def log_message(self, *args, **kwargs) -> None:
        return

    def _send(self, body: dict) -> None:
        raw = json.dumps(body).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self) -> None:  # noqa: N802
        type(self).requests.append(f"GET {self.path}")
        self._send({"models": self.loaded})

    def do_POST(self) -> None:  # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)))
        type(self).requests.append(f"POST {self.path} {body['model']}")
        if not self.loaded:
            type(self).loaded = [{"name": body["model"], "context_length": self.window}]
        self._send({"model": body["model"], "done": True, "done_reason": "load"})


def _ollama(loaded: list[dict]):
    return type("Ollama", (_Ollama,), {"loaded": loaded, "requests": []})


def _windows(monkeypatch, handler, *steps) -> list:
    answers: list = []
    with _Server(handler = handler) as upstream:
        row = {
            "id": "c1",
            "provider_type": "ollama",
            "base_url": upstream.base_url,
            "is_enabled": True,
            "models": ["qwen3:8b"],
        }
        monkeypatch.setattr(providers_route.providers_db, "get_provider", lambda _id: row)

        async def go() -> None:
            for model, before in steps:
                before()
                answers.append(await providers_route.load_served_window("c1", model))

        _run(go)
    return answers


def _window(
    monkeypatch,
    handler,
    model: str = "qwen3:8b",
) -> dict:
    return _windows(monkeypatch, handler, (model, lambda: None))[0]


def test_an_idle_ollama_model_is_loaded_to_read_its_served_window(monkeypatch):
    handler = _ollama([])
    assert _window(monkeypatch, handler) == {"context_length": 4_096}
    assert handler.requests == ["POST /api/generate qwen3:8b", "GET /api/ps"]


def test_a_loaded_ollama_model_reports_its_window(monkeypatch):
    handler = _ollama([{"name": "qwen3:8b", "context_length": 32_768}])
    assert _window(monkeypatch, handler) == {"context_length": 32_768}


def test_a_restarted_ollama_reports_its_new_window(monkeypatch):
    handler = _ollama([])

    def restart() -> None:
        handler.loaded, handler.window = [], 65_536

    answers = _windows(monkeypatch, handler, ("qwen3:8b", lambda: None), ("qwen3:8b", restart))
    assert answers == [{"context_length": 4_096}, {"context_length": 65_536}]


def test_a_model_outside_the_connection_is_never_loaded(monkeypatch):
    handler = _ollama([])
    with pytest.raises(HTTPException) as raised:
        _window(monkeypatch, handler, model = "llama3.3:70b")
    assert raised.value.status_code == 404
    assert handler.requests == []
