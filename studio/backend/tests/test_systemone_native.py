# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Focused contract tests for the text-only native Clef adapter."""

from __future__ import annotations

import io
import os
import subprocess
import threading
from types import SimpleNamespace

import pytest

from core.systemone import native_worker
from core.systemone.clef_runtime import ClefWorkerError, ClefWorkerInputError


class _Response:
    def __init__(self, status, payload):
        self.status_code, self._payload, self.text = status, payload, str(payload)

    def json(self):
        return self._payload


class _Process:
    pid = 4321

    def __init__(self):
        self.returncode = None
        self.stdout = io.StringIO()
        self.terminated = self.killed = False

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminated, self.returncode = True, 0

    def kill(self):
        self.killed, self.returncode = True, 0

    def wait(self, timeout = None):
        if self.returncode is None:
            raise subprocess.TimeoutExpired("llama-server", timeout)
        return self.returncode


class _Client:
    responses = []
    calls = []

    def __init__(self, **kwargs):
        self.closed = False

    def get(self, *args, **kwargs):
        return _Response(200, {"status": "ok"})

    def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return self.responses.pop(0)

    def close(self):
        self.closed = True


@pytest.fixture
def native(monkeypatch, tmp_path):
    binary = tmp_path / "llama-server"
    binary.write_bytes(b"native executable /v1/systemone")
    process, forgotten, commands = _Process(), [], []
    monkeypatch.setattr(native_worker, "_resolve_binary", lambda: str(binary))
    monkeypatch.setattr(native_worker.httpx, "Client", _Client)
    monkeypatch.setattr(
        native_worker.subprocess,
        "Popen",
        lambda command, **kwargs: commands.append(command) or process,
    )

    from core.inference.llama_cpp import LlamaCppBackend
    from utils import process_lifetime

    monkeypatch.setattr(LlamaCppBackend, "_find_free_port", staticmethod(lambda: 8765))
    monkeypatch.setattr(
        LlamaCppBackend, "_llama_server_env_for_binary", staticmethod(lambda _: dict(os.environ))
    )
    monkeypatch.setattr(
        LlamaCppBackend, "_clear_device_placement_env", staticmethod(lambda _: None)
    )
    monkeypatch.setattr(
        LlamaCppBackend, "_enumerated_gpu_devices", staticmethod(lambda *_: ["CUDA0"])
    )
    monkeypatch.setattr(process_lifetime, "spawn_on_lifetime_thread", lambda spawn: spawn())
    monkeypatch.setattr(process_lifetime, "adopt_pid", lambda _: None)
    monkeypatch.setattr(process_lifetime, "forget_pid", forgotten.append)
    monkeypatch.setattr(process_lifetime, "is_process_shutting_down", lambda: False)
    _Client.responses, _Client.calls = [], []
    model = SimpleNamespace(name = "clef-flash")
    path = tmp_path / "clef-flash.gguf"
    path.write_bytes(b"gguf")
    return model, path, process, forgotten, commands


@pytest.mark.parametrize("shared", [False, True])
def test_capability_probe_rejects_old_binary_and_accepts_route(tmp_path, monkeypatch, shared):
    old, current = tmp_path / "old", tmp_path / "current"
    old.write_bytes(b"no decisions here")
    current.write_bytes(b"libllama-server-impl.so" if shared else b"contains /v1/systemone route")
    if shared:
        (tmp_path / "libllama-server-impl.so").write_bytes(b"contains /v1/systemone route")

    assert not native_worker.supports_systemone(old)
    assert native_worker.supports_systemone(current)
    monkeypatch.setattr(native_worker, "_resolve_binary", lambda: str(old))
    assert native_worker.native_availability() == {
        "available": False,
        "reason": "This llama-server build does not support /v1/systemone.",
        "binary": str(old),
    }
    monkeypatch.setattr(native_worker, "_resolve_binary", lambda: str(current))
    assert native_worker.native_availability() == {
        "available": True,
        "reason": None,
        "binary": str(current),
    }


def test_native_worker_sends_normalized_text_wire_format(native):
    model, path, process, forgotten, commands = native
    _Client.responses = [
        _Response(
            200,
            {
                "model": "server-name",
                "answers": {
                    "route": {"type": "choice", "probabilities": {"a": 0.2, "b": 0.8}},
                    "priority": {"type": "score", "probabilities": {"0": 0.4, "1": 0.6}},
                },
            },
        )
    ]
    worker = native_worker.NativeWorker()
    worker.start(path, model, "cpu", threading.Event())
    result = worker.decide(
        model,
        {"text": "café"},
        {
            "route": {"type": "choice", "criteria": {"a": "accept", "b": "decline"}},
            "priority": {"type": "score", "instructions": None, "criteria": ["low", "high"]},
        },
        [],
    )

    _, request = _Client.calls[0]
    assert request["json"] == {
        "model": "clef-flash",
        "state": {"text": "café"},
        "questions": {
            "route": {
                "type": "choice",
                "criteria": {"a": "accept", "b": "decline"},
                "instructions": "null",
            },
            "priority": {"type": "score", "instructions": "null", "criteria": ["low", "high"]},
        },
    }
    assert request["headers"]["Authorization"].startswith("Bearer ")
    assert "images" not in request["json"]
    assert commands[0][commands[0].index("-ngl") + 1] == "0"
    assert commands[0][commands[0].index("-c") + 1] == str(native_worker.NATIVE_MAX_CONTEXT_TOKENS)
    assert result["model"] == "clef-flash"
    assert worker.device == "cpu" and worker.gpu_available is False
    assert worker.close() and process.terminated and forgotten == [process.pid]


def test_errors_media_refusal_and_cancellation_cleanup(native, monkeypatch):
    model, path, process, forgotten, _ = native
    _Client.responses = [_Response(401, {"error": {"message": "bad process key"}})]
    worker = native_worker.NativeWorker()
    worker.start(path, model, "gpu", threading.Event())
    with pytest.raises(ClefWorkerError, match = "HTTP 401: bad process key"):
        worker.decide(model, "state", {"q": {"type": "noul"}}, [])
    _Client.responses = [
        _Response(
            500,
            {
                "error": {
                    "message": "input (8291 tokens) is too large to process. increase the physical batch size (current batch size: 2048)"
                }
            },
        )
    ]
    with pytest.raises(ClefWorkerInputError, match = "too large"):
        worker.decide(model, "state", {"q": {"type": "noul"}}, [])
    assert worker.device == "CUDA0" and worker.gpu_available is True
    worker.cancel()
    assert process.terminated and forgotten == [process.pid]

    with pytest.raises(ClefWorkerInputError, match = "does not support images"):
        native_worker.NativeWorker().decide(model, "state", {"q": {"type": "noul"}}, [b"image"])
    assert not native_worker.supports_request({"q": {"instructions": ""}}, [])
    assert not native_worker.supports_request({"q": {"type": "score", "criteria": ["only"]}}, [])
    from core.inference.llama_cpp import LlamaCppBackend

    monkeypatch.setattr(LlamaCppBackend, "_enumerated_gpu_devices", staticmethod(lambda *_: []))
    with pytest.raises(ClefWorkerError, match = "no usable GPU"):
        native_worker.NativeWorker().start(path, model, "gpu", threading.Event())
