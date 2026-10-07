# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Contract tests for the text-only native Clef adapter; no model inference here."""

import json
import os
import subprocess
import sys
import threading
from types import SimpleNamespace

import httpx
import pytest

from core.systemone import native_worker
from core.systemone.owned_runtime import ClefWorkerError, ClefWorkerInputError


@pytest.fixture
def native(monkeypatch, tmp_path):
    from core.inference.llama_cpp import LlamaCppBackend
    from utils import process_lifetime

    binary = tmp_path / "llama-server"
    binary.write_bytes(b"native executable /v1/systemone")
    popen = subprocess.Popen
    forgotten, commands, requests, responses = [], [], [], []

    def respond(request):
        if request.url.path == "/health":
            return httpx.Response(200, json = {"status": "ok"})
        requests.append(request)
        return responses.pop(0)

    client = httpx.Client
    monkeypatch.setattr(native_worker, "_resolve_binary", lambda: str(binary))
    monkeypatch.setattr(
        native_worker.httpx,
        "Client",
        lambda **kwargs: client(transport = httpx.MockTransport(respond), **kwargs),
    )
    monkeypatch.setattr(
        native_worker.subprocess,
        "Popen",
        lambda command, **kwargs: commands.append(command) or process,
    )
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
    model = SimpleNamespace(name = "clef-flash")
    path = tmp_path / "clef-flash.gguf"
    path.write_bytes(b"gguf")
    process = popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        yield model, path, process, forgotten, commands, requests, responses
    finally:
        if process.poll() is None:
            process.kill()
        process.wait(timeout = 5)


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
    model, path, process, forgotten, commands, requests, responses = native
    responses.append(
        httpx.Response(
            200,
            json = {
                "model": "server-name",
                "answers": {
                    "route": {"type": "choice", "probabilities": {"a": 0.2, "b": 0.8}},
                    "priority": {"type": "score", "probabilities": {"0": 0.4, "1": 0.6}},
                },
            },
        )
    )
    questions = {
        "route": {
            "type": "choice",
            "criteria": {"a": "accept", "b": "decline"},
            "instructions": "route",
        },
        "priority": {"type": "score", "instructions": "priority", "criteria": ["low", "high"]},
    }
    inputs = {key: dict(question) for key, question in questions.items()}
    inputs["route"].pop("instructions")
    inputs["priority"]["instructions"] = None
    worker = native_worker.NativeWorker()
    worker.start(path, model, "cpu", threading.Event())
    result = worker.decide(model, {"text": "café"}, inputs, [])
    request = requests[0]
    payload = json.loads(request.content)
    assert payload == {"model": "clef-flash", "state": {"text": "café"}, "questions": questions}
    assert request.url.path == "/v1/systemone"
    assert request.headers["Authorization"].startswith("Bearer ")
    assert "images" not in payload
    assert commands[0][commands[0].index("-ngl") + 1] == "0"
    assert commands[0][commands[0].index("-c") + 1] == str(native_worker.NATIVE_MAX_CONTEXT_TOKENS)
    assert result["model"] == "clef-flash"
    assert worker.device == "cpu" and worker.gpu_available is False
    assert worker.close() and process.poll() is not None and forgotten == [process.pid]


def test_errors_media_refusal_and_cancellation_cleanup(native, monkeypatch):
    from core.inference.llama_cpp import LlamaCppBackend

    model, path, process, forgotten, _, _, responses = native
    responses.append(httpx.Response(401, json = {"error": {"message": "bad process key"}}))
    worker = native_worker.NativeWorker()
    worker.start(path, model, "gpu", threading.Event())
    with pytest.raises(ClefWorkerError, match = "HTTP 401: bad process key"):
        worker.decide(model, "state", {"q": {"type": "noul"}}, [])
    responses.append(
        httpx.Response(
            500,
            json = {
                "error": {
                    "message": "input (8291 tokens) is too large to process. increase the physical batch size (current batch size: 2048)"
                }
            },
        )
    )
    with pytest.raises(ClefWorkerInputError, match = "too large"):
        worker.decide(model, "state", {"q": {"type": "noul"}}, [])
    assert worker.device == "CUDA0" and worker.gpu_available is True
    worker.cancel()
    assert process.poll() is not None and forgotten == [process.pid]
    with pytest.raises(ClefWorkerInputError, match = "does not support images"):
        native_worker.NativeWorker().decide(model, "state", {"q": {"type": "noul"}}, [b"image"])
    assert not native_worker.supports_request({"q": {"instructions": ""}}, [])
    assert not native_worker.supports_request({"q": {"type": "score", "criteria": ["only"]}}, [])
    monkeypatch.setattr(LlamaCppBackend, "_enumerated_gpu_devices", staticmethod(lambda *_: []))
    with pytest.raises(ClefWorkerError, match = "no usable GPU"):
        native_worker.NativeWorker().start(path, model, "gpu", threading.Event())


@pytest.mark.parametrize("backend", ["llama.cpp", "pytorch"])
def test_cache_delete_guard_matches_the_resident_backend(monkeypatch, backend):
    from core.systemone import catalog, runtime
    from hub.services.models import deletion

    monkeypatch.setattr(
        runtime, "status", lambda: {"loaded_model": "clef-flash", "backend": backend}
    )
    monkeypatch.setattr(runtime, "loading_repo_ids", lambda: ())
    native_repo = catalog.NATIVE_CHECKPOINTS["clef-flash"].source
    torch_repo = catalog.CHECKPOINTS["clef-flash"].source
    resident, unused = (
        (native_repo, torch_repo) if backend == "llama.cpp" else (torch_repo, native_repo)
    )
    assert deletion._decisions_blocks_delete(resident) is not None
    assert deletion._decisions_blocks_delete(unused) is None
