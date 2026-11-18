# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from auth.authentication import authenticated_via_api_key, get_current_subject
import routes.inference as inference_route
from .llama_backend_double import FakeLlamaCppBackend

_SYSTEM = "You are Captain Hook. Answer every question as a pirate."


def _write_tag(root: Path, name: str, layers: list) -> None:
    tag_file = root / "manifests" / "registry.ollama.ai" / "library" / name / "latest"
    tag_file.parent.mkdir(parents = True)
    tag_file.write_text(json.dumps({"config": {}, "layers": layers}), encoding = "utf-8")


def _ollama_refs(tmp_path: Path, monkeypatch) -> dict:
    from hub.services.models import ollama

    root = tmp_path / "ollama"
    blobs = root / "blobs"
    blobs.mkdir(parents = True)
    (blobs / f"sha256-{'a' * 64}").write_bytes(b"GGUF-not-really")
    (blobs / f"sha256-{'b' * 64}").write_text(_SYSTEM, encoding = "utf-8")
    weights = {"mediaType": "application/vnd.ollama.image.model", "digest": f"sha256:{'a' * 64}"}
    system = {"mediaType": "application/vnd.ollama.image.system", "digest": f"sha256:{'b' * 64}"}
    _write_tag(root, "pirate", [weights, system])
    _write_tag(root, "plain", [weights])
    monkeypatch.setattr(ollama, "ollama_model_dirs", lambda: [root])
    return {row.model_id: row.load_id for row in ollama.scan_ollama_dir(root)}


def _ollama_ref_with_system(tmp_path: Path, monkeypatch) -> str:
    return _ollama_refs(tmp_path, monkeypatch)["ollama/pirate:latest"]


def _client(backend, monkeypatch) -> TestClient:
    monkeypatch.setattr(inference_route, "get_llama_cpp_backend", lambda: backend)
    app = FastAPI()
    app.include_router(inference_route.router)
    app.dependency_overrides[get_current_subject] = lambda: "test-user"
    app.dependency_overrides[authenticated_via_api_key] = lambda: False
    return TestClient(app)


def _sent_messages(tmp_path, monkeypatch, messages):
    sent = {}

    class _OllamaBackend(FakeLlamaCppBackend):
        model_identifier = _ollama_ref_with_system(tmp_path, monkeypatch)

        def generate_chat_completion(self, **kwargs):
            sent["messages"] = kwargs["messages"]
            yield "Arr"

    monkeypatch.setattr(inference_route, "_effective_enable_tools", lambda payload: False)
    response = _client(_OllamaBackend(), monkeypatch).post(
        "/chat/completions", json = {"messages": messages, "stream": False}
    )
    assert response.status_code == 200, response.text
    return sent["messages"]


def test_modelfile_system_prompt_reaches_the_model(tmp_path, monkeypatch):
    messages = _sent_messages(tmp_path, monkeypatch, [{"role": "user", "content": "Hi"}])

    assert messages[0]["role"] == "system"
    assert _SYSTEM in messages[0]["content"]
    assert messages[-1] == {"role": "user", "content": "Hi"}


def test_request_system_prompt_replaces_modelfile_system_prompt(tmp_path, monkeypatch):
    messages = _sent_messages(
        tmp_path,
        monkeypatch,
        [
            {"role": "system", "content": "Be brief."},
            {"role": "user", "content": "Hi"},
        ],
    )

    system = [m["content"] for m in messages if m["role"] == "system"]
    assert len(system) == 1
    assert "Be brief." in system[0]
    assert _SYSTEM not in system[0]


def test_empty_request_system_turn_suppresses_modelfile_system_prompt(tmp_path, monkeypatch):
    messages = _sent_messages(
        tmp_path,
        monkeypatch,
        [{"role": "system", "content": ""}, {"role": "user", "content": "Hi"}],
    )

    assert all(_SYSTEM not in str(m["content"]) for m in messages)


def test_tag_sharing_weights_but_not_system_is_not_already_resident(tmp_path, monkeypatch):
    from core.inference.llama_cpp import LlamaCppBackend
    from hub.services.models.ollama import ollama_model_ref_files

    refs = _ollama_refs(tmp_path, monkeypatch)
    pirate, plain = refs["ollama/pirate:latest"], refs["ollama/plain:latest"]

    def resident(ref):
        backend = FakeLlamaCppBackend()
        backend.model_identifier = ref
        backend._gguf_load_identity = LlamaCppBackend._gguf_load_source_identity(
            *ollama_model_ref_files(ref)
        )
        return backend

    assert inference_route._resident_is_still_tagged(plain, resident(plain))
    assert inference_route._resident_is_still_tagged(pirate, resident(pirate))
    assert not inference_route._resident_is_still_tagged(pirate, resident(plain))
    assert not inference_route._resident_is_still_tagged(plain, resident(pirate))


def test_tool_passthrough_body_carries_modelfile_system_prompt(tmp_path, monkeypatch):
    from models.inference import ChatCompletionRequest

    class _Backend:
        model_identifier = _ollama_ref_with_system(tmp_path, monkeypatch)
        is_vision = False
        _request_reasoning_kwargs = lambda *_args, **_kwargs: None

    tool = {"type": "function", "function": {"name": "lookup", "parameters": {"type": "object"}}}

    def body(messages):
        payload = ChatCompletionRequest(messages = messages, tools = [tool])
        return inference_route._build_openai_passthrough_body(
            payload, backend_ctx = 4096, llama_backend = _Backend()
        )["messages"]

    sent = body([{"role": "user", "content": "Hi"}])
    assert sent[0]["role"] == "system" and _SYSTEM in sent[0]["content"]
    sent = body([{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Hi"}])
    assert all(_SYSTEM not in str(m["content"]) for m in sent)


def test_context_count_prices_modelfile_system_prompt(tmp_path, monkeypatch):
    counted = []

    class _OllamaBackend(FakeLlamaCppBackend):
        model_identifier = _ollama_ref_with_system(tmp_path, monkeypatch)
        _request_reasoning_kwargs = lambda *_args, **_kwargs: None

    def _count(backend, payload, messages, *_args):
        counted.append(messages)
        return 1, None, None

    monkeypatch.setattr(inference_route, "_count_chat_context_tokens", _count)
    client = _client(_OllamaBackend(), monkeypatch)
    for messages in (
        [{"role": "user", "content": "Hi"}],
        [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Hi"}],
    ):
        response = client.post("/chat/count_tokens", json = {"messages": messages})
        assert response.status_code == 200, response.text

    assert _SYSTEM in counted[0][0]["content"]
    assert all(_SYSTEM not in str(m["content"]) for m in counted[1])


def test_anthropic_count_falls_back_to_modelfile_system_prompt(tmp_path, monkeypatch):
    import asyncio
    from types import SimpleNamespace
    from models.inference import AnthropicMessagesRequest

    counted = []

    class _Backend:
        is_loaded = True
        is_vision = False
        model_identifier = _ollama_ref_with_system(tmp_path, monkeypatch)
        context_length = 4096
        supports_tools = False
        supports_tool_passthrough = False

        def _request_reasoning_kwargs(self, *_args, **_kwargs):
            return {}

        def count_chat_tokens(self, messages, *_args, **_kwargs):
            counted.append(messages)
            return 1

    class _Request:
        state = SimpleNamespace()
        url = SimpleNamespace(path = "/v1/messages/count_tokens")
        method = "POST"

        async def is_disconnected(self):
            return False

    monkeypatch.setattr(inference_route, "get_llama_cpp_backend", lambda: _Backend())
    empty = {"type": "text", "text": ""}
    for system in (None, [empty], [empty, empty], "Be brief."):
        payload = AnthropicMessagesRequest(
            max_tokens = 16, messages = [{"role": "user", "content": "Hi"}], system = system
        )
        asyncio.run(
            inference_route.anthropic_count_tokens(payload, request = _Request(), current_subject = "t")
        )

    assert all(m[0]["role"] == "system" and _SYSTEM in m[0]["content"] for m in counted[:3])
    assert all(_SYSTEM not in str(m["content"]) for m in counted[3])
