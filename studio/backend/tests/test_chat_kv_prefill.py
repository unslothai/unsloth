# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Studio's opportunistic llama.cpp chat KV-cache prefill."""

from __future__ import annotations

import asyncio
import sys
import threading
from pathlib import Path

import httpx

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from auth.authentication import get_current_subject  # noqa: E402
from routes import inference as inference_route  # noqa: E402


class _Backend:
    is_loaded = True

    is_diffusion = False
    _is_audio = False
    base_url = "http://llama.prefill.test"

    model_identifier = "local-model"
    _openai_advertised_id = None
    _native_grant_backed = False
    _native_display_label = None
    context_length = 4096
    markup_profile = None

    supports_tools = True
    is_vision = False

    @staticmethod
    def _prompt_cache_off():
        return False

    @staticmethod
    def _request_reasoning_kwargs(enable_thinking, reasoning_effort, preserve_thinking):
        return {
            "enable_thinking": enable_thinking,
            "reasoning_effort": reasoning_effort,
            "preserve_thinking": preserve_thinking,
        }


class _Response:
    def __init__(
        self,
        status_code: int = 200,
        text: str = "{}",
    ):
        self.status_code = status_code
        self.text = text


class _Client:
    def __init__(self, response: _Response | None = None):
        self.response = response or _Response()
        self.calls: list[dict] = []
        self.closed = False

    async def post(self, url, **kwargs):
        self.calls.append({"url": url, **kwargs})
        return self.response

    async def aclose(self):
        self.closed = True


def _client(monkeypatch, backend, upstream: _Client) -> TestClient:
    monkeypatch.setattr(inference_route, "get_llama_cpp_backend", lambda: backend)
    monkeypatch.setattr(inference_route, "_cancelable_nonstreaming_client", lambda: upstream)
    app = FastAPI()
    app.include_router(inference_route.studio_router, prefix = "/api/inference")
    app.dependency_overrides[get_current_subject] = lambda: "test-user"
    return TestClient(app)


def _payload(**updates):
    body = {
        "model": "local-model",
        "messages": [
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi!"},
        ],
        "stream": True,
        "max_tokens": 77,
        "enable_thinking": False,
        "reasoning_effort": "low",
        "preserve_thinking": True,
    }
    body.update(updates)
    return body


def test_prefill_forwards_zero_token_nonstreaming_request(monkeypatch):
    upstream = _Client()
    response = _client(monkeypatch, _Backend(), upstream).post(
        "/api/inference/chat/prefill",
        json = _payload(),
    )

    assert response.status_code == 200
    assert response.json() == {"prefilled": True}
    [call] = upstream.calls
    assert call["url"] == "http://llama.prefill.test/v1/chat/completions"
    sent = call["json"]
    assert sent["messages"][-1] == {"role": "assistant", "content": "Hi!"}
    assert sent["stream"] is False
    assert sent["n_predict"] == 0
    assert "max_tokens" not in sent
    assert "stream_options" not in sent
    assert sent["chat_template_kwargs"] == {
        "enable_thinking": False,
        "reasoning_effort": "low",
        "preserve_thinking": True,
    }
    assert upstream.closed is True


def test_prefill_uses_standard_gguf_message_normalization(monkeypatch):
    upstream = _Client()
    response = _client(monkeypatch, _Backend(), upstream).post(
        "/api/inference/chat/prefill",
        json = _payload(
            messages = [
                {"role": "system", "content": "Be concise."},
                {"role": "user", "content": "First"},
                {"role": "assistant", "content": ""},
                {"role": "user", "content": "Second"},
                {"role": "assistant", "content": "Done"},
            ]
        ),
    )

    assert response.json() == {"prefilled": True}
    [call] = upstream.calls
    assert call["json"]["messages"] == [
        {"role": "system", "content": "Be concise."},
        {"role": "user", "content": "First\n\nSecond"},
        {"role": "assistant", "content": "Done"},
    ]


def test_prefill_reproduces_studio_tool_prompt(monkeypatch):
    tool = {
        "type": "function",
        "function": {
            "name": "python",
            "description": "Run Python",
            "parameters": {"type": "object", "properties": {}},
        },
    }

    async def select_tools(_payload, *, tools_on, mcp_allowed):
        assert tools_on is True
        assert mcp_allowed is False
        return [tool]

    monkeypatch.setattr(inference_route, "_select_request_tools", select_tools)
    monkeypatch.setattr(
        inference_route,
        "_build_tool_action_nudge",
        lambda **_kwargs: "USE THE PYTHON TOOL",
    )
    upstream = _Client()
    response = _client(monkeypatch, _Backend(), upstream).post(
        "/api/inference/chat/prefill",
        json = _payload(enable_tools = True),
    )

    assert response.json() == {"prefilled": True}
    [call] = upstream.calls
    sent = call["json"]
    assert sent["tools"] == [tool]
    assert sent["tool_choice"] == "auto"
    assert sent["messages"][0]["content"].endswith("USE THE PYTHON TOOL")


def test_prefill_does_not_route_external_or_unloaded_models(monkeypatch):
    upstream = _Client()
    client = _client(monkeypatch, _Backend(), upstream)
    external = client.post(
        "/api/inference/chat/prefill",
        json = _payload(provider_type = "openai"),
    )
    assert external.json() == {"prefilled": False, "reason": "unsupported"}

    unloaded = _Backend()
    unloaded.is_loaded = False
    monkeypatch.setattr(inference_route, "get_llama_cpp_backend", lambda: unloaded)
    missing = client.post("/api/inference/chat/prefill", json = _payload())
    assert missing.json() == {"prefilled": False, "reason": "unsupported"}
    assert upstream.calls == []


def test_prefill_rejects_a_stale_model_payload(monkeypatch):
    backend = _Backend()
    backend.model_identifier = "new-model"
    upstream = _Client()
    response = _client(monkeypatch, backend, upstream).post(
        "/api/inference/chat/prefill",
        json = _payload(model = "local-model"),
    )

    assert response.json() == {"prefilled": False, "reason": "stale_model"}
    assert upstream.calls == []


def test_prefill_rechecks_model_identity_before_dispatch(monkeypatch):
    backend = _Backend()
    upstream = _Client()

    async def switch_model(_payload, _backend):
        backend.model_identifier = "new-model"
        return {}

    monkeypatch.setattr(inference_route, "_build_chat_prefill_body", switch_model)
    response = _client(monkeypatch, backend, upstream).post(
        "/api/inference/chat/prefill",
        json = _payload(),
    )

    assert response.json() == {"prefilled": False, "reason": "stale_model"}
    assert upstream.calls == []


def test_prefill_rejects_audio_and_diffusion_runtimes(monkeypatch):
    upstream = _Client()
    client = _client(monkeypatch, _Backend(), upstream)

    audio = _Backend()
    audio._is_audio = True
    monkeypatch.setattr(inference_route, "get_llama_cpp_backend", lambda: audio)
    audio_response = client.post("/api/inference/chat/prefill", json = _payload())
    assert audio_response.json() == {"prefilled": False, "reason": "unsupported"}

    diffusion = _Backend()
    diffusion.is_diffusion = True
    monkeypatch.setattr(inference_route, "get_llama_cpp_backend", lambda: diffusion)
    diffusion_response = client.post("/api/inference/chat/prefill", json = _payload())
    assert diffusion_response.json() == {"prefilled": False, "reason": "unsupported"}
    assert upstream.calls == []


def test_prefill_skips_when_prompt_cache_is_disabled(monkeypatch):
    backend = _Backend()
    backend._prompt_cache_off = lambda: True
    upstream = _Client()
    response = _client(monkeypatch, backend, upstream).post(
        "/api/inference/chat/prefill",
        json = _payload(),
    )

    assert response.json() == {"prefilled": False, "reason": "unsupported"}
    assert upstream.calls == []


def test_prefill_skips_instead_of_waiting_for_admission(monkeypatch):
    class _Reservation:
        cancelled = False

        def lease_nowait(self):
            return None

        def cancel(self):
            self.cancelled = True

    reservation = _Reservation()
    monkeypatch.setattr(
        inference_route,
        "_openai_llama_admission_reserve",
        lambda **_kwargs: (reservation, object()),
    )
    upstream = _Client()
    response = _client(monkeypatch, _Backend(), upstream).post(
        "/api/inference/chat/prefill",
        json = _payload(),
    )

    assert response.json() == {"prefilled": False, "reason": "busy"}
    assert reservation.cancelled is True
    assert upstream.calls == []


def test_real_admission_preempts_active_prefill(monkeypatch):
    class _Queue:
        def reserve(self, *, capacity, config):
            return object()

    backend = _Backend()
    cancel_event = threading.Event()
    monkeypatch.setattr(inference_route, "get_llama_admission_queue", lambda _key: _Queue())
    inference_route._register_chat_prefill_cancel(backend.base_url, cancel_event)
    try:
        inference_route._openai_llama_admission_reserve(
            request = None,
            llama_backend = backend,
        )
    finally:
        inference_route._unregister_chat_prefill_cancel(backend.base_url, cancel_event)

    assert cancel_event.is_set() is True


def test_prefill_upstream_failure_is_best_effort_and_releases_resources(monkeypatch):
    class _Lease:
        released = False

        def release(self):
            self.released = True

    class _Reservation:
        def __init__(self, lease):
            self.lease = lease

        def lease_nowait(self):
            return self.lease

    lease = _Lease()
    monkeypatch.setattr(
        inference_route,
        "_openai_llama_admission_reserve",
        lambda **_kwargs: (_Reservation(lease), object()),
    )
    upstream = _Client(_Response(503, "busy"))
    response = _client(monkeypatch, _Backend(), upstream).post(
        "/api/inference/chat/prefill",
        json = _payload(),
    )

    assert response.status_code == 200
    assert response.json() == {"prefilled": False, "reason": "upstream_error"}
    assert upstream.closed is True
    assert lease.released is True


def test_prefill_network_failure_is_best_effort(monkeypatch):
    class _FailingClient(_Client):
        async def post(self, url, **kwargs):
            self.calls.append({"url": url, **kwargs})
            raise httpx.ConnectError("offline")

    upstream = _FailingClient()
    response = _client(monkeypatch, _Backend(), upstream).post(
        "/api/inference/chat/prefill",
        json = _payload(),
    )

    assert response.status_code == 200
    assert response.json() == {"prefilled": False, "reason": "upstream_error"}
    assert upstream.closed is True


def test_prefill_task_cancellation_releases_lease_and_client(monkeypatch):
    class _Lease:
        released = False

        def release(self):
            self.released = True

    class _Reservation:
        def __init__(self, lease):
            self.lease = lease

        def lease_nowait(self):
            return self.lease

    class _BlockingClient(_Client):
        def __init__(self):
            super().__init__()
            self.started = asyncio.Event()

        async def post(self, url, **kwargs):
            self.calls.append({"url": url, **kwargs})
            self.started.set()
            await asyncio.Future()

    class _ConnectedRequest:
        async def is_disconnected(self):
            return False

    lease = _Lease()
    upstream = _BlockingClient()
    monkeypatch.setattr(inference_route, "get_llama_cpp_backend", _Backend)
    monkeypatch.setattr(inference_route, "_cancelable_nonstreaming_client", lambda: upstream)
    monkeypatch.setattr(
        inference_route,
        "_openai_llama_admission_reserve",
        lambda **_kwargs: (_Reservation(lease), object()),
    )
    payload = inference_route.ChatCompletionRequest.model_validate(_payload())

    async def run():
        task = asyncio.create_task(
            inference_route.prefill_chat_cache(
                payload,
                _ConnectedRequest(),
                current_subject = "test-user",
            )
        )
        await upstream.started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(run())
    assert upstream.closed is True
    assert lease.released is True


def test_prefill_route_is_tracked_by_inference_lifecycle():
    from core.inference.llama_keepwarm import _is_inference_path
    assert _is_inference_path("/api/inference/chat/prefill") is True
