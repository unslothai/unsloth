# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""/v1/completions served by a resident managed vLLM/SGLang engine (unslothai/unsloth#7703)."""

import asyncio
import json
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from routes import inference as api
from routes import managed_engine_chat
from core.inference.api_monitor import ApiMonitor
from utils.hardware import DeviceType

LOGPROBS = {
    "tokens": ["2", "+", "2", "=", "4"],
    "token_logprobs": [None, -1.5, -0.2, -0.1, -0.05],
    "top_logprobs": [None, {"+": -1.5}, {"2": -0.2}, {"=": -0.1}, {"4": -0.05}],
    "text_offset": [0, 1, 2, 3, 4],
}


class _Request:
    state = SimpleNamespace()
    url = SimpleNamespace(path = "/v1/completions")
    method = "POST"
    headers: dict = {}
    scope: dict = {}

    def __init__(self, body):
        self._body = body

    async def json(self):
        return self._body

    async def is_disconnected(self):
        return False


def _orchestrator(engine = "vllm", alive = True):
    backend = SimpleNamespace(
        active_model_name = "unsloth/Qwen3-0.6B",
        models = {"unsloth/Qwen3-0.6B": {"engine": engine}},
        _managed_engine = SimpleNamespace(
            model = "/models/qwen3",
            context = 4096,
            base_url = "http://engine.test",
            headers = {"Authorization": "Bearer private"},
            engine = engine,
            alive = lambda: alive,
        ),
    )
    return backend


@pytest.fixture
def engine(monkeypatch):
    httpx = managed_engine_chat.httpx
    requests = []

    async def handle(request):
        assert request.url.path == "/v1/completions"
        assert request.headers["Authorization"] == "Bearer private"
        body = json.loads(request.content)
        requests.append(body)
        if body.get("prompt") == "bad":
            return httpx.Response(400, json = {"error": {"message": "invalid prompt"}})
        usage = {"prompt_tokens": 4, "completion_tokens": 1, "total_tokens": 5}
        choice = {"index": 0, "text": "2+2=4", "logprobs": LOGPROBS, "finish_reason": "length"}
        if not body.get("stream"):
            return httpx.Response(
                200,
                json = {
                    "id": "cmpl-1",
                    "object": "text_completion",
                    "model": body["model"],
                    "choices": [choice],
                    "usage": usage,
                },
            )
        events = [
            {
                "id": "cmpl-1",
                "model": body["model"],
                "choices": [{**choice, "finish_reason": None}],
            },
            {"id": "cmpl-1", "model": body["model"], "choices": [], "usage": usage},
        ]
        text = "".join("data: " + json.dumps(e) + "\n\n" for e in events) + "data: [DONE]\n\n"
        return httpx.Response(200, text = text)

    client = httpx.AsyncClient
    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **kwargs: client(transport = httpx.MockTransport(handle), **kwargs),
    )
    monitor = ApiMonitor(max_entries = 8)
    monkeypatch.setattr(api, "api_monitor", monitor)
    monkeypatch.setattr(
        api, "get_llama_cpp_backend", lambda: SimpleNamespace(is_loaded = False, context_length = None)
    )
    backend = _orchestrator()
    monkeypatch.setattr(api, "_peek_inference_backend", lambda: backend)
    monkeypatch.setattr(api, "_should_validate_before_switch", lambda: False)
    monkeypatch.setattr("utils.hardware.get_device", lambda: DeviceType.CUDA)

    async def no_switch(*_args, **_kwargs):
        return None

    monkeypatch.setattr(api, "_maybe_auto_switch_model", no_switch)
    return SimpleNamespace(backend = backend, requests = requests, monitor = monitor)


def _call(body):
    async def run():
        response = await api.openai_completions(_Request(body), current_subject = "u")
        if body.get("stream"):
            return [chunk async for chunk in response.body_iterator]
        return json.loads(response.body)

    return asyncio.run(run())


def test_non_stream_relays_logprobs_and_echo(engine):
    body = {
        "model": "unsloth/Qwen3-0.6B",
        "prompt": "2+2=",
        "logprobs": 2,
        "echo": True,
        "max_tokens": 1,
    }
    result = _call(body)
    [sent] = engine.requests
    assert sent == {**body, "model": "/models/qwen3"}
    assert result["model"] == "unsloth/Qwen3-0.6B"
    assert result["choices"][0]["logprobs"] == LOGPROBS
    [entry] = engine.monitor.snapshot()
    assert entry["status"] == "completed"
    assert engine.monitor.active_count() == 0


@pytest.mark.parametrize("include_usage", [False, True])
def test_stream_relays_logprobs_and_honours_usage_opt_in(engine, include_usage):
    body = {"prompt": "2+2=", "logprobs": 1, "stream": True}
    if include_usage:
        body["stream_options"] = {"include_usage": True}
    chunks = _call(body)
    [sent] = engine.requests
    assert sent["stream_options"] == {"include_usage": True}
    events = [
        json.loads(chunk[5:])
        for chunk in b"".join(chunks).decode().split("\n\n")
        if chunk.startswith("data:") and chunk[5:].strip() != "[DONE]"
    ]
    assert events[0]["model"] == "unsloth/Qwen3-0.6B"
    assert events[0]["choices"][0]["logprobs"] == LOGPROBS
    assert any(e.get("usage") for e in events) == include_usage
    assert b"data: [DONE]" in b"".join(chunks)
    assert engine.monitor.active_count() == 0


def test_engine_rejection_keeps_its_status(engine):
    with pytest.raises(HTTPException) as caught:
        _call({"prompt": "bad"})
    assert caught.value.status_code == 400
    [entry] = engine.monitor.snapshot()
    assert entry["status"] == "error"
    assert engine.monitor.active_count() == 0


def test_echo_reaches_a_resident_engine_before_any_switch(engine, monkeypatch):
    monkeypatch.setattr(api, "_should_validate_before_switch", lambda: True)
    _call({"prompt": "hi", "echo": True, "logprobs": 1, "max_tokens": 1})
    assert engine.requests[0]["echo"] is True


def test_dead_engine_and_no_engine_still_503(engine, monkeypatch):
    for backend in (_orchestrator(alive = False), SimpleNamespace(active_model_name = None, models = {})):
        monkeypatch.setattr(api, "_peek_inference_backend", lambda backend = backend: backend)
        with pytest.raises(HTTPException) as caught:
            _call({"prompt": "hi"})
        assert caught.value.status_code == 503
    assert engine.requests == []


def test_safetensors_model_without_engine_still_503(engine, monkeypatch):
    backend = _orchestrator()
    backend.models["unsloth/Qwen3-0.6B"] = {}
    monkeypatch.setattr(api, "_peek_inference_backend", lambda: backend)
    with pytest.raises(HTTPException) as caught:
        _call({"prompt": "hi"})
    assert caught.value.status_code == 503
    assert engine.requests == []


def test_echo_still_refused_without_a_resident_engine(engine, monkeypatch):
    monkeypatch.setattr(api, "_should_validate_before_switch", lambda: True)
    monkeypatch.setattr(api, "_peek_inference_backend", lambda: None)
    with pytest.raises(HTTPException) as caught:
        _call({"prompt": "hi", "echo": True})
    assert caught.value.status_code == 400
    assert engine.requests == []


def test_loaded_gguf_wins_over_engine(engine, monkeypatch):
    captured = []
    monkeypatch.setattr(
        api,
        "get_llama_cpp_backend",
        lambda: SimpleNamespace(
            is_loaded = True,
            base_url = "http://llama.test",
            context_length = 4096,
            model_identifier = "gguf",
        ),
    )
    monkeypatch.setattr(api, "_cancelable_nonstreaming_client", lambda: _PostClient(captured))
    monkeypatch.setattr(api, "_fill_recommended_sampling_completions", lambda *a, **k: None)
    _call({"prompt": "hi"})
    assert captured == ["http://llama.test/v1/completions"]
    assert engine.requests == []


class _PostClient:
    def __init__(self, captured):
        self.captured = captured

    async def post(self, url, **_kwargs):
        self.captured.append(url)
        return managed_engine_chat.httpx.Response(
            200, json = {"id": "cmpl-x", "choices": [{"text": "ok", "index": 0}]}
        )

    async def aclose(self):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        pass


def test_unstarted_stream_releases_tracking_when_the_upstream_close_fails(engine, monkeypatch):
    """The SSE wrapper swallows this hook's error, so a skipped release leaves the model busy."""
    from state import active_generations

    async def boom(self):
        raise RuntimeError("transport gone")

    active_generations.reset_for_tests()

    async def run():
        response = await api.openai_completions(
            _Request({"prompt": "2+2=", "stream": True}), current_subject = "u"
        )
        assert active_generations.count() == 1
        monkeypatch.setattr(managed_engine_chat.httpx.Response, "aclose", boom)
        with pytest.raises(RuntimeError):
            await response._unstarted_cleanup()

    asyncio.run(run())
    assert active_generations.count() == 0
