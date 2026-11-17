# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import contextlib
import json

from fastapi import FastAPI
from fastapi.testclient import TestClient

from auth.authentication import get_current_subject
from core.inference.llama_cpp import LlamaCppBackend
import routes.inference as inference_route
from .llama_backend_double import FakeLlamaCppBackend

_ANSWER = 'Use re.sub(r"<think>.*?</think>", "", text) to drop the block.'


def _sse(delta: dict) -> str:
    return "data: " + json.dumps({"choices": [{"index": 0, "delta": delta}]}) + "\n"


def _llama_server(monkeypatch, frames):
    real = LlamaCppBackend.__new__(LlamaCppBackend)
    real._process = object()
    real._healthy = True
    real._port = 48847
    real._api_key = None
    real._effective_context_length = 4096
    real._supports_reasoning = True
    real._reasoning_always_on = True
    real._reasoning_style = "enable_thinking"
    real._supports_preserve_thinking = False

    @contextlib.contextmanager
    def fake_stream(
        _client,
        _url,
        _payload,
        _cancel_event,
        headers = None,
        first_token_deadline = None,
    ):
        yield type("R", (), {"status_code": 200, "chunks": frames, "close": lambda self: None})()

    monkeypatch.setattr(real, "_stream_with_retry", fake_stream)
    monkeypatch.setattr(
        real,
        "_iter_text_cancellable",
        lambda response, _c, first_token_deadline = None: iter(response.chunks),
    )
    monkeypatch.setattr(real, "_maybe_recover_from_mtp_crash", lambda *_a, **_k: False)
    return real


class _ThinkingGguf(FakeLlamaCppBackend):
    reasoning_always_on = True
    supports_reasoning = True

    def __init__(self, real):
        self._real = real

    def generate_chat_completion(self, **kwargs):
        return self._real.generate_chat_completion(**kwargs)


def _stream(monkeypatch, frames):
    backend = _ThinkingGguf(_llama_server(monkeypatch, frames))
    monkeypatch.setattr(inference_route, "get_llama_cpp_backend", lambda: backend)
    monkeypatch.setattr(inference_route, "_effective_enable_tools", lambda payload: False)
    app = FastAPI()
    app.include_router(inference_route.router)
    app.dependency_overrides[get_current_subject] = lambda: "test-user"
    response = TestClient(app).post(
        "/chat/completions",
        json = {"messages": [{"role": "user", "content": "hi"}], "stream": True},
    )
    assert response.status_code == 200
    reasoning = content = ""
    for line in response.text.splitlines():
        if not line.startswith("data: {"):
            continue
        for choice in json.loads(line[len("data: ") :]).get("choices") or []:
            delta = choice.get("delta") or {}
            reasoning += delta.get("reasoning_content") or ""
            content += delta.get("content") or ""
    return reasoning, content


def test_literal_think_tags_in_the_answer_stay_answer_text(monkeypatch):
    frames = [_sse({"reasoning_content": "Plan the regex."})]
    frames += [_sse({"content": piece}) for piece in (_ANSWER[:12], _ANSWER[12:30], _ANSWER[30:])]
    frames.append("data: [DONE]\n")

    reasoning, content = _stream(monkeypatch, frames)

    assert reasoning == "Plan the regex."
    assert content == _ANSWER


def test_a_trace_that_quotes_the_closing_tag_stays_reasoning(monkeypatch):
    trace = "I'll match <think> up to </think> lazily."
    frames = [_sse({"reasoning_content": trace[:20]}), _sse({"reasoning_content": trace[20:]})]
    frames += [_sse({"content": "Here it is."}), "data: [DONE]\n"]

    reasoning, content = _stream(monkeypatch, frames)

    assert reasoning == trace
    assert content == "Here it is."


def test_a_reply_cut_off_while_thinking_leaves_no_stray_closing_tag(monkeypatch):
    frames = [_sse({"reasoning_content": "Still thinking about <think> tags"})]
    frames.append(
        "data: "
        + json.dumps({"choices": [{"index": 0, "delta": {}, "finish_reason": "length"}]})
        + "\n"
    )
    frames.append("data: [DONE]\n")

    reasoning, content = _stream(monkeypatch, frames)

    assert reasoning == "Still thinking about <think> tags"
    assert content == ""


def test_a_later_reasoning_segment_stays_reasoning(monkeypatch):
    frames = [
        _sse({"reasoning_content": "Plan."}),
        _sse({"content": "Literal <think>x</think> first. "}),
        _sse({"reasoning_content": "Check."}),
        _sse({"content": _ANSWER}),
        "data: [DONE]\n",
    ]

    reasoning, content = _stream(monkeypatch, frames)

    assert reasoning == "Plan.Check."
    assert content == "Literal <think>x</think> first. " + _ANSWER


def test_a_trace_quoting_tool_tags_keeps_the_answer_with_tools_on(monkeypatch):
    trace = "Quote </think> and </function>."

    class _ToolGguf(FakeLlamaCppBackend):
        supports_tools = True
        reasoning_always_on = True
        supports_reasoning = True

        def _request_reasoning_kwargs(self, *args, **kwargs):
            return {}

        def generate_chat_completion_with_tools(self, **kwargs):
            prov = kwargs["reasoning_provenance"]
            prov["wrapped"] += 1
            prov.setdefault("wraps", []).append({"len": len(trace)})
            yield {"type": "content", "text": "<think>" + trace}
            yield {"type": "content", "text": "<think>" + trace + "</think>OK"}
            yield {"type": "metadata", "finish_reason": "stop"}

    monkeypatch.setattr(inference_route, "get_llama_cpp_backend", lambda: _ToolGguf())
    monkeypatch.setattr(inference_route, "_effective_enable_tools", lambda payload: True)

    async def _select(payload, **_kwargs):
        return [{"type": "function", "function": {"name": "web_search"}}]

    monkeypatch.setattr(inference_route, "_select_request_tools", _select)
    app = FastAPI()
    app.include_router(inference_route.router)
    app.dependency_overrides[get_current_subject] = lambda: "test-user"
    response = TestClient(app).post(
        "/chat/completions",
        json = {
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
            "enable_tools": True,
        },
        headers = {"X-Unsloth-Events": "1"},
    )
    assert response.status_code == 200, response.text
    reasoning = content = ""
    for line in response.text.splitlines():
        if not line.startswith("data: {"):
            continue
        for choice in json.loads(line[len("data: ") :]).get("choices") or []:
            delta = choice.get("delta") or {}
            reasoning += delta.get("reasoning_content") or ""
            content += delta.get("content") or ""

    assert reasoning == trace
    assert content == "OK"


def _stream_with_tools(monkeypatch, frames):
    real = _llama_server(monkeypatch, frames)

    class _ToolGguf(_ThinkingGguf):
        supports_tools = True

        def _request_reasoning_kwargs(self, *args, **kwargs):
            return {}

        def generate_chat_completion_with_tools(self, **kwargs):
            return self._real.generate_chat_completion_with_tools(**kwargs)

    monkeypatch.setattr(inference_route, "get_llama_cpp_backend", lambda: _ToolGguf(real))
    monkeypatch.setattr(inference_route, "_effective_enable_tools", lambda payload: True)

    async def _select(payload, **_kwargs):
        return [{"type": "function", "function": {"name": "web_search"}}]

    monkeypatch.setattr(inference_route, "_select_request_tools", _select)
    app = FastAPI()
    app.include_router(inference_route.router)
    app.dependency_overrides[get_current_subject] = lambda: "test-user"
    response = TestClient(app).post(
        "/chat/completions",
        json = {
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
            "enable_tools": True,
        },
        headers = {"X-Unsloth-Events": "1"},
    )
    assert response.status_code == 200, response.text
    reasoning = content = ""
    for line in response.text.splitlines():
        if not line.startswith("data: {"):
            continue
        for choice in json.loads(line[len("data: ") :]).get("choices") or []:
            delta = choice.get("delta") or {}
            reasoning += delta.get("reasoning_content") or ""
            content += delta.get("content") or ""
    return reasoning, content


def test_a_later_reasoning_segment_stays_reasoning_with_tools_on(monkeypatch):
    frames = [
        _sse({"reasoning_content": "Plan."}),
        _sse({"content": "Literal <think>x</think> first. "}),
        _sse({"reasoning_content": "Check."}),
        _sse({"content": "Answer"}),
        "data: [DONE]\n",
    ]

    reasoning, content = _stream_with_tools(monkeypatch, frames)

    assert reasoning == "Plan.Check."
    assert content == "Literal <think>x</think> first. Answer"


def test_a_trace_quoting_a_tool_call_keeps_the_answer_with_tools_on(monkeypatch):
    trace = "Quote </think> and <tool_call>{}</tool_call>."
    frames = [_sse({"reasoning_content": trace}), _sse({"content": "OK"}), "data: [DONE]\n"]

    reasoning, content = _stream_with_tools(monkeypatch, frames)

    assert reasoning == trace
    assert content == "OK"
