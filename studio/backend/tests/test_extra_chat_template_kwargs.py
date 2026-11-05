# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A caller's own chat_template_kwargs (e.g. a model's ``verification`` tag) must reach llama-server
beside Studio's reasoning keys instead of being dropped (#16545)."""

from __future__ import annotations

import contextlib
import copy
import json
import sys
from pathlib import Path

import pytest

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from core.inference.llama_cpp import LlamaCppBackend
from models.inference import ChatCompletionRequest, ChatMessage, ResponsesRequest
from routes.inference import (
    _build_chat_request,
    _build_openai_passthrough_body,
    _extra_chat_template_kwargs,
    _normalize_chat_reasoning_controls,
)

_LOOKUP_TOOL = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Search the web.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
}


def _sse(delta: dict) -> str:
    return "data: " + json.dumps({"choices": [{"index": 0, "delta": delta}]}) + "\n"


def _done() -> str:
    return "data: [DONE]\n"


def _tool_call() -> str:
    return _sse(
        {
            "tool_calls": [
                {
                    "index": 0,
                    "id": "call_0",
                    "type": "function",
                    "function": {
                        "name": "web_search",
                        "arguments": json.dumps({"query": "q"}),
                    },
                }
            ]
        }
    )


def _reasoning_backend() -> LlamaCppBackend:
    backend = LlamaCppBackend.__new__(LlamaCppBackend)
    backend._supports_reasoning = True
    backend._reasoning_always_on = False
    backend._reasoning_style = "enable_thinking"
    backend._supports_preserve_thinking = False
    return backend


def _backend(monkeypatch, streams: list, payloads: list) -> LlamaCppBackend:
    backend = _reasoning_backend()
    backend._process = object()
    backend._healthy = True
    backend._port = 48853
    backend._api_key = None
    backend._effective_context_length = 4096

    @contextlib.contextmanager
    def fake_stream_with_retry(_client, _url, payload, _cancel_event, **_kwargs):
        payloads.append(copy.deepcopy(payload))
        yield type("FakeResponse", (), {"status_code": 200, "chunks": streams.pop(0)})()

    def fake_iter_text_cancellable(
        response,
        _cancel_event,
        first_token_deadline = None,
    ):
        yield from response.chunks

    monkeypatch.setattr(backend, "_stream_with_retry", fake_stream_with_retry)
    monkeypatch.setattr(backend, "_iter_text_cancellable", fake_iter_text_cancellable)
    monkeypatch.setattr(backend, "_maybe_recover_from_mtp_crash", lambda *_a, **_k: False)
    return backend


def _chat_payload(**fields) -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model = "default", messages = [{"role": "user", "content": "hi"}], **fields
    )


def test_plain_generation_sends_the_extra_keys(monkeypatch):
    payloads: list[dict] = []
    backend = _backend(monkeypatch, [[_sse({"content": "hi"}), _done()]], payloads)

    list(
        backend.generate_chat_completion(
            messages = [{"role": "user", "content": "hi"}],
            enable_thinking = True,
            chat_template_kwargs = {"verification": True},
        )
    )

    [payload] = payloads
    assert payload["chat_template_kwargs"] == {"verification": True, "enable_thinking": True}


def test_extra_keys_go_out_even_without_a_reasoning_control(monkeypatch):
    payloads: list[dict] = []
    backend = _backend(monkeypatch, [[_sse({"content": "hi"}), _done()]], payloads)

    list(
        backend.generate_chat_completion(
            messages = [{"role": "user", "content": "hi"}],
            chat_template_kwargs = {"verification": True},
        )
    )

    [payload] = payloads
    assert payload["chat_template_kwargs"] == {"verification": True}


def test_studio_reasoning_keys_win_over_the_callers(monkeypatch):
    payloads: list[dict] = []
    backend = _backend(monkeypatch, [[_sse({"content": "hi"}), _done()]], payloads)

    list(
        backend.generate_chat_completion(
            messages = [{"role": "user", "content": "hi"}],
            enable_thinking = True,
            chat_template_kwargs = {"enable_thinking": False, "verification": True},
        )
    )

    [payload] = payloads
    assert payload["chat_template_kwargs"] == {"enable_thinking": True, "verification": True}


def test_no_extra_keys_keeps_the_payload_unchanged(monkeypatch):
    payloads: list[dict] = []
    backend = _backend(monkeypatch, [[_sse({"content": "hi"}), _done()]], payloads)

    list(backend.generate_chat_completion(messages = [{"role": "user", "content": "hi"}]))

    [payload] = payloads
    assert "chat_template_kwargs" not in payload


def test_every_tool_loop_round_sends_the_extra_keys(monkeypatch):
    payloads: list[dict] = []
    backend = _backend(
        monkeypatch,
        [[_tool_call(), _done()], [_sse({"content": "done"}), _done()]],
        payloads,
    )
    monkeypatch.setattr(
        "core.inference.tools.execute_tool",
        lambda name, arguments, **_kwargs: "a result",
    )

    list(
        backend.generate_chat_completion_with_tools(
            messages = [{"role": "user", "content": "search"}],
            tools = [_LOOKUP_TOOL],
            enable_thinking = True,
            max_tool_iterations = 3,
            chat_template_kwargs = {"verification": True},
        )
    )

    assert len(payloads) == 2
    for payload in payloads:
        assert payload["chat_template_kwargs"]["verification"] is True
        assert payload["chat_template_kwargs"]["enable_thinking"] is True


def test_tool_loop_final_pass_sends_the_extra_keys(monkeypatch):
    payloads: list[dict] = []
    backend = _backend(monkeypatch, [[_sse({"content": "done"}), _done()]], payloads)

    list(
        backend.generate_chat_completion_with_tools(
            messages = [{"role": "user", "content": "hi"}],
            tools = [],
            enable_thinking = True,
            max_tool_iterations = 0,
            chat_template_kwargs = {"verification": True},
        )
    )

    [payload] = payloads
    assert payload["chat_template_kwargs"] == {"verification": True, "enable_thinking": True}


def test_the_route_lifts_reasoning_keys_and_keeps_the_rest():
    payload = _chat_payload(
        chat_template_kwargs = {"enable_thinking": False, "verification": True, "mode": "strict"}
    )

    _normalize_chat_reasoning_controls(payload)

    assert payload.enable_thinking is False
    assert _extra_chat_template_kwargs(payload) == {"verification": True, "mode": "strict"}


@pytest.mark.parametrize(
    "template_kwargs",
    [None, {}, {"enable_thinking": True}, "verification"],
)
def test_nothing_extra_to_forward(template_kwargs):
    fields = {} if template_kwargs is None else {"chat_template_kwargs": template_kwargs}
    payload = _chat_payload(**fields)

    _normalize_chat_reasoning_controls(payload)

    assert _extra_chat_template_kwargs(payload) is None


def test_client_tools_passthrough_sends_the_extra_keys():
    payload = _chat_payload(
        chat_template_kwargs = {"enable_thinking": True, "verification": True},
        tools = [_LOOKUP_TOOL],
    )
    _normalize_chat_reasoning_controls(payload)

    body = _build_openai_passthrough_body(
        payload, backend_ctx = 4096, llama_backend = _reasoning_backend()
    )

    assert body["chat_template_kwargs"] == {"verification": True, "enable_thinking": True}


def test_responses_requests_carry_the_extra_keys():
    payload = ResponsesRequest(
        input = "hi",
        chat_template_kwargs = {"enable_thinking": True, "verification": True},
    )
    messages = [ChatMessage(role = "user", content = "hi")]

    chat_req = _build_chat_request(payload, messages, stream = False)

    assert chat_req.enable_thinking is True
    assert _extra_chat_template_kwargs(chat_req) == {"verification": True}


def test_the_public_token_count_prices_the_extra_keys(monkeypatch):
    from .test_openai_auto_switch import _count_request, _count_tokens_backend, _counted_body

    _switched, counted = _count_tokens_backend(monkeypatch)
    _counted_body(
        _count_request(
            [{"role": "user", "content": "hi"}],
            chat_template_kwargs = {"enable_thinking": True, "verification": True},
        )
    )

    assert counted["chat_template_kwargs"] == {"verification": True, "enable_thinking": True}


def test_the_default_system_probe_renders_with_the_extra_keys(monkeypatch):
    import routes.inference as inference_route
    from utils.current_date_prompt_settings import template_system_turn
    from datetime import date

    backend = _reasoning_backend()
    backend._process = object()
    backend._healthy = True
    monkeypatch.setattr(inference_route, "get_llama_cpp_backend", lambda: backend)
    payload = _chat_payload(
        chat_template_kwargs = {
            "enable_thinking": True,
            "verification": True,
            "nested": {"a": 1},
            "documents": None,
        }
    )
    _normalize_chat_reasoning_controls(payload)

    controls = inference_route._template_controls(payload)

    assert dict(controls) == {"verification": True, "enable_thinking": True}
    template = (
        "{% if messages[0].role != 'system' %}<system>{{ 'VERIFY' if verification else 'PLAIN' }}</system>"
        "{% endif %}{% for m in messages %}<{{ m.role }}>{{ m.content }}</{{ m.role }}>{% endfor %}"
    )
    assert template_system_turn(template, date(2026, 1, 1), False, controls)[1] == "VERIFY"


def test_in_process_backends_probe_without_the_extras(monkeypatch):
    import routes.inference as inference_route

    backend = _reasoning_backend()
    backend._process = None
    monkeypatch.setattr(inference_route, "get_llama_cpp_backend", lambda: backend)
    payload = _chat_payload(chat_template_kwargs = {"enable_thinking": True, "verification": True})
    _normalize_chat_reasoning_controls(payload)

    assert dict(inference_route._template_controls(payload)) == {"enable_thinking": True}
