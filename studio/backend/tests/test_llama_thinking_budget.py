# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A per-request thinking budget must reach every llama-server request a generation makes."""

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


def _backend(monkeypatch, streams: list, payloads: list) -> LlamaCppBackend:
    backend = LlamaCppBackend.__new__(LlamaCppBackend)
    backend._process = object()
    backend._healthy = True
    backend._port = 48852
    backend._api_key = None
    backend._effective_context_length = 4096
    backend._supports_reasoning = True
    backend._reasoning_always_on = False
    backend._reasoning_style = "enable_thinking"
    backend._supports_preserve_thinking = False

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


@pytest.mark.parametrize("budget", [None, 128])
def test_plain_generation_sends_the_budget(monkeypatch, budget):
    payloads: list[dict] = []
    backend = _backend(monkeypatch, [[_sse({"content": "hi"}), _done()]], payloads)

    list(
        backend.generate_chat_completion(
            messages = [{"role": "user", "content": "hi"}],
            enable_thinking = True,
            thinking_budget_tokens = budget,
        )
    )

    [payload] = payloads
    assert payload.get("thinking_budget_tokens") == budget
    assert payload["chat_template_kwargs"] == {"enable_thinking": True}


@pytest.mark.parametrize("budget", [None, 128])
def test_every_tool_loop_round_sends_the_budget(monkeypatch, budget):
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
            thinking_budget_tokens = budget,
        )
    )

    assert len(payloads) == 2
    assert [p.get("thinking_budget_tokens") for p in payloads] == [budget, budget]


@pytest.mark.parametrize("budget", [None, 128])
def test_tool_loop_final_pass_sends_the_budget(monkeypatch, budget):
    payloads: list[dict] = []
    backend = _backend(monkeypatch, [[_sse({"content": "done"}), _done()]], payloads)

    list(
        backend.generate_chat_completion_with_tools(
            messages = [{"role": "user", "content": "hi"}],
            tools = [],
            enable_thinking = True,
            max_tool_iterations = 0,
            thinking_budget_tokens = budget,
        )
    )

    [payload] = payloads
    assert payload.get("thinking_budget_tokens") == budget
