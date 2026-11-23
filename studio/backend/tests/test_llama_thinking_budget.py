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


def _assert_budget(sent, requested):
    # No requested budget: the window's own, which leaves the answer room (cut_thought.window_budget).
    if requested is None:
        assert isinstance(sent, int) and 0 < sent < 4096
    else:
        assert sent == requested


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
    _assert_budget(payload.get("thinking_budget_tokens"), budget)
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
    for payload in payloads:
        _assert_budget(payload.get("thinking_budget_tokens"), budget)


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
    _assert_budget(payload.get("thinking_budget_tokens"), budget)


def _gemma4_prompt_tokens(
    messages,
    _system = None,
    _tools = None,
    **kwargs,
) -> int:
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    template = (Path(_BACKEND_DIR) / "assets" / "chat_templates" / "gemma-4.jinja").read_text()
    env = ImmutableSandboxedEnvironment(
        trim_blocks = True, lstrip_blocks = True, extensions = ["jinja2.ext.loopcontrols"]
    )
    rendered = env.from_string(template).render(
        messages = messages,
        add_generation_prompt = True,
        bos_token = "<bos>",
        **(kwargs.get("chat_template_kwargs") or {}),
    )
    return len(rendered) // 4


def test_earlier_thoughts_the_template_drops_leave_the_budget_alone(monkeypatch):
    payloads: list[dict] = []
    backend = _backend(monkeypatch, [[_sse({"content": "hi"}), _done()]], payloads)
    backend._effective_context_length = 16384
    monkeypatch.setattr(backend, "count_chat_tokens", _gemma4_prompt_tokens)
    messages = []
    for turn in range(6):
        messages.append({"role": "user", "content": f"Question {turn}?"})
        messages.append(
            {
                "role": "assistant",
                "content": f"Answer {turn}.",
                "reasoning_content": "Working it out step by step. " * 300,
            }
        )
    messages.append({"role": "user", "content": "And the last one?"})

    list(backend.generate_chat_completion(messages = messages, enable_thinking = True, max_tokens = 8192))

    [payload] = payloads
    assert "thinking_budget_tokens" not in payload
