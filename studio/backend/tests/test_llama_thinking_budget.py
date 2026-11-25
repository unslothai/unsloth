# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A per-request thinking budget must reach every llama-server request a generation makes."""

from __future__ import annotations

import contextlib
import copy
import json
import sys
import threading
from pathlib import Path

import pytest

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from core.inference import cut_thought
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
    backend.counts = []
    backend.counted_tokens = 2000

    def count_chat_tokens(
        messages,
        _system = None,
        tools = None,
        **kwargs,
    ):
        backend.counts.append({"messages": messages, "tools": tools, **kwargs})
        return backend.counted_tokens

    monkeypatch.setattr(backend, "count_chat_tokens", count_chat_tokens)
    return backend


def _assert_budget(sent, requested):
    # No requested budget: the window's own, sized from the counted prompt.
    if requested is None:
        assert sent == cut_thought.window_budget(4096, 2000)
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
    assert any(m.get("role") == "tool" for m in backend.counts[-1]["messages"])
    for count in backend.counts:
        assert count["tools"] and count["chat_template_kwargs"] == {"enable_thinking": True}


@pytest.mark.parametrize("budget", [None, 128])
def test_tool_loop_final_pass_sends_the_budget(monkeypatch, budget):
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
            messages = [{"role": "user", "content": "hi"}],
            tools = [_LOOKUP_TOOL],
            enable_thinking = True,
            max_tool_iterations = 1,
            thinking_budget_tokens = budget,
        )
    )

    final = payloads[-1]
    assert final.get("tool_choice") == "none"
    _assert_budget(final.get("thinking_budget_tokens"), budget)
    assert any(m.get("role") == "tool" for m in backend.counts[-1]["messages"])
    assert backend.counts[-1]["tools"]
    assert backend.counts[-1]["chat_template_kwargs"] == {"enable_thinking": True}


# A template that keeps every earlier assistant turn's thought.
_KEEPS_THOUGHTS = (
    "{% for m in messages %}<|{{ m.role }}|>{% if m.reasoning_content %}"
    "<think>{{ m.reasoning_content }}</think>{% endif %}{{ m.content }}{% endfor %}<|assistant|>"
)


def _render_counter(template: str):
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    env = ImmutableSandboxedEnvironment(
        trim_blocks = True, lstrip_blocks = True, extensions = ["jinja2.ext.loopcontrols"]
    )
    compiled = env.from_string(template)

    def count(
        messages,
        _system = None,
        _tools = None,
        **kwargs,
    ) -> int:
        rendered = compiled.render(
            messages = messages,
            add_generation_prompt = True,
            bos_token = "<bos>",
            **(kwargs.get("chat_template_kwargs") or {}),
        )
        return len(rendered) // 4

    return count


@pytest.mark.parametrize("keeps_thoughts", [False, True])
def test_the_budget_counts_only_the_thoughts_the_template_keeps(monkeypatch, keeps_thoughts):
    payloads: list[dict] = []
    backend = _backend(monkeypatch, [[_sse({"content": "hi"}), _done()]], payloads)
    backend._effective_context_length = 16384
    template = (
        _KEEPS_THOUGHTS
        if keeps_thoughts
        else (Path(_BACKEND_DIR) / "assets" / "chat_templates" / "gemma-4.jinja").read_text(
            encoding = "utf-8"
        )
    )
    monkeypatch.setattr(backend, "count_chat_tokens", _render_counter(template))
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
    if keeps_thoughts:
        assert 0 < payload["thinking_budget_tokens"] < 16384 - 6 * 300 * 28 // 4
    else:
        assert "thinking_budget_tokens" not in payload


def test_the_budget_count_follows_the_request_it_prices(monkeypatch):
    payloads: list[dict] = []
    backend = _backend(monkeypatch, [[_sse({"content": "more"}), _done()]], payloads)
    cancel_event = threading.Event()
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "", "reasoning_content": "Partway"},
    ]

    list(
        backend.generate_chat_completion(
            messages = messages,
            enable_thinking = True,
            continue_final_message = True,
            cancel_event = cancel_event,
        )
    )

    [payload] = payloads
    assert payload["continue_final_message"] is True
    [count] = backend.counts
    assert count["continue_final_message"] is True
    assert not count["should_abort"]()
    cancel_event.set()
    assert count["should_abort"]()
    budget = backend._window_thinking_budget(
        messages, None, None, 7, True, None, cancel_event = cancel_event
    )
    assert budget == 7
    assert len(backend.counts) == 1
