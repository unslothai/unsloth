# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A pause inside a textual tool call replays the prose, not the half-written call."""

from __future__ import annotations

import contextlib
import copy
import json
import threading

from core.inference import llama_preemption as preemption
from core.inference.llama_cpp import LlamaCppBackend
import pytest

pytestmark = pytest.mark.usefixtures("preemption_opted_in")

_TOOL = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "search",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
}

_CALL = '<tool_call>{"name": "web_search", "arguments": {"query": "kernel"}}</tool_call>'


def _delta(content: str) -> str:
    return "data: " + json.dumps({"choices": [{"index": 0, "delta": {"content": content}}]}) + "\n"


def _finish() -> str:
    return (
        "data: "
        + json.dumps({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})
        + "\n"
    )


def _done() -> str:
    return "data: [DONE]\n"


def _run(monkeypatch, streams, *, pause_after: int):
    """Streams in order; the first attempt pauses after its first ``pause_after`` data chunks."""
    signal = preemption.PreemptSignal()
    payloads: list[dict] = []
    executed: list[dict] = []
    pending = [list(s) for s in streams]
    backend = LlamaCppBackend.__new__(LlamaCppBackend)
    backend._process = object()
    backend._healthy = True
    backend._port = 48861
    backend._api_key = None
    backend._effective_context_length = 4096
    backend._supports_reasoning = False
    backend._reasoning_always_on = False
    backend._reasoning_style = "enable_thinking"
    backend._supports_preserve_thinking = False

    @contextlib.contextmanager
    def fake_stream_with_retry(
        _client,
        _url,
        payload,
        _cancel_event,
        headers = None,
        first_token_deadline = None,
        preempt_event = None,
    ):
        payloads.append(copy.deepcopy(payload))
        yield type("FakeResponse", (), {"status_code": 200, "chunks": pending.pop(0)})()

    def fake_iter_text_cancellable(
        response,
        _cancel_event,
        first_token_deadline = None,
        preempt_event = None,
    ):
        attempt = len(payloads) - 1
        seen = 0
        for chunk in response.chunks:
            yield chunk
            if chunk.startswith("data: {"):
                seen += 1
            if attempt == 0 and seen == pause_after:
                signal.request("kv_pressure")
                raise preemption.LlamaStreamPreempted

    class _Policy:
        def should_preempt(self):
            return False

        def on_preempted(self, checkpoint):
            pass

        def await_resume(self, timeout = None):
            return True

        def on_resumed(self):
            pass

    def execute(name, arguments, **_kwargs):
        executed.append({"name": name, "arguments": arguments})
        return "Linux kernel 6.10."

    monkeypatch.setattr(backend, "_stream_with_retry", fake_stream_with_retry)
    monkeypatch.setattr(backend, "_iter_text_cancellable", fake_iter_text_cancellable)
    monkeypatch.setattr(backend, "_maybe_recover_from_mtp_crash", lambda *_a, **_k: False)
    monkeypatch.setattr("core.inference.tools.execute_tool", execute)
    events = list(
        backend.generate_chat_completion_with_tools(
            messages = [{"role": "user", "content": "what kernel is current?"}],
            tools = [_TOOL],
            cancel_event = threading.Event(),
            preempt_event = signal,
            preempt_policy = _Policy(),
            max_tool_iterations = 1,
            permission_mode = "off",
        )
    )
    return payloads, executed, events


def test_the_fragment_is_dropped_and_the_call_runs_once(monkeypatch):
    payloads, executed, _events = _run(
        monkeypatch,
        [
            [
                _delta("First, some context. "),
                _delta('<tool_call>{"name": "web_search", '),
                _delta('"arguments": {"query":'),
                _finish(),
                _done(),
            ],
            [_delta(_CALL), _finish(), _done()],
            [_delta("It is 6.10."), _finish(), _done()],
        ],
        pause_after = 3,
    )
    resumed = payloads[1]
    trailing = resumed["messages"][-1]
    assert trailing["role"] == "assistant"
    assert "<tool_call>" not in trailing["content"], trailing["content"]
    assert trailing["content"].startswith("First, some context.")
    assert [call["name"] for call in executed] == ["web_search"], executed


def test_plain_prose_is_still_replayed_verbatim(monkeypatch):
    payloads, executed, _events = _run(
        monkeypatch,
        [
            [_delta("Once upon "), _delta("a time"), _finish(), _done()],
            [_delta(" there was a cat."), _finish(), _done()],
        ],
        pause_after = 2,
    )
    assert payloads[1]["messages"][-1]["content"] == "Once upon a time"
    assert executed == []
