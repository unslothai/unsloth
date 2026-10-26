# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A final answering pass paused after its thought replays its answer, not its display."""

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
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
    },
}


def _chunk(delta: dict, finish = None) -> str:
    choice = {"index": 0, "delta": delta}
    if finish:
        choice["finish_reason"] = finish
    return "data: " + json.dumps({"choices": [choice]}) + "\n"


def _end() -> list[str]:
    return [_chunk({}, "stop"), "data: [DONE]\n"]


def _tool_round() -> list[str]:
    call = {
        "index": 0,
        "id": "call_search",
        "type": "function",
        "function": {"name": "web_search", "arguments": json.dumps({"query": "kernel"})},
    }
    return [_chunk({"tool_calls": [call]}), "data: [DONE]\n"]


class _Policy:
    def __init__(self):
        self.checkpoints: list = []

    def should_preempt(self) -> bool:
        return False

    def on_preempted(self, checkpoint):
        self.checkpoints.append(checkpoint)

    def await_resume(self, timeout = None) -> bool:
        return True

    def on_resumed(self) -> None:
        pass


def _run(monkeypatch, streams, pause_after: dict):
    """``pause_after`` maps an attempt to the data chunk it is paused after."""
    signal = preemption.PreemptSignal()
    policy = _Policy()
    payloads: list[dict] = []
    queued = [list(stream) for stream in streams]
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
        yield type("FakeResponse", (), {"status_code": 200, "chunks": queued.pop(0)})()

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
                if pause_after.get(attempt) == seen:
                    signal.request("kv_pressure")
                    raise preemption.LlamaStreamPreempted

    monkeypatch.setattr(backend, "_stream_with_retry", fake_stream_with_retry)
    monkeypatch.setattr(backend, "_iter_text_cancellable", fake_iter_text_cancellable)
    monkeypatch.setattr(backend, "_maybe_recover_from_mtp_crash", lambda *_a, **_k: False)
    monkeypatch.setattr(
        "core.inference.tools.execute_tool", lambda name, arguments, **_kwargs: "Linux 6.10."
    )
    events = list(
        backend.generate_chat_completion_with_tools(
            messages = [{"role": "user", "content": "what kernel is current?"}],
            tools = [_TOOL],
            cancel_event = threading.Event(),
            preempt_event = signal,
            preempt_policy = policy,
            max_tool_iterations = 1,
            permission_mode = "off",
        )
    )
    return payloads, policy, events


def _trailing(payload: dict) -> dict:
    trailing = payload["messages"][-1]
    assert trailing["role"] == "assistant"
    return trailing


def test_the_resume_extends_the_answer_alone(monkeypatch):
    payloads, policy, _events = _run(
        monkeypatch,
        [
            _tool_round(),
            [
                _chunk({"reasoning_content": "Let me check."}),
                _chunk({"content": "The kernel"}),
                *_end(),
            ],
            [_chunk({"content": " is 6.10."}), *_end()],
        ],
        pause_after = {1: 2},
    )
    assert policy.checkpoints[0].visible_text == "The kernel"
    trailing = _trailing(payloads[2])
    assert "<think>" not in str(trailing.get("content")), trailing
    assert "Let me check." not in str(
        trailing.get("content")
    ), "the thought went back inside the answer it was extending"
    assert "The kernel" in str(trailing.get("content")), trailing


def test_a_second_pause_sends_only_what_is_new(monkeypatch):
    payloads, policy, events = _run(
        monkeypatch,
        [
            _tool_round(),
            [
                _chunk({"reasoning_content": "Let me check."}),
                _chunk({"content": "The kernel"}),
                *_end(),
            ],
            [_chunk({"content": " is"}), *_end()],
            [_chunk({"content": " 6.10."}), *_end()],
        ],
        pause_after = {1: 2, 2: 1},
    )
    assert [c.visible_text for c in policy.checkpoints] == ["The kernel", " is"]
    trailing = _trailing(payloads[3])
    content = str(trailing.get("content"))
    assert "<think>" not in content and "Let me check." not in content, trailing
    assert content.count("The kernel") == 1 and content.count(" is") == 1, trailing
    shown = [e["text"] for e in events if e.get("type") == "content"]
    assert shown and shown[-1].endswith("The kernel is 6.10."), shown[-1:]
