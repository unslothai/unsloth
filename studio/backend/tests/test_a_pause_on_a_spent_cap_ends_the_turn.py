# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A pause that lands on a fully charged cap ends the turn instead of resuming.

The resumed attempt would be floored at one token, so it replays the whole prompt through
admission to emit a token the caller never allowed. The plain path already stops there; the
tool round loop and the final answering pass are the two surfaces that did not.
"""

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

# Eight characters, which the pause charges as two tokens: exactly the cap below.
_CAP = 2
_SPENDS_THE_CAP = "abcdefgh"


def _delta(content: str) -> str:
    return "data: " + json.dumps({"choices": [{"index": 0, "delta": {"content": content}}]}) + "\n"


def _finish(reason: str = "stop") -> str:
    return (
        "data: "
        + json.dumps({"choices": [{"index": 0, "delta": {}, "finish_reason": reason}]})
        + "\n"
    )


def _done() -> str:
    return "data: [DONE]\n"


def _tool_call() -> list[str]:
    return [
        "data: "
        + json.dumps(
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_search",
                                    "type": "function",
                                    "function": {
                                        "name": "web_search",
                                        "arguments": json.dumps({"query": "kernel"}),
                                    },
                                }
                            ]
                        },
                    }
                ]
            }
        )
        + "\n",
        _done(),
    ]


class _RecordingPolicy:
    def __init__(self):
        self.events: list[str] = []
        self.checkpoints: list[preemption.StreamCheckpoint] = []

    def should_preempt(self) -> bool:
        return False

    def on_preempted(self, checkpoint):
        self.events.append("preempted")
        self.checkpoints.append(checkpoint)

    def await_resume(self, timeout = None) -> bool:
        self.events.append("awaited")
        return True

    def on_resumed(self) -> None:
        self.events.append("resumed")


class _Recorder:
    """A backend that pauses a chosen attempt after its first content delta."""

    def __init__(self, monkeypatch, streams, *, signal, pause_attempts):
        self.payloads: list[dict] = []
        self.signal = signal
        self.pause_attempts = set(pause_attempts)
        self._streams = [list(stream) for stream in streams]
        backend = LlamaCppBackend.__new__(LlamaCppBackend)
        backend._process = object()
        backend._healthy = True
        backend._port = 48853
        backend._api_key = None
        backend._effective_context_length = 4096
        backend._supports_reasoning = False
        backend._reasoning_always_on = False
        backend._reasoning_style = "enable_thinking"
        backend._supports_preserve_thinking = False
        self.backend = backend

        recorder = self

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
            recorder.payloads.append(copy.deepcopy(payload))
            assert recorder._streams, "the turn opened an upstream request it had no stream for"
            yield type(
                "FakeResponse", (), {"status_code": 200, "chunks": recorder._streams.pop(0)}
            )()

        def fake_iter_text_cancellable(
            response,
            _cancel_event,
            first_token_deadline = None,
            preempt_event = None,
        ):
            attempt = len(recorder.payloads) - 1
            for chunk in response.chunks:
                yield chunk
                if attempt in recorder.pause_attempts and chunk.startswith("data: {"):
                    recorder.signal.request("kv_pressure")
                    raise preemption.LlamaStreamPreempted

        monkeypatch.setattr(backend, "_stream_with_retry", fake_stream_with_retry)
        monkeypatch.setattr(backend, "_iter_text_cancellable", fake_iter_text_cancellable)
        monkeypatch.setattr(backend, "_maybe_recover_from_mtp_crash", lambda *_a, **_k: False)
        monkeypatch.setattr(
            "core.inference.tools.execute_tool",
            lambda name, arguments, **_kwargs: "Linux kernel 6.10.",
        )


def _run(recorder, *, signal, policy):
    return list(
        recorder.backend.generate_chat_completion_with_tools(
            messages = [{"role": "user", "content": "what kernel is current?"}],
            tools = [_TOOL],
            cancel_event = threading.Event(),
            preempt_event = signal,
            preempt_policy = policy,
            max_tokens = _CAP,
            max_tool_iterations = 1,
            permission_mode = "off",
        )
    )


def _metadata(events) -> dict:
    metas = [event for event in events if event.get("type") == "metadata"]
    assert metas, "the turn ended without a finish reason at all"
    return metas[-1]


class TestTheRoundLoopStopsOnASpentCap:
    def test_it_does_not_reopen_the_stream_for_one_more_token(self, monkeypatch):
        signal = preemption.PreemptSignal()
        policy = _RecordingPolicy()
        recorder = _Recorder(
            monkeypatch,
            [[_delta(_SPENDS_THE_CAP), _finish(), _done()]],
            signal = signal,
            pause_attempts = (0,),
        )
        events = _run(recorder, signal = signal, policy = policy)

        assert len(recorder.payloads) == 1, (
            "the round loop resumed a turn whose cap was already fully charged, replaying "
            f"the prompt to decode past it; payloads: {len(recorder.payloads)}"
        )
        assert policy.events == ["preempted"], (
            "the participant waited for room it could not use, and was told it resumed; "
            f"got {policy.events}"
        )
        assert {"type": "preempt", "state": "resumed"} not in events
        meta = _metadata(events)
        assert meta["finish_reason"] == "length"
        assert meta["usage"]["completion_tokens"] == _CAP


class TestTheFinalPassStopsOnASpentCap:
    def test_it_does_not_reopen_the_stream_for_one_more_token(self, monkeypatch):
        signal = preemption.PreemptSignal()
        policy = _RecordingPolicy()
        recorder = _Recorder(
            monkeypatch,
            [_tool_call(), [_delta(_SPENDS_THE_CAP), _finish(), _done()]],
            signal = signal,
            pause_attempts = (1,),
        )
        events = _run(recorder, signal = signal, policy = policy)

        assert len(recorder.payloads) == 2, (
            "the final pass resumed a turn whose cap was already fully charged, replaying "
            f"the prompt to decode past it; payloads: {len(recorder.payloads)}"
        )
        assert policy.events == ["preempted"], (
            "the participant waited for room it could not use, and was told it resumed; "
            f"got {policy.events}"
        )
        assert {"type": "preempt", "state": "resumed"} not in events
        meta = _metadata(events)
        assert meta["finish_reason"] == "length"
        assert meta["usage"]["completion_tokens"] == _CAP
