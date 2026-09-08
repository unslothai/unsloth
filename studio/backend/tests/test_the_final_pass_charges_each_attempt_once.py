# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""What the final answering pass tells the preemptor about a paused attempt."""

from __future__ import annotations

import contextlib
import copy
import json
import threading

from core.inference import llama_preemption as preemption
from core.inference.llama_cpp import _TOKEN_REPORT_EVERY, LlamaCppBackend


_TOOL = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "search",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
    },
}

# Enough chunks that the batched reporter fires at least once per attempt, and enough
# beyond it that a carried-over counter reports a different number from a reset one.
_CHUNKS_PER_ATTEMPT = _TOKEN_REPORT_EVERY + 8


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


def _answer_stream(letter: str) -> list[str]:
    """One character per chunk, which is what a token delta usually is."""
    return [_delta(letter) for _ in range(_CHUNKS_PER_ATTEMPT)] + [_finish(), _done()]


def _tool_call(call_id: str = "call_search") -> list[str]:
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
                                    "id": call_id,
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
        self.checkpoints: list[preemption.StreamCheckpoint] = []

    def should_preempt(self) -> bool:
        return False

    def on_preempted(self, checkpoint) -> None:
        self.checkpoints.append(checkpoint)

    def await_resume(self, timeout = None) -> bool:
        return True

    def on_resumed(self) -> None:
        return None

    def on_declined(self) -> None:
        return None


class _Recorder:
    """A backend that preempts a chosen attempt after a set number of chunks."""

    def __init__(self, monkeypatch, streams, *, signal, pause_attempts, pause_after):
        self.payloads: list[dict] = []
        self.signal = signal
        self.pause_attempts = set(pause_attempts)
        self.pause_after = pause_after
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
            stream = recorder._streams.pop(0)
            yield type("FakeResponse", (), {"status_code": 200, "chunks": stream})()

        def fake_iter_text_cancellable(
            response,
            _cancel_event,
            first_token_deadline = None,
            preempt_event = None,
        ):
            attempt = len(recorder.payloads) - 1
            seen = 0
            for chunk in response.chunks:
                yield chunk
                if not chunk.startswith("data: {"):
                    continue
                seen += 1
                if attempt in recorder.pause_attempts and seen >= recorder.pause_after:
                    # Pressure noticed mid-stream, which is when it really is.
                    recorder.signal.request("kv_pressure")
                    raise preemption.LlamaStreamPreempted

        monkeypatch.setattr(backend, "_stream_with_retry", fake_stream_with_retry)
        monkeypatch.setattr(backend, "_iter_text_cancellable", fake_iter_text_cancellable)
        monkeypatch.setattr(backend, "_maybe_recover_from_mtp_crash", lambda *_a, **_k: False)
        monkeypatch.setattr(
            "core.inference.tools.execute_tool",
            lambda name, arguments, **_kwargs: "Linux kernel 6.10.",
        )


def _paused_final_run(monkeypatch, *, pause_attempts = (1,)):
    """A tool round, then a final pass that pauses partway and is resumed."""
    signal = preemption.PreemptSignal()
    policy = _RecordingPolicy()
    reports: list[int] = []
    recorder = _Recorder(
        monkeypatch,
        [
            _tool_call(),
            _answer_stream("a"),
            _answer_stream("b"),
        ],
        signal = signal,
        pause_attempts = pause_attempts,
        pause_after = _CHUNKS_PER_ATTEMPT,
    )
    events = list(
        recorder.backend.generate_chat_completion_with_tools(
            messages = [{"role": "user", "content": "what kernel is current?"}],
            tools = [_TOOL],
            cancel_event = threading.Event(),
            preempt_event = signal,
            preempt_policy = policy,
            max_tool_iterations = 1,
            permission_mode = "off",
            on_tokens = reports.append,
        )
    )
    return recorder, policy, reports, events


class TestTheLiveCountIsPerAttempt:
    def test_the_resumed_attempt_is_counted_from_zero(self, monkeypatch):
        recorder, policy, reports, _events = _paused_final_run(monkeypatch)
        assert len(recorder.payloads) == 3, (
            "expected the tool round, the paused final pass and its resume; "
            f"got {len(recorder.payloads)}"
        )
        assert policy.checkpoints, "the final pass never paused, so nothing here was tested"
        assert reports == [_TOKEN_REPORT_EVERY, _TOKEN_REPORT_EVERY], (
            "the count carried across the resume, so the sweep was told this chat had "
            "grown by both attempts while `note_replayed` had already added the first "
            "one to its baseline: the same tokens twice, and an eviction to make room "
            f"that was never taken. Reported {reports}"
        )

    def test_the_report_still_fires_at_all(self, monkeypatch):
        """The reset must not turn into never reporting: `observe` is the only thing that plans an
        eviction and `on_tokens` is the only thing that calls it.
        """
        _recorder, _policy, reports, _events = _paused_final_run(monkeypatch)
        assert len(reports) == 2


class TestThePauseChargeIsTheObservedCount:
    def test_the_charge_is_at_least_what_the_attempt_decoded(self, monkeypatch):
        _recorder, policy, _reports, _events = _paused_final_run(monkeypatch)
        checkpoint = policy.checkpoints[0]
        assert checkpoint.visible_text == "a" * _CHUNKS_PER_ATTEMPT
        # One character per chunk, so the four-characters-per-token estimate is a quarter
        # of the truth. That is the shape of the undercharge on token-dense text, where a
        # token really is about one character.
        estimate = len(checkpoint.visible_text) // 4
        assert checkpoint.charged_tokens == _CHUNKS_PER_ATTEMPT, (
            "the pause was charged the character estimate while the attempt's own chunk "
            f"count was known: {checkpoint.charged_tokens} against {_CHUNKS_PER_ATTEMPT}"
        )
        assert checkpoint.charged_tokens > estimate

    def test_the_estimate_still_floors_it(self, monkeypatch):
        """The estimate stays the lower bound, so chunks that are not one per token cannot undercharge."""
        _recorder, policy, _reports, _events = _paused_final_run(monkeypatch)
        checkpoint = policy.checkpoints[0]
        assert checkpoint.charged_tokens >= len(checkpoint.visible_text) // 4

    def test_the_charge_is_spent_from_the_caller_s_cap_once(self, monkeypatch):
        """The same figure spends `max_tokens` down, so it must be added once."""
        _recorder, _policy, _reports, events = _paused_final_run(monkeypatch)
        metadata = [event for event in events if event.get("type") == "metadata"]
        assert metadata, "the turn reported no usage at all"
        assert (metadata[-1].get("usage") or {}).get("completion_tokens", 0) >= (
            _CHUNKS_PER_ATTEMPT
        ), "the paused attempt's tokens were not reported to the caller"
