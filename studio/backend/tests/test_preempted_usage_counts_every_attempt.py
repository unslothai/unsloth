# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The usage a paused chat reports covers every attempt, not just the last one.

A preemption closes the upstream request and re-opens it with the partial moved into the
prompt (``continue_final_message``), so llama-server counts each attempt on its own and
the final usage chunk the client receives described the LAST request alone: chats that
streamed 8000 plus characters through the GUI reported 259, 151, 1 and 401 completion
tokens, the last of which was whatever the tail happened to be. Every attempt decoded real
tokens and the user saw all of them, so the reported ``completion_tokens`` is their sum.

``prompt_tokens`` deliberately stays the last attempt's: each resume re-sends the same
conversation with the partial appended, so the earlier prompts are prefixes of the final
one and adding them would count the same conversation several times over.
"""

from __future__ import annotations

import contextlib
import copy
import json
import threading

import pytest

from core.inference import llama_preemption as preemption
from core.inference.llama_cpp import LlamaCppBackend


def _delta(content: str, **extra) -> str:
    chunk = {"choices": [{"index": 0, "delta": {"content": content}}], **extra}
    return "data: " + json.dumps(chunk) + "\n"


def _usage(prompt_tokens: int, completion_tokens: int) -> str:
    chunk = {
        "choices": [],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }
    return "data: " + json.dumps(chunk) + "\n"


def _finish(reason: str = "stop") -> str:
    return (
        "data: "
        + json.dumps({"choices": [{"index": 0, "delta": {}, "finish_reason": reason}]})
        + "\n"
    )


def _done() -> str:
    return "data: [DONE]\n"


class _Upstream:
    """A fake llama-server that pauses partway through the attempts it is told to."""

    def __init__(self, monkeypatch, streams, *, signal, pause_after):
        self.payloads: list[dict] = []
        self.signal = signal
        # attempt index -> how many data chunks it serves before the pause.
        self.pause_after = dict(pause_after)
        self._streams = [list(stream) for stream in streams]
        self.backend = LlamaCppBackend.__new__(LlamaCppBackend)
        backend = self.backend
        backend._process = object()
        backend._healthy = True
        backend._port = 48851
        backend._api_key = None
        backend._effective_context_length = 4096
        backend._supports_reasoning = False
        backend._reasoning_always_on = False
        backend._reasoning_style = "enable_thinking"
        backend._supports_preserve_thinking = False

        upstream = self

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
            upstream.payloads.append(copy.deepcopy(payload))
            yield type(
                "FakeResponse", (), {"status_code": 200, "chunks": upstream._streams.pop(0)}
            )()

        def fake_iter_text_cancellable(
            response,
            _cancel_event,
            first_token_deadline = None,
            preempt_event = None,
        ):
            attempt = len(upstream.payloads) - 1
            budget = upstream.pause_after.get(attempt)
            served = 0
            for chunk in response.chunks:
                yield chunk
                if chunk.startswith("data: {"):
                    served += 1
                    if budget is not None and served >= budget:
                        upstream.signal.request("kv_pressure")
                        raise preemption.LlamaStreamPreempted

        monkeypatch.setattr(backend, "_stream_with_retry", fake_stream_with_retry)
        monkeypatch.setattr(backend, "_iter_text_cancellable", fake_iter_text_cancellable)
        monkeypatch.setattr(backend, "_maybe_recover_from_mtp_crash", lambda *_a, **_k: False)


class _Policy:
    def should_preempt(self) -> bool:
        return False

    def on_preempted(self, checkpoint) -> None:
        pass

    def await_resume(self, timeout = None) -> bool:
        return True

    def on_resumed(self) -> None:
        pass


def _reported_usage(backend, *, signal):
    events = list(
        backend.generate_chat_completion(
            messages = [{"role": "user", "content": "write me an essay"}],
            cancel_event = threading.Event(),
            preempt_event = signal,
            preempt_policy = _Policy(),
        )
    )
    metadata = [
        event for event in events if isinstance(event, dict) and event.get("type") == "metadata"
    ]
    assert len(metadata) == 1, "a resumed turn reports its usage once, at the end"
    return metadata[0]["usage"]


class TestOnePause:
    def test_the_paused_attempts_tokens_are_counted(self, monkeypatch):
        """The attempt that was cut decoded 12 tokens and the client was shown them."""
        signal = preemption.PreemptSignal()
        upstream = _Upstream(
            monkeypatch,
            [
                [
                    _delta("Once "),
                    _delta("upon a ", timings = {"predicted_n": 12, "predicted_ms": 90.0}),
                    _finish("length"),
                    _done(),
                ],
                [_delta("time."), _usage(40, 7), _finish(), _done()],
            ],
            signal = signal,
            pause_after = {0: 2},
        )
        usage = _reported_usage(upstream.backend, signal = signal)
        assert len(upstream.payloads) == 2
        assert usage["completion_tokens"] == 19, (
            "12 decoded before the pause plus 7 after it. Reporting 7 alone is what a chat "
            "that streamed thousands of characters used to claim."
        )

    def test_the_prompt_is_the_last_attempts_and_the_total_agrees(self, monkeypatch):
        signal = preemption.PreemptSignal()
        upstream = _Upstream(
            monkeypatch,
            [
                [
                    _delta("Once "),
                    _delta("upon a ", timings = {"predicted_n": 12, "predicted_ms": 90.0}),
                    _finish("length"),
                    _done(),
                ],
                [_delta("time."), _usage(40, 7), _finish(), _done()],
            ],
            signal = signal,
            pause_after = {0: 2},
        )
        usage = _reported_usage(upstream.backend, signal = signal)
        # The resumed prompt carries the partial, so the first attempt's prompt is a prefix
        # of it; summing prompts would report the same conversation twice.
        assert usage["prompt_tokens"] == 40
        assert usage["total_tokens"] == 40 + 19

    def test_without_timings_the_chunks_decoded_are_the_estimate(self, monkeypatch):
        """A build that does not send per-chunk timings still must not report zero for the
        aborted attempt: one chunk is about one token, which is what the rest of the
        preemption path already assumes."""
        signal = preemption.PreemptSignal()
        upstream = _Upstream(
            monkeypatch,
            [
                [_delta("a"), _delta("b"), _delta("c"), _finish("length"), _done()],
                [_delta("d"), _usage(40, 5), _finish(), _done()],
            ],
            signal = signal,
            pause_after = {0: 3},
        )
        usage = _reported_usage(upstream.backend, signal = signal)
        assert usage["completion_tokens"] == 3 + 5


class TestSeveralPauses:
    def test_every_attempt_is_added(self, monkeypatch):
        signal = preemption.PreemptSignal()
        upstream = _Upstream(
            monkeypatch,
            [
                [_delta("one ", timings = {"predicted_n": 10}), _finish("length"), _done()],
                [_delta("two ", timings = {"predicted_n": 20}), _finish("length"), _done()],
                [_delta("three."), _usage(64, 3), _finish(), _done()],
            ],
            signal = signal,
            pause_after = {0: 1, 1: 1},
        )
        usage = _reported_usage(upstream.backend, signal = signal)
        assert len(upstream.payloads) == 3
        assert usage["completion_tokens"] == 33
        assert usage["total_tokens"] == 64 + 33


class TestATurnThatWasNeverPaused:
    def test_reports_exactly_what_the_server_said(self, monkeypatch):
        signal = preemption.PreemptSignal()
        upstream = _Upstream(
            monkeypatch,
            [[_delta("all in one go"), _usage(31, 9), _finish(), _done()]],
            signal = signal,
            pause_after = {},
        )
        usage = _reported_usage(upstream.backend, signal = signal)
        assert len(upstream.payloads) == 1
        assert usage == {"prompt_tokens": 31, "completion_tokens": 9, "total_tokens": 40}


@pytest.mark.parametrize(
    ("earlier", "expected"),
    [
        (0, {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7}),
        (11, {"prompt_tokens": 5, "completion_tokens": 13, "total_tokens": 18}),
    ],
)
def test_the_helper_leaves_an_unpaused_turn_alone(earlier, expected):
    from core.inference.llama_cpp import _usage_with_earlier_attempts
    usage = {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7}
    assert _usage_with_earlier_attempts(usage, earlier) == expected
