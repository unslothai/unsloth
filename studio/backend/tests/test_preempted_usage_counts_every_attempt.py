# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The usage a paused chat reports covers every attempt, not just the last one."""

from __future__ import annotations


import pytest

from core.inference import llama_preemption as preemption

from .preempt_fakes import (
    PreemptRecorder,
    RecordingPolicy as _Policy,
    delta as _delta,
    done as _done,
    finish as _finish,
    run_plain,
    usage as _usage,
)


def _Upstream(monkeypatch, streams, *, signal, pause_after):
    return PreemptRecorder(monkeypatch, streams, signal = signal, pause_after = pause_after)


def _reported_usage(backend, *, signal):
    events = run_plain(
        backend,
        signal = signal,
        policy = _Policy(),
        prompt = "write me an essay",
    )
    metadata = [
        event for event in events if isinstance(event, dict) and event.get("type") == "metadata"
    ]
    assert len(metadata) == 1, "a resumed turn reports its usage once, at the end"
    return metadata[0]["usage"]


class TestOnePause:
    def test_the_paused_attempts_tokens_are_counted(self, monkeypatch):
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
