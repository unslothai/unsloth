# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""What the final answering pass tells the preemptor about a paused attempt."""

from __future__ import annotations

import threading

from core.inference import llama_preemption as preemption
from core.inference.llama_cpp import _TOKEN_REPORT_EVERY

from .preempt_fakes import (
    DecliningPolicy as _RecordingPolicy,
    PreemptRecorder,
    delta as _delta,
    done as _done,
    finish as _finish,
    tool_call as _tool_call,
    web_search_tool,
)

_TOOL = web_search_tool()


def _Recorder(monkeypatch, streams, *, signal, pause_attempts, pause_after):
    return PreemptRecorder(
        monkeypatch,
        streams,
        signal = signal,
        pause_attempts = pause_attempts,
        pause_after = pause_after,
        port = 48853,
        execute_tool = True,
    )


# Enough chunks that the batched reporter fires at least once per attempt, and enough
# beyond it that a carried-over counter reports a different number from a reset one.
_CHUNKS_PER_ATTEMPT = _TOKEN_REPORT_EVERY + 8


def _answer_stream(letter: str) -> list[str]:
    return [_delta(letter) for _ in range(_CHUNKS_PER_ATTEMPT)] + [_finish(), _done()]


def _paused_final_run(monkeypatch, *, pause_attempts = (1,)):
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
        _recorder, policy, _reports, _events = _paused_final_run(monkeypatch)
        checkpoint = policy.checkpoints[0]
        assert checkpoint.charged_tokens >= len(checkpoint.visible_text) // 4

    def test_the_charge_is_spent_from_the_caller_s_cap_once(self, monkeypatch):
        _recorder, _policy, _reports, events = _paused_final_run(monkeypatch)
        metadata = [event for event in events if event.get("type") == "metadata"]
        assert metadata, "the turn reported no usage at all"
        assert (metadata[-1].get("usage") or {}).get("completion_tokens", 0) >= (
            _CHUNKS_PER_ATTEMPT
        ), "the paused attempt's tokens were not reported to the caller"
