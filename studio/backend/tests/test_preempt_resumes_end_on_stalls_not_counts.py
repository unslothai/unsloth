# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A paused chat gives up only when its resumes stop producing output, not after a count."""

from __future__ import annotations

import json

from core.inference import llama_preemption as preemption
from core.inference.llama_preemption import (
    ControllerPreemptionPolicy,
    PreemptionController,
    PreemptSignal,
    StreamCheckpoint,
)
import pytest

from .test_llama_plain_chat_preempt_resume import (
    _RecordingPolicy,
    _Recorder,
    _delta,
    _done,
    _finish,
    _run,
)

pytestmark = pytest.mark.usefixtures("preemption_opted_in")

_PAUSES = 64


def _role_only() -> str:
    return "data: " + json.dumps({"choices": [{"index": 0, "delta": {"role": "assistant"}}]}) + "\n"


def _policy(gen_id: str) -> ControllerPreemptionPolicy:
    controller = PreemptionController(f"stall-{gen_id}")
    controller.configure(budget = 16384, kv_unified = True)
    signal = PreemptSignal()
    # No lease, so a granted wait answers at once and only the give-up rule is exercised.
    controller.register(gen_id, tokens = 100, signal = signal)
    return ControllerPreemptionPolicy(controller, gen_id, signal)


class TestThePolicy:
    def test_many_pauses_with_progress_keep_resuming(self):
        policy = _policy("busy")
        for n in range(1, _PAUSES + 1):
            policy.on_preempted(
                StreamCheckpoint(visible_text = "word " * 8, charged_tokens = 8, resumes = n)
            )
            assert policy.await_resume(timeout = 1.0) is True, f"gave up at pause {n}"

    def test_pauses_without_output_give_up(self):
        policy = _policy("stuck")
        limit = preemption.DEFAULT_MAX_STALLED_RESUMES
        for n in range(1, limit):
            policy.on_preempted(StreamCheckpoint(charged_tokens = 0, resumes = n))
            assert policy.await_resume(timeout = 1.0) is True
        policy.on_preempted(StreamCheckpoint(charged_tokens = 0, resumes = limit))
        assert policy.await_resume(timeout = 1.0) is False

    def test_progress_resets_the_stall_count(self):
        policy = _policy("mixed")
        limit = preemption.DEFAULT_MAX_STALLED_RESUMES
        for n in range(1, 4 * limit):
            charged = 5 if n % (limit - 1) == 0 else 0
            policy.on_preempted(StreamCheckpoint(charged_tokens = charged, resumes = n))
            assert policy.await_resume(timeout = 1.0) is True, f"gave up at pause {n}"


class TestAPlainChat:
    def test_sixty_four_pauses_with_progress_complete(self, monkeypatch):
        signal = PreemptSignal()
        policy = _RecordingPolicy()
        streams = [[_delta(f"w{n} "), _finish(), _done()] for n in range(_PAUSES)]
        streams.append([_delta("end."), _finish(), _done()])
        recorder = _Recorder(monkeypatch, streams, signal = signal, pause_attempts = range(_PAUSES))
        events = _run(recorder.backend, signal = signal, policy = policy)
        assert len(recorder.payloads) == _PAUSES + 1
        assert policy.events.count("awaited") == _PAUSES
        assert not any(
            isinstance(e, dict) and e.get("type") == "context_truncated" for e in events
        ), "gave up"
        finishes = [
            e.get("finish_reason")
            for e in events
            if isinstance(e, dict) and e.get("type") == "metadata"
        ]
        assert finishes[-1] == "stop", finishes

    def test_pauses_without_output_give_up(self, monkeypatch):
        signal = PreemptSignal()
        policy = _RecordingPolicy()
        limit = preemption.DEFAULT_MAX_STALLED_RESUMES
        streams = [[_role_only(), _finish(), _done()] for _ in range(limit + 4)]
        recorder = _Recorder(monkeypatch, streams, signal = signal, pause_attempts = range(limit + 4))
        events = _run(recorder.backend, signal = signal, policy = policy)
        assert all(c.charged_tokens == 0 for c in policy.checkpoints)
        assert len(recorder.payloads) == limit
        assert policy.events.count("awaited") == limit - 1
        finishes = [
            e.get("finish_reason")
            for e in events
            if isinstance(e, dict) and e.get("type") == "metadata"
        ]
        assert finishes[-1] == "length", finishes
