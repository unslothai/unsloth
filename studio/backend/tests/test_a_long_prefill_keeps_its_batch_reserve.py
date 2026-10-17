# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A prefill that is still advancing keeps its batch reserve past the announcement TTL."""

from __future__ import annotations

import json

from core.inference import llama_preemption as preemption
from core.inference.llama_cpp import LlamaCppBackend
from core.inference.llama_preemption import PreemptSignal, PreemptionController
import pytest

pytestmark = pytest.mark.usefixtures("preemption_opted_in")


class _Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def _controller() -> PreemptionController:
    made = PreemptionController("http://127.0.0.1:65101")
    made.configure(budget = 16384, kv_unified = True, slots = 4)
    return made


class TestTheReserveFollowsProgress:
    def _announced(self, monkeypatch):
        clock = _Clock()
        monkeypatch.setattr(preemption.time, "monotonic", clock)
        controller = _controller()
        signal = PreemptSignal()
        controller.register("long", tokens = 12000, signal = signal)
        controller.participant("long").announce_prefill(12000)
        return clock, controller, signal

    def test_progress_keeps_it_past_the_ttl(self, monkeypatch):
        clock, controller, signal = self._announced(monkeypatch)
        clock.now += preemption.PENDING_PREFILL_TTL_S - 10
        signal.note_prefill_progress()
        clock.now += 20
        assert (
            controller.snapshot().prefilling == 12000
        ), "the prompt is still going in, so the next batch still needs its room"

    def test_without_progress_it_still_expires(self, monkeypatch):
        clock, controller, _signal = self._announced(monkeypatch)
        clock.now += preemption.PENDING_PREFILL_TTL_S + 10
        assert controller.snapshot().prefilling == 0


class _Stream:
    def __init__(self, chunks):
        self._chunks = chunks

    def iter_text(self):
        yield from self._chunks

    def close(self):
        pass


def _progress(processed):
    payload = {"prompt_progress": {"total": 4000, "processed": processed, "cache": 0, "time_ms": 1}}
    return "data: " + json.dumps(payload) + "\n\n"


def test_the_stream_reports_its_progress_to_the_signal(monkeypatch):
    monkeypatch.setattr(
        LlamaCppBackend, "_set_stream_read_timeout", staticmethod(lambda *a, **k: None)
    )
    signal = PreemptSignal()
    assert signal.prefill_progress_at == 0.0
    chunks = [
        _progress(1024),
        _progress(2048),
        'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n',
    ]
    list(LlamaCppBackend._iter_text_cancellable(_Stream(chunks), None, preempt_event = signal))
    assert signal.prefill_progress_at > 0.0
