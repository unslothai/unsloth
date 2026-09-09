# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The sweep counts output, a spent cap ends the turn, and the extras size the batch."""

import json

from core.inference import llama_preemption as preemption
from core.inference.llama_cpp import _TOKEN_REPORT_EVERY

from .preempt_fakes import (
    PreemptRecorder,
    RecordingPolicy,
    clean_admission_queues,
    clean_preemption_registry,
    delta,
    done,
    finish,
    run_plain,
)

# pytest finds these by name; named here so the import reads as a use.
_FIXTURES = (clean_admission_queues, clean_preemption_registry)


def _opener() -> str:
    """llama-server's first frame: the role, no content."""
    return "data: " + json.dumps({"choices": [{"index": 0, "delta": {"role": "assistant"}}]}) + "\n"


class TestOnlyOutputCountsTowardsTheSweep:
    """The role opener and the finish frame add no cell; counting them let a sweep fire on
    the finish frame and pick the request that had just finished as its victim."""

    def test_the_opener_and_the_finish_frame_do_not_count(self, monkeypatch):
        signal = preemption.PreemptSignal()
        reports: list[int] = []
        stream = [_opener()] + [delta("x")] * (_TOKEN_REPORT_EVERY - 1) + [finish(), done()]
        recorder = PreemptRecorder(monkeypatch, [stream], signal = signal)
        run_plain(
            recorder.backend, signal = signal, policy = RecordingPolicy(), on_tokens = reports.append
        )
        assert reports == [], f"a frame with no output counted: {reports}"

    def test_a_full_batch_of_output_still_reports(self, monkeypatch):
        signal = preemption.PreemptSignal()
        reports: list[int] = []
        stream = [_opener()] + [delta("x")] * _TOKEN_REPORT_EVERY + [finish(), done()]
        recorder = PreemptRecorder(monkeypatch, [stream], signal = signal)
        run_plain(
            recorder.backend, signal = signal, policy = RecordingPolicy(), on_tokens = reports.append
        )
        assert reports == [_TOKEN_REPORT_EVERY]


class TestASpentCapIsNotReopened:
    """A pause landing after the caller's cap was spent used to reopen upstream for the
    one-token floor, once per pause; the partial is the answer."""

    def test_the_partial_ends_the_turn_with_length(self, monkeypatch):
        signal = preemption.PreemptSignal()
        policy = RecordingPolicy()
        recorder = PreemptRecorder(
            monkeypatch,
            [[delta("x")] * 8 + [finish(), done()], [delta(" more"), finish(), done()]],
            signal = signal,
            pause_attempts = (0,),
            pause_after = 8,
        )
        events = run_plain(recorder.backend, signal = signal, policy = policy, max_tokens = 8)

        assert len(recorder.payloads) == 1, "the spent cap was reopened upstream"
        assert "awaited" not in policy.events, "a spent cap queued for room it cannot use"
        dicts = [e for e in events if isinstance(e, dict)]
        assert [e["finish_reason"] for e in dicts if e.get("type") == "metadata"][-1] == "length"
        # The caller's own cap ended it, which is not a give-up.
        assert not any(e.get("type") == "context_truncated" for e in dicts)


class TestThePassThroughBatchSizesTheReserve:
    def test_the_extras_win_over_the_typed_field(self):
        import routes.inference as inference

        class _Typed:
            requested_n_batch = 512
            extra_args = ["--batch-size", "8192"]

        class _LastWins:
            requested_n_batch = None
            extra_args = ["-b", "1024", "--batch-size=4096"]

        class _Empty:
            requested_n_batch = 512
            extra_args = []

        # The launcher's own flag comes before the extras, so the extras win at launch.
        assert inference._openai_llama_effective_batch_tokens(_Typed()) == 8192
        assert inference._openai_llama_effective_batch_tokens(_LastWins()) == 4096
        assert inference._openai_llama_effective_batch_tokens(_Empty()) == 512
