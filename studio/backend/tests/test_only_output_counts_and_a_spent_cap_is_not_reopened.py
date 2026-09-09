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
    run_tool_loop,
    web_search_tool,
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

    def test_the_tool_round_counts_only_output_too(self, monkeypatch):
        def run(stream):
            signal = preemption.PreemptSignal()
            reports: list[int] = []
            recorder = PreemptRecorder(monkeypatch, [stream], signal = signal)
            run_tool_loop(
                recorder.backend,
                signal = signal,
                policy = RecordingPolicy(),
                tools = [web_search_tool()],
                on_tokens = reports.append,
            )
            return reports

        short = [_opener()] + [delta("x")] * (_TOKEN_REPORT_EVERY - 1) + [finish(), done()]
        assert run(short) == [], "the opener or the finish frame counted on the tool round"
        full = [_opener()] + [delta("x")] * _TOKEN_REPORT_EVERY + [finish(), done()]
        assert run(full) == [_TOKEN_REPORT_EVERY]

    def test_every_reader_shares_the_output_predicate(self):
        import inspect

        from core.inference.llama_cpp import LlamaCppBackend

        predicate = (
            'delta.get("content") or delta.get("reasoning_content") or delta.get("tool_calls")'
        )
        plain = " ".join(inspect.getsource(LlamaCppBackend.generate_chat_completion).split())
        tools = " ".join(
            inspect.getsource(LlamaCppBackend.generate_chat_completion_with_tools).split()
        )
        assert plain.count(predicate) == 1
        # The tool round and the final pass.
        assert tools.count(predicate) == 2


class TestARefusedResumeChargesTheAttemptOnce:
    """The interrupted attempt's decode goes into the accumulators, and the refused ending
    built its metadata from the same reading again: seven tokens reported as fourteen."""

    def test_the_tool_round_reports_the_attempt_once(self, monkeypatch):
        signal = preemption.PreemptSignal()
        stream = [
            delta("x", timings = {"prompt_n": 100, "predicted_n": n, "predicted_ms": 10 * n})
            for n in range(1, 8)
        ] + [finish(), done()]
        recorder = PreemptRecorder(
            monkeypatch, [stream], signal = signal, pause_attempts = (0,), pause_after = 7
        )
        items = run_tool_loop(
            recorder.backend,
            signal = signal,
            policy = RecordingPolicy(resume = False),
            tools = [web_search_tool()],
        )
        assert len(recorder.payloads) == 1
        metadata = [i for i in items if isinstance(i, dict) and i.get("type") == "metadata"][-1]
        assert metadata["finish_reason"] == "length"
        assert metadata["usage"]["completion_tokens"] == 7, metadata["usage"]
        assert metadata["usage"]["total_tokens"] == 107, metadata["usage"]
        assert metadata["timings"]["predicted_ms"] == 70, metadata["timings"]


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

    def test_llama_arg_batch_is_the_batch_the_child_runs_when_nothing_is_emitted(self):
        """llama-server reads the env whenever no flag is emitted, so an operator who
        exports 8192 and states nothing prefills in 8192 and must be reserved for it."""
        import routes.inference as inference

        class _Unstated:
            requested_n_batch = None
            _requested_n_batch = None
            extra_args = []

        class _Typed:
            requested_n_batch = 512
            extra_args = []

        class _Extras:
            requested_n_batch = None
            extra_args = ["--batch-size", "4096"]

        env = {"LLAMA_ARG_BATCH": "8192"}
        assert inference._openai_llama_effective_batch_tokens(_Unstated(), env = env) == 8192
        # Both are emitted as flags, and llama.cpp's flags beat its env.
        assert inference._openai_llama_effective_batch_tokens(_Typed(), env = env) == 512
        assert inference._openai_llama_effective_batch_tokens(_Extras(), env = env) == 4096
        assert inference._openai_llama_effective_batch_tokens(_Unstated(), env = {}) == 2048

    def test_an_unusable_llama_arg_batch_falls_back_to_the_default(self):
        import routes.inference as inference

        class _Unstated:
            requested_n_batch = None
            _requested_n_batch = None
            extra_args = []

        for raw in ("", "   ", "not-a-number", "0", "-1"):
            assert (
                inference._openai_llama_effective_batch_tokens(
                    _Unstated(), env = {"LLAMA_ARG_BATCH": raw}
                )
                == 2048
            ), raw

    def test_the_process_environment_is_the_default_source(self, monkeypatch):
        import routes.inference as inference

        class _Unstated:
            requested_n_batch = None
            _requested_n_batch = None
            extra_args = []

        monkeypatch.setenv("LLAMA_ARG_BATCH", "6144")
        assert inference._openai_llama_effective_batch_tokens(_Unstated()) == 6144
        monkeypatch.delenv("LLAMA_ARG_BATCH")
        assert inference._openai_llama_effective_batch_tokens(_Unstated()) == 2048
