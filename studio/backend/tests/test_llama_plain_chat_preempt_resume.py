# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Pausing and finishing an ordinary chat, with no tools anywhere: the seam the client
sees, what the attempt is charged, what it reports, and how it gives up."""

from __future__ import annotations

import contextlib
import copy
import threading

import pytest

from core.inference import llama_preemption as preemption
from core.inference.llama_cpp import (
    PREEMPT_GAVE_UP_REASON,
    LlamaCppBackend,
)

from .preempt_fakes import (
    PreemptRecorder,
    RecordingPolicy as _RecordingPolicy,
    delta as _delta,
    done as _done,
    finish as _finish,
    reasoning as _reasoning,
    run_plain as _run,
    usage as _usage,
)


def _Recorder(
    monkeypatch,
    streams,
    *,
    signal,
    pause_attempts = (0,),
    **kwargs,
):
    return PreemptRecorder(
        monkeypatch,
        streams,
        signal = signal,
        pause_attempts = pause_attempts,
        port = 48849,
        **kwargs,
    )


def _two_part(monkeypatch, signal, **kwargs):
    return _Recorder(
        monkeypatch,
        [
            [_delta("Once upon a time"), _finish(), _done()],
            [_delta(" there was a cat."), _finish(), _done()],
        ],
        signal = signal,
        **kwargs,
    )


def _texts(events) -> list[str]:
    return [event for event in events if isinstance(event, str)]


def _assembled(events) -> str:
    """The route reads cumulative snapshots and forwards the difference."""
    prev = ""
    out = ""
    for cumulative in _texts(events):
        out += cumulative[len(prev) :]
        prev = cumulative
    return out


def _monotonic(events) -> None:
    snapshots = _texts(events)
    for earlier, later in zip(snapshots, snapshots[1:]):
        assert later.startswith(
            earlier
        ), f"a snapshot went backwards across the pause: {earlier!r} then {later!r}"


def _preempts(events) -> list[dict]:
    return [e for e in events if isinstance(e, dict) and e.get("type") == "preempt"]


def _gave_up(events) -> list[dict]:
    return [
        e
        for e in events
        if isinstance(e, dict)
        and e.get("type") == "context_truncated"
        and e.get("reason") == PREEMPT_GAVE_UP_REASON
    ]


def _metadata(events) -> list[dict]:
    return [e for e in events if isinstance(e, dict) and e.get("type") == "metadata"]


class TestAPlainChatPauses:
    def test_a_paused_chat_is_reopened_with_the_partial_to_extend(self, monkeypatch):
        signal = preemption.PreemptSignal()
        recorder = _two_part(monkeypatch, signal)
        _run(recorder.backend, signal = signal, policy = _RecordingPolicy())
        assert len(recorder.payloads) == 2, (
            "a paused chat with no tools must be re-opened, not abandoned. Abandoning is "
            "what this surface did for its whole life before it was armed."
        )
        resumed = recorder.payloads[1]
        assert resumed.get("continue_final_message") is True
        assert resumed.get("add_generation_prompt") is False
        trailing = resumed["messages"][-1]
        assert trailing["role"] == "assistant"
        assert "Once upon a time" in trailing["content"]

    def test_the_handshake_runs_in_order_including_on_resumed(self, monkeypatch):
        signal = preemption.PreemptSignal()
        policy = _RecordingPolicy()
        _run(_two_part(monkeypatch, signal).backend, signal = signal, policy = policy)
        assert policy.events == ["preempted", "awaited", "resumed"]


class TestThePauseIsVisibleToTheClient:
    def test_a_pause_and_its_resume_are_both_announced(self, monkeypatch):
        signal = preemption.PreemptSignal()
        events = _run(
            _two_part(monkeypatch, signal).backend, signal = signal, policy = _RecordingPolicy()
        )
        assert [e["state"] for e in _preempts(events)] == ["paused", "resumed"]

    def test_the_pause_is_announced_after_the_lease_goes_back(self, monkeypatch):
        signal = preemption.PreemptSignal()
        order: list[str] = []

        class _OrderingPolicy(_RecordingPolicy):
            def on_preempted(self, checkpoint):
                order.append("lease-returned")
                super().on_preempted(checkpoint)

        recorder = _two_part(monkeypatch, signal)
        for chunk in recorder.backend.generate_chat_completion(
            messages = [{"role": "user", "content": "hi"}],
            cancel_event = threading.Event(),
            preempt_event = signal,
            preempt_policy = _OrderingPolicy(),
        ):
            if isinstance(chunk, dict) and chunk.get("type") == "preempt":
                order.append(f"announced-{chunk['state']}")
        assert order[:2] == ["lease-returned", "announced-paused"]

    def test_a_chat_that_never_pauses_announces_nothing(self, monkeypatch):
        signal = preemption.PreemptSignal()
        recorder = _Recorder(
            monkeypatch,
            [[_delta("Once upon a time"), _finish(), _done()]],
            signal = signal,
            pause_attempts = (),
        )
        events = _run(recorder.backend, signal = signal, policy = _RecordingPolicy())
        assert _preempts(events) == []


class TestTheSeamIsSeamless:
    def test_two_pauses_keep_both_prefixes(self, monkeypatch):
        signal = preemption.PreemptSignal()
        recorder = _Recorder(
            monkeypatch,
            [
                [_delta("one"), _finish(), _done()],
                [_delta(" two"), _finish(), _done()],
                [_delta(" three"), _delta(" four"), _finish(), _done()],
            ],
            signal = signal,
            pause_attempts = (0, 1),
        )
        events = _run(recorder.backend, signal = signal, policy = _RecordingPolicy())
        _monotonic(events)
        assert _assembled(events) == "one two three four"
        assert recorder.payloads[2]["messages"][-1]["content"] == "one two"

    def test_a_thought_interrupted_mid_way_stays_one_thought(self, monkeypatch):
        signal = preemption.PreemptSignal()
        recorder = _Recorder(
            monkeypatch,
            [
                [_reasoning("Let me"), _reasoning(" think"), _finish(), _done()],
                [_reasoning(" harder."), _delta("Answer."), _finish(), _done()],
            ],
            signal = signal,
        )
        recorder.backend._supports_reasoning = True
        events = _run(recorder.backend, signal = signal, policy = _RecordingPolicy())
        _monotonic(events)
        assert _assembled(events) == "<think>Let me harder.</think>Answer."

    def test_a_pause_mid_thought_resumes_the_thought_not_the_answer(self, monkeypatch):
        signal = preemption.PreemptSignal()
        recorder = _Recorder(
            monkeypatch,
            [
                [_reasoning("Let me"), _finish(), _done()],
                [_reasoning(" think."), _delta("Answer."), _finish(), _done()],
            ],
            signal = signal,
        )
        recorder.backend._supports_reasoning = True
        _run(recorder.backend, signal = signal, policy = _RecordingPolicy())
        resumed = recorder.payloads[1]["messages"][-1]
        assert resumed["role"] == "assistant"
        assert "<think>" not in (
            resumed.get("content") or ""
        ), "the open thought was replayed as visible content with a literal tag"
        assert resumed.get("reasoning_content") == "Let me"
        assert recorder.payloads[1].get("continue_final_message") is True


class TestTheCapIsSpentDownAcrossResumes:
    def test_a_stated_max_tokens_shrinks_on_resume(self, monkeypatch):
        signal = preemption.PreemptSignal()
        recorder = _Recorder(
            monkeypatch,
            [
                [_delta("x" * 400), _finish(), _done()],
                [_delta(" done."), _finish(), _done()],
            ],
            signal = signal,
        )
        _run(recorder.backend, signal = signal, policy = _RecordingPolicy(), max_tokens = 500)
        first, second = (p.get("max_tokens") for p in recorder.payloads)
        assert first == 500
        assert second is not None and second < first, (
            f"resumed attempt asked for {second} after already producing ~100 tokens "
            f"of a {first} cap"
        )
        assert second >= 1, "never zero: a request for no tokens returns an empty turn"

    def test_the_admission_allowance_bounds_the_wire_cap_on_every_attempt(self, monkeypatch):
        signal = preemption.PreemptSignal()
        recorder = _two_part(monkeypatch, signal)
        _run(
            recorder.backend,
            signal = signal,
            policy = _RecordingPolicy(),
            max_tokens = 3000,
            admission_output_allowance = 512,
        )
        assert [p.get("max_tokens") for p in recorder.payloads] == [512, 512], (
            "the room admission reserved has to land on the request, and on the resumed "
            "one too, or a pause hands the turn a fresh allowance it was never granted"
        )

    def test_an_unstated_max_tokens_is_left_alone(self, monkeypatch):
        signal = preemption.PreemptSignal()
        recorder = _two_part(monkeypatch, signal)
        _run(recorder.backend, signal = signal, policy = _RecordingPolicy())
        assert recorder.payloads[0].get("max_tokens") == recorder.payloads[1].get("max_tokens")


class TestThePauseIsChargedWhatTheAttemptDecoded:
    _DENSE = "天地玄黄宇宙洪荒"  # eight CJK tokens; chars // 4 calls this two

    def test_token_dense_text_is_not_charged_by_its_character_count(self, monkeypatch):
        signal = preemption.PreemptSignal()
        policy = _RecordingPolicy()
        recorder = _Recorder(
            monkeypatch,
            [
                [*[_delta(ch) for ch in self._DENSE], _finish(), _done()],
                [_delta("。"), _finish(), _done()],
            ],
            signal = signal,
            pause_after = len(self._DENSE),
        )
        _run(recorder.backend, signal = signal, policy = policy, prompt = "写一首诗")
        charged = policy.checkpoints[0].charged_tokens
        assert charged >= len(self._DENSE), (
            f"{len(self._DENSE)} tokens were streamed and {charged} were charged; the "
            "difference is cells the watermark cannot see and output cap the caller "
            "never agreed to"
        )


class TestTheUsageCoversEveryAttempt:
    @staticmethod
    def _usage(recorder, signal):
        events = _run(recorder.backend, signal = signal, policy = _RecordingPolicy())
        metadata = _metadata(events)
        assert len(metadata) == 1, "a resumed turn reports its usage once, at the end"
        return metadata[0]["usage"]

    def test_the_paused_attempts_tokens_are_counted_and_the_prompt_is_the_last(self, monkeypatch):
        signal = preemption.PreemptSignal()
        recorder = _Recorder(
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
            pause_after = 2,
        )
        usage = self._usage(recorder, signal)
        assert usage["completion_tokens"] == 19, (
            "12 decoded before the pause plus 7 after it. Reporting 7 alone is what a "
            "chat that streamed thousands of characters used to claim."
        )
        # The resumed prompt carries the partial, so summing prompts would report the
        # same conversation twice.
        assert usage["prompt_tokens"] == 40
        assert usage["total_tokens"] == 40 + 19

    def test_the_turn_ends_with_the_partial_a_notice_and_a_length_finish(self, monkeypatch):
        signal = preemption.PreemptSignal()
        recorder = _Recorder(
            monkeypatch,
            [
                [
                    _delta("Once "),
                    _delta("upon "),
                    _delta("a "),
                    _delta("time"),
                    _finish(),
                    _done(),
                ],
            ],
            signal = signal,
            pause_after = 4,
        )
        events = _run(recorder.backend, signal = signal, policy = _RecordingPolicy(resume = False))

        assert len(recorder.payloads) == 1, "a refused resume must not re-open the request"
        assert _assembled(events) == "Once upon a time"
        assert len(_gave_up(events)) == 1
        kinds = [e.get("type") for e in events if isinstance(e, dict)]
        assert kinds.index("preempt") < kinds.index("context_truncated") < kinds.index("metadata")
        metadata = _metadata(events)
        assert (
            len(metadata) == 1 and metadata[0]["finish_reason"] == "length"
        ), "an incomplete turn reported as anything else tells the client it is done"
        assert metadata[0]["usage"].get("completion_tokens"), (
            "four deltas were streamed and shown; reporting zero completion tokens for "
            "them corrupts every usage-based client and monitor"
        )


class TestThePauseCanLandBeforeTheStreamOpens:
    @staticmethod
    def _pause_on_first_open(
        monkeypatch,
        recorder,
        signal,
        payloads = None,
    ):
        real_stream = recorder.backend._stream_with_retry
        opened = {"n": 0}

        @contextlib.contextmanager
        def _open(_client, _url, payload, *args, **kwargs):
            opened["n"] += 1
            if payloads is not None:
                payloads.append(copy.deepcopy(payload))
            if opened["n"] == 1:
                signal.request("kv_pressure")
                raise preemption.LlamaStreamPreempted
            with real_stream(_client, _url, payload, *args, **kwargs) as response:
                yield response

        monkeypatch.setattr(recorder.backend, "_stream_with_retry", _open)
        return opened

    def test_a_pause_during_stream_setup_still_resumes(self, monkeypatch):
        signal = preemption.PreemptSignal()
        policy = _RecordingPolicy()
        recorder = _Recorder(
            monkeypatch,
            [[_delta("a full answer"), _finish(), _done()]],
            signal = signal,
            pause_attempts = (),
        )
        opened = self._pause_on_first_open(monkeypatch, recorder, signal)
        events = _run(recorder.backend, signal = signal, policy = policy)
        assert opened["n"] == 2, "a pause before the first token must still be resumed"
        assert policy.events == ["preempted", "awaited", "resumed"]
        assert "a full answer" in _assembled(events)

    def test_an_armed_signal_stops_the_request_before_it_is_sent(self):
        class _ExplodingClient:
            def stream(self, *args, **kwargs):
                raise AssertionError(
                    "llama-server was asked to prefill a prompt for a chat that had "
                    "already been told to stop"
                )

        signal = preemption.PreemptSignal()
        signal.request("kv_pressure")
        with pytest.raises(preemption.LlamaStreamPreempted):
            with LlamaCppBackend._stream_with_retry(
                _ExplodingClient(),
                "http://127.0.0.1:1/v1/chat/completions",
                {"messages": []},
                None,
                preempt_event = signal,
            ):
                raise AssertionError("the stream opened despite a pending preemption")


class TestNothingChangesForCallersThatDoNotPreempt:
    def test_no_policy_means_the_stream_is_untouched(self, monkeypatch):
        signal = preemption.PreemptSignal()
        recorder = _Recorder(
            monkeypatch,
            [[_delta("Once upon a time"), _finish(), _done()]],
            signal = signal,
            pause_attempts = (),
        )
        events = _run(recorder.backend, signal = signal, policy = None)
        assert len(recorder.payloads) == 1
        assert _assembled(events) == "Once upon a time"
