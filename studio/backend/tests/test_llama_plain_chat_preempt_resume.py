# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Pausing and finishing an ordinary chat, with no tools anywhere: the seam the client
sees, what the attempt is charged, what it reports, and how it gives up."""

from __future__ import annotations

import contextlib
import copy
import json
import threading

import pytest

from core.inference import llama_preemption as preemption
from core.inference.llama_cpp import (
    PREEMPT_GAVE_UP_REASON,
    LlamaCppBackend,
    _preempt_gave_up_event,
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


def _empty_delta() -> str:
    return "data: " + json.dumps({"choices": [{"index": 0, "delta": {}}]}) + "\n"


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
        assert later.startswith(earlier), (
            f"a snapshot went backwards across the pause: {earlier!r} then {later!r}"
        )


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

    def test_the_visible_text_is_not_replayed(self, monkeypatch):
        signal = preemption.PreemptSignal()
        events = _run(
            _two_part(monkeypatch, signal).backend, signal = signal, policy = _RecordingPolicy()
        )
        _monotonic(events)
        assert _assembled(events) == "Once upon a time there was a cat."


class TestThePauseIsVisibleToTheClient:
    def test_a_pause_and_its_resume_are_both_announced(self, monkeypatch):
        signal = preemption.PreemptSignal()
        events = _run(
            _two_part(monkeypatch, signal).backend, signal = signal, policy = _RecordingPolicy()
        )
        assert [e["state"] for e in _preempts(events)] == ["paused", "resumed"]

    def test_a_pause_that_never_resumes_still_announces_itself(self, monkeypatch):
        signal = preemption.PreemptSignal()
        recorder = _Recorder(
            monkeypatch, [[_delta("Once upon a time"), _finish(), _done()]], signal = signal
        )
        events = _run(recorder.backend, signal = signal, policy = _RecordingPolicy(resume = False))
        assert [e["state"] for e in _preempts(events)] == ["paused"]

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

    def test_a_resume_paused_before_its_first_token_still_continues_the_partial(self, monkeypatch):
        signal = preemption.PreemptSignal()
        recorder = _Recorder(
            monkeypatch,
            [
                [_delta("Introduction: The"), _finish(), _done()],
                [_empty_delta(), _delta("never reached"), _finish(), _done()],
                [_delta(" Paradigm"), _finish(), _done()],
            ],
            signal = signal,
            pause_attempts = (0, 1),
        )
        events = _run(recorder.backend, signal = signal, policy = _RecordingPolicy())
        third = recorder.payloads[2]
        assert third["messages"][-1] == {"role": "assistant", "content": "Introduction: The"}
        assert third.get("continue_final_message") is True, (
            "the partial went back as a finished turn: the model will answer again from "
            "the top and the client will see the answer twice"
        )
        _monotonic(events)
        assert _assembled(events) == "Introduction: The Paradigm"

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
        assert "<think>" not in (resumed.get("content") or ""), (
            "the open thought was replayed as visible content with a literal tag"
        )
        assert resumed.get("reasoning_content") == "Let me"
        assert recorder.payloads[1].get("continue_final_message") is True

    def test_a_reasoning_only_answer_keeps_both_attempts(self, monkeypatch):
        signal = preemption.PreemptSignal()
        recorder = PreemptRecorder(
            monkeypatch,
            [
                [_reasoning("The first half. "), _finish(), _done()],
                [_reasoning("The second half."), _finish(), _done()],
            ],
            signal = signal,
            pause_attempts = (0,),
            port = 48853,
            supports_reasoning = True,
            reasoning_always_on = True,
        )
        events = _run(
            recorder.backend,
            signal = signal,
            policy = _RecordingPolicy(),
            prompt = "answer me",
            promote_reasoning_only = True,
        )
        final = _texts(events)[-1]
        thought, _, fallback = final.partition("</think>")
        assert "The first half. " in thought and "The second half." in thought, final
        # For a reasoning-only model the promoted fallback IS the answer, and it was
        # built from the resumed attempt alone.
        assert "The first half. " in fallback and "The second half." in fallback, final


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

    def test_without_timings_the_chunks_decoded_are_the_estimate(self, monkeypatch):
        signal = preemption.PreemptSignal()
        recorder = _Recorder(
            monkeypatch,
            [
                [_delta("a"), _delta("b"), _delta("c"), _finish("length"), _done()],
                [_delta("d"), _usage(40, 5), _finish(), _done()],
            ],
            signal = signal,
            pause_after = 3,
        )
        assert self._usage(recorder, signal)["completion_tokens"] == 3 + 5

    def test_every_attempt_is_added(self, monkeypatch):
        signal = preemption.PreemptSignal()
        recorder = _Recorder(
            monkeypatch,
            [
                [_delta("one ", timings = {"predicted_n": 10}), _finish("length"), _done()],
                [_delta("two ", timings = {"predicted_n": 20}), _finish("length"), _done()],
                [_delta("three."), _usage(64, 3), _finish(), _done()],
            ],
            signal = signal,
            pause_attempts = (0, 1),
        )
        usage = self._usage(recorder, signal)
        assert len(recorder.payloads) == 3
        assert usage["completion_tokens"] == 33
        assert usage["total_tokens"] == 64 + 33

    def test_a_turn_that_was_never_paused_reports_what_the_server_said(self, monkeypatch):
        signal = preemption.PreemptSignal()
        recorder = _Recorder(
            monkeypatch,
            [[_delta("all in one go"), _usage(31, 9), _finish(), _done()]],
            signal = signal,
            pause_attempts = (),
        )
        assert self._usage(recorder, signal) == {
            "prompt_tokens": 31,
            "completion_tokens": 9,
            "total_tokens": 40,
        }

    @pytest.mark.parametrize(
        ("earlier", "expected"),
        [
            (0, {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7}),
            (11, {"prompt_tokens": 5, "completion_tokens": 13, "total_tokens": 18}),
        ],
    )
    def test_the_helper_leaves_an_unpaused_turn_alone(self, earlier, expected):
        from core.inference.llama_cpp import _usage_with_earlier_attempts

        usage = {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7}
        assert _usage_with_earlier_attempts(usage, earlier) == expected


class TestGivingUpIsNotAnError:
    def test_the_notice_says_nothing_was_evicted_and_everything_fitted(self):
        event = _preempt_gave_up_event(4096, 512)
        assert event["type"] == "context_truncated"
        assert event["reason"] == PREEMPT_GAVE_UP_REASON
        assert event["fits"] is True
        assert event["dropped_messages"] == 0
        assert event["context_length"] == 4096
        assert 0 < event["prompt_target"] < 4096
        assert "context_length" not in _preempt_gave_up_event(None, None)

    def test_the_turn_ends_with_the_partial_a_notice_and_a_length_finish(self, monkeypatch):
        signal = preemption.PreemptSignal()
        recorder = _Recorder(
            monkeypatch,
            [
                [_delta("Once "), _delta("upon "), _delta("a "), _delta("time"), _finish(), _done()],
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
        assert len(metadata) == 1 and metadata[0]["finish_reason"] == "length", (
            "an incomplete turn reported as anything else tells the client it is done"
        )
        assert metadata[0]["usage"].get("completion_tokens"), (
            "four deltas were streamed and shown; reporting zero completion tokens for "
            "them corrupts every usage-based client and monitor"
        )

    def test_a_refused_resume_before_the_first_token_is_not_an_empty_turn(self, monkeypatch):
        signal = preemption.PreemptSignal()
        recorder = _Recorder(
            monkeypatch, [[_empty_delta(), _finish(), _done()]], signal = signal
        )
        events = _run(recorder.backend, signal = signal, policy = _RecordingPolicy(resume = False))
        assert _assembled(events) == "", "this test is only about the empty case"
        notices = _gave_up(events)
        assert len(notices) == 1 and notices[0]["context_length"] == 4096

    def test_a_resume_that_is_granted_says_nothing_of_the_kind(self, monkeypatch):
        signal = preemption.PreemptSignal()
        events = _run(
            _two_part(monkeypatch, signal).backend, signal = signal, policy = _RecordingPolicy()
        )
        assert _gave_up(events) == []
        assert _metadata(events)[-1]["finish_reason"] != "length"


class TestThePauseCanLandBeforeTheStreamOpens:
    @staticmethod
    def _pause_on_first_open(monkeypatch, recorder, signal, payloads = None):
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

    def test_the_checkpoint_of_an_empty_pause_does_not_continue(self, monkeypatch):
        signal = preemption.PreemptSignal()
        recorder = _Recorder(
            monkeypatch,
            [[_delta("a full answer"), _finish(), _done()]],
            signal = signal,
            pause_attempts = (),
        )
        payloads: list[dict] = []
        self._pause_on_first_open(monkeypatch, recorder, signal, payloads)
        _run(recorder.backend, signal = signal, policy = _RecordingPolicy())
        assert payloads[1].get("continue_final_message") is not True
        assert payloads[1]["messages"][-1]["role"] == "user"

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

    def test_a_clear_signal_does_open_the_request(self):
        opened = []

        class _RecordingClient:
            def stream(self, *args, **kwargs):
                opened.append(kwargs.get("json"))
                raise RuntimeError("stop here; the POST is all this test needs to see")

        with pytest.raises(RuntimeError):
            with LlamaCppBackend._stream_with_retry(
                _RecordingClient(),
                "http://127.0.0.1:1/v1/chat/completions",
                {"messages": []},
                None,
                preempt_event = preemption.PreemptSignal(),
            ):
                pass
        assert opened == [{"messages": []}]


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
