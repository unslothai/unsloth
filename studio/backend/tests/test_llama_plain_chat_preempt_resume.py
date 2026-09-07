# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Pausing and finishing an ordinary chat, with no tools anywhere."""

from __future__ import annotations

import contextlib
import copy
import threading

from core.inference import llama_preemption as preemption

from .preempt_fakes import (
    PreemptRecorder,
    RecordingPolicy as _RecordingPolicy,
    delta as _delta,
    done as _done,
    finish as _finish,
    run_plain as _run,
)


def _Recorder(monkeypatch, streams, *, signal, pause_attempts = (0,)):
    return PreemptRecorder(
        monkeypatch,
        streams,
        signal = signal,
        pause_attempts = pause_attempts,
        port = 48849,
    )


class TestAPlainChatPauses:
    def test_the_request_is_reopened(self, monkeypatch):
        signal = preemption.PreemptSignal()
        policy = _RecordingPolicy()
        recorder = _Recorder(
            monkeypatch,
            [
                [_delta("Once upon a time"), _finish(), _done()],
                [_delta(" there was a cat."), _finish(), _done()],
            ],
            signal = signal,
        )
        _run(recorder.backend, signal = signal, policy = policy)
        assert len(recorder.payloads) == 2, (
            "a paused chat with no tools must be re-opened, not abandoned. Abandoning is "
            "what this surface did for its whole life before it was armed."
        )

    def test_the_resumed_request_continues_the_partial(self, monkeypatch):
        signal = preemption.PreemptSignal()
        policy = _RecordingPolicy()
        recorder = _Recorder(
            monkeypatch,
            [
                [_delta("Once upon a time"), _finish(), _done()],
                [_delta(" there was a cat."), _finish(), _done()],
            ],
            signal = signal,
        )
        _run(recorder.backend, signal = signal, policy = policy)
        resumed = recorder.payloads[1]
        assert resumed.get("continue_final_message") is True
        assert resumed.get("add_generation_prompt") is False
        trailing = resumed["messages"][-1]
        assert trailing["role"] == "assistant"
        assert "Once upon a time" in trailing["content"]

    def test_the_handshake_runs_in_order_including_on_resumed(self, monkeypatch):
        signal = preemption.PreemptSignal()
        policy = _RecordingPolicy()
        recorder = _Recorder(
            monkeypatch,
            [
                [_delta("Once upon a time"), _finish(), _done()],
                [_delta(" done."), _finish(), _done()],
            ],
            signal = signal,
        )
        _run(recorder.backend, signal = signal, policy = policy)
        assert policy.events == ["preempted", "awaited", "resumed"]

    def test_the_visible_text_is_not_replayed(self, monkeypatch):
        signal = preemption.PreemptSignal()
        policy = _RecordingPolicy()
        recorder = _Recorder(
            monkeypatch,
            [
                [_delta("Once upon a time"), _finish(), _done()],
                [_delta(" there was a cat."), _finish(), _done()],
            ],
            signal = signal,
        )
        chunks = [
            c for c in _run(recorder.backend, signal = signal, policy = policy) if isinstance(c, str)
        ]
        prev = ""
        assembled = ""
        for cumulative in chunks:
            assembled += cumulative[len(prev) :]
            prev = cumulative
        assert assembled == "Once upon a time there was a cat."
        assert assembled.count("Once upon a time") == 1


class TestThePauseIsVisibleToTheClient:
    """The spin-wait the goal asks for, made visible."""

    @staticmethod
    def _events(chunks):
        return [c for c in chunks if isinstance(c, dict) and c.get("type") == "preempt"]

    def test_a_pause_and_its_resume_are_both_announced(self, monkeypatch):
        signal = preemption.PreemptSignal()
        policy = _RecordingPolicy()
        recorder = _Recorder(
            monkeypatch,
            [
                [_delta("Once upon a time"), _finish(), _done()],
                [_delta(" done."), _finish(), _done()],
            ],
            signal = signal,
        )
        events = self._events(_run(recorder.backend, signal = signal, policy = policy))
        assert [e["state"] for e in events] == ["paused", "resumed"]

    def test_a_pause_that_never_resumes_still_announces_itself(self, monkeypatch):
        signal = preemption.PreemptSignal()
        policy = _RecordingPolicy(resume = False)
        recorder = _Recorder(
            monkeypatch,
            [[_delta("Once upon a time"), _finish(), _done()]],
            signal = signal,
        )
        events = self._events(_run(recorder.backend, signal = signal, policy = policy))
        assert [e["state"] for e in events] == ["paused"]

    def test_the_pause_is_announced_after_the_lease_goes_back(self, monkeypatch):
        signal = preemption.PreemptSignal()
        order = []

        class _OrderingPolicy(_RecordingPolicy):
            def on_preempted(self, checkpoint):
                order.append("lease-returned")
                super().on_preempted(checkpoint)

        policy = _OrderingPolicy()
        recorder = _Recorder(
            monkeypatch,
            [
                [_delta("Once upon a time"), _finish(), _done()],
                [_delta(" done."), _finish(), _done()],
            ],
            signal = signal,
        )
        for chunk in recorder.backend.generate_chat_completion(
            messages = [{"role": "user", "content": "hi"}],
            cancel_event = threading.Event(),
            preempt_event = signal,
            preempt_policy = policy,
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
        chunks = _run(recorder.backend, signal = signal, policy = _RecordingPolicy())
        assert self._events(chunks) == []


class TestTheCapIsSpentDownAcrossResumes:
    def test_a_stated_max_tokens_shrinks_on_resume(self, monkeypatch):
        signal = preemption.PreemptSignal()
        policy = _RecordingPolicy()
        recorder = _Recorder(
            monkeypatch,
            [
                [_delta("x" * 400), _finish(), _done()],
                [_delta(" done."), _finish(), _done()],
            ],
            signal = signal,
        )
        _run(recorder.backend, signal = signal, policy = policy, max_tokens = 500)
        first = recorder.payloads[0].get("max_tokens")
        second = recorder.payloads[1].get("max_tokens")
        assert first == 500
        assert second is not None and second < first, (
            f"resumed attempt asked for {second} after already producing ~100 tokens "
            f"of a {first} cap"
        )
        assert second >= 1, "never zero: a request for no tokens returns an empty turn"

    def test_an_unstated_max_tokens_is_left_alone(self, monkeypatch):
        signal = preemption.PreemptSignal()
        policy = _RecordingPolicy()
        recorder = _Recorder(
            monkeypatch,
            [
                [_delta("Once upon a time"), _finish(), _done()],
                [_delta(" done."), _finish(), _done()],
            ],
            signal = signal,
        )
        _run(recorder.backend, signal = signal, policy = policy)
        assert recorder.payloads[0].get("max_tokens") == recorder.payloads[1].get("max_tokens")


class TestGivingUpIsNotAnError:
    def test_a_refused_resume_ends_the_turn_with_the_partial(self, monkeypatch):
        signal = preemption.PreemptSignal()
        policy = _RecordingPolicy(resume = False)
        recorder = _Recorder(
            monkeypatch,
            [[_delta("Once upon a time"), _finish(), _done()]],
            signal = signal,
        )
        chunks = _run(recorder.backend, signal = signal, policy = policy)
        assert len(recorder.payloads) == 1, "a refused resume must not re-open the request"
        assert any("Once upon a time" in c for c in chunks if isinstance(c, str))
        assert policy.events == ["preempted", "awaited"]


class TestThePauseCanLandBeforeTheStreamOpens:
    def test_a_pause_during_stream_setup_still_resumes(self, monkeypatch):
        signal = preemption.PreemptSignal()
        policy = _RecordingPolicy()
        recorder = _Recorder(
            monkeypatch,
            [[_delta("a full answer"), _finish(), _done()]],
            signal = signal,
            pause_attempts = (),
        )

        opened = {"n": 0}
        real_stream = recorder.backend._stream_with_retry

        @contextlib.contextmanager
        def pause_on_first_open(*args, **kwargs):
            opened["n"] += 1
            if opened["n"] == 1:
                signal.request("kv_pressure")
                raise preemption.LlamaStreamPreempted
            with real_stream(*args, **kwargs) as response:
                yield response

        monkeypatch.setattr(recorder.backend, "_stream_with_retry", pause_on_first_open)
        chunks = _run(recorder.backend, signal = signal, policy = policy)
        assert opened["n"] == 2, "a pause before the first token must still be resumed"
        assert policy.events == ["preempted", "awaited", "resumed"]
        assert any("a full answer" in c for c in chunks if isinstance(c, str))

    def test_the_checkpoint_of_an_empty_pause_does_not_continue(self, monkeypatch):
        signal = preemption.PreemptSignal()
        policy = _RecordingPolicy()
        recorder = _Recorder(
            monkeypatch,
            [[_delta("a full answer"), _finish(), _done()]],
            signal = signal,
            pause_attempts = (),
        )
        payloads = []
        real_stream = recorder.backend._stream_with_retry
        opened = {"n": 0}

        @contextlib.contextmanager
        def pause_on_first_open(_client, _url, payload, *args, **kwargs):
            opened["n"] += 1
            payloads.append(copy.deepcopy(payload))
            if opened["n"] == 1:
                signal.request("kv_pressure")
                raise preemption.LlamaStreamPreempted
            with real_stream(_client, _url, payload, *args, **kwargs) as response:
                yield response

        monkeypatch.setattr(recorder.backend, "_stream_with_retry", pause_on_first_open)
        _run(recorder.backend, signal = signal, policy = policy)
        assert payloads[1].get("continue_final_message") is not True
        assert payloads[1]["messages"][-1]["role"] == "user"


class TestNothingChangesForCallersThatDoNotPreempt:
    def test_no_policy_means_the_stream_is_untouched(self, monkeypatch):
        signal = preemption.PreemptSignal()
        recorder = _Recorder(
            monkeypatch,
            [[_delta("Once upon a time"), _finish(), _done()]],
            signal = signal,
            pause_attempts = (),
        )
        chunks = _run(recorder.backend, signal = signal, policy = None)
        assert len(recorder.payloads) == 1
        assert any("Once upon a time" in c for c in chunks if isinstance(c, str))
