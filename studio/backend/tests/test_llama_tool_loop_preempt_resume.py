# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Pausing a reply and finishing it, in one response, through the real loop."""

from __future__ import annotations

import threading

from core.inference import llama_preemption as preemption

from .preempt_fakes import (
    PreemptRecorder,
    RecordingPolicy as _RecordingPolicy,
    delta as _delta,
    done as _done,
    finish as _finish,
    run_tool_loop,
    web_search_tool,
)

_TOOL = web_search_tool(required = True)


def _Recorder(
    monkeypatch,
    streams,
    *,
    signal,
    pause_after_attempt = 0,
    pause_attempts = None,
):
    return PreemptRecorder(
        monkeypatch,
        streams,
        signal = signal,
        pause_attempts = ({pause_after_attempt} if pause_attempts is None else set(pause_attempts)),
        port = 48847,
    )


def _run(backend, *, signal, policy, **kwargs):
    return run_tool_loop(backend, signal = signal, policy = policy, tools = [_TOOL], **kwargs)


class TestThePauseIsResumed:
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
        assert len(recorder.payloads) == 2, "a paused attempt must be re-opened, not abandoned"

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
        assert (
            "Once upon a time" in trailing["content"]
        ), "the partial must go back as the turn to EXTEND"

    def test_the_policy_handshake_runs_in_order(self, monkeypatch):
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

    def test_the_checkpoint_carries_the_streamed_text(self, monkeypatch):
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
        assert policy.checkpoints[0].visible_text == "Once upon a time"
        assert policy.checkpoints[0].has_resume_point()
        assert policy.checkpoints[0].resumes == 1

    def test_token_dense_text_is_charged_by_the_chunks_seen(self, monkeypatch):
        signal = preemption.PreemptSignal()
        policy = _RecordingPolicy()
        dense = ["\u6708"] * 12  # twelve one-character chunks, three tokens by the estimate
        recorder = _Recorder(
            monkeypatch,
            [
                [_delta(piece) for piece in dense] + [_finish(), _done()],
                [_delta(" done."), _finish(), _done()],
            ],
            signal = signal,
        )
        backend = recorder.backend

        def pause_after_the_dense_run(response, _cancel_event, first_token_deadline = None, preempt_event = None):
            attempt = len(recorder.payloads) - 1
            served = 0
            for chunk in response.chunks:
                yield chunk
                if attempt == 0 and chunk.startswith("data: {"):
                    served += 1
                    if served == len(dense):
                        recorder.signal.request("kv_pressure")
                        raise preemption.LlamaStreamPreempted

        monkeypatch.setattr(backend, "_iter_text_cancellable", pause_after_the_dense_run)
        _run(backend, signal = signal, policy = policy)
        assert policy.checkpoints[0].visible_text == "".join(dense)
        assert policy.checkpoints[0].charged_tokens >= len(dense), policy.checkpoints[0].charged_tokens

    def test_the_signal_is_cleared_so_the_resume_can_run(self, monkeypatch):
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
        assert not signal.is_set()
        assert not signal.pending


class TestWhatAPauseMustNotCost:
    def test_a_resume_is_not_charged_as_a_tool_iteration(self, monkeypatch):
        pauses = 6
        signal = preemption.PreemptSignal()
        policy = _RecordingPolicy()
        recorder = _Recorder(
            monkeypatch,
            [[_delta(f"part {n} "), _finish(), _done()] for n in range(pauses + 1)],
            signal = signal,
            pause_attempts = range(pauses),
        )
        _run(recorder.backend, signal = signal, policy = policy, max_tool_iterations = 1)
        assert (
            len(recorder.payloads) == pauses + 1
        ), "a paused turn was cut short by the tool-iteration bound"


class TestWhenItCannotOrMustNotResume:
    def test_a_policy_that_gives_up_ends_the_turn_and_says_so(self, monkeypatch):
        signal = preemption.PreemptSignal()
        policy = _RecordingPolicy(resume = False)
        recorder = _Recorder(
            monkeypatch,
            [
                [_delta("Once upon a time"), _finish(), _done()],
                [_delta(" there was a cat."), _finish(), _done()],
            ],
            signal = signal,
        )
        events = _run(recorder.backend, signal = signal, policy = policy)
        assert policy.events.count("preempted") == 1, "it paused more than once"
        assert not signal.is_set(), "the signal must be cleared before the turn ends"
        # No second request: the turn ended rather than decoding without a lease.
        assert len(recorder.payloads) == 1
        assert any(
            e.get("type") == "context_truncated" and e.get("reason") == "preempt_gave_up"
            for e in events
        ), "the client was not told why the answer stopped"
        assert any(
            e.get("type") == "metadata" and e.get("finish_reason") == "length" for e in events
        ), "no terminal metadata with a length finish"

    def test_a_pause_before_the_first_token_re_issues_the_request_whole(self, monkeypatch):
        signal = preemption.PreemptSignal()
        policy = _RecordingPolicy()

        recorder = _Recorder(
            monkeypatch,
            [
                # No content delta at all before the pause.
                [_finish(), _done()],
                [_delta("a full answer."), _finish(), _done()],
            ],
            signal = signal,
        )

        # Pause on the very first chunk, which carries no content.
        def fake_iter(
            response,
            _cancel_event,
            first_token_deadline = None,
            preempt_event = None,
        ):
            attempt = len(recorder.payloads) - 1
            if attempt == 0:
                raise preemption.LlamaStreamPreempted
            yield from response.chunks

        monkeypatch.setattr(recorder.backend, "_iter_text_cancellable", fake_iter)
        _run(recorder.backend, signal = signal, policy = policy)

        assert len(recorder.payloads) == 2
        assert not recorder.payloads[1].get(
            "continue_final_message"
        ), "there was no partial, so nothing should be continued"
        assert policy.checkpoints[0].has_resume_point() is False


class TestTheDefaultsAreUnchanged:
    def test_no_signal_means_no_pause_path_at_all(self, monkeypatch):
        signal = preemption.PreemptSignal()
        recorder = _Recorder(
            monkeypatch,
            [[_delta("plain answer"), _finish(), _done()]],
            signal = signal,
            pause_after_attempt = -1,
        )
        out = list(
            recorder.backend.generate_chat_completion_with_tools(
                messages = [{"role": "user", "content": "hi"}],
                tools = [_TOOL],
                cancel_event = threading.Event(),
            )
        )
        assert len(recorder.payloads) == 1
        assert any(event.get("type") == "content" for event in out)
