# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Pausing a reply and finishing it, in one response, through the real loop.

This is the whole mechanic end to end. A pause aborts only the upstream request;
the loop stays in ``generate_chat_completion_with_tools``, keeps its
``ToolLoopController`` and its conversation, and re-opens the request with the
partial reply as a trailing assistant turn plus ``continue_final_message``.

What makes that safe is that the loop never leaves the frame. The controller's
one-shot ledger lives in memory and is built once per response, so a design that
tore the response down and resumed it as a new request would re-run one-shot
tools. These tests pin the observable consequences: the request really is
re-opened, it really carries the continuation flag and the partial text, and the
turn is not charged as a tool iteration.
"""

from __future__ import annotations

import contextlib
import copy
import json
import threading

from core.inference import llama_preemption as preemption
from core.inference.llama_cpp import LlamaCppBackend


_TOOL = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "search",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
}


def _delta(content: str) -> str:
    return "data: " + json.dumps({"choices": [{"index": 0, "delta": {"content": content}}]}) + "\n"


def _finish(reason: str = "stop") -> str:
    return (
        "data: "
        + json.dumps({"choices": [{"index": 0, "delta": {}, "finish_reason": reason}]})
        + "\n"
    )


def _done() -> str:
    return "data: [DONE]\n"


class _Recorder:
    """A backend whose stream pauses itself partway through the first attempt."""

    def __init__(
        self,
        monkeypatch,
        streams,
        *,
        signal,
        pause_after_attempt = 0,
        pause_attempts = None,
    ):
        self.payloads: list[dict] = []
        self.signal = signal
        self.pause_attempts = (
            set(pause_attempts) if pause_attempts is not None else {pause_after_attempt}
        )
        self._streams = [list(stream) for stream in streams]
        self.backend = LlamaCppBackend.__new__(LlamaCppBackend)
        backend = self.backend
        backend._process = object()
        backend._healthy = True
        backend._port = 48847
        backend._api_key = None
        backend._effective_context_length = 4096
        backend._supports_reasoning = False
        backend._reasoning_always_on = False
        backend._reasoning_style = "enable_thinking"
        backend._supports_preserve_thinking = False

        recorder = self

        @contextlib.contextmanager
        def fake_stream_with_retry(
            _client,
            _url,
            payload,
            _cancel_event,
            headers = None,
            first_token_deadline = None,
            preempt_event = None,
        ):
            recorder.payloads.append(copy.deepcopy(payload))
            stream = recorder._streams.pop(0)
            yield type("FakeResponse", (), {"status_code": 200, "chunks": stream})()

        def fake_iter_text_cancellable(
            response,
            _cancel_event,
            first_token_deadline = None,
            preempt_event = None,
        ):
            attempt = len(recorder.payloads) - 1
            for chunk in response.chunks:
                yield chunk
                if attempt in recorder.pause_attempts and chunk.startswith("data: {"):
                    # Pressure noticed mid-stream, which is when it really is.
                    recorder.signal.request("kv_pressure")
                    raise preemption.LlamaStreamPreempted

        monkeypatch.setattr(backend, "_stream_with_retry", fake_stream_with_retry)
        monkeypatch.setattr(backend, "_iter_text_cancellable", fake_iter_text_cancellable)
        monkeypatch.setattr(backend, "_maybe_recover_from_mtp_crash", lambda *_a, **_k: False)


class _RecordingPolicy:
    """Stands in for the admission side. Records the handshake order."""

    def __init__(self, *, resume = True):
        self.events: list[str] = []
        self.checkpoints: list[preemption.StreamCheckpoint] = []
        self._resume = resume

    def should_preempt(self) -> bool:
        return False

    def on_preempted(self, checkpoint):
        self.events.append("preempted")
        self.checkpoints.append(checkpoint)

    def await_resume(self, timeout = None) -> bool:
        self.events.append("awaited")
        return self._resume

    def on_resumed(self) -> None:
        self.events.append("resumed")


def _run(backend, *, signal, policy, **kwargs):
    return list(
        backend.generate_chat_completion_with_tools(
            messages = [{"role": "user", "content": "write me a poem"}],
            tools = [_TOOL],
            cancel_event = threading.Event(),
            preempt_event = signal,
            preempt_policy = policy,
            **kwargs,
        )
    )


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
        """From the stream accumulator, not the thread's assistant row, which an
        aborted attempt never writes."""
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

    def test_the_signal_is_cleared_so_the_resume_can_run(self, monkeypatch):
        """Left set, the resumed attempt would abort on its first read and spin."""
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
        """Contention must not silently shorten an agent run.

        Enough pauses to outlast the loop's own reprompt slack, because one pause
        fits inside it and would prove nothing: a chat unlucky enough to be paused
        repeatedly is exactly the case that must still finish.
        """
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
    def test_a_policy_that_gives_up_stops_pausing(self, monkeypatch):
        """It must not wait forever, and it must not pause a second time.

        Giving up leaves the round loop, which hands the turn to the existing
        final-answer pass. That pass continues the partial rather than repeating
        it, so a chat whose pause could not be honoured still gets a whole reply
        instead of a sentence that stops mid-word. What must NOT happen is another
        pause, or a hang.
        """
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
        _run(recorder.backend, signal = signal, policy = policy)
        assert policy.events.count("preempted") == 1, "it paused more than once"
        assert not signal.is_set(), "the signal must be cleared before falling through"
        # The turn was handed on rather than abandoned mid-sentence.
        assert len(recorder.payloads) == 2

    def test_a_pause_before_the_first_token_re_issues_the_request_whole(self, monkeypatch):
        """`continue_final_message` refuses an empty assistant turn, so there is
        nothing to continue and the attempt is simply sent again."""
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
