# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Pausing and finishing an ordinary chat, with no tools anywhere.

The twin of ``test_llama_tool_loop_preempt_resume``, and it exists because that one was
for a long time the ONLY coverage of preemption, which matched the code: preemption armed
on the tool-loop branch and nowhere else, so a chat that invoked no tools took an admission
lease and then decoded with no preemption at all. Measured on the plain streaming surface
at ``-c 16384`` with four chats and 3000-token prompts, the whole machinery reported
``armed 0 paused 0 gave-up 0``, and not even ``not-armed``, which is logged whenever arming
is attempted and declined.

The resume shape differs from the tool loop's and that difference is the thing to hold
still. The tool loop never leaves its frame, because its ``ToolLoopController`` ledger is
in-memory and tearing the response down would re-run one-shot tools. A plain chat has no
such ledger, so it resumes by calling the generator again with the partial appended, which
avoids re-indenting ~130 lines of a 400-line generator inside a ``while True:``. The
recursion is bounded by ``DEFAULT_MAX_PREEMPT_RESUMES``.
"""

from __future__ import annotations

import contextlib
import copy
import json
import threading

from core.inference import llama_preemption as preemption
from core.inference.llama_cpp import LlamaCppBackend


def _delta(content: str) -> str:
    return (
        "data: " + json.dumps({"choices": [{"index": 0, "delta": {"content": content}}]}) + "\n"
    )


def _finish(reason: str = "stop") -> str:
    return (
        "data: "
        + json.dumps({"choices": [{"index": 0, "delta": {}, "finish_reason": reason}]})
        + "\n"
    )


def _done() -> str:
    return "data: [DONE]\n"


class _Recorder:
    """A backend whose stream pauses itself partway through chosen attempts."""

    def __init__(self, monkeypatch, streams, *, signal, pause_attempts = (0,)):
        self.payloads: list[dict] = []
        self.signal = signal
        self.pause_attempts = set(pause_attempts)
        self._streams = [list(stream) for stream in streams]
        self.backend = LlamaCppBackend.__new__(LlamaCppBackend)
        backend = self.backend
        backend._process = object()
        backend._healthy = True
        backend._port = 48849
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
            yield type(
                "FakeResponse", (), {"status_code": 200, "chunks": recorder._streams.pop(0)}
            )()

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
                    recorder.signal.request("kv_pressure")
                    raise preemption.LlamaStreamPreempted

        monkeypatch.setattr(backend, "_stream_with_retry", fake_stream_with_retry)
        monkeypatch.setattr(backend, "_iter_text_cancellable", fake_iter_text_cancellable)
        monkeypatch.setattr(backend, "_maybe_recover_from_mtp_crash", lambda *_a, **_k: False)


class _RecordingPolicy:
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
        backend.generate_chat_completion(
            messages = [{"role": "user", "content": "write me a poem"}],
            cancel_event = threading.Event(),
            preempt_event = signal,
            preempt_policy = policy,
            **kwargs,
        )
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
        """``on_resumed`` was missing from the first version of this handler.

        It drives ``controller.note_resumed``, so without it a chat that is decoding
        again stays PAUSED in the ledger: its cells read as reclaimable, it stays a
        candidate for a pause it is no longer in, and the winner logic reasons about a
        state two moves old. Nothing else in the suite would have noticed.
        """
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
        """The client has already been streamed the partial, so the resumed attempt
        must not send it a second time."""
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
        chunks = [c for c in _run(recorder.backend, signal = signal, policy = policy)
                  if isinstance(c, str)]
        assert "".join(chunks).count("Once upon a time") == 1


class TestThePauseIsVisibleToTheClient:
    """The spin-wait the goal asks for, made visible.

    `admission-status.ts` has been able to read `preempt-paused` and `preempt-resumed`
    since it was written, and the backend had never sent either. A paused chat looked
    exactly like a wedged one: a half-written answer that stops dead and starts again
    minutes later with no explanation. The queue signals it DID send say "you have not
    started"; these say "you started, the text on screen is not lost".
    """

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
        """The case that matters most, because the turn ends there.

        A client told nothing has a truncated answer and no reason for it.
        """
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
        """Order, not just presence.

        Announcing before `on_preempted` would tell the client it is waiting for room
        that this very chat is still holding.
        """
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
        """``max_tokens`` bounds NEW tokens, and the resumed attempt starts a fresh count.

        Forwarding it unchanged let a chat preempted n times emit up to (n+1) times the
        cap it asked for, which is wrong for the client and wrong for admission, which
        charged once.
        """
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
        """``await_resume`` answering False means the room never came back.

        Ending there leaves the client the partial it has already been streamed, which
        the length-continuation path can pick up. Raising instead would turn a busy cache
        into an error, which is the behaviour this whole design replaces.
        """
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
        """The earliest possible pause, and it used to raise NameError.

        ``_open_stream`` raises ``LlamaStreamPreempted`` from the read installer, before
        the body of the ``with`` runs. ``reasoning_text`` was assigned inside that body,
        so the handler, which reads it, blew up and the client saw a 500 rather than a
        resume. Every other test here pauses mid-stream and cannot reach this.
        """
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
        """Nothing was produced, so there is nothing to continue FROM.

        ``continue_final_message`` refuses an empty assistant turn, so the attempt has to
        be re-issued whole rather than as a continuation.
        """
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
        """The default for every existing call site, and it must stay exactly as it was."""
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
