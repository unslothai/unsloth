# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A pause the stream refuses has to be handed back, not merely ignored."""

from __future__ import annotations

import contextlib
import copy
import json
import threading

from core.inference import llama_preemption as preemption
from core.inference.llama_cpp import LlamaCppBackend
from core.inference.llama_preemption import (
    ControllerPreemptionPolicy,
    ParticipantState,
    PreemptionController,
)


_TOOL = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "search",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
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


def _tool_call(call_id: str = "call_search") -> list[str]:
    return [
        "data: "
        + json.dumps(
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": call_id,
                                    "type": "function",
                                    "function": {
                                        "name": "web_search",
                                        "arguments": json.dumps({"query": "kernel"}),
                                    },
                                }
                            ]
                        },
                    }
                ]
            }
        )
        + "\n",
        _done(),
    ]


class _OldDouble:
    """A policy written against the protocol as it was, with no ``on_declined``."""

    def __init__(self):
        self.events: list[str] = []

    def should_preempt(self) -> bool:
        return False

    def on_preempted(self, checkpoint) -> None:
        self.events.append("preempted")

    def await_resume(self, timeout = None) -> bool:
        self.events.append("awaited")
        return True

    def on_resumed(self) -> None:
        self.events.append("resumed")


class _RaisingPolicy(_OldDouble):
    """Bookkeeping that fails must never take the conversation with it."""

    def on_declined(self) -> None:
        self.events.append("declined")
        raise RuntimeError("policy is broken")


class _Recorder:
    """A backend whose chosen attempt is preempted partway through."""

    def __init__(self, monkeypatch, streams, *, signal, pause_attempts):
        self.payloads: list[dict] = []
        self.signal = signal
        self.pause_attempts = set(pause_attempts)
        self._streams = [list(stream) for stream in streams]
        backend = LlamaCppBackend.__new__(LlamaCppBackend)
        backend._process = object()
        backend._healthy = True
        backend._port = 48851
        backend._api_key = None
        backend._effective_context_length = 4096
        backend._supports_reasoning = False
        backend._reasoning_always_on = False
        backend._reasoning_style = "enable_thinking"
        backend._supports_preserve_thinking = False
        self.backend = backend

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
                    raise preemption.LlamaStreamPreempted

        monkeypatch.setattr(backend, "_stream_with_retry", fake_stream_with_retry)
        monkeypatch.setattr(backend, "_iter_text_cancellable", fake_iter_text_cancellable)
        monkeypatch.setattr(backend, "_maybe_recover_from_mtp_crash", lambda *_a, **_k: False)
        monkeypatch.setattr(
            "core.inference.tools.execute_tool",
            lambda name, arguments, **_kwargs: "Linux kernel 6.10.",
        )


def _run(recorder, *, signal, policy):
    return list(
        recorder.backend.generate_chat_completion_with_tools(
            messages = [{"role": "user", "content": "what kernel is current?"}],
            tools = [_TOOL],
            cancel_event = threading.Event(),
            preempt_event = signal,
            preempt_policy = policy,
            max_tool_iterations = 1,
            permission_mode = "off",
        )
    )


def _chosen_victim(controller, gen_id, signal):
    """Register two decoding chats and let a real sweep choose `gen_id`."""
    controller.register("other", tokens = 1000)
    participant = controller.register(gen_id, tokens = 1000, signal = signal)
    victims = controller.plan_preemptions(needed = 16384)
    assert [v.gen_id for v in victims] == [gen_id], [v.gen_id for v in victims]
    assert participant.state == ParticipantState.PREEMPTING
    assert signal.is_set()
    return participant


def _capped(monkeypatch):
    """Refuse the first pause, so the capped branch is the one under test."""
    monkeypatch.setattr(preemption, "DEFAULT_MAX_PREEMPT_RESUMES", 0)


class TestTheRoundLoopHandsTheDecisionBack:
    def test_the_participant_is_selectable_again(self, monkeypatch):
        _capped(monkeypatch)
        controller = PreemptionController("declined-round")
        controller.configure(budget = 16384, kv_unified = True)
        signal = preemption.PreemptSignal()
        participant = _chosen_victim(controller, "chat", signal)
        policy = ControllerPreemptionPolicy(controller, "chat", signal, loop = None)
        recorder = _Recorder(
            monkeypatch,
            [
                _tool_call(),
                [_delta("The current kernel is 6.10."), _finish(), _done()],
            ],
            signal = signal,
            pause_attempts = (0,),
        )
        events = _run(recorder, signal = signal, policy = policy)

        assert participant.state == ParticipantState.DECODING, (
            "the refusing chat decoded a whole final answer while PREEMPTING, which no "
            "later sweep can choose and no eviction can reclaim"
        )
        assert participant.preemptable, "a decoding chat has to be selectable again"
        assert not signal.is_set() and not signal.pending
        assert any(
            event.get("type") == "content" for event in events
        ), "the turn still has to produce its answer"

    def test_the_lease_and_the_cells_stay_where_they_are(self, monkeypatch):
        """Declining is not pausing: nothing was handed back, so nothing is released."""
        _capped(monkeypatch)
        controller = PreemptionController("declined-keeps-room")
        controller.configure(budget = 16384, kv_unified = True)
        signal = preemption.PreemptSignal()

        class _Lease:
            def __init__(self):
                self.preempted = 0

            def preempt(self):
                self.preempted += 1

        lease = _Lease()
        controller.register("other", tokens = 1000)
        participant = controller.register("chat", lease = lease, tokens = 1000, signal = signal)
        assert [v.gen_id for v in controller.plan_preemptions(needed = 16384)] == ["chat"]
        policy = ControllerPreemptionPolicy(controller, "chat", signal, loop = None)
        recorder = _Recorder(
            monkeypatch,
            [
                _tool_call(),
                [_delta("Answer."), _finish(), _done()],
            ],
            signal = signal,
            pause_attempts = (0,),
        )
        _run(recorder, signal = signal, policy = policy)

        assert lease.preempted == 0, "the tokens were handed back under a stream still using them"
        assert participant.holds_kv, "its cells never left the cache"
        assert participant.tokens == 1000

    def test_the_promotion_count_only_counts_pauses_taken(self, monkeypatch):
        _capped(monkeypatch)
        controller = PreemptionController("declined-promotion")
        controller.configure(budget = 16384, kv_unified = True)
        signal = preemption.PreemptSignal()
        participant = _chosen_victim(controller, "chat", signal)
        assert participant.consecutive_preemptions == 1
        policy = ControllerPreemptionPolicy(controller, "chat", signal, loop = None)
        recorder = _Recorder(
            monkeypatch,
            [_tool_call(), [_delta("Answer."), _finish(), _done()]],
            signal = signal,
            pause_attempts = (0,),
        )
        _run(recorder, signal = signal, policy = policy)
        assert participant.consecutive_preemptions == 0, (
            "a pause that never happened must not promote this chat above chats that "
            "really did lose their work"
        )

    def test_a_policy_without_the_method_still_finishes(self, monkeypatch):
        _capped(monkeypatch)
        policy = _OldDouble()
        signal = preemption.PreemptSignal()
        recorder = _Recorder(
            monkeypatch,
            [_tool_call(), [_delta("Answer."), _finish(), _done()]],
            signal = signal,
            pause_attempts = (0,),
        )
        events = _run(recorder, signal = signal, policy = policy)
        assert policy.events == [], "the capped branch must not run the pause handshake"
        assert any(event.get("type") == "content" for event in events)

    def test_a_policy_that_raises_does_not_end_the_turn(self, monkeypatch):
        _capped(monkeypatch)
        policy = _RaisingPolicy()
        signal = preemption.PreemptSignal()
        recorder = _Recorder(
            monkeypatch,
            [_tool_call(), [_delta("Answer."), _finish(), _done()]],
            signal = signal,
            pause_attempts = (0,),
        )
        events = _run(recorder, signal = signal, policy = policy)
        assert policy.events == ["declined"]
        assert any(event.get("type") == "content" for event in events)


class TestTheFinalPassHandsTheDecisionBack:
    """The same branch at the end of the turn."""

    def test_nothing_is_left_preempting(self, monkeypatch):
        _capped(monkeypatch)
        controller = PreemptionController("declined-final")
        controller.configure(budget = 16384, kv_unified = True)
        signal = preemption.PreemptSignal()
        participant = _chosen_victim(controller, "chat", signal)
        policy = ControllerPreemptionPolicy(controller, "chat", signal, loop = None)
        recorder = _Recorder(
            monkeypatch,
            [
                _tool_call(),
                [_delta("The current kernel"), _finish(), _done()],
            ],
            signal = signal,
            # The final answering pass, not the round.
            pause_attempts = (1,),
        )
        events = _run(recorder, signal = signal, policy = policy)

        assert (
            participant.state != ParticipantState.PREEMPTING
        ), "the turn ended with the ledger still holding a chosen victim"
        assert not signal.is_set()
        assert any(
            event.get("reason") == "preempt_gave_up" for event in events
        ), "the turn ended with no notice of why"
        metadata = [event for event in events if event.get("type") == "metadata"]
        assert metadata and metadata[-1]["finish_reason"] == "length"


class TestTheControllerItself:
    def test_only_a_chosen_victim_moves(self):
        controller = PreemptionController("declined-unit")
        controller.configure(budget = 16384, kv_unified = True)
        paused = controller.register("paused", tokens = 10, state = ParticipantState.PAUSED)
        controller.note_declined("paused")
        assert paused.state == ParticipantState.PAUSED, (
            "a chat that really did pause has moved on under its own transition; putting "
            "it back to DECODING would invent a holder"
        )
        # And an unknown id is simply nothing.
        controller.note_declined("never-registered")

    def test_the_signal_is_cleared_with_the_state(self):
        controller = PreemptionController("declined-signal")
        controller.configure(budget = 16384, kv_unified = True)
        signal = preemption.PreemptSignal()
        controller.register("other", tokens = 1000)
        participant = controller.register("chat", tokens = 1000, signal = signal)
        controller.plan_preemptions(needed = 16384)
        controller.note_declined("chat")
        assert participant.state == ParticipantState.DECODING
        assert (
            not signal.is_set()
        ), "a signal left set aborts the very stream this call is letting run"
        assert not signal.pending

    def test_a_declined_chat_can_be_chosen_again(self):
        controller = PreemptionController("declined-again")
        controller.configure(budget = 16384, kv_unified = True)
        signal = preemption.PreemptSignal()
        controller.register("other", tokens = 1000)
        controller.register("chat", tokens = 1000, signal = signal)
        controller.plan_preemptions(needed = 16384)
        controller.note_declined("chat")
        assert [v.gen_id for v in controller.plan_preemptions(needed = 16384)] == [
            "chat"
        ], "the point of the handback: pressure later in the turn can ask again"

    def test_the_null_policy_answers_it(self):
        preemption.NullPreemptionPolicy().on_declined()
        deferred = preemption.DeferredPreemptionPolicy()
        deferred.on_declined()

        class _Inner:
            def __init__(self):
                self.declined = 0

            def on_declined(self):
                self.declined += 1

        inner = _Inner()
        deferred.bind(inner)
        deferred.on_declined()
        assert inner.declined == 1
