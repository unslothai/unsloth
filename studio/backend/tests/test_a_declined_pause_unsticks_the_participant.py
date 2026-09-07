# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A pause the stream refuses has to be handed back, not merely ignored."""

from __future__ import annotations


from core.inference import llama_preemption as preemption
from core.inference.llama_preemption import (
    ControllerPreemptionPolicy,
    ParticipantState,
    PreemptionController,
)

from .preempt_fakes import (
    PreemptRecorder,
    # No ``on_declined``, which is the point: injected doubles are handed straight to the
    # loop, so the call has to survive one written against the protocol as it was.
    RecordingPolicy as _OldDouble,
    delta as _delta,
    done as _done,
    finish as _finish,
    run_tool_loop,
    tool_call as _tool_call,
    web_search_tool,
)

_TOOL = web_search_tool()


class _RaisingPolicy(_OldDouble):
    """Bookkeeping that fails must never take the conversation with it."""

    def on_declined(self) -> None:
        self.events.append("declined")
        raise RuntimeError("policy is broken")


def _Recorder(monkeypatch, streams, *, signal, pause_attempts):
    return PreemptRecorder(
        monkeypatch,
        streams,
        signal = signal,
        pause_attempts = pause_attempts,
        request_pressure = False,
        execute_tool = True,
    )


def _run(recorder, *, signal, policy):
    return run_tool_loop(
        recorder.backend,
        signal = signal,
        policy = policy,
        tools = [_TOOL],
        prompt = "what kernel is current?",
        max_tool_iterations = 1,
        permission_mode = "off",
    )


def _chosen_victim(controller, gen_id, signal):
    controller.register("other", tokens = 1000)
    participant = controller.register(gen_id, tokens = 1000, signal = signal)
    victims = controller.plan_preemptions(needed = 16384)
    assert [v.gen_id for v in victims] == [gen_id], [v.gen_id for v in victims]
    assert participant.state == ParticipantState.PREEMPTING
    assert signal.is_set()
    return participant


def _capped(monkeypatch):
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
    """The same branch at the end of the turn, where teardown still has to find it clean."""

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
