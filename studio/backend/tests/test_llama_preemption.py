# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Pausing a chat instead of killing four: the signal, the checkpoint, the ledger, the
planner, the buffer arithmetic, the residency reading and the waits either side of a pause."""

from __future__ import annotations

import asyncio
import json
import threading
import time
import urllib.request

import pytest

from core.inference import llama_admission
from core.inference import llama_preemption as preemption
from core.inference.llama_admission import LlamaAdmissionConfig, LlamaAdmissionQueue
from core.inference.llama_cpp import LlamaCppBackend, _CombinedCancelEvent, _interrupt_event
from core.inference.llama_preemption import (
    DEFAULT_PREEMPT_BUFFER_MIN_TOKENS,
    DEFAULT_RESUME_WAIT_TIMEOUT_S,
    MAX_RESUME_WAIT_MULTIPLE,
    PROMOTE_AFTER_CONSECUTIVE_PREEMPTIONS,
    ControllerPreemptionPolicy,
    DeferredPreemptionPolicy,
    ParticipantState,
    PreemptSignal,
    PreemptionController,
    StreamCheckpoint,
    get_preemption_controller,
    preemption_buffer_tokens,
    read_slot_occupancy,
    reclaim_idle_slots,
    wait_for_reclaim,
)
from core.inference.llama_stats import erase_llama_slot, fetch_llama_slots

from .preempt_fakes import clean_admission_queues, clean_preemption_registry  # noqa: F401


def _controller(
    budget = 16384,
    kv_unified = True,
    key = "test",
    **kw,
):
    controller = PreemptionController(key)
    controller.configure(budget = budget, kv_unified = kv_unified, **kw)
    return controller


def _ceiling(controller):
    snapshot = controller.snapshot()
    return snapshot.budget - snapshot.buffer


def _register(
    controller,
    gen_id,
    tokens,
    state = ParticipantState.DECODING,
    **kw,
):
    return controller.register(gen_id, tokens = tokens, state = state, **kw)


def _fill(
    controller,
    gen_id,
    fraction,
    state = ParticipantState.DECODING,
):
    return _register(controller, gen_id, int(_ceiling(controller) * fraction), state = state)


async def _lease(
    queue,
    *,
    tokens,
    capacity = 4,
    budget = 16384,
):
    reservation = queue.reserve(
        capacity = capacity, config = LlamaAdmissionConfig(), tokens = tokens, budget = budget
    )
    lease = reservation.lease_nowait()
    assert lease is not None, "expected an immediate admission"
    return lease


# =============================================================================== the signal


class TestTheSignal:
    def test_a_request_is_visible_carries_its_reason_and_can_be_forgotten(self):
        signal = PreemptSignal()
        assert not signal.is_set()
        signal.request("kv_pressure")
        assert signal.is_set() and signal.pending and signal.reason == "kv_pressure"
        signal.clear()
        assert not signal.is_set() and not signal.pending and signal.reason is None

    def test_a_request_in_the_unsafe_window_is_hidden_but_never_dropped(self):
        """The property tool execution depends on."""
        signal = PreemptSignal()
        with signal.unsafe_window():
            signal.request()
            assert not signal.is_set(), "a pause must not land during tool execution"
            assert signal.pending, "but it must not be forgotten either"
            with signal.unsafe_window():
                pass
            assert not signal.is_set(), "the outer window is still open"
        assert signal.is_set()

        made_before = PreemptSignal()
        made_before.request()
        with made_before.unsafe_window():
            assert not made_before.is_set()
        assert made_before.is_set()

    def test_a_cancel_or_a_pause_interrupts_the_one_event_the_stream_takes(self):
        for trip in (lambda c, p: p.request(), lambda c, p: c.set()):
            cancel, pause = threading.Event(), PreemptSignal()
            combined = _interrupt_event(cancel, pause)
            assert isinstance(combined, _CombinedCancelEvent) and not combined.is_set()
            trip(cancel, pause)
            assert combined.is_set()
        cancel = threading.Event()
        assert _interrupt_event(cancel, None) is cancel
        assert _interrupt_event(None, None) is None

    @pytest.mark.parametrize(
        ("value", "enabled"),
        [(None, True), ("0", False), ("false", False), ("OFF", False), ("maybe", True)],
    )
    def test_the_rollout_switch_is_on_unless_the_environment_plainly_says_otherwise(
        self, monkeypatch, value, enabled
    ):
        if value is None:
            monkeypatch.delenv(preemption.PREEMPT_ENV, raising = False)
        else:
            monkeypatch.setenv(preemption.PREEMPT_ENV, value)
        assert preemption.preemption_enabled() is enabled

    def test_the_null_policy_never_pauses_and_the_deferred_one_is_inert_until_bound(self):
        """The generator is BUILT before admission returns and ITERATED after."""
        null = preemption.NullPreemptionPolicy()
        assert null.should_preempt() is False and null.await_resume() is True
        assert isinstance(null, preemption.PreemptionPolicy)

        policy = DeferredPreemptionPolicy()
        assert policy.bound is False and policy.should_preempt() is False
        assert policy.await_resume(timeout = 0.01) is False
        policy.on_resumed()
        policy.on_declined()

        controller = _controller(key = "wiring-bind")
        signal = PreemptSignal()
        controller.register("g", tokens = 10)
        policy.bind(ControllerPreemptionPolicy(controller, "g", signal))
        assert policy.bound is True and policy.should_preempt() is False
        signal.set()
        assert policy.should_preempt() is True


# =========================================================================== the checkpoint


class TestTheCheckpoint:
    """The measured livelock: ten pauses, `kept_chars=0` on every one."""

    def test_prose_is_a_resume_point_and_a_thought_is_a_reasoning_one(self):
        thought = StreamCheckpoint(visible_text = "", reasoning_text = "Let me work through")
        assert not thought.has_resume_point(), "there is no prose to extend"
        assert thought.has_reasoning_resume_point()
        assert thought.kept_chars() == len("Let me work through")

        both = StreamCheckpoint(visible_text = "The answer is", reasoning_text = "thinking")
        assert both.has_resume_point()
        assert not both.has_reasoning_resume_point(), "prose already shown must not go back"
        assert both.kept_chars() == len("The answer is")

    def test_a_reasoning_only_turn_is_resumable_and_a_tool_call_turn_is_not(self):
        from core.inference.chat_template_helpers import (
            trailing_assistant_reasoning,
            trailing_assistant_resumable,
            trailing_assistant_text,
        )

        convo = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "", "reasoning_content": "half a thought"},
        ]
        # The exact trap: "" is falsy, so every `and trailing_assistant_text(...)` gate
        # dropped the continuation flag for a turn that had real work to continue.
        assert trailing_assistant_text(convo) == ""
        assert trailing_assistant_reasoning(convo) == "half a thought"
        assert trailing_assistant_resumable(convo)
        convo[-1]["tool_calls"] = [{"id": "1", "function": {"name": "x", "arguments": "{}"}}]
        assert not trailing_assistant_resumable(convo)

    def test_assembling_carries_a_thought_as_reasoning_and_merges_a_second_pause(self):
        convo = [{"role": "user", "content": "hi"}]
        assert (
            LlamaCppBackend._assemble_preempt_resume(object(), convo, StreamCheckpoint(), "", "")
            is False
        )
        assert len(convo) == 1, "an empty assistant turn would be refused downstream"

        assert (
            LlamaCppBackend._assemble_preempt_resume(
                object(), convo, StreamCheckpoint(reasoning_text = "first half "), "", "first half "
            )
            is True
        )
        assert convo[-1]["reasoning_content"] == "first half "
        assert convo[-1]["content"] == "", "a thought in content is rendered as the answer"

        LlamaCppBackend._assemble_preempt_resume(
            object(), convo, StreamCheckpoint(reasoning_text = "second half"), "", "second half"
        )
        assert convo[-1]["reasoning_content"] == "first half second half"
        assert len(convo) == 2


# =============================================================================== the stream


class _FakeResponse:
    """Enough of httpx.Response for _iter_text_cancellable."""

    def __init__(self, chunks):
        self._chunks = list(chunks)
        self.closed = False
        self.request = None

    def iter_text(self):
        yield from self._chunks

    def close(self):
        self.closed = True


def _drain(
    response,
    *,
    cancel_event = None,
    preempt_event = None,
):
    return list(
        LlamaCppBackend._iter_text_cancellable(
            response,
            cancel_event,
            first_token_deadline = time.monotonic() + 30.0,
            preempt_event = preempt_event,
        )
    )


class TestWhichExceptionComesOutOfTheStream:
    """A pause has to be tellable from a Stop, at the exact place the stream dies."""

    def test_a_pause_raises_preempted_and_closes_the_response(self):
        pause = PreemptSignal()
        pause.request("kv_pressure")
        response = _FakeResponse(["data: a\n", "data: b\n"])
        with pytest.raises(preemption.LlamaStreamPreempted):
            _drain(response, cancel_event = threading.Event(), preempt_event = pause)
        assert response.closed

    def test_a_pause_raised_mid_stream_stops_the_rest(self):
        pause = PreemptSignal()
        seen = []

        class _Streaming(_FakeResponse):
            def iter_text(self):
                for index, chunk in enumerate(self._chunks):
                    seen.append(chunk)
                    if index == 1:
                        pause.request("kv_pressure")
                    yield chunk

        with pytest.raises(preemption.LlamaStreamPreempted):
            _drain(
                _Streaming(["a", "b", "c", "d"]),
                cancel_event = threading.Event(),
                preempt_event = pause,
            )
        assert "d" not in seen, "the stream kept reading after the pause"


# ==================================================================== the buffer arithmetic


class TestTheBufferArithmetic:
    def test_the_buffer_is_per_slot_with_a_floor_and_a_cap(self):
        # Per SLOT, not per cache: the reaction headroom scales with how many chats decode
        # at once and not with the size of the cache.
        four = preemption_buffer_tokens(16384, slots = 4)
        assert preemption_buffer_tokens(16384, slots = 8) > four, "twice the slots, twice as much"
        assert four == preemption_buffer_tokens(65536, slots = 4), "a bigger cache is not slower"
        assert preemption_buffer_tokens(2048) >= DEFAULT_PREEMPT_BUFFER_MIN_TOKENS
        # And a small share of a normal cache, or it serialises: at 15% of 16384 the
        # simulated makespan was 26890 steps against 239 at this size.
        assert four < 16384 * 0.08
        assert preemption_buffer_tokens(0) == 0

    def test_the_ceiling_is_the_budget_minus_the_buffer_and_room_asked_for_counts(self):
        controller = _controller()
        _register(controller, "a", 16384 - preemption_buffer_tokens(16384))
        assert controller.plan_preemptions() == [], "exactly at the ceiling still fits"
        _register(controller, "b", 1)
        assert controller.plan_preemptions(), "one token past it must not"

        advance = _controller(key = "advance")
        # Sized from the ceiling rather than the figure it happened to have, so raising the
        # margin does not break the property being tested.
        _register(advance, "winner", _ceiling(advance) - 2000)
        _register(advance, "other", 1500)
        assert advance.plan_preemptions() == [], "it fits under the ceiling"
        assert advance.plan_preemptions(needed = 1000), "a request for room must be counted"


class TestTheBatchReserveIsOnlyHeldWhileSomethingPrefills:
    """A quarter of the cache was held back for a prefill that was not happening."""

    BUDGET, SLOTS, N_BATCH, DRAFTS = 8192, 4, 2048, 2
    IDLE = 192 * SLOTS + DRAFTS * SLOTS  # reaction headroom plus drafts
    PREFILL = N_BATCH + DRAFTS * SLOTS  # while a chunk is in flight, and once permanent

    def _c(self, key = "test://buffer"):
        return _controller(
            budget = self.BUDGET,
            key = key,
            slots = self.SLOTS,
            draft_tokens = self.DRAFTS,
            batch_tokens = self.N_BATCH,
        )

    def _buffer(self, **kw):
        return preemption_buffer_tokens(
            self.BUDGET,
            slots = self.SLOTS,
            draft_tokens = self.DRAFTS,
            batch_tokens = self.N_BATCH,
            **kw,
        )

    def test_the_reserve_is_the_pending_prompt_bounded_by_one_chunk_or_nothing_at_all(self):
        assert self._buffer() == self.IDLE == 776
        assert self.PREFILL - self.IDLE == 1280, "the cells the old policy kept from the chats"
        assert self._buffer(pending_prefill = 300) == self.IDLE, "under the reaction headroom"
        assert self._buffer(pending_prefill = 1200) == 1200 + self.DRAFTS * self.SLOTS
        assert self._buffer(pending_prefill = 5000) == self.PREFILL

    def test_the_lower_ceiling_really_hands_the_cells_out(self):
        c = self._c()
        for i in range(self.SLOTS):
            c.register(f"chat{i}", tokens = 1700, signal = PreemptSignal())
            c.observe(f"chat{i}", 1)
        # 6804: over the old 6136 ceiling, under the 7416 that stands with nothing prefilling.
        assert c.committed_tokens() == 6804 and c.snapshot().prefilling == 0
        assert c.plan_preemptions(needed = 0) == []
        c.note_tokens("chat0", 1701 + 2100)
        assert c.snapshot().prefilling == 2100
        assert c.plan_preemptions(needed = 0), "the growing round's chunk was not covered"

        higher = self._c("test://higher")
        for i in range(self.SLOTS):
            higher.register(f"chat{i}", tokens = 1900, signal = PreemptSignal())
            higher.observe(f"chat{i}", 1)
        assert higher.committed_tokens() == 7604 > self.BUDGET - self.IDLE
        assert higher.plan_preemptions(needed = 0), "7604 is past the 7416 dynamic ceiling"


# ============================================================================== who stops


class TestWhoStops:
    def test_the_newest_chat_stops_first_and_size_does_not_decide(self):
        controller = _controller()
        _fill(controller, "big_and_early", 0.75)
        _fill(controller, "small_and_late", 0.35)
        assert {p.gen_id for p in controller.plan_preemptions()} == {"small_and_late"}

    def test_a_parked_chat_is_taken_first_and_one_running_tools_is_never_taken(self):
        controller = _controller()
        _register(controller, "winner", 9000)
        _register(controller, "decoding", 4000)
        _register(controller, "parked", 4000, state = ParticipantState.PARKED_ON_TOOL)
        _register(controller, "tools", 4000, state = ParticipantState.TOOLS_RUNNING)
        victims = [p.gen_id for p in controller.plan_preemptions()]
        assert victims[0] == "parked", "the cheapest room to take, and the benchmarked order"
        assert "tools" not in victims, "nothing decodes there, and it is the unsafe window"

    def test_a_victim_is_marked_and_signalled_together_and_still_holds_its_cells(self):
        """Four chats armed, two were chosen as victims, neither ever paused."""
        controller = _controller(key = "victim-signal")
        winner = _fill(controller, "winner", 0.75)
        victim = _fill(controller, "victim", 0.35)
        before = controller.committed_tokens()
        controller.plan_preemptions()
        assert victim.state == ParticipantState.PREEMPTING
        assert victim.preempt_event.is_set(), "the decision and the signal must not drift"
        assert not winner.preempt_event.is_set(), "the winner must keep decoding"
        assert controller.committed_tokens() == before, (
            "asking for a pause freed room that is still occupied; it is free only once "
            "the stream confirms it stopped"
        )
        controller.set_state(victim.gen_id, ParticipantState.PAUSED)
        assert controller.committed_tokens() == before - victim.tokens

    def test_the_biggest_chat_is_still_preemptable_and_the_last_holder_survives(self):
        """The epoch winner is gone, and this is what replaced it."""
        controller = _controller()
        # The huge chat is deliberately NOT the oldest: as the oldest it would be taken last
        # and then spared by the last-holder rule, making this assertion unreachable.
        _fill(controller, "oldest", 0.14)
        _fill(controller, "huge", 0.72)
        _fill(controller, "newest", 0.14)
        victims = {p.gen_id for p in controller.plan_preemptions(needed = 6000)}
        assert "huge" in victims, "an exempt chat can grow until it fills the window"
        assert "oldest" not in victims, "the last holder standing must survive"
        assert controller.snapshot().winner is None, "nobody is crowned any more"

    def test_a_lone_holder_is_not_preempted_for_a_newcomer(self):
        controller = _controller()
        _register(controller, "alone", 15000)
        assert controller.plan_preemptions(needed = 4000) == []

    def test_an_unpreemptable_holder_counts_as_the_one_left_standing(self):
        c = _controller(budget = 8192, key = "raw-standing", slots = 4, batch_tokens = 2048)
        raw = c.register("raw", tokens = 4000, state = ParticipantState.STREAMING_RAW)
        c.note_tokens("raw", 4000)
        chat = c.register("chat", tokens = 1000)
        c.note_tokens("chat", 1000)
        assert c.observe("chat", 1000) == [], "room while under the ceiling"
        # Over it, the only preemptable holder is chosen rather than spared, because the raw
        # stream is standing already and pausing nobody ends both.
        assert [v.gen_id for v in c.observe("chat", 1500)] == ["chat"]
        assert raw.state == ParticipantState.STREAMING_RAW
        assert chat.state == ParticipantState.PREEMPTING


class TestAReclaimedHolderIsNotAVictim:
    """It holds no cells, so pausing it frees none."""

    def test_a_parked_holder_whose_cells_were_erased_is_skipped_for_a_live_one(self):
        controller = _controller(key = "sweep-1", draft_tokens = 0, slots = 4, batch_tokens = 0)
        controller.register("parked", tokens = 9000)
        controller.register("live-a", tokens = 8000)
        controller.register("live-b", tokens = 8000)
        controller.note_state("parked", ParticipantState.PARKED_ON_TOOL)
        controller.note_cells_reclaimed()
        assert controller.committed_tokens() > 15616, "the sweep has to be under pressure"
        victims = [v.gen_id for v in controller.plan_preemptions(needed = 0)]
        assert victims == [
            "live-b"
        ], "newest-first among the holders that still have cells, with one left standing"
        assert not controller.participant("parked").preempt_event.is_set()


class TestStarvation:
    """Losing repeatedly must not become never finishing."""

    def _starve(self, controller):
        _fill(controller, "hog", 0.82)
        starved = _fill(controller, "starved", 0.28)
        for _ in range(PROMOTE_AFTER_CONSECUTIVE_PREEMPTIONS):
            assert [p.gen_id for p in controller.plan_preemptions()] == ["starved"]
            controller.note_resumed("starved")
        return starved

    def test_three_consecutive_preemptions_promote_a_chat_and_it_is_taken_last(self):
        controller = _controller()
        starved = self._starve(controller)
        assert starved.promoted, "three preemptions in a row must promote it"
        _fill(controller, "newcomer", 0.28)
        victims = [p.gen_id for p in controller.plan_preemptions()]
        assert victims and victims[0] != "starved", f"taken first anyway, order was {victims}"


class TestTheSwitchesThatTurnItOff:
    @pytest.mark.parametrize("budget, unified", [(16384, False), (0, True)])
    def test_a_private_cache_or_an_unknown_budget_plans_nothing(self, budget, unified):
        controller = _controller(budget = budget, kv_unified = unified)
        _register(controller, "a", 15000)
        _register(controller, "b", 15000)
        assert controller.plan_preemptions() == []

    def test_the_env_switch_disables_it(self, monkeypatch):
        controller = _controller()
        _register(controller, "a", 15000)
        _register(controller, "b", 15000)
        assert controller.plan_preemptions(), "on by default"
        controller.note_resumed("b")
        monkeypatch.setenv("UNSLOTH_LLAMA_ADMISSION_PREEMPT", "0")
        assert controller.plan_preemptions() == [] and controller.active is False

    def test_the_signal_is_cleared_with_the_state_and_it_can_be_chosen_again(self):
        controller = _controller(key = "declined-signal")
        signal = PreemptSignal()
        controller.register("other", tokens = 1000)
        participant = controller.register("chat", tokens = 1000, signal = signal)
        controller.plan_preemptions(needed = 16384)
        controller.note_declined("chat")
        assert participant.state == ParticipantState.DECODING
        assert (
            not signal.is_set() and not signal.pending
        ), "a signal left set aborts the very stream this call is letting run"
        assert [v.gen_id for v in controller.plan_preemptions(needed = 16384)] == [
            "chat"
        ], "the point of the handback: pressure later in the turn can ask again"


# ============================================================================== the ledger


class TestTheLedger:
    def test_live_growth_is_added_and_a_round_boundary_rebaselines(self):
        controller = _controller(key = "sweep")
        controller.register("a", tokens = 1000)
        controller.observe("a", 500)
        assert controller.committed_tokens() == 1500
        controller.note_tokens("a", 9000)
        controller.observe("a", 100)
        assert controller.committed_tokens() == 9100, "growth is measured from the new round"

    def test_a_round_that_grows_past_the_watermark_evicts_the_newest_arrival(self):
        controller = _controller(key = "prefill-evict", slots = 4)
        controller.register("older", tokens = 4000)
        controller.register("newer", tokens = 4000)
        assert controller.plan_preemptions() == [], "nothing to do yet"
        controller.note_tokens("newer", _ceiling(controller))
        assert [p.gen_id for p in controller.observe("newer", 0)], "nobody was asked to stop"

        sweep = _controller(key = "sweep-newest")
        for index in range(4):
            sweep.register(f"c{index}", tokens = 2000, signal = PreemptSignal())
        sweep.register("big", tokens = 9000, signal = PreemptSignal())
        assert "big" in {v.gen_id for v in sweep.observe("big", 3000)}

    def test_a_replay_raises_the_baseline_and_replays_accumulate(self):
        """Carrying a partial across a pause moves tokens from generated to prompt."""
        controller = _controller(key = "replay")
        controller.register("a", tokens = 1000)
        controller.observe("a", 500)
        controller.note_replayed("a", 500)
        controller.observe("a", 0)
        assert (
            controller.participant("a").tokens == 1500
        ), "occupancy fell back to the admission prompt and lost the replayed partial"
        for charged in (564, 59, 1079):
            controller.note_replayed("a", charged)
        controller.note_replayed("a", 0)
        controller.observe("a", 0)
        assert controller.participant("a").tokens == 1500 + 564 + 59 + 1079
        controller.note_replayed("gone", 500)  # an unknown generation is ignored

    def test_the_adapter_charges_only_what_was_actually_kept(self):
        controller = _controller(key = "adapter")
        controller.register("a", tokens = 1000)
        policy = ControllerPreemptionPolicy(controller, "a", PreemptSignal())
        policy.on_preempted(StreamCheckpoint(charged_tokens = 700))
        controller.observe("a", 0)
        assert controller.participant("a").tokens == 1000, "nothing was kept, nothing charged"
        policy.on_preempted(StreamCheckpoint(charged_tokens = 700, reasoning_text = "a thought"))
        controller.observe("a", 0)
        assert controller.participant("a").tokens == 1700

    def test_unregistering_drops_the_charge_including_the_replays(self):
        """The ledger only ever grew, so eventually nobody could be admitted."""
        controller = _controller(key = "disarm-test")
        controller.register("a", tokens = 8000)
        controller.register("b", tokens = 4000)
        controller.note_replayed("a", 4000)
        controller.observe("a", 0)
        assert controller.snapshot().committed == 16000
        controller.unregister("a")
        controller.unregister("a")  # twice is harmless
        assert controller.snapshot().committed == 4000
        assert controller.participant("a") is None
        assert controller.participant("b").state == ParticipantState.DECODING

    def test_tokens_arriving_move_a_tool_holder_back_to_decoding(self):
        """A chat that answers a tool result with another tool call is still pausable."""
        c = _controller(budget = 8192, key = "toolcall", slots = 4, batch_tokens = 2048)
        chat = c.register("chat", tokens = 1000)
        c.note_state("chat", ParticipantState.TOOLS_RUNNING)
        assert chat.preemptable is False
        c.observe("chat", 32)
        assert chat.state == ParticipantState.DECODING
        assert chat.preemptable is True and chat.tokens == 1032
        c.note_state("chat", ParticipantState.PARKED_ON_TOOL)
        c.observe("chat", 1)
        assert chat.state == ParticipantState.DECODING


class TestAParkedHolderGivesItsCellsBack:
    """A chat stopped on a tool approval must not keep waiting chats out of an empty cache."""

    def _c(self, key = "test://parked"):
        # 8192 cells and a buffer well under 2100, so a 6000-plus ceiling: the failing shape.
        return _controller(budget = 8192, key = key, slots = 4, draft_tokens = 2, batch_tokens = 2048)

    def test_a_parked_holder_whose_cells_were_reclaimed_stops_counting(self):
        c = self._c()
        leader = c.register("leader", tokens = 3847)
        c.note_tokens("leader", 3847)
        c.register("waiter", tokens = 3092, state = ParticipantState.PAUSED)
        c.note_resident(4676, 0)
        assert c.try_grant_resume("waiter", 3092) is False, "no room beside the leader"

        assert c.note_state("leader", ParticipantState.PARKED_ON_TOOL) is True
        assert c.snapshot().parked == 1 and c.note_cells_reclaimed() == 1
        c.note_resident(0, 0)
        assert leader.holds_kv is False
        assert c.snapshot().committed == 0, (
            "the cache is empty and the ledger must say so; charging cells that were "
            "erased is what kept two waiters out for three minutes"
        )
        assert c.try_grant_resume("waiter", 3092) is True

    def test_the_reclaim_hands_back_the_admission_commitment_once_per_park(self):
        class _Lease:
            yielded = 0

            def yield_parked_commitment(self):
                self.yielded += 1
                return 3847

        lease = _Lease()
        c = self._c("test://parked-lease")
        c.register("leader", tokens = 3847, lease = lease)
        c.note_state("leader", ParticipantState.PARKED_ON_TOOL)
        assert c.note_cells_reclaimed() == 1 and lease.yielded == 1
        assert c.note_cells_reclaimed() == 0, "once per park, not once per sweep"
        assert lease.yielded == 1


# =========================================================================== the residency


class TestReadingWhatTheCacheActuallyHolds:
    """`purging slot 1 with 16383 tokens`, observed 2026-09-01."""

    def test_idle_caches_are_occupancy_and_separately_reclaimable(self):
        occupancy = read_slot_occupancy(
            lambda: [
                {"id": 0, "is_processing": False, "n_prompt_tokens_cache": 16383},
                {"id": 1, "is_processing": True, "n_prompt_tokens_cache": 2000},
                {"id": 2, "is_processing": False, "n_prompt_tokens_cache": 9209},
            ]
        )
        assert occupancy["resident"] == 27592, (
            "idle caches are only recycled by llama.cpp from the KV-full retry, i.e. after "
            "a decode has already failed, so they are occupancy to us"
        )
        assert occupancy["idle_tokens"] == 25592
        assert [slot for slot, _ in occupancy["idle"]] == [0, 2], "largest idle first"

    def test_generated_tokens_are_added_to_the_cache_field_and_only_to_it(self):
        # `n_prompt_tokens` ALREADY includes them: over 128 samples `n_prompt_tokens - n_decoded`
        # is constant within a request, and adding them again scored 28238 in a 16384 cache.
        def _slot(field, processing = True):
            return [
                {
                    "id": 0,
                    "is_processing": processing,
                    field: 12632,
                    "next_token": [{"n_decoded": 6323}],
                }
            ]

        assert read_slot_occupancy(lambda: _slot("n_prompt_tokens"))["resident"] == 12632
        assert (
            read_slot_occupancy(lambda: _slot("n_prompt_tokens_cache"))["resident"] == 18955
        ), "n_prompt_tokens_cache is the prompt only, so generation is still missing"
        finished = read_slot_occupancy(lambda: _slot("n_prompt_tokens_cache", processing = False))
        assert (
            finished["resident"] == 12632
        ), "a finished slot's cache already holds the whole sequence it produced"
        assert finished["idle_tokens"] == 12632

    def test_an_unreadable_endpoint_is_not_an_empty_cache(self):
        assert read_slot_occupancy(lambda: None) is None
        assert read_slot_occupancy(lambda: []) is None


class TestTheCacheHoldsMoreThanTheLedgerKnows:
    def test_a_chat_that_has_not_prefilled_is_added_to_what_the_cache_holds(self):
        controller = _controller(key = "resident")
        controller.register("a", tokens = 2000, signal = PreemptSignal())
        controller.note_resident(16383)
        # "a" has not decoded a token, so its 2000 are NOT among the 16383 already held:
        # they are a prefill still to come, and both have to fit.
        assert controller.committed_tokens() == 18383
        controller.note_resident(None)
        assert controller.committed_tokens() == 2000, "a failed read falls back, not to zero"

    def test_a_chat_already_in_the_cache_is_not_counted_a_second_time(self):
        controller = _controller(key = "resident-measured")
        controller.register("a", tokens = 4096, signal = PreemptSignal())
        controller.observe("a", 1)
        controller.note_resident(1400)
        assert controller.committed_tokens() == 4097, "the larger of the two opinions"
        controller.note_tokens("a", 1400)
        assert controller.committed_tokens() == 1400, "counted once, not once per source"

    def test_idle_residue_does_not_stand_between_a_waiter_and_its_resume(self):
        controller = _controller(key = "idle-deadlock", slots = 4)
        controller.register("waiter", tokens = 5000, signal = PreemptSignal())
        controller.set_state("waiter", ParticipantState.PAUSED)
        controller.note_resident(21304, 21304)
        assert controller.room_for("waiter", 5000) is True, "reclaimable residue must not block"
        # 16000 live cells, minus this waiter's own 5000, still leaves 11000 that a 5000
        # token resume cannot fit beside.
        controller.note_resident(16000, 0)
        assert controller.room_for("waiter", 5000) is False

    def test_the_residency_probe_is_optional_and_can_never_raise(self):
        controller = _controller(key = "probe")
        calls = []
        controller.set_residency_probe(lambda: calls.append(1))
        controller.refresh_residency()
        assert calls == [1]
        controller.set_residency_probe(lambda: (_ for _ in ()).throw(RuntimeError("slots down")))
        controller.refresh_residency()  # must not raise
        PreemptionController("no-probe").refresh_residency()


class TestReclaimingIdleResidue:
    def test_the_busy_slot_is_never_erased_and_the_loop_stops_at_what_it_needed(self):
        occupancy = read_slot_occupancy(
            lambda: [
                {"id": 0, "is_processing": False, "n_prompt_tokens_cache": 16383},
                {"id": 1, "is_processing": True, "n_prompt_tokens_cache": 2000},
            ]
        )
        erased = []

        def _erase(slot_id):
            erased.append(slot_id)
            return dict(occupancy["idle"])[slot_id]

        assert reclaim_idle_slots(occupancy, _erase, needed = 10000) == 16383
        assert erased == [0], "the busy slot must never be erased"

    def test_a_partial_reclaim_leaves_the_rest_holding_their_cells(self):
        """`reclaim_idle_slots` stops at `needed`; `note_cells_reclaimed` does not."""
        occupancy = {
            "resident": 9000,
            "idle_tokens": 9000,
            "idle": [(0, 3000), (1, 3000), (2, 3000)],
        }
        erased: list[int] = []
        freed = reclaim_idle_slots(
            occupancy, lambda slot_id: (erased.append(slot_id), 3000)[1], needed = 3000
        )
        assert freed == 3000 and erased == [0]
        assert freed < occupancy["idle_tokens"], (
            "which is exactly the condition the caller has to check before telling the "
            "ledger that every parked holder lost its cells"
        )


class TestTheSlotProbesAreAuthenticated:
    """``/slots`` is not a public endpoint."""

    def test_both_probes_carry_the_key_or_nothing_at_all(self, monkeypatch):
        seen: list[urllib.request.Request] = []

        class _Response:
            status = 200

            def read(self):
                return json.dumps({"n_erased": 4096}).encode()

            def __enter__(self):
                return self

            def __exit__(self, *_exc):
                return False

        def _urlopen(request, timeout = None):
            # A bare URL would leave nowhere to put the header at all.
            assert isinstance(request, urllib.request.Request), request
            seen.append(request)
            return _Response()

        monkeypatch.setattr(urllib.request, "urlopen", _urlopen)
        key = {"Authorization": "Bearer secret"}
        fetch_llama_slots("http://127.0.0.1:8080", headers = key)
        assert seen[0].get_header("Authorization") == "Bearer secret"
        fetch_llama_slots("http://127.0.0.1:8080")
        assert seen[1].get_header("Authorization") is None
        assert erase_llama_slot("http://127.0.0.1:8080", 2, headers = key) == 4096
        assert seen[2].get_header("Authorization") == "Bearer secret"
        assert seen[2].get_method() == "POST"


# ================================================================ the resume grant and waits


class TestTheResumeGrantChecksRoom:
    def test_two_waiters_cannot_both_be_granted_the_same_room(self):
        controller = _controller(key = "double-grant", slots = 4)
        for gen_id in ("a", "b"):
            controller.register(gen_id, tokens = 9000, signal = PreemptSignal())
            controller.set_state(gen_id, ParticipantState.PAUSED)
        # Each fits alone; together they do not. That is exactly the shape that crashed.
        assert 9000 <= _ceiling(controller) < 18000
        assert controller.room_for("a", 9000) is True
        assert controller.room_for("b", 9000) is True, "the same answer twice, which is the bug"
        assert controller.try_grant_resume("a", 9000) is True
        assert controller.try_grant_resume("b", 9000) is False, "the first has booked the room"
        controller.note_resume_failed("a")
        assert (
            controller.try_grant_resume("b", 9000) is True
        ), "room booked by a resume that never happened must not stay booked"


class TestAChatThatOutgrewTheSharedCeiling:
    """No eviction can admit it, so waiting for room is a deadlock, not a delay."""

    def _c(self):
        return _controller(key = "solo-test", draft_tokens = 2, slots = 4)

    def test_a_want_past_the_shared_ceiling_fits_alone_and_nowhere_else(self):
        controller = self._c()
        ceiling = _ceiling(controller)
        assert not controller.outgrew_the_shared_ceiling(ceiling)
        assert controller.outgrew_the_shared_ceiling(ceiling + 1)
        assert not controller.cannot_ever_fit(ceiling + 1), "ending the turn would be premature"
        assert controller.room_for("solo", ceiling + 500), "alone, it fits"
        _register(controller, "other", 3000)
        assert not controller.room_for("solo", ceiling + 500), "beside anyone, it does not"

    def test_a_want_past_the_cache_itself_can_never_fit(self):
        controller = self._c()
        assert controller.cannot_ever_fit(16384)
        assert not controller.cannot_ever_fit(10000)


class _Lease:
    is_released = False
    tokens = 0

    async def resume_async(self, want, **kwargs):
        return True


class TestTheResumeWait:
    """A 33-minute hang with nothing decoding, observed 2026-09-01."""

    @staticmethod
    def _pinned(key):
        controller = _controller(key = key)
        controller.register("holder", tokens = 1000, signal = PreemptSignal())
        controller.note_tokens("holder", 1000)
        # Far above the ledger's own sum, which pins `committed` at a number the decoders
        # do not move, and past the ceiling, which keeps a waiter refused.
        controller.note_resident(16384)
        return controller

    @staticmethod
    def _waiting_policy(
        controller,
        gen_id,
        tokens,
        *,
        paused = False,
    ):
        loop = asyncio.new_event_loop()
        threading.Thread(target = loop.run_forever, daemon = True).start()
        controller.register(gen_id, lease = _Lease(), tokens = tokens, signal = PreemptSignal())
        if paused:
            controller.set_state(gen_id, ParticipantState.PAUSED)
        return ControllerPreemptionPolicy(controller, gen_id, PreemptSignal(), loop = loop), loop

    @staticmethod
    def _shutdown(loop):
        loop.call_soon_threadsafe(loop.stop)

    def test_stop_before_the_wait_books_no_room(self):
        # with room to spare the grant would succeed at once, and a Stop read only inside the
        # loop was skipped, sending the stopped chat into the resume
        controller = PreemptionController("stop-first")
        controller.configure(budget = 16384, kv_unified = True)
        policy, loop = self._waiting_policy(controller, "chat", 1000, paused = True)
        try:
            stop = threading.Event()
            stop.set()
            assert policy.await_resume(timeout = 5.0, cancel_event = stop) is False
            assert controller.participant("chat").state == ParticipantState.PAUSED
        finally:
            self._shutdown(loop)

    def test_the_progress_signature_sees_a_decoding_backend_committed_cannot(self):
        """ "no progress for 90.0s" about three chats decoding at full rate."""
        controller = self._pinned("pinned")
        before = controller.progress_signature()
        controller.observe("holder", 512)
        after = controller.progress_signature()
        assert after[0] == before[0] == 16384, "committed must be pinned for this test"
        assert after[1] == before[1], "and the same holder must still hold"
        assert after != before and after[2] > before[2], "the token total is what moves"
        # A fresh attempt restarting its count is not read as lost tokens.
        controller.observe("holder", 32)
        assert controller.progress_signature()[2] > after[2]
        # And a tool call and a round boundary are progress too.
        assert controller.note_state("holder", ParticipantState.TOOLS_RUNNING) is True
        tools = controller.progress_signature()
        assert tools[3] == after[3] + 1
        controller.note_tokens("holder", 1400)
        assert controller.progress_signature()[2] > tools[2]

    def test_a_stalled_wait_still_ends_near_its_timeout(self):
        controller = _controller(key = "stalled")
        controller.register("holder", tokens = 14000, signal = PreemptSignal())
        controller.note_tokens("holder", 14000)  # measured, and going nowhere
        policy, loop = self._waiting_policy(controller, "waiter", 8000)
        try:
            started = time.monotonic()
            assert policy.await_resume(timeout = 0.5) is False
            elapsed = time.monotonic() - started
        finally:
            self._shutdown(loop)
        assert 0.4 < elapsed < 5.0, f"stall must end near the timeout, took {elapsed}s"

    def test_it_waits_while_anything_moves_and_takes_the_room_when_it_is_real(self):
        timeout = 0.4
        controller = self._pinned("still-decoding")
        policy, loop = self._waiting_policy(controller, "waiter", 4000, paused = True)
        decoded = {"tokens": 0}
        stop = threading.Event()

        def decode_then_free():
            # Long enough that a wall-clock or committed-only bound has certainly fired.
            deadline = time.monotonic() + timeout * 3
            generated = 0
            while time.monotonic() < deadline and not stop.is_set():
                time.sleep(0.02)
                generated += 32
                controller.observe("holder", generated)
                decoded["tokens"] = generated
            # The blocker finishes: its cells come back and the waiter fits at last.
            controller.note_resident(1000)
            controller.note_tokens("holder", 1000)

        worker = threading.Thread(target = decode_then_free, daemon = True)
        worker.start()
        try:
            started = time.monotonic()
            resumed = policy.await_resume(timeout = timeout)
            elapsed = time.monotonic() - started
        finally:
            stop.set()
            worker.join(timeout = 5)
            self._shutdown(loop)

        assert decoded["tokens"] > 0, "the fixture never decoded; the test proves nothing"
        assert controller.committed_tokens() <= 16384
        assert elapsed > timeout * 2, (
            f"gave up after {elapsed}s of a backend generating tokens throughout; a chat "
            "queued behind live answers waits, however long that takes"
        )
        assert resumed is True

    def test_the_hard_backstop_outlasts_the_slowest_answer_measured(self):
        bound = DEFAULT_RESUME_WAIT_TIMEOUT_S * MAX_RESUME_WAIT_MULTIPLE
        assert bound >= 2 * (8192 / 2.3), f"the backstop is {bound}s, shorter than two answers"


class TestTheAdmissionLeaseGivesItsCommitmentBack:
    """`park` keeps its tokens because the task is alive. Preemption ends it."""

    @pytest.mark.asyncio
    async def test_preempt_hands_back_both_the_slot_and_the_tokens_once(self):
        queue = LlamaAdmissionQueue("k")
        lease = await _lease(queue, tokens = 4000)
        assert lease.park() is True, "park budget should be available on a 1-slot queue"
        assert queue.snapshot().committed == 4000, "park must NOT hand the tokens back"
        assert lease.preempt() is True
        assert queue.snapshot().committed == 0 and queue.snapshot().active == 0
        assert lease.is_preempted is True
        assert lease.preempt() is False, "idempotent"

    @pytest.mark.asyncio
    async def test_a_parked_lease_resumes_into_its_own_slot_and_is_not_recharged_twice(self):
        queue = LlamaAdmissionQueue("k")
        lease = await _lease(queue, tokens = 4000)
        assert lease.park() is True and lease.preempt() is True
        assert await lease.resume_async(4000, poll_s = 0.001, timeout_s = 1.0) is True
        assert queue.snapshot().active == 0, "the park still owns the slot"
        assert queue.snapshot().committed == 4000, "only the commitment came back"
        assert await lease.resume_async(9999) is True
        assert queue.snapshot().committed == 4000, "an unpreempted lease must not be re-costed"

    # ============================================================ the barrier and the registry

    def test_an_unreadable_metrics_endpoint_times_out_rather_than_blocking(self):
        clock = iter([0.0, 0.0, 1.0, 99.0])
        assert (
            wait_for_reclaim(
                lambda: None,
                target_processing = 0,
                timeout_s = 1.0,
                sleep = lambda _s: None,
                monotonic = lambda: next(clock),
            )
            is False
        )


class TestTheRegistry:
    def test_one_controller_per_key_and_an_idle_one_retires_but_a_busy_one_is_kept(self):
        assert get_preemption_controller("a") is get_preemption_controller("a")
        assert get_preemption_controller("a") is not get_preemption_controller("b")
        first = get_preemption_controller("port-1")
        get_preemption_controller("port-2")
        assert get_preemption_controller("port-1") is not first
        busy = get_preemption_controller("port-1")
        busy.register("live", tokens = 100)
        get_preemption_controller("port-2")
        assert get_preemption_controller("port-1") is busy
