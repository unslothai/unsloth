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
from core.inference.llama_cpp import (
    LlamaCppBackend,
    _CombinedCancelEvent,
    _interrupt_event,
)
from core.inference.llama_preemption import (
    DEFAULT_PREEMPT_BUFFER_MIN_TOKENS,
    DEFAULT_RESUME_WAIT_TIMEOUT_S,
    MAX_RESUME_WAIT_MULTIPLE,
    PENDING_PREFILL_TTL_S,
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


def _controller(budget = 16384, kv_unified = True, key = "test", **kw):
    controller = PreemptionController(key)
    controller.configure(budget = budget, kv_unified = kv_unified, **kw)
    return controller


def _ceiling(controller):
    snapshot = controller.snapshot()
    return snapshot.budget - snapshot.buffer


def _register(controller, gen_id, tokens, state = ParticipantState.DECODING, **kw):
    return controller.register(gen_id, tokens = tokens, state = state, **kw)


def _fill(controller, gen_id, fraction, state = ParticipantState.DECODING):
    return _register(controller, gen_id, int(_ceiling(controller) * fraction), state = state)


async def _lease(queue, *, tokens, capacity = 4, budget = 16384):
    reservation = queue.reserve(
        capacity = capacity, config = LlamaAdmissionConfig(), tokens = tokens, budget = budget
    )
    lease = reservation.lease_nowait()
    assert lease is not None, "expected an immediate admission"
    return lease


# =============================================================================== the signal


class TestTheSignalAsksAndForgets:
    def test_a_request_is_visible_carries_its_reason_and_can_be_forgotten(self):
        signal = PreemptSignal()
        assert not signal.is_set()
        signal.request("kv_pressure")
        assert signal.is_set() and signal.pending and signal.reason == "kv_pressure"
        signal.clear()
        assert not signal.is_set() and not signal.pending and signal.reason is None


class TestTheUnsafeWindow:
    """The property tool execution depends on: hidden, but never dropped."""

    def test_a_request_inside_the_window_is_hidden_and_arrives_when_it_closes(self):
        signal = PreemptSignal()
        with signal.unsafe_window():
            signal.request()
            assert not signal.is_set(), "a pause must not land during tool execution"
            assert signal.pending, "but it must not be forgotten either"
        assert signal.is_set()

    def test_a_request_made_before_the_window_is_hidden_too(self):
        signal = PreemptSignal()
        signal.request()
        with signal.unsafe_window():
            assert not signal.is_set()
        assert signal.is_set()

    def test_nesting_waits_for_the_outermost(self):
        signal = PreemptSignal()
        with signal.unsafe_window():
            with signal.unsafe_window():
                signal.request()
                assert not signal.is_set()
            assert not signal.is_set(), "the outer window is still open"
        assert signal.is_set()

    def test_no_request_means_nothing_fires_on_close(self):
        signal = PreemptSignal()
        with signal.unsafe_window():
            pass
        assert not signal.is_set()

    def test_clearing_inside_the_window_really_forgets(self):
        signal = PreemptSignal()
        signal.request()
        with signal.unsafe_window():
            signal.clear()
        assert not signal.is_set(), "a cleared request must not resurface on close"

    def test_the_window_closes_even_when_the_body_raises(self):
        signal = PreemptSignal()
        with pytest.raises(ValueError):
            with signal.unsafe_window():
                signal.request()
                raise ValueError("tool blew up")
        assert not signal.deferred
        assert signal.is_set()


class TestCancelAndPauseTogether:
    """The stream plumbing takes one event, so cancel and pause share it."""

    def test_either_one_interrupts(self):
        for trip in (lambda c, p: p.request(), lambda c, p: c.set()):
            cancel, pause = threading.Event(), PreemptSignal()
            combined = _interrupt_event(cancel, pause)
            assert isinstance(combined, _CombinedCancelEvent)
            assert not combined.is_set()
            trip(cancel, pause)
            assert combined.is_set()

    def test_one_event_passes_straight_through(self):
        cancel = threading.Event()
        assert _interrupt_event(cancel, None) is cancel
        assert _interrupt_event(None, None) is None

    def test_a_deferred_pause_does_not_interrupt(self):
        pause = PreemptSignal()
        combined = _interrupt_event(threading.Event(), pause)
        with pause.unsafe_window():
            pause.request()
            assert not combined.is_set()

    def test_wait_returns_when_the_pause_arrives_and_gives_up_without_one(self):
        pause = PreemptSignal()
        combined = _interrupt_event(threading.Event(), pause)
        assert combined.wait(timeout = 0.05) is False
        threading.Timer(0.05, pause.request).start()
        assert combined.wait(timeout = 5.0) is True


class TestTheRolloutSwitch:
    @pytest.mark.parametrize(
        ("value", "enabled"),
        [(None, True), ("0", False), ("false", False), ("OFF", False), ("maybe", True)],
    )
    def test_it_is_on_unless_the_environment_plainly_says_otherwise(
        self, monkeypatch, value, enabled
    ):
        if value is None:
            monkeypatch.delenv(preemption.PREEMPT_ENV, raising = False)
        else:
            monkeypatch.setenv(preemption.PREEMPT_ENV, value)
        assert preemption.preemption_enabled() is enabled

    def test_the_default_policy_never_pauses_and_satisfies_the_protocol(self):
        policy = preemption.NullPreemptionPolicy()
        assert policy.should_preempt() is False
        assert policy.await_resume() is True
        assert isinstance(policy, preemption.PreemptionPolicy)


class TestTheDeferredHandoff:
    """The generator is BUILT before admission returns and ITERATED after."""

    def test_unbound_is_inert_rather_than_crashing(self):
        policy = DeferredPreemptionPolicy()
        assert policy.bound is False
        assert policy.should_preempt() is False
        assert policy.await_resume(timeout = 0.01) is False
        policy.on_resumed()
        policy.on_declined()

    def test_binding_forwards(self):
        controller = _controller(key = "wiring-bind")
        signal = PreemptSignal()
        controller.register("g", tokens = 10)
        policy = DeferredPreemptionPolicy()
        policy.bind(ControllerPreemptionPolicy(controller, "g", signal))
        assert policy.bound is True
        assert policy.should_preempt() is False
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
        assert not both.has_reasoning_resume_point(), (
            "resuming as a thought would push already-visible prose back into the block"
        )
        assert both.kept_chars() == len("The answer is")

    def test_nothing_generated_and_whitespace_alike_keep_nothing(self):
        for checkpoint in (
            StreamCheckpoint(),
            StreamCheckpoint(visible_text = "  \n ", reasoning_text = " \n"),
        ):
            assert checkpoint.kept_chars() == 0 or not checkpoint.has_resume_point()
            assert not checkpoint.has_resume_point()
            assert not checkpoint.has_reasoning_resume_point()


class TestTheWireCarriesAThoughtPartial:
    def test_a_reasoning_only_turn_counts_as_resumable(self):
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

    def test_a_tool_call_turn_is_never_resumable(self):
        from core.inference.chat_template_helpers import trailing_assistant_resumable

        assert not trailing_assistant_resumable(
            [
                {"role": "user", "content": "hi"},
                {
                    "role": "assistant",
                    "content": "",
                    "reasoning_content": "thinking",
                    "tool_calls": [{"id": "1", "function": {"name": "x", "arguments": "{}"}}],
                },
            ]
        )


class TestAssemblingTheResumedTurn:
    def test_nothing_produced_assembles_nothing(self):
        convo = [{"role": "user", "content": "hi"}]
        assert (
            LlamaCppBackend._assemble_preempt_resume(object(), convo, StreamCheckpoint(), "", "")
            is False
        )
        assert len(convo) == 1, "an empty assistant turn would be refused downstream"

    def test_a_thought_is_carried_as_reasoning_not_as_the_answer(self):
        convo = [{"role": "user", "content": "hi"}]
        assert (
            LlamaCppBackend._assemble_preempt_resume(
                object(),
                convo,
                StreamCheckpoint(reasoning_text = "half a thought"),
                "",
                "half a thought",
            )
            is True
        )
        assert convo[-1]["reasoning_content"] == "half a thought"
        assert convo[-1]["content"] == "", "a thought in content would be rendered as the answer"

    def test_a_second_pause_merges_rather_than_replaces_the_thought(self):
        convo = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "", "reasoning_content": "first half "},
        ]
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


def _drain(response, *, cancel_event = None, preempt_event = None):
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

    def test_a_stop_still_ends_quietly_even_beside_a_pause(self):
        for pause in (None, PreemptSignal()):
            cancel = threading.Event()
            cancel.set()
            if pause is not None:
                pause.request()
            response = _FakeResponse(["data: a\n"])
            assert _drain(response, cancel_event = cancel, preempt_event = pause) == []
            assert response.closed

    def test_a_deferred_pause_does_not_stop_the_stream(self):
        pause = PreemptSignal()
        response = _FakeResponse(["data: a\n", "data: b\n"])
        with pause.unsafe_window():
            pause.request()
            assert _drain(
                response, cancel_event = threading.Event(), preempt_event = pause
            ) == ["data: a\n", "data: b\n"]

    def test_no_signal_at_all_is_the_old_path(self):
        response = _FakeResponse(["data: a\n", "data: b\n"])
        assert _drain(response, cancel_event = threading.Event()) == ["data: a\n", "data: b\n"]

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


class TestTheSignaturesStayBackwardsCompatible:
    """Overrides and test doubles written against the old signatures still work."""

    def test_every_funnel_defaults_the_new_arguments(self):
        import inspect

        for funnel in (
            LlamaCppBackend._iter_text_cancellable,
            LlamaCppBackend._install_cancel_aware_read,
            LlamaCppBackend._stream_with_retry,
            LlamaCppBackend._open_stream,
            LlamaCppBackend._open_chat_stream_with_respawn_retry,
        ):
            params = inspect.signature(funnel).parameters
            assert "preempt_event" in params, funnel.__name__
            assert params["preempt_event"].default is None, funnel.__name__
        params = inspect.signature(LlamaCppBackend.generate_chat_completion_with_tools).parameters
        for name in ("preempt_event", "preempt_policy", "admission_output_allowance"):
            assert name in params and params[name].default is None, name


# ==================================================================== the buffer arithmetic


class TestTheBufferArithmetic:
    def test_the_buffer_is_per_slot_with_a_floor_and_a_cap(self):
        # Per SLOT, not per cache: the reaction headroom scales with how many chats decode
        # at once and not with the size of the cache.
        four = preemption_buffer_tokens(16384, slots = 4)
        assert preemption_buffer_tokens(16384, slots = 8) > four, (
            "twice the slots generate twice as much during an eviction"
        )
        assert four == preemption_buffer_tokens(65536, slots = 4), (
            "a bigger cache does not make an eviction slower"
        )
        assert preemption_buffer_tokens(2048) >= DEFAULT_PREEMPT_BUFFER_MIN_TOKENS
        # And a small share of a normal cache, or it serialises: at 15% of 16384 the
        # simulated makespan was 26890 steps against 239 at this size.
        assert four < 16384 * 0.08
        assert preemption_buffer_tokens(0) == 0

    def test_a_tiny_cache_is_reduced_not_erased(self):
        for total in (256, 512, 1024):
            buffer = preemption_buffer_tokens(total, draft_tokens = 8, slots = 8)
            assert 0 < buffer <= total // 2, f"{total}: buffer {buffer} erases the cache"

    def test_speculative_drafts_are_reserved_on_top(self):
        plain = preemption_buffer_tokens(16384, slots = 4)
        assert preemption_buffer_tokens(16384, draft_tokens = 2, slots = 4) == plain + 8, (
            "every slot may hold n_draft unaccounted tokens"
        )
        assert preemption_buffer_tokens(16384, draft_tokens = 0, slots = 4) == plain
        # And a huge draft window still cannot swallow a small cache.
        assert preemption_buffer_tokens(512, draft_tokens = 64, slots = 8) <= 256

    def test_the_controller_reserves_them(self):
        controller = _controller(key = "spec", draft_tokens = 2, slots = 4)
        # Derived, not a constant: the ratio is tunable and was raised after measurement.
        assert controller.snapshot().buffer == preemption_buffer_tokens(
            16384, draft_tokens = 2, slots = 4
        )
        assert controller.snapshot().buffer > preemption_buffer_tokens(16384)

    def test_the_ceiling_is_the_budget_minus_the_buffer(self):
        controller = _controller()
        ceiling = 16384 - preemption_buffer_tokens(16384)
        _register(controller, "a", ceiling)
        assert controller.plan_preemptions() == [], "exactly at the ceiling still fits"
        _register(controller, "b", 1)
        assert controller.plan_preemptions(), "one token past it must not"

    def test_room_asked_for_in_advance_counts(self):
        controller = _controller()
        # Sized from the ceiling rather than the figure it happened to have, so raising the
        # margin does not break the property being tested.
        _register(controller, "winner", _ceiling(controller) - 2000)
        _register(controller, "other", 1500)
        assert controller.plan_preemptions() == [], "it fits under the ceiling"
        assert controller.plan_preemptions(needed = 1000), "a request for room must be counted"

    def test_the_cache_is_never_handed_out_to_the_last_token(self):
        """Four chats died with preemption working perfectly, 2026-09-01."""
        for total in (2048, 4096, 16384, 65536, 262144):
            for slots in (2, 3, 4, 8):
                buffer = preemption_buffer_tokens(total, draft_tokens = 2, slots = slots)
                assert buffer >= 2 * slots, f"{total}/{slots}: drafts not covered"
                assert 0 < buffer < total, f"{total}/{slots}: no headroom"


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

    def test_nothing_pending_holds_back_only_reaction_headroom_and_drafts(self):
        assert self._buffer() == self.IDLE == 776
        assert self.PREFILL - self.IDLE == 1280, "the cells the old policy kept from the chats"
        c = self._c()
        for i in range(self.SLOTS):
            c.register(f"chat{i}", tokens = 800, signal = PreemptSignal())
            c.observe(f"chat{i}", 32)  # a token proves the prompt is in
        assert c.snapshot().prefilling == 0
        assert c.snapshot().buffer == self.IDLE

    def test_a_short_prompt_reserves_its_own_length_and_a_long_one_reserves_a_chunk(self):
        assert self._buffer(pending_prefill = 300) == self.IDLE
        assert self._buffer(pending_prefill = 1200) == 1200 + self.DRAFTS * self.SLOTS
        assert self._buffer(pending_prefill = 5000) == self.PREFILL

    def test_registering_a_prompt_announces_it_and_the_first_token_retires_it(self):
        c = self._c()
        c.register("chat", tokens = 1200, signal = PreemptSignal())
        assert c.snapshot().prefilling == 1200
        assert c.snapshot().buffer == 1200 + self.DRAFTS * self.SLOTS
        c.observe("chat", 1)
        assert c.snapshot().prefilling == 0
        assert c.snapshot().buffer == self.IDLE

    def test_a_round_boundary_announces_only_the_growth(self):
        c = self._c()
        c.register("chat", tokens = 6000, signal = PreemptSignal())
        c.observe("chat", 10)
        c.note_tokens("chat", 6040)  # 6010 held, 6040 stated: 30 new tokens, not 6040
        assert c.snapshot().prefilling == 30
        c.observe("chat", 0)
        assert c.snapshot().prefilling == 30, "a sweep does not retire a pending prefill"

        shrank = self._c("test://shrank")
        shrank.register("chat", tokens = 6000, signal = PreemptSignal())
        shrank.observe("chat", 10)
        shrank.note_tokens("chat", 3000)
        assert shrank.snapshot().prefilling == 0, "a round that shrank announces nothing"

    def test_granting_a_resume_announces_the_replay_and_a_failed_grant_hands_it_back(self):
        c = self._c()
        paused = c.register("paused", tokens = 3000, signal = PreemptSignal())
        c.observe("paused", 10)
        c.set_state("paused", ParticipantState.PAUSED)
        assert c.snapshot().prefilling == 0, "a paused chat submits nothing"
        assert c.try_grant_resume("paused", 3200) is True
        assert paused.state == ParticipantState.DECODING
        assert c.snapshot().prefilling == 3200
        assert c.snapshot().buffer == self.PREFILL
        c.note_resume_failed("paused")
        assert c.snapshot().prefilling == 0
        assert c.snapshot().buffer == self.IDLE

    def test_several_prefills_at_once_still_share_one_chunk(self):
        c = self._c()
        for name in "abc":
            c.register(name, tokens = 900, signal = PreemptSignal())
        assert c.snapshot().prefilling == 2700
        assert c.snapshot().buffer == self.PREFILL

    def test_a_stale_announcement_expires(self):
        c = self._c()
        chat = c.register("chat", tokens = 5000, signal = PreemptSignal())
        assert c.snapshot().buffer == self.PREFILL
        chat.pending_prefill_at -= PENDING_PREFILL_TTL_S + 1.0
        assert c.snapshot().prefilling == 0
        assert c.snapshot().buffer == self.IDLE

    def test_the_arriving_chats_chunk_is_reserved_before_the_sweep_that_arms_it(self):
        c = self._c()
        c.register("incumbent", tokens = 5800, signal = PreemptSignal())
        c.observe("incumbent", 0)
        assert c.observe("incumbent", 100) == [], "5800 alone is under the idle ceiling"
        c.register("arriving", tokens = 2000, signal = PreemptSignal())
        assert c.plan_preemptions(needed = 0), (
            "the arriving chat's chunk was not reserved before the sweep"
        )

        # And the same reserve is visible to a sweep driven by another chat's tokens.
        other = self._c("test://pending-seen")
        other.register("decoder", tokens = 5800, signal = PreemptSignal())
        other.observe("decoder", 100)
        other.register("prefiller", tokens = 2000, signal = PreemptSignal())
        assert other.snapshot().prefilling == 2000
        assert other.observe("decoder", 120), "a sweep on another chat missed the reserve"

    def test_room_for_charges_the_asker_for_the_batch_it_is_about_to_submit_but_once(self):
        c = self._c()
        c.register("holder", tokens = 4300, signal = PreemptSignal())
        c.observe("holder", 10)
        c.register("waiter", tokens = 10, signal = PreemptSignal())
        c.observe("waiter", 1)
        c.set_state("waiter", ParticipantState.PAUSED)
        # 4310 + 2500 is inside the idle ceiling of 7416, outside the 6136 that granting a
        # replay bigger than one chunk would itself create.
        assert c.room_for("waiter", 2500) is False
        assert c.room_for("waiter", 600) is True

        solo = self._c("test://solo")
        solo.register("solo", tokens = 3000, signal = PreemptSignal())
        assert solo.snapshot().prefilling == 3000
        assert solo.room_for("solo", 3000) is True, "its own announcement is charged once"

    def test_the_lower_ceiling_really_hands_the_cells_out(self):
        c = self._c()
        for i in range(self.SLOTS):
            c.register(f"chat{i}", tokens = 1700, signal = PreemptSignal())
            c.observe(f"chat{i}", 1)
        # 6804: over the old 6136 ceiling, under the 7416 that stands with nothing prefilling.
        assert c.committed_tokens() == 6804 and c.snapshot().prefilling == 0
        assert c.plan_preemptions(needed = 0) == []
        # And one of them reaching a round boundary brings the ceiling back down.
        c.note_tokens("chat0", 1701 + 2100)
        assert c.snapshot().prefilling == 2100
        assert c.plan_preemptions(needed = 0), "the growing round's chunk was not covered"

    def test_it_still_fires_once_the_lower_ceiling_is_passed(self):
        c = self._c()
        for i in range(self.SLOTS):
            c.register(f"chat{i}", tokens = 1900, signal = PreemptSignal())
            c.observe(f"chat{i}", 1)
        assert c.committed_tokens() == 7604 > self.BUDGET - self.IDLE
        assert c.plan_preemptions(needed = 0), "7604 is past the 7416 dynamic ceiling"

    def test_a_reclaimed_holder_re_announces_when_it_decodes(self):
        c = self._c()
        c.register("parked", tokens = 3000, signal = PreemptSignal())
        c.observe("parked", 10)
        c.note_state("parked", ParticipantState.PARKED_ON_TOOL)
        c.note_cells_reclaimed()
        assert c.snapshot().prefilling == 0, "cells that are gone submit nothing"
        c.note_state("parked", ParticipantState.DECODING)
        assert c.snapshot().prefilling == 3010, "the whole prompt goes back in"

    def test_a_lone_chat_still_needs_room_for_its_own_prefill_batch(self):
        """`Context size has been exceeded` six times a run, on four consecutive runs."""
        c = _controller(key = "solo-batch", slots = 4, draft_tokens = 6, batch_tokens = 2048)
        c.register("only", tokens = 100, signal = PreemptSignal())
        assert c.room_for("only", 16297) is False, (
            "87 cells is not enough for a 2048 token prefill batch"
        )
        assert c.room_for("only", 14000) is True

    @pytest.mark.parametrize(
        ("env", "expected_idle"),
        [
            ({"UNSLOTH_LLAMA_PREEMPT_STATIC_BATCH": "1"}, PREFILL),
            ({"UNSLOTH_LLAMA_PREEMPT_BUFFER_PER_SLOT": "300"}, 300 * SLOTS + DRAFTS * SLOTS),
            (
                {
                    "UNSLOTH_LLAMA_PREEMPT_BUFFER_PER_SLOT": "800",
                    "UNSLOTH_LLAMA_PREEMPT_STATIC_BATCH": "1",
                },
                800 * SLOTS + DRAFTS * SLOTS,
            ),
        ],
    )
    def test_the_overrides_restore_the_permanent_terms_and_compose(
        self, monkeypatch, env, expected_idle
    ):
        for name, value in env.items():
            monkeypatch.setenv(name, value)
        assert self._buffer() == expected_idle

    def test_an_admitted_prompt_can_be_told_to_stop_costing_a_batch(self, monkeypatch):
        """`UNSLOTH_LLAMA_PREEMPT_BATCH_ONLY_UNCHARGED=1`, off by default."""
        monkeypatch.setenv("UNSLOTH_LLAMA_PREEMPT_BATCH_ONLY_UNCHARGED", "1")
        c = self._c("test://uncharged")
        c.register("arriving", tokens = 3499, signal = PreemptSignal())
        assert c.snapshot().prefilling == 0
        assert c.snapshot().buffer == self.IDLE
        assert c.committed_tokens() == 3499, "its charge is still reserved, where it belongs"
        # A round boundary's growth still costs one.
        c.observe("arriving", 20)
        c.note_tokens("arriving", 7000)
        assert c.snapshot().buffer == self.PREFILL

    def test_the_default_still_reserves_for_an_arrival(self):
        c = self._c("test://charged-default")
        c.register("arriving", tokens = 3499, signal = PreemptSignal())
        assert c.snapshot().prefilling == 3499
        assert c.snapshot().buffer == self.PREFILL


# ============================================================================== who stops


class TestWhoStops:
    def test_the_newest_chat_stops_first_and_size_does_not_decide(self):
        controller = _controller()
        _fill(controller, "big_and_early", 0.75)
        _fill(controller, "small_and_late", 0.35)
        assert {p.gen_id for p in controller.plan_preemptions()} == {"small_and_late"}

    def test_a_parked_chat_is_taken_before_a_decoding_one(self):
        controller = _controller()
        _register(controller, "winner", 9000)
        _register(controller, "decoding", 4000)
        _register(controller, "parked", 4000, state = ParticipantState.PARKED_ON_TOOL)
        assert [p.gen_id for p in controller.plan_preemptions()][0] == "parked"

    def test_a_chat_running_tools_is_never_preempted(self):
        controller = _controller()
        _register(controller, "winner", 9000)
        _register(controller, "tools", 8000, state = ParticipantState.TOOLS_RUNNING)
        assert "tools" not in {p.gen_id for p in controller.plan_preemptions()}, (
            "nothing is decoding there, and it is the unsafe window"
        )

    def test_a_queued_chat_is_not_a_victim(self):
        controller = _controller()
        _register(controller, "winner", 15000)
        queued = _register(controller, "queued", 0, state = ParticipantState.QUEUED)
        _register(controller, "other", 2000)
        assert queued not in controller.plan_preemptions()

    def test_only_as_many_as_needed_are_paused(self):
        controller = _controller()
        _fill(controller, "first", 0.55)
        _register(controller, "second", 400)
        _register(controller, "third", 400)
        _fill(controller, "fourth", 0.48)
        victims = [p.gen_id for p in controller.plan_preemptions()]
        assert victims == ["fourth"], f"one victim was enough, got {victims}"

    def test_the_sweep_takes_everyone_when_the_room_demands_it(self):
        controller = _controller()
        for name in ("a", "b", "c"):
            _fill(controller, name, 0.33)
        victims = [p.gen_id for p in controller.plan_preemptions(needed = 14000)]
        assert len(victims) >= 2, f"the sweep stopped early, got {victims}"

    def test_a_victim_is_marked_and_signalled_together_and_still_holds_its_cells(self):
        """Four chats armed, two were chosen as victims, neither ever paused."""
        controller = _controller(key = "victim-signal")
        winner = _fill(controller, "winner", 0.75)
        victim = _fill(controller, "victim", 0.35)
        before = controller.committed_tokens()
        controller.plan_preemptions()
        assert victim.state == ParticipantState.PREEMPTING
        assert victim.preempt_event.is_set(), "the decision and the signal must not drift apart"
        assert not winner.preempt_event.is_set(), "the winner must keep decoding"
        assert controller.committed_tokens() == before, (
            "asking for a pause freed room that is still occupied; it is free only once "
            "the stream confirms it stopped"
        )
        controller.set_state(victim.gen_id, ParticipantState.PAUSED)
        assert controller.committed_tokens() == before - victim.tokens

    def test_an_already_asked_victim_is_not_asked_twice(self):
        controller = _controller(key = "no-double")
        for index in range(4):
            controller.register(f"c{index}", tokens = 4096, signal = PreemptSignal())
        first = {v.gen_id for v in controller.plan_preemptions(needed = 4096)}
        second = {v.gen_id for v in controller.plan_preemptions(needed = 4096)}
        assert not (first & second), f"re-selected {first & second}"


class TestNobodyIsExemptAndSomebodyIsAlwaysLeftStanding:
    """The epoch winner is gone, and this is what replaced it."""

    def test_the_biggest_chat_is_still_preemptable(self):
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

    def test_a_holder_already_preempting_does_not_count_as_standing(self):
        c = _controller(budget = 8192, key = "mid-sweep", slots = 4, batch_tokens = 2048)
        a = c.register("a", tokens = 1000)
        c.note_tokens("a", 3500)
        b = c.register("b", tokens = 1000)
        c.note_tokens("b", 3500)
        chosen = c.observe("b", 2500)
        assert len(chosen) == 1
        # A sweep between the decision and the pause must not take the last decoder.
        assert c.observe("a" if chosen[0] is b else "b", 2600) == []
        assert (a if chosen[0] is b else b).state == ParticipantState.DECODING

    def test_a_lone_preemptable_holder_is_never_paused(self):
        c = _controller(budget = 8192, key = "lone", slots = 4, batch_tokens = 2048)
        only = c.register("only", tokens = 1000)
        assert c.observe("only", 7000) == []
        assert only.state == ParticipantState.DECODING


class TestTheArmingSweepPrefersTheNewcomer:
    """A chat armed into a full cache becomes its own victim, and that costs it nothing."""

    def _full(self):
        controller = _controller(budget = 8192, key = "arming", slots = 4)
        for index in range(3):
            gen_id = f"decoding-{index}"
            controller.register(gen_id, tokens = 2032, signal = PreemptSignal())
            controller.note_tokens(gen_id, 2032)
        controller.register("arriving", tokens = 2032, signal = PreemptSignal())
        return controller

    def test_the_arriving_chat_is_the_only_victim_and_holds_no_cells_to_free(self):
        controller = self._full()
        assert [v.gen_id for v in controller.plan_preemptions(needed = 0)] == ["arriving"], (
            "newest-first is the benchmarked policy and the newcomer is the newest; "
            "sparing it would evict a chat that is decoding to admit one that is not"
        )
        for index in range(3):
            participant = controller.participant(f"decoding-{index}")
            assert participant.state == ParticipantState.DECODING
            assert not participant.preempt_event.is_set()
        assert controller.participant("arriving").measured is False

    def test_losing_repeatedly_still_promotes_it(self):
        controller = self._full()
        for _ in range(PROMOTE_AFTER_CONSECUTIVE_PREEMPTIONS):
            controller.plan_preemptions(needed = 0)
            controller.set_state("arriving", ParticipantState.DECODING)
        assert controller.participant("arriving").promoted is True


class TestAReclaimedHolderIsNotAVictim:
    """It holds no cells, so pausing it frees none."""

    def _three(self, key):
        controller = _controller(key = key, draft_tokens = 0, slots = 4, batch_tokens = 0)
        controller.register("parked", tokens = 9000)
        controller.register("live-a", tokens = 8000)
        controller.register("live-b", tokens = 8000)
        controller.note_state("parked", ParticipantState.PARKED_ON_TOOL)
        controller.note_cells_reclaimed()
        return controller

    def test_a_parked_holder_whose_cells_were_erased_is_not_chosen(self):
        controller = self._three("http://sweep-1")
        assert controller.committed_tokens() > 15616, "the sweep has to be under pressure"
        victims = [v.gen_id for v in controller.plan_preemptions(needed = 0)]
        assert "parked" not in victims, (
            "a holder whose cells are already gone frees nothing; choosing it spends the "
            "sweep on a chat that cannot give anything back"
        )
        assert not controller.participant("parked").preempt_event.is_set()
        assert victims == ["live-b"], (
            "newest-first among the holders that still have cells, with one left standing"
        )


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
        assert victims and victims[0] != "starved", (
            f"the promoted chat was taken first anyway, order was {victims}"
        )

    def test_the_debt_clears_once_it_runs_again_unmolested(self):
        controller = _controller()
        starved = self._starve(controller)
        controller.unregister("hog")
        controller.note_resumed("starved")
        controller.plan_preemptions()
        assert starved.consecutive_preemptions <= PROMOTE_AFTER_CONSECUTIVE_PREEMPTIONS


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
        assert controller.plan_preemptions() == []
        assert controller.active is False


class TestADeclinedPauseIsHandedBack:
    def test_only_a_chosen_victim_moves(self):
        controller = _controller(key = "declined-unit")
        paused = controller.register("paused", tokens = 10, state = ParticipantState.PAUSED)
        controller.note_declined("paused")
        assert paused.state == ParticipantState.PAUSED, (
            "a chat that really did pause has moved on under its own transition; putting "
            "it back to DECODING would invent a holder"
        )
        controller.note_declined("never-registered")  # and an unknown id is simply nothing

    def test_the_signal_is_cleared_with_the_state_and_it_can_be_chosen_again(self):
        controller = _controller(key = "declined-signal")
        signal = PreemptSignal()
        controller.register("other", tokens = 1000)
        participant = controller.register("chat", tokens = 1000, signal = signal)
        controller.plan_preemptions(needed = 16384)
        controller.note_declined("chat")
        assert participant.state == ParticipantState.DECODING
        assert not signal.is_set() and not signal.pending, (
            "a signal left set aborts the very stream this call is letting run"
        )
        assert [v.gen_id for v in controller.plan_preemptions(needed = 16384)] == ["chat"], (
            "the point of the handback: pressure later in the turn can ask again"
        )


# ============================================================================== the ledger


class TestTheLedger:
    def test_live_growth_is_added_and_a_round_boundary_rebaselines(self):
        controller = _controller(key = "sweep")
        controller.register("a", tokens = 1000)
        controller.observe("a", 500)
        assert controller.committed_tokens() == 1500
        controller.note_tokens("a", 9000)
        controller.observe("a", 100)
        assert controller.committed_tokens() == 9100, (
            "growth after a round must be measured from the round, not from admission"
        )

    def test_a_round_that_grows_past_the_watermark_evicts_someone(self):
        controller = _controller(key = "prefill-evict", slots = 4)
        controller.register("older", tokens = 4000)
        controller.register("newer", tokens = 4000)
        assert controller.plan_preemptions() == [], "nothing to do yet"
        controller.note_tokens("newer", _ceiling(controller))
        assert [p.gen_id for p in controller.observe("newer", 0)], (
            "the round grew past the watermark and nobody was asked to stop"
        )

    def test_the_newest_arrival_is_the_first_victim_of_a_watermark_sweep(self):
        controller = _controller(key = "sweep-newest")
        for index in range(4):
            controller.register(f"c{index}", tokens = 2000, signal = PreemptSignal())
        controller.register("big", tokens = 9000, signal = PreemptSignal())
        assert "big" in {v.gen_id for v in controller.observe("big", 3000)}

    def test_a_replay_raises_the_baseline_and_replays_accumulate(self):
        """Carrying a partial across a pause moves tokens from generated to prompt."""
        controller = _controller(key = "replay")
        controller.register("a", tokens = 1000)
        controller.observe("a", 500)
        controller.note_replayed("a", 500)
        controller.observe("a", 0)
        assert controller.participant("a").tokens == 1500, (
            "occupancy fell back to the admission prompt and lost the replayed partial"
        )
        for charged in (564, 59, 1079):
            controller.note_replayed("a", charged)
        controller.observe("a", 0)
        assert controller.participant("a").tokens == 1500 + 564 + 59 + 1079
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

    @pytest.mark.asyncio
    async def test_a_lease_reports_its_charge_and_a_finished_chat_stops_counting(self):
        """`getattr(lease, "tokens", 0)` silently returned 0 before this property existed."""
        queue = llama_admission.get_llama_admission_queue("wiring-tokens")
        controller = _controller(key = "wiring-prune")
        leases = []
        for index in range(2):
            lease = await _lease(queue, tokens = 4096)
            assert lease.tokens == 4096, "the charge must be readable, not silently zero"
            assert lease.is_released is False
            leases.append(lease)
            controller.register(f"gen{index}", lease = lease, tokens = 4096)
        assert controller.committed_tokens() == 8192
        leases[0].release()
        assert leases[0].is_released is True
        assert controller.committed_tokens() == 4096, (
            "a finished generation still counted against the budget"
        )

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

    def test_only_live_states_move_and_only_a_change_is_reported(self):
        c = _controller(budget = 8192, key = "states", slots = 4)
        c.register("p", tokens = 10, state = ParticipantState.PAUSED)
        assert c.note_state("p", ParticipantState.TOOLS_RUNNING) is False
        c.register("q", tokens = 10)
        assert c.note_state("q", ParticipantState.PAUSED) is False, "PAUSED is the preemptor's"
        assert c.note_state("q", ParticipantState.TOOLS_RUNNING) is True
        assert c.note_state("q", ParticipantState.TOOLS_RUNNING) is False, "no change, no report"


class TestAParkedHolderGivesItsCellsBack:
    """A chat stopped on a tool approval must not keep waiting chats out of an empty cache."""

    def _c(self, key = "test://parked"):
        # 8192 cells and a buffer well under 2100, so a 6000-plus ceiling: the failing shape.
        return _controller(
            budget = 8192, key = key, slots = 4, draft_tokens = 2, batch_tokens = 2048
        )

    def test_a_parked_holder_whose_cells_were_reclaimed_stops_counting(self):
        c = self._c()
        leader = c.register("leader", tokens = 3847)
        c.note_tokens("leader", 3847)
        c.register("waiter", tokens = 3092, state = ParticipantState.PAUSED)
        c.note_resident(4676, 0)
        assert c.try_grant_resume("waiter", 3092) is False, "no room beside the leader"

        assert c.note_state("leader", ParticipantState.PARKED_ON_TOOL) is True
        assert c.snapshot().parked == 1
        assert c.note_cells_reclaimed() == 1
        c.note_resident(0, 0)

        assert leader.holds_kv is False
        assert c.snapshot().committed == 0, (
            "the cache is empty and the ledger must say so; charging cells that were "
            "erased is what kept two waiters out for three minutes"
        )
        assert c.try_grant_resume("waiter", 3092) is True

    def test_the_charge_comes_back_when_the_holder_decodes_again(self):
        c = self._c("test://parked-back")
        leader = c.register("leader", tokens = 3847)
        c.note_tokens("leader", 3847)
        c.note_state("leader", ParticipantState.TOOLS_RUNNING)
        c.note_cells_reclaimed()
        assert leader.holds_kv is False
        c.note_state("leader", ParticipantState.DECODING)
        assert leader.holds_kv is True and leader.tokens == 3847

    def test_a_reclaim_leaves_decoding_holders_alone(self):
        c = self._c("test://parked-live")
        a = c.register("a", tokens = 1000)
        c.note_tokens("a", 1000)
        assert c.note_cells_reclaimed() == 0
        assert a.holds_kv is True

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
        assert c.note_cells_reclaimed() == 1
        assert lease.yielded == 1
        assert c.note_cells_reclaimed() == 0, "once per park, not once per sweep"
        assert lease.yielded == 1

    @pytest.mark.asyncio
    async def test_yield_parked_commitment_frees_the_pool_and_recost_takes_it_back(self):
        queue = llama_admission.get_llama_admission_queue("http://lease.test")
        lease = await _lease(queue, tokens = 3847, budget = 8192)
        assert queue.committed_now() == 3847
        assert lease.park() is True  # park() wants a running loop for the waiters it may wake

        assert lease.yield_parked_commitment() == 3847
        assert queue.committed_now() == 0
        assert lease.yield_parked_commitment() == 0, "nothing left to hand back"

        other = await _lease(queue, tokens = 3092, budget = 8192)
        assert queue.committed_now() == 3092, "somebody else can take the room now"
        assert lease.recost(3847) is True, "the parked chat's next round is charged again"
        assert queue.committed_now() == 3092 + 3847
        lease.release()
        other.release()
        assert queue.committed_now() == 0


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
            "idle caches are only recycled by llama.cpp from the KV-full retry, i.e. "
            "after a decode has already failed, so they are occupancy to us"
        )
        assert occupancy["idle_tokens"] == 25592
        assert [slot for slot, _ in occupancy["idle"]] == [0, 2], "largest idle first"

    def test_generated_tokens_are_added_to_the_cache_field_and_only_to_it(self):
        # `n_prompt_tokens` ALREADY includes them: measured over 128 samples,
        # `n_prompt_tokens - n_decoded` is constant within a request to within 3 tokens,
        # and adding them again scored 28238 in a 16384-cell cache.
        assert (
            read_slot_occupancy(
                lambda: [
                    {
                        "id": 0,
                        "is_processing": True,
                        "n_prompt_tokens": 12632,
                        "next_token": [{"n_decoded": 6323}],
                    }
                ]
            )["resident"]
            == 12632
        )
        assert (
            read_slot_occupancy(
                lambda: [
                    {
                        "id": 0,
                        "is_processing": True,
                        "n_prompt_tokens_cache": 12632,
                        "next_token": [{"n_decoded": 6323}],
                    }
                ]
            )["resident"]
            == 18955
        ), "n_prompt_tokens_cache is the prompt only, so generation is still missing"

    def test_a_finished_slot_is_not_counted_twice(self):
        occupancy = read_slot_occupancy(
            lambda: [
                {
                    "id": 0,
                    "is_processing": False,
                    "n_prompt_tokens_cache": 9000,
                    "next_token": [{"n_decoded": 4000}],
                }
            ]
        )
        assert occupancy["resident"] == 9000, (
            "its cache already holds the whole sequence it produced, so the generated "
            "half must not be added a second time"
        )
        assert occupancy["idle_tokens"] == 9000

    @pytest.mark.parametrize("shape", [[{"n_decoded": 500}], {"n_decoded": 500}])
    def test_the_decoded_count_is_read_from_either_shape(self, shape):
        slots = [
            {"id": 0, "is_processing": True, "n_prompt_tokens_cache": 1000, "next_token": shape}
        ]
        assert read_slot_occupancy(lambda: slots)["resident"] == 1500

    @pytest.mark.parametrize("shape", [None, [], "nonsense", {"n_decoded": "abc"}, {}])
    def test_a_missing_or_malformed_next_token_reads_as_zero(self, shape):
        slots = [{"id": 0, "is_processing": True, "n_prompt_tokens": 1000, "next_token": shape}]
        assert read_slot_occupancy(lambda: slots)["resident"] == 1000

    def test_an_unreadable_endpoint_is_not_an_empty_cache(self):
        assert read_slot_occupancy(lambda: None) is None
        assert read_slot_occupancy(lambda: []) is None

    def test_a_parked_sequence_lives_in_host_ram_and_holds_no_cells(self):
        occupancy = read_slot_occupancy(
            lambda: [
                {"id": 0, "is_processing": True, "is_preempted": False, "n_prompt_tokens": 3000},
                {"id": 1, "is_processing": True, "is_preempted": True, "n_prompt_tokens": 4000},
                {
                    "id": 2,
                    "is_processing": False,
                    "is_preempted": False,
                    "n_prompt_tokens_cache": 500,
                },
            ]
        )
        assert occupancy["resident"] == 3500
        assert occupancy["idle_tokens"] == 500


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
        assert controller.committed_tokens() == 4097, (
            "the ledger is the larger of the two opinions about the same cells"
        )
        controller.note_tokens("a", 1400)
        assert controller.committed_tokens() == 1400, (
            "a measured chat must be counted once, not once per source"
        )

    def test_a_lagging_reading_cannot_shrink_a_measured_chat(self):
        controller = _controller(key = "resident-lag")
        controller.register("a", tokens = 9000, signal = PreemptSignal())
        controller.observe("a", 0)
        controller.note_resident(12)  # mid prefill, almost nothing visible yet
        assert controller.committed_tokens() == 9000

    def test_a_reading_above_the_cache_is_clamped_to_it(self):
        controller = _controller(key = "clamp", slots = 4)
        controller.register("a", tokens = 100, signal = PreemptSignal())
        controller.observe("a", 0)
        controller.note_resident(21304, 0)
        assert controller.committed_tokens() == 16384

    def test_idle_residue_does_not_stand_between_a_waiter_and_its_resume(self):
        controller = _controller(key = "idle-deadlock", slots = 4)
        controller.register("waiter", tokens = 5000, signal = PreemptSignal())
        controller.set_state("waiter", ParticipantState.PAUSED)
        controller.note_resident(21304, 21304)
        assert controller.room_for("waiter", 5000) is True, (
            "a cache holding nothing but reclaimable residue must not block a resume"
        )
        # 16000 live cells, minus this waiter's own 5000, still leaves 11000 that a 5000
        # token resume cannot fit beside.
        controller.note_resident(16000, 0)
        assert controller.room_for("waiter", 5000) is False

    def test_a_resume_is_refused_against_a_cache_only_slots_can_see(self):
        controller = _controller(key = "resident-room")
        controller.register("a", tokens = 2000, signal = PreemptSignal())
        controller.note_resident(16383)
        assert controller.room_for("a", 4000) is False
        controller.note_resident(3000)
        assert controller.room_for("a", 4000) is True

    def test_the_residency_probe_is_optional_and_can_never_raise(self):
        controller = _controller(key = "probe")
        calls = []
        controller.set_residency_probe(lambda: calls.append(1))
        controller.refresh_residency()
        assert calls == [1]

        def boom():
            raise RuntimeError("slots endpoint down")

        controller.set_residency_probe(boom)
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

        def _erase(slot_id):
            erased.append(slot_id)
            return 3000

        freed = reclaim_idle_slots(occupancy, _erase, needed = 3000)
        assert freed == 3000 and erased == [0]
        assert freed < occupancy["idle_tokens"], (
            "which is exactly the condition the caller has to check before telling the "
            "ledger that every parked holder lost its cells"
        )

    def test_a_failing_erase_does_not_take_the_generation_with_it(self):
        occupancy = read_slot_occupancy(
            lambda: [{"id": 0, "is_processing": False, "n_prompt_tokens_cache": 900}]
        )

        def _boom(_slot_id):
            raise RuntimeError("endpoint disabled")

        assert reclaim_idle_slots(occupancy, _boom, needed = 500) == 0
        assert reclaim_idle_slots(occupancy, lambda _i: 900, needed = 0) == 0


class TestTheSlotProbesAreAuthenticated:
    """``/slots`` is not a public endpoint."""

    class _Response:
        status = 200

        def __init__(self, body):
            self._body = body

        def read(self):
            return self._body

        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return False

    def _capture(self, monkeypatch, body):
        seen: list[urllib.request.Request] = []

        def _urlopen(request, timeout = None):
            # Both helpers must send a Request, not a bare URL, or there is nowhere to
            # put the header at all.
            assert isinstance(request, urllib.request.Request), request
            seen.append(request)
            return self._Response(body)

        monkeypatch.setattr(urllib.request, "urlopen", _urlopen)
        return seen

    def test_the_read_carries_the_key_or_nothing_at_all(self, monkeypatch):
        seen = self._capture(monkeypatch, json.dumps([]).encode())
        fetch_llama_slots("http://127.0.0.1:8080", headers = {"Authorization": "Bearer secret"})
        assert seen[0].get_header("Authorization") == "Bearer secret"
        fetch_llama_slots("http://127.0.0.1:8080")
        assert seen[1].get_header("Authorization") is None

    def test_the_erase_carries_the_key_and_posts(self, monkeypatch):
        seen = self._capture(monkeypatch, json.dumps({"n_erased": 4096}).encode())
        freed = erase_llama_slot(
            "http://127.0.0.1:8080", 2, headers = {"Authorization": "Bearer secret"}
        )
        assert freed == 4096
        assert seen[0].get_header("Authorization") == "Bearer secret"
        assert seen[0].get_method() == "POST"


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
        assert controller.try_grant_resume("b", 9000) is False, (
            "the second waiter must be refused: the first has already booked the room"
        )
        controller.note_resume_failed("a")
        assert controller.try_grant_resume("b", 9000) is True, (
            "room booked by a resume that never happened must not stay booked"
        )

    def test_no_room_while_the_others_still_hold_it(self):
        """44 preemptions across four chats, one producing 611 characters in 374 seconds."""
        controller = _controller(key = "thrash")
        for index in range(2):
            controller.register(f"c{index}", tokens = 7000, signal = PreemptSignal())
        assert controller.room_for("c0", 12000) is False, (
            "a resume was permitted while the cache was already over its watermark"
        )
        controller.set_state("c1", ParticipantState.PAUSED)
        assert controller.room_for("c0", 12000) is True

    def test_a_generation_does_not_count_against_its_own_resume(self):
        controller = _controller(key = "thrash-solo")
        controller.register("solo", tokens = 9000, signal = PreemptSignal())
        assert controller.room_for("solo", 9000) is True

    def test_the_gate_is_off_when_preemption_is(self, monkeypatch):
        monkeypatch.setenv("UNSLOTH_LLAMA_ADMISSION_PREEMPT", "0")
        controller = _controller(key = "thrash-off")
        for index in range(4):
            controller.register(f"c{index}", tokens = 9000, signal = PreemptSignal())
        assert controller.room_for("c0", 9000) is True, (
            "with preemption off nothing should wait on its watermark"
        )


class TestAChatThatOutgrewTheSharedCeiling:
    """No eviction can admit it, so waiting for room is a deadlock, not a delay."""

    def _c(self):
        return _controller(key = "solo-test", draft_tokens = 2, slots = 4)

    def test_a_want_past_the_shared_ceiling_is_recognised_and_fits_alone(self):
        controller = self._c()
        ceiling = _ceiling(controller)
        assert not controller.outgrew_the_shared_ceiling(ceiling)
        assert controller.outgrew_the_shared_ceiling(ceiling + 1)
        assert not controller.cannot_ever_fit(ceiling + 1), (
            "it fits with the cache to itself, so ending the turn would be premature"
        )
        assert controller.room_for("solo", ceiling + 500), "alone, it fits"
        _register(controller, "other", 3000)
        assert not controller.room_for("solo", ceiling + 500), "beside anyone, it does not"

    def test_a_want_past_the_cache_itself_can_never_fit(self):
        controller = self._c()
        assert controller.cannot_ever_fit(16384)
        assert not controller.cannot_ever_fit(10000)

    def test_the_solo_ceiling_beats_the_shared_one_by_more_than_a_rounding(self):
        controller = self._c()
        snapshot = controller.snapshot()
        shared = snapshot.budget - snapshot.buffer
        solo = next(
            w for w in range(snapshot.budget, shared, -1) if not controller.cannot_ever_fit(w)
        )
        # Most of the reaction headroom comes back, a lone chat having nobody to react to.
        assert solo - shared >= snapshot.buffer // 2, (
            f"solo ceiling {solo} barely clears the shared {shared} against a buffer of "
            f"{snapshot.buffer}; the reaction headroom is still charged to a chat that "
            f"has nobody to react to"
        )

    def test_none_of_this_applies_when_preemption_is_off(self):
        controller = self._c()
        controller.configure(budget = 0, kv_unified = False)
        assert not controller.outgrew_the_shared_ceiling(999999)
        assert not controller.cannot_ever_fit(999999)


class TestTheProgressSignature:
    """"no progress for 90.0s" about three chats decoding at full rate."""

    def _pinned(self, key):
        controller = _controller(key = key)
        controller.register("holder", tokens = 1000, signal = PreemptSignal())
        controller.note_tokens("holder", 1000)
        # Far above the ledger's own sum, which pins `committed` at a number the decoders
        # do not move, and past the ceiling, which keeps a waiter refused.
        controller.note_resident(16384)
        return controller

    def test_a_generated_token_moves_it_though_committed_does_not(self):
        controller = self._pinned("pinned")
        before = controller.progress_signature()
        controller.observe("holder", 512)
        after = controller.progress_signature()
        assert after[0] == before[0] == 16384, "committed must be pinned for this test"
        assert after[1] == before[1], "and the same holder must still hold"
        assert after != before and after[2] > before[2], "the token total is what moves"

    def test_a_resumed_attempt_restarting_its_count_is_not_read_as_lost_tokens(self):
        controller = self._pinned("restarts")
        controller.observe("holder", 512)
        mid = controller.progress_signature()[2]
        controller.observe("holder", 32)  # a fresh attempt, first report
        assert controller.progress_signature()[2] > mid

    def test_a_tool_call_and_a_round_boundary_both_count_as_progress(self):
        controller = self._pinned("tools")
        before = controller.progress_signature()
        assert controller.note_state("holder", ParticipantState.TOOLS_RUNNING) is True
        after = controller.progress_signature()
        assert after[3] == before[3] + 1 and after != before
        tokens_before = after[2]
        controller.note_tokens("holder", 1400)
        assert controller.progress_signature()[2] > tokens_before


class _Lease:
    is_released = False
    tokens = 0

    async def resume_async(self, want, **kwargs):
        return True


class TestTheResumeWait:
    """A 33-minute hang with nothing decoding, observed 2026-09-01."""

    @staticmethod
    def _waiting_policy(controller, gen_id, tokens, *, paused = False):
        loop = asyncio.new_event_loop()
        threading.Thread(target = loop.run_forever, daemon = True).start()
        controller.register(gen_id, lease = _Lease(), tokens = tokens, signal = PreemptSignal())
        if paused:
            controller.set_state(gen_id, ParticipantState.PAUSED)
        return ControllerPreemptionPolicy(controller, gen_id, PreemptSignal(), loop = loop), loop

    @staticmethod
    def _shutdown(loop):
        loop.call_soon_threadsafe(loop.stop)

    def test_an_unstated_timeout_still_returns_promptly_rather_than_blocking(self):
        for gen_id in ("known", "missing"):
            controller = _controller(key = f"prompt-{gen_id}")
            if gen_id == "known":
                controller.register(gen_id, tokens = 100, signal = PreemptSignal())
            policy = ControllerPreemptionPolicy(controller, gen_id, PreemptSignal(), loop = None)
            started = time.monotonic()
            assert policy.await_resume() in (True, False)
            assert time.monotonic() - started < 5, f"{gen_id}: await_resume blocked"
        assert DEFAULT_RESUME_WAIT_TIMEOUT_S > 0

    def test_giving_up_is_reported_as_false_not_raised(self):
        controller = _controller(key = "gave-up")
        policy = ControllerPreemptionPolicy(controller, "missing", PreemptSignal(), loop = None)
        assert policy.await_resume(timeout = 0.01) is False

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

    @pytest.mark.parametrize("direction", ["draining", "growing"])
    def test_progress_buys_more_patience_than_the_wall_clock_allows(self, direction):
        controller = _controller(key = f"progressing-{direction}")
        # Sized so the waiter genuinely CANNOT fit: 6000 + 4000 fits, so the waiter would
        # be granted room on its first look and the test would measure nothing.
        start = 14000 if direction == "draining" else 13000
        controller.register("holder", tokens = start, signal = PreemptSignal())
        controller.note_tokens("holder", start)
        stop = threading.Event()

        def move():
            held = start
            while not stop.is_set() and (held > 1000 if direction == "draining" else held < 15500):
                time.sleep(0.05)
                held += -200 if direction == "draining" else 100
                controller.note_tokens("holder", held)

        worker = threading.Thread(target = move, daemon = True)
        worker.start()
        policy, loop = self._waiting_policy(controller, "waiter", 4000)
        try:
            started = time.monotonic()
            policy.await_resume(timeout = 0.3)
            elapsed = time.monotonic() - started
        finally:
            stop.set()
            worker.join(timeout = 5)
            self._shutdown(loop)
        assert elapsed > 0.4, (
            f"gave up after {elapsed}s while the server was working throughout; a stall "
            "deadline must reset on progress, and growth is evidence of work"
        )

    def test_it_waits_while_anything_moves_and_takes_the_room_when_it_is_real(self):
        timeout = 0.4
        controller = self_pinned = _controller(key = "still-decoding")
        controller.register("holder", tokens = 1000, signal = PreemptSignal())
        controller.note_tokens("holder", 1000)
        controller.note_resident(16384)
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
        assert self_pinned.committed_tokens() <= 16384
        assert elapsed > timeout * 2, (
            f"gave up after {elapsed}s of a backend generating tokens throughout; a chat "
            "queued behind live answers waits, however long that takes"
        )
        assert resumed is True

    def test_the_hard_backstop_outlasts_the_slowest_answer_measured(self):
        bound = DEFAULT_RESUME_WAIT_TIMEOUT_S * MAX_RESUME_WAIT_MULTIPLE
        assert bound >= 2 * (8192 / 2.3), (
            f"the backstop is {bound}s, shorter than two answers at the rate measured"
        )


class TestTheRecostWaitIsPatientWithABusyLeader:
    """A round waiting for room behind a decoding leader is queued, not stuck."""

    @staticmethod
    def _two(queue):
        a = queue.reserve(
            capacity = 4, config = LlamaAdmissionConfig(), tokens = 6000, budget = 8192
        ).lease_nowait()
        b = queue.reserve(
            capacity = 4, config = LlamaAdmissionConfig(), tokens = 1000, budget = 8192
        ).lease_nowait()
        assert a is not None and b is not None
        return a, b

    @pytest.mark.asyncio
    async def test_a_moving_progress_signature_keeps_the_wait_alive(self):
        queue = llama_admission.get_llama_admission_queue("http://patient.test")
        leader, waiter = self._two(queue)
        ticks = {"n": 0}

        def progress():
            ticks["n"] += 1  # somebody is decoding: a new value every read
            return ticks["n"]

        outcome = {}

        def wait():
            outcome["granted"] = waiter.recost_waiting(
                4000, timeout_s = 0.3, poll_s = 0.02, progress = progress, gen_id = "waiter"
            )

        thread = threading.Thread(target = wait)
        thread.start()
        await asyncio.sleep(0.8)  # well past timeout_s
        assert thread.is_alive(), "the wait gave up while the pool was visibly moving"
        leader.release()
        thread.join(timeout = 5)
        assert outcome["granted"] is True
        assert queue.committed_now() == 4000
        waiter.release()

    @pytest.mark.asyncio
    async def test_a_frozen_pool_still_gives_up_on_schedule(self):
        queue = llama_admission.get_llama_admission_queue("http://frozen.test")
        leader, waiter = self._two(queue)
        started = time.monotonic()
        granted = waiter.recost_waiting(
            4000, timeout_s = 0.3, poll_s = 0.02, progress = lambda: "same", gen_id = "waiter"
        )
        elapsed = time.monotonic() - started
        assert granted is False
        assert 0.25 <= elapsed < 2.0, elapsed
        assert queue.committed_now() == 6000 + 1000, "back at the old figure, as before"
        leader.release()
        waiter.release()

    @pytest.mark.asyncio
    async def test_the_hard_deadline_bounds_a_pool_that_moves_forever(self):
        queue = llama_admission.get_llama_admission_queue("http://forever.test")
        leader, waiter = self._two(queue)
        ticks = {"n": 0}

        def progress():
            ticks["n"] += 1
            return ticks["n"]

        started = time.monotonic()
        granted = waiter.recost_waiting(
            4000, timeout_s = 0.05, poll_s = 0.01, progress = progress, gen_id = "waiter"
        )
        elapsed = time.monotonic() - started
        assert granted is False
        # 0.05 s * the repark multiple (20) = 1 s, give or take a poll.
        assert 0.9 <= elapsed < 3.0, elapsed
        leader.release()
        waiter.release()

    @pytest.mark.asyncio
    async def test_no_progress_callable_behaves_as_before(self):
        queue = llama_admission.get_llama_admission_queue("http://plain.test")
        leader, waiter = self._two(queue)
        started = time.monotonic()
        assert waiter.recost_waiting(4000, timeout_s = 0.2, poll_s = 0.02) is False
        assert time.monotonic() - started < 2.0
        leader.release()
        waiter.release()


# ========================================================== the commitment that comes back


class TestTheCommitmentActuallyComesBack:
    """`park` keeps its tokens because the task is alive. Preemption ends it."""

    @pytest.mark.asyncio
    async def test_preempt_hands_back_both_the_slot_and_the_tokens_once(self):
        queue = LlamaAdmissionQueue("k")
        lease = await _lease(queue, tokens = 4000)
        assert lease.park() is True, "park budget should be available on a 1-slot queue"
        assert queue.snapshot().committed == 4000, "park must NOT hand the tokens back"
        assert lease.preempt() is True
        after = queue.snapshot()
        assert after.committed == 0 and after.active == 0
        assert lease.is_preempted is True
        assert lease.preempt() is False, "idempotent"

    @pytest.mark.asyncio
    async def test_a_released_lease_cannot_be_preempted_and_never_double_refunds(self):
        queue = LlamaAdmissionQueue("k")
        released = await _lease(queue, tokens = 4000)
        released.release()
        assert released.preempt() is False
        holder = await _lease(queue, tokens = 4000)
        victim = await _lease(queue, tokens = 4000)
        assert victim.preempt() is True
        victim.release()
        assert queue.snapshot().committed == 4000, "only the untouched holder should remain"
        assert holder.is_released is False

    @pytest.mark.asyncio
    async def test_the_freed_room_admits_a_waiter(self):
        queue = LlamaAdmissionQueue("k")
        big = await _lease(queue, tokens = 12000, capacity = 2)
        reservation = queue.reserve(
            capacity = 2, config = LlamaAdmissionConfig(), tokens = 8000, budget = 16384
        )
        assert reservation.lease_nowait() is None, "should not fit beside the big holder"
        assert big.preempt() is True
        await asyncio.sleep(0)
        assert reservation.lease_nowait() is not None


class TestResumeDoesNotChargeTwice:
    @pytest.mark.asyncio
    async def test_resume_restores_exactly_what_it_took_or_a_larger_figure(self):
        queue = LlamaAdmissionQueue("k")
        lease = await _lease(queue, tokens = 4000)
        lease.preempt()
        assert await lease.resume_async(4000) is True
        assert queue.snapshot().committed == 4000, "resume must commit once, not twice"
        assert queue.snapshot().active == 1 and lease.is_preempted is False
        lease.preempt()
        assert await lease.resume_async(5000) is True
        assert queue.snapshot().committed == 5000

    @pytest.mark.asyncio
    async def test_a_lease_that_was_never_preempted_resumes_to_a_no_op(self):
        queue = LlamaAdmissionQueue("k")
        lease = await _lease(queue, tokens = 4000)
        assert await lease.resume_async(9999) is True
        assert queue.snapshot().committed == 4000, "an unpreempted lease must not be re-costed"

    @pytest.mark.asyncio
    async def test_a_release_during_the_wait_strands_nothing(self):
        queue = LlamaAdmissionQueue("k")
        lease = await _lease(queue, tokens = 4000)
        lease.preempt()
        lease.release()
        assert await lease.resume_async(4000) is False
        assert queue.snapshot().committed == 0 and queue.snapshot().active == 0

    @pytest.mark.asyncio
    async def test_a_pool_that_never_moves_times_out_rather_than_freezing_the_queue(self):
        queue = LlamaAdmissionQueue("k")
        await _lease(queue, tokens = 15000, capacity = 2)
        victim = await _lease(queue, tokens = 1000, capacity = 2)
        victim.preempt()
        assert await victim.resume_async(15000, poll_s = 0.001, timeout_s = 0.05) is False
        assert queue.snapshot().committed == 15000, "a failed resume must commit nothing"

    @pytest.mark.asyncio
    async def test_a_draining_pool_buys_more_patience_than_the_wall_clock(self):
        queue = LlamaAdmissionQueue("k")
        blocker = await _lease(queue, tokens = 15000, capacity = 2)
        victim = await _lease(queue, tokens = 1000, capacity = 2)
        # Parked, so the resume takes the commitment-only path, which is the loop the live
        # give-up came out of.
        assert victim.park() is True
        victim.preempt()

        # Budget 16384, blocker holding 15000, victim wanting 5000. Draining 1000 every
        # 20ms reaches the point it fits at ~80ms, past the 50ms flat deadline, while
        # never pausing longer than 50ms: a flat deadline fires, a stall deadline never does.
        async def drain_slowly():
            for _ in range(6):
                await asyncio.sleep(0.02)
                blocker.recost(max(0, blocker.tokens - 1000))

        drainer = asyncio.ensure_future(drain_slowly())
        try:
            resumed = await victim.resume_async(5000, poll_s = 0.005, timeout_s = 0.05)
        finally:
            await drainer
        assert resumed is True, (
            "gave up while the pool was draining; the deadline must reset on progress"
        )

    @pytest.mark.asyncio
    async def test_a_parked_lease_never_takes_a_second_slot(self):
        queue = LlamaAdmissionQueue("k")
        lease = await _lease(queue, tokens = 4000)
        assert lease.park() is True
        assert lease.preempt() is True
        assert await lease.resume_async(4000, poll_s = 0.001, timeout_s = 1.0) is True
        assert queue.snapshot().active == 0, "the park still owns the slot"
        assert queue.snapshot().committed == 4000, "only the commitment came back"

    @pytest.mark.asyncio
    async def test_a_cancelled_resume_gives_up_rather_than_spinning(self):
        queue = LlamaAdmissionQueue("k")
        await _lease(queue, tokens = 16000, capacity = 2)
        victim = await _lease(queue, tokens = 300, capacity = 2)
        victim.preempt()
        cancel = threading.Event()
        cancel.set()
        assert await victim.resume_async(9000, cancel_event = cancel) is False


# ============================================================ the barrier and the registry


class TestTheReclaimBarrier:
    def test_it_waits_until_the_count_falls(self):
        readings = iter(
            [{"requests_processing": 3.0}, {"requests_processing": 2.0}, {"requests_processing": 1.0}]
        )
        assert (
            wait_for_reclaim(lambda: next(readings), target_processing = 1, sleep = lambda _s: None)
            is True
        )

    def test_it_returns_at_once_when_already_clear(self):
        calls = []

        def _scrape():
            calls.append(1)
            return {"requests_processing": 0.0}

        assert wait_for_reclaim(_scrape, target_processing = 1, sleep = lambda _s: None) is True
        assert len(calls) == 1

    def test_a_server_without_the_counter_is_not_a_confirmation(self):
        assert (
            wait_for_reclaim(
                lambda: {"n_decode_total": 5.0}, target_processing = 0, sleep = lambda _s: None
            )
            is False
        )

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

    def test_it_never_claims_to_know_which_generation_finished(self):
        import inspect

        signature = inspect.signature(wait_for_reclaim)
        assert "target_processing" in signature.parameters
        assert "gen_id" not in signature.parameters


class TestTheRegistry:
    def test_one_controller_per_key(self):
        assert get_preemption_controller("a") is get_preemption_controller("a")
        assert get_preemption_controller("a") is not get_preemption_controller("b")

    def test_an_empty_controller_is_evicted_when_a_new_load_arrives(self):
        first = get_preemption_controller("port-1")
        get_preemption_controller("port-2")
        assert get_preemption_controller("port-1") is not first, "an idle controller retires"

    def test_a_controller_with_work_in_flight_is_kept(self):
        busy = get_preemption_controller("port-1")
        busy.register("live", tokens = 100)
        get_preemption_controller("port-2")
        assert get_preemption_controller("port-1") is busy, "an in-flight controller survives"
