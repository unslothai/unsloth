# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Pausing a chat instead of killing four.

On 2026-09-01 four tool chats on `-c 16384 --parallel 4 --kv-unified` were each admitted
at their share, all generated into the one shared pool, and llama-server errored EVERY
processing slot at once. These cover the half that decides who stops: the commitment
really coming back, the epoch holding still, the starved chat being promoted, and a
resume that does not charge twice.
"""

import asyncio

import pytest

from core.inference.llama_admission import (
    LlamaAdmissionConfig,
    LlamaAdmissionQueue,
    reset_llama_admission_queues,
)
from core.inference.llama_preemption import (
    DEFAULT_PREEMPT_BUFFER_MIN_TOKENS,
    DEFAULT_PREEMPT_BUFFER_RATIO,
    PROMOTE_AFTER_CONSECUTIVE_PREEMPTIONS,
    ParticipantState,
    PreemptionController,
    get_preemption_controller,
    preemption_buffer_tokens,
    reset_preemption_controllers,
    wait_for_reclaim,
)


@pytest.fixture(autouse = True)
def _clean_registries():
    reset_llama_admission_queues()
    reset_preemption_controllers()
    yield
    reset_llama_admission_queues()
    reset_preemption_controllers()


def _controller(budget = 16384, kv_unified = True):
    controller = PreemptionController("test")
    controller.configure(budget = budget, kv_unified = kv_unified)
    return controller


def _register(controller, gen_id, tokens, state = ParticipantState.DECODING):
    return controller.register(gen_id, tokens = tokens, state = state)


async def _lease(queue, *, tokens, capacity = 4, budget = 16384):
    reservation = queue.reserve(
        capacity = capacity,
        config = LlamaAdmissionConfig(),
        tokens = tokens,
        budget = budget,
    )
    lease = reservation.lease_nowait()
    assert lease is not None, "expected an immediate admission"
    return lease


class TestTheCommitmentActuallyComesBack:
    """`park` keeps its tokens because the task is alive. Preemption ends it."""

    @pytest.mark.asyncio
    async def test_preempt_hands_back_both_the_slot_and_the_tokens(self):
        queue = LlamaAdmissionQueue("k")
        lease = await _lease(queue, tokens = 4000)
        assert queue.snapshot().committed == 4000
        assert queue.snapshot().active == 1

        assert lease.preempt() is True
        after = queue.snapshot()
        assert after.committed == 0, "the KV commitment must come back, unlike park()"
        assert after.active == 0, "the slot must come back too"
        assert lease.is_preempted is True

    @pytest.mark.asyncio
    async def test_park_still_keeps_its_tokens(self):
        """The contrast that makes preempt() a different method and not a flag.

        A parked task is still alive at llama-server, so its cells are still resident and
        its commitment must stay. Preemption ends the task, so the commitment goes.
        """
        queue = LlamaAdmissionQueue("k")
        lease = await _lease(queue, tokens = 4000)
        assert lease.park() is True, "park budget should be available on a 1-slot queue"
        assert queue.snapshot().committed == 4000, "park must NOT hand the tokens back"
        assert lease.preempt() is True
        assert queue.snapshot().committed == 0, "preempt must"

    @pytest.mark.asyncio
    async def test_preempt_is_idempotent(self):
        queue = LlamaAdmissionQueue("k")
        lease = await _lease(queue, tokens = 4000)
        assert lease.preempt() is True
        assert lease.preempt() is False
        assert queue.snapshot().committed == 0

    @pytest.mark.asyncio
    async def test_a_released_lease_cannot_be_preempted(self):
        queue = LlamaAdmissionQueue("k")
        lease = await _lease(queue, tokens = 4000)
        lease.release()
        assert lease.preempt() is False
        assert queue.snapshot().committed == 0

    @pytest.mark.asyncio
    async def test_releasing_after_a_preempt_does_not_double_refund(self):
        """Both give room back; together they must not invent any."""
        queue = LlamaAdmissionQueue("k")
        holder = await _lease(queue, tokens = 4000)
        victim = await _lease(queue, tokens = 4000)
        assert victim.preempt() is True
        victim.release()
        assert queue.snapshot().committed == 4000, "only the untouched holder should remain"

    @pytest.mark.asyncio
    async def test_the_freed_room_admits_a_waiter(self):
        """The point of preempting: someone else gets in."""
        queue = LlamaAdmissionQueue("k")
        big = await _lease(queue, tokens = 12000, capacity = 2)
        reservation = queue.reserve(
            capacity = 2,
            config = LlamaAdmissionConfig(),
            tokens = 8000,
            budget = 16384,
        )
        assert reservation.lease_nowait() is None, "should not fit beside the big holder"
        assert big.preempt() is True
        await asyncio.sleep(0)
        assert reservation.lease_nowait() is not None, "the freed room should admit the waiter"


class TestResumeDoesNotChargeTwice:
    @pytest.mark.asyncio
    async def test_resume_restores_exactly_what_it_took(self):
        queue = LlamaAdmissionQueue("k")
        lease = await _lease(queue, tokens = 4000)
        lease.preempt()
        assert await lease.resume_async(4000) is True
        assert queue.snapshot().committed == 4000, "resume must commit once, not twice"
        assert queue.snapshot().active == 1
        assert lease.is_preempted is False

    @pytest.mark.asyncio
    async def test_resume_may_take_a_larger_figure_than_it_gave_back(self):
        """A resumed run carries the partial it already generated."""
        queue = LlamaAdmissionQueue("k")
        lease = await _lease(queue, tokens = 4000)
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
        assert queue.snapshot().committed == 0, "a resumed-then-released lease must strand nothing"
        assert queue.snapshot().active == 0

    @pytest.mark.asyncio
    async def test_a_resume_that_cannot_fit_gives_up_instead_of_freezing_the_queue(self):
        """Unbounded, this holds room nobody else can plan against. `recost_waiting` is
        bounded for the same reason."""
        queue = LlamaAdmissionQueue("k")
        blocker = await _lease(queue, tokens = 15000, capacity = 2)
        victim = await _lease(queue, tokens = 1000, capacity = 2)
        victim.preempt()
        assert blocker is not None
        resumed = await victim.resume_async(15000, poll_s = 0.001, timeout_s = 0.05)
        assert resumed is False
        assert queue.snapshot().committed == 15000, "a failed resume must commit nothing"

    @pytest.mark.asyncio
    async def test_a_parked_lease_never_takes_a_second_slot(self):
        """Its slot belongs to the park machinery and comes back through unpark_async;
        a ticket here would put one lease in two slots."""
        queue = LlamaAdmissionQueue("k")
        lease = await _lease(queue, tokens = 4000)
        assert lease.park() is True
        assert lease.preempt() is True
        assert await lease.resume_async(4000, poll_s = 0.001, timeout_s = 1.0) is True
        assert queue.snapshot().active == 0, "the park still owns the slot"
        assert queue.snapshot().committed == 4000, "only the commitment came back"

    @pytest.mark.asyncio
    async def test_a_cancelled_resume_gives_up_rather_than_spinning(self):
        import threading

        queue = LlamaAdmissionQueue("k")
        blocker = await _lease(queue, tokens = 16000, capacity = 2)
        victim = await _lease(queue, tokens = 300, capacity = 2)
        victim.preempt()
        assert blocker is not None
        cancel = threading.Event()
        cancel.set()
        assert await victim.resume_async(9000, cancel_event = cancel) is False


class TestTheBufferArithmetic:
    def test_the_buffer_is_a_ratio_of_the_cache_with_a_floor(self):
        """Pinned as a ratio, not as 820.

        The literal was the five per cent this started at. It was raised after
        measurement: at five per cent the watermark sat at 95% of the cache and
        llama-server still entered its shrinking-batch retry ten times in one run, which
        is the path the speculative sub-batch error comes from. It is tunable now, so a
        test that hardcodes the figure just has to be edited again next time.
        """
        assert preemption_buffer_tokens(16384) == pytest.approx(
            16384 * DEFAULT_PREEMPT_BUFFER_RATIO, rel = 0.01
        )
        assert preemption_buffer_tokens(2048) >= DEFAULT_PREEMPT_BUFFER_MIN_TOKENS
        assert preemption_buffer_tokens(0) == 0
        # Still never the whole cache, whatever the ratio is set to.
        assert preemption_buffer_tokens(16384) < 16384 // 2 + 1

    def test_nothing_is_preempted_while_it_fits(self):
        controller = _controller(budget = 16384)
        _register(controller, "a", 2000)
        _register(controller, "b", 2000)
        assert controller.plan_preemptions() == []

    def test_the_ceiling_is_the_budget_minus_the_buffer(self):
        controller = _controller(budget = 16384)
        ceiling = 16384 - preemption_buffer_tokens(16384)
        _register(controller, "a", ceiling)
        assert controller.plan_preemptions() == [], "exactly at the ceiling still fits"
        _register(controller, "b", 1)
        assert controller.plan_preemptions(), "one token past it must not"

    def test_room_asked_for_in_advance_counts(self):
        controller = _controller(budget = 16384)
        ceiling = 16384 - preemption_buffer_tokens(16384)
        # Sized from the ceiling rather than from the 15564 it happened to be at five
        # per cent, so raising the margin does not break the property being tested.
        _register(controller, "winner", ceiling - 2000)
        _register(controller, "other", 1500)
        assert controller.plan_preemptions() == [], "it fits under the ceiling"
        assert controller.plan_preemptions(needed = 1000), "a request for room must be counted"

    def test_a_lone_holder_is_not_preempted_for_a_newcomer(self):
        """There is no victim but itself, and an in-flight conversation beats one that
        has not started. The wait line already holds the newcomer, exactly as it does
        for a reparker."""
        controller = _controller(budget = 16384)
        _register(controller, "alone", 15000)
        assert controller.plan_preemptions(needed = 4000) == []


class TestWhoStops:
    def test_the_longest_chat_keeps_decoding(self):
        controller = _controller(budget = 16384)
        _register(controller, "small", 5000)
        _register(controller, "longest", 11000)
        victims = {p.gen_id for p in controller.plan_preemptions()}
        assert "longest" not in victims, "longest wins"
        assert "small" in victims

    def test_a_parked_chat_is_taken_before_a_decoding_one(self):
        """It holds KV and consumes no compute, so its room is the cheapest."""
        controller = _controller(budget = 16384)
        _register(controller, "winner", 9000)
        _register(controller, "decoding", 4000)
        _register(controller, "parked", 4000, state = ParticipantState.PARKED_ON_TOOL)
        victims = [p.gen_id for p in controller.plan_preemptions()]
        assert victims[0] == "parked"

    def test_a_chat_running_tools_is_never_preempted(self):
        controller = _controller(budget = 16384)
        _register(controller, "winner", 9000)
        _register(controller, "tools", 8000, state = ParticipantState.TOOLS_RUNNING)
        victims = {p.gen_id for p in controller.plan_preemptions()}
        assert "tools" not in victims, "nothing is decoding there, and it is the unsafe window"

    def test_only_as_many_as_needed_are_paused(self):
        """'Pause all but one' is the worst case, not the first move."""
        controller = _controller(budget = 16384)
        _register(controller, "winner", 8000)
        _register(controller, "big", 7000)
        _register(controller, "small_a", 400)
        _register(controller, "small_b", 400)
        victims = [p.gen_id for p in controller.plan_preemptions()]
        assert victims == ["big"], f"one victim should have been enough, got {victims}"

    def test_a_victim_is_marked_and_signalled_together(self):
        """Marked PREEMPTING, not PAUSED.

        This asserted PAUSED until a live run on 2026-09-01 showed why that is wrong:
        PAUSED stops counting the victim's KV, so the room was treated as free the
        instant a victim was chosen rather than when its stream actually stopped. Four
        chats were then admitted against a cache the controller believed was half empty,
        and the model ran out of context space exactly as it did before preemption
        existed. A victim holds its cells until the pause is confirmed.
        """
        controller = _controller(budget = 16384)
        _register(controller, "winner", 11000)
        victim = _register(controller, "victim", 5000)
        before = controller.committed_tokens()
        controller.plan_preemptions()
        assert victim.state == ParticipantState.PREEMPTING
        assert victim.preempt_event.is_set(), "the decision and the signal must not drift apart"
        assert controller.committed_tokens() == before, (
            "asking for a pause must not free room that is still occupied"
        )

    def test_a_queued_chat_is_not_a_victim(self):
        controller = _controller(budget = 16384)
        _register(controller, "winner", 15000)
        queued = _register(controller, "queued", 0, state = ParticipantState.QUEUED)
        _register(controller, "other", 2000)
        assert queued not in controller.plan_preemptions()


class TestTheEpochHoldsStill:
    def test_the_winner_does_not_change_when_a_victim_overtakes_it(self):
        """Without an epoch the two would trade places forever."""
        controller = _controller(budget = 16384)
        _register(controller, "a", 11000)
        b = _register(controller, "b", 5000)
        assert {p.gen_id for p in controller.plan_preemptions()} == {"b"}
        # b resumes and grows past a; a must still hold the epoch.
        controller.note_resumed("b")
        b.tokens = 14000
        assert {p.gen_id for p in controller.plan_preemptions()} == {"b"}
        assert controller.snapshot().winner == "a"

    def test_the_epoch_ends_when_the_winner_blocks_on_a_tool(self):
        controller = _controller(budget = 16384)
        _register(controller, "a", 11000)
        _register(controller, "b", 5000)
        controller.plan_preemptions()
        assert controller.snapshot().winner == "a"
        controller.set_state("a", ParticipantState.PARKED_ON_TOOL)
        assert controller.snapshot().winner is None, "blocking on a tool ends the epoch"

    def test_the_epoch_ends_when_the_winner_finishes(self):
        controller = _controller(budget = 16384)
        _register(controller, "a", 11000)
        _register(controller, "b", 5000)
        controller.plan_preemptions()
        controller.unregister("a")
        assert controller.snapshot().winner is None

    def test_a_new_winner_is_chosen_after_the_epoch_ends(self):
        controller = _controller(budget = 16384)
        _register(controller, "a", 11000)
        b = _register(controller, "b", 5000)
        controller.plan_preemptions()
        controller.unregister("a")
        controller.note_resumed("b")
        b.tokens = 15000
        _register(controller, "c", 2000)
        controller.plan_preemptions()
        assert controller.snapshot().winner == "b"


class TestStarvation:
    def _starve(self, controller):
        """Drive `starved` through three preemptions in a row.

        The two must actually overflow the ceiling (15564 of 16384), or nothing is
        preempted and the rule is never exercised.
        """
        _register(controller, "hog", 12000)
        starved = _register(controller, "starved", 4000)
        for _ in range(PROMOTE_AFTER_CONSECUTIVE_PREEMPTIONS):
            assert [p.gen_id for p in controller.plan_preemptions()] == ["starved"]
            controller.note_resumed("starved")
        return starved

    def test_three_consecutive_preemptions_promote_a_chat(self):
        controller = _controller(budget = 16384)
        starved = self._starve(controller)
        assert starved.promoted, "three preemptions in a row must promote it"
        # End the hog's epoch; the promoted chat now outranks longest-wins, even though
        # the newcomer is three times its size.
        controller.unregister("hog")
        _register(controller, "hog2", 12000)
        controller.plan_preemptions()
        assert controller.snapshot().winner == "starved"

    def test_being_crowned_clears_the_debt(self):
        controller = _controller(budget = 16384)
        starved = self._starve(controller)
        controller.unregister("hog")
        _register(controller, "hog2", 12000)
        controller.plan_preemptions()
        assert starved.consecutive_preemptions == 0, "the starvation is cured once it wins"

    def test_a_preemption_after_a_reprieve_is_not_consecutive(self):
        """The rule is three IN A ROW; winning in between clears the count."""
        controller = _controller(budget = 16384)
        starved = self._starve(controller)
        controller.unregister("hog")
        _register(controller, "hog2", 12000)
        controller.plan_preemptions()          # starved is crowned, debt cleared
        controller.set_state("starved", ParticipantState.PARKED_ON_TOOL)
        controller.note_resumed("starved")
        controller.plan_preemptions()
        assert starved.consecutive_preemptions <= 1


class TestTheSwitchesThatTurnItOff:
    def test_a_private_cache_per_slot_is_never_preempted(self):
        """Without --kv-unified each slot owns its own cells; nobody can overrun anyone."""
        controller = _controller(budget = 16384, kv_unified = False)
        _register(controller, "a", 15000)
        _register(controller, "b", 15000)
        assert controller.plan_preemptions() == []
        assert controller.active is False

    def test_an_unknown_budget_plans_nothing(self):
        controller = _controller(budget = 0)
        _register(controller, "a", 15000)
        _register(controller, "b", 15000)
        assert controller.plan_preemptions() == []

    def test_the_env_switch_disables_it(self, monkeypatch):
        controller = _controller(budget = 16384)
        _register(controller, "a", 15000)
        _register(controller, "b", 15000)
        assert controller.plan_preemptions(), "on by default"
        controller.note_resumed("b")
        monkeypatch.setenv("UNSLOTH_LLAMA_ADMISSION_PREEMPT", "0")
        assert controller.plan_preemptions() == []
        assert controller.active is False

    def test_it_is_not_gated_on_idle_slot_clearing(self):
        """The reactive purge is gated on kv_unified ALONE (server-context.cpp:1656), so
        gating here on --cache-idle-slots would disable preemption on Windows full GPU
        offload (#5692), which is exactly where the clamp is the only other protection."""
        import inspect

        import core.inference.llama_preemption as module

        for name, member in vars(module).items():
            if callable(member) and getattr(member, "__module__", None) == module.__name__:
                try:
                    body = inspect.getsource(member)
                except (OSError, TypeError):
                    continue
                code = "\n".join(
                    line for line in body.split("\n") if not line.strip().startswith("#")
                )
                assert "idle_slot_clearing_active" not in code, name


class TestTheReclaimBarrier:
    def test_it_waits_until_the_count_falls(self):
        readings = iter([
            {"requests_processing": 3.0},
            {"requests_processing": 2.0},
            {"requests_processing": 1.0},
        ])
        assert wait_for_reclaim(
            lambda: next(readings),
            target_processing = 1,
            sleep = lambda _s: None,
        ) is True

    def test_it_returns_at_once_when_already_clear(self):
        calls = []

        def _scrape():
            calls.append(1)
            return {"requests_processing": 0.0}

        assert wait_for_reclaim(_scrape, target_processing = 1, sleep = lambda _s: None) is True
        assert len(calls) == 1

    def test_a_server_without_the_counter_is_not_a_confirmation(self):
        assert wait_for_reclaim(
            lambda: {"n_decode_total": 5.0},
            target_processing = 0,
            sleep = lambda _s: None,
        ) is False

    def test_an_unreadable_metrics_endpoint_times_out_rather_than_blocking(self):
        clock = iter([0.0, 0.0, 1.0, 99.0])
        assert wait_for_reclaim(
            lambda: None,
            target_processing = 0,
            timeout_s = 1.0,
            sleep = lambda _s: None,
            monotonic = lambda: next(clock),
        ) is False

    def test_it_never_claims_to_know_which_generation_finished(self):
        """It is a barrier, not an attribution: the gauge cannot say who owns a slot."""
        import inspect

        from core.inference import llama_preemption

        signature = inspect.signature(llama_preemption.wait_for_reclaim)
        assert "target_processing" in signature.parameters
        assert "gen_id" not in signature.parameters


class TestTheRegistry:
    def test_one_controller_per_key(self):
        assert get_preemption_controller("a") is get_preemption_controller("a")
        assert get_preemption_controller("a") is not get_preemption_controller("b")

    def test_an_empty_controller_is_evicted_when_a_new_load_arrives(self):
        first = get_preemption_controller("port-1")
        get_preemption_controller("port-2")
        assert get_preemption_controller("port-1") is not first, "an idle controller should retire"

    def test_a_controller_with_work_in_flight_is_kept(self):
        busy = get_preemption_controller("port-1")
        busy.register("live", tokens = 100)
        get_preemption_controller("port-2")
        assert get_preemption_controller("port-1") is busy, "an in-flight controller must survive"
