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


def _ceiling(controller):
    """The live watermark. Tests size themselves against this rather than a literal.

    Hardcoded token counts broke every time the buffer changed, most recently when it
    stopped being a fraction of the cache and became a per-slot reserve: fourteen tests
    failed for one intended change, none of them because the behaviour was wrong.
    """
    snapshot = controller.snapshot()
    return snapshot.budget - snapshot.buffer


def _fill(controller, gen_id, fraction, state = ParticipantState.DECODING):
    """Register a participant holding `fraction` of the ceiling."""
    return _register(controller, gen_id, int(_ceiling(controller) * fraction), state = state)


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
    def test_the_buffer_is_per_slot_with_a_floor(self):
        """Pinned as a shape, not as a number.

        This has now been a literal 820, a ratio of 5%, a ratio of 15%, and a per-slot
        reserve. Each rewrite broke the tests that named the previous figure, so this
        asserts the properties that must hold under any of them.
        """
        # Per SLOT, not per cache. The reaction headroom the buffer buys is what can be
        # generated between a sweep and a victim's stream actually stopping, which scales
        # with how many chats decode at once and not with the size of the cache.
        four = preemption_buffer_tokens(16384, slots = 4)
        eight = preemption_buffer_tokens(16384, slots = 8)
        assert eight > four, "twice the slots generate twice as much during an eviction"
        assert preemption_buffer_tokens(16384, slots = 4) == preemption_buffer_tokens(
            65536, slots = 4
        ), "a bigger cache does not make an eviction slower"
        assert preemption_buffer_tokens(2048) >= DEFAULT_PREEMPT_BUFFER_MIN_TOKENS
        # And it must be a small share of a normal cache, or it serialises: at 15% of
        # 16384 the simulated makespan was 26890 steps against 239 at this size.
        assert four < 16384 * 0.08
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
    def test_the_newest_chat_stops_first(self):
        """vLLM V1's rule: evict the most recently arrived, so work done is work kept.

        This asserted the opposite until 2026-09-03, that the LONGEST chat keeps decoding
        and the small one stops, on the reasoning that the fewest victims free the most
        room. Simulated across nine load regimes at 60 seeds each, largest-first ranked
        5th of 7 policies (mean rank 4.25) and last on fairness, because the largest chat
        carries the most work to discard and the most tokens to replay on resume.
        Newest-first ranked best overall at 2.89 and best of all on completions.
        """
        controller = _controller(budget = 16384)
        _fill(controller, "older", 0.35)
        _fill(controller, "newer", 0.75)
        victims = {p.gen_id for p in controller.plan_preemptions()}
        assert "newer" in victims, "the most recently arrived chat should stop first"
        assert "older" not in victims

    def test_size_does_not_decide(self):
        """Registration order does, so a big early chat outranks a small late one."""
        controller = _controller(budget = 16384)
        _fill(controller, "big_and_early", 0.75)
        _fill(controller, "small_and_late", 0.35)
        victims = {p.gen_id for p in controller.plan_preemptions()}
        assert victims == {"small_and_late"}

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
        """'Pause all but one' is the worst case, not the first move.

        Newest-first can need more victims than largest-first to free the same room, so
        this asserts the stopping rule rather than a fixed victim list: the sweep must
        stop as soon as the projection fits.
        """
        controller = _controller(budget = 16384)
        _fill(controller, "first", 0.55)
        _register(controller, "second", 400)
        _register(controller, "third", 400)
        _fill(controller, "fourth", 0.48)
        victims = [p.gen_id for p in controller.plan_preemptions()]
        assert victims, "something had to stop"
        assert "first" not in victims, "the oldest chat should be the last to go"
        # Newest first: fourth, then third, then second. It stops once it fits.
        assert victims == ["fourth"], f"one victim was enough, got {victims}"

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
        _fill(controller, "winner", 0.75)
        victim = _fill(controller, "victim", 0.35)
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


class TestNobodyIsExemptFromEviction:
    """The epoch winner is gone, and this is what replaced it.

    A single generation used to be crowned and held unpreemptable until it stopped
    decoding, so that two chats could not trade places forever. Three things retired it.
    It never measurably reduced thrash in simulation. It cost completions under tool load
    (6.57 against 6.77 of eight chats over nine regimes). And in a live run the exempt
    chat simply grew until it had filled the entire 16384 window, at which point
    llama-server truncated its turn and the chat had nowhere left to continue.

    Starvation is now handled by promotion after repeated preemptions, which protects a
    loser without handing anyone the whole cache.
    """

    def test_the_biggest_chat_is_still_preemptable(self):
        controller = _controller(budget = 16384)
        # Registration order matters now, so the huge chat is deliberately NOT the oldest:
        # as the oldest it would be taken last and then spared by the last-holder rule,
        # which would make this assertion unreachable rather than true.
        _fill(controller, "oldest", 0.14)
        _fill(controller, "huge", 0.72)
        _fill(controller, "newest", 0.14)
        victims = {p.gen_id for p in controller.plan_preemptions(needed = 6000)}
        assert "huge" in victims, "an exempt chat can grow until it fills the window"
        assert "oldest" not in victims, "the last holder standing must survive"

    def test_there_is_no_winner_in_the_snapshot(self):
        controller = _controller(budget = 16384)
        _register(controller, "a", 9000)
        _register(controller, "b", 6000)
        controller.plan_preemptions()
        assert controller.snapshot().winner is None

    def test_the_sweep_takes_everyone_when_the_room_demands_it(self):
        """The worst case must remain reachable: all but one can stop."""
        controller = _controller(budget = 16384)
        for name in ("a", "b", "c"):
            _fill(controller, name, 0.33)
        victims = [p.gen_id for p in controller.plan_preemptions(needed = 14000)]
        assert len(victims) >= 2, f"the sweep stopped early, got {victims}"

class TestStarvation:
    """Losing repeatedly must not become never finishing.

    The protection used to be a crown: the starved chat became the exempt epoch winner
    and could not be touched. The crown is gone, so the debt now changes the eviction
    ORDER instead, promoting a repeatedly-preempted chat behind everyone else.
    """

    def _starve(self, controller):
        """Drive `starved` through three preemptions in a row."""
        _fill(controller, "hog", 0.82)
        starved = _fill(controller, "starved", 0.28)
        for _ in range(PROMOTE_AFTER_CONSECUTIVE_PREEMPTIONS):
            assert [p.gen_id for p in controller.plan_preemptions()] == ["starved"]
            controller.note_resumed("starved")
        return starved

    def test_three_consecutive_preemptions_promote_a_chat(self):
        controller = _controller(budget = 16384)
        starved = self._starve(controller)
        assert starved.promoted, "three preemptions in a row must promote it"

    def test_a_promoted_chat_is_taken_last(self):
        """The point of the promotion: newest-first no longer applies to it.

        Without this the starved chat, being the newest registration, would keep being
        chosen first and would never finish.
        """
        controller = _controller(budget = 16384)
        starved = self._starve(controller)
        _fill(controller, "newcomer", 0.28)
        victims = [p.gen_id for p in controller.plan_preemptions()]
        assert victims, "something had to stop"
        assert victims[0] != "starved", (
            f"the promoted chat was taken first anyway, order was {victims}"
        )

    def test_the_debt_clears_once_it_runs_again_unmolested(self):
        controller = _controller(budget = 16384)
        starved = self._starve(controller)
        controller.unregister("hog")
        controller.note_resumed("starved")
        controller.plan_preemptions()
        assert starved.consecutive_preemptions <= PROMOTE_AFTER_CONSECUTIVE_PREEMPTIONS

    def test_a_preemption_after_a_reprieve_is_not_consecutive(self):
        """The rule is three IN A ROW, so a clean stretch resets the count."""
        controller = _controller(budget = 16384)
        starved = self._starve(controller)
        controller.unregister("hog")
        controller.note_resumed("starved")
        starved.consecutive_preemptions = 0
        _register(controller, "hog2", 12000)
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


class TestAPauseMidThoughtKeepsTheThought:
    """The measured livelock: ten pauses, `kept_chars=0` on every one.

    Run of 2026-09-02, Qwen3 at a 16384 window with four chats. Every pause landed
    inside the thought block, so `visible_text` was empty, `has_resume_point()` said
    there was nothing to continue, and each resume re-issued the request whole. One chat
    was paused five times, charged 551 then 29 then 829 then 4196 tokens, and was still
    unfinished when the 1500s deadline cut the run. Preemption was not pausing that
    chat, it was repeatedly destroying its work.
    """

    def test_a_thought_is_a_resume_point(self):
        from core.inference.llama_preemption import StreamCheckpoint

        cp = StreamCheckpoint(visible_text = "", reasoning_text = "Let me work through")
        assert not cp.has_resume_point(), "there is no prose to extend"
        assert cp.has_reasoning_resume_point()
        assert cp.kept_chars() == len("Let me work through")

    def test_prose_wins_when_both_are_present(self):
        from core.inference.llama_preemption import StreamCheckpoint

        cp = StreamCheckpoint(visible_text = "The answer is", reasoning_text = "thinking")
        assert cp.has_resume_point()
        assert not cp.has_reasoning_resume_point(), (
            "resuming as a thought would push already-visible prose back into the block"
        )
        assert cp.kept_chars() == len("The answer is")

    def test_nothing_generated_keeps_nothing(self):
        from core.inference.llama_preemption import StreamCheckpoint

        cp = StreamCheckpoint()
        assert cp.kept_chars() == 0
        assert not cp.has_resume_point()
        assert not cp.has_reasoning_resume_point()

    def test_whitespace_is_not_progress(self):
        from core.inference.llama_preemption import StreamCheckpoint

        cp = StreamCheckpoint(visible_text = "  \n ", reasoning_text = " \n")
        assert not cp.has_resume_point()
        assert not cp.has_reasoning_resume_point()


class TestTheWireCarriesAThoughtPartial:
    def test_a_reasoning_only_turn_counts_as_resumable(self):
        from core.inference.chat_template_helpers import (
            trailing_assistant_resumable,
            trailing_assistant_reasoning,
            trailing_assistant_text,
        )

        convo = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "", "reasoning_content": "half a thought"},
        ]
        # The exact trap: "" is falsy, so every `and trailing_assistant_text(...)` gate
        # dropped the continuation flag for a turn that had real work to continue.
        assert trailing_assistant_text(convo) == ""
        assert not trailing_assistant_text(convo)
        assert trailing_assistant_reasoning(convo) == "half a thought"
        assert trailing_assistant_resumable(convo)

    def test_a_tool_call_turn_is_never_resumable(self):
        from core.inference.chat_template_helpers import trailing_assistant_resumable

        convo = [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": "",
                "reasoning_content": "thinking",
                "tool_calls": [{"id": "1", "function": {"name": "x", "arguments": "{}"}}],
            },
        ]
        assert not trailing_assistant_resumable(convo)

    def test_the_payload_gate_uses_the_resumable_test(self):
        from pathlib import Path

        from core.inference import llama_cpp

        source = Path(llama_cpp.__file__).read_text()
        assert 'if continue_final_message and trailing_assistant_resumable(conversation):' in source

    def test_the_splice_path_still_uses_visible_text_only(self):
        """The manual splice appends its result as VISIBLE text.

        Handing it a thought would paste the reasoning into the answer, which is why
        this is a second predicate rather than a change to the first one.
        """
        from pathlib import Path

        from core.inference import chat_template_helpers

        source = Path(chat_template_helpers.__file__).read_text()
        splice = source.split("def render_prompt_with_boundary", 1)[1].split("\ndef ", 1)[0]
        assert "trailing_assistant_text(messages)" in splice
        assert "trailing_assistant_reasoning" not in splice


class TestReplayedWorkIsStillCharged:
    """Carrying a partial across a pause moves tokens from generated to prompt.

    The run of 2026-09-02 that first kept thoughts across a pause: six pauses carrying
    2001, 2510, 8019, 3017, 1743 and 121 characters, and the crash the whole feature
    exists to prevent came back, four context-exhaustion errors and 38 KV retries where
    the previous build had none. The resumed request replays the partial as prompt while
    the stream's counter restarts at zero, so `base_tokens + generated` measured the
    original prompt and missed everything replayed on top of it.
    """

    def _controller(self, budget = 16384):
        return _controller(budget = budget)

    def test_a_replay_raises_the_baseline(self):
        controller = self._controller()
        controller.register("a", tokens = 1000)
        controller.observe("a", 500)
        assert controller.participant("a").tokens == 1500

        controller.note_replayed("a", 500)
        # Those 500 are now prompt, and the next attempt's counter starts at zero.
        controller.observe("a", 0)
        assert controller.participant("a").tokens == 1500, (
            "occupancy fell back to the admission prompt and lost the replayed partial"
        )

    def test_replays_accumulate_across_pauses(self):
        controller = self._controller()
        controller.register("a", tokens = 1000)
        for charged in (564, 59, 1079, 507):
            controller.note_replayed("a", charged)
        controller.observe("a", 0)
        assert controller.participant("a").tokens == 1000 + 564 + 59 + 1079 + 507

    def test_a_pause_with_nothing_kept_is_not_charged(self):
        """A pause before the first token replays nothing, so it costs nothing."""
        controller = self._controller()
        controller.register("a", tokens = 1000)
        controller.note_replayed("a", 0)
        controller.observe("a", 0)
        assert controller.participant("a").tokens == 1000

    def test_an_unknown_generation_is_ignored(self):
        controller = self._controller()
        controller.note_replayed("gone", 500)  # must not raise

    def test_the_adapter_charges_only_what_was_actually_kept(self):
        from core.inference.llama_preemption import (
            ControllerPreemptionPolicy,
            PreemptSignal,
            StreamCheckpoint,
        )

        controller = self._controller()
        controller.register("a", tokens = 1000)
        policy = ControllerPreemptionPolicy(controller, "a", PreemptSignal())

        # Decoded tokens but nothing carried: re-issued whole, so nothing is replayed.
        policy.on_preempted(StreamCheckpoint(charged_tokens = 700))
        controller.observe("a", 0)
        assert controller.participant("a").tokens == 1000

        policy.on_preempted(
            StreamCheckpoint(charged_tokens = 700, reasoning_text = "a thought")
        )
        controller.observe("a", 0)
        assert controller.participant("a").tokens == 1700


class TestAChatThatOutgrewTheSharedCeiling:
    """No eviction can admit it, so waiting for room is a deadlock, not a delay.

    The buffer holds back a proportional watermark plus the drafts of every slot. A chat
    needing more than `budget - buffer` therefore cannot be admitted or resumed no matter
    who is evicted, and the resume wait had no answer for that: it waited until its client
    disconnected. Observed live as one chat of four open for a full 2400s deadline while
    llama-server sat idle with every slot released and requests_processing at 0.

    Simulated over the tool-heavy regime, letting such a chat take the cache alone moved
    makespan from 166799 steps to 924 and starvation from 1.6 chats of eight to none.
    """

    def _controller_with_drafts(self, budget = 16384, drafts = 2, slots = 4):
        from core.inference.llama_preemption import PreemptionController

        controller = PreemptionController("solo-test")
        controller.configure(
            budget = budget, kv_unified = True, draft_tokens = drafts, slots = slots
        )
        return controller

    def test_a_want_past_the_shared_ceiling_is_recognised(self):
        controller = self._controller_with_drafts()
        snapshot = controller.snapshot()
        ceiling = snapshot.budget - snapshot.buffer
        assert not controller.outgrew_the_shared_ceiling(ceiling)
        assert controller.outgrew_the_shared_ceiling(ceiling + 1)

    def test_it_may_resume_once_the_cache_is_its_own(self):
        controller = self._controller_with_drafts()
        snapshot = controller.snapshot()
        ceiling = snapshot.budget - snapshot.buffer
        want = ceiling + 500
        # Alone, it fits: the reaction buffer protects concurrent growers, and there are
        # none. This is the assertion whose absence was the hang.
        assert controller.room_for("solo", want)

    def test_it_may_not_resume_beside_anyone(self):
        controller = self._controller_with_drafts()
        snapshot = controller.snapshot()
        want = snapshot.budget - snapshot.buffer + 500
        _register(controller, "other", 3000)
        assert not controller.room_for("solo", want)

    def test_a_want_past_the_cache_itself_can_never_fit(self):
        controller = self._controller_with_drafts()
        assert controller.cannot_ever_fit(16384)
        assert not controller.cannot_ever_fit(10000)

    def test_the_solo_ceiling_beats_the_shared_one_by_more_than_a_rounding(self):
        """Keeping the ratio in the solo case made this worth six tokens on a 16384
        cache, so a chat needing 15915 still could not run and the deadlock stood."""
        controller = self._controller_with_drafts()
        snapshot = controller.snapshot()
        shared = snapshot.budget - snapshot.buffer
        # Probe the solo ceiling through the public question rather than the private one.
        solo = next(
            w for w in range(snapshot.budget, shared, -1) if not controller.cannot_ever_fit(w)
        )
        # Most of the reaction headroom comes back, since a lone chat has nobody to
        # react to. Expressed against the buffer rather than as a literal: this said
        # "> 1000" while the buffer was 2458 tokens, and the buffer is 776 now.
        assert solo - shared >= snapshot.buffer // 2, (
            f"solo ceiling {solo} barely clears the shared {shared} against a buffer of "
            f"{snapshot.buffer}; the reaction headroom is still being charged to a chat "
            f"that has nobody to react to"
        )

    def test_none_of_this_applies_when_preemption_is_off(self):
        controller = self._controller_with_drafts()
        controller.configure(budget = 0, kv_unified = False)
        assert not controller.outgrew_the_shared_ceiling(999999)
        assert not controller.cannot_ever_fit(999999)
