# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Resume order at the controller's room gate, and residency samples that started too early."""

import asyncio
import threading
import time

import pytest

from core.inference.llama_preemption import (
    ControllerPreemptionPolicy,
    ParticipantState,
    PreemptionController,
    PreemptSignal,
    reset_preemption_controllers,
)

pytestmark = pytest.mark.usefixtures("preemption_opted_in")


@pytest.fixture(autouse = True)
def _clean_registry():
    reset_preemption_controllers()
    yield
    reset_preemption_controllers()


class _Lease:
    def __init__(self, tokens = 1000):
        self.tokens = tokens
        self.slot = 0
        self.is_released = False

    def preempt(self):
        return True

    async def resume_async(self, tokens, **kwargs):
        return True


def _controller(budget = 16384, slots = 4):
    controller = PreemptionController("k")
    controller.configure(budget = budget, kv_unified = True, slots = slots)
    return controller


def _paused(controller, gen_id, tokens):
    controller.register(gen_id, lease = _Lease(tokens), tokens = tokens, signal = PreemptSignal())
    controller.set_state(gen_id, ParticipantState.PAUSED)


class TestAnOlderResumeKeepsItsPlaceAtTheRoomGate:
    """The ceiling is 16384 - 768 = 15616; a decoder holds 10000, so 7000 never fits and
    3000 always does."""

    def test_a_stream_of_smaller_resumes_does_not_starve_it(self):
        controller = _controller()
        controller.register("d", lease = _Lease(10000), tokens = 10000, signal = PreemptSignal())
        controller.observe("d", 0)
        _paused(controller, "old", 7000)
        controller.begin_resume("old", 7000)
        assert controller.try_grant_resume("old", 7000) is False
        for i in range(6):
            gen_id = f"s{i}"
            _paused(controller, gen_id, 3000)
            controller.begin_resume(gen_id, 3000)
            assert controller.try_grant_resume(gen_id, 3000) is False, "overtook the older resume"
            controller.end_resume(gen_id)
            controller.unregister(gen_id)

    def test_the_older_resume_takes_the_room_when_it_comes(self):
        controller = _controller()
        controller.register("d", lease = _Lease(10000), tokens = 10000, signal = PreemptSignal())
        controller.observe("d", 0)
        _paused(controller, "old", 7000)
        controller.begin_resume("old", 7000)
        assert controller.try_grant_resume("old", 7000) is False
        controller.unregister("d")
        assert controller.try_grant_resume("old", 7000) is True
        assert controller.participant("old").state == ParticipantState.RESUMING
        controller.end_resume("old")
        # And the line is clear again, so a later resume is not held back by a ghost.
        _paused(controller, "s", 3000)
        controller.begin_resume("s", 3000)
        assert controller.try_grant_resume("s", 3000) is True

    def test_a_wait_that_gives_up_frees_the_line(self):
        controller = _controller()
        controller.register("d", lease = _Lease(10000), tokens = 10000, signal = PreemptSignal())
        controller.observe("d", 0)
        _paused(controller, "old", 7000)
        controller.begin_resume("old", 7000)
        _paused(controller, "s", 3000)
        controller.begin_resume("s", 3000)
        assert controller.try_grant_resume("s", 3000) is False
        controller.end_resume("old")
        assert controller.try_grant_resume("s", 3000) is True, "the line still holds a dead place"

    def test_a_dead_waiter_does_not_hold_the_room_forever(self):
        controller = _controller()
        lease = _Lease(7000)
        controller.register("old", lease = lease, tokens = 7000, signal = PreemptSignal())
        controller.set_state("old", ParticipantState.PAUSED)
        controller.begin_resume("old", 7000)
        lease.is_released = True
        _paused(controller, "s", 3000)
        controller.begin_resume("s", 3000)
        # _prune_locked drops the released lease; its place has to go with it.
        controller.snapshot()
        assert controller.try_grant_resume("s", 3000) is True
        assert controller.resume_queue_depth() == 1

    def test_an_uncontended_resume_is_granted_at_once(self):
        controller = _controller()
        _paused(controller, "solo", 7000)
        assert controller.resume_queue_depth() == 0
        assert controller.try_grant_resume("solo", 7000) is True

    def test_a_caller_without_a_place_waits_behind_the_line(self):
        controller = _controller()
        controller.register("d", lease = _Lease(10000), tokens = 10000, signal = PreemptSignal())
        controller.observe("d", 0)
        _paused(controller, "old", 7000)
        controller.begin_resume("old", 7000)
        _paused(controller, "s", 3000)
        assert controller.try_grant_resume("s", 3000) is False

    def test_the_room_a_line_is_owed_is_not_spent_by_a_solo_grant(self):
        """The cache empties: the escape that lets a chat past the shared ceiling run alone
        must not spend it on the wait behind."""
        controller = _controller()
        _paused(controller, "old", 7000)
        controller.begin_resume("old", 7000)
        _paused(controller, "big", 15000)
        controller.begin_resume("big", 15000)
        assert controller.try_grant_resume("big", 15000) is False
        controller.end_resume("old")
        assert controller.try_grant_resume("big", 15000) is True


class _RefusingController(PreemptionController):
    """__slots__ makes the real one unpatchable, so refuse room from a subclass instead."""

    def __init__(self):
        super().__init__("k")
        self.seen = []

    def try_grant_resume(self, gen_id, want):
        self.seen.append(self.resume_queue_depth())
        return False


class TestAwaitResumeTakesItsPlaceBeforeTheRoomTest:
    def test_the_place_is_taken_before_the_first_room_test_and_dropped_on_give_up(self):
        controller = _RefusingController()
        controller.configure(budget = 16384, kv_unified = True, slots = 4)
        signal = PreemptSignal()
        controller.register("old", lease = _Lease(7000), tokens = 7000, signal = signal)
        controller.set_state("old", ParticipantState.PAUSED)
        loop = asyncio.new_event_loop()
        thread = threading.Thread(target = loop.run_forever, daemon = True)
        thread.start()
        seen = controller.seen
        try:
            policy = ControllerPreemptionPolicy(controller, "old", signal, loop = loop)
            assert policy.await_resume(timeout = 0.15) is False
        finally:
            loop.call_soon_threadsafe(loop.stop)
            thread.join(timeout = 5)
            loop.close()
        assert seen and seen[0] == 1, "the room was tested before the place in line was taken"
        assert controller.resume_queue_depth() == 0, "the give-up left its place in the line"

    def test_an_older_wait_is_not_overtaken_while_it_is_still_waiting(self):
        controller = _controller()
        signal = PreemptSignal()
        controller.register("old", lease = _Lease(7000), tokens = 7000, signal = signal)
        controller.set_state("old", ParticipantState.PAUSED)
        controller.register("d", lease = _Lease(10000), tokens = 10000, signal = PreemptSignal())
        controller.observe("d", 0)
        loop = asyncio.new_event_loop()
        thread = threading.Thread(target = loop.run_forever, daemon = True)
        thread.start()
        policy = ControllerPreemptionPolicy(controller, "old", signal, loop = loop)
        waiter = threading.Thread(target = policy.await_resume, kwargs = {"timeout": 2.0})
        waiter.start()
        try:
            deadline = time.monotonic() + 2.0
            while controller.resume_queue_depth() == 0 and time.monotonic() < deadline:
                time.sleep(0.01)
            assert controller.resume_queue_depth() == 1, "the wait never took a place in line"
            _paused(controller, "s", 3000)
            controller.begin_resume("s", 3000)
            assert controller.try_grant_resume("s", 3000) is False
        finally:
            waiter.join(timeout = 10)
            loop.call_soon_threadsafe(loop.stop)
            thread.join(timeout = 5)
            loop.close()
        assert not waiter.is_alive(), "the older wait never ended"
        assert controller.resume_queue_depth() <= 1


class TestAResidencySampleIsOrderedAgainstTheMark:
    def test_a_sample_in_flight_before_the_mark_does_not_promote(self):
        controller = _controller()
        controller.register(
            "raw", lease = _Lease(4000), tokens = 4000, state = ParticipantState.STREAMING_RAW
        )
        controller.note_resident(5000)
        assert controller.snapshot().committed == 9000, "unmeasured: residency plus the charge"
        # A probe leaves for /slots, and the prefill finishes while it is in flight.
        epoch = controller.residency_epoch()
        controller.note_measured("raw")
        controller.note_resident(5000, started_at_seq = epoch)
        assert (
            controller.snapshot().committed == 9000
        ), "a reading taken before the prefill swallowed the whole charge"
        # The next probe starts after the mark, and that one does hold its cells.
        later = controller.residency_epoch()
        controller.note_resident(5000, started_at_seq = later)
        assert controller.snapshot().committed == 5000

    def test_a_sample_started_after_the_mark_still_promotes(self):
        controller = _controller()
        controller.register(
            "raw", lease = _Lease(4000), tokens = 4000, state = ParticipantState.STREAMING_RAW
        )
        controller.note_resident(4000)
        controller.note_measured("raw")
        epoch = controller.residency_epoch()
        controller.note_resident(4000, started_at_seq = epoch)
        assert controller.snapshot().committed == 4000

    def test_only_the_marks_the_sample_predates_are_held_back(self):
        controller = _controller()
        for gen_id in ("a", "b"):
            controller.register(
                gen_id, lease = _Lease(1000), tokens = 1000, state = ParticipantState.STREAMING_RAW
            )
        controller.note_resident(6000)
        controller.note_measured("a")
        epoch = controller.residency_epoch()
        controller.note_measured("b")
        controller.note_resident(6000, started_at_seq = epoch)
        assert controller.participant("a").measured is True
        assert controller.participant("b").measured is False
        assert controller.snapshot().committed == 7000

    def test_an_older_sample_finishing_after_a_newer_one_is_dropped(self):
        # Two probes leave together; the arming one outlives its join window and lands
        # after the token-path one that read a fuller cache.
        controller = _controller()
        controller.register(
            "raw", lease = _Lease(1000), tokens = 1000, state = ParticipantState.STREAMING_RAW
        )
        controller.note_measured("raw")
        arming = controller.residency_epoch()
        newer = controller.residency_epoch()
        controller.note_resident(9000, started_at_seq = newer)
        controller.note_resident(2000, started_at_seq = arming)
        assert controller.snapshot().committed == 9000, "the older count came back"
        controller.note_resident(None, started_at_seq = arming)
        assert controller.snapshot().committed == 9000, "a stale failed read cleared it"
        # A probe sent after the recorded one is the newest word, whatever it says.
        later = controller.residency_epoch()
        controller.note_resident(2000, started_at_seq = later)
        assert controller.snapshot().committed == 2000

    def test_a_caller_that_states_no_epoch_keeps_the_old_behaviour(self):
        controller = _controller()
        controller.register(
            "raw", lease = _Lease(4000), tokens = 4000, state = ParticipantState.STREAMING_RAW
        )
        controller.note_resident(5000)
        controller.note_measured("raw")
        controller.note_resident(5000)
        assert controller.snapshot().committed == 5000
