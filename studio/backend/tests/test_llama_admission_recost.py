# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Re-costing a live lease as its tool loop grows.

#9392 admitted generations against the KV cache instead of the slot count, after two chats
killed each other on a 2048-token cache (565 + 1485, neither too long alone). It could not
know a tool loop's final size, since each round appends its results and re-sends the
conversation, so it reserved the WHOLE cache: airtight, and it made every tool chat run
alone (any lit pill sets ``enable_tools``). Measured on a 262144 cache, four tool chats
reached first token at 0.1s, 2.8s, 4.6s and 8.8s, one after another.

Re-costing is the alternative that PR named and skipped. These pin the properties that
make it safe to call from inside a running generator.
"""

from __future__ import annotations

import asyncio
import threading
import time

import pytest

from core.inference.llama_admission import LlamaAdmissionConfig, LlamaAdmissionQueue


def _reserve(queue, *, capacity, tokens, budget):
    return queue.reserve(
        capacity = capacity,
        config = LlamaAdmissionConfig(),
        tokens = tokens,
        budget = budget,
    )


def _lease(
    queue,
    *,
    capacity = 4,
    tokens,
    budget,
):
    """reserve() reads the running loop, so every test here is async."""
    reservation = _reserve(queue, capacity = capacity, tokens = tokens, budget = budget)
    lease = reservation.lease_nowait()
    assert lease is not None, "expected this reservation to be admitted"
    return lease


class TestTheQueueSide:
    @pytest.mark.asyncio
    async def test_growth_that_fits_is_applied(self):
        queue = LlamaAdmissionQueue("test")
        _lease(queue, tokens = 1000, budget = 4096)
        assert queue.try_recost(1000, 2000) is True
        assert queue.snapshot().committed == 2000

    @pytest.mark.asyncio
    async def test_growth_that_does_not_fit_is_refused_and_changes_nothing(self):
        """Refused, not blocked: this runs inside the generator, where a round that waited
        could be waiting on a holder that is waiting on it."""
        queue = LlamaAdmissionQueue("test")
        _lease(queue, tokens = 2000, budget = 4096)
        _lease(queue, tokens = 2000, budget = 4096)
        assert queue.snapshot().committed == 4000
        assert queue.try_recost(2000, 3000) is False
        assert queue.snapshot().committed == 4000, "a refused growth must not move anything"

    @pytest.mark.asyncio
    async def test_a_lone_holder_may_grow_past_the_budget(self):
        """The escape admission uses: refusing the only holder stalls a conversation
        nothing else can unblock, and llama.cpp surfaces a real overflow itself."""
        queue = LlamaAdmissionQueue("test")
        _lease(queue, tokens = 2000, budget = 4096)
        assert queue.try_recost(2000, 9000) is True
        assert queue.snapshot().committed == 9000

    @pytest.mark.asyncio
    async def test_shrinking_always_applies(self):
        queue = LlamaAdmissionQueue("test")
        _lease(queue, tokens = 2000, budget = 4096)
        _lease(queue, tokens = 2000, budget = 4096)
        assert queue.try_recost(2000, 500) is True
        assert queue.snapshot().committed == 2500

    @pytest.mark.asyncio
    async def test_no_budget_means_nothing_to_account(self):
        queue = LlamaAdmissionQueue("test")
        assert queue.try_recost(1000, 999999) is True


class TestTheLeaseSide:
    @pytest.mark.asyncio
    async def test_recost_moves_the_queue_and_the_lease_together(self):
        queue = LlamaAdmissionQueue("test")
        lease = _lease(queue, tokens = 1000, budget = 8192)
        assert lease.recost(2500) is True
        assert queue.snapshot().committed == 2500
        # Release must hand back the NEW figure, not the one it was admitted on.
        lease.release()
        assert queue.snapshot().committed == 0

    @pytest.mark.asyncio
    async def test_a_refused_recost_leaves_release_correct(self):
        """The leak this guards: if the queue took the growth while the lease kept the old
        number, release would hand back less than it holds and strand the difference."""
        queue = LlamaAdmissionQueue("test")
        first = _lease(queue, tokens = 3000, budget = 4096)
        second = _lease(queue, tokens = 1000, budget = 4096)
        assert second.recost(3000) is False
        second.release()
        first.release()
        assert queue.snapshot().committed == 0

    @pytest.mark.asyncio
    async def test_a_released_lease_recosts_to_nothing(self):
        queue = LlamaAdmissionQueue("test")
        lease = _lease(queue, tokens = 1000, budget = 4096)
        lease.release()
        assert lease.recost(3000) is True, "a finished run is not an error"
        assert queue.snapshot().committed == 0, "and must not re-commit anything"

    @pytest.mark.asyncio
    async def test_recost_is_idempotent_at_the_same_size(self):
        queue = LlamaAdmissionQueue("test")
        lease = _lease(queue, tokens = 1000, budget = 4096)
        for _ in range(5):
            assert lease.recost(1000) is True
        assert queue.snapshot().committed == 1000

    @pytest.mark.asyncio
    async def test_concurrent_recost_and_release_do_not_strand_tokens(self):
        """release() takes the lease lock and then the queue's; recost takes them in the
        same order, which is what keeps this from deadlocking or leaking."""
        for _ in range(40):
            queue = LlamaAdmissionQueue("test")
            lease = _lease(queue, tokens = 1000, budget = 1_000_000)
            barrier = threading.Barrier(2)

            def grow(lease = lease, barrier = barrier):
                barrier.wait()
                lease.recost(5000)

            def drop(lease = lease, barrier = barrier):
                barrier.wait()
                lease.release()

            # Daemon throughout this file: a failed assertion leaves a growth thread
            # spinning, and a live non-daemon one wedges exit instead of reporting it.
            threads = [
                threading.Thread(target = grow, daemon = True),
                threading.Thread(target = drop, daemon = True),
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            assert queue.snapshot().committed == 0, "a race left tokens committed with no holder"


class TestFourToolChatsTogether:
    @pytest.mark.asyncio
    async def test_four_tool_chats_are_admitted_and_can_each_grow_a_little(self):
        """The behaviour the change exists for, end to end at the queue level."""
        queue = LlamaAdmissionQueue("test")
        budget = 262144
        share = budget // 4
        leases = [_lease(queue, tokens = share, budget = budget) for _ in range(4)]
        assert queue.snapshot().committed == budget, "all four tool chats admitted at once"
        # The cache is exactly full, so nobody may grow at anyone else's expense.
        assert leases[0].recost(share + 1000) is False
        # ... until someone finishes.
        leases[3].release()
        assert leases[0].recost(share + 1000) is True


class TestWaitingForRoomInsteadOfRunningOverIt:
    """``recost`` alone accounts for growth without enforcing it.

    A refused recost leaves the run at its old figure and it sends the bigger prompt
    anyway. Four loops opening at a share each and growing together is then the measured
    failure: llama.cpp halves the batch to 1, gives up, and ``Context size has been
    exceeded`` clears EVERY decoding slot, not just the one that overflowed.
    ``recost_waiting`` is what makes the accounting binding.
    """

    @pytest.mark.asyncio
    async def test_growth_that_fits_never_touches_the_wait_line(self):
        queue = LlamaAdmissionQueue("test")
        lease = _lease(queue, tokens = 1000, budget = 8192)
        assert lease.recost_waiting(2000) is True
        assert queue.snapshot().committed == 2000
        assert queue._reparking == 0, "a growth that fit should not have yielded anything"

    @pytest.mark.asyncio
    async def test_a_waiter_is_let_in_when_a_holder_finishes(self):
        queue = LlamaAdmissionQueue("test")
        first = _lease(queue, tokens = 2000, budget = 4096)
        second = _lease(queue, tokens = 2000, budget = 4096)

        done: list = []

        def grow():
            done.append(second.recost_waiting(3500, poll_s = 0.01))

        thread = threading.Thread(target = grow, daemon = True)
        thread.start()
        # It cannot proceed: 2000 is still held by `first` and 3500 does not fit beside it.
        thread.join(0.2)
        assert thread.is_alive(), "expected the growth to wait, not to be refused"
        # Yielding first is what makes this resolvable at all.
        assert queue.snapshot().committed == 2000
        first.release()
        thread.join(5)
        assert not thread.is_alive()
        assert done == [True]
        assert queue.snapshot().committed == 3500

    @pytest.mark.asyncio
    async def test_four_loops_growing_together_do_not_overcommit(self):
        """The deadlock case that decides the design. Every holder wants more than a
        share, so blocking while still holding leaves all four waiting on each other.
        Yielding first means _committed strictly falls, so somebody fits."""
        queue = LlamaAdmissionQueue("test")
        budget = 4096
        leases = [_lease(queue, tokens = budget // 4, budget = budget) for _ in range(4)]
        assert queue.snapshot().committed == budget

        results: list = []
        lock = threading.Lock()

        def grow(lease):
            ok = lease.recost_waiting(budget // 2, poll_s = 0.01)
            with lock:
                results.append(ok)
            # Finishing is what lets the next one in.
            lease.release()

        threads = [threading.Thread(target = grow, args = (lease,), daemon = True) for lease in leases]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(20)
        assert not any(t.is_alive() for t in threads), "a growth deadlocked"
        assert results == [True] * 4
        assert queue.snapshot().committed == 0
        assert queue._reparking == 0

    @pytest.mark.asyncio
    async def test_the_budget_is_never_exceeded_while_they_wait(self):
        """Only one reparker may win the committed-is-zero escape: four racing an empty
        cache would each see zero and each admit itself."""
        queue = LlamaAdmissionQueue("test")
        budget = 4096
        leases = [_lease(queue, tokens = budget // 4, budget = budget) for _ in range(4)]
        peak = []
        stop = threading.Event()

        def watch():
            while not stop.is_set():
                peak.append(queue.snapshot().committed)
                time.sleep(0.005)

        watcher = threading.Thread(target = watch, daemon = True)
        watcher.start()

        def grow(lease):
            lease.recost_waiting(budget, poll_s = 0.01)
            lease.release()

        threads = [threading.Thread(target = grow, args = (lease,), daemon = True) for lease in leases]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(20)
        stop.set()
        watcher.join(5)
        assert max(peak) <= budget, f"committed reached {max(peak)} against a {budget} cache"

    @pytest.mark.asyncio
    async def test_cancelling_a_waiting_round_restores_its_commitment(self):
        """Stop pressed while waiting. Until the run releases it still occupies
        llama-server's cache, so the pool must know about it."""
        queue = LlamaAdmissionQueue("test")
        first = _lease(queue, tokens = 3000, budget = 4096)
        second = _lease(queue, tokens = 1000, budget = 4096)
        cancel = threading.Event()

        out: list = []

        def grow():
            out.append(second.recost_waiting(4000, cancel_event = cancel, poll_s = 0.01))

        thread = threading.Thread(target = grow, daemon = True)
        thread.start()
        thread.join(0.2)
        assert thread.is_alive()
        cancel.set()
        thread.join(5)
        assert out == [False], "a cancelled wait reports that it did not take the new size"
        assert queue.snapshot().committed == 4000, "the old commitment is back"
        assert queue._reparking == 0
        second.release()
        first.release()
        assert queue.snapshot().committed == 0

    @pytest.mark.asyncio
    async def test_a_reparker_is_not_overtaken_by_a_new_arrival(self):
        """An in-flight conversation beats one that has not started. Otherwise a steady
        arrival rate holds a growing run at its opening size indefinitely."""
        queue = LlamaAdmissionQueue("test")
        first = _lease(queue, tokens = 2000, budget = 4096)
        second = _lease(queue, tokens = 2000, budget = 4096)

        def grow():
            second.recost_waiting(3000, poll_s = 0.01)

        thread = threading.Thread(target = grow, daemon = True)
        thread.start()
        thread.join(0.2)
        assert thread.is_alive()
        # A newcomer arrives while the reparker waits, and must not be granted the room
        # the reparker just gave up.
        newcomer = queue.reserve(
            capacity = 4, config = LlamaAdmissionConfig(), tokens = 1000, budget = 4096
        )
        assert newcomer.lease_nowait() is None, "a new arrival overtook a growing run"
        first.release()
        thread.join(5)
        assert not thread.is_alive()
        # Behind the reparker, not instead of it. The reclaim brings the barrier down and
        # is the last thing to touch the queue, so if it does not run admission itself a
        # request that fits in the room left over waits out the whole grown run -- and
        # since a queued waiter shuts reserve()'s fast path, so does every later arrival.
        for _ in range(100):
            await asyncio.sleep(0.01)
            if newcomer.lease_nowait() is not None:
                break
        assert (
            newcomer.lease_nowait() is not None
        ), "the last repark barrier came down without re-running admission"
        assert queue.snapshot().committed == 4000


class TestGivingUpTheWait:
    """What a reparker owes the pool when its wait ends without the bigger commitment.

    ``yield_commitment`` has already taken the old figure off ``_committed``, but the
    lease still occupies that KV at llama-server. Handing it back is a CORRECTION, not a
    request, and a full cache must not be able to refuse it: a lease that records a
    commitment it never restored subtracts it again on release, leaving phantom room.
    """

    @pytest.mark.asyncio
    async def test_a_full_cache_cannot_refuse_the_restore(self):
        queue = LlamaAdmissionQueue("test")
        loser = _lease(queue, tokens = 1024, budget = 4096)
        winner = _lease(queue, tokens = 1024, budget = 4096)
        assert queue.snapshot().committed == 2048

        cancel = threading.Event()
        out: list = []

        def grow():
            out.append(loser.recost_waiting(4096, cancel_event = cancel, poll_s = 0.01))

        thread = threading.Thread(target = grow, daemon = True)
        thread.start()
        while queue.snapshot().committed != 1024:
            await asyncio.sleep(0.005)
        # The other run takes the whole cache while this one is parked, leaving no room
        # for the restore to ask for.
        assert winner.recost(4096) is True
        assert queue.snapshot().committed == 4096

        cancel.set()
        thread.join(5)
        assert out == [False]
        assert queue._reparking == 0
        # Both leases really hold their figures at llama-server, so the pool says so even
        # over the budget: that is what is genuinely resident.
        assert (
            queue.snapshot().committed == 4096 + 1024
        ), "a restore the cache could not fit was dropped instead of recorded"

        loser.release()
        assert (
            queue.snapshot().committed == 4096
        ), "release subtracted a commitment that was never restored"
        # And the phantom room that leak created must not admit anyone.
        newcomer = queue.reserve(
            capacity = 4, config = LlamaAdmissionConfig(), tokens = 1000, budget = 4096
        )
        assert (
            newcomer.lease_nowait() is None
        ), "a newcomer was admitted into room the winner is still using"
        winner.release()

    @pytest.mark.asyncio
    async def test_a_released_lease_restores_nothing(self):
        """release() already handed back the 0 held while parked, so re-committing here
        would strand the difference for the life of the process."""
        queue = LlamaAdmissionQueue("test")
        holder = _lease(queue, tokens = 2000, budget = 4096)
        _lease(queue, tokens = 2000, budget = 4096)

        cancel = threading.Event()
        thread = threading.Thread(
            target = lambda: holder.recost_waiting(4000, cancel_event = cancel, poll_s = 0.01),
            daemon = True,
        )
        thread.start()
        while queue.snapshot().committed != 2000:
            await asyncio.sleep(0.005)
        holder.release()
        cancel.set()
        thread.join(5)
        assert not thread.is_alive()
        assert queue._reparking == 0
        assert queue.snapshot().committed == 2000, "the released lease was re-committed"


class TestYieldingIsGatedOnTheServerActuallyClearing:
    """A slot being idle is not the same as its KV cells being reclaimed.

    Under ``--kv-unified`` a finished round's cells stay resident until ``prompt_clear()``,
    which llama-server runs only under ``--cache-idle-slots``. ``--cache-ram 0``
    force-disables that, and Studio emits it on Windows under full GPU offload (#5692)
    next to ``--kv-unified``. Yielding there hands a second caller occupied room.
    """

    @pytest.mark.asyncio
    async def test_growth_that_fits_does_not_care(self):
        """The cheap path never yields anything, so gating must not disturb it."""
        queue = LlamaAdmissionQueue("test")
        holder = _lease(queue, tokens = 1000, budget = 4096)
        assert holder.recost_waiting(2000, allow_yield = False) is True
        assert queue.snapshot().committed == 2000

    @pytest.mark.asyncio
    async def test_growth_that_does_not_fit_declines_instead_of_yielding(self):
        queue = LlamaAdmissionQueue("test")
        holder = _lease(queue, tokens = 2000, budget = 4096)
        _lease(queue, tokens = 2000, budget = 4096)

        assert holder.recost_waiting(4000, allow_yield = False) is False
        assert holder._tokens == 2000, "the old commitment was not kept"
        assert queue.snapshot().committed == 4000, "capacity was handed out twice"
        assert queue._reparking == 0

    @pytest.mark.asyncio
    async def test_it_does_not_block(self):
        """Declining is the pre-existing behaviour: do not wait for room that is never
        coming back on this server."""
        queue = LlamaAdmissionQueue("test")
        holder = _lease(queue, tokens = 2000, budget = 4096)
        _lease(queue, tokens = 2000, budget = 4096)

        started = time.monotonic()
        assert holder.recost_waiting(4000, allow_yield = False, timeout_s = 30.0, poll_s = 0.5) is False
        assert time.monotonic() - started < 1.0

    @pytest.mark.asyncio
    async def test_yielding_is_still_the_default(self):
        """Old callers, and every server that does clear, keep the waiting behaviour."""
        queue = LlamaAdmissionQueue("test")
        holder = _lease(queue, tokens = 2000, budget = 4096)
        other = _lease(queue, tokens = 2000, budget = 4096)

        thread = threading.Thread(
            target = lambda: holder.recost_waiting(4000, poll_s = 0.01),
            daemon = True,
        )
        thread.start()
        while queue.snapshot().committed != 2000:
            await asyncio.sleep(0.005)
        other.release()
        thread.join(5)
        assert not thread.is_alive()
        assert queue.snapshot().committed == 4000
