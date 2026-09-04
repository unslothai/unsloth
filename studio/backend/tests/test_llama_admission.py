# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

import asyncio
import os
import sys
import threading

import pytest

_backend = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, _backend)

from core.inference import llama_admission
from core.inference.llama_admission import (
    ADMISSION_CONTROL_ENV,
    ADMISSION_KEEPALIVE_INTERVAL_ENV,
    ADMISSION_MAX_QUEUE_ENV,
    ADMISSION_QUEUE_PER_SLOT_ENV,
    ADMISSION_QUEUE_TIMEOUT_ENV,
    DEFAULT_ADMISSION_KEEPALIVE_INTERVAL_S,
    DEFAULT_ADMISSION_MAX_QUEUE,
    DEFAULT_ADMISSION_QUEUE_TIMEOUT_S,
    LlamaAdmissionConfig,
    LlamaAdmissionQueueFull,
    get_llama_admission_queue,
    llama_admission_config_from_env,
    reset_llama_admission_queues,
)


_ADMISSION_ENV = (
    ADMISSION_CONTROL_ENV,
    ADMISSION_QUEUE_TIMEOUT_ENV,
    ADMISSION_KEEPALIVE_INTERVAL_ENV,
    ADMISSION_MAX_QUEUE_ENV,
    ADMISSION_QUEUE_PER_SLOT_ENV,
    *llama_admission._LEGACY_ENV.values(),
)


@pytest.fixture(autouse = True)
def _reset_queues(monkeypatch):
    # Clear ambient settings for every test, not just the ones that remember to:
    # a canonical name set on the machine silently beats the legacy name a test
    # is exercising, and the queue registry is process-global.
    for name in _ADMISSION_ENV:
        monkeypatch.delenv(name, raising = False)
    reset_llama_admission_queues()
    yield
    reset_llama_admission_queues()


def test_admission_config_defaults(monkeypatch):
    for name in (
        ADMISSION_CONTROL_ENV,
        ADMISSION_QUEUE_TIMEOUT_ENV,
        ADMISSION_KEEPALIVE_INTERVAL_ENV,
        ADMISSION_MAX_QUEUE_ENV,
        ADMISSION_QUEUE_PER_SLOT_ENV,
        "UNSLOTH_OPENAI_COMPAT_ADMISSION_CONTROL",
        "UNSLOTH_OPENAI_COMPAT_ADMISSION_QUEUE_TIMEOUT",
        "UNSLOTH_OPENAI_COMPAT_ADMISSION_KEEPALIVE_INTERVAL",
        "UNSLOTH_OPENAI_COMPAT_ADMISSION_MAX_QUEUE",
    ):
        monkeypatch.delenv(name, raising = False)

    config = llama_admission_config_from_env()

    # Literals, not the module constants: comparing a default to itself would let
    # any future value change through silently.
    assert config.enabled is True
    assert config.queue_timeout_s is None  # wait forever
    assert config.keepalive_interval_s == 5.0
    assert config.max_queue is None  # no absolute cap
    assert config.queue_per_slot == 16
    assert (DEFAULT_ADMISSION_QUEUE_TIMEOUT_S, DEFAULT_ADMISSION_MAX_QUEUE) == (None, None)
    assert DEFAULT_ADMISSION_KEEPALIVE_INTERVAL_S == 5.0


def test_admission_config_env_overrides(monkeypatch):
    monkeypatch.setenv(ADMISSION_CONTROL_ENV, "off")
    monkeypatch.setenv(ADMISSION_QUEUE_TIMEOUT_ENV, "0")
    monkeypatch.setenv(ADMISSION_KEEPALIVE_INTERVAL_ENV, "0.25")
    monkeypatch.setenv(ADMISSION_MAX_QUEUE_ENV, "0")

    config = llama_admission_config_from_env()

    assert config.enabled is False
    assert config.queue_timeout_s is None
    assert config.keepalive_interval_s == 0.25
    assert config.max_queue is None


def test_admission_config_honors_legacy_openai_compat_env(monkeypatch):
    # The queue is shared with /v1/messages now, but existing OPENAI_COMPAT
    # settings must keep working.
    monkeypatch.setenv("UNSLOTH_OPENAI_COMPAT_ADMISSION_MAX_QUEUE", "7")
    monkeypatch.setenv("UNSLOTH_OPENAI_COMPAT_ADMISSION_CONTROL", "off")

    config = llama_admission_config_from_env()

    assert config.max_queue == 7
    assert config.enabled is False


def test_admission_config_prefers_neutral_env_over_legacy(monkeypatch):
    monkeypatch.setenv("UNSLOTH_OPENAI_COMPAT_ADMISSION_MAX_QUEUE", "7")
    monkeypatch.setenv(ADMISSION_MAX_QUEUE_ENV, "3")

    assert llama_admission_config_from_env().max_queue == 3


def test_admission_config_positive_queue_timeout_env(monkeypatch):
    monkeypatch.setenv(ADMISSION_QUEUE_TIMEOUT_ENV, "600")

    config = llama_admission_config_from_env()

    assert config.queue_timeout_s == 600.0


def test_fifo_capacity_one_grants_next_waiter_on_release():
    async def _run():
        queue = get_llama_admission_queue("http://llama.test")
        config = LlamaAdmissionConfig()

        first = queue.reserve(capacity = 1, config = config)
        second = queue.reserve(capacity = 1, config = config)
        third = queue.reserve(capacity = 1, config = config)

        first_lease = first.lease_nowait()
        assert first_lease is not None
        assert second.lease_nowait() is None
        assert third.lease_nowait() is None
        assert queue.snapshot().queued == 2

        first_lease.release()
        second_lease = await second.wait(0.1)
        assert second_lease is not None
        assert third.lease_nowait() is None

        second_lease.release()
        third_lease = await third.wait(0.1)
        assert third_lease is not None
        third_lease.release()

        snapshot = queue.snapshot()
        assert snapshot.active == 0
        assert snapshot.queued == 0

    asyncio.run(_run())


def test_pool_hands_out_distinct_slots_and_reuses_them():
    async def _run():
        queue = get_llama_admission_queue("http://llama.test")
        config = LlamaAdmissionConfig()

        leases = [queue.reserve(capacity = 3, config = config).lease_nowait() for _ in range(3)]
        assert sorted(lease.slot for lease in leases) == [0, 1, 2]  # one slot each
        snapshot = queue.snapshot()
        assert (snapshot.active, snapshot.free, snapshot.capacity) == (3, 0, 3)

        # A freed slot returns to the pool and is handed to the next caller.
        freed = leases[1].slot
        leases[1].release()
        assert queue.snapshot().free == 1
        reused = queue.reserve(capacity = 3, config = config).lease_nowait()
        assert reused.slot == freed

        reused.release()
        leases[0].release()
        leases[2].release()
        snapshot = queue.snapshot()
        assert (snapshot.active, snapshot.free) == (0, 3)

    asyncio.run(_run())


def test_pool_waiter_is_handed_a_real_slot():
    async def _run():
        queue = get_llama_admission_queue("http://llama.test")
        config = LlamaAdmissionConfig()

        held = queue.reserve(capacity = 1, config = config).lease_nowait()
        waiting = queue.reserve(capacity = 1, config = config)
        assert waiting.lease_nowait() is None
        assert queue.snapshot().free == 0

        held.release()
        granted = await waiting.wait(0.1)
        assert granted is not None and granted.slot == 0  # the slot just freed
        granted.release()

    asyncio.run(_run())


def test_shrinking_capacity_retires_slots_beyond_the_new_pool():
    async def _run():
        queue = get_llama_admission_queue("http://llama.test")
        config = LlamaAdmissionConfig()

        leases = [queue.reserve(capacity = 4, config = config).lease_nowait() for _ in range(4)]
        assert queue.snapshot().capacity == 4

        # llama-server reloaded with fewer --parallel slots; in-flight holders keep
        # running and their slots retire instead of returning to the smaller pool.
        shrunk = queue.reserve(capacity = 2, config = config)
        assert shrunk.lease_nowait() is None  # all 4 still held, nothing free
        for lease in leases:
            lease.release()

        granted = await shrunk.wait(0.1)
        assert granted is not None and granted.slot < 2
        granted.release()
        snapshot = queue.snapshot()
        assert (snapshot.capacity, snapshot.active, snapshot.free) == (2, 0, 2)

    asyncio.run(_run())


def test_queue_limit_scales_with_the_serving_slots():
    # The wait line follows --parallel: 16 per slot, floored at 64 so a 1-slot
    # backend keeps the depth it had before scaling existed.
    config = LlamaAdmissionConfig()
    assert config.queue_limit(4) == 64  # --parallel 4  (the default)
    assert config.queue_limit(8) == 128  # --parallel 8
    assert config.queue_limit(16) == 256
    assert config.queue_limit(1) == 64  # floor, not 16
    assert config.queue_limit(2) == 64  # floor, not 32
    # An explicit cap wins, and a None multiplier means an unbounded line.
    assert LlamaAdmissionConfig(max_queue = 5).queue_limit(8) == 5
    assert LlamaAdmissionConfig(queue_per_slot = None).queue_limit(8) is None
    # Non-positive settings mean unbounded, never "reject everything".
    assert LlamaAdmissionConfig(max_queue = 0).queue_limit(4) is None
    assert LlamaAdmissionConfig(max_queue = -1).queue_limit(4) is None
    assert LlamaAdmissionConfig(queue_per_slot = 0).queue_limit(4) is None
    assert LlamaAdmissionConfig(queue_per_slot = -3).queue_limit(4) is None


def test_queue_limit_rejects_only_once_the_line_is_full():
    async def _run():
        queue = get_llama_admission_queue("http://llama.test")
        # Explicit cap, so the test drives rejection without standing up the 64
        # waiters the scaled floor would otherwise require.
        config = LlamaAdmissionConfig(max_queue = 4)

        held = [queue.reserve(capacity = 2, config = config).lease_nowait() for _ in range(2)]
        parked = [queue.reserve(capacity = 2, config = config) for _ in range(4)]
        assert queue.snapshot().queued == 4

        with pytest.raises(LlamaAdmissionQueueFull):
            queue.reserve(capacity = 2, config = config)

        for reservation in parked:
            reservation.cancel()
        for lease in held:
            lease.release()

    asyncio.run(_run())


def test_waiting_is_never_timed_out_by_default():
    # "Wait forever": the default config sets no queue timeout at all.
    assert llama_admission_config_from_env().queue_timeout_s is None
    assert LlamaAdmissionConfig().queue_timeout_s is None


def test_single_request_at_a_time_never_queues_or_allocates_waiters():
    # The common serving case: one request in flight at a time must take a slot
    # straight away and never touch the wait line.
    async def _run():
        queue = get_llama_admission_queue("http://llama.test")
        config = LlamaAdmissionConfig()
        for _ in range(50):
            reservation = queue.reserve(capacity = 4, config = config)
            lease = reservation.lease_nowait()
            assert lease is not None  # admitted immediately
            assert queue.snapshot().queued == 0  # nobody ever lined up
            lease.release()
        snapshot = queue.snapshot()
        assert (snapshot.active, snapshot.free, snapshot.queued) == (0, 4, 0)

    asyncio.run(_run())


def test_unbounded_queue_keeps_waiting_instead_of_rejecting():
    # queue_per_slot None is the "pool + unbounded wait line" mode: nothing is
    # ever rejected, callers just line up for the next free slot.
    async def _run():
        queue = get_llama_admission_queue("http://llama.test")
        config = LlamaAdmissionConfig(max_queue = None, queue_per_slot = None)

        held = queue.reserve(capacity = 1, config = config).lease_nowait()
        waiters = [queue.reserve(capacity = 1, config = config) for _ in range(200)]
        assert queue.snapshot().queued == 200  # no LlamaAdmissionQueueFull

        held.release()
        first = await waiters[0].wait(0.1)
        assert first is not None
        first.release()
        for waiter in waiters[1:]:
            waiter.cancel()

    asyncio.run(_run())


def test_queue_full_rejects_excess_waiter():
    async def _run():
        queue = get_llama_admission_queue("http://llama.test")
        config = LlamaAdmissionConfig(max_queue = 1)

        first = queue.reserve(capacity = 1, config = config)
        queued = queue.reserve(capacity = 1, config = config)

        assert first.lease_nowait() is not None
        assert queued.lease_nowait() is None
        with pytest.raises(LlamaAdmissionQueueFull):
            queue.reserve(capacity = 1, config = config)

    asyncio.run(_run())


def test_disabled_admission_bypasses_active_slot_limit():
    async def _run():
        queue = get_llama_admission_queue("http://llama.test")
        config = LlamaAdmissionConfig(enabled = False)

        first = queue.reserve(capacity = 1, config = config)
        second = queue.reserve(capacity = 1, config = config)

        assert first.lease_nowait() is not None
        assert second.lease_nowait() is not None
        assert queue.snapshot().active == 0
        assert queue.snapshot().queued == 0

    asyncio.run(_run())


def test_cancelling_promoted_waiter_releases_slot():
    async def _run():
        queue = get_llama_admission_queue("http://llama.test")
        config = LlamaAdmissionConfig()

        first = queue.reserve(capacity = 1, config = config)
        second = queue.reserve(capacity = 1, config = config)
        first_lease = first.lease_nowait()

        first_lease.release()
        await asyncio.sleep(0)
        second.cancel()

        snapshot = queue.snapshot()
        assert snapshot.active == 0
        assert snapshot.queued == 0

    asyncio.run(_run())


def test_cancelling_promoted_waiter_before_delivery_releases_slot():
    async def _run():
        queue = get_llama_admission_queue("http://llama.test")
        config = LlamaAdmissionConfig()

        first = queue.reserve(capacity = 1, config = config)
        second = queue.reserve(capacity = 1, config = config)
        first_lease = first.lease_nowait()

        first_lease.release()
        second.cancel()

        snapshot = queue.snapshot()
        assert snapshot.active == 0
        assert snapshot.queued == 0

    asyncio.run(_run())


def test_external_waiter_future_cancel_invalidates_reservation():
    async def _run():
        queue = get_llama_admission_queue("http://llama.test")
        config = LlamaAdmissionConfig()

        first = queue.reserve(capacity = 1, config = config)
        second = queue.reserve(capacity = 1, config = config)
        first_lease = first.lease_nowait()
        assert first_lease is not None
        assert second._waiter is not None

        second._waiter.future.cancel()

        assert second.lease_nowait() is None
        assert second.is_cancelled is True
        assert await second.wait(0.01) is None

        first_lease.release()
        snapshot = queue.snapshot()
        assert snapshot.active == 0
        assert snapshot.queued == 0

    asyncio.run(_run())


def test_wait_returns_none_when_waiter_future_cancelled_during_wait():
    async def _run():
        queue = get_llama_admission_queue("http://llama.test")
        config = LlamaAdmissionConfig()

        first = queue.reserve(capacity = 1, config = config)
        second = queue.reserve(capacity = 1, config = config)
        first_lease = first.lease_nowait()
        assert first_lease is not None
        assert second._waiter is not None

        wait_task = asyncio.create_task(second.wait(1.0))
        await asyncio.sleep(0)
        second._waiter.future.cancel()

        assert await asyncio.wait_for(wait_task, timeout = 0.1) is None
        assert second.is_cancelled is True

        first_lease.release()
        snapshot = queue.snapshot()
        assert snapshot.active == 0
        assert snapshot.queued == 0

    asyncio.run(_run())


def test_capacity_increase_promotes_existing_waiter_fifo():
    async def _run():
        queue = get_llama_admission_queue("http://llama.test")
        config = LlamaAdmissionConfig()

        first = queue.reserve(capacity = 1, config = config)
        second = queue.reserve(capacity = 1, config = config)

        first_lease = first.lease_nowait()
        assert first_lease is not None
        assert second.lease_nowait() is None
        assert queue.snapshot().active == 1
        assert queue.snapshot().queued == 1

        third = queue.reserve(capacity = 2, config = config)

        second_lease = await second.wait(0.1)
        assert second_lease is not None
        assert third.lease_nowait() is None

        snapshot = queue.snapshot()
        assert snapshot.capacity == 2
        assert snapshot.active == 2
        assert snapshot.queued == 1

        first_lease.release()
        third_lease = await third.wait(0.1)
        assert third_lease is not None

        second_lease.release()
        third_lease.release()
        snapshot = queue.snapshot()
        assert snapshot.active == 0
        assert snapshot.queued == 0

    asyncio.run(_run())


def test_lease_release_is_idempotent_under_concurrent_calls():
    async def _run():
        queue = get_llama_admission_queue("http://llama.test")
        config = LlamaAdmissionConfig()

        reservation = queue.reserve(capacity = 1, config = config)
        lease = reservation.lease_nowait()
        assert lease is not None

        threads = [threading.Thread(target = lease.release) for _ in range(16)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        snapshot = queue.snapshot()
        assert snapshot.active == 0
        assert snapshot.queued == 0

    asyncio.run(_run())


def test_releasing_a_stale_lease_does_not_free_someone_elses_slot():
    # The concurrent test above passes without the _released guard: the racing
    # calls all target a still-live slot, which the bitmask already absorbs. The
    # case the guard exists for is a slot released twice with a reuse in between.
    # It is live: _wait_for_openai_admission_non_streaming releases and re-raises,
    # then the caller's finally cancels the reservation and releases the same
    # lease again, by which point the slot can belong to another request.
    async def _run():
        queue = get_llama_admission_queue("http://llama.test")
        config = LlamaAdmissionConfig()

        stale = queue.reserve(capacity = 1, config = config).lease_nowait()
        stale.release()
        other = queue.reserve(capacity = 1, config = config).lease_nowait()
        assert other.slot == stale.slot  # the slot got reused

        stale.release()
        assert queue.snapshot().active == 1, "stale release handed back a live slot"
        other.release()
        assert queue.snapshot().active == 0

    asyncio.run(_run())


def test_grant_reclaims_the_slot_when_the_waiters_loop_is_gone():
    # _grant_waiters_locked takes the slot before scheduling delivery, so if the
    # schedule fails the bit is already set. Leaving it set strands the slot for
    # good, because _free is rebuilt from the bitmask.
    queue = get_llama_admission_queue("http://llama.test")
    config = LlamaAdmissionConfig()
    held = None

    dead = asyncio.new_event_loop()
    try:

        async def _fill_and_queue():
            nonlocal held
            held = queue.reserve(capacity = 1, config = config).lease_nowait()
            assert queue.reserve(capacity = 1, config = config).lease_nowait() is None

        dead.run_until_complete(_fill_and_queue())
    finally:
        dead.close()

    held.release()  # grant path now hits the closed loop
    assert queue.snapshot().active == 0
    assert queue.is_idle()


def test_cancel_returns_the_granted_slot_when_the_waiters_loop_is_gone():
    # Routes cancel() from finally blocks, so a raise here would mask their
    # exception and skip the release that hands the granted slot back.
    queue = get_llama_admission_queue("http://llama.test")
    config = LlamaAdmissionConfig()
    held = reservation = None

    dead = asyncio.new_event_loop()
    try:

        async def _fill_and_queue():
            nonlocal held, reservation
            held = queue.reserve(capacity = 1, config = config).lease_nowait()
            reservation = queue.reserve(capacity = 1, config = config)

        dead.run_until_complete(_fill_and_queue())
        held.release()  # promotes the waiter, so cancel() has a lease to return
    finally:
        dead.close()

    reservation.cancel()
    assert queue.snapshot().active == 0
    assert queue.is_idle()


def test_delivery_to_an_already_finished_waiter_releases_the_slot():
    # A slot is taken before delivery is scheduled, so if the waiter finishes in
    # that window someone has to hand it back. _deliver_lease does it twice over,
    # in the dead-waiter branch and in the InvalidStateError backstop; this pins
    # the outcome, not which one. Reaches into the waiter because no public call
    # leaves that window open: queue.cancel() reclaims granted_lease itself.
    async def _run():
        queue = get_llama_admission_queue("http://llama.test")
        config = LlamaAdmissionConfig()

        held = queue.reserve(capacity = 1, config = config).lease_nowait()
        reservation = queue.reserve(capacity = 1, config = config)
        waiter = reservation._waiter

        held.release()  # schedules _deliver_lease, sets granted_lease
        waiter.future.cancel()  # finishes the future before the callback runs
        assert waiter.granted_lease is not None
        await asyncio.sleep(0)  # let the callback run

        assert queue.snapshot().active == 0
        assert queue.is_idle()

    asyncio.run(_run())


def test_new_key_evicts_idle_prior_load_queues():
    # Each model load carries a fresh ephemeral port, so a new base_url key must
    # not leave the drained queues from earlier loads accumulating forever.
    get_llama_admission_queue("http://127.0.0.1:1001")
    get_llama_admission_queue("http://127.0.0.1:1002")
    assert set(llama_admission._QUEUES) == {"http://127.0.0.1:1002"}

    get_llama_admission_queue("http://127.0.0.1:1003")
    assert set(llama_admission._QUEUES) == {"http://127.0.0.1:1003"}


def test_new_key_retains_in_flight_prior_load_queue():
    config = LlamaAdmissionConfig()
    busy = get_llama_admission_queue("http://127.0.0.1:2001")

    async def _run():
        reservation = busy.reserve(capacity = 1, config = config)
        lease = reservation.lease_nowait()
        assert lease is not None

        # A new load must not drop a queue that still has an in-flight request.
        get_llama_admission_queue("http://127.0.0.1:2002")
        assert set(llama_admission._QUEUES) == {"http://127.0.0.1:2001", "http://127.0.0.1:2002"}

        # Once it drains, the next load reclaims it.
        lease.release()
        get_llama_admission_queue("http://127.0.0.1:2003")
        assert set(llama_admission._QUEUES) == {"http://127.0.0.1:2003"}

    asyncio.run(_run())


def test_capacity_shrink_never_admits_past_the_new_ceiling():
    # A load that downshifts --parallel (or an unload resetting it to 1) shrinks the
    # pool while slots are still held. Those holdovers keep occupying the backend, so
    # they must count against the ceiling; sizing on free ids alone over-admits.
    async def _run():
        queue = get_llama_admission_queue("http://llama.test")
        config = LlamaAdmissionConfig()

        held = [queue.reserve(capacity = 4, config = config).lease_nowait() for _ in range(4)]
        assert all(lease is not None for lease in held)
        waiter = queue.reserve(capacity = 4, config = config)

        queue.reserve(capacity = 1, config = config)  # capacity collapses to 1
        # Release the one id that still falls inside the shrunk pool, so it goes
        # back on the free list; ids at or above capacity retire instead.
        low = min(held, key = lambda lease: lease.slot)
        assert low.slot == 0
        low.release()

        # The other 3 holdovers are still generating, which already meets the new
        # ceiling, so the freed id must not be handed on. Gating on "is an id free"
        # alone grants it here and puts 4 generations on a 1-slot backend.
        with pytest.raises(asyncio.TimeoutError):
            await waiter.wait(0.2)
        assert queue.snapshot().active == 3

        waiter.cancel()
        for lease in held:
            if lease is not low:
                lease.release()

    asyncio.run(_run())


def test_queue_per_slot_env_is_parsed(monkeypatch):
    monkeypatch.setenv(ADMISSION_QUEUE_PER_SLOT_ENV, "4")
    assert llama_admission_config_from_env().queue_limit(32) == 128
    # Non-positive asks for an unbounded line rather than rejecting everything.
    monkeypatch.setenv(ADMISSION_QUEUE_PER_SLOT_ENV, "0")
    assert llama_admission_config_from_env().queue_limit(32) is None


def test_max_queue_zero_from_env_is_unbounded_end_to_end(monkeypatch):
    # Guards the whole env path, not just the parsed field: a regression that let
    # queue_per_slot survive MAX_QUEUE=0 would silently re-bound the line.
    monkeypatch.setenv(ADMISSION_MAX_QUEUE_ENV, "0")
    config = llama_admission_config_from_env()
    assert config.max_queue is None and config.queue_per_slot is None
    assert config.queue_limit(1) is None and config.queue_limit(64) is None


def test_legacy_env_fallback_covers_every_setting(monkeypatch):
    for canonical, legacy in llama_admission._LEGACY_ENV.items():
        monkeypatch.delenv(canonical, raising = False)
        monkeypatch.setenv(legacy, "0" if "CONTROL" in canonical else "7")
    config = llama_admission_config_from_env()
    assert config.enabled is False
    assert config.queue_timeout_s == 7.0
    assert config.keepalive_interval_s == 7.0
    assert config.max_queue == 7


def test_empty_canonical_env_falls_through_to_legacy(monkeypatch):
    # The branch _raw_env exists for: set but blank must not mask the legacy name.
    monkeypatch.setenv(ADMISSION_CONTROL_ENV, "   ")
    monkeypatch.setenv(llama_admission._LEGACY_ENV[ADMISSION_CONTROL_ENV], "0")
    assert llama_admission_config_from_env().enabled is False


def test_explicit_queue_per_slot_is_not_floored(monkeypatch):
    # The floor exists so a 1-slot backend keeps its old depth by default, not to
    # override an operator who asked for a shallow line.
    monkeypatch.setenv(ADMISSION_QUEUE_PER_SLOT_ENV, "2")
    config = llama_admission_config_from_env()
    assert config.queue_limit(1) == 2
    assert config.queue_limit(8) == 16

    # Unset, the default multiplier is floored instead.
    monkeypatch.delenv(ADMISSION_QUEUE_PER_SLOT_ENV, raising = False)
    assert llama_admission_config_from_env().queue_limit(1) == 64

    # A value that does not parse falls back to the default multiplier, so it has
    # to keep the default's floor. Otherwise a typo quietly shrinks the line 4x.
    for garbage in ("abc", "1e3", "16.0"):
        monkeypatch.setenv(ADMISSION_QUEUE_PER_SLOT_ENV, garbage)
        assert llama_admission_config_from_env().queue_limit(1) == 64, garbage


def test_module_imports_on_python_39(monkeypatch):
    """No 3.10+ API on an import path. The package declares >=3.9 but CI only
    runs 3.12, so a regression here would ship broken."""
    import ast
    import pathlib

    src = pathlib.Path(llama_admission.__file__).read_text(encoding = "utf-8")
    tree = ast.parse(src)

    # int.bit_count() (3.10+)
    assert not [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == "bit_count"
    ]
    # dataclass(slots = ...) is 3.10+, so every dataclass must take it through
    # the version gate instead of naming it. A new one that forgets the gate
    # loses slots silently, so require the **_SLOTS unpack rather than allow it.
    seen = 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
        if name != "dataclass":
            continue
        seen += 1
        assert "slots" not in {kw.arg for kw in node.keywords}
        assert [
            kw
            for kw in node.keywords
            if kw.arg is None and getattr(kw.value, "id", None) == "_SLOTS"
        ], ast.dump(node)
    assert seen


def test_slots_gate_matches_the_running_interpreter():
    """The gate is only worth having if it actually applies where it can."""
    import sys

    gated = (LlamaAdmissionConfig, llama_admission.LlamaAdmissionSnapshot, llama_admission._Waiter)
    if sys.version_info >= (3, 10):
        assert llama_admission._SLOTS == {"slots": True}
        for cls in gated:
            assert getattr(cls, "__slots__", None), cls
    else:
        assert llama_admission._SLOTS == {}

    # Construct through the gate either way: slots=True rebuilds the class, so a
    # field it cannot carry over would only show up on instantiation.
    config = LlamaAdmissionConfig(max_queue = 7)
    assert config.max_queue == 7 and config.queue_limit(4) == 7
    assert llama_admission.LlamaAdmissionSnapshot("k", 1, 1, 0).capacity == 1


def test_held_count_tracks_the_bitmask():
    # _held replaces int.bit_count(); the two must never drift apart.
    async def _run():
        queue = get_llama_admission_queue("http://llama.test")
        config = LlamaAdmissionConfig()
        popcount = lambda: bin(queue._in_use).count("1")

        leases = [queue.reserve(capacity = 4, config = config).lease_nowait() for _ in range(4)]
        assert queue._held == popcount() == 4
        leases[1].release()
        assert queue._held == popcount() == 3
        shrunk = queue.reserve(capacity = 2, config = config)  # shrink with slots held
        assert queue._held == popcount() == 3
        shrunk.cancel()  # else it is granted a slot as the others drain
        for lease in leases:
            lease.release()
        assert queue._held == popcount() == 0

    asyncio.run(_run())


def test_snapshot_free_never_exceeds_what_can_be_admitted():
    # After a shrink, low ids can sit in _free while holdovers fill the ceiling.
    # Reporting them as free made the admission log contradict itself.
    async def _run():
        queue = get_llama_admission_queue("http://llama.test")
        config = LlamaAdmissionConfig()

        held = [queue.reserve(capacity = 4, config = config).lease_nowait() for _ in range(4)]
        queue.reserve(capacity = 1, config = config)  # capacity collapses to 1
        min(held, key = lambda lease: lease.slot).release()

        snapshot = queue.snapshot()
        assert snapshot.free == 0, snapshot  # nothing is actually takeable
        assert snapshot.active == 3
        for lease in held:
            lease.release()

    asyncio.run(_run())


def test_a_newcomer_does_not_barge_past_a_parked_waiter():
    # Anti-starvation, pinned as behaviour rather than as the `if not self._waiters`
    # check: _take_slot_locked consults _can_admit_locked anyway, so either alone
    # refuses the newcomer. This fails if both ever go.
    async def _run():
        queue = get_llama_admission_queue("http://llama.test")
        config = LlamaAdmissionConfig()

        held = queue.reserve(capacity = 1, config = config).lease_nowait()
        parked = queue.reserve(capacity = 1, config = config)
        assert parked.lease_nowait() is None

        held.release()
        newcomer = queue.reserve(capacity = 1, config = config)
        assert newcomer.lease_nowait() is None, "newcomer barged past the parked waiter"
        assert (await parked.wait(0.1)) is not None

    asyncio.run(_run())


def test_dead_waiters_stop_counting_against_the_queue_limit():
    # A future cancelled out of band leaves the entry in the deque: cancel() is not
    # called, so only the prune drops it. Without that, depth, is_idle() and the
    # queue-full limit all drift for the life of the queue.
    async def _run():
        queue = get_llama_admission_queue("http://llama.test")
        config = LlamaAdmissionConfig(max_queue = 2)

        held = queue.reserve(capacity = 1, config = config).lease_nowait()
        first = queue.reserve(capacity = 1, config = config)
        second = queue.reserve(capacity = 1, config = config)
        assert queue.snapshot().queued == 2
        with pytest.raises(LlamaAdmissionQueueFull):
            queue.reserve(capacity = 1, config = config)

        first._waiter.future.cancel()
        second._waiter.future.cancel()
        assert queue.snapshot().queued == 0, "dead waiters still occupy the line"
        # The freed depth is usable again, and an idle queue is evictable.
        queue.reserve(capacity = 1, config = config).cancel()
        held.release()
        assert queue.is_idle()

    asyncio.run(_run())


def test_parking_frees_the_slot_for_a_waiter():
    """A holder waiting on a tool approval must not hold a decode slot.

    It is not generating, and with several prompts unanswered every slot would
    be held by a run parked on a human while llama-server sits idle.
    """

    async def _run():
        queue = get_llama_admission_queue("http://llama.test")
        config = LlamaAdmissionConfig()

        first = queue.reserve(capacity = 1, config = config)
        second = queue.reserve(capacity = 1, config = config)
        first_lease = first.lease_nowait()
        assert first_lease is not None
        assert second.lease_nowait() is None

        first_lease.park()
        assert first_lease.slot is None, "the slot went back to the pool"
        second_lease = await second.wait(0.1)
        assert second_lease is not None, "parking did not free the slot"

        # The parked holder keeps its lease, so releasing it is still correct.
        first_lease.unpark()
        first_lease.release()
        second_lease.release()
        assert queue.snapshot().active == 0

    asyncio.run(_run())


def test_unpark_without_park_is_a_no_op():
    async def _run():
        queue = get_llama_admission_queue("http://llama.test")
        config = LlamaAdmissionConfig()

        first = queue.reserve(capacity = 1, config = config)
        first_lease = first.lease_nowait()
        assert first_lease is not None
        first_lease.unpark()
        first_lease.unpark()

        second = queue.reserve(capacity = 1, config = config)
        assert second.lease_nowait() is None, "capacity leaked past the limit"

    asyncio.run(_run())


def test_releasing_a_parked_lease_leaves_the_queue_evictable():
    # is_idle() drives registry eviction, and a parked holder owns no slot, so
    # nothing but the parked count keeps its queue alive. A stuck count would
    # pin every dead queue for the life of the process.
    async def _run():
        queue = get_llama_admission_queue("http://llama.test")
        config = LlamaAdmissionConfig()

        lease = queue.reserve(capacity = 1, config = config).lease_nowait()
        lease.park()
        assert not queue.is_idle(), "a parked holder is coming back to this queue"
        lease.release()
        assert queue.is_idle()

    asyncio.run(_run())


def test_unpark_waits_instead_of_putting_two_holders_on_one_slot():
    # park() hands the freed slot to a waiter, so by the time the user answers an approval
    # prompt someone else may be decoding in it. Resuming regardless left two holders
    # against capacity 1, and the resumed tool loop went past the admission limit.
    async def scenario():
        queue = get_llama_admission_queue("http://llama.test")
        config = LlamaAdmissionConfig()

        a = queue.reserve(capacity = 1, config = config)
        a_lease = a.lease_nowait()
        assert a_lease is not None, "A takes the only slot"
        b = queue.reserve(capacity = 1, config = config)
        assert b.lease_nowait() is None, "B waits behind A"

        a_lease.park()  # A parks on an approval prompt; its slot goes to B
        b_lease = await asyncio.wait_for(b.wait(timeout_s = 1), timeout = 2)
        assert b_lease is not None, "B was granted the parked slot"

        # A answers the prompt while B is still decoding: it must WAIT.
        resumed = asyncio.ensure_future(a_lease.unpark_async(poll_s = 0.01))
        await asyncio.sleep(0.05)
        assert not resumed.done(), "A must not resume while B holds the slot"
        assert queue.snapshot().active <= 1, "never over capacity while waiting"

        b_lease.release()
        await asyncio.wait_for(resumed, timeout = 2)
        assert a_lease.slot is not None, "A took a real slot back"
        assert queue.snapshot().active <= 1, "still within capacity after resuming"

    asyncio.run(scenario())


def test_unpark_gives_up_when_the_caller_is_cancelled():
    # A holder being torn down must not sit in the wait loop.
    async def scenario():
        queue = get_llama_admission_queue("http://llama.test")
        config = LlamaAdmissionConfig()
        a = queue.reserve(capacity = 1, config = config)
        a_lease = a.lease_nowait()
        assert a_lease is not None
        b = queue.reserve(capacity = 1, config = config)
        a_lease.park()
        assert await asyncio.wait_for(b.wait(timeout_s = 1), timeout = 2) is not None

        ev = threading.Event()
        waiting = asyncio.ensure_future(a_lease.unpark_async(cancel_event = ev, poll_s = 0.01))
        await asyncio.sleep(0.03)
        assert not waiting.done()
        ev.set()
        await asyncio.wait_for(waiting, timeout = 2)
        assert a_lease.slot is None, "gave up without a slot rather than over-admitting"

    asyncio.run(scenario())


def test_an_approved_chat_is_not_overtaken_by_later_arrivals():
    # A parks on an approval prompt, B takes the slot, C arrives afterwards. release() grants
    # under the same lock, so a plain poll in unpark_async never saw a free slot: A waited
    # behind every later arrival and starved.
    async def scenario():
        queue = get_llama_admission_queue("http://llama.test")
        config = LlamaAdmissionConfig()

        a = queue.reserve(capacity = 1, config = config)
        a_lease = a.lease_nowait()
        assert a_lease is not None
        b = queue.reserve(capacity = 1, config = config)
        a_lease.park()  # A's slot goes to B
        b_lease = await asyncio.wait_for(b.wait(timeout_s = 1), timeout = 2)
        assert b_lease is not None

        # A is approved and starts waiting; C arrives only after that.
        resumed = asyncio.ensure_future(a_lease.unpark_async(poll_s = 0.01))
        await asyncio.sleep(0.03)
        c = queue.reserve(capacity = 1, config = config)
        assert c.lease_nowait() is None

        b_lease.release()  # the slot frees exactly once
        await asyncio.wait_for(resumed, timeout = 2)
        # A resumed; C is still queued behind it rather than having overtaken it.
        assert c.lease_nowait() is None
        assert queue.snapshot().active <= 1

    asyncio.run(scenario())


def test_two_approved_chats_do_not_block_each_other():
    # A bare pending-count made every approved holder count against every other: park A, admit
    # and park B, admit C, approve both, and once C released the predicate stayed false forever.
    async def scenario():
        queue = get_llama_admission_queue("http://llama.test")
        config = LlamaAdmissionConfig()

        a = queue.reserve(capacity = 1, config = config)
        a_lease = a.lease_nowait()
        assert a_lease is not None
        b = queue.reserve(capacity = 1, config = config)
        a_lease.park()  # A parks; B is admitted
        b_lease = await asyncio.wait_for(b.wait(timeout_s = 1), timeout = 2)
        assert b_lease is not None

        c = queue.reserve(capacity = 1, config = config)
        b_lease.park()  # B parks too; C is admitted
        c_lease = await asyncio.wait_for(c.wait(timeout_s = 1), timeout = 2)
        assert c_lease is not None

        # Both approvals come back while C is still decoding.
        first = asyncio.ensure_future(a_lease.unpark_async(poll_s = 0.01))
        await asyncio.sleep(0.02)
        second = asyncio.ensure_future(b_lease.unpark_async(poll_s = 0.01))
        await asyncio.sleep(0.02)
        assert not first.done() and not second.done()

        c_lease.release()
        # The earlier approval goes first; the other follows once it releases.
        await asyncio.wait_for(first, timeout = 2)
        assert not second.done(), "the second approval waits its turn, not forever"
        a_lease.release()
        await asyncio.wait_for(second, timeout = 2)
        assert queue.snapshot().active <= 1

    asyncio.run(scenario())


def test_an_immediate_arrival_cannot_take_an_approved_chats_slot():
    # The fairness reservation lived only in _grant_waiters_locked. reserve()'s fast path
    # ignored it, so a request arriving in the window between the slot freeing and the
    # approved chat's next poll took the slot straight off the top.
    async def scenario():
        queue = get_llama_admission_queue("http://llama.test")
        config = LlamaAdmissionConfig()

        a = queue.reserve(capacity = 1, config = config)
        a_lease = a.lease_nowait()
        assert a_lease is not None
        a_lease.park()  # A is on an approval prompt; its slot is up for grabs
        b = queue.reserve(capacity = 1, config = config)
        b_lease = b.lease_nowait()
        assert b_lease is not None

        resumed = asyncio.ensure_future(a_lease.unpark_async(poll_s = 0.01))
        await asyncio.sleep(0.03)  # A is approved and now holds a ticket

        # No await between these two: C arrives before A's poll can run again.
        b_lease.release()
        c = queue.reserve(capacity = 1, config = config)
        assert c.lease_nowait() is None, "the freed slot is reserved for the approved chat"

        await asyncio.wait_for(resumed, timeout = 2)
        assert queue.snapshot().active <= 1

    asyncio.run(scenario())


def test_parking_is_bounded_so_the_thread_pool_cannot_be_drained(monkeypatch):
    # A pending prompt parks an executor thread (the loop blocks inside
    # to_thread(next, gen)) and frees a slot that admits another run which can
    # park too, so unbounded parking drains the pool the generators run on.
    # Pinned because the real budget follows the runner's usable CPUs.
    monkeypatch.setattr(llama_admission, "_executor_workers", lambda: 32)

    async def scenario():
        queue = get_llama_admission_queue("http://llama.test")
        config = LlamaAdmissionConfig()
        limit = llama_admission._max_parked(1)
        assert limit >= 1

        leases = []
        for _ in range(limit):
            lease = queue.reserve(capacity = 1, config = config).lease_nowait()
            assert lease is not None and lease.park()
            leases.append(lease)

        refused = queue.reserve(capacity = 1, config = config).lease_nowait()
        assert refused is not None
        assert not refused.park(), "parking is unbounded"
        # Refusing means keeping the slot, the old behaviour, not an error.
        assert refused.slot is not None
        assert queue.snapshot().active == 1

        leases[0].unpark()
        assert refused.park(), "budget was not returned"
        for lease in leases[1:] + [refused]:
            lease.release()
        leases[0].release()

    asyncio.run(scenario())


def test_the_park_budget_is_shared_by_every_queue(monkeypatch):
    # One executor, so a per-queue budget would be handed out again to every
    # backend and to every reload onto a fresh ephemeral port.
    monkeypatch.setattr(llama_admission, "_executor_workers", lambda: 32)

    async def scenario():
        config = LlamaAdmissionConfig()
        first = get_llama_admission_queue("http://llama.test:1")
        second = get_llama_admission_queue("http://llama.test:2")
        limit = llama_admission._max_parked(1)

        for index in range(limit):
            queue = first if index % 2 == 0 else second
            lease = queue.reserve(capacity = 1, config = config).lease_nowait()
            assert lease.park()

        spare = second.reserve(capacity = 1, config = config).lease_nowait()
        assert not spare.park(), "each queue got its own budget"

        # A reset drops the queues the count was claimed against, so it must drop
        # the count too or the leak shrinks the budget process-wide.
        reset_llama_admission_queues()
        revived = get_llama_admission_queue("http://llama.test:1")
        fresh = revived.reserve(capacity = 1, config = config).lease_nowait()
        assert fresh.park(), "reset leaked the park count"
        fresh.release()

    asyncio.run(scenario())


def test_the_park_budget_leaves_the_executor_room_to_work(monkeypatch):
    # The pool already permits `capacity` pending prompts and every park admits
    # one more, so the budget must account for both. Swept across executor sizes
    # rather than read off this host, since a container gets a small one.
    for cpus in (1, 2, 4, 8, 16, 28, 64):
        workers = min(32, cpus + 4)
        monkeypatch.setattr(llama_admission, "_executor_workers", lambda w = workers: w)
        reserve = llama_admission._executor_reserve(workers)
        assert reserve >= 2, f"{workers} workers left no reserve"

        # Even the smallest executor fits the two simultaneous prompts #7455 needs.
        assert llama_admission._max_parked(1) >= 2, f"no room for two on {workers} workers"
        assert llama_admission._max_parked(1) <= workers // 2
        # A backend whose --parallel alone fills the executor gets no parks.
        assert llama_admission._max_parked(workers) == 0
        for capacity in range(0, workers + 8):
            budget = llama_admission._max_parked(capacity)
            assert budget >= 0, f"negative budget at capacity {capacity}"
            assert (
                budget == 0 or capacity + budget <= workers - reserve
            ), f"{workers} workers: capacity {capacity} plus {budget} parks leaves no room"


def test_the_park_budget_follows_the_executors_own_cpu_count(monkeypatch):
    # 3.13 sizes ThreadPoolExecutor from process_cpu_count(), which honours CPU
    # affinity and cgroup quotas; cpu_count() would budget from the whole host
    # inside a one-core container. Pulled apart here, since they usually match.
    import concurrent.futures

    monkeypatch.setattr(os, "cpu_count", lambda: 64)
    if hasattr(os, "process_cpu_count"):
        monkeypatch.setattr(os, "process_cpu_count", lambda: 1)
    # Against the real thing rather than the formula: the default executor is a
    # plain ThreadPoolExecutor(), so its own sizing is the answer on any version.
    with concurrent.futures.ThreadPoolExecutor() as pool:
        assert llama_admission._executor_workers() == pool._max_workers


def test_the_stream_retries_a_park_that_was_refused():
    # _park_admission short-circuits on `on == _parked`, so recording a refused
    # park as parked would skip every later approval in the run even once the
    # budget frees up. Structural because that only shows on a second approval.
    import ast

    # Read rather than import: routes.inference pulls in the whole app.
    route = os.path.join(_backend, "routes", "inference.py")
    with open(route, encoding = "utf-8") as handle:
        tree = ast.parse(handle.read())
    helpers = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "_park_admission"
    ]
    assert len(helpers) == 1, f"expected one _park_admission, found {len(helpers)}"

    guards = [
        node
        for node in ast.walk(helpers[0])
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.UnaryOp)
        and isinstance(node.test.op, ast.Not)
        and isinstance(node.test.operand, ast.Call)
        and getattr(node.test.operand.func, "attr", None) == "park"
        and getattr(node.test.operand.func.value, "id", None) == "lease"
    ]
    assert len(guards) == 1, "lease.park()'s answer is ignored"
    assert all(
        isinstance(stmt, ast.Return) for stmt in guards[0].body
    ), "a refused park must leave _parked alone, so a later approval retries it"


def test_the_park_budget_counts_every_live_backend(monkeypatch):
    # base_url takes a fresh port on every load, so a reload mints a queue while
    # the old one drains. Prompts on both park threads of the one executor, so a
    # budget sized from either backend alone lets them add up past the reserve.
    monkeypatch.setattr(llama_admission, "_executor_workers", lambda: 32)

    async def scenario():
        config = LlamaAdmissionConfig()
        old = get_llama_admission_queue("http://llama.test:1")
        draining = old.reserve(capacity = 16, config = config).lease_nowait()
        assert draining is not None  # in flight, so the registry keeps this queue

        new = get_llama_admission_queue("http://llama.test:2")
        lease = new.reserve(capacity = 16, config = config).lease_nowait()
        assert lease is not None

        # 16 slots each against 32 workers: their prompts alone can fill it.
        assert llama_admission._max_parked(16) > 0, "this test needs a budget to remove"
        assert not lease.park(), "budget sized from one backend of two"

        draining.release()  # the old backend drains and is up for eviction
        assert lease.park(), "an idle backend still counted against the budget"
        lease.release()

    asyncio.run(scenario())


def test_the_park_budget_is_freed_when_the_prompt_is_answered(monkeypatch):
    # The executor thread comes back the moment the answer arrives, before the
    # resume queues for a slot. Holding the budget until the slot lands refuses
    # someone else's park, and that someone holds the slot the resumer wants.
    monkeypatch.setattr(llama_admission, "_executor_workers", lambda: 32)

    async def scenario():
        config = LlamaAdmissionConfig()
        queue = get_llama_admission_queue("http://llama.test")

        parked = []
        for _ in range(llama_admission._max_parked(1)):
            lease = queue.reserve(capacity = 1, config = config).lease_nowait()
            assert lease is not None and lease.park()
            parked.append(lease)

        blocked = queue.reserve(capacity = 1, config = config).lease_nowait()
        assert blocked is not None
        assert not blocked.park(), "the budget was not full to begin with"

        # One prompt is answered. Its slot is taken, so the resume queues for one.
        resumed = asyncio.ensure_future(parked[0].unpark_async(poll_s = 0.01))
        await asyncio.sleep(0.05)
        assert not resumed.done(), "the resume needs to still be waiting for its slot"

        assert blocked.park(), "budget held for a prompt wait that is over"
        # Which is what frees the slot the resumer was waiting for.
        await asyncio.wait_for(resumed, timeout = 2)
        for lease in parked[1:] + [blocked]:
            lease.release()
        parked[0].release()

    asyncio.run(scenario())


def test_releasing_a_parked_holder_returns_its_budget(monkeypatch):
    # A client that disconnects on the prompt releases straight out of parked,
    # never unparking. Its executor thread went with it, so keeping the budget
    # would lose one for the life of the process.
    monkeypatch.setattr(llama_admission, "_executor_workers", lambda: 32)

    async def scenario():
        config = LlamaAdmissionConfig()
        queue = get_llama_admission_queue("http://llama.test")

        parked = []
        for _ in range(llama_admission._max_parked(1)):
            lease = queue.reserve(capacity = 1, config = config).lease_nowait()
            assert lease is not None and lease.park()
            parked.append(lease)

        blocked = queue.reserve(capacity = 1, config = config).lease_nowait()
        assert blocked is not None
        assert not blocked.park(), "the budget was not full to begin with"

        parked[0].release()
        assert blocked.park(), "a released park never gave its budget back"
        for lease in parked[1:] + [blocked]:
            lease.release()

    asyncio.run(scenario())
