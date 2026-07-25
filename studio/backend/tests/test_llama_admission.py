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


@pytest.fixture(autouse = True)
def _reset_queues():
    reset_llama_admission_queues()
    yield
    reset_llama_admission_queues()


def test_admission_config_defaults(monkeypatch):
    for name in (
        ADMISSION_CONTROL_ENV,
        ADMISSION_QUEUE_TIMEOUT_ENV,
        ADMISSION_KEEPALIVE_INTERVAL_ENV,
        ADMISSION_MAX_QUEUE_ENV,
        "UNSLOTH_OPENAI_COMPAT_ADMISSION_CONTROL",
        "UNSLOTH_OPENAI_COMPAT_ADMISSION_QUEUE_TIMEOUT",
        "UNSLOTH_OPENAI_COMPAT_ADMISSION_KEEPALIVE_INTERVAL",
        "UNSLOTH_OPENAI_COMPAT_ADMISSION_MAX_QUEUE",
    ):
        monkeypatch.delenv(name, raising = False)

    config = llama_admission_config_from_env()

    assert config.enabled is True
    assert config.queue_timeout_s == DEFAULT_ADMISSION_QUEUE_TIMEOUT_S
    assert config.keepalive_interval_s == DEFAULT_ADMISSION_KEEPALIVE_INTERVAL_S
    assert config.max_queue == DEFAULT_ADMISSION_MAX_QUEUE


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
    # The wait line follows --parallel: 16 per slot by default.
    config = LlamaAdmissionConfig()
    assert config.queue_limit(4) == 64  # --parallel 4  (the default)
    assert config.queue_limit(8) == 128  # --parallel 8
    assert config.queue_limit(1) == 16
    # An explicit cap wins, and a None multiplier means an unbounded line.
    assert LlamaAdmissionConfig(max_queue = 5).queue_limit(8) == 5
    assert LlamaAdmissionConfig(queue_per_slot = None).queue_limit(8) is None


def test_scaled_queue_limit_rejects_only_past_slots_times_multiplier():
    async def _run():
        queue = get_llama_admission_queue("http://llama.test")
        config = LlamaAdmissionConfig(queue_per_slot = 2)  # capacity 2 -> 4 waiters

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
