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
        assert set(llama_admission._QUEUES) == {
            "http://127.0.0.1:2001",
            "http://127.0.0.1:2002",
        }

        # Once it drains, the next load reclaims it.
        lease.release()
        get_llama_admission_queue("http://127.0.0.1:2003")
        assert set(llama_admission._QUEUES) == {"http://127.0.0.1:2003"}

    asyncio.run(_run())
