# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A preempted holder owns no slot but is coming back to its queue, exactly as a parked one is.
Evicting that queue as idle when another model's queue is created hands later arrivals a fresh
pool while the paused chat resumes against the orphan: more leases than the capacity."""

from __future__ import annotations

import asyncio
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.inference.llama_admission import (
    LlamaAdmissionConfig,
    get_llama_admission_queue,
    reset_llama_admission_queues,
)

_KEY = "http://llama.preempted-lifetime.test:1"
_OTHER = "http://llama.preempted-lifetime.test:2"


@pytest.fixture(autouse = True)
def _reset():
    reset_llama_admission_queues()
    yield
    reset_llama_admission_queues()


def _two_holders_one_preempted():
    queue = get_llama_admission_queue(_KEY)
    config = LlamaAdmissionConfig()
    paused = queue.reserve(capacity = 2, config = config).lease_nowait()
    other = queue.reserve(capacity = 2, config = config).lease_nowait()
    assert paused is not None and other is not None
    assert paused.preempt()
    other.release()
    return queue, config, paused


def test_a_preempted_holder_keeps_the_queue_from_eviction():
    async def _run():
        queue, _config, paused = _two_holders_one_preempted()
        assert not queue.is_idle()
        get_llama_admission_queue(_OTHER)
        assert get_llama_admission_queue(_KEY) is queue
        paused.release()

    asyncio.run(_run())


def test_resume_never_admits_more_than_the_capacity():
    async def _run():
        queue, config, paused = _two_holders_one_preempted()
        get_llama_admission_queue(_OTHER)
        arrivals = [
            get_llama_admission_queue(_KEY).reserve(capacity = 2, config = config).lease_nowait()
            for _ in range(2)
        ]
        resumed = await paused.resume_async(0, timeout_s = 0.05, poll_s = 0.005)
        live = sum(lease is not None for lease in arrivals) + int(resumed)
        assert live <= 2, (live, resumed, arrivals)
        for lease in arrivals:
            if lease is not None:
                lease.release()
        paused.release()

    asyncio.run(_run())


@pytest.mark.parametrize("ending", ["resume", "release"])
def test_the_queue_is_idle_again_once_the_holder_ends(ending):
    async def _run():
        queue, _config, paused = _two_holders_one_preempted()
        if ending == "resume":
            assert await paused.resume_async(0, timeout_s = 0.5, poll_s = 0.005)
            assert not queue.is_idle()
        paused.release()
        assert queue.is_idle()
        # A second release, or a preempt after release, must not move the count.
        paused.release()
        assert not paused.preempt()
        assert queue.is_idle()

    asyncio.run(_run())


def test_a_holder_preempted_twice_is_counted_once():
    async def _run():
        queue, _config, paused = _two_holders_one_preempted()
        assert await paused.resume_async(0, timeout_s = 0.5, poll_s = 0.005)
        assert paused.preempt()
        assert not queue.is_idle()
        assert await paused.resume_async(0, timeout_s = 0.5, poll_s = 0.005)
        paused.release()
        assert queue.is_idle()

    asyncio.run(_run())
