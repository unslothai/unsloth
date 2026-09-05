# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A round waiting for room behind a decoding leader is queued, not stuck.

`recost_waiting` reset its stall clock only when the pool's commitment FELL. That ledger
moves at round boundaries and nowhere else, so a tool-loop chat waiting for its next round
behind a leader decoding at full rate saw nothing move for `timeout_s`, decided the pool
was stuck, and went on at its old figure on top of a full cache. Measured 2026-09-05 on
the 35B model at -c 8192: five minutes of silence per round for the waiting chat while the
leader generated 220 tokens a second, then an uncharged round and a preemption.

The wait now also watches a `progress` signature, the preemptor's, which moves with every
token anybody decodes. Any change resets the clock; the hard deadline still bounds a pool
that moves forever without ever fitting this lease.
"""

from __future__ import annotations

import asyncio
import threading
import time

import pytest

from core.inference import llama_admission
from core.inference.llama_admission import LlamaAdmissionConfig


@pytest.fixture(autouse = True)
def _fresh_queues():
    llama_admission.reset_llama_admission_queues()
    yield
    llama_admission.reset_llama_admission_queues()


def _two_leases(queue, *, leader: int, waiter: int, budget: int):
    a = queue.reserve(capacity = 4, config = LlamaAdmissionConfig(), tokens = leader, budget = budget).lease_nowait()
    b = queue.reserve(capacity = 4, config = LlamaAdmissionConfig(), tokens = waiter, budget = budget).lease_nowait()
    assert a is not None and b is not None
    return a, b


@pytest.mark.asyncio
async def test_a_moving_progress_signature_keeps_the_wait_alive():
    queue = llama_admission.get_llama_admission_queue("http://patient.test")
    leader, waiter = _two_leases(queue, leader = 6000, waiter = 1000, budget = 8192)

    ticks = {"n": 0}

    def progress():
        # Somebody is decoding: a new value every read.
        ticks["n"] += 1
        return ticks["n"]

    outcome = {}

    def wait():
        # Wants more than fits beside the leader; the leader lets go after a while.
        outcome["granted"] = waiter.recost_waiting(
            4000, timeout_s = 0.3, poll_s = 0.02, progress = progress, gen_id = "waiter"
        )

    t = threading.Thread(target = wait)
    t.start()
    await asyncio.sleep(0.8)  # well past timeout_s: without progress this would have given up
    assert t.is_alive(), "the wait gave up while the pool was visibly moving"
    leader.release()
    t.join(timeout = 5)
    assert outcome["granted"] is True
    assert queue.committed_now() == 4000
    waiter.release()


@pytest.mark.asyncio
async def test_a_frozen_pool_still_gives_up_on_schedule():
    queue = llama_admission.get_llama_admission_queue("http://frozen.test")
    leader, waiter = _two_leases(queue, leader = 6000, waiter = 1000, budget = 8192)
    started = time.monotonic()
    granted = waiter.recost_waiting(
        4000, timeout_s = 0.3, poll_s = 0.02, progress = lambda: "same", gen_id = "waiter"
    )
    elapsed = time.monotonic() - started
    assert granted is False
    assert 0.25 <= elapsed < 2.0, elapsed
    # Back at the old figure, as before.
    assert queue.committed_now() == 6000 + 1000
    leader.release()
    waiter.release()


@pytest.mark.asyncio
async def test_the_hard_deadline_bounds_a_pool_that_moves_forever():
    queue = llama_admission.get_llama_admission_queue("http://forever.test")
    leader, waiter = _two_leases(queue, leader = 6000, waiter = 1000, budget = 8192)
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
async def test_no_progress_callable_behaves_as_before():
    queue = llama_admission.get_llama_admission_queue("http://plain.test")
    leader, waiter = _two_leases(queue, leader = 6000, waiter = 1000, budget = 8192)
    started = time.monotonic()
    assert waiter.recost_waiting(4000, timeout_s = 0.2, poll_s = 0.02) is False
    assert time.monotonic() - started < 2.0
    leader.release()
    waiter.release()
