# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""A chat parked on a tool approval gives its cached context back on demand."""

import asyncio
from types import SimpleNamespace

import pytest
import core.inference.llama_cpp as llama_cpp
from core.inference.llama_admission import (
    LlamaAdmissionConfig,
    LlamaAdmissionQueue,
    reset_llama_admission_queues,
)


@pytest.fixture(autouse = True)
def reset():
    reset_llama_admission_queues()
    yield
    reset_llama_admission_queues()


def reserve(queue, tokens, capacity):
    config = LlamaAdmissionConfig()
    return queue.reserve(capacity = capacity, config = config, tokens = tokens, budget = 35000)


def test_a_parked_holder_keeps_its_context_until_someone_is_blocked_on_it():
    async def scenario():
        queue = LlamaAdmissionQueue("demand")
        parked = reserve(queue, 29000, 4).lease_nowait()
        assert parked.park()
        assert queue.snapshot().committed == 29000, "parking returns the slot, not the KV"
        assert not parked.reclaim_would_admit(), "erasing costs a reprocess, admits nobody"
        waiter = reserve(queue, 8750, 4)
        assert waiter.lease_nowait() is None
        assert parked.reclaim_would_admit()
        waiter.cancel()
        assert not parked.reclaim_would_admit()

    asyncio.run(scenario())


def test_a_resume_wins_its_room_back_and_is_not_overtaken():
    async def scenario():
        queue = LlamaAdmissionQueue("resume")
        first = reserve(queue, 29000, 4).lease_nowait()
        assert first.park()
        assert first.release_parked_cache() == 29000
        busy = reserve(queue, 20000, 4).lease_nowait()  # blocks the resume, not a small arrival
        resumed = asyncio.create_task(first.unpark_async(poll_s = 0.001))
        await asyncio.sleep(0.01)
        assert not resumed.done(), "a spare slot does not justify exceeding the KV budget"
        arrivals = [reserve(queue, 8000, 4) for _ in range(4)]
        assert not any(a.lease_nowait() for a in arrivals), "the approved chat is owed this"
        busy.release()
        await asyncio.wait_for(resumed, 1)
        assert queue.snapshot().committed == 29000, "the queue drains to the resumer first"
        for arrival in arrivals:
            arrival.cancel()
        first.release()
        assert queue.snapshot().committed == 0, "what the resume took back is given back"

    asyncio.run(scenario())


@pytest.mark.parametrize("timeout_s", [None, 0.01])
def test_a_claim_it_cannot_pay_for_does_not_pin_the_slot_a_later_one_needs(timeout_s):
    """The first needs KV the second holds; the second needs the slot the first reserves."""

    async def scenario():
        queue = LlamaAdmissionQueue(f"deadlock-{timeout_s}")
        first = reserve(queue, 10000, 1).lease_nowait()
        assert first.park()
        assert first.release_parked_cache() == 10000
        second = reserve(queue, 35000, 1).lease_nowait()
        assert second is not None, "the whole cache, freed by the first chat's erasure"
        assert second.park()
        ahead = asyncio.create_task(first.unpark_async(poll_s = 0.001, timeout_s = timeout_s))
        await asyncio.sleep(0.05)
        assert not ahead.done(), "the cache is committed; 10000 more does not fit"
        await asyncio.wait_for(second.unpark_async(poll_s = 0.001), 1)
        second.release()
        await asyncio.wait_for(ahead, 1)
        assert queue.snapshot().committed == 10000

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "save_dir,other,body,ok",
    [
        ("/tmp/slots", False, {"id_slot": 2, "n_erased": 12}, True),
        ("/tmp/slots", False, {"id_slot": 1, "n_erased": 12}, False),
        ("/tmp/slots", False, {"id_slot": 2}, False),
        ("/tmp/slots", True, None, False),
        (None, False, None, False),
    ],
)
def test_erasure_unconfirmed_keeps_the_context(monkeypatch, save_dir, other, body, ok):
    server = SimpleNamespace(base_url = "http://s", _auth_headers = {}, _slot_save_dir = save_dir)

    sent = []

    def post(url, **kwargs):
        sent.append((url, kwargs["params"]))
        return SimpleNamespace(raise_for_status = lambda: None, json = lambda: body)

    monkeypatch.setattr(llama_cpp.httpx, "post", post)
    target = "http://other" if other else server.base_url
    assert llama_cpp.LlamaCppBackend.release_idle_chat_slot(server, target, 2) is ok
    # A swallowed request reads as a refusal, so the call itself is the assertion.
    assert sent == ([] if body is None else [("http://s/slots/2", {"action": "erase"})])
