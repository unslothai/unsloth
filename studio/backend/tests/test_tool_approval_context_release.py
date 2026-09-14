# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

import asyncio
import threading
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


def reserve(
    queue,
    tokens,
    capacity = 4,
):
    return queue.reserve(
        capacity = capacity,
        config = LlamaAdmissionConfig(),
        tokens = tokens,
        budget = 35000,
    )


def backend(slot_save_dir = "/tmp/slots"):
    return SimpleNamespace(
        base_url = "http://127.0.0.1:9999",
        _auth_headers = {},
        _slot_save_dir = slot_save_dir,
    )


def test_parking_alone_keeps_the_context_reserved():
    async def scenario():
        queue = LlamaAdmissionQueue("park-only")
        first = reserve(queue, 29000).lease_nowait()
        assert first.park()
        assert queue.snapshot().committed == 29000
        assert first.parked_tokens == 29000
        first.release()
        assert queue.is_idle()

    asyncio.run(scenario())


def test_reclaim_is_offered_only_while_someone_is_blocked_on_the_context():
    async def scenario():
        queue = LlamaAdmissionQueue("on-demand")
        first = reserve(queue, 29000).lease_nowait()
        assert first.park()
        assert not first.reclaim_would_admit(), "nobody is waiting, so erasing buys nothing"
        waiter = reserve(queue, 8750)
        assert waiter.lease_nowait() is None
        assert first.reclaim_would_admit()
        waiter.cancel()
        assert not first.reclaim_would_admit()
        first.release()
        assert queue.is_idle()

    asyncio.run(scenario())


def test_reclaim_is_refused_when_the_waiter_cannot_get_a_slot_either():
    async def scenario():
        queue = LlamaAdmissionQueue("slot-bound")
        first = reserve(queue, 30000, capacity = 1).lease_nowait()
        pending = reserve(queue, 4000, capacity = 1)
        assert pending.lease_nowait() is None
        assert first.park()
        await asyncio.sleep(0)
        other = pending.lease_nowait()
        assert other is not None, "parking hands the one slot on"
        waiter = reserve(queue, 5000, capacity = 1)
        assert waiter.lease_nowait() is None
        # 34000 of 35000 committed AND the only slot taken, so erasing would leave the
        # waiter room it still cannot use.
        assert not queue._fits_budget_locked(5000)
        assert not first.reclaim_would_admit(), "the pool is full; room is not the wall"
        waiter.cancel()
        other.release()
        first.release()
        assert queue.is_idle()

    asyncio.run(scenario())


def test_a_lapsed_resume_is_still_a_reason_to_reclaim():
    async def scenario():
        queue = LlamaAdmissionQueue("lapsed-claimant")
        resuming = reserve(queue, 20000).lease_nowait()
        parked = reserve(queue, 8000).lease_nowait()
        assert resuming.park()
        assert resuming.release_parked_cache() == 20000
        newcomer = reserve(queue, 8000).lease_nowait()
        assert newcomer is not None
        resumed = asyncio.create_task(resuming.unpark_async(poll_s = 0.001, timeout_s = 0.01))
        await asyncio.sleep(0.05)
        assert not resumed.done()
        # The deadline gave up priority, not the 20000 it still has to win back.
        assert parked.park()
        assert parked.reclaim_would_admit()
        assert parked.release_parked_cache() == 8000
        await asyncio.wait_for(resumed, 1)
        newcomer.release()
        resuming.release()
        parked.release()
        assert queue.is_idle()

    asyncio.run(scenario())


def test_parked_holders_reclaim_together_when_no_single_erasure_is_enough():
    async def scenario():
        queue = LlamaAdmissionQueue("two-parked")
        first = reserve(queue, 15000).lease_nowait()
        second = reserve(queue, 15000).lease_nowait()
        for lease in (first, second):
            assert lease.park()
        waiter = reserve(queue, 25000)
        assert waiter.lease_nowait() is None
        # Neither erasure alone leaves room for 25000, but both together do, so neither
        # holder may sit the round out.
        assert first.reclaim_would_admit()
        assert first.release_parked_cache() == 15000
        assert waiter.lease_nowait() is None
        assert second.reclaim_would_admit()
        assert second.release_parked_cache() == 15000
        await asyncio.sleep(0)
        admitted = waiter.lease_nowait()
        assert admitted is not None
        admitted.release()
        first.release()
        second.release()
        assert queue.is_idle()

    asyncio.run(scenario())


def test_a_parked_holder_reclaims_for_an_approved_chat_that_is_resuming():
    async def scenario():
        queue = LlamaAdmissionQueue("resume-claimant")
        resuming = reserve(queue, 20000).lease_nowait()
        parked = reserve(queue, 8000).lease_nowait()
        assert resuming.park()
        assert resuming.release_parked_cache() == 20000
        assert parked.park()
        newcomer = reserve(queue, 8000).lease_nowait()
        assert newcomer is not None
        resumed = asyncio.create_task(resuming.unpark_async(poll_s = 0.001))
        await asyncio.sleep(0.01)
        assert not resumed.done()
        # Nothing is queued and only the newcomer is decoding, but a chat the user
        # already approved cannot resume until the parked one gives its cache back.
        assert parked.reclaim_would_admit()
        assert parked.release_parked_cache() == 8000
        await asyncio.wait_for(resumed, 1)
        newcomer.release()
        resuming.release()
        parked.release()
        assert queue.is_idle()

    asyncio.run(scenario())


def test_reclaimed_approval_admits_waiter_and_resume_waits_for_context():
    async def scenario():
        queue = LlamaAdmissionQueue("approval")
        first = reserve(queue, 29000).lease_nowait()
        second = reserve(queue, 8750)
        assert second.lease_nowait() is None
        assert first.park()
        assert first.release_parked_cache() == 29000
        await asyncio.sleep(0)
        other = second.lease_nowait()
        assert other is not None
        assert queue.snapshot().committed == 8750
        resumed = asyncio.create_task(first.unpark_async(poll_s = 0.001))
        await asyncio.sleep(0.01)
        assert not resumed.done(), "spare slots do not justify exceeding the KV budget"
        other.release()
        await asyncio.wait_for(resumed, 1)
        assert queue.snapshot().committed == 29000
        first.release()
        assert queue.snapshot().committed == 0
        assert queue.is_idle()

    asyncio.run(scenario())


def test_a_resuming_approval_is_not_overtaken_by_new_arrivals():
    async def scenario():
        queue = LlamaAdmissionQueue("no-overtake")
        first = reserve(queue, 29000).lease_nowait()
        assert first.park()
        assert first.release_parked_cache() == 29000
        # Holds enough that the approved chat cannot resume yet, but leaves room a
        # smaller arrival would otherwise slip into.
        busy = reserve(queue, 20000).lease_nowait()
        assert busy is not None
        resumed = asyncio.create_task(first.unpark_async(poll_s = 0.001))
        await asyncio.sleep(0.01)
        assert not resumed.done()
        arrivals = []
        for _ in range(8):
            arrival = reserve(queue, 8000)
            assert arrival.lease_nowait() is None, "the approved chat is owed this room"
            arrivals.append(arrival)
            await asyncio.sleep(0.002)
        busy.release()
        await asyncio.wait_for(resumed, 1)
        assert queue.snapshot().committed == 29000, "the queue drains to the resumer first"
        assert not any(arrival.lease_nowait() for arrival in arrivals)
        first.release()
        for arrival in arrivals:
            arrival.cancel()
        assert queue.is_idle()

    asyncio.run(scenario())


def test_a_later_approval_does_not_take_room_owed_to_an_earlier_one():
    async def scenario():
        queue = LlamaAdmissionQueue("resume-order")
        first = reserve(queue, 20000).lease_nowait()
        second = reserve(queue, 10000).lease_nowait()
        for lease in (first, second):
            assert lease.park()
            assert lease.release_parked_cache() > 0
        busy = reserve(queue, 20000).lease_nowait()
        assert busy is not None
        resumed = [
            asyncio.create_task(lease.unpark_async(poll_s = 0.001)) for lease in (first, second)
        ]
        await asyncio.sleep(0.02)
        assert not any(task.done() for task in resumed), "10000 fits, but 20000 is owed first"
        busy.release()
        await asyncio.wait_for(asyncio.gather(*resumed), 1)
        assert queue.snapshot().committed == 30000
        first.release()
        second.release()
        assert queue.is_idle()

    asyncio.run(scenario())


def test_a_resume_that_waits_too_long_stops_holding_room_but_never_overcommits():
    async def scenario():
        queue = LlamaAdmissionQueue("bounded")
        first = reserve(queue, 29000).lease_nowait()
        assert first.park()
        assert first.release_parked_cache() == 29000
        other = reserve(queue, 8750).lease_nowait()
        assert other is not None
        resumed = asyncio.create_task(first.unpark_async(poll_s = 0.001, timeout_s = 0.05))
        await asyncio.sleep(0.15)
        assert not resumed.done(), "an exhausted cache fails every decoding slot at once"
        # The lapsed claim is what the deadline buys: arrivals are no longer held off.
        late = reserve(queue, 8000).lease_nowait()
        assert late is not None
        late.release()
        other.release()
        await asyncio.wait_for(resumed, 1)
        assert queue.snapshot().committed == 29000
        first.release()
        assert queue.is_idle()

    asyncio.run(scenario())


def test_a_blocked_resume_does_not_hide_a_smaller_claimant_behind_it():
    async def scenario():
        queue = LlamaAdmissionQueue("masking")
        resuming = reserve(queue, 29000).lease_nowait()
        assert resuming.park()
        assert resuming.release_parked_cache() == 29000
        parked = reserve(queue, 8000).lease_nowait()
        assert parked.park()
        busy = reserve(queue, 20000).lease_nowait()
        assert busy is not None
        resumed = asyncio.create_task(resuming.unpark_async(poll_s = 0.001, timeout_s = 0.01))
        await asyncio.sleep(0.05)
        assert not resumed.done()
        waiter = reserve(queue, 10000)
        assert waiter.lease_nowait() is None
        # Erasing 8000 cannot seat the 29000 resume, but it does seat the 10000 waiter,
        # so the resume must not stand in for the whole queue.
        assert parked.reclaim_would_admit()
        assert parked.release_parked_cache() == 8000
        await asyncio.sleep(0)
        admitted = waiter.lease_nowait()
        assert admitted is not None
        admitted.release()
        busy.release()
        await asyncio.wait_for(resumed, 1)
        resuming.release()
        parked.release()
        assert queue.is_idle()

    asyncio.run(scenario())


def test_a_lapsed_approval_stops_holding_room_from_a_later_one():
    async def scenario():
        queue = LlamaAdmissionQueue("lapsed-ahead")
        first = reserve(queue, 20000).lease_nowait()
        assert first.park()
        assert first.release_parked_cache() == 20000
        second = reserve(queue, 10000).lease_nowait()
        assert second.park()
        assert second.release_parked_cache() == 10000
        busy = reserve(queue, 20000).lease_nowait()
        assert busy is not None
        ahead = asyncio.create_task(first.unpark_async(poll_s = 0.001, timeout_s = 0.01))
        await asyncio.sleep(0.05)
        assert not ahead.done(), "20000 does not fit beside the 20000 already decoding"
        # The lapsed claim no longer reserves room, so the smaller approval behind it
        # resumes rather than queueing on a deadline that has already passed.
        behind = asyncio.create_task(second.unpark_async(poll_s = 0.001))
        await asyncio.wait_for(behind, 1)
        assert not ahead.done()
        busy.release()
        await asyncio.wait_for(ahead, 1)
        first.release()
        second.release()
        assert queue.is_idle()

    asyncio.run(scenario())


def test_a_resume_one_poll_from_its_slot_is_no_reason_to_erase():
    async def scenario():
        queue = LlamaAdmissionQueue("about-to-be-served")
        parked = reserve(queue, 9000, capacity = 3).lease_nowait()
        others = [reserve(queue, 4000, capacity = 3).lease_nowait() for _ in range(2)]
        assert all(others)
        assert parked.park()
        last = reserve(queue, 4000, capacity = 3).lease_nowait()
        assert last is not None, "the parked holder's slot goes to the next chat"
        # Registers a resume ticket that cannot be served on this poll, then sleeps past
        # the rest of the test.
        resumed = asyncio.create_task(parked.unpark_async(poll_s = 5))
        await asyncio.sleep(0.01)
        last.release()
        # That ticket owes nothing -- its cache was never erased -- and a slot just came
        # free, so it is one poll from decoding. Erasing would buy it nothing.
        assert not parked.reclaim_would_admit()
        resumed.cancel()
        await asyncio.gather(resumed, return_exceptions = True)
        for lease in others:
            lease.release()
        parked.release()

    asyncio.run(scenario())


def test_an_approval_waiting_for_room_does_not_pin_the_slot_a_later_one_needs():
    """Two approvals answered at once must not wait on each other forever.

    The first needs KV the second is still holding; the second needs only the slot, which
    the first was reserving. Nothing decodes, so nothing ever frees either.
    """

    async def scenario():
        queue = LlamaAdmissionQueue("resume-deadlock")
        first = reserve(queue, 10000, capacity = 1).lease_nowait()
        assert first.park()
        assert first.release_parked_cache() == 10000
        second = reserve(queue, 35000, capacity = 1).lease_nowait()
        assert second is not None, "the whole cache, freed by the first chat's erasure"
        assert second.park()
        ahead = asyncio.create_task(first.unpark_async(poll_s = 0.001))
        await asyncio.sleep(0.01)
        assert not ahead.done(), "35000 is committed; 10000 more does not fit"
        behind = asyncio.create_task(second.unpark_async(poll_s = 0.001))
        await asyncio.wait_for(behind, 1)
        assert not ahead.done()
        second.release()
        await asyncio.wait_for(ahead, 1)
        assert queue.snapshot().committed == 10000
        first.release()
        assert queue.is_idle()

    asyncio.run(scenario())


def test_a_lapsed_approval_still_does_not_pin_the_slot_a_later_one_needs():
    """The same deadlock, reached after the first approval's deadline passes.

    Lapsing changes what a holder reserves from others, never what it still needs, so it
    must not start holding a slot back the moment its deadline goes by.
    """

    async def scenario():
        queue = LlamaAdmissionQueue("lapsed-deadlock")
        first = reserve(queue, 10000, capacity = 1).lease_nowait()
        assert first.park()
        assert first.release_parked_cache() == 10000
        second = reserve(queue, 35000, capacity = 1).lease_nowait()
        assert second is not None
        assert second.park()
        ahead = asyncio.create_task(first.unpark_async(poll_s = 0.001, timeout_s = 0.01))
        await asyncio.sleep(0.05)
        assert not ahead.done()
        behind = asyncio.create_task(second.unpark_async(poll_s = 0.001))
        await asyncio.wait_for(behind, 1)
        second.release()
        await asyncio.wait_for(ahead, 1)
        assert queue.snapshot().committed == 10000
        first.release()
        assert queue.is_idle()

    asyncio.run(scenario())


def test_an_approval_that_can_afford_its_room_keeps_its_turn_for_the_slot():
    async def scenario():
        queue = LlamaAdmissionQueue("resume-order-slots")
        first = reserve(queue, 1000, capacity = 1).lease_nowait()
        assert first.park()
        second = reserve(queue, 1000, capacity = 1).lease_nowait()
        assert second is not None
        assert second.park()
        busy = reserve(queue, 1000, capacity = 1).lease_nowait()
        assert busy is not None
        # The first approval registers its ticket, then sleeps past the rest of the test,
        # so the second gets every poll and still must not overtake it.
        ahead = asyncio.create_task(first.unpark_async(poll_s = 5))
        await asyncio.sleep(0.01)
        behind = asyncio.create_task(second.unpark_async(poll_s = 0.001))
        await asyncio.sleep(0.01)
        busy.release()
        await asyncio.sleep(0.05)
        assert not behind.done(), "the one slot is owed to the approval that asked first"
        ahead.cancel()
        behind.cancel()
        await asyncio.gather(ahead, behind, return_exceptions = True)
        first.release()
        second.release()

    asyncio.run(scenario())


def test_a_lease_released_mid_resume_stops_holding_room_for_arrivals():
    async def scenario():
        queue = LlamaAdmissionQueue("released-mid-resume")
        first = reserve(queue, 29000).lease_nowait()
        assert first.park()
        assert first.release_parked_cache() == 29000
        other = reserve(queue, 8750).lease_nowait()
        assert other is not None
        resumed = asyncio.create_task(first.unpark_async(poll_s = 0.001))
        await asyncio.sleep(0.01)
        assert not resumed.done()
        first.release()
        await asyncio.wait_for(resumed, 1)
        late = reserve(queue, 1000).lease_nowait()
        assert late is not None, "a lease that is gone must not reserve the cache"
        late.release()
        other.release()
        assert queue.is_idle()

    asyncio.run(scenario())


def test_cancelled_resume_does_not_subtract_another_chats_tokens():
    async def scenario():
        queue = LlamaAdmissionQueue("cancel")
        first = reserve(queue, 29000).lease_nowait()
        assert first.park()
        assert first.release_parked_cache() == 29000
        other = reserve(queue, 8750).lease_nowait()
        cancel = threading.Event()
        cancel.set()
        await first.unpark_async(cancel_event = cancel)
        first.release()
        assert queue.snapshot().committed == 8750
        other.release()
        assert queue.is_idle()

    asyncio.run(scenario())


def test_release_during_readmission_returns_newly_acquired_tokens():
    async def scenario():
        queue = LlamaAdmissionQueue("released-resume")
        first = reserve(queue, 29000).lease_nowait()
        assert first.park()
        assert first.release_parked_cache() == 29000
        other = reserve(queue, 8750).lease_nowait()
        resumed = asyncio.create_task(first.unpark_async(poll_s = 0.001))
        await asyncio.sleep(0.01)
        first.release()
        other.release()
        await asyncio.wait_for(resumed, 1)
        assert queue.snapshot().committed == 0
        assert queue.is_idle()

    asyncio.run(scenario())


def test_a_released_lease_reclaims_nothing():
    async def scenario():
        queue = LlamaAdmissionQueue("released")
        first = reserve(queue, 29000).lease_nowait()
        assert first.park()
        first.release()
        assert first.release_parked_cache() == 0
        assert first.parked_tokens == 0
        assert queue.snapshot().committed == 0

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "chunk,expected",
    [
        # What llama-server sends for the final chat-stream result with verbose on and
        # response_fields narrowed to id_slot.
        ({"choices": [{"delta": {}}], "__verbose": {"id_slot": 3}}, 3),
        ({"choices": [{"delta": {"content": "hi"}}]}, None),
        ({"__verbose": {"id_slot": 0}}, 0),
        ({"__verbose": {}}, None),
        ({"__verbose": {"id_slot": -1}}, None),
        ({"__verbose": {"id_slot": "2"}}, None),
        ({"__verbose": {"id_slot": True}}, None),
        ({"__verbose": None}, None),
        ("data", None),
    ],
)
def test_the_decode_slot_is_read_only_from_a_verbose_result(chunk, expected):
    assert llama_cpp.LlamaCppBackend.decode_slot_from_chunk(chunk) == expected


@pytest.mark.parametrize(
    "body,expected",
    [
        ({"id_slot": 2, "n_erased": 123}, True),
        ({"id_slot": 1, "n_erased": 123}, False),
        ({"id_slot": 2}, False),
    ],
)
def test_engine_erasure_requires_matching_acknowledgement(monkeypatch, body, expected):
    engine = backend()

    def post(url, **kwargs):
        assert url == engine.base_url + "/slots/2"
        assert kwargs["params"] == {"action": "erase"}
        assert kwargs["trust_env"] is False
        return SimpleNamespace(raise_for_status = lambda: None, json = lambda: body)

    monkeypatch.setattr(llama_cpp.httpx, "post", post)
    assert llama_cpp.LlamaCppBackend.release_idle_chat_slot(engine, engine.base_url, 2) is expected


def test_erasure_does_not_target_a_replacement_server(monkeypatch):
    engine = backend()

    def post(*args, **kwargs):
        pytest.fail("must not erase the replacement server's slot")

    monkeypatch.setattr(llama_cpp.httpx, "post", post)
    assert not llama_cpp.LlamaCppBackend.release_idle_chat_slot(engine, "http://127.0.0.1:9998", 2)


def test_erasure_is_not_attempted_without_the_endpoint(monkeypatch):
    engine = backend(slot_save_dir = None)

    def post(*args, **kwargs):
        pytest.fail("the engine was started without --slot-save-path")

    monkeypatch.setattr(llama_cpp.httpx, "post", post)
    assert not llama_cpp.LlamaCppBackend.release_idle_chat_slot(engine, engine.base_url, 2)


def test_erasure_failure_keeps_reservation_conservative(monkeypatch):
    engine = backend()

    def post(*args, **kwargs):
        raise TimeoutError("engine did not acknowledge")

    monkeypatch.setattr(llama_cpp.httpx, "post", post)
    assert not llama_cpp.LlamaCppBackend.release_idle_chat_slot(engine, engine.base_url, 2)
