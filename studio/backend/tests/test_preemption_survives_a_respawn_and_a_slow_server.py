# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Three ways a preempted chat lost its place that had nothing to do with the cache."""

import asyncio
import inspect
import re
import threading
import time

import pytest

import routes.inference as inference
from core.inference.llama_admission import LlamaAdmissionConfig, LlamaAdmissionQueue
from core.inference.llama_cpp import LlamaCppBackend
from core.inference.llama_preemption import PreemptionController


def _ticking():
    """A progress signature that moves on every reading, like a decoding backend."""
    box = {"n": 0}

    def _read():
        box["n"] += 1
        return (box["n"],)

    return _read


def _frozen():
    return (0,)


async def _lease(
    queue,
    *,
    tokens,
    capacity = 1,
    budget = 16384,
):
    reservation = queue.reserve(
        capacity = capacity,
        config = LlamaAdmissionConfig(),
        tokens = tokens,
        budget = budget,
    )
    lease = reservation.lease_nowait()
    assert lease is not None, "expected an immediate admission"
    return lease


class TestTheKeySurvivesARespawn:
    def _backend(self, port):
        backend = LlamaCppBackend.__new__(LlamaCppBackend)
        backend._port = port
        backend._admission_key = None
        backend._respawn_replay = False
        return backend

    def test_a_user_load_stamps_a_fresh_key(self):
        backend = self._backend(65001)
        backend._stamp_admission_key()
        assert backend.admission_key == "http://127.0.0.1:65001"
        backend._port = 65002
        backend._stamp_admission_key()
        assert backend.admission_key == "http://127.0.0.1:65002", "a new load is a new pool"

    def test_a_respawn_keeps_the_key_it_was_launched_under(self):
        backend = self._backend(65001)
        backend._stamp_admission_key()
        backend._respawn_replay = True
        backend._port = 65003
        backend._stamp_admission_key()
        assert backend.base_url == "http://127.0.0.1:65003", "the wire still goes to the new port"
        assert (
            backend.admission_key == "http://127.0.0.1:65001"
        ), "the ledger and the queue must still be the ones the live chats registered in"

    def test_a_backend_without_a_key_falls_back_to_its_url(self):
        backend = self._backend(65004)
        assert backend.admission_key == "http://127.0.0.1:65004"

    def test_every_surface_asks_the_backend_for_its_key(self):
        class _Keyed:
            base_url = "http://127.0.0.1:2"
            admission_key = "http://127.0.0.1:1"

        class _Bare:
            base_url = "http://127.0.0.1:3"

        assert inference._preempt_key(_Keyed()) == "http://127.0.0.1:1"
        assert inference._preempt_key(_Bare()) == "http://127.0.0.1:3"
        assert inference._preempt_key(object()) == "llama-server"
        source = inspect.getsource(inference)
        inline = source.count('getattr(llama_backend, "base_url", "llama-server")')
        assert inline == 1, (
            "the URL fallback belongs inside _preempt_key alone; an inline copy is a "
            "surface that loses its participants on the next respawn"
        )

    def test_the_respawn_replays_under_the_flag_and_every_port_pick_stamps(self):
        source = inspect.getsource(LlamaCppBackend._respawn_if_dead)
        flag_on = source.index("self._respawn_replay = True")
        replay = source.index("self.load_model(intent)")
        flag_off = source.index("self._respawn_replay = False")
        assert flag_on < replay < flag_off
        module = inspect.getsource(inspect.getmodule(LlamaCppBackend))
        picks = [m.end() for m in re.finditer(r"self\._port = self\._find_free_port\(\)", module)]
        assert picks, "expected the port picks this test guards"
        for end in picks:
            following = module[end : end + 120]
            assert (
                "self._stamp_admission_key()" in following
            ), "a port pick without a stamp leaves admission_key pointing at a dead server"


class TestAParkedResumeWaitsThroughAMovingPool:
    @pytest.mark.asyncio
    async def test_a_flat_deadline_still_gives_up_on_a_frozen_pool(self):
        queue = LlamaAdmissionQueue("k")
        holder = await _lease(queue, tokens = 1000)
        started = time.monotonic()
        slot = await queue.acquire_parked_slot(
            tokens = 0,
            poll_s = 0.01,
            deadline = started + 0.15,
            patience = 0.15,
            hard_deadline = started + 3.0,
            progress = _frozen,
        )
        assert slot is None
        assert time.monotonic() - started < 1.0, "nothing moved, so the stall bound must fire"
        holder.release()

    @pytest.mark.asyncio
    async def test_a_moving_pool_keeps_the_wait_alive_until_the_slot_comes_back(self):
        queue = LlamaAdmissionQueue("k")
        holder = await _lease(queue, tokens = 1000)

        async def _release_later():
            await asyncio.sleep(0.5)
            holder.release()

        asyncio.ensure_future(_release_later())
        started = time.monotonic()
        slot = await queue.acquire_parked_slot(
            tokens = 0,
            poll_s = 0.01,
            deadline = started + 0.15,
            patience = 0.15,
            hard_deadline = started + 5.0,
            progress = _ticking(),
        )
        waited = time.monotonic() - started
        assert slot is not None, (
            "the slot came back at 0.5s while the pool was moving; a 0.15s flat deadline "
            "abandoned this turn with every slot busy and healthy"
        )
        assert waited >= 0.4
        queue.release(slot, 0)

    @pytest.mark.asyncio
    async def test_the_hard_deadline_ends_a_wait_the_pool_never_serves(self):
        queue = LlamaAdmissionQueue("k")
        holder = await _lease(queue, tokens = 1000)
        started = time.monotonic()
        slot = await queue.acquire_parked_slot(
            tokens = 0,
            poll_s = 0.01,
            deadline = started + 0.1,
            patience = 0.1,
            hard_deadline = started + 0.3,
            progress = _ticking(),
        )
        assert slot is None
        assert 0.25 <= time.monotonic() - started < 1.5
        holder.release()

    @pytest.mark.asyncio
    async def test_resume_async_takes_the_signature_through_to_the_slot_wait(self):
        queue = LlamaAdmissionQueue("k")
        first = await _lease(queue, tokens = 1000)
        assert first.preempt() is True
        second = await _lease(queue, tokens = 1000)

        async def _release_later():
            await asyncio.sleep(0.5)
            second.release()

        asyncio.ensure_future(_release_later())
        started = time.monotonic()
        resumed = await first.resume_async(1200, poll_s = 0.01, timeout_s = 0.15, progress = _ticking())
        assert resumed is True, "a preempted chat waiting behind a live answer must get its turn"
        assert time.monotonic() - started >= 0.4
        assert queue.snapshot().committed == 1200
        first.release()

    @pytest.mark.asyncio
    async def test_resume_async_without_a_signature_keeps_the_old_flat_bound(self):
        queue = LlamaAdmissionQueue("k")
        first = await _lease(queue, tokens = 1000)
        assert first.preempt() is True
        second = await _lease(queue, tokens = 1000)
        started = time.monotonic()
        resumed = await first.resume_async(1200, poll_s = 0.01, timeout_s = 0.15)
        assert resumed is False
        assert time.monotonic() - started < 1.0
        second.release()


BASE = "http://127.0.0.1:65009"


class _Backend:
    base_url = BASE
    _kv_cache_unified = True
    context_length = 16384


class _Lease:
    tokens = 2000
    slot = 0
    finished = False


@pytest.fixture
def contended(monkeypatch):
    """Two holders, so the disarm decides the cells are wanted and goes to the wire."""
    made = PreemptionController(BASE)
    made.configure(budget = 16384, kv_unified = True, slots = 4)
    made.register("leaving", lease = _Lease(), tokens = 2000)
    made.register("staying", lease = _Lease(), tokens = 2000)
    monkeypatch.setattr(inference, "get_preemption_controller", lambda key: made)
    monkeypatch.setenv("UNSLOTH_LLAMA_ADMISSION_PREEMPT", "1")
    return made


@pytest.fixture
def slow_probe(monkeypatch):
    """A slots probe that takes 0.3s and records the thread it ran on."""
    seen = {"thread": None, "erased": []}

    def _fetch(base, **_kw):
        seen["thread"] = threading.current_thread()
        time.sleep(0.3)
        return [{"id": 0, "is_processing": False, "n_ctx_used": 2000}]

    monkeypatch.setattr(inference, "fetch_llama_slots", _fetch)
    monkeypatch.setattr(
        inference,
        "read_slot_occupancy",
        lambda scrape: (scrape() and {"idle": [0], "resident": 2000, "idle_tokens": 2000}),
    )

    def _reclaim(
        occupancy,
        erase,
        *,
        needed = 0,
    ):
        for slot_id in occupancy.get("idle", []):
            seen["erased"].append(slot_id)
            erase(slot_id)
        return 2000

    monkeypatch.setattr(inference, "reclaim_idle_slots", _reclaim)
    monkeypatch.setattr(inference, "erase_llama_slot", lambda base, slot_id, **_kw: True)
    return seen


class TestTheDisarmDoesNotHoldTheLoop:
    def test_from_an_async_caller_the_wire_work_leaves_the_loop_thread(self, contended, slow_probe):
        async def _route_body():
            started = time.monotonic()
            inference._openai_llama_preemption_disarm(llama_backend = _Backend(), gen_id = "leaving")
            held = time.monotonic() - started
            # The ledger is exact the instant the call returns.
            assert contended.participant("leaving") is None
            # Let the worker finish, then look at where the probe ran.
            await asyncio.sleep(0.6)
            return held

        held = asyncio.run(_route_body())
        assert held < 0.2, f"the disarm held the event loop for {held:.2f}s of blocking HTTP"
        assert slow_probe["thread"] is not None, "the reclaim must still happen"
        assert slow_probe["thread"] is not threading.main_thread()
        assert slow_probe["erased"] == [0]

    def test_from_a_plain_caller_the_reclaim_runs_inline(self, contended, slow_probe):
        inference._openai_llama_preemption_disarm(llama_backend = _Backend(), gen_id = "leaving")
        assert slow_probe["thread"] is threading.main_thread()
        assert slow_probe["erased"] == [0]
