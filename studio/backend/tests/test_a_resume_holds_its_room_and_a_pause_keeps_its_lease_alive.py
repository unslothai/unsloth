# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Seven ways the ledger and the wait drifted from the cache between a pause and its resume."""

import asyncio
import inspect
import threading
import time

import pytest

import core.inference.chat_generation_runs as runs
import core.inference.llama_cpp as llama_mod
import routes.inference as inference
from core.inference import llama_preemption as preemption
from core.inference.llama_admission import LlamaAdmissionConfig, LlamaAdmissionQueue
from core.inference.llama_preemption import (
    ParticipantState,
    PreemptionController,
    PreemptSignal,
    reset_preemption_controllers,
)
from .test_llama_tool_loop_preempt_resume import (
    _delta,
    _done,
    _finish,
    _Recorder,
    _RecordingPolicy,
    _run,
)


@pytest.fixture(autouse = True)
def _clean_registry():
    reset_preemption_controllers()
    yield
    reset_preemption_controllers()


class _Lease:
    def __init__(self, tokens = 2000):
        self.tokens = tokens
        self.slot = 0
        self.finished = False


def _controller(budget = 16384):
    controller = PreemptionController("k")
    controller.configure(budget = budget, kv_unified = True, slots = 4)
    return controller


class TestAGrantedResumeIsNotAVictimUntilItDecodes:
    def test_the_grant_marks_resuming_and_holds_the_room(self):
        controller = _controller()
        controller.register("a", lease = _Lease(), tokens = 1000, signal = PreemptSignal())
        controller.set_state("a", ParticipantState.PAUSED)
        before = controller.snapshot().committed
        assert controller.try_grant_resume("a", 3000) is True
        participant = controller.participant("a")
        assert participant.state == ParticipantState.RESUMING
        assert participant.holds_kv, "the booked room has to count"
        assert not participant.preemptable, "and it cannot be chosen"
        assert controller.snapshot().committed >= before + 3000

    def test_a_sweep_does_not_choose_it(self):
        controller = _controller(budget = 8192)
        controller.register("a", lease = _Lease(), tokens = 1000, signal = PreemptSignal())
        controller.set_state("a", ParticipantState.PAUSED)
        assert controller.try_grant_resume("a", 3000) is True
        # Two decoders and a full cache: the sweep must pick among them, never the resume.
        controller.register("b", lease = _Lease(), tokens = 3000, signal = PreemptSignal())
        controller.register("c", lease = _Lease(), tokens = 3000, signal = PreemptSignal())
        controller.note_resident(8192)
        chosen = controller.observe("b", 500)
        assert all(p.gen_id != "a" for p in chosen), "the resume was chosen as a victim"
        assert controller.participant("a").state == ParticipantState.RESUMING
        assert not controller.participant("a").preempt_event.is_set()

    def test_a_failed_resume_goes_back_to_paused(self):
        controller = _controller()
        controller.register("a", lease = _Lease(), tokens = 1000, signal = PreemptSignal())
        controller.set_state("a", ParticipantState.PAUSED)
        assert controller.try_grant_resume("a", 3000) is True
        controller.note_resume_failed("a")
        assert controller.participant("a").state == ParticipantState.PAUSED

    def test_a_resume_that_happened_decodes(self):
        controller = _controller()
        controller.register("a", lease = _Lease(), tokens = 1000, signal = PreemptSignal())
        controller.set_state("a", ParticipantState.PAUSED)
        assert controller.try_grant_resume("a", 3000) is True
        controller.note_resumed("a")
        assert controller.participant("a").state == ParticipantState.DECODING

    def test_tokens_arriving_also_mean_decoding(self):
        controller = _controller()
        controller.register("a", lease = _Lease(), tokens = 1000, signal = PreemptSignal())
        controller.set_state("a", ParticipantState.PAUSED)
        assert controller.try_grant_resume("a", 3000) is True
        controller.observe("a", 5)
        assert controller.participant("a").state == ParticipantState.DECODING


class TestARefusedResumeLeavesTheHolderPaused:
    def test_the_round_loop_does_not_report_a_resume_that_did_not_happen(self, monkeypatch):
        signal = preemption.PreemptSignal()
        policy = _RecordingPolicy(resume = False)
        recorder = _Recorder(
            monkeypatch,
            [
                [_delta("Once upon a time"), _finish(), _done()],
                [_delta(" there was a cat."), _finish(), _done()],
            ],
            signal = signal,
        )
        _run(recorder.backend, signal = signal, policy = policy)
        assert policy.events.count("preempted") == 1
        assert "resumed" not in policy.events, (
            "on_resumed ran for a refused resume, so the ledger carried a DECODING holder "
            "with its whole charge while the turn was ending with its lease preempted"
        )

    def test_both_handshakes_gate_on_the_answer(self):
        source = inspect.getsource(llama_mod.LlamaCppBackend.generate_chat_completion_with_tools)
        for flag in ("_resumed", "_resumed_f"):
            gate = source.index(f"if {flag}:\n")
            assert "preempt_policy.on_resumed()" in source[gate : gate + 400]


class TestThePauseSaysItIsStillWaiting:
    def test_keepalives_are_yielded_while_the_policy_blocks(self, monkeypatch):
        monkeypatch.setattr(llama_mod, "_PREEMPT_KEEPALIVE_S", 0.05)

        class _Slow:
            def await_resume(self, cancel_event = None):
                time.sleep(0.3)
                return True

        events = []

        def drive():
            result = yield from llama_mod._await_resume(_Slow(), threading.Event())
            events.append(("result", result))

        for event in drive():
            events.append(event)
        keepalives = [e for e in events if e == {"type": "preempt", "state": "keepalive"}]
        assert len(keepalives) >= 3, events
        assert events[-1] == ("result", True)

    def test_an_older_policy_is_called_without_the_event(self):
        class _Old:
            def await_resume(self):
                return False

        def drive():
            return (yield from llama_mod._await_resume(_Old(), threading.Event()))

        gen = drive()
        with pytest.raises(StopIteration) as stop:
            while True:
                next(gen)
        assert stop.value.value is False

    def test_a_policy_that_raises_raises_here(self):
        class _Bad:
            def await_resume(self, cancel_event = None):
                raise RuntimeError("no loop")

        def drive():
            return (yield from llama_mod._await_resume(_Bad(), None))

        with pytest.raises(RuntimeError):
            list(drive())

    def test_the_routes_forward_it_and_the_run_loop_renews_on_it(self):
        assert inference._OPENAI_PREEMPT_SSE_BY_STATE["keepalive"] == ": preempt-keepalive\n\n"
        assert runs._PREEMPT_KEEPALIVE_MARKER == ": preempt-keepalive"
        source = inspect.getsource(runs.ChatGenerationSupervisor)
        assert "_PREEMPT_KEEPALIVE_MARKER in text" in source
        routes = inspect.getsource(inference)
        assert (
            routes.count("_OPENAI_PREEMPT_SSE_BY_STATE.get(") >= 2
        ), "the stream consumers map preempt events through the table"
        backend = inspect.getsource(llama_mod)
        assert backend.count("yield from _await_resume(preempt_policy, cancel_event)") == 3


async def _lease(
    queue,
    *,
    tokens,
    capacity = 2,
    budget = 100,
):
    reservation = queue.reserve(
        capacity = capacity, config = LlamaAdmissionConfig(), tokens = tokens, budget = budget
    )
    lease = reservation.lease_nowait()
    assert lease is not None
    return lease


class TestResumeTicketsKeepTheirOrderForRoomToo:
    @pytest.mark.asyncio
    async def test_a_later_smaller_resume_does_not_overtake_an_earlier_one(self):
        """60 of 100 committed; the first resume wants 70, the second 40."""
        queue = LlamaAdmissionQueue("k")
        holder = await _lease(queue, tokens = 60)
        started = time.monotonic()
        first = asyncio.ensure_future(
            queue.acquire_parked_slot(tokens = 70, poll_s = 0.01, deadline = started + 2.0)
        )
        await asyncio.sleep(0.05)
        second = asyncio.ensure_future(
            queue.acquire_parked_slot(tokens = 40, poll_s = 0.01, deadline = started + 0.3)
        )
        await asyncio.sleep(0.15)
        assert not first.done() and not second.done()
        got_second = await second
        assert got_second is None, "the later ticket was admitted into room the earlier one is owed"
        holder.release()
        got_first = await first
        assert got_first is not None
        assert queue.snapshot().committed == 70
        queue.release(got_first, 70)

    @pytest.mark.asyncio
    async def test_a_fresh_arrival_leaves_the_room_a_ticket_is_owed(self):
        queue = LlamaAdmissionQueue("k")
        holder = await _lease(queue, tokens = 60)
        waiting = asyncio.ensure_future(
            queue.acquire_parked_slot(tokens = 70, poll_s = 0.01, deadline = time.monotonic() + 2.0)
        )
        await asyncio.sleep(0.05)
        reservation = queue.reserve(
            capacity = 4, config = LlamaAdmissionConfig(), tokens = 30, budget = 100
        )
        assert reservation.lease_nowait() is None, "30 more would leave the ticket short"
        holder.release()
        slot = await waiting
        assert slot is not None
        queue.release(slot, 70)
        try:
            reservation.cancel()
        except Exception:
            pass


class TestARawHolderIsMeasuredOnceItProduces:
    def test_measured_stops_the_double_count(self):
        controller = _controller()
        controller.register(
            "raw", lease = _Lease(4000), tokens = 4000, state = ParticipantState.STREAMING_RAW
        )
        controller.note_resident(4000)
        assert controller.snapshot().committed == 8000, "unmeasured: residency plus the charge"
        controller.note_measured("raw")
        assert controller.snapshot().committed == 4000
        assert controller.participant("raw").state == ParticipantState.STREAMING_RAW
        controller.note_measured("nobody")  # idempotent, tolerant

    def test_every_raw_loop_marks_at_its_first_data_line(self):
        source = inspect.getsource(inference)
        assert source.count("_raw_measured = False") == 3
        assert source.count("_openai_llama_note_raw_measured(") >= 4  # the def and three calls

    def test_the_non_streaming_raw_requests_are_measured_at_registration(self):
        # No data line to mark them at, so the charge sat on top of the residency
        # `/slots` reported for the whole answer.
        source = inspect.getsource(inference)
        assert source.count("measured = True,  # non-streaming") == 2
        helper = inspect.getsource(inference._openai_llama_count_raw_holder)
        assert "controller.note_measured(gen_id)" in helper

    def test_the_helper_reaches_the_controller(self, monkeypatch):
        controller = _controller()
        controller.register(
            "raw", lease = _Lease(), tokens = 2000, state = ParticipantState.STREAMING_RAW
        )
        monkeypatch.setattr(inference, "get_preemption_controller", lambda key: controller)

        class _Backend:
            base_url = "http://127.0.0.1:1"

        inference._openai_llama_note_raw_measured(llama_backend = _Backend(), gen_id = "raw")
        assert controller.participant("raw").measured is True


BASE = "http://127.0.0.1:65031"


class _Backend:
    base_url = BASE
    _kv_cache_unified = True
    context_length = 16384


class TestAPartialEraseReclaimsNothingGlobal:
    def test_the_early_reclaim_guards_like_the_later_one(self):
        source = inspect.getsource(inference)
        early = source.index('"reclaimed-idle-early"')
        window = source[early : early + 1600]
        assert 'if freed >= int(occupancy.get("idle_tokens") or 0):' in window
        assert window.index("if freed >= int") < window.index("controller.note_cells_reclaimed(")

    def _wired(self, monkeypatch, *, freed, readings):
        controller = PreemptionController(BASE)
        controller.configure(budget = 16384, kv_unified = True, slots = 4)
        controller.register("leaving", lease = _Lease(), tokens = 2000)
        controller.register("parked", lease = _Lease(), tokens = 2000)
        controller.set_state("parked", ParticipantState.PARKED_ON_TOOL)
        monkeypatch.setattr(inference, "get_preemption_controller", lambda key: controller)
        monkeypatch.setenv("UNSLOTH_LLAMA_ADMISSION_PREEMPT", "1")
        monkeypatch.setattr(inference, "fetch_llama_slots", lambda base, headers = None: [])
        it = iter(readings)
        monkeypatch.setattr(inference, "read_slot_occupancy", lambda scrape: next(it))
        monkeypatch.setattr(
            inference, "reclaim_idle_slots", lambda occupancy, erase, *, needed = 0: freed
        )
        monkeypatch.setattr(inference, "erase_llama_slot", lambda base, slot_id, headers = None: True)
        return controller

    def test_a_partial_erase_leaves_the_parked_holders_charge(self, monkeypatch):
        controller = self._wired(
            monkeypatch,
            freed = 1000,
            readings = [
                {"idle": [0, 1], "resident": 6000, "idle_tokens": 4000},
                {"idle": [1], "resident": 5000, "idle_tokens": 3000},
            ],
        )
        inference._openai_llama_preemption_disarm(llama_backend = _Backend(), gen_id = "leaving")
        assert (
            controller.participant("parked").cells_reclaimed is False
        ), "one of two idle slots went and the parked holder was told its cells were gone"

    def test_the_replacement_sample_is_a_fresh_reading(self, monkeypatch):
        controller = self._wired(
            monkeypatch,
            freed = 4000,
            readings = [
                {"idle": [0, 1], "resident": 6000, "idle_tokens": 4000},
                # A live chat prefilled while the erases ran: the cache is fuller than
                # `old - freed` says.
                {"idle": [], "resident": 5500, "idle_tokens": 0},
            ],
        )
        inference._openai_llama_preemption_disarm(llama_backend = _Backend(), gen_id = "leaving")
        assert (
            controller.snapshot().committed >= 5500
        ), "the worker wrote a stale `old - freed` over a newer residency sample"
        assert controller.participant("parked").cells_reclaimed is True

    def test_a_failed_re_read_keeps_the_newest_sample(self, monkeypatch):
        controller = self._wired(
            monkeypatch,
            freed = 4000,
            readings = [{"idle": [0, 1], "resident": 6000, "idle_tokens": 4000}, None],
        )
        controller.note_resident(7000)
        inference._openai_llama_preemption_disarm(llama_backend = _Backend(), gen_id = "leaving")
        assert controller.snapshot().committed >= 7000
