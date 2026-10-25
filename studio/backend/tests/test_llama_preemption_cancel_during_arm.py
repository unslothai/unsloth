# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A request cancelled while it arms preemption, after its reservation is taken and before any
stream exists, must leave the admission queue and the controller as it found them."""

from __future__ import annotations

import asyncio
import concurrent.futures

import pytest

from core.inference.llama_admission import LlamaAdmissionConfig, get_llama_admission_queue
from core.inference.llama_preemption import PreemptSignal, get_preemption_controller
import routes.inference as inference

from .preempt_fakes import clean_admission_queues, clean_preemption_registry
from .test_llama_preemption_is_armed_on_chat_surfaces import BASE, _ArmedBackend, _backend, _post

_FIXTURES = (clean_admission_queues, clean_preemption_registry)


pytestmark = pytest.mark.usefixtures("preemption_opted_in")


async def _cancelled_residency_read(controller):
    """The residency read the arm awaits, cancelled as a client disconnect would."""
    asyncio.current_task().cancel()
    await asyncio.sleep(0)


def _ledger_is_empty(url):
    queue = get_llama_admission_queue(url).snapshot()
    controller = get_preemption_controller(url).snapshot()
    assert queue.active == 0 and queue.committed == 0, f"the admission lease leaked: {queue}"
    assert queue.free == queue.capacity, f"a slot never came back: {queue}"
    assert (
        controller.holders == 0 and controller.committed == 0
    ), f"the participant stayed enrolled: {controller}"


class TestTheChatRoutes:
    @pytest.mark.parametrize("tools", [True, False], ids = ["tool-loop", "plain"])
    @pytest.mark.parametrize("stream", [False, True], ids = ["json", "sse"])
    def test_four_cancelled_arms_give_every_slot_back(self, monkeypatch, stream, tools):
        backend = _ArmedBackend(tools = tools)
        backend.count_chat_tokens = lambda *_args, **_kwargs: 32
        backend._request_reasoning_kwargs = lambda *_args, **_kwargs: None
        monkeypatch.setattr(
            inference, "_refresh_residency_before_planning_async", _cancelled_residency_read
        )
        # Four, one per slot: a leak per request leaves the queue full for the fifth.
        for _ in range(4):
            with pytest.raises((asyncio.CancelledError, concurrent.futures.CancelledError)):
                _post(monkeypatch, backend, stream = stream)
            _ledger_is_empty(BASE)
        assert not backend.seen, "the request began generating, so it was not cancelled mid-arm"


class TestTheArmHelpers:
    def _reserve(self, url):
        return get_llama_admission_queue(url).reserve(
            capacity = 4, config = LlamaAdmissionConfig(), budget = 16384, tokens = 4096
        )

    @pytest.mark.asyncio
    async def test_a_cancelled_arm_drops_its_enrolment(self, monkeypatch):
        url = "http://127.0.0.1:1/"
        monkeypatch.setattr(
            inference, "_refresh_residency_before_planning_async", _cancelled_residency_read
        )
        reservation = self._reserve(url)
        with pytest.raises(asyncio.CancelledError):
            await inference._openai_llama_preemption_arm_async(
                request = None,
                llama_backend = _backend(url = url),
                reservation = reservation,
                gen_id = "cancelled",
                signal = PreemptSignal(),
            )
        assert get_preemption_controller(url).participant("cancelled") is None
        reservation.cancel()

    @pytest.mark.asyncio
    async def test_a_cancelled_raw_holder_count_drops_its_enrolment(self, monkeypatch):
        url = "http://127.0.0.1:1/"
        monkeypatch.setattr(
            inference, "_refresh_residency_before_planning_async", _cancelled_residency_read
        )
        reservation = self._reserve(url)
        with pytest.raises(asyncio.CancelledError):
            await inference._openai_llama_count_raw_holder_async(
                llama_backend = _backend(url = url),
                lease = reservation.lease_nowait(),
                gen_id = "raw-cancelled",
            )
        assert get_preemption_controller(url).participant("raw-cancelled") is None
        reservation.cancel()


class TestStopDuringTheEvictionWait:
    """Stop pressed while an armed chat waits for the chats its plan paused: the wait can run a
    minute, so Stop has to reach the request and end it before anything goes upstream, not be
    answered `{"cancelled": 0}` and then end on the eviction 503."""

    CANCEL_ID = "stop-during-the-arm"

    @pytest.mark.parametrize("tools", [True, False], ids = ["tool-loop", "plain"])
    @pytest.mark.parametrize("stream", [False, True], ids = ["json", "sse"])
    def test_stop_reaches_the_arm_and_ends_the_request(self, monkeypatch, stream, tools):
        import threading
        import time

        from core.inference.llama_preemption import ParticipantState

        from .test_llama_preemption_is_armed_on_chat_surfaces import _client

        backend = _ArmedBackend(tools = tools)
        backend.count_chat_tokens = lambda *_args, **_kwargs: 32
        backend._request_reasoning_kwargs = lambda *_args, **_kwargs: None
        monkeypatch.setattr(inference, "_UNPAUSABLE_VICTIM_WAIT_S", 3.0)
        monkeypatch.setattr(inference, "_UNPAUSABLE_VICTIM_HARD_WAIT_S", 3.0)
        real_plan = inference._openai_llama_preemption_plan
        stopped: list = []

        def _plan(controller, *, gen_id, **kwargs):
            policy = real_plan(controller, gen_id = gen_id, **kwargs)
            # A victim still prefilling: told to pause, it holds its cells all wait long.
            controller.register("wedged", tokens = 100)
            controller.set_state("wedged", ParticipantState.PREEMPTING)
            controller.participant(gen_id).awaiting_victims = ["wedged"]

            def _press_stop():
                stopped.append(inference._cancel_by_cancel_id_or_stash(self.CANCEL_ID))

            timer = threading.Timer(0.2, _press_stop)
            timer.daemon = True
            timer.start()
            return policy

        monkeypatch.setattr(inference, "_openai_llama_preemption_plan", _plan)
        body = {
            "messages": [{"role": "user", "content": "write me an essay"}],
            "stream": stream,
            "cancel_id": self.CANCEL_ID,
        }
        if tools:
            body["enable_tools"] = True
        started = time.monotonic()
        response = _client(monkeypatch, backend, reraise = False).post(
            "/chat/completions", json = body, headers = {"X-Unsloth-Events": "1"}
        )
        waited = time.monotonic() - started
        get_preemption_controller(BASE).unregister("wedged")

        assert stopped and stopped[0] >= 1, "Stop found no request to cancel"
        assert waited < 2.0, f"Stop waited {waited:.1f}s for the victim"
        assert response.status_code != 503 and b"Timed out" not in response.content
        if not stream or tools:
            assert response.status_code == 499, response.status_code
        assert not backend.seen, "the request went upstream after Stop"
        _ledger_is_empty(BASE)
