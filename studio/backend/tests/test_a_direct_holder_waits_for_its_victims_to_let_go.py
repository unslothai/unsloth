# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""An unpausable request waits for the chats its plan paused before it goes upstream."""

import asyncio
import inspect
import threading
import time
from types import SimpleNamespace

import pytest

from core.inference import llama_preemption as preemption
from core.inference.llama_admission import LlamaAdmissionTimeout
from core.inference.llama_preemption import ParticipantState, get_preemption_controller
import routes.inference as inference

pytestmark = pytest.mark.usefixtures("preemption_opted_in")


@pytest.fixture
def backend(monkeypatch):
    monkeypatch.setenv(preemption.PREEMPT_ENV, "1")
    monkeypatch.setattr(inference, "_openai_llama_preemption_will_apply", lambda *a, **k: True)
    monkeypatch.setattr(inference, "_openai_llama_admission_budget", lambda *a, **k: 16384)
    monkeypatch.setattr(inference, "_refresh_residency_before_planning", lambda controller: None)
    fake = SimpleNamespace(base_url = "http://127.0.0.1:65076", _admission_key = "direct-wait-test")
    controller = get_preemption_controller(inference._preempt_key(fake))
    controller.configure(budget = 16384, kv_unified = True)
    controller.register("chat", tokens = 15000)
    controller.note_measured("chat")
    controller.note_resident(15000, started_at_seq = controller.residency_epoch())
    yield fake, controller
    for gen_id in list(controller._participants):
        controller.unregister(gen_id)


def _pause_later(controller, after_s):
    def run():
        time.sleep(after_s)
        controller.set_state("chat", ParticipantState.PAUSED)

    thread = threading.Thread(target = run, daemon = True)
    thread.start()
    return thread


def test_an_async_route_sends_only_once_the_victim_let_go(backend):
    fake, controller = backend

    async def route():
        holder = inference._openai_llama_count_completions_holder(fake, "x" * 16000)
        assert controller.participant("chat").state == ParticipantState.PREEMPTING
        _pause_later(controller, 0.3)
        started = time.monotonic()
        await inference._openai_llama_let_victims_go(fake, holder)
        return holder, time.monotonic() - started, controller.participant("chat").state

    holder, waited, state = asyncio.run(route())
    assert state == ParticipantState.PAUSED, "the request went upstream while the victim decoded"
    assert waited >= 0.25
    inference._openai_llama_release_completions_holder(fake, holder)


def test_a_worker_thread_caller_waits_inline(backend):
    fake, controller = backend
    _pause_later(controller, 0.3)
    started = time.monotonic()
    holder = inference._openai_llama_count_direct_holder(fake, 9000, "direct-holder")
    assert time.monotonic() - started >= 0.25
    assert controller.participant("chat").state == ParticipantState.PAUSED
    inference._openai_llama_release_completions_holder(fake, holder)


def test_a_stuck_victim_is_waited_on_only_so_long(backend, monkeypatch):
    fake, controller = backend
    monkeypatch.setattr(inference, "_UNPAUSABLE_VICTIM_WAIT_S", 0.2)
    monkeypatch.setattr(inference, "_UNPAUSABLE_VICTIM_HARD_WAIT_S", 0.5, raising = False)
    started = time.monotonic()
    with pytest.raises(LlamaAdmissionTimeout):
        inference._openai_llama_count_direct_holder(fake, 9000, "direct-holder")
    assert 0.15 <= time.monotonic() - started < 2.0
    assert controller.participant("chat").state == ParticipantState.PREEMPTING
    assert set(controller._participants) == {"chat"}, "the refused holder stayed registered"


def test_every_unpausable_route_waits_before_it_sends():
    source = "".join(inspect.getsource(inference).split())
    for gen_id in (
        "_cmpl_holder",
        "resp_id",
        "message_id",
        "completion_id",
        "_raw_gen_id",
        "_tts_holder",
    ):
        assert f"await_openai_llama_let_victims_go(" in source
        assert f"_openai_llama_let_victims_go(llama_backend,{gen_id})" in source or (
            f"_openai_llama_let_victims_go(get_llama_cpp_backend(),{gen_id})" in source
        ), gen_id


def test_a_victim_parked_on_a_tool_is_not_waited_on(backend):
    fake, controller = backend
    controller.note_state("chat", ParticipantState.PARKED_ON_TOOL)
    started = time.monotonic()
    holder = inference._openai_llama_count_direct_holder(fake, 9000, "direct-holder")
    assert controller.participant("chat").state == ParticipantState.PREEMPTING
    assert time.monotonic() - started < 1.0, "a parked victim never reads its signal"
    inference._openai_llama_release_completions_holder(fake, holder)


def test_a_second_holder_waits_for_the_eviction_the_first_one_began(monkeypatch):
    """The plan counts a chat already told to pause as gone, so the second raw request picked no
    victim of its own and went upstream while that chat still held 6000 of 8192 cells."""
    import asyncio as _asyncio
    from types import SimpleNamespace as _NS

    from core.inference import llama_preemption as _preemption
    from core.inference.llama_admission import LlamaAdmissionTimeout as _Timeout
    from core.inference.llama_preemption import ParticipantState as _State
    from core.inference.llama_preemption import get_preemption_controller as _controller_for
    import routes.inference as _inf

    monkeypatch.setenv(_preemption.PREEMPT_ENV, "1")
    monkeypatch.setattr(_inf, "_UNPAUSABLE_VICTIM_WAIT_S", 0.01)
    monkeypatch.setattr(_inf, "_UNPAUSABLE_VICTIM_HARD_WAIT_S", 0.05)
    backend = _NS(
        base_url = "http://pending-victim-test",
        context_length = 8192,
        _kv_cache_context_total = 8192,
        _kv_cache_unified = True,
        effective_parallel_slots = 4,
        is_vision = False,
        server_preempts_kv = False,
    )
    controller = _controller_for(_inf._preempt_key(backend))
    controller.configure(budget = 8192, kv_unified = True, slots = 4)
    reservations = []

    def payload(cap):
        return _NS(
            messages = [{"role": "user", "content": "hi"}], max_tokens = cap, max_completion_tokens = None
        )

    def reserve(
        prompt,
        cap,
        pausable = True,
    ):
        reservation, _ = _inf._openai_llama_admission_reserve(
            request = None,
            llama_backend = backend,
            payload = payload(cap),
            counted_prompt_tokens = prompt,
            pausable = pausable,
        )
        reservations.append(reservation)
        return reservation.lease_nowait()

    async def route():
        chat = reserve(1000, 6000)
        controller.register("chat", lease = chat, tokens = chat.tokens, prompt_tokens = 1000)
        controller.observe("chat", 5000, plan = False)
        controller.note_measured("chat")
        controller.note_resident(6000, started_at_seq = controller.residency_epoch())
        first = reserve(1000, 1000, False)
        await _inf._openai_llama_count_raw_holder_async(
            llama_backend = backend, lease = first, gen_id = "raw-a", payload = payload(1000)
        )
        assert controller.participant("chat").state == _State.PREEMPTING
        second = reserve(2500, 1000, False)
        await _inf._openai_llama_count_raw_holder_async(
            llama_backend = backend, lease = second, gen_id = "raw-b", payload = payload(1000)
        )
        assert "chat" in (
            controller.participant("raw-b").awaiting_victims or []
        ), "the second holder did not wait for the eviction the first one began"
        with pytest.raises(_Timeout):
            await _inf._openai_llama_let_victims_go(backend, "raw-b")

    try:
        _asyncio.run(route())
    finally:
        for reservation in reservations:
            reservation.cancel()
        for gen_id in ("chat", "raw-a", "raw-b"):
            controller.unregister(gen_id)
