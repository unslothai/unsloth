# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""An unpausable request waits for the chats its plan paused before it goes upstream.

Planning only marks the victims PREEMPTING and sets their signals; they keep decoding into their
cells until their streams see it and close upstream. A direct prompt sent at once could outrun
llama-server's KV-full retries, which halve the batch within one decode and then end every slot.
"""

import asyncio
import inspect
import threading
import time
from types import SimpleNamespace

import pytest

from core.inference import llama_preemption as preemption
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
    started = time.monotonic()
    holder = inference._openai_llama_count_direct_holder(fake, 9000, "direct-holder")
    assert 0.15 <= time.monotonic() - started < 2.0
    assert controller.participant("chat").state == ParticipantState.PREEMPTING
    inference._openai_llama_release_completions_holder(fake, holder)


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
