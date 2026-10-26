# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A /v1/completions or /v1/embeddings request on the event loop plans its room on a fresh cache
reading, as the raw relays do. Planning on the reading already held missed the cells a running
holder took since, picked no victim, and sent the request into a full cache."""

import asyncio
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
    fake = SimpleNamespace(base_url = "http://127.0.0.1:65077", admission_key = "direct-fresh-test")
    controller = get_preemption_controller(inference._preempt_key(fake))
    controller.configure(budget = 16384, kv_unified = True)
    controller.register("chat", tokens = 15000)
    controller.note_measured("chat")
    controller.note_resident(15000, started_at_seq = controller.residency_epoch())

    def probe():
        # The cache grew by a running raw holder's decode since the last reading.
        epoch = controller.residency_epoch()
        time.sleep(0.05)
        controller.note_resident(16000, started_at_seq = epoch)

    controller.set_residency_probe(probe)
    yield fake, controller
    controller.set_residency_probe(None)
    for gen_id in list(controller._participants):
        controller.unregister(gen_id)


@pytest.mark.parametrize(
    "count, payload",
    [
        (inference._openai_llama_count_completions_holder_async, list(range(500))),
        (inference._openai_llama_count_embeddings_holder_async, list(range(500))),
    ],
)
def test_a_route_on_the_loop_reads_the_cache_before_it_plans(backend, count, payload):
    fake, controller = backend

    async def route():
        return await count(fake, payload)

    holder = asyncio.run(route())
    assert holder is not None
    assert (
        controller.participant("chat").state == ParticipantState.PREEMPTING
    ), "the plan ran on the stale reading and made no room for the request"
    inference._openai_llama_release_completions_holder(fake, holder)


def test_every_loop_route_counts_through_the_reading_variant():
    import inspect
    for route in (inference.openai_completions, inference.openai_embeddings):
        source = "".join(inspect.getsource(route).split())
        assert "_openai_llama_count_completions_holder(" not in source
        assert "_openai_llama_count_embeddings_holder(" not in source


def test_a_tts_holder_on_the_loop_plans_on_the_fresh_reading(backend, monkeypatch):
    fake, controller = backend
    monkeypatch.setattr(inference, "get_llama_cpp_backend", lambda: fake)

    async def route():
        # As _generate_tts_wav runs: read, then the sync context manager counts and plans.
        await inference._openai_llama_read_cache_for_direct_holder(fake)
        with inference._direct_llama_request(kv_tokens = 500) as holder:
            return holder, controller.participant("chat").state

    holder, state = asyncio.run(route())
    assert holder is not None
    assert state == ParticipantState.PREEMPTING, "the TTS plan ran on the stale reading"


def test_the_tts_route_reads_the_cache_before_its_holder_plans():
    import inspect

    source = "".join(inspect.getsource(inference._generate_tts_wav).split())
    read = source.index("await_openai_llama_read_cache_for_direct_holder(get_llama_cpp_backend())")
    assert read < source.index("with_direct_llama_request(")
