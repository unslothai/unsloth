# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Arming a chat reads the cache on a worker thread so the event loop is not held. When the
server parks slots itself Studio plans no victims, so that thread was started and joined on
every admission for a reading nothing used."""

import asyncio
from types import SimpleNamespace

import pytest

from core.inference import llama_preemption as preemption
from core.inference.llama_preemption import PreemptSignal, get_preemption_controller
import routes.inference as inference_route


@pytest.fixture(autouse = True)
def _clean(monkeypatch):
    monkeypatch.delenv(preemption.PREEMPT_ENV, raising = False)
    monkeypatch.delenv(preemption.PREEMPT_MODE_ENV, raising = False)
    preemption.note_server_parking(False)
    yield
    preemption.note_server_parking(False)


@pytest.fixture
def started(monkeypatch):
    names = []
    real = inference_route.threading.Thread

    class _Recording(real):
        def __init__(self, *args, **kwargs):
            names.append(kwargs.get("name"))
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(inference_route.threading, "Thread", _Recording)
    return names


def _backend(*, server_parks, base_url):
    return SimpleNamespace(
        base_url = base_url,
        context_length = 16384,
        _kv_cache_context_total = 16384,
        effective_parallel_slots = 4,
        _kv_cache_unified = True,
        server_preempts_kv = server_parks,
    )


def _arm(backend):
    async def run():
        reservation, _config = inference_route._openai_llama_admission_reserve(
            request = None,
            llama_backend = backend,
            payload = SimpleNamespace(
                messages = [{"role": "user", "content": "hi"}],
                max_tokens = None,
                max_completion_tokens = None,
            ),
            counted_prompt_tokens = 100,
        )
        try:
            return inference_route._openai_llama_preemption_arm(
                request = None,
                llama_backend = backend,
                reservation = reservation,
                gen_id = "chat-1",
                signal = PreemptSignal(),
                loop = asyncio.get_running_loop(),
            )
        finally:
            reservation.cancel()

    return asyncio.run(run())


def test_a_parking_server_starts_no_residency_thread(started):
    preemption.note_server_parking(True)
    backend = _backend(server_parks = True, base_url = "http://parks.test")
    assert _arm(backend) is not None, "the chat must still be armed"
    assert get_preemption_controller(inference_route._preempt_key(backend)).server_mode
    assert "preempt-arm-residency" not in started


def test_studio_owned_preemption_still_reads_the_cache_first(monkeypatch, started):
    monkeypatch.setenv(preemption.PREEMPT_MODE_ENV, preemption.PREEMPT_MODE_STUDIO)
    backend = _backend(server_parks = False, base_url = "http://studio.test")
    assert _arm(backend) is not None
    assert "preempt-arm-residency" in started
