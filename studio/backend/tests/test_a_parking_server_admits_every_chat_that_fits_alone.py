# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""When llama-server parks slots itself, Studio admission gates on slots, not on the KV sum.

Measured at -c 8192 --parallel 4 with --preempt-ram: four chats of about 3000 prompt tokens each
were charged about 4020 apiece, so the ledger let two in and held two at ``admission-wait``
while the server, which would have parked a resident to make room, idled half its slots.
Without server parking the cache budget still applies, exactly as on main.
"""

import asyncio
from types import SimpleNamespace

import pytest

from core.inference import llama_preemption as preemption
import routes.inference as inference_route


@pytest.fixture(autouse = True)
def _clean(monkeypatch):
    monkeypatch.delenv(preemption.PREEMPT_ENV, raising = False)
    monkeypatch.delenv(preemption.PREEMPT_MODE_ENV, raising = False)
    preemption.note_server_parking(False)
    yield
    preemption.note_server_parking(False)


_ids = iter(range(1_000_000))


def _backend(*, server_parks):
    return SimpleNamespace(
        base_url = f"http://fits-alone-{next(_ids)}.test",
        context_length = 8192,
        _kv_cache_context_total = 8192,
        effective_parallel_slots = 4,
        _kv_cache_unified = True,
        is_vision = False,
        server_preempts_kv = server_parks,
    )


def _admitted(backend, *, prompt = 3000, max_tokens = 512, chats = 4):
    async def run():
        reservations = []
        for _ in range(chats):
            reservation, _config = inference_route._openai_llama_admission_reserve(
                request = None,
                llama_backend = backend,
                payload = SimpleNamespace(
                    messages = [{"role": "user", "content": "hi"}],
                    max_tokens = max_tokens,
                    max_completion_tokens = None,
                ),
                counted_prompt_tokens = prompt,
            )
            reservations.append(reservation)
        granted = sum(1 for r in reservations if r.lease_nowait() is not None)
        for reservation in reservations:
            reservation.cancel()
        return granted

    return asyncio.run(run())


@pytest.mark.parametrize("max_tokens", [None, 512, 8192])
def test_a_parking_server_takes_every_chat_its_slots_can(max_tokens):
    preemption.note_server_parking(True)
    assert _admitted(_backend(server_parks = True), max_tokens = max_tokens) == 4


def test_slots_still_bound_a_parking_server():
    preemption.note_server_parking(True)
    assert _admitted(_backend(server_parks = True), chats = 6) == 4


@pytest.mark.parametrize("max_tokens", [None, 512])
def test_without_server_parking_the_cache_budget_still_holds_them_back(max_tokens):
    """Main's arithmetic: 3000 + 512 (stated) or 3000 + 1024 (unstated) each, two of 8192."""
    assert preemption.preemption_enabled() is False
    assert _admitted(_backend(server_parks = False), max_tokens = max_tokens) == 2


def test_the_kill_switch_keeps_the_budget_on_a_parking_server(monkeypatch):
    monkeypatch.setenv(preemption.PREEMPT_ENV, "0")
    preemption.note_server_parking(True)
    assert _admitted(_backend(server_parks = True)) == 2
