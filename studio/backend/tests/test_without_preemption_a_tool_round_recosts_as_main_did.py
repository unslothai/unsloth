# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""With preemption off a tool round is re-costed as before preemption existed.

Main waited for room on a wall clock where yielding was safe, and when the ledger declined
carried on at the old figure. The preemption work refuses such a round instead, re-counts the
wire prompt, and waits on a stall clock; all three belong to the share pricing and must not reach
a server that does not park.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from core.inference import llama_preemption as preemption
from core.inference.llama_admission import LlamaAdmissionLease, LlamaAdmissionRecostRefused
import routes.inference as inf


@pytest.fixture(autouse = True)
def _server_that_does_not_park(monkeypatch):
    monkeypatch.delenv(preemption.PREEMPT_ENV, raising = False)
    monkeypatch.delenv(preemption.PREEMPT_MODE_ENV, raising = False)
    preemption.note_server_parking(False)
    yield
    preemption.note_server_parking(False)


class _Payload:
    def __init__(self, **fields):
        self.__dict__.update(fields)

    def __getattr__(self, _name):
        return None


_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web",
            "parameters": {"type": "object", "properties": {"q": {"type": "string"}}},
        },
    }
]
_SYSTEM = "Answer briefly. " * 60
_GROWN = "more " * 2000
_CONVERSATION = [
    {"role": "system", "content": _SYSTEM},
    {"role": "user", "content": "find it " * 300},
    {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "web_search", "arguments": "{}"}}
        ],
    },
    {"role": "tool", "tool_call_id": "c1", "content": "result " * 500},
]

_ids = iter(range(1_000_000))


def _backend(budget, slots):
    return SimpleNamespace(
        base_url = f"http://recost-{next(_ids)}",
        effective_parallel_slots = slots,
        _kv_cache_context_total = budget,
        context_length = budget,
        _kv_cache_unified = True,
        is_vision = False,
    )


def _open(backend):
    payload = _Payload(system = _SYSTEM, messages = [{"role": "user", "content": "q"}])
    reservation, _config = inf._openai_llama_admission_reserve(
        request = None,
        llama_backend = backend,
        payload = payload,
        tool_loop = True,
        injected_tools = _TOOLS,
    )
    return reservation, payload


def _recost(reservation, payload, backend, conversation = _CONVERSATION):
    return inf._openai_llama_admission_recost(
        reservation,
        conversation,
        request = None,
        llama_backend = backend,
        payload = payload,
        injected_tools = _TOOLS,
        wire_tools = _TOOLS,
    )


def test_a_round_the_ledger_declines_carries_on_at_its_old_figure():
    """Two chats fill a 4096 cache; one grows past its share where yielding is unsafe."""

    async def run():
        backend = _backend(4096, 2)
        first, payload = _open(backend)
        second, _ = _open(backend)
        lease = first.lease_nowait()
        assert second.lease_nowait() is not None
        held = lease.tokens
        grown = _CONVERSATION + [{"role": "user", "content": _GROWN}]
        assert _recost(first, payload, backend, grown) is None
        after = lease.tokens
        first.cancel()
        second.cancel()
        return held, after

    held, after = asyncio.run(run())
    assert after == held


def test_the_wait_is_main_wall_clock(monkeypatch):
    calls = []
    real = LlamaAdmissionLease.recost_waiting

    def recording(self, tokens, **kwargs):
        calls.append(kwargs)
        return real(self, tokens, **kwargs)

    monkeypatch.setattr(LlamaAdmissionLease, "recost_waiting", recording)

    async def run():
        backend = _backend(16384, 4)
        reservation, payload = _open(backend)
        _recost(reservation, payload, backend)
        reservation.cancel()

    asyncio.run(run())
    assert calls and calls[0].get("stall_clock") is False
    assert calls[0].get("progress") is None


def test_with_share_pricing_a_declined_round_is_refused(share_pricing):
    async def run():
        backend = _backend(4096, 2)
        first, payload = _open(backend)
        second, _ = _open(backend)
        try:
            with pytest.raises(LlamaAdmissionRecostRefused):
                _recost(first, payload, backend, _CONVERSATION + [{"role": "user", "content": _GROWN}])
        finally:
            first.cancel()
            second.cancel()

    asyncio.run(run())
