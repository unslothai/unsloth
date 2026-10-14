# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""With preemption off a tool round is re-costed exactly as before preemption existed.

Main charged the round's conversation, catalogue, top-level fields and media, waited for room on
a wall clock where yielding was safe, and when the ledger declined carried on at the old figure.
The preemption work refuses such a round instead, re-counts the wire prompt, waits on a stall
clock and prices the finalized conversation in place of the payload; all of that belongs to the
share pricing and must not reach a server that does not park. The figures below are main's
(cc7e8aaeb) for the same calls.
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

# (opening charge, charge after the round re-cost) on main.
MAIN_RECOST = {(16384, 4): (4096, 4096), (262144, 4): (65536, 65536), (4096, 2): (2048, 3106)}

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


@pytest.mark.parametrize("shape", sorted(MAIN_RECOST))
def test_the_round_is_charged_what_main_charged(shape):
    async def run():
        backend = _backend(*shape)
        reservation, payload = _open(backend)
        lease = reservation.lease_nowait()
        opened = lease.tokens
        assert _recost(reservation, payload, backend) is None, "no wire cap without preemption"
        recosted = lease.tokens
        reservation.cancel()
        return opened, recosted

    assert asyncio.run(run()) == MAIN_RECOST[shape]


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


_ANTHROPIC_TOOLS = [
    {
        "name": "web_search",
        "description": "Search",
        "input_schema": {"type": "object", "properties": {"q": {"type": "string"}}},
    }
]
_ANTHROPIC_SYSTEM = "You are careful. " * 100
_TRANSLATED = [
    {"role": "system", "content": _ANTHROPIC_SYSTEM},
    {"role": "user", "content": "hello " * 200},
]
# Main's opening charge for the Anthropic surface: passthrough, tool loop, plain, at four slots.
MAIN_ANTHROPIC_OPENING = {
    16384: {"raw": 2218, "tool": 4096, "plain": 2173},
    262144: {"raw": 2218, "tool": 65536, "plain": 2173},
}


@pytest.mark.parametrize("budget", sorted(MAIN_ANTHROPIC_OPENING))
@pytest.mark.parametrize("shape", ["raw", "tool", "plain"])
def test_an_anthropic_request_opens_at_mains_charge(budget, shape):
    """As `_admitted_anthropic` prices it: the translated messages, and the conversation."""
    tool_loop = shape == "tool"
    payload = _Payload(
        system = _ANTHROPIC_SYSTEM,
        tools = None if shape == "plain" else _ANTHROPIC_TOOLS,
        messages = [{"role": "user", "content": "hello " * 200}],
        max_tokens = 1000,
    )
    charged = inf._openai_llama_admission_tokens(
        payload,
        budget = budget,
        capacity = 4,
        tool_loop = tool_loop,
        injected_tools = inf._anthropic_admission_injected_tools(
            None if shape == "plain" else _TOOLS, tool_loop = tool_loop
        ),
        conversation = _TRANSLATED,
        messages_override = _TRANSLATED,
        context_window = budget,
    )
    assert charged == MAIN_ANTHROPIC_OPENING[budget][shape]
