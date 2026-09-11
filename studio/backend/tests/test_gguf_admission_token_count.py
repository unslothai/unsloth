# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Numeric text must not overcommit the shared KV pool through a character estimate."""

import asyncio
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from core.inference.llama_admission import LlamaAdmissionQueue
import routes.inference as inference


def _backend(count):
    return SimpleNamespace(
        base_url = "test-token-count",
        context_length = 30000,
        effective_parallel_slots = 3,
        count_chat_tokens = Mock(return_value = count),
        _request_reasoning_kwargs = Mock(return_value = {"enable_thinking": False}),
    )


def _payload():
    return SimpleNamespace(
        messages = [{"role": "user", "content": " ".join(map(str, range(1000, 5500)))}],
        max_tokens = 1024,
        enable_thinking = False,
        reasoning_effort = None,
        preserve_thinking = None,
    )


def test_numeric_prompts_wait_for_room_but_short_prompts_still_overlap(monkeypatch):
    async def scenario(count):
        queue = LlamaAdmissionQueue("counted")
        monkeypatch.setattr(inference, "get_llama_admission_queue", lambda _: queue)
        backend = _backend(count)
        payload = _payload()
        reservations = []
        for _ in range(3):
            reservation, _ = await inference._reserve_counted_gguf_chat(
                request = None,
                llama_backend = backend,
                payload = payload,
                messages = payload.messages,
            )
            reservations.append(reservation)
        leases = [r.lease_nowait() for r in reservations]
        if count == 22560:
            assert leases[0] is not None
            assert leases[1:] == [None, None]
            assert queue.snapshot().committed == 23584
            leases[0].release()
            await asyncio.sleep(0)
            assert reservations[1].lease_nowait() is not None
        else:
            assert all(lease is not None for lease in leases)
        assert backend.count_chat_tokens.call_count == 3

    asyncio.run(scenario(22560))
    asyncio.run(scenario(100))


def test_counts_prepared_messages_and_tools_with_generation_options():
    backend = _backend(22560)
    payload = _payload()
    prepared = [{"role": "system", "content": "Injected system instructions"}] + payload.messages
    tools = [{"type": "function", "function": {"name": "lookup", "parameters": {}}}]
    count = inference._count_gguf_admission_prompt(backend, payload, prepared, tools)
    assert count == 22560
    args, kwargs = backend.count_chat_tokens.call_args
    assert args[0] == prepared
    assert kwargs == dict(
        tools = tools,
        strict = True,
        chat_template_kwargs = {"enable_thinking": False},
        continue_final_message = False,
    )
    assert (
        inference._openai_llama_admission_tokens(
            payload,
            budget = 30000,
            capacity = 3,
            injected_tools = tools,
            counted_prompt_tokens = count,
        )
        == 23584
    )


@pytest.mark.parametrize("result", [0, -1, None, True, "22560"])
def test_invalid_counts_reserve_the_entire_pool(result):
    payload = _payload()
    assert (
        inference._count_gguf_admission_prompt(_backend(result), payload, payload.messages) == 30000
    )


def test_tokenizer_failure_reserves_the_entire_pool():
    backend = _backend(0)
    backend.count_chat_tokens.side_effect = RuntimeError("tokenizer unavailable")
    payload = _payload()
    assert inference._count_gguf_admission_prompt(backend, payload, payload.messages) == 30000


def test_tool_recost_keeps_the_model_count(monkeypatch):
    async def scenario():
        queue = LlamaAdmissionQueue("recost")
        monkeypatch.setattr(inference, "get_llama_admission_queue", lambda _: queue)
        backend = _backend(22560)
        payload = _payload()
        reservation, _ = await inference._reserve_counted_gguf_chat(
            request = None,
            llama_backend = backend,
            payload = payload,
            messages = payload.messages,
            tool_loop = True,
        )
        assert reservation.lease_nowait() is not None
        inference._openai_llama_admission_recost(
            reservation,
            payload.messages,
            request = None,
            llama_backend = backend,
            payload = payload,
            count_prepared_prompt = True,
        )
        assert queue.snapshot().committed == 23584

    asyncio.run(scenario())
