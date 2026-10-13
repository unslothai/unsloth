# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Arming a chat takes the gate every other holder takes, and the prompt admission counted."""

from __future__ import annotations

from types import SimpleNamespace

from core.inference.llama_admission import (
    ADMISSION_CONTROL_ENV,
    ADMISSION_KV_BUDGET_ENV,
    LlamaAdmissionConfig,
    get_llama_admission_queue,
)
from core.inference.llama_preemption import PreemptSignal, get_preemption_controller
import routes.inference as inference
import pytest

pytestmark = [pytest.mark.usefixtures("preemption_opted_in"), pytest.mark.asyncio]


def _backend(url: str):
    return SimpleNamespace(
        base_url = url,
        context_length = 16384,
        _kv_cache_context_total = 16384,
        _kv_cache_unified = True,
        effective_parallel_slots = 4,
    )


def _payload():
    from models.inference import ChatCompletionRequest
    return ChatCompletionRequest(
        messages = [{"role": "user", "content": "0123456789 " * 40}],
        stream = True,
        enable_tools = False,
    )


def _arm(
    url: str,
    *,
    counted = None,
    tokens = 12000,
):
    queue = get_llama_admission_queue(url)
    reservation = queue.reserve(
        capacity = 4, config = LlamaAdmissionConfig(), budget = 16384, tokens = tokens
    )
    if counted is not None:
        reservation.counted_prompt_tokens = counted
    return inference._openai_llama_preemption_arm(
        request = None,
        llama_backend = _backend(url),
        reservation = reservation,
        gen_id = "chat",
        signal = PreemptSignal(),
        payload = _payload(),
    )


class TestTheAccountingGate:
    @pytest.mark.parametrize("switch", [ADMISSION_CONTROL_ENV, ADMISSION_KV_BUDGET_ENV])
    async def test_a_chat_is_not_armed_with_the_accounting_off(self, monkeypatch, switch):
        monkeypatch.setenv(switch, "0")
        url = f"http://127.0.0.1:65081/{switch}/"
        assert _arm(url) is None
        assert (
            get_preemption_controller(url).participant("chat") is None
        ), "armed against a lease the ledger never charged"

    @pytest.mark.parametrize("switch", [ADMISSION_CONTROL_ENV, ADMISSION_KV_BUDGET_ENV])
    async def test_its_observer_is_inert_too(self, monkeypatch, switch):
        monkeypatch.setenv(switch, "0")
        _refresh, observe, _note = inference._openai_llama_residency_observer(
            llama_backend = _backend("http://127.0.0.1:65082/"), completion_id = "chat"
        )
        assert observe is None

    async def test_with_the_accounting_on_it_still_arms(self):
        url = "http://127.0.0.1:65083/"
        assert _arm(url) is not None
        assert get_preemption_controller(url).participant("chat") is not None


class TestTheCountedPrompt:
    async def test_arming_takes_the_count_admission_priced(self):
        url = "http://127.0.0.1:65084/"
        assert _arm(url, counted = 9000) is not None
        participant = get_preemption_controller(url).participant("chat")
        assert (
            participant.prompt_tokens == 9000
        ), f"re-estimated from characters: {participant.prompt_tokens}"

    async def test_without_a_count_the_estimate_stands(self):
        url = "http://127.0.0.1:65085/"
        assert _arm(url) is not None
        participant = get_preemption_controller(url).participant("chat")
        assert 0 < participant.prompt_tokens < 9000

    async def test_the_reservation_carries_the_count(self):
        reservation, _config = inference._openai_llama_admission_reserve(
            request = None,
            llama_backend = _backend("http://127.0.0.1:65086/"),
            payload = _payload(),
            counted_prompt_tokens = 9000,
        )
        try:
            assert reservation.counted_prompt_tokens == 9000
        finally:
            reservation.cancel()
