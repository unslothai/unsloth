# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A raw holder registers with its prompt apart from its output allowance."""

from __future__ import annotations

import inspect
import re

from core.inference.llama_preemption import PreemptionController
import routes.inference as inference
import pytest

pytestmark = pytest.mark.usefixtures("preemption_opted_in")

BASE = "http://127.0.0.1:65073"


class _Backend:
    base_url = BASE
    _kv_cache_unified = True
    context_length = 16384


class _Lease:
    tokens = 6000

    def yield_parked_commitment(self, *, charged_at = None):
        return 0


@pytest.fixture
def controller(monkeypatch):
    made = PreemptionController(BASE)
    made.configure(budget = 16384, kv_unified = True, slots = 4)
    monkeypatch.setattr(inference, "get_preemption_controller", lambda key: made)
    monkeypatch.setattr(inference, "_openai_llama_preemption_will_apply", lambda *a, **k: True)
    monkeypatch.setattr(
        inference, "_plan_for_unpausable_holder", lambda controller, gen_id, **_kwargs: None
    )
    return made


def _count(
    monkeypatch,
    priced,
    payload = object(),
):
    def price(*args, **kwargs):
        if isinstance(priced, Exception):
            raise priced
        return priced

    monkeypatch.setattr(inference, "_openai_llama_admission_charged_prompt_tokens", price)
    inference._openai_llama_count_raw_holder(
        llama_backend = _Backend(), lease = _Lease(), gen_id = "raw", payload = payload
    )


def test_the_prefill_announced_is_the_prompt(monkeypatch, controller):
    _count(monkeypatch, 300)
    assert (
        controller.snapshot().prefilling == 300
    ), "the output allowance was announced as a prefill"
    assert controller.participant("raw").tokens == 6000


def test_an_unpriced_prompt_still_announces_the_charge(monkeypatch, controller):
    _count(monkeypatch, RuntimeError("no count"))
    assert controller.snapshot().prefilling == 6000


def test_no_payload_still_announces_the_charge(monkeypatch, controller):
    _count(monkeypatch, 300, payload = None)
    assert controller.snapshot().prefilling == 6000


def test_a_prompt_past_the_charge_is_capped(monkeypatch, controller):
    _count(monkeypatch, 9000)
    assert controller.snapshot().prefilling == 6000


def test_every_raw_surface_hands_over_what_it_was_priced_from():
    source = inspect.getsource(inference)
    calls = re.findall(r"_openai_llama_count_raw_holder_async\((.*?)\n\s*\)", source, re.S)
    calls = [c for c in calls if "llama_backend = llama_backend" in c]
    assert len(calls) == 4, len(calls)
    assert all("payload = " in call for call in calls), calls
