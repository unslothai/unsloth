# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A controller is idle between being fetched for an enrolment and the register that follows.
Another backend's first lookup in that window retired it, so the chat registered on an orphan
and every later lookup for its key found a different, empty controller."""

from types import SimpleNamespace

import pytest

from core.inference import llama_preemption as preemption
from core.inference.llama_admission import LlamaAdmissionConfig, get_llama_admission_queue
from core.inference.llama_preemption import PreemptSignal, get_preemption_controller
import routes.inference as inference

pytestmark = pytest.mark.usefixtures("preemption_opted_in")


@pytest.mark.asyncio
async def test_another_backends_lookup_mid_enrolment_keeps_the_controller(monkeypatch):
    monkeypatch.setenv(preemption.PREEMPT_ENV, "1")
    monkeypatch.setattr(inference, "_openai_llama_preemption_will_apply", lambda *a, **k: True)
    monkeypatch.setattr(inference, "_openai_llama_admission_budget", lambda *a, **k: 16384)
    primary = SimpleNamespace(base_url = "http://127.0.0.1:65001", _admission_key = "primary-model")
    key = inference._preempt_key(primary)
    reservation = get_llama_admission_queue(key).reserve(
        capacity = 4, config = LlamaAdmissionConfig(), budget = 16384, tokens = 1000
    )
    assert reservation.lease_nowait() is not None

    def _configure(controller, request, llama_backend):
        controller.configure(budget = 16384, kv_unified = True)
        # The extra model's first chat arrives while this enrolment has not registered yet.
        get_preemption_controller("extra-model")

    monkeypatch.setattr(inference, "_openai_llama_configure_controller", _configure)
    enrolled = inference._openai_llama_preemption_enrol(
        request = None,
        llama_backend = primary,
        reservation = reservation,
        gen_id = "chat",
        signal = PreemptSignal(),
    )
    assert enrolled is not None
    current = get_preemption_controller(key)
    assert current is enrolled, "the enrolment registered on a controller the registry dropped"
    assert current.participant("chat") is not None


def test_a_retired_load_with_nothing_in_flight_is_still_dropped():
    first = get_preemption_controller("old-load")
    get_preemption_controller("new-load")
    assert get_preemption_controller("old-load") is not first
