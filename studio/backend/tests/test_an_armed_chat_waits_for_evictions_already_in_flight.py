# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A chat armed while another arrival's victim is still decoding waits for that victim too.

The plan counts chats already told to pause as gone, so a second arrival picks no victim of its own
and used to go upstream at once, its prefill landing beside a victim still writing into its cells.
The unpausable holders already wait for evictions in flight when they have no room; the armed chat
now does the same."""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace
import uuid

import pytest

from core.inference import llama_preemption as preemption
from core.inference.llama_preemption import (
    ParticipantState,
    PreemptSignal,
    get_preemption_controller,
)
import routes.inference as inference

pytestmark = pytest.mark.usefixtures("preemption_opted_in")


@pytest.fixture
def busy(monkeypatch):
    """8192 cells: A decodes in 7000 of them, and B's plan has just told A to pause."""
    monkeypatch.setenv(preemption.PREEMPT_ENV, "1")
    monkeypatch.setattr(inference, "_refresh_residency_before_planning", lambda controller: None)
    fake = SimpleNamespace(base_url = f"http://in-flight-{uuid.uuid4().hex}.test")
    controller = get_preemption_controller(inference._preempt_key(fake))
    controller.configure(budget = 8192, kv_unified = True)
    controller.register("A", tokens = 7000, signal = PreemptSignal())
    controller.note_measured("A")
    controller.note_resident(7000, started_at_seq = controller.residency_epoch())
    controller.register("B", tokens = 1500, signal = PreemptSignal())
    assert [v.gen_id for v in controller.plan_preemptions(needed = 0)] == ["A"]
    assert controller.participant("A").state == ParticipantState.PREEMPTING

    def _enrol(*, gen_id, signal, **_kwargs):
        controller.register(gen_id, tokens = 1500, signal = signal)
        return controller

    monkeypatch.setattr(inference, "_openai_llama_preemption_enrol", _enrol)
    yield fake, controller
    for gen_id in list(controller._participants):
        controller.unregister(gen_id)


def test_a_later_arrival_waits_on_the_victim_another_plan_chose(busy):
    _fake, controller = busy
    controller.register("C", tokens = 1500, signal = PreemptSignal())
    inference._openai_llama_preemption_plan(controller, gen_id = "C", signal = PreemptSignal())
    assert controller.participant("C").awaiting_victims == ["A"], (
        "C goes upstream while A still decodes in 7000 of 8192 cells"
    )


def test_the_arm_returns_only_once_that_victim_let_go(busy):
    fake, controller = busy

    def _pause_later():
        time.sleep(0.3)
        controller.set_state("A", ParticipantState.PAUSED)

    threading.Thread(target = _pause_later, daemon = True).start()
    started = time.monotonic()
    policy = inference._openai_llama_preemption_arm(
        llama_backend = fake, gen_id = "C", signal = PreemptSignal()
    )
    assert policy is not None
    assert time.monotonic() - started >= 0.25, "the prefill went upstream beside a decoding victim"
    assert controller.participant("A").state == ParticipantState.PAUSED


def test_an_arrival_with_room_does_not_wait_on_them(busy):
    _fake, controller = busy
    controller.unregister("B")
    controller.register("C", tokens = 1, signal = PreemptSignal())
    if not controller.room_for("C", 1):
        pytest.skip("no room for a one-token arrival in this ledger")
    inference._openai_llama_preemption_plan(controller, gen_id = "C", signal = PreemptSignal())
    assert controller.participant("C").awaiting_victims == []
