# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A chat armed in Studio mode whose plan paused a decoding chat waits for that chat to let go
before its own request goes upstream: the victim keeps decoding into its cells until its stream
sees the signal, and a prefill sent meanwhile overruns the cache it is freeing. The unpausable
holders already wait; the armed chat used to return straight from the plan."""

from __future__ import annotations

import asyncio
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
def armed(monkeypatch):
    monkeypatch.setenv(preemption.PREEMPT_ENV, "1")
    monkeypatch.setattr(inference, "_refresh_residency_before_planning", lambda controller: None)

    async def _no_read(controller):
        return None

    monkeypatch.setattr(inference, "_refresh_residency_before_planning_async", _no_read)
    fake = SimpleNamespace(base_url = f"http://armed-wait-{uuid.uuid4().hex}.test")
    controller = get_preemption_controller(inference._preempt_key(fake))
    controller.configure(budget = 16384, kv_unified = True)
    controller.register("chat", tokens = 15000)
    controller.note_measured("chat")
    controller.note_resident(15000, started_at_seq = controller.residency_epoch())

    def _enrol(*, gen_id, signal, **_kwargs):
        controller.register(gen_id, tokens = 4000, signal = signal)
        return controller

    monkeypatch.setattr(inference, "_openai_llama_preemption_enrol", _enrol)
    yield fake, controller
    for gen_id in list(controller._participants):
        controller.unregister(gen_id)


def _pause_later(controller, after_s):
    def run():
        time.sleep(after_s)
        controller.set_state("chat", ParticipantState.PAUSED)

    thread = threading.Thread(target = run, daemon = True)
    thread.start()
    return thread


def test_the_async_arm_returns_only_once_the_victim_let_go(armed):
    fake, controller = armed

    async def arm():
        _pause_later(controller, 0.3)
        started = time.monotonic()
        policy = await inference._openai_llama_preemption_arm_async(
            llama_backend = fake, gen_id = "new", signal = PreemptSignal()
        )
        return policy, time.monotonic() - started, controller.participant("chat").state

    policy, waited, state = asyncio.run(arm())
    assert policy is not None
    assert state == ParticipantState.PAUSED, "the prefill went upstream while the victim decoded"
    assert waited >= 0.25


def test_the_sync_arm_waits_inline(armed):
    fake, controller = armed
    _pause_later(controller, 0.3)
    started = time.monotonic()
    policy = inference._openai_llama_preemption_arm(
        llama_backend = fake, gen_id = "new", signal = PreemptSignal()
    )
    assert policy is not None
    assert time.monotonic() - started >= 0.25
    assert controller.participant("chat").state == ParticipantState.PAUSED


def test_a_victim_that_never_lets_go_does_not_refuse_the_chat(armed, monkeypatch):
    fake, controller = armed
    monkeypatch.setattr(inference, "_UNPAUSABLE_VICTIM_WAIT_S", 0.1)
    monkeypatch.setattr(inference, "_UNPAUSABLE_VICTIM_HARD_WAIT_S", 0.3)
    policy = inference._openai_llama_preemption_arm(
        llama_backend = fake, gen_id = "new", signal = PreemptSignal()
    )
    assert policy is not None
    assert controller.participant("chat").state == ParticipantState.PREEMPTING


class _Lease:
    tokens = 4000


class _Reservation:
    def lease_nowait(self):
        return _Lease()


def test_a_tool_round_waits_for_the_chats_its_sweep_paused(armed, monkeypatch):
    """The next round's prefill goes upstream straight after the publish: the same gap."""
    fake, controller = armed
    controller.register("round", tokens = 100, signal = PreemptSignal())
    controller.note_measured("round")
    monkeypatch.setattr(
        inference, "_openai_llama_admission_charged_prompt_tokens", lambda *a, **k: 3000
    )
    _pause_later(controller, 0.3)
    started = time.monotonic()
    inference._openai_llama_publish_round_charge(
        llama_backend = fake,
        gen_id = "round",
        reservation = _Reservation(),
        payload = None,
        conversation = [],
        rendered_tools = None,
        observe_tokens = lambda generated: controller.plan_preemptions(needed = 0, exclude = "round"),
    )
    assert time.monotonic() - started >= 0.25
    assert controller.participant("chat").state == ParticipantState.PAUSED
