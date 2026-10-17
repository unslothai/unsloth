# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Idle residue a cached reading saw is offered before a contended plan pauses a live chat."""

from __future__ import annotations

from core.inference.llama_preemption import ParticipantState, PreemptionController
import routes.inference as inference
import pytest

pytestmark = pytest.mark.usefixtures("preemption_opted_in")


BASE = "http://127.0.0.1:65079"


class _Backend:
    base_url = BASE
    _kv_cache_unified = True
    context_length = 8192


class _Lease:
    def yield_parked_commitment(self, *, charged_at = None):
        return 0


def _setup(monkeypatch):
    controller = PreemptionController(BASE)
    controller.configure(budget = 8192, kv_unified = True, slots = 4)
    for gen_id in ("old", "live"):
        controller.register(gen_id, lease = _Lease(), tokens = 1000, prompt_tokens = 1000)
        controller.note_measured(gen_id)
    monkeypatch.setattr(inference, "get_preemption_controller", lambda key: controller)
    monkeypatch.setattr(inference, "fetch_llama_slots", lambda base, headers = None: [])
    state = {"resident": 4000, "idle": [(3, 2000)], "idle_tokens": 2000, "reads": 0}

    def read(scrape):
        state["reads"] += 1
        return {k: v for k, v in state.items() if k != "reads"}

    monkeypatch.setattr(inference, "read_slot_occupancy", read)
    erased: list = []

    def reclaim(
        occupancy,
        erase,
        *,
        needed = 0,
        **_kwargs,
    ):
        if not occupancy.get("idle"):
            return 0
        erased.append(_preempting(controller))
        state.update(resident = state["resident"] - 2000, idle = [], idle_tokens = 0)
        return 2000

    monkeypatch.setattr(inference, "reclaim_idle_slots", reclaim)
    refresh, observe, _note = inference._openai_llama_residency_observer(
        llama_backend = _Backend(), completion_id = "live"
    )
    refresh(controller, force = True)
    assert erased == [] and controller.snapshot().committed == 4000
    return controller, observe, erased, state


def _preempting(controller) -> list:
    return [
        p.gen_id
        for p in controller._participants.values()
        if p.state == ParticipantState.PREEMPTING
    ]


def test_the_residue_goes_before_anyone_is_paused(monkeypatch):
    controller, observe, erased, _state = _setup(monkeypatch)
    controller.note_tokens("live", 5000, 5000)
    snapshot = controller.snapshot()
    assert snapshot.committed > snapshot.budget - snapshot.buffer
    observe(0)
    assert erased == [[]], f"a live chat was signalled before the residue went: {erased}"
    assert _preempting(controller) == [], "paused for room the idle slot gave"
    assert controller.snapshot().committed == 6000


def test_a_reading_that_offered_its_residue_is_not_read_again(monkeypatch):
    controller, observe, _erased, state = _setup(monkeypatch)
    controller.note_tokens("live", 5000, 5000)
    observe(0)
    reads = state["reads"]
    controller.note_tokens("live", 9000, 9000)
    observe(0)
    assert state["reads"] == reads


def test_under_the_ceiling_the_throttle_holds(monkeypatch):
    controller, observe, erased, state = _setup(monkeypatch)
    reads = state["reads"]
    controller.note_tokens("live", 2000, 2000)
    observe(0)
    assert state["reads"] == reads and erased == []
