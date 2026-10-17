# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A choice undone as its tool starts is planned again, and idle residue goes for pending pressure.

A victim chosen as its stream finished a tool call has the choice undone, since nothing reads
the signal while the tool runs. The plan that chose it counted its cells as leaving; an
unpausable holder plans once and reports no growth, so nothing else would sweep before that
room is used. And the early idle reclaim measured only the reading, so a prompt registered but
not yet in the cache left the idle residue standing while the plan after it paused a live chat.
"""

from __future__ import annotations

from core.inference.llama_preemption import ParticipantState, PreemptionController
import routes.inference as inference
import pytest

pytestmark = pytest.mark.usefixtures("preemption_opted_in")

BASE = "http://127.0.0.1:65074"


class _Lease:
    def yield_parked_commitment(self, *, charged_at = None):
        return 0


def _three_decoders() -> PreemptionController:
    controller = PreemptionController(BASE)
    controller.configure(budget = 8192, kv_unified = True, slots = 4)
    for gen_id in ("a", "b", "c"):
        controller.register(gen_id, lease = _Lease(), tokens = 3000)
        controller.note_measured(gen_id)
    return controller


class TestARevokedVictimIsReplaced:
    def test_another_is_chosen_while_the_room_is_still_owed(self):
        controller = _three_decoders()
        chosen = [p.gen_id for p in controller.plan_preemptions(needed = 0)]
        assert len(chosen) == 1, chosen
        victim = chosen[0]
        assert controller.note_state(victim, ParticipantState.TOOLS_RUNNING) is True
        assert controller.participant(victim).state == ParticipantState.TOOLS_RUNNING
        replaced = [
            gen_id
            for gen_id in ("a", "b", "c")
            if controller.participant(gen_id).state == ParticipantState.PREEMPTING
        ]
        assert (
            replaced and victim not in replaced
        ), "the undone choice left the cache over its ceiling with nobody asked to stop"

    def test_nobody_is_chosen_once_the_room_is_not_owed(self):
        controller = _three_decoders()
        victim = controller.plan_preemptions(needed = 0)[0].gen_id
        survivor = next(g for g in ("a", "b", "c") if g != victim)
        controller.unregister(survivor)
        assert controller.note_state(victim, ParticipantState.TOOLS_RUNNING) is True
        assert not any(
            p.state == ParticipantState.PREEMPTING
            for p in (controller.participant(g) for g in ("a", "b", "c"))
            if p is not None
        )

    def test_an_ordinary_transition_plans_nothing(self, monkeypatch):
        controller = _three_decoders()
        planned: list = []
        monkeypatch.setattr(
            PreemptionController,
            "plan_preemptions",
            lambda self, **kwargs: planned.append(kwargs) or [],
        )
        assert controller.note_state("a", ParticipantState.TOOLS_RUNNING) is True
        assert controller.note_state("a", ParticipantState.TOOLS_RUNNING) is False
        assert planned == []


class _Backend:
    base_url = BASE
    _kv_cache_unified = True
    _kv_cache_context_total = 16384
    context_length = 16384


def test_a_pending_prompt_reclaims_idle_residue_before_the_plan(monkeypatch):
    controller = PreemptionController(BASE)
    controller.configure(budget = 16384, kv_unified = True, slots = 4)
    controller.register("live", lease = _Lease(), tokens = 2500)
    controller.note_measured("live")
    controller.register("fresh", lease = _Lease(), tokens = 8000, prompt_tokens = 8000)
    occupancy = {"idle": [(3, 6000)], "resident": 8500, "idle_tokens": 6000}
    monkeypatch.setattr(inference, "get_preemption_controller", lambda key: controller)
    monkeypatch.setattr(inference, "fetch_llama_slots", lambda base, headers = None: [])
    monkeypatch.setattr(inference, "read_slot_occupancy", lambda scrape: dict(occupancy))
    erased: list = []

    def reclaim(
        occupancy,
        erase,
        *,
        needed = 0,
        **_kwargs,
    ):
        erased.append(needed)
        return 6000

    monkeypatch.setattr(inference, "reclaim_idle_slots", reclaim)
    refresh, _observe, _note = inference._openai_llama_residency_observer(
        llama_backend = _Backend(), completion_id = "live"
    )
    snapshot = controller.snapshot()
    ceiling = snapshot.budget - snapshot.buffer
    assert occupancy["resident"] < ceiling < occupancy["resident"] + 8000
    refresh(controller, force = True)
    assert erased and erased[0] > 0, (
        "the reading sat under the ceiling, so the idle residue stayed while the plan "
        "counted the pending prompt"
    )
