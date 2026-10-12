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
        # Its stream had already emitted a tool call; the loop starts the tool.
        assert controller.note_state(victim, ParticipantState.TOOLS_RUNNING) is True
        assert controller.participant(victim).state == ParticipantState.TOOLS_RUNNING
        replaced = [
            gen_id
            for gen_id in ("a", "b", "c")
            if controller.participant(gen_id).state == ParticipantState.PREEMPTING
        ]
        assert replaced and victim not in replaced, (
            "the undone choice left the cache over its ceiling with nobody asked to stop"
        )

    def test_nobody_is_chosen_once_the_room_is_not_owed(self):
        controller = _three_decoders()
        victim = controller.plan_preemptions(needed = 0)[0].gen_id
        # A third chat finished meanwhile: the room the plan wanted is there now.
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
