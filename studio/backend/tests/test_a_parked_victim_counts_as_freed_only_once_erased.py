# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A holder parked on a tool approval is freed by the erase, not by its signal.

It is blocked on the user and never reads the signal, so choosing it frees nothing until an
idle-slot erase takes its cells. With `/slots` unavailable or the erase failing, the plan counted
its cells as leaving anyway, then netted them out of every later plan, and a decoder grew into
cells that were still resident.
"""

from __future__ import annotations

from core.inference.llama_preemption import ParticipantState, PreemptionController
import pytest

pytestmark = pytest.mark.usefixtures("preemption_opted_in")


class _Lease:
    def yield_parked_commitment(self, *, charged_at = None):
        return 0


def _over_with_a_parked_holder() -> PreemptionController:
    controller = PreemptionController("http://127.0.0.1:65075")
    controller.configure(budget = 16384, kv_unified = True, slots = 4)
    controller.register("parked", lease = _Lease(), tokens = 6000)
    controller.note_state("parked", ParticipantState.PARKED_ON_TOOL)
    controller.register("old", lease = _Lease(), tokens = 5000)
    controller.register("new", lease = _Lease(), tokens = 5500)
    return controller


def test_a_decoder_is_chosen_too_while_the_parked_cells_are_still_resident():
    controller = _over_with_a_parked_holder()
    chosen = [p.gen_id for p in controller.plan_preemptions(needed = 0)]
    assert chosen[0] == "parked", "the erase still has to target the parked slot"
    assert (
        "new" in chosen
    ), "the parked holder was counted freed though nothing has erased it, so no decoder stopped"


def test_a_later_plan_still_counts_the_unerased_parked_cells():
    controller = _over_with_a_parked_holder()
    controller.plan_preemptions(needed = 0)
    # The decoder's pause lands; the parked holder's erase never does.
    controller.note_state("new", ParticipantState.DECODING)
    controller.unregister("new")
    controller.register("next", lease = _Lease(), tokens = 5500)
    chosen = [p.gen_id for p in controller.plan_preemptions(needed = 0)]
    assert "next" in chosen, "a later plan netted out cells nobody erased"


def test_an_erased_parked_holder_frees_its_room():
    controller = _over_with_a_parked_holder()
    controller.plan_preemptions(needed = 0)
    controller.unregister("new")
    # The erase landed: the parked holder no longer holds cells.
    assert (
        controller.note_cells_reclaimed({"parked": controller.participant("parked").park_seq}) == 1
    )
    controller.register("next", lease = _Lease(), tokens = 5000)
    assert controller.plan_preemptions(needed = 0) == []
