# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""An approval landing while an idle-slot erase runs does not hide the erase."""

from __future__ import annotations

from core.inference.llama_preemption import ParticipantState, PreemptionController
import pytest

pytestmark = pytest.mark.usefixtures("preemption_opted_in")


class _Lease:
    charge_seq = 1

    def __init__(self):
        self.yielded = 0

    def yield_parked_commitment(self, *, charged_at = None):
        self.yielded += 1
        return 0


def _parked_on_an_approval():
    controller = PreemptionController("http://127.0.0.1:65079")
    controller.configure(budget = 16384, kv_unified = True, slots = 4)
    lease = _Lease()
    controller.register("chat", lease = lease, tokens = 4000, prompt_tokens = 4000)
    controller.note_tokens("chat", 4000, 4000)
    controller.note_measured("chat")
    controller.note_state("chat", ParticipantState.PARKED_ON_TOOL)
    return controller, lease


@pytest.mark.parametrize("report", ["note_state", "set_state"])
def test_the_erase_is_recorded_though_the_tool_started(report):
    controller, lease = _parked_on_an_approval()
    reading = controller.parked_holders()
    getattr(controller, report)("chat", ParticipantState.TOOLS_RUNNING)
    assert controller.note_cells_reclaimed(reading) == 1
    participant = controller.participant("chat")
    assert participant.cells_reclaimed is True, "the erase took its cells but was not recorded"
    assert lease.yielded == 1


def test_a_park_after_decoding_is_still_a_new_one():
    controller, _lease = _parked_on_an_approval()
    reading = controller.parked_holders()
    controller.note_state("chat", ParticipantState.DECODING)
    controller.note_state("chat", ParticipantState.PARKED_ON_TOOL)
    assert controller.note_cells_reclaimed(reading) == 0
    assert controller.participant("chat").cells_reclaimed is False
