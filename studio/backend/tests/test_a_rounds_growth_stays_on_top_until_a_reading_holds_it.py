# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A measured holder's round growth stays on top of the reading until a reading holds it."""

from __future__ import annotations

from core.inference.llama_preemption import PreemptionController
import pytest

pytestmark = pytest.mark.usefixtures("preemption_opted_in")


class _Lease:
    def yield_parked_commitment(self, *, charged_at = None):
        return 0


def _measured_holder_beside_residue() -> PreemptionController:
    controller = PreemptionController("http://127.0.0.1:65075")
    controller.configure(budget = 32768, kv_unified = True, slots = 4)
    controller.register("chat", lease = _Lease(), tokens = 4000, prompt_tokens = 4000)
    controller.note_tokens("chat", 4000, 4000)
    controller.note_measured("chat")
    controller.note_resident(10000, started_at_seq = controller.residency_epoch())
    assert controller.participant("chat").measured is True
    return controller


def test_the_growth_is_counted_on_top_of_the_reading():
    controller = _measured_holder_beside_residue()
    controller.note_tokens("chat", 9000, 9000)
    assert (
        controller.snapshot().committed == 10000 + 5000
    ), "the round's growth was folded into a reading taken before it was sent"


def test_a_probe_in_flight_does_not_confirm_it():
    controller = _measured_holder_beside_residue()
    in_flight = controller.residency_epoch()
    controller.note_tokens("chat", 9000, 9000)
    controller.note_resident(10000, started_at_seq = in_flight)
    assert controller.snapshot().committed == 15000


def test_a_probe_before_the_round_is_sent_does_not_confirm_it():
    controller = _measured_holder_beside_residue()
    controller.note_tokens("chat", 9000, 9000)
    controller.note_resident(10000, started_at_seq = controller.residency_epoch())
    assert controller.snapshot().committed == 15000
    assert controller.participant("chat").unconfirmed_growth == 5000


def test_a_reading_started_after_its_first_token_folds_it_in():
    controller = _measured_holder_beside_residue()
    controller.note_tokens("chat", 9000, 9000)
    controller.observe("chat", 1, plan = False)
    controller.note_resident(15000, started_at_seq = controller.residency_epoch())
    assert controller.snapshot().committed == 15000
    assert controller.participant("chat").unconfirmed_growth == 0


def test_a_resume_is_not_granted_into_the_growth():
    controller = _measured_holder_beside_residue()
    controller.register("waiter", lease = _Lease(), tokens = 1000)
    controller.note_tokens("chat", 9000, 9000)
    assert controller.room_for("waiter", 18000) is False
    assert controller.room_for("waiter", 2000) is True
