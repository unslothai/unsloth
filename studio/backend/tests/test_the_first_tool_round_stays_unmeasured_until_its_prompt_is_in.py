# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The first tool round is published before its request opens, so it is not measured yet."""

from __future__ import annotations

from core.inference.llama_preemption import PreemptionController
import pytest

pytestmark = pytest.mark.usefixtures("preemption_opted_in")


class _Lease:
    def yield_parked_commitment(self, *, charged_at = None):
        return 0


def _controller() -> PreemptionController:
    made = PreemptionController("http://127.0.0.1:65075")
    made.configure(budget = 16384, kv_unified = True, slots = 4)
    return made


def test_a_probe_from_before_the_prompt_does_not_fold_it_in():
    controller = _controller()
    controller.register("other", lease = _Lease(), tokens = 2000)
    controller.note_measured("other")
    controller.note_resident(6000, started_at_seq = controller.residency_epoch())
    controller.register("loop", lease = _Lease(), tokens = 5000, prompt_tokens = 5000)
    in_flight = controller.residency_epoch()
    controller.note_tokens("loop", 5000, 5000)
    assert controller.participant("loop").measured is False
    controller.note_resident(6000, started_at_seq = in_flight)
    assert (
        controller.snapshot().committed >= 6000 + 5000
    ), "the first round's prompt was folded into a reading taken before it was sent"


def test_its_first_token_and_a_later_reading_fold_it_in():
    controller = _controller()
    controller.register("loop", lease = _Lease(), tokens = 5000, prompt_tokens = 5000)
    controller.note_tokens("loop", 5000, 5000)
    controller.observe("loop", 1)
    controller.note_resident(5001, started_at_seq = controller.residency_epoch())
    assert controller.participant("loop").measured is True
