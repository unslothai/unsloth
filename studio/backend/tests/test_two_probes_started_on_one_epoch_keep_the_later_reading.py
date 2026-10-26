# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Two probes that start with nothing recorded between them are still ordered."""

from __future__ import annotations

from core.inference.llama_preemption import PreemptionController
import pytest

pytestmark = pytest.mark.usefixtures("preemption_opted_in")


def _controller() -> PreemptionController:
    made = PreemptionController("http://127.0.0.1:65078")
    made.configure(budget = 16384, kv_unified = True, slots = 4)
    return made


def test_the_later_probe_is_kept_after_the_earlier_one_lands():
    controller = _controller()
    controller.note_resident(1000)
    earlier = controller.residency_epoch()
    later = controller.residency_epoch()
    assert controller.note_resident(3000, started_at_seq = earlier) is not None
    assert (
        controller.note_resident(9000, started_at_seq = later) is not None
    ), "a probe that started after the recorded one was dropped as stale"
    assert controller._resident == 9000


def test_the_earlier_probe_landing_last_is_still_dropped():
    controller = _controller()
    controller.note_resident(1000)
    earlier = controller.residency_epoch()
    later = controller.residency_epoch()
    assert controller.note_resident(9000, started_at_seq = later) is not None
    assert controller.note_resident(3000, started_at_seq = earlier) is None
    assert controller._resident == 9000


def test_a_mark_after_both_starts_still_waits_for_a_later_probe():
    controller = _controller()
    controller.register("chat", tokens = 4000)
    earlier = controller.residency_epoch()
    later = controller.residency_epoch()
    controller.note_measured("chat")
    controller.note_resident(4000, started_at_seq = earlier)
    controller.note_resident(4000, started_at_seq = later)
    assert controller.participant("chat").measured is False
    controller.note_resident(4000, started_at_seq = controller.residency_epoch())
    assert controller.participant("chat").measured is True
