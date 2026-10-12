# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A resume counts every replay already granted on top of the cache reading.

The reading cannot show a replay that has not prefilled yet. Taking the larger of the reading
and the ledger let a second resume be granted the room a first one was already given: 9000
resident plus two 4000 replays committed 17000 cells of a 16384 cache.
"""

from __future__ import annotations

from core.inference.llama_preemption import ParticipantState, PreemptionController, PreemptSignal
import pytest

pytestmark = pytest.mark.usefixtures("preemption_opted_in")


def _cache() -> PreemptionController:
    controller = PreemptionController("resume-sum")
    controller.configure(budget = 16384, kv_unified = True, slots = 4)
    controller.register("live", tokens = 5000, signal = PreemptSignal())
    controller.note_measured("live")
    for gen_id in ("a", "b"):
        controller.register(gen_id, tokens = 4000, signal = PreemptSignal())
        controller.set_state(gen_id, ParticipantState.PAUSED)
    # The live chat plus a finished request's cells that are not idle residue.
    controller.note_resident(9000, 0)
    return controller


def test_a_second_resume_is_refused_the_room_the_first_was_given():
    controller = _cache()
    assert controller.try_grant_resume("a", 4000) is True
    assert controller.participant("a").measured is False
    assert (
        controller.room_for("b", 4000) is False
    ), "both replays were granted: 9000 resident plus 8000 still to prefill"
    assert controller.committed_tokens() <= 16384


def test_one_resume_still_fits():
    controller = _cache()
    assert controller.room_for("a", 4000) is True
