# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A pause holds no cells, so its replay estimate cannot come off the residency reading.

Subtracting it credited the paused chat with room the reading says is somebody else's, and
the residency check exists precisely to catch a ledger that has drifted below the truth.
"""

from __future__ import annotations

from core.inference import llama_preemption as preemption
from core.inference.llama_preemption import ParticipantState
import pytest

pytestmark = pytest.mark.usefixtures("preemption_opted_in")


def _controller(key: str) -> preemption.PreemptionController:
    controller = preemption.PreemptionController(key)
    controller.configure(budget = 16384, kv_unified = True, slots = 4, batch_tokens = 2048)
    return controller


class TestAPausedHolderSubtractsNothing:
    def test_a_resume_is_refused_when_the_cache_is_already_full(self):
        """Ledger 4000, reading 11000, and an 8000-token replay would make it 19000."""
        controller = _controller("http://resume-probe")
        controller.register("live", tokens = 4000)
        controller.note_measured("live")
        controller.register("paused", tokens = 8000, state = ParticipantState.PAUSED)
        controller.note_resident(11000, reclaimable = 0)

        assert controller.try_grant_resume("paused", 8000) is False
        assert controller.room_for("paused", 8000) is False

    def test_the_refusal_is_the_reading_and_not_the_ledger(self):
        """Same ledger, a reading the slots have since freed: the resume goes ahead."""
        controller = _controller("http://resume-probe-free")
        controller.register("live", tokens = 4000)
        controller.note_measured("live")
        controller.register("paused", tokens = 8000, state = ParticipantState.PAUSED)
        controller.note_resident(4000, reclaimable = 0)

        assert controller.try_grant_resume("paused", 8000) is True

    def test_an_idle_residue_is_still_erased_for_the_waiter(self):
        """Reclaimable cells are waiting to be erased, not standing in the way."""
        controller = _controller("http://resume-probe-idle")
        controller.register("live", tokens = 4000)
        controller.note_measured("live")
        controller.register("paused", tokens = 8000, state = ParticipantState.PAUSED)
        controller.note_resident(11000, reclaimable = 7000)

        assert controller.try_grant_resume("paused", 8000) is True

    def test_a_reclaimed_holder_owns_no_cells_either(self):
        """Its charge describes cells an idle-slot reclaim already erased."""
        controller = _controller("http://resume-probe-reclaimed")
        controller.register("live", tokens = 4000)
        controller.note_measured("live")
        controller.register("parked", tokens = 8000, state = ParticipantState.PARKED_ON_TOOL)
        controller.note_measured("parked")
        controller.note_cells_reclaimed()
        controller.note_resident(11000, reclaimable = 0)

        assert controller.room_for("parked", 8000) is False


class TestAResidentHolderStillSubtractsItsOwn:
    def test_a_measured_holder_is_not_charged_for_its_own_cells_twice(self):
        """It IS in the reading, so asking for its own size again must not double it."""
        controller = _controller("http://resume-probe-resident")
        controller.register("mine", tokens = 8000)
        controller.note_measured("mine")
        controller.note_resident(11000, reclaimable = 0)

        # 11000 resident, 8000 of it this holder's: 3000 belongs to anybody else.
        assert controller.room_for("mine", 8000) is True

    def test_a_grant_that_has_not_prefilled_yet_owns_nothing(self):
        """RESUMING holds KV in the ledger, but its prompt is not in the cache."""
        controller = _controller("http://resume-probe-resuming")
        controller.register("mine", tokens = 8000, state = ParticipantState.RESUMING)
        controller.note_resident(11000, reclaimable = 0)

        assert controller.room_for("mine", 8000) is False


class TestARawHolderIsAddedOnTopUntilItsPrefillLands:
    """A counted holder is inside the residency sample only once llama-server has its
    prompt. Marked measured at registration, the sample from before its prefill swallows
    it, and the decoders beside it read as under the watermark by its whole prompt."""

    def test_unmeasured_it_is_added_and_measured_it_is_inside_the_sample(self):
        controller = _controller("http://raw-holder")
        controller.register("chat", tokens = 6000)
        controller.note_measured("chat")
        controller.note_resident(10000, reclaimable = 0)
        controller.register("raw", tokens = 3000, state = ParticipantState.STREAMING_RAW)
        assert controller.committed_tokens() == 13000
        # Its first data line: the prompt is resident, but the reading in hand is from
        # before it, and folding the charge into that reading would swallow it.
        controller.note_measured("raw")
        assert controller.committed_tokens() == 13000
        # The next reading has it, so it is no longer added on top.
        controller.note_resident(13000, reclaimable = 0)
        assert controller.committed_tokens() == 13000

    def test_a_failed_reading_does_not_fold_the_raw_charge(self):
        controller = _controller("http://raw-holder-unread")
        clock = [100.0]
        controller._clock = lambda: clock[0]
        controller.register("chat", tokens = 6000)
        controller.note_measured("chat")
        controller.note_resident(10000, reclaimable = 0)
        controller.register("raw", tokens = 3000, state = ParticipantState.STREAMING_RAW)
        controller.note_measured("raw")
        controller.note_resident(None)
        # One failed read keeps the last count, and the raw holder is still added on top.
        assert controller.committed_tokens() == 13000
        clock[0] += preemption._RESIDENT_HOLD_S + 1
        controller.note_resident(None)
        # The probe is gone: the ledger is the only figure, and it has every holder.
        assert controller.committed_tokens() == 9000
        controller.note_resident(13000, reclaimable = 0)
        assert controller.committed_tokens() == 13000

    def test_without_any_reading_the_mark_is_immediate(self):
        controller = _controller("http://raw-holder-noprobe")
        controller.register("raw", tokens = 3000, state = ParticipantState.STREAMING_RAW)
        controller.note_measured("raw")
        controller.note_resident(3000, reclaimable = 0)
        assert controller.committed_tokens() == 3000
