# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Generated tokens are written INTO the reservation, not on top of it.

Counting the charge and the output again made four chats holding 3472 cells read as 7568
against a 7424 ceiling, and the newest was preempted for room nobody needed.
"""

from __future__ import annotations

from core.inference import llama_preemption as preemption
from core.inference.llama_preemption import ParticipantState
import pytest

pytestmark = pytest.mark.usefixtures("preemption_opted_in")


_PROMPT = 100
_ALLOWANCE = 1024
_GENERATED = 768


def _controller(key: str) -> preemption.PreemptionController:
    controller = preemption.PreemptionController(key)
    controller.configure(budget = 8192, kv_unified = True, slots = 4, batch_tokens = 2048)
    return controller


def _four_chats(controller, *, split: bool):
    for index in range(4):
        controller.register(
            str(index),
            tokens = _PROMPT + _ALLOWANCE,
            prompt_tokens = _PROMPT if split else None,
        )
        controller.note_measured(str(index))
    controller.note_resident(4 * (_PROMPT + _GENERATED), reclaimable = 0)
    victims = []
    for index in range(4):
        victims.extend(p.gen_id for p in controller.observe(str(index), _GENERATED))
    return victims


class TestTheLedgerMatchesTheCache:
    def test_four_default_chats_are_counted_at_what_they_hold(self):
        controller = _controller("http://spent-allowance")
        victims = _four_chats(controller, split = True)

        assert controller.committed_tokens() == 4 * (_PROMPT + _GENERATED)
        assert victims == [], f"preempted {victims} with the cache less than half full"

    def test_without_the_split_the_reservation_is_still_carried_whole(self):
        """Nothing safe can be subtracted from a charge nobody split, so it stands."""
        controller = _controller("http://spent-allowance-unsplit")
        victims = _four_chats(controller, split = False)

        assert controller.committed_tokens() == 4 * (_PROMPT + _ALLOWANCE + _GENERATED)
        assert victims, "the unsplit charge is the arithmetic the finding described"

    def test_a_prompt_is_never_recorded_above_the_charge_it_came_from(self):
        controller = _controller("http://spent-allowance-clamp")
        participant = controller.register("one", tokens = 500, prompt_tokens = 9000)
        assert participant.prompt_tokens == 500


class TestAnOverrunIsStillSeen:
    """The whole point of the sweep: the charge is optimistic, so growth past it counts."""

    def test_generating_past_the_allowance_grows_the_ledger(self):
        controller = _controller("http://spent-allowance-overrun")
        controller.register("hog", tokens = _PROMPT + _ALLOWANCE, prompt_tokens = _PROMPT)
        controller.note_measured("hog")
        controller.observe("hog", 6000)

        assert controller.committed_tokens() >= _PROMPT + 6000


class TestARoundBoundaryRestatesTheConversation:
    def test_the_previous_round_s_output_is_inside_the_new_prompt(self):
        controller = _controller("http://spent-allowance-round")
        controller.register("loop", tokens = _PROMPT + _ALLOWANCE, prompt_tokens = _PROMPT)
        controller.note_measured("loop")
        controller.observe("loop", _GENERATED)
        assert controller.committed_tokens() == _PROMPT + _GENERATED

        # The round appends the answer and a tool result, and re-costs on the total.
        grown_prompt = _PROMPT + _GENERATED + 400
        controller.note_tokens("loop", grown_prompt + _ALLOWANCE, grown_prompt)
        assert controller.committed_tokens() == grown_prompt

        # And the next attempt's own count is added to THAT, not to the charge.
        controller.observe("loop", 50)
        assert controller.committed_tokens() == grown_prompt + 50


class TestAPausedHolderIsUnaffected:
    def test_a_pause_still_holds_no_cells(self):
        controller = _controller("http://spent-allowance-paused")
        controller.register("paused", tokens = _PROMPT + _ALLOWANCE, prompt_tokens = _PROMPT)
        controller.note_measured("paused")
        controller.note_state("paused", ParticipantState.DECODING)
        controller.observe("paused", _GENERATED)
        controller.set_state("paused", ParticipantState.PAUSED)

        assert controller.committed_tokens() == 0
