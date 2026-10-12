# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A round re-charged before a reclaim, and published after it, still replays its whole prompt.

The reclaim erases the parked holder's cells, then finds the lease re-charged and restores the
holder. The round that re-charged has not published yet; when it does, it read the restored
holder as a kept prefix, announced only its growth and marked itself measured.
"""

from __future__ import annotations

from core.inference.llama_preemption import ParticipantState, PreemptionController
import pytest

pytestmark = pytest.mark.usefixtures("preemption_opted_in")


class _RechargedLease:
    """The real lease refuses a yield for a commitment it no longer holds."""

    def __init__(self):
        self.charge_seq = 7

    def yield_parked_commitment(self, *, charged_at = None):
        return 0 if charged_at != self.charge_seq else 1000


def _parked_then_recharged():
    controller = PreemptionController("http://127.0.0.1:65093")
    controller.configure(budget = 16384, kv_unified = True, slots = 4)
    lease = _RechargedLease()
    controller.register("chat", lease = lease, tokens = 6000, prompt_tokens = 6000)
    controller.note_tokens("chat", 6000, 6000)
    controller.note_state("chat", ParticipantState.TOOLS_RUNNING)
    seen = controller.parked_holders()
    # The tool came back and `recost_waiting` re-charged, but the round has not published.
    lease.charge_seq += 1
    assert controller.note_cells_reclaimed(seen) == 1
    return controller


def test_the_published_round_announces_the_whole_prompt():
    controller = _parked_then_recharged()
    controller.note_tokens("chat", 6400, 6400)
    assert controller.snapshot().prefilling == 6400, (
        "the erase took every cell, so the round prefills its whole prompt, not its growth"
    )
    assert controller.participant("chat").measured is False


def test_the_marker_is_spent_by_that_round():
    controller = _parked_then_recharged()
    controller.note_tokens("chat", 6400, 6400)
    controller.observe("chat", 10)
    controller.note_tokens("chat", 6500, 6500)
    # The next round is growth again: the prompt plus the 10 tokens written are resident.
    assert controller.snapshot().prefilling == 6500 - 6410
    assert controller.participant("chat").measured is True


def test_a_token_first_spends_it_too():
    controller = _parked_then_recharged()
    controller.note_state("chat", ParticipantState.DECODING)
    controller.observe("chat", 5)
    assert controller.participant("chat").replay_pending is False


def test_the_sweep_after_the_round_keeps_it_unmeasured():
    # `_openai_llama_publish_round_charge` reports zero right after `note_tokens`: no token was
    # decoded, so the replayed prompt is still going in and its announcement stands.
    controller = _parked_then_recharged()
    controller.note_tokens("chat", 6400, 6400)
    controller.observe("chat", 0)
    assert controller.participant("chat").measured is False
    assert controller.snapshot().prefilling == 6400
