# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Two ways a parallel round's bookkeeping drifted from what the cache and the tools did.

A parked holder whose tool came back and re-charged before the reclaim scanned it keeps its
new commitment, so it must keep counting; and a padded tool name is the same tool to
`prepare_call`, so it is the same call to the overlap gate.
"""

from __future__ import annotations

from core.inference.llama_preemption import ParticipantState, PreemptionController
from core.inference.tool_loop_controller import ToolLoopController
import pytest

pytestmark = pytest.mark.usefixtures("preemption_opted_in")


class _RechargedLease:
    """Re-charged before the reclaim's yield: the real lease refuses a stale sequence."""

    def __init__(self):
        self.charge_seq = 7
        self.asked: list = []

    def yield_parked_commitment(self, *, charged_at = None):
        self.asked.append(charged_at)
        return 0 if charged_at != self.charge_seq else 1000


def _controller() -> PreemptionController:
    made = PreemptionController("http://127.0.0.1:65091")
    made.configure(budget = 16384, kv_unified = True, slots = 4)
    return made


class TestARefusedYieldRestoresTheHolder:
    def test_a_holder_recharged_before_the_scan_keeps_counting(self):
        controller = _controller()
        lease = _RechargedLease()
        controller.register("chat", lease = lease, tokens = 6000)
        controller.note_state("chat", ParticipantState.TOOLS_RUNNING)
        seen = controller.parked_holders()
        # The tool came back and the round re-charged; the ledger has its new prompt.
        lease.charge_seq += 1
        controller.note_tokens("chat", 6400, 6400)
        assert controller.note_cells_reclaimed(seen) == 1
        assert lease.asked == [7], "the yield names the commitment that parked"
        participant = controller.participant("chat")
        assert participant.cells_reclaimed is False and participant.holds_kv, (
            "the lease kept the new commitment, so the cells it is filling must count"
        )
        assert controller.snapshot().prefilling == 6400

    def test_a_granted_yield_still_releases_it(self):
        controller = _controller()
        lease = _RechargedLease()
        controller.register("chat", lease = lease, tokens = 6000)
        controller.note_state("chat", ParticipantState.TOOLS_RUNNING)
        lease.charge_seq = controller.participant("chat").park_charge_seq
        assert controller.note_cells_reclaimed(controller.parked_holders()) == 1
        assert controller.participant("chat").holds_kv is False


class TestAPaddedNameIsTheSameCall:
    def test_the_round_keys_match(self):
        tool = {
            "type": "function",
            "function": {
                "name": "terminal",
                "parameters": {"type": "object", "properties": {"command": {"type": "string"}}},
            },
        }
        controller = ToolLoopController(tools = [tool])
        calls = [
            {"function": {"name": "terminal", "arguments": '{"command": "echo hi"}'}},
            {"function": {"name": " terminal ", "arguments": '{"command": "echo hi"}'}},
        ]
        assert controller.call_key(calls[0]) == controller.call_key(calls[1])
