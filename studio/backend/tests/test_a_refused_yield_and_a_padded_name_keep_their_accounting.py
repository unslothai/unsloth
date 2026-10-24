# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Two ways a parallel round's bookkeeping drifted from what the cache and the tools did."""

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
        lease.charge_seq += 1
        controller.note_tokens("chat", 6400, 6400)
        assert controller.note_cells_reclaimed(seen) == 1
        assert lease.asked == [7], "the yield names the commitment that parked"
        participant = controller.participant("chat")
        assert (
            participant.cells_reclaimed is False and participant.holds_kv
        ), "the lease kept the new commitment, so the cells it is filling must count"
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


class TestAReplayedPromptIsNotMeasuredUntilItIsIn:
    def test_a_round_after_an_erase_counts_on_top_of_the_reading(self):
        controller = _controller()
        controller.register("chat", lease = _RechargedLease(), tokens = 6000)
        controller.note_tokens("chat", 6000, 6000)
        controller.note_state("chat", ParticipantState.PARKED_ON_TOOL)
        lease = controller.participant("chat").lease
        lease.charge_seq = controller.participant("chat").park_charge_seq
        assert controller.note_cells_reclaimed(controller.parked_holders()) == 1
        controller.note_resident(5000, started_at_seq = controller.residency_epoch())
        controller.note_tokens("chat", 6200, 6200)
        assert controller.participant("chat").measured is False
        assert (
            controller.snapshot().committed == 5000 + 6200
        ), "the replay was folded into cells the cache holds for someone else"
        controller.observe("chat", 1)
        controller.note_resident(11200, started_at_seq = controller.residency_epoch())
        assert controller.participant("chat").measured is True


class TestAPaddedNameIsGatedAsItsTool:
    def test_the_round_preflight_asks_for_a_padded_terminal(self, monkeypatch):
        from state import tool_policy
        call = {"function": {"name": " terminal ", "arguments": '{"command": "rm -rf x"}'}}
        assert (
            tool_policy.round_may_prompt(
                [call],
                confirm_tool_calls = True,
                bypass_permissions = False,
                permission_mode = "off",
                sandbox_level = "low",
                is_high_risk = lambda name, arguments: name == "terminal",
                never_needs = lambda name: False,
            )
            is True
        )


class TestAChoiceThatLandsAsAToolStartsIsUndone:
    def test_the_tool_is_recorded_and_the_signal_cleared(self):
        controller = _controller()
        controller.register("old", lease = _RechargedLease(), tokens = 9000)
        chat = controller.register("chat", lease = _RechargedLease(), tokens = 9000)
        chosen = [p.gen_id for p in controller.plan_preemptions(needed = 4000)]
        assert chosen == ["chat"]
        assert chat.state == ParticipantState.PREEMPTING and chat.preempt_event.is_set()
        assert controller.note_state("chat", ParticipantState.TOOLS_RUNNING) is True
        assert chat.state == ParticipantState.TOOLS_RUNNING
        assert not chat.preempt_event.is_set()
        assert chat.consecutive_preemptions == 0


class TestTheFirstPrefillIsThePromptOnly:
    def test_registration_announces_the_prompt_not_the_allowance(self):
        controller = _controller()
        controller.register("chat", lease = _RechargedLease(), tokens = 9000, prompt_tokens = 500)
        assert controller.snapshot().prefilling == 500
        controller.restart("chat")
        assert controller.snapshot().prefilling == 500

    def test_an_unknown_split_still_announces_the_charge(self):
        controller = _controller()
        controller.register("chat", lease = _RechargedLease(), tokens = 9000)
        assert controller.snapshot().prefilling == 9000
