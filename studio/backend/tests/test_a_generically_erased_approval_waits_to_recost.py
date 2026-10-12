# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A round recosting after the generic idle-slot reclaim erased its parked cells may wait.

The route's own approval erase sets a flag that lets the next round yield and wait for room.
The generic reclaim tells only the ledger, so where proactive idle clearing is off (Windows full
GPU offload) the grown round was refused at once, and a tool that had already run never had its
result sent back to the model.
"""

from __future__ import annotations

import inspect

from core.inference.llama_preemption import ParticipantState, PreemptionController
import routes.inference as inference
import pytest

pytestmark = pytest.mark.usefixtures("preemption_opted_in")

BASE = "http://127.0.0.1:65075"


class _Backend:
    base_url = BASE
    _kv_cache_unified = True
    context_length = 8192


class _Lease:
    charge_seq = 3

    def yield_parked_commitment(self, *, charged_at = None):
        return 1000


@pytest.fixture
def controller(monkeypatch):
    made = PreemptionController(BASE)
    made.configure(budget = 8192, kv_unified = True, slots = 4)
    monkeypatch.setattr(inference, "get_preemption_controller", lambda key: made)
    return made


def _parked(controller) -> None:
    controller.register("chat", lease = _Lease(), tokens = 4000, prompt_tokens = 4000)
    controller.note_tokens("chat", 4000, 4000)
    controller.note_state("chat", ParticipantState.PARKED_ON_TOOL)


def test_a_generic_reclaim_lets_the_next_round_wait(controller):
    _parked(controller)
    assert inference._openai_llama_self_erased(_Backend(), "chat") is False
    # The token path's idle reclaim takes every idle slot, this parked chat's included.
    assert controller.note_cells_reclaimed(controller.parked_holders()) == 1
    # Answered: the tool runs, and the next round recosts before it decodes.
    controller.note_state("chat", ParticipantState.TOOLS_RUNNING)
    assert inference._openai_llama_self_erased(_Backend(), "chat") is True
    controller.note_state("chat", ParticipantState.DECODING)
    assert inference._openai_llama_self_erased(_Backend(), "chat") is False


def test_a_reclaim_that_found_the_lease_recharged_still_counts(controller):
    _parked(controller)
    lease = controller.participant("chat").lease
    seen = controller.parked_holders()
    lease.charge_seq += 1
    assert controller.note_cells_reclaimed(seen) == 1
    # Restored to count again, but its cells were erased all the same.
    assert inference._openai_llama_self_erased(_Backend(), "chat") is True


def test_an_unknown_holder_or_the_switch_off_is_not_erased(controller, monkeypatch):
    assert inference._openai_llama_self_erased(_Backend(), "gone") is False
    monkeypatch.setattr(inference, "preemption_enabled", lambda: False)
    assert inference._openai_llama_self_erased(_Backend(), "gone") is False


def test_both_tool_loops_recost_with_it():
    source = inspect.getsource(inference)
    assert (
        "cache_is_empty = erased\n                    or _openai_llama_self_erased(llama_backend, completion_id),"
        in source
    )
    anthropic = source[source.index("def _anthropic_recost(") :]
    anthropic = anthropic[: anthropic.index("_openai_llama_publish_round_charge(")]
    assert "cache_is_empty = _openai_llama_self_erased(llama_backend, message_id)," in anthropic
