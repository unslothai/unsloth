# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A chat that answers a tool result with another tool call must still be pausable.

WHAT WENT WRONG

Four browser chats on the 35B model at -c 8192, 2026-09-05 02:54. Chat 0 wrote its
answer through a terminal tool (`cat > MANUAL.md << EOF ...`), so its second round was
one long tool call: tool-call deltas, no content. The route had reported TOOLS_RUNNING at
the first tool start and would report DECODING only at the next content chunk, which
never came. TOOLS_RUNNING is not preemptable. The sweep saw two holders, one of them
unpreemptable, spared the other as the last one standing, and chose nobody while the
pool climbed 5216 -> 8288 past a 6136 ceiling. llama-server then ended both chats with
`Context size has been exceeded`. Reproduced in temp/repro_tools_running.py.

THE RULES THIS PINS

1. A generated token is proof of decoding: `observe` moves a TOOLS_RUNNING or
   PARKED_ON_TOOL holder to DECODING by itself.
2. The sweep leaves one HOLDER standing, not one preemptable victim. Beside a holder it
   cannot choose (a raw passthrough, a chat genuinely in a tool), every preemptable chat
   may be chosen; a holder already PREEMPTING is on its way out and does not count.
3. The tool-loop route reports DECODING on streamed tool arguments, and compares against
   the ledger's own state rather than a local mirror, so a state the controller moved on
   its own is not mistaken for a repeat.
"""

from __future__ import annotations

import pytest

from core.inference.llama_preemption import (
    ParticipantState,
    PreemptionController,
)
import routes.inference as inference_route


def _controller() -> PreemptionController:
    made = PreemptionController("test://toolcall")
    made.configure(budget = 8192, kv_unified = True, slots = 4, draft_tokens = 2, batch_tokens = 2048)
    return made


class TestObserveMeansDecoding:
    def test_tokens_arriving_move_a_tool_holder_back_to_decoding(self):
        c = _controller()
        chat = c.register("chat", tokens = 1000)
        c.note_state("chat", ParticipantState.TOOLS_RUNNING)
        assert chat.preemptable is False
        c.observe("chat", 32)
        assert chat.state == ParticipantState.DECODING
        assert chat.preemptable is True
        assert chat.tokens == 1032

    def test_a_parked_holder_that_produces_tokens_is_decoding_too(self):
        c = _controller()
        chat = c.register("chat", tokens = 1000)
        c.note_state("chat", ParticipantState.PARKED_ON_TOOL)
        c.observe("chat", 1)
        assert chat.state == ParticipantState.DECODING

    def test_the_run_that_failed(self):
        """The shape of 02:54: leader decoding, the other holder streaming a tool call."""
        c = _controller()
        leader = c.register("leader", tokens = 1000)
        c.note_tokens("leader", 3000)
        tool_chat = c.register("toolchat", tokens = 1000)
        c.note_tokens("toolchat", 1500)
        c.note_state("toolchat", ParticipantState.TOOLS_RUNNING)

        chosen = []
        for generated in (500, 1000, 1500, 2000):
            chosen.extend(c.observe("toolchat", generated))
        # Past the 6136 ceiling with the tool chat's own tokens: somebody must have
        # been chosen, and both remain accounted for.
        assert chosen, "the pool passed its ceiling and nobody was paused"
        assert {v.gen_id for v in chosen} <= {"leader", "toolchat"}
        assert c.snapshot().committed > 6136
        # The victim is signalled and the other holder is left standing.
        victim = chosen[0]
        assert victim.state == ParticipantState.PREEMPTING
        assert victim.preempt_event.is_set()
        other = leader if victim is tool_chat else tool_chat
        assert other.state == ParticipantState.DECODING


class TestOneHolderStanding:
    def test_an_unpreemptable_holder_counts_as_the_one_left_standing(self):
        c = _controller()
        raw = c.register("raw", tokens = 4000, state = ParticipantState.STREAMING_RAW)
        c.note_tokens("raw", 4000)
        chat = c.register("chat", tokens = 1000)
        c.note_tokens("chat", 1000)
        # Room while under the ceiling.
        assert c.observe("chat", 1000) == []
        # Over it, the only preemptable holder is chosen rather than spared, because
        # the raw stream is standing already and pausing nobody ends both.
        victims = c.observe("chat", 1500)
        assert [v.gen_id for v in victims] == ["chat"]
        assert raw.state == ParticipantState.STREAMING_RAW
        assert chat.state == ParticipantState.PREEMPTING

    def test_a_holder_already_preempting_does_not_count_as_standing(self):
        c = _controller()
        a = c.register("a", tokens = 1000)
        c.note_tokens("a", 3500)
        b = c.register("b", tokens = 1000)
        c.note_tokens("b", 3500)
        first = c.observe("b", 2500)
        assert len(first) == 1
        chosen = first[0]
        # A sweep between the decision and the pause must not take the last decoder.
        again = c.observe("a" if chosen is b else "b", 2600)
        assert again == []
        survivor = a if chosen is b else b
        assert survivor.state == ParticipantState.DECODING

    def test_a_lone_preemptable_holder_is_never_paused(self):
        c = _controller()
        only = c.register("only", tokens = 1000)
        assert c.observe("only", 7000) == []
        assert only.state == ParticipantState.DECODING


class TestTheRouteReports:
    def test_note_state_reads_the_ledger_not_a_mirror(self, monkeypatch):
        """After the controller moved the chat to DECODING by itself, a later
        TOOLS_RUNNING report must still go through."""
        controller = _controller()
        monkeypatch.setattr(
            inference_route, "get_preemption_controller", lambda key: controller
        )
        backend = type("B", (), {"base_url": "http://tool.test"})()
        _refresh, _observe, note_state = inference_route._openai_llama_residency_observer(
            llama_backend = backend, completion_id = "chat"
        )
        chat = controller.register("chat", tokens = 1000)
        assert note_state(ParticipantState.TOOLS_RUNNING) == ParticipantState.TOOLS_RUNNING
        assert chat.state == ParticipantState.TOOLS_RUNNING
        # Tokens arrive: the ledger says DECODING on its own.
        controller.observe("chat", 32)
        assert note_state(None) == ParticipantState.DECODING
        # The next tool start is not a repeat of the earlier report.
        assert note_state(ParticipantState.TOOLS_RUNNING) == ParticipantState.TOOLS_RUNNING
        assert chat.state == ParticipantState.TOOLS_RUNNING
