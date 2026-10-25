# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Long NPC conversations: older turns fold into a per-NPC summary that the reply prompt keeps in its stable prefix."""

import pytest

from core.flows.nodes import world as world_nodes
from core.flows.registry import FlowNodeError
from core.flows.world import (
    SUMMARY_CHARS,
    SUMMARY_SCHEMA,
    WorldError,
    clean_summary,
    npc_prompt,
    resolve_world,
)

from .flows_player_fakes import FakeCtx, run
from .test_flows_world import RAW
from .test_flows_world_sim import CLASSIC_PROMPT, CLASSIC_SYSTEM, _classic_state

SUMMARY = {"text": "Pip promised to bring me a blue lure. Pip's sister is called Wren.", "upto": 9}


def _turns(start, n):
    return [
        {"role": "player" if i % 2 == 0 else "npc", "text": f"line {i}", "seq": i}
        for i in range(start, start + n)
    ]


def test_summary_sits_in_the_stable_prefix():
    a_sys, a_prompt = npc_prompt(
        _classic_state(summary = SUMMARY, history = _turns(10, 4)), "Stay in character."
    )
    b_sys, _ = npc_prompt(
        _classic_state(summary = SUMMARY, history = _turns(10, 9), player_message = "and now?"),
        "Stay in character.",
    )
    assert SUMMARY["text"] in a_sys and SUMMARY["text"] not in a_prompt
    assert a_sys == b_sys
    assert npc_prompt(_classic_state(), "Stay in character.") == (CLASSIC_SYSTEM, CLASSIC_PROMPT)


def test_a_new_item_or_variable_keeps_the_conversation_in_the_cached_prefix():
    import os

    history = _turns(0, 8)
    _, before = npc_prompt(_classic_state(history = history), "Stay in character.")
    _, after = npc_prompt(
        _classic_state(history = history, inventory = ["rod", "pie"], vars = {"reputation": 5}),
        "Stay in character.",
    )
    shared = os.path.commonprefix([before, after])
    assert "line 7" in shared and before != after


def test_turns_already_summarised_are_not_repeated():
    _, prompt = npc_prompt(
        _classic_state(summary = SUMMARY, history = _turns(0, 14)), "Stay in character."
    )
    assert "line 9" not in prompt and "line 10" in prompt and "line 13" in prompt


def test_clean_summary_bounds_and_rejects_empty():
    assert clean_summary({"summary": "  We   met. "}) == "We met."
    long = " ".join(f"Sentence {i} is here." for i in range(100))
    out = clean_summary({"summary": long})
    assert len(out) <= SUMMARY_CHARS and out.endswith(".")
    for bad in ({"summary": "  "}, {}, None):
        with pytest.raises(WorldError):
            clean_summary(bad)


def test_summarise_task_folds_turns():
    state = {
        **_classic_state(),
        "task": "summarise",
        "summary": SUMMARY,
        "turns": _turns(10, 10),
    }
    ctx = FakeCtx(
        "world.npc_reply",
        inputs = {"state": state},
        llm = lambda **kw: {"summary": "Pip owes me a lure."},
    )
    out = run(world_nodes.world_npc_reply(ctx))
    assert out == {"reply": {"summary": "Pip owes me a lure."}}
    call = ctx.llm_calls[0]
    assert call["schema"] is SUMMARY_SCHEMA
    # Live calls send no fixed seed: Studio turns the prompt cache off for a seeded request.
    assert call.get("seed") is None
    assert SUMMARY["text"] in call["prompt"] and "line 19" in call["prompt"]
    ctx = FakeCtx("world.npc_reply", inputs = {"state": state}, llm = lambda **kw: {"summary": ""})
    with pytest.raises(FlowNodeError):
        run(world_nodes.world_npc_reply(ctx))


def test_summarise_setting_reaches_the_player():
    nodes = [{"id": "w", "type": "world.spec", "params": {"world": RAW}}]
    on = resolve_world({"nodes": [*nodes, {"id": "r", "type": "world.npc_reply", "params": {}}]})
    off = resolve_world(
        {"nodes": [*nodes, {"id": "r", "type": "world.npc_reply", "params": {"summarise": False}}]}
    )
    assert "summarise" not in on and off["summarise"] is False
