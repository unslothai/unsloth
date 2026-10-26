# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import copy

from core.flows.nodes import world as world_nodes
from core.flows.world import (
    chat_prompt,
    chat_schema,
    clean_options,
    memory_lines,
    normalize_world,
    options_blocks,
    validate_chat,
)

from .flows_player_fakes import run
from .test_flows_world_sim import SIM_RAW, SimCtx, _record

LIFE = {
    "options": True,
    "plan_cache": 999,
    "reflect_at": 150,
    "rate_batch": 0,
    "relationships": True,
}


def _world(**life):
    raw = copy.deepcopy(SIM_RAW)
    raw["simulation"] = {**raw["simulation"], **life}
    raw["items"] = [{"name": "Fishing Rod"}]
    return normalize_world(raw, seed = 1)


def test_life_fields_are_opt_in():
    classic = _world()["simulation"]
    assert not {"options", "plan_cache", "reflect_at", "rate_batch", "relationships"} & set(classic)
    sim = _world(**LIFE)["simulation"]
    assert sim["options"] is True and sim["relationships"] is True
    assert sim["plan_cache"] == 256 and sim["reflect_at"] == 150 and "rate_batch" not in sim
    assert "options" not in _world(options = "yes")["simulation"]


def test_option_plan_returns_termination_conditions():
    world = _world(**LIFE)
    seen = {}

    def llm(**kw):
        seen.update(kw)
        base = {"at": "00:00", "need": "energy", "above": 0, "item": ""}
        return {
            "options": [
                {
                    **base,
                    "activity": "Bake",
                    "place": "Cafe",
                    "action": "work",
                    "until": "time",
                    "at": "12:00",
                },
                {
                    **base,
                    "activity": "Eat",
                    "place": "Cafe",
                    "action": "eat",
                    "until": "need",
                    "need": "hunger",
                    "above": 90,
                },
                {
                    **base,
                    "activity": "Fetch",
                    "place": "Pier",
                    "action": "go_to",
                    "until": "item",
                    "item": "Fishing Rod",
                },
                {**base, "activity": "Walk", "place": "Pier", "action": "wander", "until": "place"},
                {
                    **base,
                    "activity": "Late",
                    "place": "Pier",
                    "action": "work",
                    "until": "time",
                    "at": "07:00",
                },
                {**base, "activity": "Moon", "place": "Moon", "action": "work", "until": "place"},
                {
                    **base,
                    "activity": "Odd",
                    "place": "Pier",
                    "action": "work",
                    "until": "item",
                    "item": "Ghost",
                },
            ]
        }

    state = {
        "world": world,
        "npc_id": "bea",
        "memory": [],
        "clock": 1440 + 600,
        "options": True,
        "needs": {"hunger": 40},
    }
    out = run(
        world_nodes.world_npc_plan(SimCtx("world.npc_plan", inputs = {"state": state}, llm = llm))
    )["plan"]
    assert out["options"] == [
        {
            "activity": "Bake",
            "place": "cafe",
            "kind": "work",
            "until": {"type": "time", "at": 1440 + 720},
        },
        {
            "activity": "Eat",
            "place": "cafe",
            "kind": "eat",
            "until": {"type": "need", "need": "hunger", "above": 90},
        },
        {
            "activity": "Fetch",
            "place": "pier",
            "kind": "go_to",
            "until": {"type": "item", "item": "fishing_rod"},
        },
        {
            "activity": "Walk",
            "place": "pier",
            "kind": "wander",
            "until": {"type": "place", "place": "pier"},
        },
    ]
    assert out["plan"][0] == {"start": 2040, "minutes": 120, "activity": "Bake", "place": "cafe"}
    assert len(out["plan"]) == 4 and out["day"] == 1
    item = seen["schema"]["properties"]["options"]["items"]
    assert item["properties"]["until"]["enum"] == ["time", "need", "item", "place"]
    assert item["properties"]["item"]["enum"] == ["", "Fishing Rod"]
    assert "hunger 40" in seen["prompt"] and "until 22:00" in seen["prompt"]


def test_option_plan_needs_both_world_flag_and_request():
    for world, state_flag in ((_world(**LIFE), False), (_world(), True)):
        ctx = SimCtx(
            "world.npc_plan",
            inputs = {
                "state": {"world": world, "npc_id": "bea", "clock": 600, "options": state_flag}
            },
            llm = lambda **kw: {"plan": []},
        )
        out = run(world_nodes.world_npc_plan(ctx))["plan"]
        assert "options" not in out and "plan" in ctx.llm_calls[0]["schema"]["properties"]


def test_rating_is_one_batched_call():
    world = _world(**LIFE)
    texts = [f"saw thing {i}" for i in range(40)]
    ctx = SimCtx(
        "world.npc_plan",
        inputs = {"state": {"world": world, "task": "rate", "texts": texts}},
        llm = lambda **kw: {"ratings": [12, 0, 7]},
    )
    out = run(world_nodes.world_npc_plan(ctx))["plan"]
    assert len(ctx.llm_calls) == 1
    assert out["ratings"][:3] == [10, 1, 7] and len(out["ratings"]) == 32
    assert (
        "32. saw thing 31" in ctx.llm_calls[0]["prompt"]
        and "saw thing 32" not in ctx.llm_calls[0]["prompt"]
    )
    empty = SimCtx(
        "world.npc_plan", inputs = {"state": {"world": world, "task": "rate", "texts": []}}
    )
    assert run(world_nodes.world_npc_plan(empty))["plan"] == {"ratings": []} and not empty.llm_calls


def test_reflection_is_one_call_with_citations():
    world = _world(**LIFE)
    memory = [_record(i, f"fact {i}") for i in range(3)]
    memory[1]["source"] = ["old_tom"]
    ctx = SimCtx(
        "world.npc_plan",
        inputs = {"state": {"world": world, "task": "reflect", "npc_id": "bea", "memory": memory}},
        llm = lambda **kw: {
            "insights": [
                {"text": " Tom  gossips ", "cites": [2, 9, True], "importance": 7},
                {"text": ""},
            ]
        },
    )
    out = run(world_nodes.world_npc_plan(ctx))["plan"]
    assert len(ctx.llm_calls) == 1
    assert out["insights"] == [{"text": "Tom gossips", "importance": 7, "cites": ["m1"]}]
    assert "2. fact 1 (heard from Old Tom)" in ctx.llm_calls[0]["prompt"]


def test_social_chat_maps_rumour_sources_and_feelings():
    world = _world(**LIFE)
    a, b = world["npcs"]
    raw = {
        "lines": [{"speaker": "Bea", "text": "Hi"}, {"speaker": "Old Tom", "text": "Hmph"}],
        "a_learns": [
            {"text": "The pier is closed", "importance": 6, "from": 1},
            {"text": "", "from": 1},
            {"text": "Rain", "importance": 2, "from": 0},
        ],
        "b_learns": [{"text": "Party tonight", "importance": 7, "from": 5}],
        "a_feels": {"affinity": 9, "fact": "  Tom fishes at dawn "},
        "b_feels": {"affinity": -1, "fact": ""},
    }
    heard = _record(4, "The pier is closed")
    heard["source"] = ["bea"]
    state = {
        "world": world,
        "a": a["id"],
        "b": b["id"],
        "clock": 600,
        "memory_a": [],
        "memory_b": [heard],
        "rel_a": {"affinity": 4, "facts": ["owes me a pie"]},
    }
    ctx = SimCtx("world.npc_chat", inputs = {"state": state}, llm = lambda **kw: copy.deepcopy(raw))
    chat = run(world_nodes.world_npc_chat(ctx))["chat"]
    assert chat["a_learns"] == [
        {"text": "The pier is closed", "importance": 6, "from": "m4"},
        {"text": "Rain", "importance": 2},
    ]
    assert chat["b_learns"] == [{"text": "Party tonight", "importance": 7}]
    assert chat["a_feels"] == {"affinity": 3, "fact": "Tom fishes at dawn"}
    assert chat["b_feels"] == {"affinity": -1, "fact": ""}
    call = ctx.llm_calls[0]
    assert {"a_feels", "b_feels"} <= set(call["schema"]["required"])
    assert "Bea feels warm towards Old Tom (affinity +4); knows: owes me a pie" in call["prompt"]
    assert "1. The pier is closed (heard from Bea)" in call["prompt"]


def test_classic_chat_and_memory_text_are_unchanged():
    world = _world()
    a, b = world["npcs"]
    assert chat_schema(a, b, False) == chat_schema(a, b)
    assert "a_feels" not in chat_schema(a, b)["properties"]
    mem = [_record(0, "x")]
    assert chat_prompt(world, a, b, mem, mem, {"clock": 5}) == chat_prompt(
        world, a, b, mem, mem, {"clock": 5}, False
    )
    raw = {
        "lines": [{"speaker": "Bea", "text": "Hi"}],
        "a_learns": [{"text": "y", "from": 1}],
        "b_learns": [],
    }
    assert validate_chat(raw, a, b) == {
        "a": a["id"],
        "b": b["id"],
        "lines": [{"npc": "bea", "text": "Hi"}],
        "a_learns": [{"text": "y", "importance": 4}],
        "b_learns": [],
    }
    assert memory_lines([{"text": "x"}]) == "- x"
    chain = {"text": "x", "source": ["bea", "old_tom", "player"]}
    assert memory_lines([chain], {"bea": "Bea", "old_tom": "Old Tom"}) == (
        "- x (heard from Bea, who heard it from Old Tom, who heard it from the player)"
    )


def test_reply_prompt_shows_who_told_the_npc():
    world = _world(**LIFE)
    rec = _record(0, "The mill burned down", 8)
    rec["source"] = ["old_tom"]
    state = {
        "world": world,
        "npc_id": "bea",
        "history": [],
        "player_message": "Who told you?",
        "memory": [rec],
        "clock": 600,
    }
    ctx = SimCtx(
        "world.npc_reply", inputs = {"state": state}, llm = lambda **kw: {"text": "Old Tom did."}
    )
    run(world_nodes.world_npc_reply(ctx))
    assert "- The mill burned down (heard from Old Tom)" in ctx.llm_calls[0]["prompt"]


def test_clean_options_drops_past_times_and_blocks_chain():
    world = _world(**LIFE)
    base = {
        "place": "Cafe",
        "action": "eat",
        "need": "fun",
        "above": 50,
        "item": "",
        "activity": "x",
    }
    opts = clean_options(
        {"options": [{**base, "until": "time", "at": "09:00"}, {**base, "until": "need"}]},
        world,
        600,
    )
    assert opts == [
        {
            "activity": "x",
            "place": "cafe",
            "kind": "eat",
            "until": {"type": "need", "need": "fun", "above": 50},
        }
    ]
    assert options_blocks(opts, 600) == [
        {"start": 600, "minutes": 60, "activity": "x", "place": "cafe"}
    ]
