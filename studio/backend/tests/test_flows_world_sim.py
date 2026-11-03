# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import copy

import pytest

from core.flows.nodes import world as world_nodes
from core.flows.registry import FlowNodeError
from core.flows.schema import WorldSpec
from core.flows.world import (
    NPC_MEMORY_REPLY_SCHEMA,
    NPC_REPLY_SCHEMA,
    WORLD_ENTRY_TYPES,
    clock_text,
    normalize_world,
    npc_prompt,
    resolve_world_report,
    validate_chat,
)

from .flows_player_fakes import FakeCtx, run
from .test_flows_world import RAW

# The classic prompt, frozen before simulation existed: worlds without it must prompt byte for byte the same.
CLASSIC_SYSTEM = (
    'You are Old Tom, a character in the game world "Mossy Hollow".\nPersona: grumpy fisher\nWorld: A sleepy village.\n'
    "Stay in character.\nReply as JSON. `text` is what you say. Only set `complete_quest` to one of your quest ids "
    "when the player has earned it (they carry the needed item). Use `give_item` only for an item id from the world "
    "list that fits the story; use empty strings when nothing happens.\nSet `action` to follow_player when you "
    "agree to come along with the player, stop_following when you part ways, or go_to with a `place` id when you "
    "agree to head somewhere: Pond (id pond), Mill (id mill). Leave it empty otherwise."
)
CLASSIC_PROMPT = (
    "Conversation so far:\nPlayer: hello\nOld Tom: Hmph.\nYour quests:\n- Lost rod (id q): Find my rod Needs item "
    "id rod.\nWorld items: Fishing Rod (id rod), Berry Pie (id pie)\nPlayer (Pip) carries: Berry Pie (id pie)\n"
    "Variables: {'reputation': 2}\nPlayer: Seen my rod?\nOld Tom:"
)

SIM_RAW = {
    **RAW,
    "npcs": [
        {
            "name": "Bea",
            "persona": "cheerful baker",
            "routine": "bakes at dawn, sells bread at the market",
            "relationships": {"Old Tom": "her grumpy neighbour", "Nobody": "x"},
            "seed_memories": [
                "I am hosting a party",
                {"text": "Tom owes me a pie", "importance": 12},
            ],
        },
        {"name": "Old Tom", "persona": "grumpy fisher"},
    ],
    "simulation": {
        "places": [
            {"name": "Cafe", "x": 2, "y": 2, "w": 4, "h": 3},
            {"name": "Pier", "x": 9, "y": 9, "w": 3, "h": 2},
        ]
    },
}


class SimCtx(FakeCtx):
    def __init__(
        self,
        *a,
        embeddings: bool = False,
        **kw,
    ):
        super().__init__(*a, **kw)
        self.embeddings = embeddings
        self.embedded: list[list[str]] = []

    async def call(
        self,
        method,
        path,
        json = None,
        **kw,
    ):
        assert (method, path) == ("POST", "/v1/embeddings")
        self.embedded.append(list(json["input"]))
        if not self.embeddings:
            raise FlowNodeError("no embedding model")
        vec = lambda t: [t.lower().count(c) + 0.01 for c in "abcdefghijklmnopqrstuvwxyz"]
        return {"data": [{"index": i, "embedding": vec(t)} for i, t in enumerate(json["input"])]}


def _classic_state(**kw):
    state = {
        "world": normalize_world(RAW, seed = 1),
        "npc_id": "bea",
        "history": [{"role": "player", "text": "hello"}, {"role": "npc", "text": "Hmph."}],
        "player_message": "Seen my rod?",
        "vars": {"reputation": 2},
        "inventory": ["pie"],
        "completed_quests": [],
    }
    state.update(kw)
    return state


def _record(
    i,
    text,
    importance = 3,
    t = 8.0,
):
    return {
        "id": f"m{i}",
        "text": text,
        "kind": "observation",
        "t_created": t,
        "t_access": t,
        "importance": importance,
    }


def test_worlds_without_simulation_are_unchanged():
    w = normalize_world(RAW, seed = 1)
    assert "simulation" not in w
    assert all(not {"routine", "relationships", "seed_memories"} & set(n) for n in w["npcs"])
    assert npc_prompt(_classic_state(), "Stay in character.") == (CLASSIC_SYSTEM, CLASSIC_PROMPT)
    ctx = SimCtx(
        "world.npc_reply",
        params = {"rules": "Stay in character."},
        inputs = {"state": _classic_state()},
        llm = lambda **kw: {"text": "Hmph.", "remember": [{"text": "x", "importance": 3}]},
    )
    reply = run(world_nodes.world_npc_reply(ctx))["reply"]
    assert reply == {"text": "Hmph."}
    call = ctx.llm_calls[0]
    assert (call["system"], call["prompt"]) == (CLASSIC_SYSTEM, CLASSIC_PROMPT)
    assert call["schema"] is NPC_REPLY_SCHEMA and call["max_tokens"] == 320 and not ctx.embedded
    _, warnings = resolve_world_report(
        {"nodes": [{"id": "w", "type": "world.spec", "params": {"world": RAW}}]}
    )
    assert not any("Simulation" in x for x in warnings)


def test_simulation_section_is_normalised():
    w = normalize_world(SIM_RAW, seed = 3)
    WorldSpec.model_validate(w)
    sim = w["simulation"]
    assert [p["id"] for p in sim["places"]] == ["cafe", "pier"]
    assert (
        sim["minutes_per_second"] == 2.0
        and sim["calls_per_minute"] == 4
        and sim["start_minute"] == 480
    )
    bea = w["npcs"][0]
    assert bea["relationships"] == {"old_tom": "her grumpy neighbour"}
    assert bea["seed_memories"] == [
        {"text": "I am hosting a party", "importance": 5},
        {"text": "Tom owes me a pie", "importance": 10},
    ]
    from_regions = normalize_world({**RAW, "simulation": True}, seed = 1)["simulation"]["places"]
    assert [p["name"] for p in from_regions] == ["Pond", "Mill"]
    bare = normalize_world(
        {"npcs": [{"name": "A"}], "simulation": {"calls_per_minute": 999}}, seed = 1
    )
    assert [p["id"] for p in bare["simulation"]["places"]] == ["town_square"] and bare[
        "simulation"
    ]["calls_per_minute"] == 30
    assert {"npc_plan", "npc_chat"} <= set(WORLD_ENTRY_TYPES.values())


def test_reply_with_memory_retrieves_and_records_in_one_call():
    world = normalize_world(SIM_RAW, seed = 1)
    memory = [
        _record(0, "I am hosting a Valentine's Day party at the cafe", 8),
        _record(1, "The weather was nice", 1),
        _record(2, "I sold bread to Tom", 3),
    ]
    state = {
        "world": world,
        "npc_id": "bea",
        "history": [],
        "player_message": "Any party plans?",
        "memory": memory,
        "clock": 600,
    }
    ctx = SimCtx(
        "world.npc_reply",
        inputs = {"state": state},
        llm = lambda **kw: {
            "text": "Come to my party!",
            "remember": [
                {"text": "The player asked about my party", "importance": 4},
                {"text": " "},
                {"text": "b"},
                {"text": "c"},
            ],
        },
    )
    reply = run(world_nodes.world_npc_reply(ctx))["reply"]
    assert reply["text"] == "Come to my party!"
    assert reply["remember"] == [
        {"text": "The player asked about my party", "importance": 4},
        {"text": "b", "importance": 4},
    ]
    assert reply["recalled"][0] == "m0" and len(reply["recalled"]) == 3
    assert len(ctx.llm_calls) == 1
    call = ctx.llm_calls[0]
    assert call["schema"] is NPC_MEMORY_REPLY_SCHEMA
    assert (
        "It is Day 1, 10:00." in call["prompt"]
        and "- I am hosting a Valentine's Day party" in call["prompt"]
    )
    assert (
        "About Old Tom: her grumpy neighbour" in call["system"] and "`remember`" in call["system"]
    )
    assert call["system"].startswith(
        npc_prompt({**state, "memory": None}, "Stay in character. Keep replies under 60 words.")[0]
    )
    assert ctx.embedded and any("keywords" in m for _, m in ctx.logs)


def test_recall_uses_embeddings_and_caches_vectors():
    world_nodes._VECTORS.clear()
    memory = [_record(i, t) for i, t in enumerate(["zzz party", "apples", "zebra zoo"])]
    ctx = SimCtx("world.npc_reply", embeddings = True)
    picked = run(world_nodes._recall(ctx, memory, "zzz", 9.0, 2))
    assert [r["id"] for r in picked] == ["m0", "m2"]
    assert ctx.embedded == [["zzz", "zzz party", "apples", "zebra zoo"]]
    run(world_nodes._recall(ctx, memory, "apples", 9.0, 2))
    assert ctx.embedded[1] == ["apples"]
    assert run(world_nodes._recall(ctx, None, "x", 0.0)) == []
    assert "embedding" not in memory[0]


def test_recall_does_not_reuse_vectors_from_another_embedding_model():
    world_nodes._VECTORS.clear()
    spaces = {
        "old": {"fruit": [1.0, 0.0], "apple": [1.0, 0.0], "pear": [0.0, 1.0]},
        "new": {"fruit": [1.0, 0.0], "apple": [0.0, 1.0], "pear": [1.0, 0.0]},
    }
    active = ["old"]
    embedded = []

    class TwoModels(SimCtx):
        async def call(
            self,
            method,
            path,
            json = None,
            **kw,
        ):
            embedded.append(list(json["input"]))
            space = spaces[active[0]]
            data = [{"index": i, "embedding": space[t]} for i, t in enumerate(json["input"])]
            return {"model": active[0], "data": data}

    ctx = TwoModels("world.npc_reply")
    memory = [_record(0, "apple", 5), _record(1, "pear", 5)]
    assert run(world_nodes._recall(ctx, memory, "fruit", 9.0, 1))[0]["id"] == "m0"
    active[0] = "new"
    run(world_nodes._recall(ctx, memory, "fruit", 9.0, 1))
    picked = run(world_nodes._recall(ctx, memory, "fruit", 9.0, 1))
    assert embedded[-1] == ["fruit", "apple", "pear"]
    assert picked[0]["id"] == "m1"


def test_plan_places_are_restricted():
    world = normalize_world(SIM_RAW, seed = 1)
    seen = {}

    def llm(**kw):
        seen.update(kw)
        return {
            "plan": [
                {"start": "07:00", "minutes": 60, "activity": "Bake", "place": "Cafe"},
                {"start": "09:00", "minutes": 90, "activity": "Fly", "place": "Moon"},
                {"start": "25:00", "minutes": 60, "activity": "x", "place": "Pier"},
                {"start": "12:30", "minutes": 9999, "activity": "Fish", "place": "Pier"},
            ]
        }

    state = {
        "world": world,
        "npc_id": "bea",
        "memory": [_record(0, "Party at the cafe at 17:00", 8)],
        "clock": 1440 + 400,
    }
    out = run(
        world_nodes.world_npc_plan(SimCtx("world.npc_plan", inputs = {"state": state}, llm = llm))
    )["plan"]
    assert seen.get("seed") is None
    assert out["npc"] == "bea" and out["day"] == 1
    assert out["plan"] == [
        {"start": 1440 + 420, "minutes": 60, "activity": "Bake", "place": "cafe"},
        {"start": 1440 + 750, "minutes": 720, "activity": "Fish", "place": "pier"},
    ]
    place = seen["schema"]["properties"]["plan"]["items"]["properties"]["place"]
    assert place["enum"] == ["Cafe", "Pier"]
    assert "It is day 2" in seen["prompt"] and "Party at the cafe" in seen["prompt"]
    with pytest.raises(FlowNodeError):
        run(
            world_nodes.world_npc_plan(SimCtx("world.npc_plan", inputs = {"state": _classic_state()}))
        )


def test_encounter_output_schema():
    world = normalize_world(SIM_RAW, seed = 1)
    a, b = world["npcs"]
    raw = {
        "lines": [
            {"speaker": "Bea", "text": "Party at the cafe tonight!"},
            {"speaker": "Ghost", "text": "boo"},
            {"speaker": "old tom", "text": "  Hmph,   fine. "},
        ],
        "a_learns": [],
        "b_learns": [{"text": "Bea is hosting a party at the cafe tonight", "importance": 7}],
    }
    state = {
        "world": world,
        "a": a["id"],
        "b": b["id"],
        "clock": 1000,
        "place": "Cafe",
        "memory_a": [_record(0, "I am hosting a party", 8)],
        "memory_b": [],
    }
    ctx = SimCtx("world.npc_chat", inputs = {"state": state}, llm = lambda **kw: copy.deepcopy(raw))
    chat = run(world_nodes.world_npc_chat(ctx))["chat"]
    assert ctx.llm_calls[0].get("seed") is None
    assert chat["lines"] == [
        {"npc": a["id"], "text": "Party at the cafe tonight!"},
        {"npc": b["id"], "text": "Hmph, fine."},
    ]
    assert chat["b_learns"] == [
        {"text": "Bea is hosting a party at the cafe tonight", "importance": 7}
    ]
    assert chat["a_learns"] == [] and chat["a_recalled"] == ["m0"] and chat["b_recalled"] == []
    schema = ctx.llm_calls[0]["schema"]
    assert schema["properties"]["lines"]["items"]["properties"]["speaker"]["enum"] == [
        "Bea",
        "Old Tom",
    ]
    assert schema["properties"]["lines"]["maxItems"] == 6
    assert (
        "at Cafe" in ctx.llm_calls[0]["prompt"]
        and "I am hosting a party" in ctx.llm_calls[0]["prompt"]
    )
    with pytest.raises(FlowNodeError):
        run(
            world_nodes.world_npc_chat(
                SimCtx("world.npc_chat", inputs = {"state": {**state, "b": a["id"]}})
            )
        )
    with pytest.raises(FlowNodeError):
        run(
            world_nodes.world_npc_chat(
                SimCtx("world.npc_chat", inputs = {"state": state}, llm = lambda **kw: {"lines": []})
            )
        )
    with pytest.raises(Exception):
        validate_chat("nope", a, b)


def test_clock_text():
    assert clock_text(0) == "Day 1, 00:00"
    assert clock_text(1440 + 17 * 60 + 5) == "Day 2, 17:05"
    assert clock_text(None) == "Day 1, 00:00"


def test_simulation_without_plan_or_chat_nodes_warns():
    doc = {
        "nodes": [
            {"id": "w", "type": "world.spec", "params": {"world": SIM_RAW}},
            {"id": "r", "type": "world.npc_reply"},
        ]
    }
    world, warnings = resolve_world_report(doc)
    assert world["entry_points"] == {"npc_reply": "r"}
    assert any("Simulation" in x for x in warnings)
    doc["nodes"] += [{"id": "p", "type": "world.npc_plan"}, {"id": "c", "type": "world.npc_chat"}]
    world, warnings = resolve_world_report(doc)
    assert (
        world["entry_points"] == {"npc_reply": "r", "npc_plan": "p", "npc_chat": "c"}
        and not warnings
    )


def test_the_memory_prompt_keeps_a_stable_prefix_across_turns():
    state = {
        "world": normalize_world(SIM_RAW, seed = 1),
        "npc_id": "bea",
        "history": [{"role": "player", "text": "hello"}, {"role": "npc", "text": "Morning!"}],
        "player_message": "Any party plans?",
        "clock": 8 * 60,
    }
    first = npc_prompt(state, "Stay in character.", [_record(0, "I bake at dawn")])
    later = npc_prompt(
        {**state, "clock": 9 * 60}, "Stay in character.", [_record(1, "Tom owes me a pie")]
    )
    assert first[0] == later[0]
    stable = first[1].split("It is ")[0]
    assert stable and later[1].startswith(stable) and "Conversation so far" in stable


def test_a_slow_embedder_never_holds_up_a_live_reply(monkeypatch):
    import asyncio
    import time

    world_nodes._VECTORS.clear()
    calls = []

    async def slow_embed(ctx, texts):
        calls.append(list(texts))
        await asyncio.sleep(1.5)
        return [[1.0, 0.0] for _ in texts], None

    monkeypatch.setattr(world_nodes, "_embed", slow_embed)
    monkeypatch.setattr(world_nodes, "REPLY_RECALL_BUDGET_S", 0.2)
    memory = [
        _record(0, "I am hosting a party at the cafe", 8),
        _record(1, "The weather was nice", 1),
    ]
    state = {
        "world": normalize_world(SIM_RAW, seed = 1),
        "npc_id": "bea",
        "history": [],
        "player_message": "Any party plans?",
        "memory": memory,
        "clock": 600,
    }
    ctx = SimCtx(
        "world.npc_reply",
        inputs = {"state": state},
        llm = lambda **kw: {"text": "Come along!", "remember": []},
    )

    async def go():
        started = time.monotonic()
        reply = (await world_nodes.world_npc_reply(ctx))["reply"]
        took = time.monotonic() - started
        await asyncio.sleep(1.6)  # the embedder finishes in the background
        return reply, took

    reply, took = run(go())
    assert reply["text"] == "Come along!" and took < 1.0
    assert any("keywords this turn" in m for _, m in ctx.logs)
    assert {"I am hosting a party at the cafe", "The weather was nice"} <= set(world_nodes._VECTORS)
