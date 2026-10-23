# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import pytest

from core.flows.nodes import world as world_nodes
from core.flows.registry import FlowNodeError
from core.flows.schema import WorldSpec
from core.flows.world import (
    BIOME_TILES,
    WorldError,
    normalize_world,
    npc_prompt,
    resolve_world,
    resolve_world_report,
    validate_npc_reply,
)

from .flows_player_fakes import FakeCtx, run

RAW = {
    "title": "Mossy Hollow",
    "intro_text": "A sleepy village.",
    "player_name": "Pip",
    "npcs": [
        {
            "id": "Baker Bea",
            "name": "Bea",
            "persona": "cheerful baker",
            "greeting": "Fresh bread!",
            "sprite_prompt": "baker",
        },
        {
            "id": "bea",
            "name": "Old Tom",
            "persona": "grumpy fisher",
            "greeting": "Hmph.",
            "sprite_prompt": "fisher",
        },
        {"name": ""},
    ],
    "items": [
        {"id": "rod", "name": "Fishing Rod", "description": "", "sprite_prompt": "rod"},
        {"id": "pie", "name": "Berry Pie", "description": "", "sprite_prompt": "pie"},
    ],
    "quests": [
        {
            "id": "q",
            "title": "Lost rod",
            "description": "Find my rod",
            "giver": "Old Tom",
            "requires_item": "Fishing Rod",
            "reward_text": "Thanks",
        },
        {"title": "Ghost", "giver": "nobody", "requires_item": "unicorn"},
    ],
    "regions": [{"name": "Pond", "on_enter_text": "Ripples."}, {"name": "Mill"}],
    "vars": {"Reputation": 0},
}


def test_normalize_world_repairs_references():
    w = normalize_world(RAW, seed = 5, biome = "forest")
    WorldSpec.model_validate(w)
    ids = [n["id"] for n in w["npcs"]]
    assert ids == ["baker_bea", "bea"] and len(set(ids)) == 2
    q = w["quests"][0]
    assert q["giver"] == "bea" and q["requires_item"] == "rod"
    ghost = w["quests"][1]
    assert ghost["giver"] in ids and "requires_item" not in ghost
    assert w["biome"] == "forest" and w["seed"] == 5 and w["vars"] == {"reputation": 0}
    assert all(0 <= r["x"] < w["width"] and r["w"] > 0 for r in w["regions"])
    with pytest.raises(WorldError):
        normalize_world("x")


def _state(**kw):
    w = normalize_world(RAW, seed = 1)
    base = {
        "world": w,
        "npc_id": "bea",
        "history": [],
        "player_message": "hi",
        "vars": {},
        "inventory": [],
    }
    base.update(kw)
    return base


def test_npc_reply_validation_only_allows_known_effects():
    s = _state(inventory = ["rod"])
    out = validate_npc_reply(
        {
            "text": " Oh   my rod! ",
            "give_item": "pie",
            "take_item": "sword",
            "complete_quest": "q",
            "set": [{"var": "reputation", "value": 2}, {"var": "gold", "value": 99}],
        },
        s,
    )
    assert out == {
        "text": "Oh my rod!",
        "give_item": "pie",
        "complete_quest": "q",
        "take_item": "rod",
        "set": {"reputation": 2},
    }
    # Quest needs the item; another NPC cannot complete it; done quests stay done.
    assert "complete_quest" not in validate_npc_reply(
        {"text": "x", "complete_quest": "q"}, _state()
    )
    assert "complete_quest" not in validate_npc_reply(
        {"text": "x", "complete_quest": "q"}, _state(npc_id = "baker_bea", inventory = ["rod"])
    )
    assert "complete_quest" not in validate_npc_reply(
        {"text": "x", "complete_quest": "q"}, _state(inventory = ["rod"], completed_quests = ["q"])
    )
    assert "give_item" not in validate_npc_reply(
        {"text": "x", "give_item": "pie"}, _state(inventory = ["pie"])
    )
    assert validate_npc_reply("plain text", _state()) == {"text": "plain text"}
    with pytest.raises(WorldError):
        validate_npc_reply({"text": "x"}, _state(npc_id = "ghost"))


def test_npc_prompt_mentions_quest_and_inventory():
    system, prompt = npc_prompt(
        _state(inventory = ["rod"], history = [{"role": "player", "text": "yo"}]), "Be brief."
    )
    assert "Old Tom" in system and "Be brief." in system
    assert (
        "Lost rod (id q)" in prompt and "Fishing Rod (id rod)" in prompt and "Player: yo" in prompt
    )


def test_resolve_world_prefers_downstream_and_adds_npc_nodes():
    w = normalize_world(RAW, seed = 1)
    doc = {
        "id": "f",
        "name": "w",
        "seed": 3,
        "baked": {},
        "nodes": [
            {"id": "gen", "type": "world.generator", "params": {}, "pos": [0, 0]},
            {"id": "art", "type": "world.art", "params": {}, "pos": [1, 0]},
            {
                "id": "npc",
                "type": "world.npc",
                "params": {"name": "Wizard Wen", "persona": "wise"},
                "pos": [0, 1],
            },
            {"id": "reply", "type": "world.npc_reply", "params": {}, "pos": [0, 2]},
        ],
        "edges": [
            {
                "id": "e",
                "source": "gen",
                "source_port": "world",
                "target": "art",
                "target_port": "world",
                "kind": "data",
            }
        ],
    }
    out = resolve_world(doc, {"gen": {"world": w}, "art": {"world": {**w, "title": "Painted"}}})
    assert out["title"] == "Painted"
    assert [n["name"] for n in out["npcs"]][-1] == "Wizard Wen"
    assert out["entry_points"] == {"npc_reply": "reply"}
    world, warnings = resolve_world_report({"id": "f", "name": "w", "nodes": [], "edges": []}, {})
    assert world is None and warnings


def test_world_spec_param_authored():
    doc = {
        "id": "f",
        "name": "w",
        "nodes": [{"id": "s", "type": "world.spec", "params": {"world": RAW}, "pos": [0, 0]}],
        "edges": [],
    }
    w, warnings = resolve_world_report(doc, {})
    assert w["title"] == "Mossy Hollow" and any("NPC Reply" in x for x in warnings)


def test_generator_and_art_handlers():
    ctx = FakeCtx(
        "world.generator",
        params = {"premise": "a cozy village", "biome": "town"},
        seed = 11,
        llm = lambda **kw: RAW,
    )
    w = run(world_nodes.world_generator(ctx))["world"]
    assert w["biome"] == "town" and w["player"]["name"] == "Pip" and w["seed"] == 11
    art = FakeCtx("world.art", inputs = {"world": w})
    painted = run(world_nodes.world_art(art))["world"]
    assert set(painted["tileset"]) == set(BIOME_TILES["town"])
    assert (
        painted["player"]["sprite"]
        and all(n["sprite"] for n in painted["npcs"])
        and all(i["sprite"] for i in painted["items"])
    )
    assert len(art.cutouts) == 1 + len(w["npcs"]) + len(w["items"])
    with pytest.raises(FlowNodeError):
        run(world_nodes.world_generator(FakeCtx("world.generator")))


def test_npc_reply_handler():
    s = _state(inventory = ["rod"])
    ctx = FakeCtx(
        "world.npc_reply",
        inputs = {"state": s},
        llm = lambda **kw: {"text": "My rod!", "complete_quest": "q"},
    )
    reply = run(world_nodes.world_npc_reply(ctx))["reply"]
    assert reply == {"text": "My rod!", "complete_quest": "q", "take_item": "rod"}
    with pytest.raises(FlowNodeError):
        run(world_nodes.world_npc_reply(FakeCtx("world.npc_reply")))


def test_npc_node():
    out = run(
        world_nodes.world_npc(FakeCtx("world.npc", params = {"name": "Old Tom", "persona": "p"}))
    )["npc"]
    assert out["id"] == "old_tom"
