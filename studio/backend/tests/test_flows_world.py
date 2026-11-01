# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import pytest

from core.flows.nodes import world as world_nodes
from core.flows.registry import FlowNodeError
from core.flows.schema import WorldSpec
from core.flows.world import (
    BIOME_TILES,
    apply_play_view,
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
    w = normalize_world(RAW, seed=5, biome="forest")
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
    w = normalize_world(RAW, seed=1)
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
    s = _state(inventory=["rod"])
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
    assert "complete_quest" not in validate_npc_reply(
        {"text": "x", "complete_quest": "q"}, _state()
    )
    assert "complete_quest" not in validate_npc_reply(
        {"text": "x", "complete_quest": "q"}, _state(npc_id="baker_bea", inventory=["rod"])
    )
    assert "complete_quest" not in validate_npc_reply(
        {"text": "x", "complete_quest": "q"}, _state(inventory=["rod"], completed_quests=["q"])
    )
    assert validate_npc_reply({"text": "My rod!"}, _state(inventory=["rod"])) == {
        "text": "My rod!",
        "complete_quest": "q",
        "take_item": "rod",
    }
    assert "complete_quest" not in validate_npc_reply(
        {"text": "x"}, _state(npc_id="baker_bea", inventory=["rod"])
    )
    assert "give_item" not in validate_npc_reply({"text": "Here.", "give_item": "rod"}, _state())
    assert "give_item" not in validate_npc_reply(
        {"text": "x", "give_item": "pie"}, _state(inventory=["pie"])
    )
    assert validate_npc_reply("plain text", _state()) == {"text": "plain text"}
    with pytest.raises(WorldError):
        validate_npc_reply({"text": "x"}, _state(npc_id="ghost"))


def test_npc_reply_actions_are_a_fixed_validated_set():
    s = _state()
    assert validate_npc_reply({"text": "Lead on!", "action": "follow_player"}, s) == {
        "text": "Lead on!",
        "action": "follow_player",
    }
    assert (
        validate_npc_reply({"text": "Bye", "action": "stop_following"}, s)["action"]
        == "stop_following"
    )
    asked = _state(player_message="Meet me at the pond later?")
    assert validate_npc_reply(
        {"text": "See you there", "action": "go_to", "place": "pond"}, asked
    ) == {
        "text": "See you there",
        "action": "go_to",
        "place": "pond",
    }
    assert "action" not in validate_npc_reply(
        {"text": "Off I go", "action": "go_to", "place": "pond"}, s
    )
    for msg, want in (
        ("Could you follow me? I want to show you something.", "follow_player"),
        ("Come with me to the pond", "follow_player"),
        ("Thanks, you can go home now. Stop following me.", "stop_following"),
        ("Please don't follow me", "stop_following"),
        ("Please don\u2019t follow me", "stop_following"),
        ("Do not come with me", "stop_following"),
        ("No need to walk with me", "stop_following"),
        ("I don't want you to follow me", "stop_following"),
        ("Don't stop following me", None),
    ):
        out = validate_npc_reply(
            {"text": "Sure", "action": "go_to", "place": "pond"}, _state(player_message=msg)
        )
        assert out.get("action") == want and "place" not in out, msg
    for bad in (
        {"action": "burn_down_the_mill"},
        {"action": "go_to", "place": "the_moon"},
        {"action": "go_to"},
        {"action": "Follow_Player"},
    ):
        out = validate_npc_reply({"text": "Hm.", **bad}, s)
        assert out == {"text": "Hm."}, bad
    system, prompt = npc_prompt(_state(following=True), "")
    assert "follow_player" in system and "Pond (id pond)" in system
    assert "You are walking with the player." in prompt
    assert "walking with" not in npc_prompt(_state(), "")[1]


def test_a_villager_who_stays_put_takes_no_movement_action():
    s = _state(player_message="Could you follow me?")
    w = s["world"]
    s["world"] = {
        **w,
        "npcs": [{**n, "wander": False} if n["id"] == "bea" else n for n in w["npcs"]],
    }
    assert "action" not in validate_npc_reply({"text": "Sure", "action": "follow_player"}, s)
    assert "follow_player" not in npc_prompt(s, "")[0]


def test_simulation_needs_flag_defaults_on():
    w = normalize_world({**RAW, "simulation": True}, seed=1)
    assert w["simulation"]["needs"] is True
    w = normalize_world({**RAW, "simulation": {"needs": False}}, seed=1)
    assert w["simulation"]["needs"] is False
    assert "simulation" not in normalize_world(RAW, seed=1)


def test_npc_prompt_mentions_quest_and_inventory():
    system, prompt = npc_prompt(
        _state(inventory=["rod"], history=[{"role": "player", "text": "yo"}]), "Be brief."
    )
    assert "Old Tom" in system and "Be brief." in system
    assert (
        "Lost rod (id q)" in prompt and "Fishing Rod (id rod)" in prompt and "Player: yo" in prompt
    )


def test_resolve_world_prefers_downstream_and_adds_npc_nodes():
    w = normalize_world(RAW, seed=1)
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


def test_render_and_view3d_are_kept_and_cleaned():
    plain = normalize_world(RAW, seed=1)
    assert "render" not in plain and "view3d" not in plain
    w = normalize_world(
        {
            **RAW,
            "render": "3d",
            "view3d": {
                "time_of_day": "dusk",
                "sky": " #AABBCC ",
                "fog": 7,
                "camera_height": 0.01,
                "walk_speed": "fast",
                "terrain": 1.5,
                "colour": "red",
            },
        },
        seed=1,
    )
    assert w["render"] == "3d"
    assert w["view3d"] == {
        "time_of_day": "dusk",
        "sky": "#aabbcc",
        "fog": 1.0,
        "camera_height": 0.3,
        "terrain": 1.5,
    }
    WorldSpec.model_validate(w)
    bad = normalize_world(
        {**RAW, "render": "4d", "view3d": {"time_of_day": "noon", "sky": "blue"}}, seed=1
    )
    assert "render" not in bad and "view3d" not in bad


def test_world_player_node_view_needs_no_rerun():
    def doc(params: dict) -> dict:
        return {
            "id": "f",
            "name": "w",
            "nodes": [
                {
                    "id": "s",
                    "type": "world.spec",
                    "params": {"world": {**RAW, "render": "2d"}},
                    "pos": [0, 0],
                },
                {"id": "play", "type": "world.play", "params": params, "pos": [1, 0]},
                {"id": "amb", "type": "input.audio", "params": {}, "pos": [0, 1]},
            ],
            "edges": [
                {
                    "id": "e1",
                    "source": "s",
                    "source_port": "world",
                    "target": "play",
                    "target_port": "world",
                    "kind": "data",
                },
                {
                    "id": "e2",
                    "source": "amb",
                    "source_port": "audio",
                    "target": "play",
                    "target_port": "ambient",
                    "kind": "data",
                },
            ],
        }

    sound = {"kind": "audio", "url": "/x.mp3"}
    w, _ = resolve_world_report(doc({}), {"amb": {"audio": sound}})
    assert w["render"] == "2d" and w["view3d"] == {"ambient": sound}
    w, _ = resolve_world_report(doc({"view": "auto", "time_of_day": "auto"}), {})
    assert w["render"] == "2d" and "view3d" not in w
    w, _ = resolve_world_report(doc({"view": "3d", "time_of_day": "night"}), {})
    assert w["render"] == "3d" and w["view3d"] == {"time_of_day": "night"}


def test_the_world_player_fed_by_the_chosen_world_sets_the_view():
    raw = {**RAW, "render": "2d"}
    d = {
        "id": "f",
        "name": "w",
        "nodes": [
            {
                "id": "stray",
                "type": "world.play",
                "params": {"view": "2d", "time_of_day": "dawn"},
                "pos": [0, 2],
            },
            {"id": "s", "type": "world.spec", "params": {"world": raw}, "pos": [0, 0]},
            {
                "id": "play",
                "type": "world.play",
                "params": {"view": "3d", "time_of_day": "night"},
                "pos": [1, 0],
            },
        ],
        "edges": [
            {
                "id": "e1",
                "source": "s",
                "source_port": "world",
                "target": "play",
                "target_port": "world",
                "kind": "data",
            },
        ],
    }
    w, _ = resolve_world_report(d, {})
    assert w["render"] == "3d" and w["view3d"]["time_of_day"] == "night"
    assert w["player_node"] == "play"


def test_generator_and_art_handlers():
    ctx = FakeCtx(
        "world.generator",
        params={"premise": "a cozy village", "biome": "town"},
        seed=11,
        llm=lambda **kw: RAW,
    )
    w = run(world_nodes.world_generator(ctx))["world"]
    assert w["biome"] == "town" and w["player"]["name"] == "Pip" and w["seed"] == 11
    art = FakeCtx("world.art", inputs={"world": w})
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
    s = _state(inventory=["rod"])
    ctx = FakeCtx(
        "world.npc_reply",
        inputs={"state": s},
        llm=lambda **kw: {"text": "My rod!", "complete_quest": "q"},
    )
    reply = run(world_nodes.world_npc_reply(ctx))["reply"]
    assert reply == {"text": "My rod!", "complete_quest": "q", "take_item": "rod"}
    assert ctx.llm_calls[0].get("seed") is None
    with pytest.raises(FlowNodeError):
        run(world_nodes.world_npc_reply(FakeCtx("world.npc_reply")))


def test_npc_node():
    out = run(
        world_nodes.world_npc(FakeCtx("world.npc", params={"name": "Old Tom", "persona": "p"}))
    )["npc"]
    assert out["id"] == "old_tom"


def test_props3d_are_cleaned_and_old_worlds_get_none():
    plain = normalize_world(RAW, seed=1)
    assert "props3d" not in plain
    asset = {"kind": "model", "ref": "sha256:" + "a" * 64, "url": "/api/flows/assets/x"}
    w = normalize_world(
        {
            **RAW,
            "props3d": [
                {"id": "house-1", "kind": "house", "x": 3.5, "y": 4.5, "rot": 450, "scale": 9},
                {"id": "house-1", "kind": "tree", "x": 1, "y": 1},  # duplicate id
                {"id": "lamp-1", "kind": "lamp", "x": -4, "y": 10_000},
                {"id": "m-1", "kind": "model", "x": 2, "y": 2, "asset": asset, "name": " Boat "},
                {"id": "m-2", "kind": "model", "x": 2, "y": 2},  # model without a file
                {"id": "m-3", "kind": "model", "x": 2, "y": 2, "asset": {"ref": "https://x/y.glb"}},
                {"id": "bad id", "kind": "rock", "x": 1, "y": 1},
                {"id": "ufo-1", "kind": "ufo", "x": 1, "y": 1},
                {"id": "nan-1", "kind": "rock", "x": float("nan"), "y": 1},
            ],
        },
        seed=1,
    )
    props = {p["id"]: p for p in w["props3d"]}
    assert list(props) == ["house-1", "lamp-1", "m-1"]
    assert props["house-1"] == {
        "id": "house-1", "kind": "house", "x": 3.5, "y": 4.5, "rot": 90.0, "scale": 5.0, "solid": True,
    }  # fmt: skip
    assert props["lamp-1"]["solid"] is False  # lamps are walked past by default
    assert props["lamp-1"]["x"] == 0.0 and props["lamp-1"]["y"] < w["height"]
    assert props["m-1"]["asset"] == {
        "kind": "model", "ref": asset["ref"], "mime": "model/gltf-binary", "url": asset["url"],
    }  # fmt: skip
    assert props["m-1"]["name"] == "Boat"
    WorldSpec.model_validate(w)


def test_scenery_saved_on_the_world_player_needs_no_rerun():
    def doc(params: dict) -> dict:
        return {
            "id": "f",
            "name": "w",
            "nodes": [
                {"id": "s", "type": "world.spec", "params": {"world": RAW}, "pos": [0, 0]},
                {"id": "play", "type": "world.play", "params": params, "pos": [1, 0]},
            ],
            "edges": [
                {
                    "id": "e1",
                    "source": "s",
                    "source_port": "world",
                    "target": "play",
                    "target_port": "world",
                    "kind": "data",
                },
            ],
        }

    w, _ = resolve_world_report(doc({}), {})
    assert "props3d" not in w
    crate = {"id": "crate-1", "kind": "crate", "x": 5.5, "y": 6.5}
    w, _ = resolve_world_report(doc({"props3d": [crate, {"kind": "nope"}]}), {})
    assert w["props3d"] == [{**crate, "rot": 0.0, "scale": 1.0, "solid": True}]
    w, _ = resolve_world_report(
        {
            **doc({"props3d": []}),
            "nodes": [
                {
                    "id": "s",
                    "type": "world.spec",
                    "params": {"world": {**RAW, "props3d": [crate]}},
                    "pos": [0, 0],
                },
                {"id": "play", "type": "world.play", "params": {"props3d": []}, "pos": [1, 0]},
            ],
        },
        {},
    )
    assert w.get("props3d") == []
