# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Action play in the 3D view: opt-in per world, through the World Player's Gameplay setting."""

from core.flows.schema import WorldSpec
from core.flows.world import action_of, apply_play_view, normalize_world, props3d

RAW = {
    "title": "Arena",
    "npcs": [{"name": "Grim", "persona": "drifter", "greeting": "Hm."}],
    "items": [],
    "quests": [],
    "regions": [],
}


def _doc(**params):
    return {"nodes": [{"id": "play", "type": "world.play", "params": params}], "edges": []}


def test_action_is_off_unless_the_world_or_its_player_turns_it_on():
    world = normalize_world(dict(RAW), seed = 1)
    assert "action" not in world
    assert "action" not in apply_play_view(dict(world), _doc(view = "3d"))
    assert action_of({"movement": False, "shooter": False}) is None
    assert action_of("yes") is None


def test_the_gameplay_setting_picks_movement_and_shooter():
    world = normalize_world(dict(RAW), seed = 1)
    assert apply_play_view(dict(world), _doc(gameplay = "action"))["action"] == {
        "movement": True,
        "shooter": False,
    }
    shooter = apply_play_view(dict(world), _doc(gameplay = "shooter"))["action"]
    assert shooter == {"movement": True, "shooter": True, "verbs": True}
    armed = normalize_world({**RAW, "action": {"shooter": True}}, seed = 1)
    assert armed["action"] == {"movement": True, "shooter": True}
    assert "action" not in apply_play_view(dict(armed), _doc(gameplay = "explore"))


def test_verbs_are_opt_in_and_old_action_specs_keep_their_shape():
    assert action_of({"movement": True}) == {"movement": True, "shooter": False}
    assert action_of({"verbs": True}) == {"movement": True, "shooter": False, "verbs": True}
    assert action_of({"verbs": False, "movement": True}) == {"movement": True, "shooter": False}
    old = normalize_world({**RAW, "action": {"movement": True, "shooter": True}}, seed = 1)
    assert old["action"] == {"movement": True, "shooter": True}
    new = normalize_world({**RAW, "action": {"shooter": True, "verbs": True}}, seed = 1)
    assert new["action"] == {"movement": True, "shooter": True, "verbs": True}
    WorldSpec.model_validate(new)


DOORS = [
    {"id": "gate", "kind": "door", "x": 5.5, "y": 5.5, "locked_by": "key", "open": False},
    {"id": "hatch", "kind": "door", "x": 7.5, "y": 5.5, "locked_by": "ghost"},
    {"id": "box", "kind": "chest", "x": 6.5, "y": 8.5, "contents": ["key", "key", "ghost", 3]},
    {"id": "pull", "kind": "lever", "x": 9.5, "y": 8.5, "targets": ["gate", "box", "nowhere"]},
    {"id": "junk", "kind": "crate", "x": 2.5, "y": 2.5, "locked_by": "key", "contents": ["key"]},
]


def test_doors_chests_and_levers_keep_valid_links_only():
    world = normalize_world(
        {**RAW, "items": [{"id": "key", "name": "Brass key"}], "props3d": DOORS}, seed = 1
    )
    by = {p["id"]: p for p in world["props3d"]}
    assert by["gate"]["locked_by"] == "key" and by["gate"]["open"] is False and by["gate"]["solid"]
    assert "locked_by" not in by["hatch"]  # an unknown item cannot lock a door forever
    assert by["box"]["contents"] == ["key"]
    assert by["pull"]["targets"] == ["gate"] and by["pull"]["solid"] is False
    assert set(by["junk"]) == {"id", "kind", "x", "y", "rot", "scale", "solid"}
    WorldSpec.model_validate(world)
    # The player node's edited props are cleaned against the world's items the same way.
    played = apply_play_view(dict(world), _doc(props3d = DOORS))
    assert {p["id"]: p.get("contents") for p in played["props3d"]}["box"] == ["key"]


def test_props_without_item_context_keep_well_formed_ids():
    out = {p["id"]: p for p in props3d(DOORS, 20, 20)}
    assert out["hatch"]["locked_by"] == "ghost"
    assert out["box"]["contents"] == ["key", "ghost"]


def test_new_action_and_prop_fields_round_trip_through_normalize_save_and_the_player_node():
    import json

    raw = {
        **RAW,
        "items": [{"id": "key", "name": "Brass key"}],
        "action": {"shooter": True, "verbs": True},
        "props3d": DOORS,
    }
    world = normalize_world(raw, seed = 1)
    # Saved as JSON and normalised again (a reload, or an export read back): nothing changes or drops.
    again = normalize_world(json.loads(json.dumps(world)), seed = 1)
    assert again["action"] == world["action"] == {"movement": True, "shooter": True, "verbs": True}
    assert again["props3d"] == world["props3d"]
    # The scene editor saves props on the player node; they come back with their links intact.
    played = apply_play_view(dict(world), _doc(props3d = json.loads(json.dumps(world["props3d"]))))
    assert played["props3d"] == world["props3d"]
