# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Action play in the 3D view: opt-in per world, through the World Player's Gameplay setting."""

from core.flows.world import action_of, apply_play_view, normalize_world

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
    # An old flow (no Gameplay setting) plays as before.
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
    assert shooter == {"movement": True, "shooter": True}
    # Explore on the player turns off what the world asked for.
    armed = normalize_world({**RAW, "action": {"shooter": True}}, seed = 1)
    assert armed["action"] == {"movement": True, "shooter": True}
    assert "action" not in apply_play_view(dict(armed), _doc(gameplay = "explore"))
