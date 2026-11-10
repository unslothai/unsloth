# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0


from core.flows.schema import WorldSpec
from core.flows.world import (
    NPC_REPLY_SCHEMA,
    normalize_world,
    npc_prompt,
    resolve_world,
    validate_director,
    validate_npc_reply,
)

from .test_flows_world import RAW
from .test_flows_world import _state as reply_state
from .test_flows_world_director import _state as director_state

GOODS = [{"id": "bread", "name": "Bread", "price": 3}, {"id": "fish", "name": "Fish", "price": 4}]


def _world(params):
    nodes = [
        {"id": "w", "type": "world.spec", "params": {"world": RAW}},
        {"id": "r", "type": "world.npc_reply", "params": params},
    ]
    return resolve_world({"nodes": nodes})


def test_autonomy_is_off_by_default_and_each_level_turns_on_its_flags():
    assert "autonomy" not in _world({})
    assert "autonomy" not in _world({"autonomy": "off"})
    assert "autonomy" not in _world({"autonomy": "everything"})
    village = _world({"autonomy": "village"})["autonomy"]
    assert village == {
        "trade": True,
        "homes": True,
        "quests": False,
        "rivals": False,
        "enemies": False,
    }
    assert all(_world({"autonomy": "full"})["autonomy"].values())


def test_roles_and_autonomy_survive_normalising_and_junk_is_dropped():
    raw = {
        **RAW,
        "autonomy": {"enemies": True, "junk": True},
        "npcs": [
            {**RAW["npcs"][0], "role": "hostile"},
            {**RAW["npcs"][1], "role": "dragon"},
        ],
    }
    w = normalize_world(raw, seed = 1)
    WorldSpec.model_validate(w)
    assert w["autonomy"] == {
        "trade": False,
        "homes": False,
        "quests": False,
        "rivals": False,
        "enemies": True,
    }
    assert w["npcs"][0]["role"] == "hostile" and "role" not in w["npcs"][1]
    assert "autonomy" not in normalize_world(RAW, seed = 1)


def test_a_trade_offer_is_only_something_the_villager_sells():
    assert "offer_trade" in NPC_REPLY_SCHEMA["properties"]
    s = reply_state(goods = GOODS)
    assert validate_npc_reply({"text": "Fish!", "offer_trade": "fish"}, s)["offer_trade"] == "fish"
    assert "offer_trade" not in validate_npc_reply({"text": "A sword!", "offer_trade": "sword"}, s)
    asked = reply_state(goods = GOODS, player_message = "What do you have for sale?")
    assert validate_npc_reply({"text": "Lots."}, asked)["offer_trade"] == "bread"
    assert "offer_trade" not in validate_npc_reply(
        {"text": "Fish!", "offer_trade": "fish"},
        reply_state(player_message = "what do you have for sale?"),
    )


def test_role_and_wares_reach_the_prompt_only_with_autonomy():
    plain, _ = npc_prompt(reply_state(), "Stay in character.")
    assert "rival" not in plain and "You sell" not in plain
    hostile, _ = npc_prompt(reply_state(role = "hostile", goods = GOODS), "Stay in character.")
    assert "HOSTILE to the player" in hostile and "never hurt" in hostile
    assert hostile.index("Your role") < hostile.index("Persona:")
    assert "Bread (id bread, 3 coins)" in hostile and "offer_trade" in hostile
    rival, _ = npc_prompt(reply_state(role = "rival"), "Stay in character.")
    assert "RIVAL" in rival and rival.index("Your role") < rival.index("Persona:")


def test_the_director_adds_rivals_and_hostiles_only_when_allowed():
    raw = {
        "title": "Watch Out",
        "giver": "Pond Keeper",
        "item": "bell",
        "new_npc_name": "Pond Keeper",
        "new_npc_role": "hostile",
    }
    allowed = validate_director(raw, director_state(allow_npc = True, allow_hostile = True))
    assert allowed["npcs"][0]["role"] == "hostile"
    plain = validate_director(raw, director_state(allow_npc = True))
    assert "role" not in plain["npcs"][0]
    wrong = validate_director(
        {**raw, "new_npc_role": "rival"}, director_state(allow_npc = True, allow_hostile = True)
    )
    assert "role" not in wrong["npcs"][0]


def test_the_director_is_told_about_roles_only_when_allowed():
    from core.flows.world import director_prompt

    with_roles, _ = director_prompt(director_state(allow_npc = True, allow_rival = True))
    assert "new_npc_role" in with_roles and "rival" in with_roles and "hostile" not in with_roles
    without, _ = director_prompt(director_state(allow_npc = True))
    assert "new_npc_role" not in without
