# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Worlds that keep growing: the director's patch only adds, and every reference in it resolves."""

import pytest

from core.flows.nodes import world as world_nodes
from core.flows.registry import FlowNodeError
from core.flows.world import (
    DIRECTOR_SCHEMA,
    WorldError,
    director_prompt,
    resolve_world,
    validate_director,
)

from .flows_player_fakes import FakeCtx, run
from .test_flows_world import RAW

WORLD = {
    "title": "Lantern Village",
    "intro_text": "A quiet village.",
    "npcs": [{"id": "old_tom", "name": "Old Tom"}, {"id": "mira", "name": "Mira"}],
    "items": [{"id": "rod", "name": "Fishing Rod"}, {"id": "bell", "name": "Golden Bell"}],
    "quests": [
        {"id": "lost_rod", "title": "The Lost Rod", "giver": "old_tom", "requires_item": "rod"}
    ],
    "regions": [{"id": "pond", "name": "Pond"}, {"id": "market", "name": "Market"}],
}


def _state(**kw):
    return {
        "task": "director",
        "world": WORLD,
        "inventory": [],
        "picked": [],
        "completed_quests": [],
        **kw,
    }


def test_quest_from_existing_giver_and_unused_item():
    patch = validate_director(
        {"title": "Ring the Bell", "giver": "Mira", "item": "Golden Bell"}, _state()
    )
    assert patch["quest"]["giver"] == "mira" and patch["quest"]["requires_item"] == "bell"
    assert patch["quest"]["id"] == "ring_the_bell" and patch["npcs"] == [] and patch["items"] == []


def test_new_npc_and_item_get_fresh_ids():
    raw = {
        "title": "Mira",  # the quest title slugs to an id a villager already has
        "giver": "Old Tom",
        "item": "",
        "new_item_name": "Old Tom",  # so does the new item's name
        "new_npc_name": "Pond Keeper",
        "region": "market",
    }
    patch = validate_director(raw, _state(allow_item = True, allow_npc = True))
    existing = {"old_tom", "mira", "rod", "bell", "lost_rod", "pond", "market"}
    new_ids = [patch["quest"]["id"], *(i["id"] for i in patch["items"])]
    assert not existing & set(new_ids) and len(set(new_ids)) == len(new_ids)
    assert patch["items"][0]["region"] == "market"
    assert patch["npcs"] == [] and patch["quest"]["giver"] == "old_tom"
    patch = validate_director(
        {"title": "Help", "giver": "Pond Keeper", "item": "bell", "new_npc_name": "Pond Keeper"},
        _state(allow_npc = True),
    )
    assert patch["npcs"][0]["id"] == "pond_keeper" and patch["quest"]["giver"] == "pond_keeper"


def test_dangling_references_are_rejected():
    with pytest.raises(WorldError):
        validate_director({"title": "X", "giver": "Nobody", "item": "bell"}, _state())
    with pytest.raises(WorldError):
        validate_director(
            {"title": "X", "giver": "Ann", "item": "bell", "new_npc_name": "Ann"}, _state()
        )
    with pytest.raises(WorldError):
        validate_director({"title": "X", "giver": "mira", "item": "Teapot"}, _state())
    with pytest.raises(WorldError):
        validate_director({"title": "", "giver": "mira", "item": "bell"}, _state())


def test_items_already_used_or_held_are_not_asked_for_again():
    for item, state in (
        ("rod", _state()),
        ("bell", _state(inventory = ["bell"])),
        ("bell", _state(picked = ["bell"])),
    ):
        with pytest.raises(WorldError):
            validate_director({"title": "Again", "giver": "mira", "item": item}, state)
    patch = validate_director(
        {"title": "Again", "giver": "mira", "item": "rod", "new_item_name": "Lantern"},
        _state(allow_item = True),
    )
    assert patch["quest"]["requires_item"] == "lantern"
    patch = validate_director(
        {"title": "Tea", "giver": "mira", "item": "Teapot"}, _state(allow_item = True)
    )
    assert patch["quest"]["requires_item"] == "teapot" and patch["items"][0]["name"] == "Teapot"
    patch = validate_director(
        {"title": "Chime", "giver": "mira", "item": "wind_chime"}, _state(allow_item = True)
    )
    assert (
        patch["items"][0]["name"] == "Wind chime"
        and patch["quest"]["requires_item"] == "wind_chime"
    )
    with pytest.raises(WorldError):
        validate_director(
            {"title": "Again", "giver": "mira", "item": "rod"}, _state(allow_item = True)
        )


def test_text_is_bounded():
    patch = validate_director(
        {
            "title": "T" * 500,
            "description": "d" * 5000,
            "giver": "mira",
            "item": "bell",
            "reward_text": "r" * 900,
        },
        _state(),
    )
    q = patch["quest"]
    assert len(q["title"]) == 120 and len(q["description"]) == 500 and len(q["reward_text"]) == 200
    long = "The bell rings. " * 50
    q = validate_director(
        {"title": "T", "description": long, "giver": "mira", "item": "bell"}, _state()
    )["quest"]
    assert q["description"].endswith("rings.") and len(q["description"]) <= 500


def test_director_task_returns_a_patch():
    ctx = FakeCtx(
        "world.npc_reply",
        inputs = {"state": _state(events = ["Quest complete: The Lost Rod"])},
        llm = lambda **kw: {"title": "Ring the Bell", "giver": "mira", "item": "bell"},
    )
    out = run(world_nodes.world_npc_reply(ctx))
    assert out["reply"]["patch"]["quest"]["requires_item"] == "bell"
    call = ctx.llm_calls[0]
    assert call["schema"]["required"] == DIRECTOR_SCHEMA["required"]
    assert call["schema"]["properties"]["giver"]["enum"] == ["old_tom", "mira"]
    assert call["schema"]["properties"]["item"]["enum"] == ["bell"]
    assert (
        "The Lost Rod (open)" in call["prompt"] and "Quest complete: The Lost Rod" in call["prompt"]
    )
    ctx = FakeCtx(
        "world.npc_reply", inputs = {"state": _state()}, llm = lambda **kw: {"title": "x", "giver": "?"}
    )
    patch = run(world_nodes.world_npc_reply(ctx))["reply"]["patch"]
    assert len(ctx.llm_calls) == 2, "one more try, then the built-in quest"
    assert patch["quest"]["requires_item"] == "bell" and patch["quest"]["giver"] in (
        "old_tom",
        "mira",
    )
    assert set(patch) == {"npcs", "items", "quest"}
    ctx = FakeCtx(
        "world.npc_reply",
        inputs = {"state": _state(picked = ["bell"])},
        llm = lambda **kw: {"title": "x", "giver": "?"},
    )
    with pytest.raises(FlowNodeError):
        run(world_nodes.world_npc_reply(ctx))


def test_a_refused_quest_is_asked_again_with_the_reason():
    answers = iter(
        [
            {"title": "Again", "giver": "mira", "item": "rod"},  # the rod is already asked for
            {"title": "Ring the Bell", "giver": "mira", "item": "bell"},
        ]
    )
    ctx = FakeCtx("world.npc_reply", inputs = {"state": _state()}, llm = lambda **kw: next(answers))
    out = run(world_nodes.world_npc_reply(ctx))
    assert out["reply"]["patch"]["quest"]["requires_item"] == "bell"
    assert "refused" not in ctx.llm_calls[0]["prompt"]
    assert (
        "Your last answer was refused: `item` is 'rod', not one of 'bell'"
        in ctx.llm_calls[1]["prompt"]
    )
    assert ctx.llm_calls[0].get("seed") is None and ctx.llm_calls[1].get("seed") is None


def test_prompt_offers_new_things_only_when_allowed():
    system, _ = director_prompt(_state())
    assert "new villager" not in system and "new item" not in system and "region" not in system
    system, _ = director_prompt(_state(allow_item = True, allow_npc = True))
    assert "new villager" in system and "new item" in system


def test_prompt_lists_only_items_a_new_quest_may_ask_for():
    _, prompt = director_prompt(_state())
    assert (
        "bell: Golden Bell" in prompt and "rod:" not in prompt
    )  # the rod is what The Lost Rod asks for
    system, prompt = director_prompt(_state(picked = ["bell"], allow_item = True))
    assert "bell:" not in prompt and "Every item is already used" in system


def test_new_quests_setting_reaches_the_player():
    nodes = [{"id": "w", "type": "world.spec", "params": {"world": RAW}}]
    on = resolve_world({"nodes": [*nodes, {"id": "r", "type": "world.npc_reply", "params": {}}]})
    off = resolve_world(
        {"nodes": [*nodes, {"id": "r", "type": "world.npc_reply", "params": {"new_quests": False}}]}
    )
    assert "new_quests" not in on and off["new_quests"] is False
