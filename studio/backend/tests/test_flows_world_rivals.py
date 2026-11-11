# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Village Rivals: the `rivals` block normalises, the strategist costs one call and is repaired, and a leader's
offer survives only when the engine could carry it out."""

import pytest
from pydantic import ValidationError

from core.flows import rivals
from core.flows.registry import FlowNodeError
from core.flows.examples.build_examples import example
from core.flows.nodes import world as world_nodes
from core.flows.schema import WorldSpec
from core.flows.world import normalize_world, resolve_world

from .flows_player_fakes import FakeCtx, run

RAW = {
    "title": "Three Villages",
    "npcs": [{"name": "Bea"}, {"name": "Bo"}, {"name": "Ada"}],
    "regions": [{"name": "North"}, {"name": "East"}, {"name": "South"}],
}
FACTIONS = [
    {"id": "us", "name": "Us", "player": True, "region": "North", "leader": "Bea"},
    {"id": "oak", "name": "Oakvale", "region": "east", "leader": "bo", "persona": "proud masons"},
    {"id": "ash", "name": "Ashford", "leader": "nobody"},
]


def _world(**rv):
    return normalize_world({**RAW, "rivals": {"factions": FACTIONS, **rv}})


def _digest(**you):
    return {
        "turn": 3,
        "you": {
            "id": "oak",
            "name": "Oakvale",
            "stock": {"food": 8, "wood": 6, "stone": 3, "gold": 3},
            **you,
        },
        "others": [
            {"id": "us", "name": "Us", "player": True, "relation": "peace"},
            {"id": "ash", "name": "Ashford", "player": False, "relation": "peace"},
        ],
        "recent": ["Ashford claimed new land."],
    }


def _state(task, **kw):
    return {
        "task": task,
        "world": _world(),
        "faction": "oak",
        "leader": {"name": "Bo", "persona": "a stone mason"},
        "digest": _digest(),
        "player": "us",
        "intent": {
            "posture": {"economy": 0.6, "expansion": 0.2, "military": 0.1, "diplomacy": 0.1},
            "target": None,
            "goals": ["grow"],
        },
        **kw,
    }


def test_rivals_normalise_ids_player_leaders_and_regions():
    w = _world(max_turns = 999, hysteresis = "x")
    fs = w["rivals"]["factions"]
    assert [f["id"] for f in fs] == ["us", "oak", "ash"]
    assert [f["player"] for f in fs] == [True, False, False]
    assert [f.get("leader") for f in fs] == ["bea", "bo", None]
    assert [f["region"] for f in fs] == ["north", "east", "south"], "ash gets the free region"
    assert w["rivals"]["max_turns"] == 200 and w["rivals"]["hysteresis"] == 0.15
    two_players = normalize_world(
        {**RAW, "rivals": {"factions": [{**f, "player": True} for f in FACTIONS]}}
    )
    assert [f["player"] for f in two_players["rivals"]["factions"]] == [True, False, False]
    clash = normalize_world(
        {**RAW, "rivals": {"factions": [{"name": "Bea"}, {"name": "Bea"}, *FACTIONS[1:]]}}
    )
    ids = [f["id"] for f in clash["rivals"]["factions"]]
    assert len(set(ids) | {"bea", "bo", "ada"}) == len(ids) + 3, "faction ids never shadow npc ids"
    five = normalize_world({**RAW, "rivals": {"factions": [{"name": f"F{i}"} for i in range(6)]}})
    assert len(five["rivals"]["factions"]) == 4


def test_worlds_without_rivals_are_unchanged():
    assert "rivals" not in normalize_world(RAW)
    assert "rivals" not in normalize_world({**RAW, "rivals": {"factions": [{"name": "Alone"}]}})
    assert "rivals" not in normalize_world({**RAW, "rivals": "yes"})
    assert WorldSpec.model_validate({"title": "x"}).rivals is None
    with pytest.raises(ValidationError):
        WorldSpec.model_validate({"title": "x", "rivals": {"factions": [{"id": "a", "name": "A"}]}})
    with pytest.raises(ValidationError):
        WorldSpec.model_validate(
            {"title": "x", "rivals": {"factions": [{"id": "a", "name": "A"}] * 2, "turns": 5}}
        )


def test_the_example_world_plays_rivals_through_resolve_world():
    doc = example("village-rivals")
    world = resolve_world(doc)
    assert world is not None and world["entry_points"].get("npc_reply") == "reply"
    fs = world["rivals"]["factions"]
    assert [f["player"] for f in fs].count(True) == 1 and len(fs) == 3
    regions = {r["id"] for r in world["regions"]}
    assert all(f["region"] in regions for f in fs)
    npcs = {n["id"] for n in world["npcs"]}
    assert all(f["leader"] in npcs for f in fs)


def test_strategist_is_one_call_and_repairs_the_answer():
    answers = {
        "economy": 2,
        "expansion": 0,
        "military": 1,
        "diplomacy": "x",
        "target": "Ashford",
        "goals": ["g" * 200, "", "two", "three", "four"],
        "say": "We grow.",
    }
    ctx = FakeCtx(
        "world.npc_reply", inputs = {"state": _state("rivals_strategist")}, llm = lambda **kw: answers
    )
    out = run(world_nodes.world_npc_reply(ctx))
    plan = out["reply"]["plan"]
    assert len(ctx.llm_calls) == 1 and ctx.llm_calls[0]["schema"] is rivals.STRATEGIST_SCHEMA
    assert plan["posture"] == {"economy": 0.5, "expansion": 0.0, "military": 0.5, "diplomacy": 0.0}
    assert plan["target"] == "ash" and plan["goals"] == ["g" * 60, "two"]
    ctx = FakeCtx(
        "world.npc_reply", inputs = {"state": _state("rivals_strategist")}, llm = lambda **kw: None
    )
    plan = run(world_nodes.world_npc_reply(ctx))["reply"]["plan"]
    assert len(ctx.llm_calls) == 1, "a bad answer is not asked again"
    assert plan["posture"] == {k: 0.25 for k in rivals.POSTURE_KEYS} and plan["target"] is None
    assert plan["goals"] == ["grow the village"]
    seen = []

    def unparsable(**kw):
        seen.append(kw.get("retry_bad_json"))
        raise FlowNodeError("The model did not return JSON")

    ctx = FakeCtx("world.npc_reply", inputs = {"state": _state("rivals_strategist")}, llm = unparsable)
    plan = run(world_nodes.world_npc_reply(ctx))["reply"]["plan"]
    assert seen == [False] and plan["goals"] == [
        "grow the village"
    ], "unparsable JSON is repaired, not retried"
    self_target = rivals.validate_strategist({"target": "oak"}, _state("rivals_strategist"))
    assert self_target["target"] is None


def test_strategist_prompt_is_bounded_and_carries_the_digest():
    big = _digest(goals = ["x" * 5000], queue = ["farm"] * 2000)
    system, prompt = rivals.strategist_prompt(_state("rivals_strategist", digest = big))
    assert len(prompt) <= rivals.DIGEST_CHARS + 40
    system, prompt = rivals.strategist_prompt(_state("rivals_strategist"))
    assert '"turn":3' in prompt and "Oakvale" in system and "proud masons" in system


def _say(raw, **kw):
    ctx = FakeCtx(
        "world.npc_reply",
        inputs = {"state": _state("rivals_diplomacy", player_message = "hi", **kw)},
        llm = lambda **_: raw,
    )
    out = run(world_nodes.world_npc_reply(ctx))
    assert len(ctx.llm_calls) == 1 and ctx.llm_calls[0]["schema"] is rivals.DIPLOMACY_SCHEMA
    return out["reply"], ctx.llm_calls[0]


def test_diplomacy_keeps_offers_the_engine_can_honour():
    trade = {
        "text": "Two stone for three bread.",
        "offer": "trade",
        "give": "stone",
        "give_amount": 2,
        "want": "food",
        "want_amount": 3,
    }
    reply, call = _say(trade)
    assert reply["offer"] == {
        "kind": "trade",
        "give": "stone",
        "give_amount": 2,
        "want": "food",
        "want_amount": 3,
    }
    assert reply["text"] == "Two stone for three bread." and "filtered" not in reply
    assert "Your village leans to economy" in call["prompt"] and "Player: hi" in call["prompt"]
    reply, _ = _say({**trade, "give": "gold", "give_amount": 9, "text": "I will give you 9 gold!"})
    assert (
        reply["offer"] == {"kind": "none"} and reply["filtered"] and "9 gold" not in reply["text"]
    )
    assert "does not have 9 gold" in reply["refused"]
    reply, _ = _say(
        {
            "text": "Peace!",
            "offer": "peace",
            "give": "",
            "give_amount": 0,
            "want": "",
            "want_amount": 0,
        }
    )
    assert reply["offer"] == {"kind": "none"}, "not at war"
    at_war = _digest()
    at_war["others"][0]["relation"] = "war"
    reply, _ = _say(
        {
            "text": "Enough blood.",
            "offer": "peace",
            "give": "",
            "give_amount": 0,
            "want": "",
            "want_amount": 0,
        },
        digest = at_war,
    )
    assert reply["offer"] == {"kind": "peace"}


def test_diplomacy_follows_the_intent_and_drops_empty_promises():
    hostile = {
        "posture": {"military": 0.7, "economy": 0.3},
        "target": "us",
        "goals": ["take their fields"],
    }
    ally = {
        "text": "Let us be allies.",
        "offer": "ally",
        "give": "",
        "give_amount": 0,
        "want": "",
        "want_amount": 0,
    }
    reply, call = _say(ally, intent = hostile)
    assert reply["offer"] == {"kind": "none"} and reply["text"] != ally["text"]
    assert "do not offer an alliance" in call["prompt"] and "set against Us" in call["prompt"]
    reply, _ = _say(ally)
    assert reply["offer"] == {"kind": "ally"}
    promise = {
        "text": "Deal! We will send you wood tomorrow.",
        "offer": "none",
        "give": "",
        "give_amount": 0,
        "want": "",
        "want_amount": 0,
    }
    reply, _ = _say(promise)
    assert reply["filtered"] and "send you wood" not in reply["text"]
    chat = {**promise, "text": "The weather is fine in Oakvale."}
    reply, _ = _say(chat)
    assert reply["text"] == chat["text"] and "filtered" not in reply


def test_unknown_faction_is_an_error():
    from core.flows.registry import FlowNodeError

    ctx = FakeCtx(
        "world.npc_reply",
        inputs = {"state": _state("rivals_strategist", faction = "ghost")},
        llm = lambda **kw: {},
    )
    with pytest.raises(FlowNodeError):
        run(world_nodes.world_npc_reply(ctx))
    assert ctx.llm_calls == []


def test_schemas_are_flat_and_closed():
    for schema in (rivals.STRATEGIST_SCHEMA, rivals.DIPLOMACY_SCHEMA):
        assert schema["additionalProperties"] is False
        assert set(schema["required"]) == set(schema["properties"])
        for prop in schema["properties"].values():
            assert prop["type"] != "object"
            assert prop.get("items", {}).get("type") in (None, "string")


def test_system_prompts_stay_byte_identical_and_the_digest_comes_last():
    later = _digest(goals = ["new"], turn = 9)
    later["recent"] = ["Oakvale raided Ashford."]
    hostile = {"posture": {"military": 0.9}, "target": "us", "goals": ["war"]}
    s1, p1 = rivals.strategist_prompt(_state("rivals_strategist"))
    s2, p2 = rivals.strategist_prompt(_state("rivals_strategist", digest = later))
    assert s1 == s2 and p1 != p2
    assert p2.endswith(rivals.digest_text(later) + "\n")
    d1, q1 = rivals.diplomacy_prompt(_state("rivals_diplomacy", player_message = "hi"))
    d2, q2 = rivals.diplomacy_prompt(
        _state(
            "rivals_diplomacy",
            player_message = "and now?",
            digest = later,
            intent = hostile,
            history = [{"role": "player", "text": "hi"}],
        )
    )
    assert d1 == d2, "intent and talk stay out of the system prompt"
    assert q2.endswith(rivals.digest_text(later) + "\n") and "set against Us" in q2
