# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Mayor mode: the `city` block, and the one model call a month that only writes words grounded in the numbers."""

import copy
import json

import pytest

from core.flows.city import (
    CITY_EVENTS,
    MAYOR_SCHEMA,
    MAYOR_SYSTEM,
    CityError,
    mayor_prompt,
    normalize_city,
    validate_mayor,
)
from core.flows.examples.build_examples import build
from core.flows.nodes import world as world_nodes
from core.flows.schema import WorldSpec
from core.flows.world import normalize_world, resolve_world

from .flows_player_fakes import FakeCtx, run

DIGEST = {
    "title": "Meadowbrook",
    "kind": "month",
    "petitions": True,
    "events": True,
    "month": 5,
    "funds": 17325,
    "tax": 7,
    "tax_income": 41,
    "road_cost": 6,
    "cash_flow": 35,
    "residential_pop": 23,
    "commercial_pop": 3,
    "industrial_pop": 4,
    "total_pop": 9,
    "demand_r": 1460,
    "demand_c": -380,
    "demand_i": 1210,
    "land_value_avg": 87,
    "pollution_avg": 31,
    "crime_avg": 58,
    "power_capacity": 700,
    "power_used": 63,
    "trips_ok": 14,
    "trips_failed": 2,
    "trips_no_road": 1,
    "zones": [
        {"id": "P1", "kind": "P", "pop": 0, "power": 1, "road": 0, "name": ""},
        {"id": "R2", "kind": "R", "pop": 6, "power": 1, "road": 1, "name": "Rose Row"},
        {"id": "C3", "kind": "C", "pop": 3, "power": 1, "road": 1, "name": ""},
        {"id": "I4", "kind": "I", "pop": 4, "power": 0, "road": 1, "name": ""},
    ],
    "residents": [
        {
            "id": "rosa",
            "name": "Rosa Park",
            "persona": "A retired teacher.",
            "zone": "R2",
            "land_value": 92,
            "pollution": 44,
            "crime": 51,
            "traffic": 150,
            "power": 1,
            "road": 1,
        }
    ],
    "unnamed": "C3",
}


def _state(**over) -> dict:
    d = copy.deepcopy(DIGEST)
    d.update(over)
    return {"task": "mayor", "city": d}


def _answer(**over) -> dict:
    out = {
        "advice": "Industry in I4 has no power; connect it and demand for industry (1210) will turn into jobs.",
        "petition_1_citizen": "rosa",
        "petition_1_metric": "pollution",
        "petition_1_text": "The air at home reads 44 on pollution, please move the factories away.",
        "petition_2_citizen": "",
        "petition_2_metric": "",
        "petition_2_text": "",
        "event": "none",
        "event_zone": "",
        "event_size": 0,
        "name_zone": "C3",
        "name": "Market Corner",
    }
    out.update(over)
    return out


def test_city_block_is_opt_in_and_old_worlds_are_unchanged():
    assert normalize_city(None) is None and normalize_city({"enabled": False}) is None
    assert normalize_city(True) == {
        "enabled": True,
        "funds": 20000,
        "tax": 7,
        "advisor": True,
        "petitions": True,
        "events": True,
    }
    assert normalize_city({"funds": -5, "tax": 99, "events": False}) == {
        "enabled": True,
        "funds": 0,
        "tax": 20,
        "advisor": True,
        "petitions": True,
        "events": False,
    }
    raw = {"title": "Old", "npcs": [{"name": "Tom"}]}
    old = normalize_world(raw, seed = 3)
    assert "city" not in old
    assert normalize_world(old, seed = 3) == old
    w = normalize_world({**raw, "city": {"tax": 9}}, seed = 3)
    assert w["city"]["tax"] == 9
    assert {k: v for k, v in w.items() if k != "city"} == old
    WorldSpec.model_validate(w)
    with pytest.raises(ValueError):
        WorldSpec.model_validate({**w, "city": {"tax": 50}})


def test_schema_is_flat_strict_and_has_no_tile_fields():
    props = MAYOR_SCHEMA["properties"]
    assert MAYOR_SCHEMA["additionalProperties"] is False
    assert set(MAYOR_SCHEMA["required"]) == set(props)
    assert all(v["type"] in ("string", "integer") for v in props.values()), "one level only"
    assert props["event"]["enum"] == list(CITY_EVENTS)
    assert not any(k in props for k in ("x", "y", "tile", "tiles", "coords", "position"))


def test_prompt_carries_the_exact_numbers_last_and_a_constant_system():
    system, prompt = mayor_prompt(_state())
    assert system == MAYOR_SYSTEM
    for needle in (
        "Funds 17325",
        "tax rate 7",
        "tax income 41",
        "cash flow 35",
        "residential 23, commercial 3, industrial 4, total 9",
        "residential 1460, commercial -380, industrial 1210",
        "land value 87, pollution 31, crime 58",
        "capacity 700, used 63",
        "passed 14, failed 2, no road 1",
        "- I4 (industrial, pop 4, power no, road yes)",
        "rosa lives in R2: land_value 92, pollution 44, crime 51, traffic 150, power 1, road 1",
        "Name zone: C3",
    ):
        assert needle in prompt, needle
    lines = prompt.splitlines()
    first_number = next(i for i, l in enumerate(lines) if l.startswith("Month "))
    stable = [l for l in lines[:first_number] if not l.startswith(("- ", "Name zone"))]
    assert not any(c.isdigit() for l in stable for c in l), "only the numbers block carries numbers"

    system2, prompt2 = mayor_prompt(_state(funds = 1, demand_r = -2000, month = 6))
    assert system2.encode() == system.encode()
    assert prompt2 != prompt
    common = next(i for i, (a, b) in enumerate(zip(prompt, prompt2)) if a != b)
    assert prompt[:common].count("\n") >= first_number, "the stable part is the shared prefix"
    for rule in ("never give coordinates", "never ask", "places every road"):
        assert rule in system
    _, advice = mayor_prompt(_state(kind = "advice"))
    assert "Petitions: off" in advice and "Events: off" in advice and "Name zone: none" in advice
    with pytest.raises(CityError):
        mayor_prompt({"task": "mayor"})


def test_grounded_advice_petitions_event_and_name_are_kept():
    out = validate_mayor(_answer(), _state())
    assert out["advice_source"] == "model"
    assert out["petitions"] == [
        {
            "citizen": "rosa",
            "zone": "R2",
            "metric": "pollution",
            "value": 44,
            "text": "The air at home reads 44 on pollution, please move the factories away.",
        }
    ]
    assert out["event"] is None
    assert out["name"] == {"zone": "C3", "name": "Market Corner"}


@pytest.mark.parametrize(
    "advice",
    [
        "Pollution is 99, far too high.",  # a number the report does not have
        "Zone R9 needs a road.",  # a zone that does not exist
        "Put a road at (12, 7).",  # a tile coordinate
        "You have 17325.9 in funds.",  # a reported number with an invented fraction
        "",
    ],
)
def test_ungrounded_advice_falls_back_to_the_numbers(advice):
    out = validate_mayor(_answer(advice = advice), _state())
    assert out["advice_source"] == "fallback"
    assert "Population 9, funds 17325, cash flow 35 a month." in out["advice"]


def test_petitions_are_grounded_in_the_residents_home_numbers():
    wrong = validate_mayor(_answer(petition_1_text = "Pollution is 80 here!"), _state())
    assert (
        wrong["petitions"][0]["text"]
        == "Please do something about pollution at home in R2: it is 44."
    )
    bare = validate_mayor(_answer(petition_1_text = "The air smells awful."), _state())
    assert bare["petitions"][0]["text"] == "The air smells awful. (pollution 44)"
    by_name = validate_mayor(
        _answer(petition_1_citizen = "Rosa Park", petition_1_metric = "crime"), _state()
    )
    assert by_name["petitions"][0]["citizen"] == "rosa" and by_name["petitions"][0]["value"] == 51
    stranger = validate_mayor(_answer(petition_1_citizen = "bob"), _state())
    assert stranger["petitions"] == []
    twice = validate_mayor(
        _answer(petition_2_citizen = "rosa", petition_2_metric = "crime", petition_2_text = "x"), _state()
    )
    assert len(twice["petitions"]) == 1
    off = validate_mayor(_answer(), _state(kind = "advice"))
    assert off["petitions"] == [] and off["name"] is None


def test_events_come_only_from_the_menu_with_a_fitting_zone():
    def ev(**kw):
        return validate_mayor(_answer(**kw), _state())["event"]

    assert ev(event = "festival", event_size = 9) == {"kind": "festival", "zone": "", "size": 3}
    assert ev(event = "fire", event_zone = "R2", event_size = 2) == {
        "kind": "fire",
        "zone": "R2",
        "size": 2,
    }
    assert ev(event = "fire", event_zone = "P1", event_size = 1) is None
    assert ev(event = "strike", event_zone = "R2", event_size = 1) is None
    assert ev(event = "strike", event_zone = "I4", event_size = 0)["size"] == 1
    assert ev(event = "meteor", event_zone = "R2", event_size = 1) is None
    assert validate_mayor(_answer(event = "boom"), _state(events = False))["event"] is None


def test_the_handler_makes_exactly_one_call_and_never_retries():
    ctx = FakeCtx("world.npc_reply", inputs = {"state": _state()}, llm = lambda **kw: _answer())
    out = run(world_nodes.world_npc_reply(ctx))["reply"]["mayor"]
    assert len(ctx.llm_calls) == 1
    call = ctx.llm_calls[0]
    assert call["system"] == MAYOR_SYSTEM and call["schema"] is MAYOR_SCHEMA
    assert out["petitions"][0]["value"] == 44
    bad = FakeCtx("world.npc_reply", inputs = {"state": _state()}, llm = lambda **kw: {"advice": 7})
    out = run(world_nodes.world_npc_reply(bad))["reply"]["mayor"]
    assert len(bad.llm_calls) == 1 and out["advice_source"] == "fallback"
    json.dumps(out)


def test_an_unparsable_mayor_answer_falls_back_after_one_request(monkeypatch):
    from core.flows.nodes import _common

    async def no_load(ctx, kind, model):
        return None

    monkeypatch.setattr(_common, "ensure_model", no_load)
    answers = iter(["not json at all", json.dumps(_answer())])

    class RealJsonCtx(FakeCtx):
        def __init__(self):
            super().__init__("world.npc_reply", inputs = {"state": _state()})
            self.posts = 0

        async def llm_json(self, **kw):
            return await _common.llm_json(self, **kw)

        def on_cancel(self, fn):
            pass

        async def call(
            self,
            method,
            path,
            json = None,
            **kw,
        ):
            if path == "/api/inference/cancel":
                return {}
            self.posts += 1
            return {"choices": [{"message": {"content": next(answers)}}]}

    ctx = RealJsonCtx()
    out = run(world_nodes.world_npc_reply(ctx))["reply"]["mayor"]
    assert ctx.posts == 1
    assert out["advice_source"] == "fallback"


def test_the_mayor_example_turns_city_on():
    doc = next(d for d in build() if d["id"] == "mayor-meadow")
    world = resolve_world(doc)
    assert world["city"]["enabled"] is True and world["city"]["funds"] == 20000
    assert world["entry_points"]["npc_reply"] == "reply"
    assert "simulation" not in world


CITY_ALL = {
    "enabled": True,
    "funds": 4321,
    "tax": 12,
    "advisor": False,
    "petitions": False,
    "events": False,
}


def test_every_city_field_round_trips_normalize_and_export_import():
    from core.flows import package
    from core.flows.package import read_package

    w = normalize_world({"title": "T", "npcs": [{"name": "Tom"}], "city": CITY_ALL}, seed = 2)
    assert w["city"] == CITY_ALL
    assert normalize_world(w, seed = 2)["city"] == CITY_ALL
    doc = copy.deepcopy(next(d for d in build() if d["id"] == "mayor-meadow"))
    node = next(n for n in doc["nodes"] if n["id"] == "world")
    node["params"]["world"]["city"] = CITY_ALL

    async def read_media(_ref):
        raise AssertionError("no media")

    for mode in ("json", "package"):
        data, _, _ = run(package.export_flow(doc, mode, read_media))
        parsed, _, _ = read_package(data)
        assert resolve_world(parsed)["city"] == CITY_ALL, mode


def test_city_wins_over_rivals_with_a_warning():
    from core.flows.world import resolve_world_report

    w = normalize_world({"title": "T", "city": True, "rivals": {"factions": []}}, seed = 1)
    assert "rivals" not in w and w["city"]["enabled"]
    doc = copy.deepcopy(next(d for d in build() if d["id"] == "mayor-meadow"))
    next(n for n in doc["nodes"] if n["id"] == "world")["params"]["world"]["rivals"] = {
        "factions": []
    }
    world, warnings = resolve_world_report(doc)
    assert "rivals" not in world
    assert any("only the Mayor panel" in x for x in warnings)
