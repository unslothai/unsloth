# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Schema-first authoring: the repair loop, the scene solver, describe-a-place and raids."""

import pytest

from core.flows.authoring import AuthoringError, constrained_call, schema_errors, string_enum
from core.flows.nodes import world as world_nodes
from core.flows.nodes._common import first_json
from core.flows.raid import ENEMY_KINDS, fallback_raid, generate_raid, raid_schema
from core.flows.registry import FlowNodeError
from core.flows.scene_solver import (
    KINDS,
    SCENE_SCHEMA,
    Grid,
    fallback_scene,
    solve,
    validate_scene,
)
from core.flows.world import DIRECTOR_SCHEMA, director_schema, props3d

from .flows_player_fakes import FakeCtx, run

SCHEMA = {
    "type": "object",
    "properties": {
        "kind": string_enum(["tree", "rock"]),
        "count": {"type": "integer", "minimum": 1},
    },
    "required": ["kind", "count"],
}


class TextCtx(FakeCtx):
    """A model that answers in text, parsed the way llm_json parses a real reply."""

    async def llm_json(self, **kw):
        self.llm_calls.append(kw)
        return first_json(self.llm(**kw))


def _replies(*texts):
    it = iter(texts)
    return lambda **kw: next(it)


def _ok(raw):
    if raw["count"] > 5:
        raise AuthoringError("count must be 5 or less")
    return raw


# --- the helper ---


def _well_formed(schema, path = "$"):
    """Every enum non-empty and of its type, every required key declared."""
    if "enum" in schema:
        assert schema["enum"] and all(isinstance(v, str) for v in schema["enum"]), path
    for key in schema.get("required") or []:
        assert key in schema["properties"], f"{path}.{key}"
    for key, sub in (schema.get("properties") or {}).items():
        _well_formed(sub, f"{path}.{key}")
    if isinstance(schema.get("items"), dict):
        _well_formed(schema["items"], f"{path}[]")


def test_schemas_are_well_formed_and_fallbacks_satisfy_them():
    state = {"world": {"npcs": [{"id": "a", "name": "A"}], "items": [], "regions": []}}
    for schema in (SCENE_SCHEMA, raid_schema(["Bell"]), director_schema(state), DIRECTOR_SCHEMA):
        _well_formed(schema)
    assert SCENE_SCHEMA["properties"]["props"]["items"]["properties"]["kind"]["enum"] == list(KINDS)
    assert "model" not in KINDS
    assert director_schema(state)["properties"]["item"] == {"type": "string"}, "no empty enum"
    assert schema_errors(fallback_raid(1), raid_schema([])) == []
    spec = fallback_scene("a market")
    assert schema_errors(spec, SCENE_SCHEMA) == []
    assert schema_errors({"kind": "house", "count": True}, SCHEMA) == [
        "`kind` is 'house', not one of 'tree', 'rock'",
        "`count` should be integer",
    ]


def test_valid_answer_needs_one_call():
    ctx = TextCtx("world.describe_place", llm = _replies('{"kind": "tree", "count": 2}'))
    out = run(constrained_call(ctx, SCHEMA, "go\n", _ok))
    assert out.value == {"kind": "tree", "count": 2} and out.attempts == 1 and not out.fallback
    assert ctx.llm_calls[0]["schema"] is SCHEMA


def test_invalid_json_once_then_valid_is_repaired():
    ctx = TextCtx(
        "world.describe_place",
        llm = _replies(
            '{"kind": "tree", "count": ', 'Sure! ```json\n{"kind": "rock", "count": 3}\n```'
        ),
    )
    out = run(constrained_call(ctx, SCHEMA, "go\n", _ok))
    assert out.value == {"kind": "rock", "count": 3} and out.attempts == 2
    assert "refused" not in ctx.llm_calls[0]["prompt"]
    assert "Your last answer was refused: it was not valid JSON" in ctx.llm_calls[1]["prompt"]


def test_schema_and_validation_errors_are_fed_back():
    ctx = TextCtx(
        "world.describe_place",
        llm = _replies(
            '{"kind": "house", "count": 2}',
            '{"kind": "tree", "count": 9}',
            '{"kind": "tree", "count": 4}',
        ),
    )
    out = run(constrained_call(ctx, SCHEMA, "go\n", _ok))
    assert out.value["count"] == 4 and out.attempts == 3
    assert "`kind` is 'house'" in ctx.llm_calls[1]["prompt"]
    assert "count must be 5 or less" in ctx.llm_calls[2]["prompt"]


def test_fallback_after_repair_max_is_deterministic():
    def go():
        ctx = TextCtx("world.describe_place", llm = lambda **kw: "no idea")
        out = run(
            constrained_call(ctx, SCHEMA, "go\n", _ok, fallback = lambda: fallback_scene("forest"))
        )
        return ctx, out

    ctx, a = go()
    _, b = go()
    assert len(ctx.llm_calls) == 3 and a.fallback and a.attempts == 3
    assert a.value == b.value and a.value["name"] == "Grove"
    assert any(level == "warning" for level, _ in ctx.logs)


def test_without_fallback_the_last_reason_is_raised():
    ctx = TextCtx("world.describe_place", llm = lambda **kw: '{"kind": "tree", "count": 8}')
    with pytest.raises(FlowNodeError, match = "count must be 5 or less"):
        run(constrained_call(ctx, SCHEMA, "go\n", _ok, repair_max = 1))
    assert len(ctx.llm_calls) == 2


def test_model_setup_errors_are_not_swallowed_by_the_fallback():
    def boom(**kw):
        raise FlowNodeError("Load a chat model first")

    ctx = FakeCtx("world.describe_place", llm = boom)
    with pytest.raises(FlowNodeError, match = "Load a chat model"):
        run(constrained_call(ctx, SCHEMA, "go\n", _ok, fallback = lambda: {}))


# --- the solver ---

OPEN = ["#" * 20] + ["#" + "." * 18 + "#" for _ in range(12)] + ["#" * 20]
# a corridor joining two rooms: a solid prop on it would cut the right room off
CORRIDOR = (
    ["############"]
    + ["#...#......#"] * 2
    + ["#..........#"]
    + ["#...#......#"] * 2
    + ["############"]
)


def _spec(
    props,
    relations = (),
    areas = None,
):
    return validate_scene(
        {
            "name": "t",
            "areas": areas if areas is not None else [{"id": "main", "size": "medium"}],
            "props": props,
            "relations": list(relations),
        }
    )


def _tiles(out):
    return [(int(p["x"]), int(p["y"])) for p in out["props3d"]]


def test_solver_places_everything_on_free_reachable_tiles():
    rows = list(OPEN)
    rows[5] = "#" + "." * 6 + "#####" + "." * 7 + "#"
    rows[8] = rows[8][:9] + "R" + rows[8][10:]
    spec = _spec(
        [
            {"id": "well", "kind": "well", "count": 1, "area": "main"},
            {"id": "stalls", "kind": "stall", "count": 4, "area": "main"},
            {"id": "benches", "kind": "bench", "count": 2, "area": "main"},
        ],
        [
            {"type": "near", "a": "stalls", "b": "well"},
            {"type": "facing", "a": "benches", "b": "well"},
        ],
    )
    grid = Grid(rows, (2, 2))
    out = solve(spec, grid, focus = (9, 8), seed = 3)
    tiles = _tiles(out)
    assert len(tiles) == 7 and len(set(tiles)) == 7 and not out["dropped"] and not out["relaxed"]
    assert all(rows[y][x] == "." for x, y in tiles), "never on a wall or a reserved tile"
    assert out["reachable"] and out["open_tiles_cut"] == 0
    well = tiles[0]
    assert all(max(abs(x - well[0]), abs(y - well[1])) <= 2 for x, y in tiles[1:5])
    assert props3d(out["props3d"], 20, 14) == out["props3d"], "already clean props3d entries"
    bench = out["props3d"][5]
    assert bench["solid"] is False and bench["kind"] == "bench"
    assert solve(spec, Grid(rows, (2, 2)), focus = (9, 8), seed = 3) == out, "deterministic"


def test_solid_props_never_cut_off_open_ground():
    spec = _spec([{"id": "rocks", "kind": "rock", "count": 6, "area": "main"}])
    out = solve(spec, Grid(CORRIDOR, (1, 1)), focus = (4, 3), seed = 0)
    assert (4, 3) not in _tiles(out), "the doorway stays open"
    assert out["reachable"] and out["open_tiles_cut"] == 0 and len(out["props3d"]) == 6


def test_unsatisfiable_relations_are_relaxed_in_priority_order():
    spec = _spec(
        [
            {"id": "well", "kind": "well", "count": 1, "area": "mid"},
            {"id": "crates", "kind": "crate", "count": 2, "area": "mid"},
        ],
        [
            {"type": "against_wall", "a": "crates", "b": ""},
            {"type": "near", "a": "crates", "b": "well"},
        ],
        areas = [{"id": "mid", "size": "small"}],
    )
    big = ["#" * 30] + ["#" + "." * 28 + "#" for _ in range(20)] + ["#" * 30]
    out = solve(spec, Grid(big, (2, 2)), focus = (15, 10), seed = 0)
    assert len(out["props3d"]) == 3 and not out["dropped"]
    # the area is far from any wall: `near` goes first, then `against_wall`; `in_area` would go last
    assert out["relaxed"] and all(r["dropped"] == ["near", "against_wall"] for r in out["relaxed"])
    tiles = _tiles(out)
    assert all(max(abs(x - 15), abs(y - 10)) <= 2 for x, y in tiles), "still in the area"


def test_against_wall_holds_when_a_wall_is_close():
    spec = _spec(
        [{"id": "crates", "kind": "crate", "count": 3, "area": "main"}],
        [{"type": "against_wall", "a": "crates", "b": ""}],
    )
    out = solve(spec, Grid(OPEN, (5, 5)), focus = (2, 6), seed = 0)
    assert not out["relaxed"]
    for x, y in _tiles(out):
        assert "#" in (OPEN[y][x - 1], OPEN[y][x + 1], OPEN[y - 1][x], OPEN[y + 1][x])


def test_no_room_drops_props_instead_of_failing():
    tiny = ["#####", "#...#", "#####"]
    out = solve(
        _spec([{"id": "trees", "kind": "tree", "count": 5, "area": "main"}]),
        Grid(tiny, (1, 1)),
        seed = 0,
    )
    assert len(out["props3d"]) + len(out["dropped"]) == 5 and out["dropped"] and out["reachable"]


def test_scene_validation_names_the_bad_reference():
    with pytest.raises(AuthoringError, match = "area 'plaza'"):
        _spec([{"id": "w", "kind": "well", "count": 1, "area": "plaza"}])
    with pytest.raises(AuthoringError, match = "prop 'ghost'"):
        _spec(
            [{"id": "w", "kind": "well", "count": 1, "area": "main"}],
            [{"type": "near", "a": "ghost", "b": "w"}],
        )
    with pytest.raises(AuthoringError, match = "keep it to 40"):
        _spec([{"id": f"t{i}", "kind": "tree", "count": 8, "area": "main"} for i in range(6)])


# --- the node ---

SPEC_JSON = (
    '{"name": "Market", "areas": [{"id": "square", "size": "medium"}], "props": ['
    '{"id": "well", "kind": "well", "count": 1, "area": "square"},'
    '{"id": "stalls", "kind": "stall", "count": 3, "area": "square"}],'
    ' "relations": [{"type": "near", "a": "stalls", "b": "well"}]}'
)


def test_describe_place_node_repairs_and_solves():
    bad = SPEC_JSON.replace('"area": "square"}]', '"area": "plaza"}]')
    ctx = TextCtx(
        "world.describe_place",
        inputs = {
            "request": {
                "text": "a market",
                "rows": OPEN,
                "spawn": {"x": 2, "y": 2},
                "taken": ["well-1"],
            }
        },
        llm = _replies(bad, SPEC_JSON),
    )
    place = run(world_nodes.world_describe_place(ctx))["place"]
    assert "area 'plaza'" in ctx.llm_calls[1]["prompt"]
    assert place["fallback"] is False and place["attempts"] == 2 and place["name"] == "Market"
    assert [p["kind"] for p in place["props3d"]] == ["well", "stall", "stall", "stall"]
    assert place["props3d"][0]["id"] == "well-2", "ids never clash with the scene's"
    assert place["reachable"]


def test_npc_reply_task_describes_a_place_with_the_fallback():
    ctx = TextCtx(
        "world.npc_reply",
        inputs = {
            "state": {"task": "describe_place", "text": "a quiet garden", "width": 16, "height": 12}
        },
        llm = lambda **kw: "I cannot do that",
    )
    place = run(world_nodes.world_npc_reply(ctx))["reply"]["place"]
    assert place["fallback"] and place["name"] == "Garden" and len(ctx.llm_calls) == 3
    assert {p["kind"] for p in place["props3d"]} == {"tree", "bench", "lamp"}
    with pytest.raises(FlowNodeError, match = "Describe the place"):
        run(
            world_nodes.world_npc_reply(
                FakeCtx("world.npc_reply", inputs = {"state": {"task": "describe_place"}})
            )
        )


# --- raids ---


def test_raid_from_the_model_keeps_to_the_enums():
    raw = {
        "theme": "goblin warren",
        "camp_name": "Mudhole",
        "roster": [{"kind": "grunt", "count": 3}, {"kind": "archer", "count": 2}],
        "loot": ["gem", "Golden Bell"],
        "barks": [
            {"event": e, "text": f"{e} line {i}"}
            for e in ("spot", "lost", "hurt", "flank", "retreat")
            for i in range(3)
        ],
    }
    ctx = FakeCtx("world.describe_place", llm = lambda **kw: raw)
    out = run(generate_raid(ctx, {"title": "V", "items": [{"id": "bell", "name": "Golden Bell"}]}))
    assert not out.fallback and out.value["loot"] == ["gem", "Golden Bell"]
    assert (
        all(r["kind"] in ENEMY_KINDS for r in out.value["roster"]) and len(out.value["barks"]) == 15
    )
    assert "Golden Bell" in ctx.llm_calls[0]["schema"]["properties"]["loot"]["items"]["enum"]


def test_raid_falls_back_to_the_same_raid_for_the_same_seed():
    ctx = FakeCtx("world.describe_place", llm = lambda **kw: {"theme": "bandit camp", "roster": []})
    a = run(generate_raid(ctx, seed = 9)).value
    b = fallback_raid(9)
    assert a == b and len(ctx.llm_calls) == 3
    assert 12 <= len(a["barks"]) <= 24 and {x["event"] for x in a["barks"]} == {
        "spot",
        "lost",
        "hurt",
        "flank",
        "retreat",
    }
