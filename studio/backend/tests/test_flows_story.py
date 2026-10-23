# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import pytest

from core.flows.nodes import story as story_nodes
from core.flows.registry import FlowNodeError
from core.flows.schema import StoryManifest
from core.flows.story import (
    StoryCompileError,
    compile_story,
    compile_story_report,
    parse_lines,
    repair_generated,
    repair_live_scene,
    story_schema,
    walk_check,
)

from .flows_player_fakes import FakeCtx, run


def N(
    id,
    type,
    x = 0,
    **params,
):
    return {"id": id, "type": type, "v": 1, "params": params, "pos": [x, 0]}


def F(src, port, dst):
    return {
        "id": f"{src}-{port}-{dst}",
        "source": src,
        "source_port": port,
        "target": dst,
        "target_port": "in",
        "kind": "flow",
    }


def D(src, port, dst, tport):
    return {
        "id": f"{src}-{port}-{dst}-{tport}",
        "source": src,
        "source_port": port,
        "target": dst,
        "target_port": tport,
        "kind": "data",
    }


def authored_doc():
    nodes = [
        N("start", "story.start", 0, title = "Moon", vars = {"courage": 0}, mode = "fixed"),
        N("mo", "story.character", 0, name = "Mo", look = "a sleepy sloth", color = "#aabbcc"),
        N(
            "s1",
            "story.scene",
            100,
            title = "Porch",
            lines = "Narrator: Night fell.\nMo (sad): The moon is gone!\nPip: Who?",
        ),
        N(
            "c1",
            "story.choice",
            200,
            prompt = "Where to?",
            choices = ["Forest", "Lake", "Nowhere"],
            effects = {"0": {"courage": {"op": "+", "args": [{"var": "courage"}, 1]}}},
            conditions = {"1": {"op": ">", "args": [{"var": "courage"}, 5]}},
        ),
        N("s2", "story.scene", 300, lines = "Mo: Trees!"),
        N("b1", "story.branch", 400, condition = {"op": ">=", "args": [{"var": "courage"}, 1]}),
        N("good", "story.ending", 500, title = "Found it", kind = "good", text = "The moon smiled."),
        N("s3", "story.scene", 300, lines = "Mo: Water!"),
        N("set", "story.set", 350, set = {"courage": 3}),
        N("r1", "story.random", 400, weights = [1, 2]),
        N("bad", "story.ending", 500, title = "Lost", kind = "bad"),
    ]
    edges = [
        F("start", "next", "s1"),
        F("s1", "next", "c1"),
        F("c1", "choice_1", "s2"),
        F("c1", "choice_2", "s3"),
        F("s2", "next", "b1"),
        F("b1", "then", "good"),
        F("b1", "else", "bad"),
        F("s3", "next", "set"),
        F("set", "next", "r1"),
        F("r1", "path_1", "good"),
        F("r1", "path_2", "bad"),
        D("mo", "character", "s1", "characters"),
    ]
    return {"id": "f", "name": "Moon flow", "nodes": nodes, "edges": edges, "baked": {}}


def test_compile_authored_graph():
    m, warnings = compile_story_report(authored_doc())
    StoryManifest.model_validate(m)
    assert m["start"] == "s1" and m["title"] == "Moon" and m["vars"] == {"courage": 0}
    scenes = {s["id"]: s for s in m["scenes"]}
    # The scene flowing into the Choice node carries the menu itself.
    s1 = scenes["s1"]
    assert [c["text"] for c in s1["choices"]] == ["Forest", "Lake"]
    assert s1["choices"][0]["set"]["courage"]["op"] == "+"
    assert s1["choices"][1]["if"]["op"] == ">"
    assert s1["lines"][-1]["text"] == "Where to?"
    # Pip was not a Character node: auto-created; Mo's expression parsed.
    chars = {c["name"]: c for c in m["characters"]}
    assert set(chars) == {"Mo", "Pip"}
    assert s1["lines"][1] == {"speaker": "mo", "text": "The moon is gone!", "expression": "sad"}
    assert s1["lines"][0]["speaker"] is None if "speaker" in s1["lines"][0] else True
    assert set(s1["on_stage"]) == {"mo", "pip"}
    assert [c["text"] for c in scenes["b1"]["choices"]] == ["", ""]
    assert scenes["r1"]["random"] == [
        {"weight": 1.0, "goto": "good"},
        {"weight": 2.0, "goto": "bad"},
    ]
    assert scenes["set"]["on_enter"] == {"courage": 3}
    assert scenes["good"]["ending"]["kind"] == "good"
    assert any("Nowhere" in w for w in warnings)
    assert (
        walk_check(m) == [] and "c1" not in scenes
    )  # the merged Choice node leaves no stray scene


def test_baked_generated_story_wins_and_entry_points():
    doc = authored_doc()
    gen = repair_generated(_raw_story(), seed = 1)
    doc["nodes"] += [
        N("gen", "story.generator", 0),
        N("ill", "story.illustrate", 100),
        N("next", "story.next_scene", 0),
    ]
    doc["edges"].append(D("gen", "story", "ill", "story"))
    gen_ill = {**gen, "title": "Illustrated"}
    m = compile_story(doc, {"gen": {"story": gen}, "ill": {"story": gen_ill}})
    assert m["title"] == "Illustrated"
    assert m["entry_points"] == {"next_scene": "next"}
    doc["baked"] = {"gen": {"story": {**gen, "title": "Baked gen"}}}
    # Baked values come first, but the most downstream node still wins.
    assert compile_story(doc, {"ill": {"story": gen_ill}})["title"] == "Illustrated"
    assert compile_story(doc, {})["title"] == "Baked gen"


def test_no_story_raises():
    with pytest.raises(StoryCompileError):
        compile_story({"id": "f", "name": "x", "nodes": [N("a", "input.text")], "edges": []})


def test_parse_lines_ignores_sentences_with_colons():
    chars = []
    lines = parse_lines("It was late: very late.\nNarrator: Hi\nOld Man Willow: Shh", chars)
    assert lines[0] == {"speaker": None, "text": "It was late: very late."}
    assert lines[1]["speaker"] is None
    assert chars[0]["name"] == "Old Man Willow" and lines[2]["speaker"] == chars[0]["id"]


def _raw_story():
    return {
        "title": "The Lost Moon",
        "start": "Porch",
        "vars": [{"name": "Courage", "value": 0}],
        "characters": [
            {"id": "mo", "name": "Mo", "look": "sloth"},
            {"id": "owl", "name": "Owl", "look": "owl"},
        ],
        "scenes": [
            {
                "id": "porch",
                "title": "Porch",
                "background_prompt": "porch",
                "lines": [{"speaker": "Mo", "text": "Gone!"}],
                "choices": [
                    {"text": "Ask Owl", "goto": "tree", "effects": [{"var": "courage", "add": 1}]},
                    {"text": "Sleep", "goto": "nowhere"},
                ],
                "is_ending": False,
            },
            {
                "id": "tree",
                "title": "Tree",
                "lines": [{"speaker": "owl", "text": "Hoo"}],
                "choices": [{"text": "Back", "goto": "porch"}, {"text": "Loop", "goto": "loop"}],
                "is_ending": False,
            },
            {
                "id": "loop",
                "title": "Loop",
                "lines": [{"speaker": "narrator", "text": "Round"}],
                "choices": [{"text": "Again", "goto": "tree"}],
                "is_ending": False,
            },
            {
                "id": "sky",
                "title": "Sky",
                "lines": [{"speaker": "ghost", "text": "Boo"}],
                "choices": [],
                "is_ending": False,
            },
        ],
    }


def test_repair_generated_guarantees_endings():
    m = repair_generated(_raw_story(), endings = 2, seed = 3)
    StoryManifest.model_validate(m)
    scenes = {s["id"]: s for s in m["scenes"]}
    assert m["start"] == "porch"
    # "nowhere" is dropped; porch/tree/loop is a closed loop, so it gets a way out to the ending.
    assert [c["text"] for c in scenes["porch"]["choices"]] == ["Ask Owl", "Find another way"]
    assert scenes["porch"]["choices"][1]["goto"] == "sky"
    assert scenes["porch"]["choices"][0]["set"] == {
        "courage": {"op": "+", "args": [{"var": "courage"}, 1]}
    }
    assert m["vars"] == {"courage": 0}
    assert scenes["sky"]["ending"]  # no ending written: the last scene ends it
    assert scenes["sky"]["lines"][0].get("speaker") is None  # unknown speaker -> narrator
    assert walk_check(m) == []


def test_repair_rejects_garbage():
    with pytest.raises(StoryCompileError):
        repair_generated("nope")
    with pytest.raises(StoryCompileError):
        repair_generated({"scenes": []})


def test_schema_is_plain_json_schema():
    s = story_schema(8, 3)
    assert s["properties"]["scenes"]["maxItems"] == 8
    assert "oneOf" not in str(s) and "anyOf" not in str(s)


def test_generator_handler_uses_seed_and_repairs():
    ctx = FakeCtx(
        "story.generator",
        params = {"premise": "a sloth loses the moon", "scenes": 4, "endings": 2},
        seed = 99,
        llm = lambda **kw: _raw_story(),
    )
    out = run(story_nodes.story_generator(ctx))
    assert out["story"]["title"] == "The Lost Moon"
    call = ctx.llm_calls[0]
    assert call["seed"] == 99 and "a sloth loses the moon" in call["prompt"]
    assert call["schema"]["properties"]["scenes"]["maxItems"] == 4
    with pytest.raises(FlowNodeError):
        run(story_nodes.story_generator(FakeCtx("story.generator", params = {"premise": ""})))


def test_illustrate_paints_each_scene_and_cuts_portraits():
    story = repair_generated(_raw_story(), seed = 1)
    ctx = FakeCtx("story.illustrate", inputs = {"story": story})
    out = run(story_nodes.story_illustrate(ctx))["story"]
    assert all(s.get("background") for s in out["scenes"])
    assert all(c["portrait"]["ref"].endswith("-cut") for c in out["characters"])
    assert len(ctx.cutouts) == len(out["characters"])
    seeds = [i["seed"] for i in ctx.images]
    assert len(set(seeds)) == len(seeds)
    # Deterministic: the same seed paints with the same per-scene seeds.
    ctx2 = FakeCtx("story.illustrate", inputs = {"story": story})
    run(story_nodes.story_illustrate(ctx2))
    assert [i["seed"] for i in ctx2.images] == seeds


def test_narrate_dedupes_lines_per_voice():
    story = repair_generated(_raw_story(), seed = 1)
    ctx = FakeCtx("story.narrate", inputs = {"story": story}, params = {"narrator_voice": "af_heart"})
    out = run(story_nodes.story_narrate(ctx))["story"]
    assert all(ln.get("voice") for s in out["scenes"] for ln in s["lines"])
    assert ctx.spoken


def test_next_scene_live():
    state = {
        "story": {
            "title": "T",
            "characters": [{"id": "mo", "name": "Mo"}],
            "scenes": [{"id": "porch"}],
        },
        "scene_id": "porch",
        "vars": {},
        "history": ["Mo: hi"],
        "choice": "Ask Owl",
    }
    raw = {
        "title": "Owl Tree",
        "background_prompt": "big oak",
        "lines": [{"speaker": "Mo", "text": "Hello"}],
        "choices": [
            {"text": "Climb", "effects": [{"var": "courage", "add": 2}]},
            {"text": "Home", "goto": "porch"},
        ],
        "is_ending": False,
    }
    ctx = FakeCtx("story.next_scene", inputs = {"state": state}, llm = lambda **kw: raw)
    scene = run(story_nodes.story_next_scene(ctx))["scene"]
    assert scene["id"].startswith("gen_") and scene["generate_background"] is True
    assert scene["choices"][0]["goto"].startswith("gen_") and scene["choices"][1]["goto"] == "porch"
    assert scene["lines"][0]["speaker"] == "mo"
    # Same state -> same ids (players can cache and dedupe).
    assert repair_live_scene(raw, state)["id"] == scene["id"]
    end = repair_live_scene(
        {"title": "Fin", "lines": [], "is_ending": True, "ending_kind": "good"}, state
    )
    assert end["ending"]["kind"] == "good" and not end["choices"]


def test_scene_art():
    ctx = FakeCtx(
        "story.scene_art", inputs = {"prompt": "a moonlit lake"}, params = {"style": "watercolor"}
    )
    img = run(story_nodes.story_scene_art(ctx))["image"]
    assert img["width"] == 1344 and "watercolor" in ctx.images[0]["prompt"]


def test_infinite_mode_keeps_open_exits():
    doc = {
        "id": "f",
        "name": "d",
        "baked": {},
        "nodes": [
            N("start", "story.start", 0, title = "D", mode = "infinite"),
            N("s", "story.scene", 1, lines = "Narrator: Dark."),
            N("c", "story.choice", 2, choices = ["In", "Out"]),
            N("next", "story.next_scene", 3),
        ],
        "edges": [F("start", "next", "s"), F("s", "next", "c")],
    }
    m = compile_story(doc)
    assert m["mode"] == "infinite" and [c["goto"] for c in m["scenes"][0]["choices"]] == [
        "gen_c_1",
        "gen_c_2",
    ]
    doc[
        "nodes"
    ].pop()  # without a Next Scene node the open exits cannot be written: fall back to live
    m2, warnings = compile_story_report(doc)
    assert m2["mode"] == "live" and m2["scenes"][0]["choices"] == [] and warnings


def test_choices_beyond_four_exits_are_reported():
    doc = {
        "id": "f",
        "name": "d",
        "baked": {},
        "nodes": [
            N("start", "story.start", 0, title = "D", mode = "infinite"),
            N("s", "story.scene", 1, lines = "Narrator: Dark."),
            N("c", "story.choice", 2, choices = ["A", "B", "C", "D", "Fifth"]),
            N("next", "story.next_scene", 3),
        ],
        "edges": [F("start", "next", "s"), F("s", "next", "c")],
    }
    m, warnings = compile_story_report(doc)
    assert len(m["scenes"][0]["choices"]) == 4
    assert any("'Fifth'" in w for w in warnings)
