# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import json

from core.flows.nodes import story as story_nodes
from core.flows.schema import StoryManifest
from core.flows.story import (
    compile_story_report,
    parse_lines,
    repair_generated,
    repair_live_scene,
    story_schema,
)

from .flows_player_fakes import FakeCtx, run
from .test_flows_story import D, F, N, _raw_story, authored_doc

CHARS = [{"id": "mo", "name": "Mo"}, {"id": "captain_lu", "name": "Captain Lu"}]


def test_directions_parse_into_text_less_stage_lines():
    warnings, scene = [], {}
    lines = parse_lines(
        "[scene with fade]\n"
        "[show Mo happy at left with dissolve]\n"
        "Mo (happy): Hello!\n"
        "[show Captain Lu at tworight] [sound Door Creak]\n"
        "[pause 1.5]\n"
        "[hide Mo with slide]",
        CHARS,
        warnings = warnings,
        scene = scene,
    )
    assert scene == {"transition": "fade"}
    assert warnings == []
    assert lines == [
        {
            "speaker": None,
            "text": "",
            "stage": {"show": "mo", "expression": "happy", "at": "left", "with": "dissolve"},
        },
        {"speaker": "mo", "text": "Hello!", "expression": "happy"},
        {"speaker": None, "text": "", "stage": {"show": "captain_lu", "at": "tworight"}},
        {"speaker": None, "text": "", "stage": {"sound": "door_creak"}},
        {"speaker": None, "text": "", "stage": {"pause": 1.5}},
        {"speaker": None, "text": "", "stage": {"hide": "mo", "with": "slideleft"}},
    ]


def test_unknown_directions_warn_and_are_skipped():
    warnings = []
    lines = parse_lines(
        "[dance wildly]\n[show Nobody]\n[pause soon]\n[show Mo at nowhere with sparkle]\n[sound]\nMo: Still here.",
        CHARS,
        warnings = warnings,
    )
    assert [ln["text"] for ln in lines] == ["[dance wildly]", "", "Still here."]
    assert lines[1]["stage"] == {"show": "mo"}
    assert len(warnings) == 6
    assert any("not a direction" in w for w in warnings)
    assert any("sparkle" in w for w in warnings) and any("nowhere" in w for w in warnings)
    # Old stage notes stay narration without a warning; no warnings list never crashes.
    w2 = []
    assert parse_lines("[Thunder rumbles]", CHARS, warnings = w2)[0]["text"] == "[Thunder rumbles]"
    assert w2 == []
    assert [ln["text"] for ln in parse_lines("[dance]\n[scene with fade]", CHARS)] == ["[dance]"]
    # A bracket inside a sentence is dialogue, not a direction.
    assert parse_lines("Mo: I [really] mean it.", CHARS)[0]["text"] == "I [really] mean it."


def _staged_doc():
    doc = authored_doc()
    for n in doc["nodes"]:
        if n["id"] == "s1":
            n["params"]["lines"] = (
                "[scene with flash]\nNarrator: Night fell.\n[show Pip at right with dissolve]\n"
                "Pip: Who?\n[sound door]\n[show Mo sad]\nMo: The moon is gone!\n[hide Ghost]"
            )
    doc["nodes"] += [
        N("pip", "story.character", 0, name = "Pip"),
        N("snd", "story.sound", 0, name = "Door"),
        N("door_audio", "input.audio", 0),
        N("mo_ex", "story.expressions", 0),
    ]
    doc["edges"] += [
        D("door_audio", "audio", "snd", "audio"),
        D("mo_ex", "expressions", "mo", "expressions"),
    ]
    outputs = {
        "door_audio": {"audio": {"kind": "audio", "ref": "asset:door"}},
        "mo_ex": {"expressions": {"Sad": {"kind": "image", "ref": "image:mo-sad"}}},
    }
    return doc, outputs


def test_authored_scene_compiles_stage_sounds_and_expressions():
    doc, outputs = _staged_doc()
    m, warnings = compile_story_report(doc, outputs)
    StoryManifest.model_validate(m)
    s1 = next(s for s in m["scenes"] if s["id"] == "s1")
    assert s1["transition"] == "flash"
    # Mo is wired in; Pip comes on by direction, so is not on stage from the start.
    assert s1["on_stage"] == ["mo"]
    stages = [ln["stage"] for ln in s1["lines"] if "stage" in ln]
    assert stages == [
        {"show": "pip", "at": "right", "with": "dissolve"},
        {"sound": "door"},
        {"show": "mo", "expression": "sad"},
    ]
    assert m["sounds"] == {"door": {"kind": "audio", "ref": "asset:door"}}
    mo = next(c for c in m["characters"] if c["id"] == "mo")
    assert mo["expressions"] == {"sad": {"kind": "image", "ref": "image:mo-sad"}}
    assert any("Ghost" in w for w in warnings)


def test_undefined_sound_and_soundless_block_warn():
    doc, outputs = _staged_doc()
    del outputs["door_audio"]
    m, warnings = compile_story_report(doc, outputs)
    assert "sounds" not in m
    assert any("no audio yet" in w for w in warnings)
    assert any("'door' is not defined" in w for w in warnings)


def test_old_authored_and_generated_manifests_are_unchanged():
    m, _ = compile_story_report(authored_doc())
    text = json.dumps(m)
    for key in ('"stage"', '"transition"', '"sounds"'):
        assert key not in text
    gen = repair_generated(_raw_story(), seed = 1)
    assert '"stage"' not in json.dumps(gen) and "sounds" not in gen
    schema = story_schema(4, 2)
    assert (
        "expression"
        not in schema["properties"]["scenes"]["items"]["properties"]["lines"]["items"]["properties"]
    )


def test_generator_tags_expressions_only_when_asked():
    raw = _raw_story()
    raw["scenes"][0]["lines"][0]["expression"] = "Happy"
    raw["scenes"][0]["lines"][0]["speaker"] = raw["characters"][0]["id"]
    ctx = FakeCtx(
        "story.generator",
        params = {"premise": "p", "scenes": 4, "expressions": ["happy", "sad"]},
        llm = lambda **kw: raw,
    )
    out = run(story_nodes.story_generator(ctx))["story"]
    line_schema = ctx.llm_calls[0]["schema"]["properties"]["scenes"]["items"]["properties"][
        "lines"
    ]["items"]
    assert line_schema["properties"]["expression"]["enum"] == ["neutral", "happy", "sad"]
    assert out["scenes"][0]["lines"][0]["expression"] == "happy"
    live = repair_live_scene(
        {"lines": [{"speaker": "mo", "text": "Hi", "expression": "neutral"}]},
        {"story": {"characters": [{"id": "mo", "name": "Mo"}], "scenes": []}},
    )
    assert "expression" not in live["lines"][0]


def test_expressions_block_uses_reference_workflow_with_one_seed():
    base = {"kind": "image", "ref": "image:base"}
    ctx = FakeCtx(
        "story.expressions",
        inputs = {"portrait": base},
        params = {"name": "Mo", "look": "a sleepy sloth", "expressions": ["Happy", "sad", "happy"]},
        seed = 5,
    )
    out = run(story_nodes.story_expressions(ctx))["expressions"]
    assert list(out) == ["happy", "sad"]
    assert all(v["ref"].endswith("-cut") for v in out.values())
    assert [c["workflow"] for c in ctx.images] == ["reference", "reference"]
    assert all(c["init_image"] == base and c["seed"] == 5 for c in ctx.images)
    assert "facial expression to happy" in ctx.images[0]["prompt"]
    assert "smile" in ctx.images[0]["prompt"]


def test_illustrate_makes_expression_portraits_from_the_uncut_base():
    story = repair_generated(_raw_story(), seed = 1)
    ctx = FakeCtx("story.illustrate", inputs = {"story": story}, params = {"expressions": ["happy"]})
    out = run(story_nodes.story_illustrate(ctx))["story"]
    for c in out["characters"]:
        assert set(c["expressions"]) == {"happy"}
        assert c["expressions"]["happy"]["ref"].endswith("-cut")
    refs = [i for i in ctx.images if i.get("workflow") == "reference"]
    assert len(refs) == len(out["characters"])
    assert all(not r["init_image"]["ref"].endswith("-cut") for r in refs)
    # Off by default: no extra images.
    ctx2 = FakeCtx("story.illustrate", inputs = {"story": story})
    run(story_nodes.story_illustrate(ctx2))
    assert not any(i.get("workflow") for i in ctx2.images)


def test_a_pause_that_is_not_a_finite_number_is_skipped():
    for bad in ("nan", "inf", "-inf"):
        warnings = []
        lines = parse_lines(f"[pause {bad}]\nMo: Still here.", CHARS, warnings = warnings)
        assert [ln["text"] for ln in lines] == ["Still here."], bad
        assert any("pause takes a number" in w for w in warnings)
