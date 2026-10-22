# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import json
from pathlib import Path

import pytest

from core.flows import compiler
from core.flows.examples.build_examples import build
from core.flows.schema import FlowDocument
from core.flows.specs import SPECS_BY_TYPE
from core.flows.story import compile_story_report, walk_check
from core.flows.world import normalize_world, resolve_world_report

BUILT = {d["id"]: d for d in build()}
IDS = sorted(BUILT)


def _example(name: str) -> dict:
    return json.loads(json.dumps(BUILT[name]))


def test_there_are_ten_examples():
    assert len(BUILT) == 10


@pytest.mark.parametrize("name", IDS)
def test_example_is_valid(name):
    doc = _example(name)
    FlowDocument.model_validate(doc)
    result = compiler.validate(doc)
    assert result["ok"], result["errors"]
    assert doc["meta"]["example"] is True and doc["meta"]["tags"]
    model_nodes = [
        n for n in doc["nodes"] if SPECS_BY_TYPE[n["type"]]["model_kind"] and not n.get("disabled")
    ]
    repos = {m["repo"] for m in doc["requirements"]["models"]}
    for n in model_nodes:
        choice = n["params"].get("model")
        if choice:
            assert choice["repo"] in repos, f"{n['id']} uses a model the flow does not declare"
    assert {(n["pos"][0], n["pos"][1]) for n in doc["nodes"]}.__len__() == len(
        doc["nodes"]
    ), "overlapping nodes"


@pytest.mark.parametrize("name", [i for i in IDS if BUILT[i]["kind"] == "story"])
def test_story_examples_play_without_running(name):
    doc = _example(name)
    manifest, warnings = compile_story_report(doc, {})
    assert not warnings
    problems = walk_check(manifest)
    if manifest["mode"] == "infinite":
        assert all(p.endswith("never reaches an ending") for p in problems)
    else:
        assert problems == []
        assert any(s.get("ending") for s in manifest["scenes"])


def test_lantern_picnic_needs_no_models():
    doc = _example("lantern-picnic")
    assert doc["requirements"]["models"] == []
    assert not [n for n in doc["nodes"] if SPECS_BY_TYPE[n["type"]]["lane"] == "gpu"]


def test_pocket_village_plays_after_a_run():
    doc = _example("pocket-village")
    world = normalize_world(
        {
            "title": "Hollow",
            "npcs": [{"name": "Bea"}],
            "items": [{"name": "Bell"}],
            "quests": [{"title": "Bell", "giver": "Bea", "requires_item": "Bell"}],
        },
        seed = 1,
    )
    w, warnings = resolve_world_report(doc, {"art": {"world": world}})
    assert not warnings
    assert [n["name"] for n in w["npcs"]] == ["Bea", "Grandma Fern"]
    assert w["entry_points"] == {"npc_reply": "reply"}


@pytest.mark.parametrize("name", ["lantern-picnic", "sloth-moon"])
def test_frontend_story_fixtures_are_fresh(name):
    doc = _example(name)
    fixture = (
        Path(compiler.__file__).parents[3]
        / "frontend"
        / "tests"
        / "fixtures"
        / f"flows-{name}.story.json"
    )
    from core.flows.story import compile_story

    assert (
        json.loads(fixture.read_text(encoding = "utf-8")) == compile_story(doc, {})
    ), 'refresh the fixture: json.dumps(compile_story(example), separators=(",", ":")) -> studio/frontend/tests/fixtures'
