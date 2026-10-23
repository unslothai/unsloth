# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import asyncio
import json
from pathlib import Path
from typing import Any, Callable

import pytest

from core.flows import compiler
from core.flows.compiler import parse_document
from core.flows.engine import FlowEngine
from core.flows.examples.build_examples import build
from core.flows.nodes import text as text_nodes
from core.flows.nodes import transform
from core.flows.registry import load_handlers
from core.flows.schema import FlowDocument
from core.flows.specs import SPECS_BY_TYPE
from core.flows.story import compile_story_report, walk_check
from core.flows.world import normalize_world, resolve_world_report

from .flows_player_fakes import FakeCtx, run

BUILT = {d["id"]: d for d in build()}
IDS = sorted(BUILT)


def _example(name: str) -> dict:
    return json.loads(json.dumps(BUILT[name]))


def test_the_example_count():
    assert len(BUILT) == 46


@pytest.mark.parametrize("name", IDS)
def test_example_is_valid(name):
    doc = _example(name)
    FlowDocument.model_validate(doc)
    result = compiler.validate(doc)
    assert result["ok"], result["errors"]
    assert [w for w in result["warnings"] if not w["message"].startswith("Disabled")] == []
    for n in doc["nodes"]:
        known = {p["id"]: p for p in SPECS_BY_TYPE[n["type"]]["params"]}
        assert set(n["params"]) <= set(known), f"{n['id']} sets params its block does not have"
        for key, value in n["params"].items():
            assert value in (known[key].get("options") or [value]), f"{n['id']}.{key}={value!r}"
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


def test_smallville_lite_is_a_simulated_village():
    w, warnings = resolve_world_report(_example("smallville-lite"))
    assert not warnings
    assert w["entry_points"] == {"npc_reply": "reply", "npc_plan": "plan", "npc_chat": "chat"}
    assert len(w["npcs"]) == 6 and len(w["simulation"]["places"]) == 5
    isabella = w["npcs"][0]
    assert "Valentine's Day party at Hobbs Cafe" in isabella["seed_memories"][0]["text"]
    assert isabella["seed_memories"][0]["importance"] == 8
    assert w["vars"] == {"party_attendance": 0}
    assert "Park et al. 2023" in _example("smallville-lite")["notes"][0]["text"]


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


def test_contact_cleanup_list_blocks_clean_the_rows():
    params = {n["id"]: n["params"] for n in _example("contact-cleanup")["nodes"]}
    rows = [
        {"name": "Ana Lima", "email": "ANA@example.com", "company": "Acme"},
        {"name": "ben ortiz", "email": "ben@example.org", "company": ""},
        {"name": "Ana Lima", "email": "ana@example.com", "company": "Acme Inc"},
        {"name": "Chris Wu", "email": "", "company": "Globex"},
    ]
    kept = run(
        transform.filter_list(
            FakeCtx("list.filter", params = params["has_email"], inputs = {"items": rows})
        )
    )
    assert [r["name"] for r in kept["dropped"]] == ["Chris Wu"]
    tidy = run(
        transform.edit_fields(
            FakeCtx("json.set", params = params["tidy"], inputs = {"value": kept["items"]})
        )
    )
    unique = run(
        transform.dedupe_list(
            FakeCtx("list.dedupe", params = params["unique"], inputs = {"items": tidy["value"]})
        )
    )
    out = run(
        transform.sort_list(
            FakeCtx("list.sort", params = params["sorted"], inputs = {"items": unique["items"]})
        )
    )
    assert [(r["name"], r["email"], r["company"]) for r in out["items"]] == [
        ("Ana Lima", "ana@example.com", "Acme"),
        ("ben ortiz", "ben@example.org", "unknown"),
    ]


def _run_example(name: str, stubs: dict) -> tuple[Any, dict]:
    """Run a bundled example through the engine: the real CPU blocks, stubs for the model and file blocks."""
    eng = FlowEngine()
    eng.handlers = {**load_handlers(), **stubs}
    eng.mint_key = lambda account: ("sk-unsloth-test", None)
    eng.revoke_key = lambda key_id: None
    doc = _example(name)

    async def go():
        parsed, errors = parse_document(doc)
        assert not errors, errors
        rs = await eng.start_run(app = None, flow_id = doc["id"], doc = parsed, options = {})
        await asyncio.wait_for(rs.task, timeout = 30)
        return rs

    rs = asyncio.run(go())
    assert rs.status == "completed", {n: s.error for n, s in rs.nodes.items() if s.error}
    return rs, rs.outputs


def _calls(seen: list) -> Callable:
    async def handler(ctx):
        seen.append((ctx.node_id, dict(ctx.params), dict(ctx.inputs)))
        return {}

    return handler


async def _a_document(ctx):
    return {"file": {"kind": "document", "ref": "document:resume.pdf", "name": "resume.pdf"}}


def test_document_audiobook_reads_each_passage_and_joins_them():
    spoken, joined = [], []
    text = " ".join(f"word{i}" for i in range(1000))

    async def load(ctx):
        assert ctx.inputs["file"]["ref"] == "document:resume.pdf"
        return {"text": text, "truncated": False}

    async def speak(ctx):
        spoken.append(ctx.inputs["text"])
        return {"audio": {"kind": "audio", "ref": f"audio:{len(spoken)}"}}

    async def concat(ctx):
        joined.append([c["ref"] for c in ctx.inputs["clips"]])
        return {"audio": {"kind": "audio", "ref": "audio:book"}}

    saved: list = []
    rs, outputs = _run_example(
        "document-audiobook",
        {
            "input.file": _a_document,
            "doc.load": load,
            "audio.speak": speak,
            "audio.concat": concat,
            "output.library": _calls(saved),
        },
    )
    # 1000 words in 120-word passages is 9; For Each reads the first 8, in order.
    assert outputs["passages"][0].startswith("word0 ") and len(outputs["passages"]) == 9
    assert spoken == outputs["passages"][:8]
    assert joined == [[f"audio:{i}" for i in range(1, 9)]]
    assert rs.nodes["listen"].outputs == {"value": {"kind": "audio", "ref": "audio:book"}}
    assert saved[0][2]["value"] == [{"kind": "audio", "ref": "audio:book"}]


def test_resume_tailor_matches_then_writes_from_the_resume():
    prompts: dict[str, str] = {}
    fit = {
        "matches": [{"requirement": "SQL", "evidence": "Wrote daily SQL reports"}],
        "gaps": ["Tableau", "dbt"],
        "keywords": ["SQL", "dashboards"],
    }

    async def load(ctx):
        return {
            "text": "Ana Lima. Analyst at Acme: wrote daily SQL reports in pandas.",
            "truncated": False,
        }

    async def generate(ctx):
        prompts[ctx.node_id] = text_nodes._prompt(ctx, "prompt", "prompt")
        if ctx.node_id == "fit":
            return {"text": json.dumps(fit), "json": fit}
        made = {"bullets": ["Built daily SQL reports"], "cover_letter": "Dear team, ..."}
        return {"text": json.dumps(made), "json": made}

    _, outputs = _run_example(
        "resume-tailor", {"input.file": _a_document, "doc.load": load, "llm.generate": generate}
    )
    assert outputs == {
        "missing skills": ["Tableau", "dbt"],
        "tailored bullets": ["Built daily SQL reports"],
        "cover letter": "Dear team, ...",
    }
    # Both steps see the job post and the resume; the writer also gets the match notes.
    for step in ("fit", "write"):
        assert "Data Analyst, Riverside Health" in prompts[step]
        assert "wrote daily SQL reports in pandas" in prompts[step]
    assert '"requirement"' not in prompts["fit"] and '"requirement"' in prompts["write"]


def test_study_cards_dedupes_questions_and_prints_the_deck():
    cards = [
        {"question": "Where does the Calvin cycle run?", "answer": "In the stroma."},
        {"question": "What does RuBisCO fix?", "answer": "Carbon dioxide."},
        {"question": "where does the calvin cycle run?", "answer": "The stroma."},
    ]
    seen: list[str] = []

    async def generate(ctx):
        seen.append(text_nodes._prompt(ctx, "prompt", "prompt"))
        return {"text": json.dumps({"cards": cards}), "json": {"cards": cards}}

    rs, outputs = _run_example("study-cards", {"llm.generate": generate})
    assert "RuBisCO fixes carbon dioxide" in seen[0]
    assert outputs["cards"] == cards[:2]
    assert outputs["duplicates removed"] == 1
    assert rs.nodes["show"].outputs["value"] == (
        "Q: Where does the Calvin cycle run?\nA: In the stroma.\n\nQ: What does RuBisCO fix?\nA: Carbon dioxide."
    )
