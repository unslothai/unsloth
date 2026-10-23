# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import json
from pathlib import Path

import pytest

from core.flows import compiler
from core.flows.examples.build_examples import build
from core.flows.nodes import transform
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
    assert len(BUILT) == 47


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


async def _run_example(
    name: str,
    stubs: dict,
    params: dict | None = None,
):
    """Run a bundled example through the real engine and CPU blocks; `stubs` replace the model blocks."""
    import asyncio

    from core.flows.compiler import parse_document
    from core.flows.engine import FlowEngine
    from core.flows.registry import load_handlers

    eng = FlowEngine()
    eng.handlers = {**load_handlers(), **stubs}
    eng.mint_key = lambda account: ("sk-unsloth-test", None)
    eng.revoke_key = lambda key_id: None
    parsed, errors = parse_document(_example(name))
    assert not errors, errors
    rs = await eng.start_run(app = None, flow_id = name, doc = parsed, options = {"params": params or {}})
    await asyncio.wait_for(rs.task, timeout = 60)
    failed = {nid: s.error for nid, s in rs.nodes.items() if s.status == "failed"}
    return rs, failed


def test_inventory_reconcile_lists_the_discrepancies_with_no_model():
    import asyncio

    doc = _example("inventory-reconcile")
    assert doc["requirements"]["models"] == []
    assert not [n for n in doc["nodes"] if SPECS_BY_TYPE[n["type"]]["lane"] == "gpu"]
    rs, failed = asyncio.run(_run_example("inventory-reconcile", {}))
    assert rs.status == "completed" and not failed, failed
    rows = [
        (r["sku"], r["status"], r["difference"], r["value"]) for r in rs.outputs["discrepancies"]
    ]
    assert rows == [
        ("BAG-01", "not counted", -12, -48),
        ("STK-10", "short", -24, -28.8),
        ("TEE-M", "short", -3, -27),
        ("CAP-01", "over", 2, 14.5),
        ("PIN-05", "not in system", 15, 0),
    ]
    totals = rs.nodes["summary"].outputs["value"]
    assert totals["kind"] == "table"
    by_status = {r["status"]: r for r in totals["rows"]}
    assert set(by_status) == {"ok", "short", "over", "not counted", "not in system"}
    assert by_status["short"]["value"] == pytest.approx(-55.8)
    assert by_status["ok"]["difference"] == 0


def test_release_notes_sorts_each_change_and_keeps_breaking_ones_apart():
    import asyncio

    seen: dict = {"extract": [], "generate": []}

    async def extract(ctx):
        text = ctx.inputs["text"]
        seen["extract"].append(text)
        lowered = text.lower()
        kind = (
            "breaking"
            if "removed" in lowered or "drop support" in lowered
            else "fix"
            if lowered.startswith("fix")
            else "internal"
            if "ci" in lowered.split()
            else "docs"
            if lowered.startswith("document")
            else "feature"
        )
        return {"json": {"kind": kind, "note": f"note: {text}"}, "text": kind}

    async def generate(ctx):
        seen["generate"].append(ctx.inputs)
        return {"text": "## 2.4.0 notes", "json": None}

    rs, failed = asyncio.run(
        _run_example("release-notes", {"llm.extract": extract, "llm.generate": generate})
    )
    assert rs.status == "completed" and not failed, failed
    changes = next(n for n in _example("release-notes")["nodes"] if n["id"] == "changes")["params"][
        "items"
    ]
    assert seen["extract"] == changes
    assert rs.outputs["release_notes"] == "## 2.4.0 notes"
    assert rs.outputs["breaking"] == [
        "note: Rename the --out flag to --output (old flag removed)",
        "note: Drop support for Python 3.8",
    ]
    (inputs,) = seen["generate"]
    assert inputs["prompt"] == "Write the release notes for version 2.4.0."
    (context,) = inputs["context"]
    assert len(context) == len(changes) - 1 and all(c["kind"] != "internal" for c in context)
    assert [c["note"] for c in rs.nodes["left_out"].outputs["value"]] == [
        "note: Bump the CI runner image"
    ]


def test_localization_kit_translates_every_string_per_language():
    import asyncio

    calls: list = []

    async def generate(ctx):
        strings = ctx.inputs["context"][0]
        calls.append((ctx.inputs["prompt"], strings))
        language = ctx.inputs["prompt"].split(" into ")[1].rstrip(".")
        return {
            "text": "",
            "json": {
                "language": language,
                "strings": [{"key": k, "text": v} for k, v in strings.items()],
            },
        }

    rs, failed = asyncio.run(_run_example("localization-kit", {"llm.generate": generate}))
    assert rs.status == "completed" and not failed, failed
    languages = ["Spanish", "German", "Japanese", "Brazilian Portuguese"]
    assert [c[0] for c in calls] == [
        f"Translate every string in the context into {l}." for l in languages
    ]
    from core.flows.examples.build_examples import LOCALIZATION_STRINGS

    assert all(c[1] == LOCALIZATION_STRINGS for c in calls)
    kit = rs.outputs["translations"]
    assert [k["language"] for k in kit] == languages
    assert [s["key"] for s in kit[0]["strings"]] == list(LOCALIZATION_STRINGS)


def test_campaign_kit_crops_one_image_into_three_captioned_ads():
    import asyncio
    import io

    from PIL import Image

    from core.flows import assets

    painted: list = []

    async def generate(ctx):
        return {
            "text": "",
            "json": {
                "headline": "Slow mornings, bottled",
                "tagline": "Cold brew for people who take their time.",
                "image_prompt": "a frosted bottle of cold brew on moss",
            },
        }

    async def paint(ctx):
        painted.append(ctx.inputs.get("prompt"))
        buf = io.BytesIO()
        Image.new("RGB", (1024, 1024), (40, 90, 60)).save(buf, "PNG")
        image = await ctx.save_asset(buf.getvalue(), "image/png", kind = "image")
        return {"image": image, "images": [image]}

    async def library(ctx):
        return {"saved": [i["ref"] for i in ctx.inputs["value"][0]]}

    rs, failed = asyncio.run(
        _run_example(
            "campaign-kit",
            {"llm.generate": generate, "image.generate": paint, "output.library": library},
        )
    )
    assert rs.status == "completed" and not failed, failed
    assert painted == ["a frosted bottle of cold brew on moss"]
    assert rs.outputs["copy"]["headline"] == "Slow mornings, bottled"
    ads = rs.nodes["ads"].outputs["list"]
    sizes = []
    for ad in ads:
        data, _ = assets.read(assets.parse_ref(ad["ref"]))
        img = Image.open(io.BytesIO(data)).convert("RGB")
        sizes.append(img.size)
        # The headline is drawn at the top of every ad over the flat test photo.
        assert len(img.crop((0, 0, img.width, img.height // 3)).getcolors(1 << 20)) > 1
    assert sizes == [(1080, 1080), (1080, 1350), (1200, 628)]
    assert len(rs.nodes["save"].outputs["saved"]) == 3
    assert rs.nodes["show"].outputs["value"]["kind"] == "image"
