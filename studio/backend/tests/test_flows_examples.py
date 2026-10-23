# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import json
from pathlib import Path

import pytest

from core.flows import compiler
from core.flows.examples.build_examples import REVIEW_TOPICS, build
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


def _png(
    size: tuple[int, int],
    color: tuple,
    mode: str = "RGB",
) -> bytes:
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.new(mode, size, color).save(buf, format = "PNG")
    return buf.getvalue()


def _run_example(name: str, stubs: dict):
    """Run a bundled example through the real engine: real CPU blocks, the given stubs for models and uploads."""
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

    async def go():
        rs = await eng.start_run(app = None, flow_id = name, doc = parsed, options = {})
        await asyncio.wait_for(rs.task, timeout = 60)
        return rs

    rs = run(go())
    failed = {nid: s.error for nid, s in rs.nodes.items() if s.status == "failed"}
    assert rs.status == "completed", failed
    return rs


def test_review_insights_counts_topics_into_a_table_and_chart():
    tags = {
        "early": ("delivery", "positive", 5),
        "snapped": ("quality", "negative", 2),
        "five minutes": ("support", "positive", 5),
        "expensive": ("price", "negative", 2),
        "Setup": ("ease of use", "positive", 5),
        "rain": ("delivery", "negative", 1),
        "Solid": ("quality", "positive", 5),
        "never got": ("support", "negative", 1),
    }
    summaries = []

    async def extract(ctx):
        text = ctx.inputs["text"]
        topic, sentiment, stars = next(v for k, v in tags.items() if k in text)
        assert ctx.params["json_schema"]["properties"]["topic"]["enum"] == REVIEW_TOPICS
        return {
            "json": {
                "topic": topic,
                "sentiment": sentiment,
                "stars": stars,
                "review": text,
                "extra": 1,
            }
        }

    async def generate(ctx):
        summaries.append(ctx.inputs["context"])
        return {"text": "Fix delivery and support replies first."}

    rs = _run_example("review-insights", {"llm.extract": extract, "llm.generate": generate})
    scores = rs.outputs["scores"]
    assert scores["kind"] == "table"
    by_topic = {r["topic"]: (r["stars"], r["n"]) for r in scores["rows"]}
    assert by_topic == {
        "delivery": (3, 2),
        "quality": (3.5, 2),
        "support": (3, 2),
        "price": (2, 1),
        "ease of use": (5, 1),
    }
    reviews = rs.outputs["reviews"]
    assert [c["id"] for c in reviews["columns"]] == ["topic", "sentiment", "stars", "review"]
    assert len(reviews["rows"]) == 8
    chart = rs.nodes["plot"].outputs["chart"]
    assert chart["kind"] == "chart" and chart["x"][:2] == ["delivery", "quality"]
    series = {s["name"]: s["y"] for s in chart["series"]}
    assert series["positive"][:2] == [1, 1] and series["negative"][:2] == [1, 1]
    assert rs.outputs["summary"] == "Fix delivery and support replies first."
    assert len(summaries) == 1 and len(summaries[0][0]) == 8


def test_image_alt_text_splits_the_vision_answer_into_fields():
    asked = []

    async def upload(ctx):
        return {
            "image": await ctx.save_asset(_png((64, 48), (200, 30, 30)), "image/png", kind = "image")
        }

    async def vision(ctx):
        asked.append((ctx.inputs["image"]["kind"], ctx.params["json_schema"]["required"]))
        value = {
            "alt_text": "A red poster on a wall",
            "description": "A plain red poster hangs on a white wall in soft daylight.",
            "visible_text": "SALE",
        }
        return {"text": json.dumps(value), "json": value}

    rs = _run_example("image-alt-text", {"input.image": upload, "llm.vision": vision})
    assert asked == [("image", ["alt_text", "description", "visible_text"])]
    assert rs.outputs == {
        "alt_text": "A red poster on a wall",
        "description": "A plain red poster hangs on a white wall in soft daylight.",
        "visible_text": "SALE",
    }
    shown = rs.nodes["show"].outputs["value"]
    assert shown[0]["kind"] == "image" and shown[1] == "A red poster on a wall"


def test_background_swap_places_the_cutout_on_the_new_scene():
    import io

    from PIL import Image

    from core.flows import assets as asset_store

    calls = {}

    async def upload(ctx):
        return {
            "image": await ctx.save_asset(
                _png((300, 200), (10, 10, 200)), "image/png", kind = "image"
            )
        }

    async def remove_bg(ctx):
        calls["subject"] = ctx.params["subject"]
        cut = await ctx.save_asset(
            _png((100, 100), (255, 0, 0, 255), "RGBA"), "image/png", kind = "image"
        )
        mask = await ctx.save_asset(_png((300, 200), 255, "L"), "image/png", kind = "image")
        return {"image": cut, "mask": mask}

    async def generate(ctx):
        calls["prompt"] = ctx.inputs["prompt"]
        bg = await ctx.save_asset(_png((400, 400), (0, 255, 0)), "image/png", kind = "image")
        return {"image": bg, "images": [bg]}

    saved = []

    async def library(ctx):
        saved.append(ctx.inputs["value"])
        return {}

    rs = _run_example(
        "background-swap",
        {
            "input.image": upload,
            "image.remove_bg": remove_bg,
            "image.generate": generate,
            "output.library": library,
        },
    )
    assert calls["subject"] == "the product" and "marble kitchen counter" in calls["prompt"]

    def pixels(media):
        data, _ = asset_store.read(asset_store.parse_ref(media["ref"]))
        return Image.open(io.BytesIO(data))

    placed = rs.nodes["place"].outputs["image"]
    im = pixels(placed).convert("RGB")
    assert im.size == (400, 400)
    # The product sits centred at 62% height on the generated background, which shows around it.
    assert im.getpixel((200, 248)) == (255, 0, 0)
    assert im.getpixel((10, 10)) == (0, 255, 0)
    assert saved and saved[0][0]["ref"] == placed["ref"]
    sheet = rs.nodes["sheet"].outputs["image"]
    assert pixels(sheet).size[0] >= 3 * 512
