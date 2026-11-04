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
from core.flows.examples.build_examples import REVIEW_TOPICS, build
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
    # 56 hand-built examples plus the 9 Data Recipes learning recipes as Recipe groups.
    assert len(BUILT) == 65


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


def test_lantern_isle_3d_opens_in_first_person():
    doc = _example("lantern-isle-3d")
    world = normalize_world({"title": "Isle", "npcs": [{"name": "Bea"}]}, seed = 1)
    w, warnings = resolve_world_report(doc, {"art": {"world": world}})
    assert not warnings
    assert w["render"] == "3d" and w["view3d"] == {"time_of_day": "dusk"}
    assert [n["name"] for n in w["npcs"]] == ["Bea", "Keeper Ines"]
    w, _ = resolve_world_report(_example("pocket-village"), {"art": {"world": world}})
    assert "render" not in w


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
        await asyncio.wait_for(rs.task, timeout = 240)
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

    from core.flows import assets as asset_store
    from PIL import Image

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
    assert im.getpixel((200, 248)) == (255, 0, 0)
    assert im.getpixel((10, 10)) == (0, 255, 0)
    assert saved and saved[0][0]["ref"] == placed["ref"]
    shown = rs.nodes["show"].outputs["value"]
    assert [v["ref"] for v in shown][1:] == [rs.nodes["cut"].outputs["image"]["ref"], placed["ref"]]


def _run_example_b(name: str, stubs: dict) -> tuple[Any, dict]:
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
    rs, outputs = _run_example_b(
        "document-audiobook",
        {
            "input.file": _a_document,
            "doc.load": load,
            "audio.speak": speak,
            "audio.concat": concat,
            "output.library": _calls(saved),
        },
    )
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

    _, outputs = _run_example_b(
        "resume-tailor", {"input.file": _a_document, "doc.load": load, "llm.generate": generate}
    )
    assert outputs == {
        "missing skills": ["Tableau", "dbt"],
        "tailored bullets": ["Built daily SQL reports"],
        "cover letter": "Dear team, ...",
    }
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

    rs, outputs = _run_example_b("study-cards", {"llm.generate": generate})
    assert "RuBisCO fixes carbon dioxide" in seen[0]
    assert outputs["cards"] == cards[:2]
    assert outputs["duplicates removed"] == 1
    assert rs.nodes["show"].outputs["value"] == (
        "Q: Where does the Calvin cycle run?\nA: In the stroma.\n\nQ: What does RuBisCO fix?\nA: Carbon dioxide."
    )


async def _run_example_c(
    name: str,
    stubs: dict,
    params: dict | None = None,
):
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
    await asyncio.wait_for(rs.task, timeout = 240)
    failed = {nid: s.error for nid, s in rs.nodes.items() if s.status == "failed"}
    return rs, failed


def test_inventory_reconcile_lists_the_discrepancies_with_no_model():
    import asyncio

    doc = _example("inventory-reconcile")
    assert doc["requirements"]["models"] == []
    assert not [n for n in doc["nodes"] if SPECS_BY_TYPE[n["type"]]["lane"] == "gpu"]
    rs, failed = asyncio.run(_run_example_c("inventory-reconcile", {}))
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
        _run_example_c("release-notes", {"llm.extract": extract, "llm.generate": generate})
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
        return {"text": "", "json": {"strings": dict(strings), "language": language}}

    rs, failed = asyncio.run(_run_example_c("localization-kit", {"llm.generate": generate}))
    assert rs.status == "completed" and not failed, failed
    languages = ["Spanish", "German", "Japanese", "Brazilian Portuguese"]
    assert [c[0] for c in calls] == [
        f"Target language: {l}. Translate every UI string in the context into {l}."
        for l in languages
    ]
    from core.flows.examples.build_examples import LOCALIZATION_STRINGS

    assert all(c[1] == LOCALIZATION_STRINGS for c in calls)
    kit = rs.outputs["translations"]
    assert [k["language"] for k in kit] == languages
    assert list(kit[0]["strings"]) == list(LOCALIZATION_STRINGS)


def test_campaign_kit_crops_one_image_into_three_captioned_ads():
    import asyncio
    import io

    from core.flows import assets
    from PIL import Image

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
        _run_example_c(
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
        assert len(img.crop((0, 0, img.width, img.height // 3)).getcolors(1 << 20)) > 1
    assert sizes == [(1080, 1080), (1080, 1350), (1200, 628)]
    assert len(rs.nodes["save"].outputs["saved"]) == 3
    shown = rs.nodes["show"].outputs["value"]
    assert [v["kind"] for v in shown] == ["image"] * 3
