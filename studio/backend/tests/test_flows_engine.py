# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import asyncio
import contextlib
import json
from pathlib import Path

import pytest

from core.flows import engine as engine_mod
from core.flows.compiler import parse_document
from core.flows.registry import FlowNodeError
from storage import flows_db
from core.flows import specs as specs_mod
from core.flows.nodes import logic
from core.flows.specs import P, inp, out, spec

from ._flows_helpers import (
    BASE_HANDLERS,
    cancelled_event,
    doc,
    edge,
    make_engine,
    node,
    node_status,
    result_handler,
    ret,
    run_async,
    run_doc,
    wait_until,
)


def _sql_recipe() -> dict:
    path = (
        Path(__file__).resolve().parents[1]
        / "core/flows/examples/learning_recipes/text-to-sql.json"
    )
    return json.loads(path.read_text(encoding = "utf-8"))


@run_async
async def test_linear_run_produces_outputs_and_ordered_events():
    eng = make_engine(dict(BASE_HANDLERS))
    d = doc(
        [
            node("a", "input.text", value = "hi"),
            node("t", "text.template", template = "<{}>"),
            node("o", "output.result", name = "answer"),
        ],
        [edge("a", "text", "t", "values"), edge("t", "text", "o", "value")],
    )
    rs = await run_doc(eng, d)
    assert rs.status == "completed"
    assert rs.outputs == {"answer": "<hi>"}
    seqs = [e["seq"] for e in rs.events]
    assert seqs == sorted(seqs) and len(set(seqs)) == len(seqs)
    stored = flows_db.events_after(rs.id, 0)
    assert [e["seq"] for e in stored] == seqs
    assert stored[-1]["type"] == "run.completed"
    assert flows_db.get_run(rs.id)["status"] == "completed"
    assert flows_db.node_results(rs.id)["t"]["outputs"] == {"text": "<hi>"}
    assert [e["seq"] for e in flows_db.events_after(rs.id, seqs[3])] == seqs[4:]


@run_async
async def test_if_skips_the_branch_not_taken_and_merge_joins():
    calls = []

    async def branch(ctx):
        return (
            {"then": ctx.inputs["value"]}
            if ctx.inputs["value"] == "yes"
            else {"else": ctx.inputs["value"]}
        )

    async def tagged(ctx):
        calls.append(ctx.node_id)
        return {"text": f"{ctx.node_id}:{ctx.inputs['values'][0]}"}

    async def merge(ctx):
        for port in ("a", "b", "c"):
            if port in ctx.inputs:
                return {"value": ctx.inputs[port]}
        return {}

    eng = make_engine(
        {**BASE_HANDLERS, "logic.if": branch, "text.template": tagged, "logic.merge": merge}
    )
    d = doc(
        [
            node("a", "input.text", value = "no"),
            node("if", "logic.if"),
            node("yes", "text.template"),
            node("no", "text.template"),
            node("m", "logic.merge"),
            node("o", "output.result", name = "r"),
        ],
        [
            edge("a", "text", "if", "value"),
            edge("if", "then", "yes", "values"),
            edge("if", "else", "no", "values"),
            edge("yes", "text", "m", "a"),
            edge("no", "text", "m", "b"),
            edge("m", "value", "o", "value"),
        ],
    )
    rs = await run_doc(eng, d)
    assert rs.status == "completed"
    assert node_status(rs, "yes") == "skipped"
    assert calls == ["no"]
    assert rs.outputs == {"r": "no:no"}


@run_async
async def test_failure_is_isolated_and_downstream_skipped():
    async def boom(ctx):
        raise FlowNodeError("nope")

    eng = make_engine({**BASE_HANDLERS, "text.split": boom})
    d = doc(
        [
            node("a", "input.text", value = "x"),
            node("bad", "text.split"),
            node("after", "list.join"),
            node("ok", "text.template", template = "[{}]"),
            node("o", "output.result", name = "ok"),
        ],
        [
            edge("a", "text", "bad", "text"),
            edge("bad", "items", "after", "items"),
            edge("a", "text", "ok", "values"),
            edge("ok", "text", "o", "value"),
        ],
    )
    rs = await run_doc(eng, d)
    assert rs.status == "failed"
    assert rs.nodes["bad"].error == "nope"
    assert node_status(rs, "after") == "skipped"
    assert rs.nodes["after"].message == "An upstream step failed"
    assert rs.outputs == {"ok": "[x]"}


@run_async
async def test_for_each_runs_body_per_item_and_collects():
    eng = make_engine(
        {
            **BASE_HANDLERS,
            "input.list": lambda ctx: ret({"items": ["a", "b", "c"]}),
            "text.template": lambda ctx: ret({"text": str(ctx.inputs["values"][0]).upper()}),
        }
    )
    d = doc(
        [
            node("l", "input.list"),
            node("fe", "logic.for_each"),
            node("t", "text.template", group = "fe"),
            node("o", "output.result", name = "all"),
        ],
        [
            edge("l", "items", "fe", "items"),
            edge("fe", "item", "t", "values"),
            edge("t", "text", "fe", "collect"),
            edge("fe", "results", "o", "value"),
        ],
    )
    rs = await run_doc(eng, d)
    assert rs.status == "completed", rs.error
    assert rs.outputs == {"all": ["A", "B", "C"]}
    iterations = [
        e["data"].get("iteration")
        for e in rs.events
        if e["type"] == "node.output" and e["node_id"] == "t"
    ]
    assert iterations == [[0], [1], [2]]


@run_async
async def test_for_each_waits_for_outside_inputs_of_its_body():
    ref_started, release_ref = asyncio.Event(), asyncio.Event()

    async def slow_ref(ctx):
        ref_started.set()
        await release_ref.wait()
        return {"text": "ref"}

    seen = []

    async def body(ctx):
        seen.append(ctx.inputs.get("values"))
        return {"text": "x"}

    eng = make_engine(
        {
            **BASE_HANDLERS,
            "llm.generate": slow_ref,
            "input.list": lambda ctx: ret({"items": ["a", "b"]}),
            "text.template": body,
        }
    )
    d = doc(
        [
            node("l", "input.list"),
            node("ref", "llm.generate"),
            node("fe", "logic.for_each"),
            node("t", "text.template", group = "fe"),
        ],
        [
            edge("l", "items", "fe", "items"),
            edge("fe", "item", "t", "values"),
            edge("ref", "text", "t", "values"),
            edge("t", "text", "fe", "collect"),
        ],
    )
    parsed, errors = parse_document(d)
    assert not errors, errors
    rs = await eng.start_run(app = None, flow_id = d["id"], doc = parsed, options = {})
    # The list is ready and the body's other input is still running: the body must not start yet.
    await wait_until(
        lambda: ref_started.is_set() and node_status(rs, "l") == "done", what = "the list and ref"
    )
    assert seen == []
    release_ref.set()
    await asyncio.wait_for(rs.task, 20)
    assert rs.status == "completed", rs.error
    assert node_status(rs, "t") != "skipped" and len(seen) == 2 and all("ref" in v for v in seen)


@run_async
async def test_cache_hits_on_second_run_and_force_bypasses():
    calls = []

    async def counted(ctx):
        calls.append(ctx.node_id)
        return {"text": "v"}

    eng = make_engine({**BASE_HANDLERS, "text.template": counted})
    d = doc(
        [node("a", "input.text", value = "k"), node("t", "text.template")],
        [edge("a", "text", "t", "values")],
    )
    first = await run_doc(eng, d)
    second = await run_doc(eng, d)
    assert calls == ["t"]
    assert node_status(first, "t") == "done" and node_status(second, "t") == "cached"
    moved = {**d, "nodes": [{**n, "pos": [500, 500]} for n in d["nodes"]]}
    assert node_status(await run_doc(eng, moved), "t") == "cached"
    await run_doc(eng, d, force_nodes = ["t"])
    assert calls == ["t", "t"]
    changed = {**d, "nodes": [d["nodes"][0], {**d["nodes"][1], "params": {"template": "other"}}]}
    await run_doc(eng, changed)
    assert calls == ["t", "t", "t"]


@run_async
async def test_side_effect_nodes_never_cache():
    calls = []

    async def save(ctx):
        calls.append(1)
        return {"saved": []}

    eng = make_engine({**BASE_HANDLERS, "output.library": save})
    d = doc(
        [node("a", "input.text", value = "k"), node("s", "output.library")],
        [edge("a", "text", "s", "value")],
    )
    await run_doc(eng, d)
    await run_doc(eng, d)
    assert calls == [1, 1]


@run_async
async def test_cancel_mid_run_keeps_finished_outputs_and_fires_hooks():
    started, resume = asyncio.Event(), asyncio.Event()
    hooked = []

    async def slow(ctx):
        async def hook():
            hooked.append(ctx.node_id)

        ctx.on_cancel(hook)
        started.set()
        await resume.wait()  # held until the test has seen cancel() answer
        ctx.check_cancelled()
        return {"text": "late"}

    eng = make_engine({**BASE_HANDLERS, "llm.generate": slow})
    d = doc(
        [node("a", "input.text", value = "p"), node("g", "llm.generate"), node("t", "text.template")],
        [edge("a", "text", "g", "prompt"), edge("g", "text", "t", "values")],
    )
    parsed, _ = parse_document(d)
    rs = await eng.start_run(app = None, flow_id = "f1", doc = parsed, options = {})
    await asyncio.wait_for(started.wait(), 60)
    assert await eng.cancel(rs.id) == "cancelling"
    resume.set()
    await asyncio.wait_for(rs.task, 60)
    assert rs.status == "cancelled"
    assert hooked == ["g"]
    assert node_status(rs, "a") == "done"
    assert node_status(rs, "g") == "cancelled"
    assert node_status(rs, "t") in ("cancelled", "skipped")
    assert flows_db.get_run(rs.id)["status"] == "cancelled"
    assert rs.events[-1]["type"] == "run.cancelled"


@run_async
async def test_cancel_says_cancelling_even_if_the_run_ends_while_hooks_run():
    started, resume = asyncio.Event(), asyncio.Event()
    held = []

    async def slow(ctx):
        async def hook():
            # The run reaches "cancelled" before cancel() answers.
            resume.set()
            await asyncio.wait({held[0].task}, timeout = 30)

        ctx.on_cancel(hook)
        started.set()
        await resume.wait()
        ctx.check_cancelled()
        return {"text": "late"}

    eng = make_engine({**BASE_HANDLERS, "llm.generate": slow})
    d = doc(
        [node("a", "input.text", value = "p"), node("g", "llm.generate")],
        [edge("a", "text", "g", "prompt")],
    )
    parsed, _ = parse_document(d)
    rs = await eng.start_run(app = None, flow_id = "f1", doc = parsed, options = {})
    held.append(rs)
    await asyncio.wait_for(started.wait(), 60)
    assert await eng.cancel(rs.id) == "cancelling"
    await asyncio.wait_for(rs.task, 60)
    assert rs.status == "cancelled"
    assert await eng.cancel(rs.id) == "cancelled"


@run_async
async def test_gpu_nodes_prefer_the_resident_model():
    order = []

    async def gen(ctx):
        repo = ctx.params["model"]["repo"]
        order.append(repo)
        ctx.resident["llm"] = repo
        return {"text": repo}

    eng = make_engine({**BASE_HANDLERS, "llm.generate": gen})
    models = ["m1", "m2", "m1", "m2", "m1"]
    nodes = [
        node(f"g{i}", "llm.generate", prompt = "x", model = {"repo": m}) for i, m in enumerate(models)
    ]
    rs = await run_doc(eng, doc(nodes))
    assert rs.status == "completed"
    assert order == ["m1", "m1", "m1", "m2", "m2"]


@run_async
async def test_target_nodes_run_only_their_ancestors():
    calls = []

    async def counted(ctx):
        calls.append(ctx.node_id)
        return {"text": ctx.node_id}

    eng = make_engine({"input.text": counted, "text.template": counted})
    d = doc(
        [node("a", "input.text"), node("b", "input.text"), node("t", "text.template")],
        [edge("a", "text", "t", "values")],
    )
    await run_doc(eng, d, target_nodes = ["t"])
    assert sorted(calls) == ["a", "t"]


@run_async
async def test_baked_outputs_are_used_without_running():
    async def never(ctx):
        raise AssertionError("baked node ran")

    eng = make_engine({**BASE_HANDLERS, "llm.generate": never})
    d = doc(
        [node("g", "llm.generate", prompt = "x"), node("o", "output.result", name = "r")],
        [edge("g", "text", "o", "value")],
        baked = {"g": {"text": "from the package"}},
    )
    rs = await run_doc(eng, d)
    assert rs.outputs == {"r": "from the package"}
    assert rs.nodes["g"].message == "Baked"


@run_async
async def test_invoke_runs_entrypoint_with_supplied_inputs():
    async def next_scene(ctx):
        return {"scene": {"state": ctx.inputs.get("state"), "seed": ctx.seed}}

    eng = make_engine({**BASE_HANDLERS, "story.next_scene": next_scene})
    d = doc([node("n", "story.next_scene")])
    parsed, _ = parse_document(d)
    out = await eng.invoke(
        app = None, flow_id = "f1", doc = parsed, entry = "n", inputs = {"state": {"hp": 3}}
    )
    assert out["scene"]["state"] == {"hp": 3}
    with pytest.raises(FlowNodeError):
        await eng.invoke(app = None, flow_id = "f1", doc = parsed, entry = "missing")
    d2 = doc([node("a", "input.text")])
    parsed2, _ = parse_document(d2)
    with pytest.raises(FlowNodeError, match = "not an entry point"):
        await eng.invoke(app = None, flow_id = "f1", doc = parsed2, entry = "a")


@run_async
async def test_cancelled_invoke_fires_the_entrypoints_stop_hooks():
    started = asyncio.Event()
    hooked = []

    async def scene_art(ctx):
        async def hook():
            hooked.append(ctx.node_id)

        ctx.on_cancel(hook)
        started.set()
        await asyncio.Event().wait()  # held until the invoke is cancelled
        return {"scene": {}}

    eng = make_engine({**BASE_HANDLERS, "story.next_scene": scene_art})
    parsed, _ = parse_document(doc([node("n", "story.next_scene")]))
    task = asyncio.create_task(eng.invoke(app = None, flow_id = "f1", doc = parsed, entry = "n"))
    await asyncio.wait_for(started.wait(), 20)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert hooked == ["n"]


@run_async
async def test_run_is_invisible_to_another_account():
    from utils.account_context import AccountContext, arun_as

    eng = make_engine(dict(BASE_HANDLERS))
    rs = await run_doc(eng, doc([node("a", "input.text", value = "x")]))
    bob = AccountContext("bob-flows", "bob")
    assert await arun_as(bob, _live(eng, rs.id)) is None


async def _live(eng, run_id):
    return eng.live(run_id)


def test_orphaned_runs_are_interrupted():
    flows_db.create_run("orphan", "f1", {})
    flows_db.update_run("orphan", status = "running")
    assert flows_db.reconcile_orphaned_runs() >= 1
    assert flows_db.get_run("orphan")["status"] == "interrupted"
    flows_db.update_run("orphan", status = "running")
    assert flows_db.get_run("orphan")["status"] == "interrupted"


def test_progress_is_coalesced(monkeypatch):
    from core.flows.engine import RunState

    times = iter([0.0, 0.05, 0.1, 0.3])
    monkeypatch.setattr(engine_mod.time, "monotonic", lambda: next(times))
    rs = RunState(id = "r", flow_id = "f", account = None, doc = {}, plan = None, options = {}, persist = False)
    for i in range(3):
        rs.progress("n", {"fraction": i / 10})
    assert [e["data"]["fraction"] for e in rs.events] == [0.0]
    rs.flush()
    assert [e["data"]["fraction"] for e in rs.events] == [0.0, 0.2]


@run_async
async def test_a_timed_out_run_stops_its_node_tasks_and_fires_hooks(monkeypatch):
    hooked, stopped = [], asyncio.Event()

    async def stuck(ctx):
        async def hook():
            hooked.append(ctx.node_id)

        ctx.on_cancel(hook)
        try:
            await asyncio.Event().wait()  # held until the time limit cancels it
        except asyncio.CancelledError:
            stopped.set()
            raise
        return {"text": "late"}

    monkeypatch.setattr(engine_mod, "MAX_RUN_SECONDS", 0.5)
    eng = make_engine({**BASE_HANDLERS, "llm.generate": stuck})
    d = doc(
        [node("a", "input.text", value = "p"), node("g", "llm.generate")],
        [edge("a", "text", "g", "prompt")],
    )
    rs = await run_doc(eng, d)
    assert rs.status == "failed" and "time limit" in rs.error
    assert hooked == ["g"]
    assert stopped.is_set()
    assert node_status(rs, "g") == "cancelled"


@run_async
async def test_stop_reaches_every_parallel_iteration_still_running():
    hooked, first_done, second_up = [], asyncio.Event(), asyncio.Event()

    async def job(ctx):
        item = ctx.inputs["values"][0]

        async def hook():
            hooked.append(item)

        ctx.on_cancel(hook)
        if item == "a":
            await asyncio.wait_for(second_up.wait(), 10)
            first_done.set()
            return {"text": item}
        stop = await cancelled_event(ctx)
        second_up.set()
        await stop.wait()
        ctx.check_cancelled()

    eng = make_engine(
        {
            **BASE_HANDLERS,
            "input.list": lambda ctx: ret({"items": ["a", "b"]}),
            "text.template": job,
        }
    )
    d = doc(
        [
            node("l", "input.list"),
            node("fe", "logic.for_each", parallel = True),
            node("t", "text.template", group = "fe"),
        ],
        [
            edge("l", "items", "fe", "items"),
            edge("fe", "item", "t", "values"),
            edge("t", "text", "fe", "collect"),
        ],
    )
    parsed, errors = parse_document(d)
    assert not errors, errors
    rs = await eng.start_run(app = None, flow_id = d["id"], doc = parsed, options = {})
    # The first iteration's hooks are dropped in the same step its handler returns.
    await asyncio.wait_for(first_done.wait(), 10)
    await eng.cancel(rs.id)
    await asyncio.wait_for(rs.task, 20)
    assert hooked == ["b"]


def test_an_imported_media_cover_shows_in_the_flow_list():
    cover = {"kind": "image", "url": "/api/flows/assets/abc.png", "ref": "sha256:abc"}
    flows_db.save_flow(
        {**doc([node("a", "input.text")]), "id": "covered", "meta": {"cover": cover}}
    )
    summary = next(f for f in flows_db.list_flows() if f["id"] == "covered")
    assert summary["cover"] == "/api/flows/assets/abc.png"


@run_async
async def test_forcing_a_for_each_reruns_its_body():
    calls = []

    async def counted(ctx):
        calls.append(ctx.inputs["values"][0])
        return {"text": str(ctx.inputs["values"][0]).upper()}

    eng = make_engine(
        {
            **BASE_HANDLERS,
            "input.list": lambda ctx: ret({"items": ["a", "b"]}),
            "text.template": counted,
        }
    )
    d = doc(
        [
            node("l", "input.list"),
            node("fe", "logic.for_each"),
            node("t", "text.template", group = "fe"),
            node("o", "output.result", name = "all"),
        ],
        [
            edge("l", "items", "fe", "items"),
            edge("fe", "item", "t", "values"),
            edge("t", "text", "fe", "collect"),
            edge("fe", "results", "o", "value"),
        ],
    )
    await run_doc(eng, d)
    await run_doc(eng, d)
    assert calls == ["a", "b"]
    rs = await run_doc(eng, d, force_nodes = ["fe"])
    assert rs.status == "completed", rs.error
    assert calls == ["a", "b", "a", "b"]


@run_async
async def test_an_error_outside_a_handler_fails_the_run():
    eng = make_engine({**BASE_HANDLERS, "input.list": lambda ctx: ret({"items": ["a"]})})
    d = doc(
        [
            node("l", "input.list"),
            node("fe", "logic.for_each", max_items = "lots"),
            node("t", "text.template", group = "fe"),
            node("o", "output.result", name = "all"),
        ],
        [
            edge("l", "items", "fe", "items"),
            edge("fe", "item", "t", "values"),
            edge("t", "text", "fe", "collect"),
            edge("fe", "results", "o", "value"),
        ],
    )
    rs = await run_doc(eng, d)
    assert rs.status == "failed", rs.status
    assert node_status(rs, "fe") == "failed"


@run_async
async def test_an_unchanged_recipe_group_hands_back_its_last_dataset():
    from core.flows.recipe_import import GENERATE_TYPE, flow_from_recipe

    sql = _sql_recipe()
    jobs = []

    async def generate(ctx):
        jobs.append(ctx.node_id)
        return {
            "dataset": {"kind": "dataset", "ref": f"recipes/r{len(jobs)}/parquet-files"},
            "preview": [],
        }

    eng = make_engine({**BASE_HANDLERS, GENERATE_TYPE: generate})
    d, _ = flow_from_recipe(sql, name = "Text to SQL")
    step = next(n for n in d["nodes"] if n["type"] == GENERATE_TYPE)
    first = await run_doc(eng, d)
    second = await run_doc(eng, d)
    assert jobs == [step["id"]], "the second run reuses the dataset"
    assert node_status(first, step["id"]) == "done" and node_status(second, step["id"]) == "cached"
    await run_doc(eng, d, force_nodes = [step["id"]])
    assert len(jobs) == 2, "Re-roll runs a fresh job"
    edited = json.loads(json.dumps(d))
    column = next(n for n in edited["nodes"] if n["type"] == "recipe.column")
    column["params"]["column"]["params"]["values"] = ["Retail", "Health"]
    await run_doc(eng, edited)
    assert len(jobs) == 3, "an edited column runs the recipe again"


@run_async
async def test_a_run_that_may_not_use_mcp_gets_no_cached_mcp_recipe_output(monkeypatch):
    from core.flows.recipe_import import GENERATE_TYPE, flow_from_recipe

    sql = _sql_recipe()
    sql["recipe"]["mcp_providers"] = [
        {"name": "docs", "provider_type": "streamable_http", "endpoint": "https://mcp.example/mcp"}
    ]
    jobs = []

    async def generate(ctx):
        jobs.append(ctx.node_id)
        return {"dataset": {"kind": "dataset", "ref": "recipes/r1/parquet-files"}, "preview": []}

    eng = make_engine({**BASE_HANDLERS, GENERATE_TYPE: generate})
    d, _ = flow_from_recipe(sql, name = "Text to SQL")
    step = next(n for n in d["nodes"] if n["type"] == GENERATE_TYPE)
    monkeypatch.setattr(flows_db, "get_capabilities", lambda flow_id: {"mcp": True})
    monkeypatch.setattr(engine_mod, "_origin", lambda account, kind: {"kind": "ui", "owner": True})
    owner = await run_doc(eng, d)
    assert node_status(owner, step["id"]) == "done" and jobs == [step["id"]]
    monkeypatch.setattr(
        engine_mod, "_origin", lambda account, kind: {"kind": "api_key", "owner": False}
    )
    api = await run_doc(eng, d)
    assert (
        node_status(api, step["id"]) == "failed"
    ), "the owner's cached MCP rows are not handed out"
    assert "Studio owner" in api.nodes[step["id"]].error and jobs == [step["id"]]


@pytest.mark.parametrize("given, kept", [(10**9, 10_000), (-5, 0)])
def test_numeric_params_are_clamped_to_the_catalog(given, kept):
    from core.flows.engine import param_values
    from core.flows.specs import SPECS_BY_TYPE

    spec = SPECS_BY_TYPE["audio.concat"]
    assert param_values(spec, {"params": {"gap_ms": given}})["gap_ms"] == kept


@run_async
async def test_ignore_baked_recomputes_shipped_outputs():
    ran = []

    async def text(ctx):
        ran.append(ctx.node_id)
        return {"text": "fresh"}

    eng = make_engine({**BASE_HANDLERS, "input.text": text})
    d = doc([node("a", "input.text", value = "x")], baked = {"a": {"text": "shipped"}})
    rs = await run_doc(eng, d)
    assert ran == [] and node_status(rs, "a") == "cached"
    rs = await run_doc(eng, d, ignore_baked = True)
    assert ran == ["a"] and rs.nodes["a"].outputs == {"text": "fresh"}


@run_async
async def test_a_full_run_leaves_player_entrypoints_idle():
    async def reply(ctx):
        raise AssertionError("an entrypoint ran in a full run")

    eng = make_engine({**BASE_HANDLERS, "story.next_scene": reply})
    d = doc([node("a", "input.text", value = "x"), node("r", "story.next_scene")])
    rs = await run_doc(eng, d)
    assert rs.status == "completed", rs.error
    assert node_status(rs, "r") == "idle"


@run_async
async def test_use_the_loaded_model_is_never_cached():
    calls = []

    async def gen(ctx):
        calls.append(ctx.params.get("model"))
        return {"text": "hi"}

    eng = make_engine({**BASE_HANDLERS, "llm.generate": gen})

    def flow(model):
        return doc(
            [node("a", "input.text", value = "p"), node("g", "llm.generate", model = model)],
            [edge("a", "text", "g", "prompt")],
        )

    await run_doc(eng, flow(None))
    second = await run_doc(eng, flow(None))
    assert len(calls) == 2 and node_status(second, "g") == "done"
    named = {"repo": "unsloth/x-GGUF", "variant": "Q4_K_M"}
    await run_doc(eng, flow(named))
    assert node_status(await run_doc(eng, flow(named)), "g") == "cached"


@run_async
async def test_a_run_value_change_is_not_served_from_the_cache():
    calls = []

    async def template(ctx):
        calls.append(ctx.flow_params.get("name"))
        return {"text": f"hi {ctx.flow_params.get('name')}"}

    eng = make_engine({**BASE_HANDLERS, "text.template": template})
    d = doc([node("t", "text.template", template = "hi {{name}}")])
    first = await run_doc(eng, d, params = {"name": "Alice"})
    second = await run_doc(eng, d, params = {"name": "Bob"})
    again = await run_doc(eng, d, params = {"name": "Bob"})
    assert calls == ["Alice", "Bob"]
    assert node_status(first, "t") == "done" and node_status(second, "t") == "done"
    assert node_status(again, "t") == "cached"


def test_cached_media_the_user_deleted_is_not_served(monkeypatch):
    from core.flows.nodes import _common
    from core.inference import audio_gallery, image_gallery, video_gallery

    present = {"keep"}
    for mod, fn in (
        (image_gallery, "owned_image_path"),
        (audio_gallery, "owned_audio_path"),
        (video_gallery, "owned_video_path"),
    ):
        monkeypatch.setattr(mod, fn, lambda media_id: "/x" if media_id in present else None)
    kept = {"images": [{"kind": "image", "ref": "image:keep"}], "text": "hi"}
    gone = {"clip": {"kind": "audio", "ref": "audio:gone"}}
    assert _common.gallery_media_present(kept)
    assert not _common.gallery_media_present(gone)
    assert _common.gallery_media_present({"image": {"kind": "image", "ref": "sha256:abc"}, "n": 3})


@run_async
async def test_a_baked_chat_answers_the_new_message_not_the_exported_one():
    from core.flows.nodes import chat

    eng = make_engine(
        {**BASE_HANDLERS, "chat.input": chat.chat_input, "chat.output": chat.chat_output}
    )
    d = doc(
        [node("i", "chat.input"), node("o", "chat.output")],
        [edge("i", "message", "o", "reply")],
        baked = {"i": {"message": "exported question", "history": []}},
    )
    parsed, _ = parse_document(d)
    out = await eng.invoke(
        app = None, flow_id = "f1", doc = parsed, entry = "o", params = {"message": "new question"}
    )
    assert out == {"reply": "new question"}


@run_async
async def test_a_for_each_failure_in_an_early_iteration_fails_the_run():
    calls = []

    async def flaky(ctx):
        calls.append(ctx.inputs.get("values"))
        if len(calls) == 1:
            raise RuntimeError("boom")
        return {"text": "ok"}

    eng = make_engine(
        {
            **BASE_HANDLERS,
            "input.list": lambda ctx: ret({"items": ["a", "b"]}),
            "text.template": flaky,
        }
    )
    d = doc(
        [
            node("l", "input.list"),
            node("fe", "logic.for_each"),
            node("t", "text.template", group = "fe"),
        ],
        [
            edge("l", "items", "fe", "items"),
            edge("fe", "item", "t", "values"),
            edge("t", "text", "fe", "collect"),
        ],
    )
    rs = await run_doc(eng, d)
    assert len(calls) == 2
    assert rs.status == "failed"


@run_async
async def test_parallel_for_each_keeps_the_cpu_limit():
    live = 0
    peak = 0
    release = asyncio.Event()

    async def slow(ctx):
        # Every iteration holds its slot until the test releases them all.
        nonlocal live, peak
        live += 1
        peak = max(peak, live)
        await release.wait()
        live -= 1
        return {"text": "ok"}

    eng = make_engine(
        {
            **BASE_HANDLERS,
            "input.list": lambda ctx: ret({"items": list(range(12))}),
            "text.template": slow,
        }
    )
    d = doc(
        [
            node("l", "input.list"),
            node("fe", "logic.for_each", parallel = True),
            node("t", "text.template", group = "fe"),
        ],
        [edge("l", "items", "fe", "items"), edge("fe", "item", "t", "values")],
    )
    parsed, errors = parse_document(d)
    assert not errors, errors
    rs = await eng.start_run(app = None, flow_id = d["id"], doc = parsed, options = {})
    limit = engine_mod.CPU_CONCURRENCY
    await wait_until(lambda: live >= limit, timeout = 30, what = "the iterations to fill the CPU limit")
    # Proving that nothing more starts needs a window: give an over-limit iteration time to show.
    with contextlib.suppress(AssertionError):
        await wait_until(lambda: live > limit, timeout = 0.5)
    release.set()
    await asyncio.wait_for(rs.task, 20)
    assert rs.status == "completed", rs.error
    assert peak == limit


@run_async
async def test_parallel_for_each_shares_one_cpu_limit_across_iterations():
    live = 0
    peak = 0
    release = asyncio.Event()

    async def slow(ctx):
        nonlocal live, peak
        live += 1
        peak = max(peak, live)
        await release.wait()
        live -= 1
        return {"text": "ok"}

    eng = make_engine(
        {
            **BASE_HANDLERS,
            "input.list": lambda ctx: ret({"items": list(range(12))}),
            "text.template": slow,
        }
    )
    # Two independent steps per iteration: a per-iteration limit would let 4 x 2 run at once.
    d = doc(
        [
            node("l", "input.list"),
            node("fe", "logic.for_each", parallel = True),
            node("t", "text.template", group = "fe"),
            node("u", "text.template", group = "fe"),
        ],
        [
            edge("l", "items", "fe", "items"),
            edge("fe", "item", "t", "values"),
            edge("fe", "item", "u", "values"),
        ],
    )
    parsed, errors = parse_document(d)
    assert not errors, errors
    rs = await eng.start_run(app = None, flow_id = d["id"], doc = parsed, options = {})
    limit = engine_mod.CPU_CONCURRENCY
    await wait_until(lambda: live >= limit, timeout = 30, what = "the steps to fill the CPU limit")
    with contextlib.suppress(AssertionError):
        await wait_until(lambda: live > limit, timeout = 0.5)
    release.set()
    await asyncio.wait_for(rs.task, 20)
    assert rs.status == "completed", rs.error
    assert peak == limit


@run_async
async def test_for_each_keeps_null_results_in_place():
    async def passthrough(ctx):
        return {"value": ctx.inputs.get("value")}

    eng = make_engine(
        {
            **BASE_HANDLERS,
            "input.list": lambda ctx: ret({"items": ["a", None, "c"]}),
            "json.passthrough": passthrough,
        }
    )
    d = doc(
        [
            node("l", "input.list"),
            node("fe", "logic.for_each"),
            node("t", "text.template", group = "fe"),
        ],
        [
            edge("l", "items", "fe", "items"),
            edge("fe", "item", "t", "values"),
            edge("t", "text", "fe", "collect"),
        ],
    )

    async def maybe_null(ctx):
        values = ctx.inputs.get("values") or []
        return {"text": None if values == [None] else "x"}

    eng.handlers["text.template"] = maybe_null
    rs = await run_doc(eng, d)
    assert rs.status == "completed", rs.error
    assert len(rs.nodes["fe"].outputs["results"]) == 3


@run_async
async def test_cancel_hooks_can_still_reach_studio():
    started = asyncio.Event()
    forwarded = []

    async def fake_forward(caller, method, path, **kw):
        forwarded.append(path)
        raise RuntimeError("stop sent")

    async def slow(ctx):
        async def hook():
            await ctx.call("POST", "/api/export/cancel", json = {})

        ctx.on_cancel(hook)
        stop = await cancelled_event(ctx)
        started.set()
        await stop.wait()
        ctx.check_cancelled()
        return {"text": "late"}

    original = engine_mod.flow_caller.forward
    engine_mod.flow_caller.forward = fake_forward
    try:
        eng = make_engine({**BASE_HANDLERS, "llm.generate": slow})
        d = doc(
            [node("a", "input.text", value = "p"), node("g", "llm.generate")],
            [edge("a", "text", "g", "prompt")],
        )
        parsed, _ = parse_document(d)
        rs = await eng.start_run(app = None, flow_id = "f1", doc = parsed, options = {})
        await asyncio.wait_for(started.wait(), 60)
        await eng.cancel(rs.id)
        await asyncio.wait_for(rs.task, 60)
    finally:
        engine_mod.flow_caller.forward = original
    assert rs.status == "cancelled"
    assert forwarded == ["/api/export/cancel"]


@run_async
async def test_a_hook_registered_after_cancel_still_fires():
    started = asyncio.Event()
    release = asyncio.Event()
    fired = []

    async def starts_a_job(ctx):
        started.set()
        await release.wait()

        async def hook():
            fired.append(ctx.node_id)

        ctx.on_cancel(hook)  # the run is already cancelling: the hook is started at once
        ctx.check_cancelled()
        return {"text": "x"}

    eng = make_engine({**BASE_HANDLERS, "llm.generate": starts_a_job})
    d = doc(
        [node("a", "input.text", value = "p"), node("g", "llm.generate")],
        [edge("a", "text", "g", "prompt")],
    )
    parsed, _ = parse_document(d)
    rs = await eng.start_run(app = None, flow_id = "f1", doc = parsed, options = {})
    await asyncio.wait_for(started.wait(), 60)
    await eng.cancel(rs.id)
    release.set()
    await asyncio.wait_for(rs.task, 60)
    await wait_until(lambda: fired == ["g"], what = "the late cancel hook")
    assert rs.status == "cancelled"


@run_async
async def test_a_request_aborted_by_stop_reads_as_cancelled():
    started = asyncio.Event()

    async def aborted(ctx):
        stop = await cancelled_event(ctx)
        started.set()
        await stop.wait()
        raise FlowNodeError(
            "Client disconnected before an upstream llama-server generation slot was available"
        )

    eng = make_engine({**BASE_HANDLERS, "llm.generate": aborted})
    d = doc(
        [node("a", "input.text", value = "p"), node("g", "llm.generate")],
        [edge("a", "text", "g", "prompt")],
    )
    parsed, _ = parse_document(d)
    rs = await eng.start_run(app = None, flow_id = "f1", doc = parsed, options = {})
    await asyncio.wait_for(started.wait(), 60)
    await eng.cancel(rs.id)
    await asyncio.wait_for(rs.task, 60)
    assert rs.status == "cancelled"
    assert node_status(rs, "g") == "cancelled" and rs.nodes["g"].error is None


@run_async
async def test_repeated_invokes_of_one_revision_compile_the_flow_once(monkeypatch):
    from core.flows import engine as engine_mod

    compiled = []
    real = engine_mod.compile_flow
    monkeypatch.setattr(engine_mod, "compile_flow", lambda d: compiled.append(1) or real(d))

    async def next_scene(ctx):
        return {"scene": {"ok": True}}

    eng = make_engine({**BASE_HANDLERS, "story.next_scene": next_scene})
    parsed, _ = parse_document(doc([node("n", "story.next_scene")]))
    for _ in range(5):
        await eng.invoke(app = None, flow_id = "f1", doc = parsed, entry = "n", revision = 1)
    assert len(compiled) == 1
    await eng.invoke(app = None, flow_id = "f1", doc = parsed, entry = "n", revision = 2)
    await eng.invoke(app = None, flow_id = "f1", doc = parsed, entry = "n")
    assert len(compiled) == 3
    await eng.release_spare_keys()


def _add_spec(monkeypatch, found: dict) -> None:
    monkeypatch.setitem(specs_mod.SPECS_BY_TYPE, found["type"], found)


def _rows_spec():
    return spec(
        "test.rows",
        "Rows",
        "logic",
        "Publishes rows as it goes.",
        outputs = [out("rows", "list")],
        params = [P("fail", "bool", False)],
    )


@run_async
async def test_partial_rows_stream_but_never_feed_the_next_step(monkeypatch):
    _add_spec(monkeypatch, _rows_spec())
    seen = {}

    async def rows(ctx):
        ctx.partial("rows", [1], step = 1, total = 3)
        ctx.partial("rows", [1, 2], step = 2, total = 3)
        return {"rows": [1, 2, 3]}

    async def result(ctx):
        seen["value"] = ctx.inputs.get("value")
        return {"value": ctx.inputs.get("value")}

    eng = make_engine({"test.rows": rows, "output.result": result})
    d = doc(
        [node("r", "test.rows"), node("o", "output.result", name = "r")],
        [edge("r", "rows", "o", "value")],
    )
    rs = await run_doc(eng, d)
    assert rs.status == "completed"
    assert seen["value"] == [1, 2, 3]
    partials = [e for e in rs.events if e["type"] == "node.partial"]
    assert partials and partials[-1]["data"] == {
        "port": "rows",
        "value": [1, 2],
        "step": 2,
        "total": 3,
    }
    assert "partial" not in rs.nodes["r"].snapshot()
    assert rs.nodes["r"].outputs == {"rows": [1, 2, 3]}


@run_async
async def test_stored_partials_keep_their_position_but_not_the_value(monkeypatch):
    from core.flows import engine as engine_mod

    _add_spec(monkeypatch, _rows_spec())
    stored = []
    real = engine_mod.flows_db.write_batch

    def capture(run_id, events, *rest):
        stored.extend(event for _, event in events)
        return real(run_id, events, *rest)

    monkeypatch.setattr(engine_mod.flows_db, "write_batch", capture)

    async def rows(ctx):
        ctx.partial("rows", list(range(500)), step = 1, total = 2)
        return {"rows": [1]}

    rs = await run_doc(make_engine({"test.rows": rows}), doc([node("r", "test.rows")]))
    live = [e for e in rs.events if e["type"] == "node.partial"]
    kept = [e for e in stored if e["type"] == "node.partial"]
    assert live and live[-1]["data"]["value"] == list(range(500))
    assert kept and all("value" not in e["data"] for e in kept)
    assert kept[-1]["data"]["step"] == 1


@run_async
async def test_a_failed_step_keeps_its_last_partial(monkeypatch):
    _add_spec(monkeypatch, _rows_spec())

    async def rows(ctx):
        ctx.partial("rows", ["a", "b"], step = 2, total = 5)
        raise FlowNodeError("the third row ran out of memory")

    eng = make_engine({"test.rows": rows})
    rs = await run_doc(eng, doc([node("r", "test.rows")]))
    snap = rs.nodes["r"].snapshot()
    assert snap["status"] == "failed"
    assert snap["partial"]["rows"]["value"] == ["a", "b"]
    assert any(e["type"] == "node.partial" for e in rs.events)


def _bindable_spec():
    return spec(
        "test.scale",
        "Scale",
        "logic",
        "Multiplies.",
        inputs = [inp("value", "number")],
        outputs = [out("value", "number")],
        params = [P("factor", "int", 2, "Factor", min = 1, max = 10, bindable = True)],
    )


def test_a_bindable_param_adds_an_optional_typed_input():
    found = _bindable_spec()
    port = next(p for p in found["inputs"] if p["id"] == "param:factor")
    assert port == {
        "id": "param:factor",
        "type": "number",
        "label": "Factor",
        "optional": True,
        "param": "factor",
    }


def test_existing_specs_publish_no_new_fields():
    for found in specs_mod.NODE_SPECS:
        assert "requires" not in found and "exclusive_gpu" not in found
        assert not any(p.get("param") for p in found["inputs"])


@run_async
async def test_a_connected_value_replaces_the_param_and_is_clamped(monkeypatch):
    _add_spec(monkeypatch, _bindable_spec())
    seen = []

    async def number(ctx):
        return {"value": ctx.params["value"]}

    async def scale(ctx):
        seen.append(ctx.params["factor"])
        assert "param:factor" not in ctx.inputs
        return {"value": ctx.inputs["value"] * ctx.params["factor"]}

    eng = make_engine(
        {"input.number": number, "test.scale": scale, "output.result": result_handler}
    )
    d = doc(
        [
            node("n", "input.number", value = 3),
            node("f", "input.number", value = 50),
            node("s", "test.scale", factor = 2),
            node("o", "output.result", name = "r"),
        ],
        [
            edge("n", "value", "s", "value"),
            {**edge("f", "value", "s", "param:factor"), "id": "f-value-s-factor"},
            edge("s", "value", "o", "value"),
        ],
    )
    rs = await run_doc(eng, d)
    assert rs.outputs == {"r": 30} and seen == [10]
    d["edges"] = [e for e in d["edges"] if e["target_port"] != "param:factor"]
    d["id"] = "f2"
    rs = await run_doc(eng, d)
    assert rs.outputs == {"r": 6}


def test_combinations_cover_every_pair_and_refuse_a_grid_too_big():
    class Ctx:
        inputs = {}
        params = {"axes": {"kv": ["f16", "q8_0"], "ctx": [4096, 8192, 16384]}, "max_items": 64}

    found = asyncio.run(logic.product(Ctx()))
    assert found["count"] == 6
    assert {"kv": "q8_0", "ctx": 16384} in found["items"]

    class Big:
        inputs = {"axes": {"a": list(range(20)), "b": list(range(20))}}
        params = {"max_items": 256}

    with pytest.raises(FlowNodeError, match = "400 combinations"):
        asyncio.run(logic.product(Big()))
