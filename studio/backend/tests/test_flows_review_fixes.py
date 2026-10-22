# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import asyncio

from core.flows import package
from core.flows.engine import param_values
from core.flows.nodes import training
from core.flows.specs import SPECS_BY_TYPE
from core.flows.story import node_value

from ._flows_helpers import doc, edge, node, run_async
from .test_flows_engine import BASE, _engine, _run, _ret, _status


def test_numeric_params_are_clamped_to_the_catalog():
    spec = SPECS_BY_TYPE["audio.concat"]
    assert param_values(spec, {"params": {"gap_ms": 10**9}})["gap_ms"] == 10_000
    assert param_values(spec, {"params": {"gap_ms": -5}})["gap_ms"] == 0


def test_latest_run_output_beats_the_baked_value():
    d = {"baked": {"g": {"story": "packaged"}}}
    assert node_value(d, {"g": {"story": "rerolled"}}, "g", "story") == "rerolled"
    assert node_value(d, {}, "g", "story") == "packaged"


def test_imported_media_urls_that_leave_studio_are_dropped():
    clean = package.scrub(
        {
            "a": {
                "kind": "image",
                "ref": "sha256:" + "0" * 64,
                "url": "https://evil.example/x.png",
            },
            "b": {"kind": "image", "ref": "image:1", "url": "/api/inference/images/gallery/1/file"},
            "c": {"kind": "image", "ref": "image:2", "url": "//evil.example/x"},
        }
    )
    assert "url" not in clean["a"] and "url" not in clean["c"]
    assert clean["b"]["url"] == "/api/inference/images/gallery/1/file"


@run_async
async def test_a_for_each_failure_in_an_early_iteration_fails_the_run():
    calls = []

    async def flaky(ctx):
        calls.append(ctx.inputs.get("values"))
        if len(calls) == 1:
            raise RuntimeError("boom")
        return {"text": "ok"}

    eng = _engine(
        {**BASE, "input.list": lambda ctx: _ret({"items": ["a", "b"]}), "text.template": flaky}
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
    rs = await _run(eng, d)
    assert len(calls) == 2
    assert rs.status == "failed"


@run_async
async def test_ignore_baked_recomputes_shipped_outputs():
    ran = []

    async def text(ctx):
        ran.append(ctx.node_id)
        return {"text": "fresh"}

    eng = _engine({**BASE, "input.text": text})
    d = doc([node("a", "input.text", value = "x")], baked = {"a": {"text": "shipped"}})
    rs = await _run(eng, d)
    assert ran == [] and _status(rs, "a") == "cached"
    rs = await _run(eng, d, ignore_baked = True)
    assert ran == ["a"] and rs.nodes["a"].outputs == {"text": "fresh"}


class _Ctx:
    def __init__(self, routes):
        self.routes, self.calls, self.hooks = routes, [], []
        self.inputs = {"dataset": {"kind": "dataset", "source": "recipe", "ref": "job1"}}
        self.params = {p["id"]: p["default"] for p in SPECS_BY_TYPE["train.finetune"]["params"]}
        self.seed, self.caller = 1, object()

    async def call(
        self,
        method,
        path,
        json = None,
        params = None,
        files = None,
        **_,
    ):
        self.calls.append((method, path, params, json, files))
        return self.routes(method, path, params, json)

    def on_cancel(self, fn):
        self.hooks.append(fn)


def test_finetune_trains_on_a_data_recipe_dataset(monkeypatch):
    rows = [{"instruction": f"q{i}", "output": f"a{i}"} for i in range(3)]

    def routes(method, path, params, body):
        if path.endswith("/jobs/job1/dataset"):
            off = params["offset"]
            return {"dataset": rows[off : off + 2] if off < 3 else [], "total": 3}
        if path == "/api/hub/datasets/upload":
            return {"stored_path": "/cache/recipe-job1.jsonl"}
        if path == "/api/train/start":
            raise RuntimeError(f"started with {body['local_datasets']}")
        raise AssertionError(path)

    monkeypatch.setattr("hub.utils.host_paths.cache_reference", lambda p: "cache:recipe-job1.jsonl")
    ctx = _Ctx(routes)
    try:
        asyncio.run(training.finetune(ctx))
    except RuntimeError as exc:
        assert "cache:recipe-job1.jsonl" in str(exc)
    else:
        raise AssertionError("training never started")
    uploaded = [c for c in ctx.calls if c[1] == "/api/hub/datasets/upload"][0][4]["file"][1]
    assert uploaded.decode().count("\n") == 2
