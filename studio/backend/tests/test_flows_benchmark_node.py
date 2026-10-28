# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import asyncio
import json

import httpx
import pytest

from core.flows import caller as flow_caller
from core.flows.nodes import benchmark
from core.flows.registry import FlowNodeError
from core.flows.specs import SPECS_BY_TYPE

ROWS = [{"instruction": f"city {i}?", "response": f"answer-{i}", "id": i} for i in range(4)]


class _Ctx:
    def __init__(self, routes, **params):
        self.routes, self.calls = routes, []
        self.inputs = {
            "checkpoint": {"kind": "checkpoint", "name": "run-1"},
            "dataset": {"kind": "dataset", "source": "recipe", "ref": "job1"},
        }
        self.params = {p["id"]: p["default"] for p in SPECS_BY_TYPE["train.benchmark"]["params"]}
        self.params.update(params)
        self.caller = object()

    async def call(self, method, path, json = None, params = None, files = None, **_):
        self.calls.append((method, path, params, json, files))
        return self.routes(method, path, params, json)

    def progress(self, *a, **k):
        pass

    def on_cancel(self, fn):
        self.cancel = fn


def _samples(correct):
    return [
        {"doc_id": i, "question": r["instruction"], "target": r["response"], "response": "x", "correct": ok}
        for i, (r, ok) in enumerate(zip(ROWS, correct))
    ]


def _routes(sent):
    def routes(method, path, params, body):
        if path.endswith("/jobs/job1/dataset"):
            return {"dataset": ROWS, "total": len(ROWS)}
        if path == "/api/hub/datasets/upload":
            return {"stored_path": f"/cache/{len(sent)}.jsonl"}
        if path == "/api/benchmarks/evals/run":
            sent.append(body)
            return {
                "success": True,
                "details": {
                    "recipe": {
                        "candidate": {"total": 4, "scored": 4, "passed": 3, "score": 0.75},
                        "baseline": {"total": 4, "scored": 4, "passed": 0, "score": 0.0},
                        "run_ids": {"candidate": "r1", "baseline": "r1_baseline"},
                    }
                },
            }
        if path == "/api/benchmarks/evals/runs/r1":
            return {"samples": _samples([True, True, False, True])}
        if path == "/api/benchmarks/evals/runs/r1_baseline":
            return {"samples": _samples([False] * 4)}
        raise AssertionError(path)

    return routes


def _uploads(ctx):
    return {
        c[4]["file"][0]: [json.loads(line) for line in c[4]["file"][1].decode().splitlines()]
        for c in ctx.calls
        if c[1] == "/api/hub/datasets/upload"
    }


@pytest.fixture(autouse = True)
def _no_storage(monkeypatch):
    monkeypatch.setattr("hub.utils.host_paths.cache_reference", lambda p: f"cache:{p}")


def test_adapter_is_scored_against_itself_off_and_failures_keep_the_row(monkeypatch):
    async def loaded(ctx):
        return "run-1", True

    monkeypatch.setattr(benchmark, "_load_target", loaded)
    sent = []
    ctx = _Ctx(_routes(sent))
    out = asyncio.run(benchmark.benchmark(ctx))
    spec = sent[0]["recipe"]
    assert sent[0]["task"] == "recipe_heldout"
    assert spec["prompt_column"] == "instruction" and spec["reference_column"] == "response"
    assert spec["baseline"] == {"mode": "adapter_off"} and spec["scorer"] == "reference"
    assert spec["model_providers"] == [{"name": "local", "is_local": True}]
    assert out["score"] == 0.75
    results = out["results"]
    assert (results["passed"], results["scored"], results["base_score"]) == (3, 4, 0.0)
    assert results["run_id"] == "r1" and len(results["preview"]) == 4
    assert results["preview"][2]["passed"] is False and results["preview"][2]["base_passed"] is False
    failed = next(rows for name, rows in _uploads(ctx).items() if name.endswith("-failures.jsonl"))
    assert failed == [{**ROWS[2], "model_output": "x", "reason": ""}]
    assert out["failures"]["rows"] == 1 and out["failures"]["preview"][0]["id"] == 2


def test_no_adapter_means_no_baseline_and_judge_scores_without_a_reference(monkeypatch):
    async def loaded(ctx):
        return "unsloth/Qwen3-0.6B", False

    monkeypatch.setattr(benchmark, "_load_target", loaded)
    sent = []
    ctx = _Ctx(_routes(sent), scorer = "LLM judge", reference_column = "")
    ctx.inputs["dataset"] = {"kind": "dataset", "source": "recipe", "ref": "job1"}
    asyncio.run(benchmark.benchmark(ctx))
    spec = sent[0]["recipe"]
    assert spec["baseline"] == {"mode": "none"}
    assert spec["scorer"] == "judge" and spec["judge_alias"] == "candidate"


def test_reference_match_needs_a_reference_column(monkeypatch):
    async def loaded(ctx):
        return "m", False

    monkeypatch.setattr(benchmark, "_load_target", loaded)
    ctx = _Ctx(lambda m, p, q, b: {"dataset": [{"prompt": "hi"}], "total": 1})
    with pytest.raises(FlowNodeError, match = "reference column"):
        asyncio.run(benchmark.benchmark(ctx))


def test_lm_eval_refuses_an_adapter_and_scores_a_gguf_with_a_limit(monkeypatch):
    async def lora(ctx):
        return "run-1", True

    monkeypatch.setattr(benchmark, "_load_target", lora)
    ctx = _Ctx(lambda *a: {}, task = "HellaSwag")
    with pytest.raises(FlowNodeError, match = "GGUF"):
        asyncio.run(benchmark.benchmark(ctx))

    async def gguf(ctx):
        return "unsloth/Qwen3-0.6B-GGUF", False

    async def resident(caller):
        return {}, {"is_gguf": True}

    monkeypatch.setattr(benchmark, "_load_target", gguf)
    monkeypatch.setattr("studio_mcp.loading._resident", resident)
    sent = []

    def routes(method, path, params, body):
        if path == "/api/benchmarks/evals/run":
            sent.append(body)
            return {"success": True, "details": {"run_id": "h1"}}
        if path == "/api/benchmarks/evals/runs/h1":
            return {
                "metrics": [{"name": "acc_norm,none", "score": 0.3}, {"name": "acc,none", "score": 0.25}],
                "samples": _samples([True, False, False, False]),
            }
        if path == "/api/hub/datasets/upload":
            return {"stored_path": "/cache/h.jsonl"}
        raise AssertionError(path)

    ctx = _Ctx(routes, task = "HellaSwag", max_rows = 10)
    out = asyncio.run(benchmark.benchmark(ctx))
    assert sent == [
        {"checkpoint_path": "unsloth/Qwen3-0.6B-GGUF", "model_source": "local", "task": "hellaswag", "limit": 10}
    ]
    # The Evals tab's headline metric, acc_norm for HellaSwag, not the first one listed.
    assert out["score"] == 0.3 and out["results"]["metric"] == "acc_norm,none"
    assert out["failures"]["rows"] == 3


def test_flows_reach_the_evals_routes_and_nothing_else_under_benchmarks():
    assert flow_caller.checked_path(httpx.URL("http://x/api/benchmarks/evals/run"), "POST")
    with pytest.raises(RuntimeError):
        flow_caller.checked_path(httpx.URL("http://x/api/benchmarks/lease"), "POST")
