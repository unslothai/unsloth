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

    def progress(self, *a, **k):
        pass

    def on_cancel(self, fn):
        self.cancel = fn


def _samples(correct):
    return [
        {
            "doc_id": i,
            "question": r["instruction"],
            "target": r["response"],
            "response": "x",
            "correct": ok,
        }
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
                        "errored": [{"index": 3, "error": "HTTP 500: boom"}],
                    }
                },
            }
        if path == "/api/benchmarks/evals/runs/r1":
            return {"samples": _samples([True, True, False])}
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
    assert spec["local_adapter"] is True
    assert spec["model_providers"] == [{"name": "local", "is_local": True}]
    assert out["score"] == 75.0
    results = out["results"]
    assert (results["passed"], results["scored"], results["base_score"]) == (3, 4, 0.0)
    assert results["run_id"] == "r1" and len(results["preview"]) == 4
    assert (
        results["preview"][2]["passed"] is False and results["preview"][2]["base_passed"] is False
    )
    assert results["preview"][3]["reason"] == "HTTP 500: boom"
    failed = next(rows for name, rows in _uploads(ctx).items() if name.endswith("-failures.jsonl"))
    assert failed == [
        {**ROWS[2], "model_output": "x", "reason": ""},
        {**ROWS[3], "model_output": None, "reason": "HTTP 500: boom"},
    ]
    assert out["failures"]["rows"] == 2 and out["failures"]["preview"][0]["id"] == 2


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
                "metrics": [
                    {"name": "acc_norm,none", "score": 0.3},
                    {"name": "acc,none", "score": 0.25},
                ],
                "samples": _samples([True, False, False, False]),
            }
        if path == "/api/hub/datasets/upload":
            return {"stored_path": "/cache/h.jsonl"}
        raise AssertionError(path)

    ctx = _Ctx(routes, task = "HellaSwag", max_rows = 10)
    out = asyncio.run(benchmark.benchmark(ctx))
    assert sent == [
        {
            "checkpoint_path": "unsloth/Qwen3-0.6B-GGUF",
            "model_source": "local",
            "task": "hellaswag",
            "limit": 10,
        }
    ]
    # The Evals tab's headline metric, acc_norm for HellaSwag, not the first one listed.
    assert out["score"] == 30.0 and out["results"]["metric"] == "acc_norm,none"
    assert out["failures"]["rows"] == 3


def test_flows_reach_the_evals_routes_and_nothing_else_under_benchmarks():
    assert flow_caller.checked_path(httpx.URL("http://x/api/benchmarks/evals/run"), "POST")
    with pytest.raises(RuntimeError):
        flow_caller.checked_path(httpx.URL("http://x/api/benchmarks/lease"), "POST")


def test_a_hub_split_is_streamed_and_cut_to_max_rows(monkeypatch):
    seen = {}

    class _Stream:
        def take(self, n):
            seen["take"] = n
            return iter(ROWS[:n])

    def load_dataset(name, subset, split, token, streaming):
        seen.update(name = name, split = split, streaming = streaming)
        return _Stream()

    monkeypatch.setattr("datasets.load_dataset", load_dataset)
    monkeypatch.setattr("core.training.account_jobs.account_hf_token", lambda token: None)
    ctx = _Ctx(lambda *a: {})
    rows = asyncio.run(
        benchmark._read_rows(ctx, {"kind": "dataset", "source": "hf", "ref": "o/d"}, 2)
    )
    assert rows == ROWS[:2]
    assert seen == {"name": "o/d", "split": "train", "streaming": True, "take": 2}


def test_a_local_file_is_streamed_too(monkeypatch):
    seen = {}

    class _Stream:
        def take(self, n):
            return iter(ROWS[:n])

    def load_dataset(builder, data_files, split, streaming):
        seen.update(builder = builder, data_files = data_files, streaming = streaming)
        return _Stream()

    monkeypatch.setattr("datasets.load_dataset", load_dataset)
    monkeypatch.setattr("core.flows.nodes.data.local_seed_path", lambda ref: "/cache/x.jsonl")
    monkeypatch.setattr("core.training.account_jobs.account_hf_token", lambda token: None)
    ctx = _Ctx(lambda *a: {})
    rows = asyncio.run(
        benchmark._read_rows(ctx, {"kind": "dataset", "source": "local", "ref": "ref:x"}, 3)
    )
    assert rows == ROWS[:3] and seen == {
        "builder": "json",
        "data_files": "/cache/x.jsonl",
        "streaming": True,
    }


def _bench_flow(
    task,
    *,
    via_export = None,
    both = False,
):
    from core.flows.compiler import compile_flow, parse_document

    from ._flows_helpers import doc, edge, node

    nodes = [
        node("data", "data.hf_dataset", repo = "a/b"),
        node("train", "train.finetune"),
        node("bench", "train.benchmark", task = task),
    ]
    edges = [edge("data", "dataset", "train", "dataset")]
    if via_export:
        nodes.append(node("export", "train.export", format = via_export))
        edges += [
            edge("train", "checkpoint", "export", "checkpoint"),
            edge("export", "model", "bench", "model"),
        ]
    if both or not via_export:
        edges.append(edge("train", "checkpoint", "bench", "checkpoint"))
    parsed, errors = parse_document(doc(nodes, edges))
    assert not errors, errors
    return compile_flow(parsed)


def test_an_lm_eval_task_on_a_fine_tune_checkpoint_is_refused_at_plan_time():
    from core.flows.compiler import CompileError

    with pytest.raises(CompileError) as found:
        _bench_flow("GSM8K")
    assert "Export Model" in str(found.value.issues) and "Recipe held-out" in str(
        found.value.issues
    )
    _bench_flow("Recipe held-out")
    _bench_flow("GSM8K", via_export = "gguf")
    with pytest.raises(CompileError, match = "Format to GGUF"):
        _bench_flow("GSM8K", via_export = "merged")
    with pytest.raises(CompileError, match = "not both"):
        _bench_flow("Recipe held-out", via_export = "gguf", both = True)


def test_an_export_wired_into_model_loads_by_export_name_and_its_quant(monkeypatch):
    from types import SimpleNamespace

    asked, loads = {}, []

    async def resolve_by(deadline, caller, **kwargs):
        asked.update(kwargs)
        return SimpleNamespace(
            ref = "ref:exports/flows-run-1-gguf", is_lora = False, gguf_variant = "Q4_K_M"
        )

    async def load_llm(caller, job, **kwargs):
        loads.append(kwargs)

    monkeypatch.setattr("studio_mcp.trained_models.resolve_by", resolve_by)
    monkeypatch.setattr("studio_mcp.loading.load_llm", load_llm)
    monkeypatch.setattr(benchmark, "note_loaded", lambda *a: None)
    ctx = _Ctx(lambda *a: {}, task = "GSM8K")
    ctx.inputs = {
        "model": {
            "kind": "model",
            "model_kind": "chat",
            "repo": "flows-run-1-gguf",
            "variant": "Q4_K_M",
        }
    }
    ctx.resident = {"llm": "old"}
    ctx.set_status = lambda *a, **k: None
    assert asyncio.run(benchmark._load_target(ctx)) == ("flows-run-1-gguf", False)
    assert (
        asked["export"] == "flows-run-1-gguf"
        and asked["variant"] == "Q4_K_M"
        and asked["checkpoint"] is None
    )
    assert loads[0]["model"] == "ref:exports/flows-run-1-gguf" and loads[0]["variant"] == "Q4_K_M"
    assert ctx.resident == {}


def test_the_score_output_is_a_percent_with_two_decimals(monkeypatch):
    async def loaded(ctx):
        return "m", False

    monkeypatch.setattr(benchmark, "_load_target", loaded)
    sent = []
    base = _routes(sent)

    def routes(method, path, params, body):
        found = base(method, path, params, body)
        if path == "/api/benchmarks/evals/run":
            found["details"]["recipe"]["candidate"]["score"] = 0.6066666666666667
        return found

    out = asyncio.run(benchmark.benchmark(_Ctx(routes)))
    assert out["score"] == 60.67 and out["results"]["score"] == 60.67
