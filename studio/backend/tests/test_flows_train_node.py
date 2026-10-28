# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import asyncio
import json

import pytest

from core.flows.nodes import training
from core.flows.registry import FlowNodeError
from core.flows.specs import SPECS_BY_TYPE


class _Ctx:
    def __init__(self, routes, **params):
        self.routes, self.calls, self.partials = routes, [], []
        self.inputs = {"dataset": {"kind": "dataset", "source": "recipe", "ref": "job1"}}
        self.params = {p["id"]: p["default"] for p in SPECS_BY_TYPE["train.finetune"]["params"]}
        self.params.update(params)
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

    def progress(self, *a, **k):
        pass

    def log(self, *a, **k):
        pass

    def partial(self, port, value, **k):
        self.partials.append((port, value))


ROWS = [{"q": f"q{i}", "a": f"a{i}"} for i in range(20)]


def _recipe_routes(started):
    def routes(method, path, params, body):
        if path.endswith("/jobs/job1/dataset"):
            off = params["offset"]
            return {"dataset": ROWS[off : off + 500], "total": len(ROWS)}
        if path == "/api/hub/datasets/upload":
            return {"stored_path": f"/cache/{len(started)}.jsonl"}
        if path == "/api/train/start":
            started.append(body)
            raise RuntimeError("started")
        raise AssertionError(path)

    return routes


def _uploads(ctx):
    return {
        c[4]["file"][0]: [json.loads(line) for line in c[4]["file"][1].decode().splitlines()]
        for c in ctx.calls
        if c[1] == "/api/hub/datasets/upload"
    }


def test_holdout_rows_never_reach_training_and_match_the_recipe_card_split(monkeypatch):
    monkeypatch.setattr("hub.utils.host_paths.cache_reference", lambda p: f"cache:{p}")
    started = []
    ctx = _Ctx(_recipe_routes(started), holdout_percent = 10, prompt_column = "q", response_column = "a")
    with pytest.raises(RuntimeError, match = "started"):
        asyncio.run(training.finetune(ctx))
    files = _uploads(ctx)
    train = next(rows for name, rows in files.items() if name.endswith("-train.jsonl"))
    held = next(rows for name, rows in files.items() if name.endswith("-holdout.jsonl"))
    assert len(train) == 18 and len(held) == 2
    assert not {r["q"] for r in train} & {r["q"] for r in held}
    ids = training.holdout_row_ids(20, 10, 3407)
    assert [r["q"] for r in held] == [f"q{i}" for i in ids]
    assert ids == training.holdout_row_ids(20, 10, 3407)
    body = started[0]
    assert body["custom_format_mapping"] == {"q": "user", "a": "assistant"}
    assert len(body["local_datasets"]) == 1 and "hf_dataset" not in body


def test_no_holdout_keeps_the_old_path(monkeypatch):
    monkeypatch.setattr("hub.utils.host_paths.cache_reference", lambda p: f"cache:{p}")
    started = []
    ctx = _Ctx(_recipe_routes(started))
    with pytest.raises(RuntimeError):
        asyncio.run(training.finetune(ctx))
    assert len(_uploads(ctx)) == 1 and "custom_format_mapping" not in started[0]


def test_one_column_alone_is_refused():
    ctx = _Ctx(_recipe_routes([]), prompt_column = "q")
    with pytest.raises(FlowNodeError, match = "both"):
        asyncio.run(training.finetune(ctx))


def test_holdout_refuses_a_dataset_too_small_to_split(monkeypatch):
    monkeypatch.setattr(training, "holdout_row_ids", lambda *a: [])
    ctx = _Ctx(_recipe_routes([]), holdout_percent = 10)
    with pytest.raises(FlowNodeError, match = "too few"):
        asyncio.run(training.finetune(ctx))


def test_holdout_row_ids_keep_at_least_one_row_and_never_all():
    assert training.holdout_row_ids(1, 50, 1) == []
    assert len(training.holdout_row_ids(3, 1, 1)) == 1
    assert len(training.holdout_row_ids(2, 50, 1)) == 1
    assert training.holdout_row_ids(100, 0, 1) == []


def test_loss_history_is_downsampled_for_the_node():
    assert training.downsample(list(range(10))) == list(range(10))
    small = training.downsample(list(range(1000)), 50)
    assert len(small) == 50 and small[0] == 0 and small[-1] == 999


def test_live_metrics_stream_to_the_node_as_a_partial(monkeypatch):
    import studio_mcp.tools.jobs as jobs

    async def run_folder(caller, out):
        return "run-1", None

    monkeypatch.setattr(jobs, "_run_folder", run_folder)
    monkeypatch.setattr(training, "POLL_S", 0)
    ctx = _Ctx(None)
    ctx.inputs = {"dataset": {"kind": "dataset", "source": "hf", "ref": "org/ds"}}
    ctx.on_cancel = lambda fn: None
    ctx.check_cancelled = lambda: None
    polls = iter(["training", "completed"])

    def routes(method, path, params, body):
        if path == "/api/train/start":
            return {"job_id": "j1"}
        if path == "/api/train/status":
            phase = next(polls)
            return {
                "job_id": "j1",
                "training_phase": phase,
                "phase": phase,
                "details": {"step": 2, "total_steps": 4, "loss": 1.5, "output_dir": "/x/run-1"},
            }
        if path == "/api/train/metrics":
            return {"loss_history": [2.0, 1.5]}
        raise AssertionError(path)

    ctx.routes = routes
    out = asyncio.run(training.finetune(ctx))
    assert ctx.partials and ctx.partials[0][0] == "metrics"
    assert ctx.partials[0][1]["loss_history"] == [2.0, 1.5]
    assert out["metrics"]["loss_history"] == [2.0, 1.5]
    assert out["holdout"] is None
    assert out["checkpoint"]["name"] == "run-1" and "output_dir" not in out["checkpoint"]


def test_holdout_splits_only_the_sliced_rows(monkeypatch):
    monkeypatch.setattr("hub.utils.host_paths.cache_reference", lambda p: f"cache:{p}")
    started = []
    ctx = _Ctx(_recipe_routes(started), holdout_percent = 10, dataset_slice_end = 9)
    with pytest.raises(RuntimeError, match = "started"):
        asyncio.run(training.finetune(ctx))
    files = _uploads(ctx)
    train = next(rows for name, rows in files.items() if name.endswith("-train.jsonl"))
    held = next(rows for name, rows in files.items() if name.endswith("-holdout.jsonl"))
    assert len(train) == 9 and len(held) == 1
    assert {r["q"] for r in train + held} == {f"q{i}" for i in range(10)}
    assert "dataset_slice_end" not in started[0]


def test_holdout_reads_hub_datasets_with_the_requests_token(monkeypatch):
    seen = {}

    def load_rows(dataset, token, end):
        seen["token"] = token
        return ROWS

    monkeypatch.setattr(training, "_load_rows", load_rows)
    monkeypatch.setattr("core.training.account_jobs.managed_account", lambda: False)
    ctx = _Ctx(_recipe_routes([]), holdout_percent = 10)
    ctx.inputs = {"dataset": {"kind": "dataset", "source": "hf", "ref": "org/gated"}}
    ctx.caller = type("C", (), {"hf_token": "hf_req"})()
    with pytest.raises(RuntimeError, match = "started"):
        asyncio.run(training.finetune(ctx))
    assert seen["token"] == "hf_req"


def test_one_column_for_both_roles_is_refused():
    ctx = _Ctx(_recipe_routes([]), prompt_column = "q", response_column = "q")
    with pytest.raises(FlowNodeError, match = "different columns"):
        asyncio.run(training.finetune(ctx))


def test_polls_read_the_loss_from_the_status_payload():
    assert training.status_losses({"metric_history": {"loss": [2.0, 1.0]}}) == [2.0, 1.0]
    assert training.status_losses({"metric_history": None}) == []
    assert training.status_losses(None) == []


def test_hub_rows_with_timestamps_still_upload(monkeypatch):
    import datetime

    import datasets

    rows = [
        {"q": f"q{i}", "a": f"a{i}", "at": datetime.datetime(2026, 1, 1, i), "x": float("nan")}
        for i in range(10)
    ]
    monkeypatch.setattr(datasets, "load_dataset", lambda *a, **k: datasets.Dataset.from_list(rows))
    loaded = training._load_rows({"source": "hf", "ref": "org/ds"}, None, None)
    assert loaded[0]["at"] == "2026-01-01 00:00:00" and loaded[0]["x"] is None
    json.dumps(loaded, allow_nan = False)
