# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import time

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


@pytest.fixture()
def db(tmp_path, monkeypatch):
    from storage import studio_db
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    monkeypatch.setattr(studio_db, "_schema_ready", False)
    return studio_db


def _client(monkeypatch):
    import eval.jobs as jobs
    from eval.runner import EvalSummary
    from auth.authentication import get_current_subject
    import routes.eval as eval_routes

    def fake_run(req, *, on_result, should_cancel):
        on_result(0, 1.0, "p", "i", "r", None, None)
        return EvalSummary(status="completed", num_scored=1, avg_score=1.0)

    mgr = jobs.EvalJobManager(run_fn=fake_run)
    monkeypatch.setattr(eval_routes, "get_eval_manager", lambda: mgr)

    app = FastAPI()
    app.dependency_overrides[get_current_subject] = lambda: "tester"
    app.include_router(eval_routes.router, prefix="/api/eval")
    return TestClient(app), mgr


def test_list_metrics(db, monkeypatch):
    client, _ = _client(monkeypatch)
    r = client.get("/api/eval/metrics")
    assert r.status_code == 200
    names = {m["name"] for m in r.json()["metrics"]}
    assert {"exact_match", "text_similarity", "json_document"} <= names


def test_start_then_run_appears_in_history(db, monkeypatch):
    client, mgr = _client(monkeypatch)
    body = {
        "model_identifier": "m",
        "dataset": {"is_local": True, "path": "d.jsonl", "split": "train"},
        "input_column": "q", "reference_column": "a", "metric_name": "exact_match",
        "limit": 1,
    }
    r = client.post("/api/eval/start", json=body)
    assert r.status_code == 200
    run_id = r.json()["run_id"]

    end = time.time() + 5
    while time.time() < end and db.get_eval_run(run_id)["status"] == "running":
        time.sleep(0.02)

    runs = client.get("/api/eval/runs").json()
    assert any(run["id"] == run_id for run in runs["runs"])
    detail = client.get(f"/api/eval/runs/{run_id}").json()
    assert detail["run"]["avg_score"] == 1.0
    assert detail["total_results"] == 1


def test_unknown_run_404(db, monkeypatch):
    client, _ = _client(monkeypatch)
    assert client.get("/api/eval/runs/does-not-exist").status_code == 404


def _client_with_run(monkeypatch, run_fn):
    import eval.jobs as jobs
    from auth.authentication import get_current_subject
    import routes.eval as eval_routes

    mgr = jobs.EvalJobManager(run_fn=run_fn)
    monkeypatch.setattr(eval_routes, "get_eval_manager", lambda: mgr)
    app = FastAPI()
    app.dependency_overrides[get_current_subject] = lambda: "tester"
    app.include_router(eval_routes.router, prefix="/api/eval")
    return TestClient(app), mgr


_START_BODY = {
    "model_identifier": "m",
    "dataset": {"is_local": True, "path": "d.jsonl", "split": "train"},
    "input_column": "q", "reference_column": "a", "metric_name": "exact_match",
    "limit": 1,
}


def test_concurrent_start_returns_409(db, monkeypatch):
    from eval.runner import EvalSummary

    def slow_run(req, *, on_result, should_cancel):
        while not should_cancel():
            time.sleep(0.02)
        return EvalSummary(status="cancelled", num_scored=0, avg_score=0.0)

    client, mgr = _client_with_run(monkeypatch, slow_run)
    first = client.post("/api/eval/start", json=_START_BODY)
    assert first.status_code == 200
    run_id = first.json()["run_id"]
    busy = client.post("/api/eval/start", json=_START_BODY)
    assert busy.status_code == 409
    mgr.cancel(run_id)  # let the slow run finish


def test_cancel_active_run(db, monkeypatch):
    from eval.runner import EvalSummary

    def slow_run(req, *, on_result, should_cancel):
        while not should_cancel():
            time.sleep(0.02)
        return EvalSummary(status="cancelled", num_scored=0, avg_score=0.0)

    client, _ = _client_with_run(monkeypatch, slow_run)
    run_id = client.post("/api/eval/start", json=_START_BODY).json()["run_id"]
    r = client.post(f"/api/eval/cancel/{run_id}")
    assert r.status_code == 200 and r.json()["cancelled"] is True


def test_cancel_unknown_run_404(db, monkeypatch):
    client, _ = _client(monkeypatch)
    assert client.post("/api/eval/cancel/nope").status_code == 404


def test_start_unknown_metric_returns_400(db, monkeypatch):
    client, _ = _client(monkeypatch)
    body = {**_START_BODY, "metric_name": "not_a_metric"}
    assert client.post("/api/eval/start", json=body).status_code == 400


def test_start_zero_limit_returns_400(db, monkeypatch):
    client, _ = _client(monkeypatch)
    body = {**_START_BODY, "limit": 0}
    assert client.post("/api/eval/start", json=body).status_code == 400
