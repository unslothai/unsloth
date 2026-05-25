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
