# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from fastapi import FastAPI
from fastapi.testclient import TestClient

from auth.authentication import get_current_subject
from routes import benchmarks as benchmarks_routes
from storage import benchmark_runs_db as db


def _result(
    variant,
    rep,
    tps,
    warmup = False,
):
    return {
        "variant": variant,
        "rep": rep,
        "warmup": warmup,
        "promptIndex": rep % 5,
        "tps": tps,
        "promptTps": 300.0,
        "promptTokens": 40,
        "genTokens": 256,
        "ttftMs": 120.5,
        "wallMs": 5000.0,
        "clientTps": None,
        "draftN": 100,
        "draftAccepted": 78,
        "loadMs": 8100.0 if rep == 0 else None,
        "at": 1_700_000_000_000 + rep,
    }


def _run(run_id = "run-1", results = None):
    return {
        "id": run_id,
        "kind": "sweep",
        "sweep": "draft",
        "model": "unsloth/Qwen3.5-4B-MTP-GGUF",
        "ggufVariant": "Q4_K_M",
        "kv": "q8_0",
        "context": 32768,
        "config": {
            "sweep": "draft",
            "variants": [{"label": "Speculation off", "load": {}}],
            "repetitions": 3,
        },
        "meta": {"gpu": "AMD Radeon AI PRO R9700", "backend": "vulkan"},
        "base": [{"field": "cache_type_kv", "label": "KV Cache Dtype", "value": "q8_0"}],
        "outcomes": [{"label": "Speculation off", "state": "done"}],
        "results": results
        if results is not None
        else [
            _result("Speculation off", 0, 10.0, warmup = True),
            _result("Speculation off", 1, 30.1),
        ],
        "createdAt": 1_700_000_000_000,
        "finishedAt": None,
    }


def _client():
    app = FastAPI()
    app.dependency_overrides[get_current_subject] = lambda: "unsloth"
    app.include_router(benchmarks_routes.router, prefix = "/api/benchmarks")
    return TestClient(app)


def test_round_trip_keeps_every_measurement_column():
    saved = db.upsert_run(_run())
    assert saved["id"] == "run-1"
    assert [r["tps"] for r in saved["results"]] == [10.0, 30.1]
    assert saved["results"][0]["warmup"] is True
    assert saved["results"][1]["warmup"] is False
    assert saved["results"][0]["loadMs"] == 8100.0
    assert saved["results"][1]["draftAccepted"] == 78
    assert saved["base"][0]["label"] == "KV Cache Dtype"
    assert saved["meta"]["gpu"] == "AMD Radeon AI PRO R9700"


def test_saving_again_replaces_results_instead_of_appending():
    db.upsert_run(_run())
    db.upsert_run(
        _run(
            results = [
                _result("Speculation off", 0, 29.7),
                _result("Speculation off", 1, 30.0),
                _result("Speculation off", 2, 30.3),
            ]
        )
    )
    run = db.get_run("run-1")
    assert [r["tps"] for r in run["results"]] == [29.7, 30.0, 30.3]


def test_list_is_newest_first_without_results_and_counts_measured_runs_only():
    db.upsert_run(_run("older"))
    later = _run("newer")
    later["createdAt"] += 1
    db.upsert_run(later)
    runs = db.list_runs()
    assert [r["id"] for r in runs] == ["newer", "older"]
    assert "results" not in runs[0]
    # One warm-up and one measured run were saved; only the measured one counts.
    assert runs[1]["resultCount"] == 1


def test_list_carries_each_rows_measured_mean():
    db.upsert_run(
        _run(
            results = [
                _result("Speculation off", 0, 5.0, warmup = True),
                _result("Speculation off", 1, 30.0),
                _result("Speculation off", 2, 32.0),
                _result("MTP 3", 1, 42.0),
            ]
        )
    )
    [run] = db.list_runs()
    assert run["resultCount"] == 3
    # The warm-up's 5.0 stays out of the mean.
    assert run["rowMeans"] == {"Speculation off": 31.0, "MTP 3": 42.0}


def test_delete_cascades_to_results():
    db.upsert_run(_run())
    assert db.delete_run("run-1") is True
    assert db.get_run("run-1") is None
    assert db.delete_run("run-1") is False
    conn = db.get_connection()
    try:
        assert conn.execute("SELECT COUNT(*) FROM benchmark_results").fetchone()[0] == 0
    finally:
        conn.close()


def test_routes_round_trip_and_refuse_a_mismatched_id():
    client = _client()
    put = client.put("/api/benchmarks/runs/run-1", json = _run())
    assert put.status_code == 200, put.text
    assert put.json()["results"][1]["tps"] == 30.1

    assert client.put("/api/benchmarks/runs/other", json = _run()).status_code == 400

    listed = client.get("/api/benchmarks/runs").json()["runs"]
    assert [r["id"] for r in listed] == ["run-1"]
    assert listed[0]["resultCount"] == 1

    one = client.get("/api/benchmarks/runs/run-1")
    assert one.status_code == 200
    assert len(one.json()["results"]) == 2

    assert client.delete("/api/benchmarks/runs/run-1").status_code == 204
    assert client.get("/api/benchmarks/runs/run-1").status_code == 404
    assert client.delete("/api/benchmarks/runs/run-1").status_code == 404


def test_unknown_kind_is_rejected_until_that_phase_ships():
    client = _client()
    body = _run()
    body["kind"] = "llama-bench"
    assert client.put("/api/benchmarks/runs/run-1", json = body).status_code == 422
