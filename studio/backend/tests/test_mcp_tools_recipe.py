# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import json

import pytest
from fastapi.responses import JSONResponse

from .mcp_harness import bodies, queries, run_tool

ERRORS = [{"message": "column q needs params", "path": "columns.0"}]
RECIPE = {"columns": [{"name": "q", "column_type": "sampler"}]}
STATUS = {
    "job_id": "job-1",
    "status": "completed",
    "stage": "done",
    "progress": {"done": 10, "total": 10, "percent": 100.0},
    "rows": 10,
    "error": None,
    "artifact_path": "/srv/unsloth/recipes/my-dataset",
    "execution_type": "full",
}
PAYLOADS = {
    ("POST", "/api/data-recipe/validate"): {"valid": False, "errors": ERRORS},
    ("POST", "/api/data-recipe/jobs"): {"job_id": "job-1"},
    ("GET", "/api/data-recipe/jobs/job-1/status"): STATUS,
    ("GET", "/api/data-recipe/jobs/current"): STATUS,
    ("GET", "/api/data-recipe/jobs/job-1/dataset"): {
        "dataset": [{"q": "a"}, {"q": "b"}],
        "total": 10,
        "limit": 2,
        "offset": 4,
    },
}


@pytest.mark.parametrize("mode", [None, "preview", "full"])
def test_validate_checks_and_preview_or_full_starts_a_job(monkeypatch, mode):
    args = {"recipe": RECIPE, **({"mode": mode} if mode else {})}
    result, studio = run_tool(monkeypatch, PAYLOADS, "run_recipe", args)
    started = [{"recipe": RECIPE, "run": {"execution_type": mode}}] if mode else []
    assert result["structuredContent"] == {
        "mode": mode or "validate",
        "valid": None if mode else False,
        "errors": None if mode else ERRORS,
        "job_id": "job-1" if mode else None,
    }
    assert bodies(studio, "/api/data-recipe/jobs") == started
    assert bodies(studio, "/api/data-recipe/validate") == ([] if mode else [{"recipe": RECIPE}])


def test_a_job_reports_its_dataset_by_name_never_its_path(monkeypatch):
    result, studio = run_tool(monkeypatch, PAYLOADS, "get_job", {"kind": "recipe", "id": "job-1"})
    out = result["structuredContent"]
    assert out["status"] == "completed"
    assert out["progress_percent"] == 100.0
    assert out["recipe"] == {
        "stage": "done",
        "rows": 10,
        "dataset": "recipes/my-dataset",
        "total_rows": None,
        "data_rows": None,
    }
    assert "/srv" not in json.dumps(out)
    assert [c[1] for c in studio.state.calls] == ["/api/data-recipe/jobs/job-1/status"]


def test_without_an_id_the_current_job_is_read(monkeypatch):
    result, studio = run_tool(monkeypatch, PAYLOADS, "get_job", {"kind": "recipe"})
    assert result["structuredContent"]["id"] == "job-1"
    assert [c[1] for c in studio.state.calls] == ["/api/data-recipe/jobs/current"]


def test_rows_page_through_the_dataset(monkeypatch):
    args = {"kind": "recipe", "id": "job-1", "rows": 2, "offset": 4}
    result, studio = run_tool(monkeypatch, PAYLOADS, "get_job", args)
    assert result["structuredContent"]["recipe"]["data_rows"] == [{"q": "a"}, {"q": "b"}]
    assert result["structuredContent"]["recipe"]["total_rows"] == 10
    assert queries(studio, "/api/data-recipe/jobs/job-1/dataset") == [{"limit": "2", "offset": "4"}]


def _missing(detail):
    return JSONResponse({"detail": detail}, status_code = 404)


def test_no_current_job_is_an_empty_result(monkeypatch):
    routes = {**PAYLOADS, ("GET", "/api/data-recipe/jobs/current"): _missing("no job")}
    result, _studio = run_tool(monkeypatch, routes, "get_job", {"kind": "recipe"})
    assert result["isError"] is False
    assert result["structuredContent"]["jobs"] == []


def test_an_unknown_recipe_id_is_still_an_error(monkeypatch):
    routes = {**PAYLOADS, ("GET", "/api/data-recipe/jobs/nope/status"): _missing("job not found")}
    result, _studio = run_tool(monkeypatch, routes, "get_job", {"kind": "recipe", "id": "nope"})
    assert result["isError"] is True
    assert "job not found" in result["content"][0]["text"]


# The dataset route answers 404 while a job runs and 422 (naming host paths) when it ended without rows.
@pytest.mark.parametrize(
    "code,detail", [(404, "dataset not ready"), (422, "dataset path missing: /srv/x")]
)
def test_rows_not_written_yet_still_report_the_status(monkeypatch, code, detail):
    routes = {
        **PAYLOADS,
        ("GET", "/api/data-recipe/jobs/job-1/status"): {**STATUS, "status": "running"},
        ("GET", "/api/data-recipe/jobs/job-1/dataset"): JSONResponse(
            {"detail": detail}, status_code = code
        ),
    }
    result, _studio = run_tool(
        monkeypatch, routes, "get_job", {"kind": "recipe", "id": "job-1", "rows": 5}
    )
    assert result["isError"] is False, result
    assert result["structuredContent"]["status"] == "running"
    assert result["structuredContent"]["recipe"]["data_rows"] is None
    assert "/srv/x" not in json.dumps(result)
