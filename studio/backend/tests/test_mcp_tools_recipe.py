# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import asyncio
import json

from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from mcp_server import create_studio_mcp

from .mcp_harness import call_tool, fake_studio, served

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
    ("POST", "/api/data-recipe/validate"): {
        "valid": False,
        "errors": [{"message": "column q needs params", "path": "columns.0"}],
    },
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


def _studio(overrides = None):
    routes = {
        key: (lambda payload: lambda request, body: payload)(value)
        for key, value in PAYLOADS.items()
    }
    routes.update(overrides or {})
    return fake_studio(routes)


def _call(monkeypatch, studio, name, args):
    with TestClient(served(create_studio_mcp(), studio, monkeypatch = monkeypatch)) as http:
        return call_tool(http, name, args)


def _bodies(studio, path):
    return [json.loads(b) for _m, p, _h, b in studio.state.calls if p == path]


def test_validate_returns_the_errors(monkeypatch):
    studio = _studio()
    result = _call(monkeypatch, studio, "run_recipe", {"recipe": RECIPE})
    assert result["structuredContent"] == {
        "mode": "validate",
        "valid": False,
        "errors": [{"message": "column q needs params", "path": "columns.0"}],
        "job_id": None,
    }
    assert _bodies(studio, "/api/data-recipe/validate") == [{"recipe": RECIPE}]
    assert _bodies(studio, "/api/data-recipe/jobs") == []


def test_preview_and_full_start_a_job_with_their_execution_type(monkeypatch):
    for mode in ("preview", "full"):
        studio = _studio()
        result = _call(monkeypatch, studio, "run_recipe", {"recipe": RECIPE, "mode": mode})
        assert result["structuredContent"] == {
            "mode": mode,
            "valid": None,
            "errors": None,
            "job_id": "job-1",
        }
        assert _bodies(studio, "/api/data-recipe/jobs") == [
            {"recipe": RECIPE, "run": {"execution_type": mode}}
        ]
        assert _bodies(studio, "/api/data-recipe/validate") == []


def test_another_running_job_is_a_tool_error(monkeypatch):
    def busy(request, body):
        return JSONResponse({"detail": "A recipe job is already running."}, status_code = 409)

    studio = _studio({("POST", "/api/data-recipe/jobs"): busy})
    result = _call(monkeypatch, studio, "run_recipe", {"recipe": RECIPE, "mode": "full"})
    assert result["isError"] is True
    assert result["content"][0]["text"] == "A recipe job is already running. (HTTP 409)"


def test_a_job_reports_its_dataset_by_name_never_its_path(monkeypatch):
    studio = _studio()
    result = _call(monkeypatch, studio, "get_job", {"kind": "recipe", "id": "job-1"})
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
    studio = _studio()
    result = _call(monkeypatch, studio, "get_job", {"kind": "recipe"})
    assert result["structuredContent"]["id"] == "job-1"
    assert [c[1] for c in studio.state.calls] == ["/api/data-recipe/jobs/current"]


def test_rows_page_through_the_dataset(monkeypatch):
    queries = []

    def page(request, body):
        queries.append(dict(request.query_params))
        return PAYLOADS[("GET", "/api/data-recipe/jobs/job-1/dataset")]

    studio = _studio({("GET", "/api/data-recipe/jobs/job-1/dataset"): page})
    result = _call(
        monkeypatch, studio, "get_job", {"kind": "recipe", "id": "job-1", "rows": 2, "offset": 4}
    )
    assert result["structuredContent"]["recipe"]["data_rows"] == [{"q": "a"}, {"q": "b"}]
    assert result["structuredContent"]["recipe"]["total_rows"] == 10
    assert queries == [{"limit": "2", "offset": "4"}]


def test_rows_are_bounded(monkeypatch):
    studio = _studio()
    for rows in (0, 501):
        result = _call(
            monkeypatch, studio, "get_job", {"kind": "recipe", "id": "job-1", "rows": rows}
        )
        assert result["isError"] is True
    assert studio.state.calls == []


def test_no_current_job_is_a_tool_error(monkeypatch):
    studio = _studio(
        {
            ("GET", "/api/data-recipe/jobs/current"): lambda r, b: JSONResponse(
                {"detail": "no job"}, status_code = 404
            )
        }
    )
    result = _call(monkeypatch, studio, "get_job", {"kind": "recipe"})
    assert result["content"][0]["text"] == "no job (HTTP 404)"


def test_recipe_annotations():
    tools = {t.name: t for t in asyncio.run(create_studio_mcp().list_tools())}
    run = tools["run_recipe"]
    assert run.annotations.readOnlyHint is False
    assert run.annotations.destructiveHint is False
    assert run.annotations.openWorldHint is False
    assert run.output_schema is not None
    assert "recipe" in tools["get_job"].parameters["properties"]["kind"]["enum"]
