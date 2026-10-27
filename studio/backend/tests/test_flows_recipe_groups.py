# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A Data Recipe as a Recipe group in a flow: schema, compile, the import routes and the run."""

import asyncio
import copy
import json
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from auth.authentication import get_current_subject
from core.flows import compiler
from core.flows.compiler import RECIPE_PARAM, CompileError, compile_flow, parse_document
from core.flows.nodes import data
from core.flows.recipe_import import COLUMN_TYPE, GENERATE_TYPE, flow_from_recipe
from core.flows.registry import FlowCancelled, FlowNodeError
from routes import flows as flows_routes
from storage import data_recipes_db

from .test_flows_recipes import _Ctx

BACKEND = Path(__file__).resolve().parents[1]
LEARNING = BACKEND.parent / "frontend" / "src" / "features" / "data-recipes" / "learning-recipes"
SQL = json.loads((LEARNING / "text-to-sql.json").read_text())


def _flow(payload = SQL) -> dict:
    doc, _ = flow_from_recipe(payload, name = "Text to SQL")
    parsed, errors = parse_document(doc)
    assert not errors, errors
    return parsed


def _column(doc: dict, name: str) -> dict:
    return next(
        n for n in doc["nodes"] if n["type"] == COLUMN_TYPE and n["params"]["column"]["name"] == name
    )


def _compiled(doc: dict) -> dict:
    plan = compile_flow(doc)
    generate = next(n for n in plan.nodes.values() if n["type"] == GENERATE_TYPE)
    return generate["params"][RECIPE_PARAM]


def test_the_backend_learning_recipes_are_the_frontend_files():
    copies = BACKEND / "core" / "flows" / "examples" / "learning_recipes"
    assert sorted(p.name for p in copies.glob("*.json")) == sorted(
        p.name for p in LEARNING.glob("*.json")
    )
    for path in LEARNING.glob("*.json"):
        assert json.loads((copies / path.name).read_text()) == json.loads(path.read_text())


def test_a_recipe_group_survives_a_save_and_a_plain_group_saves_as_before():
    doc = _flow()
    again, errors = parse_document(json.loads(json.dumps(doc)))
    assert not errors and again == doc
    group = doc["groups"][0]
    assert group["kind"] == "recipe" and group["config"]["model_configs"][0]["alias"] == "sql-pro"
    plain, _ = parse_document({**doc, "groups": [{"id": "g", "label": "G"}]})
    assert set(plain["groups"][0]) == {"id", "label", "color", "rect"}


def test_references_are_typed_wires_between_columns():
    doc = _flow()
    wires = {
        (e["source"], e["target"])
        for e in doc["edges"]
        if e["source_port"] == "out" and e["target_port"] == "in"
    }
    assert (_column(doc, "sql_prompt")["id"], _column(doc, "sql")["id"]) in wires
    assert (_column(doc, "sql")["id"], _column(doc, "sql-validator")["id"]) in wires
    assert compiler.validate(doc)["ok"]


def test_the_generate_step_runs_the_group_as_edited():
    doc = _flow()
    _column(doc, "sql")["params"]["column"]["prompt"] = "Write SQL for {{ sql_prompt }}"
    _column(doc, "instruction_phrase")["disabled"] = True
    recipe = _compiled(doc)["recipe"]
    names = [c["name"] for c in recipe["columns"]]
    assert "instruction_phrase" not in names
    assert next(c for c in recipe["columns"] if c["name"] == "sql")["prompt"].startswith("Write SQL")
    assert recipe["model_configs"] == SQL["recipe"]["model_configs"]
    assert "mcp_providers" not in recipe and "tool_configs" not in recipe


def test_column_wires_stay_inside_their_recipe():
    doc = _flow()
    sql = _column(doc, "sql")
    doc["edges"].append(
        {
            "id": "leak",
            "source": sql["id"],
            "source_port": "out",
            "target": "preview",
            "target_port": "value",
            "kind": "data",
        }
    )
    with pytest.raises(CompileError, match = "own recipe"):
        compile_flow(doc)
    doc = _flow()
    _column(doc, "sql")["group"] = None
    with pytest.raises(CompileError, match = "inside a Data Recipe group"):
        compile_flow(doc)


def test_a_loop_or_a_lost_group_fails_on_the_generate_step():
    doc = _flow()
    _column(doc, "domain")["params"]["column"] = {
        "column_type": "expression",
        "name": "domain",
        "expr": "{{ sql }}",
    }
    with pytest.raises(CompileError) as caught:
        compile_flow(doc)
    assert any("loop" in i["message"] and i.get("node_id") for i in caught.value.issues)
    doc = _flow()
    doc["groups"][0]["kind"] = None
    with pytest.raises(CompileError):
        compile_flow(doc)


def test_importing_strips_tool_servers_and_keys_and_says_so():
    payload = copy.deepcopy(SQL)
    payload["recipe"]["mcp_providers"] = [{"name": "fs", "provider_type": "stdio", "command": "x"}]
    payload["recipe"]["model_providers"][0]["api_key"] = "sk-live"
    payload["recipe"]["columns"][4]["tool_alias"] = "fs-tools"
    doc, notices = flow_from_recipe(payload, name = "SQL")
    assert len(notices) == 2
    text = json.dumps(doc)
    assert "sk-live" not in text and "stdio" not in text and "fs-tools" not in text


def _routes(job_ids, *, valid = True):
    started = iter(job_ids)
    seen = []

    def routes(method, path, params, body):
        seen.append(path)
        if path == "/api/data-recipe/validate":
            problems = [] if valid else [{"message": "Model alias sql-pro: no endpoint"}]
            return {"valid": valid, "errors": problems}
        if path == "/api/data-recipe/jobs":
            return {"job_id": next(started)}
        if path.endswith("/status"):
            return {"status": "completed", "rows": 2, "artifact_path": "/x/recipes/recipe_sql"}
        if path.endswith("/dataset"):
            return {"dataset": [{"sql_prompt": "Find", "sql": "SELECT 1"}], "total": 1}
        raise AssertionError(path)

    return routes, seen


@pytest.fixture(autouse = True)
def _fast_poll(monkeypatch):
    monkeypatch.setattr(data, "POLL_S", 0)


def test_the_group_runs_as_one_validated_job():
    routes, seen = _routes(["job1"])
    compiled = _compiled(_flow())
    ctx = _Ctx(routes, params = {RECIPE_PARAM: compiled, "rows": 2, "mode": "preview"})
    out = asyncio.run(data.recipe_group(ctx))
    assert seen[:2] == ["/api/data-recipe/validate", "/api/data-recipe/jobs"]
    sent = [c[3] for c in ctx.calls if c[1] == "/api/data-recipe/jobs"][0]
    assert sent["run"] == {"execution_type": "preview", "rows": 2}
    assert [c["name"] for c in sent["recipe"]["columns"]] == [
        c["name"] for c in SQL["recipe"]["columns"]
    ]
    assert out["dataset"]["ref"] == "recipes/recipe_sql/parquet-files"
    assert out["preview"] == [{"sql_prompt": "Find", "sql": "SELECT 1"}]
    assert ctx.hooks, "Stop must reach the recipe job"


def test_an_invalid_recipe_never_starts_a_job():
    routes, seen = _routes([], valid = False)
    ctx = _Ctx(routes, params = {RECIPE_PARAM: _compiled(_flow())})
    with pytest.raises(FlowNodeError, match = "no endpoint"):
        asyncio.run(data.recipe_group(ctx))
    assert "/api/data-recipe/jobs" not in seen


def test_stop_cancels_the_recipe_job():
    cancelled = []

    def routes(method, path, params, body):
        if path.endswith("/cancel"):
            cancelled.append(path)
            return {}
        if path.endswith("/status"):
            return {"status": "cancelled" if cancelled else "running"}
        if path == "/api/data-recipe/validate":
            return {"valid": True}
        if path == "/api/data-recipe/jobs":
            return {"job_id": "job7"}
        raise AssertionError(path)

    class Stoppable(_Ctx):
        def check_cancelled(self):
            if cancelled:
                raise FlowCancelled()

    ctx = Stoppable(routes, params = {RECIPE_PARAM: _compiled(_flow())})

    async def press_stop():
        task = asyncio.create_task(data.recipe_group(ctx))
        while not ctx.hooks:
            await asyncio.sleep(0)
        for hook in ctx.hooks:
            await hook()
        await task

    with pytest.raises(FlowCancelled):
        asyncio.run(press_stop())
    assert cancelled == ["/api/data-recipe/jobs/job7/cancel"]


class _FlowCtx(_Ctx):
    flow_id = "f1"
    node_id = "gen"


def _job_routes(final, *, preview_rows = 3):
    polls = iter(["active", final])

    def routes(method, path, params, body):
        if path == "/api/data-recipe/validate":
            return {"valid": True}
        if path == "/api/data-recipe/jobs":
            return {"job_id": "job9"}
        if path.endswith("/status"):
            state = next(polls)
            return {
                "status": state,
                "rows": 3,
                "progress": {"done": 1, "total": 3, "percent": 33.3},
                "current_column": "sql",
                "error": "llama-server went away" if state == "error" else None,
                "artifact_path": "/x/recipes/recipe_sql" if state == "completed" else None,
            }
        if path.endswith("/dataset"):
            rows = [{"sql_prompt": f"q{i}", "sql": "SELECT 1"} for i in range(preview_rows)]
            return {"dataset": rows[: params["limit"]], "total": preview_rows}
        if path.endswith("/analysis"):
            return {"num_records": preview_rows, "column_statistics": []}
        raise AssertionError(path)

    return routes


def test_each_run_is_kept_as_a_recipe_studio_execution_record(monkeypatch):
    saved = []
    from storage import flows_db

    monkeypatch.setattr(
        flows_db, "save_recipe_run", lambda flow, node, record: saved.append((flow, node, record))
    )
    ctx = _FlowCtx(
        _job_routes("completed", preview_rows = 30),
        params = {RECIPE_PARAM: _compiled(_flow()), "rows": 30, "mode": "preview"},
    )
    out = asyncio.run(data.recipe_group(ctx))
    assert [r["status"] for _, _, r in saved] == ["pending", "active", "active", "completed"]
    assert {(f, n) for f, n, _ in saved} == {("f1", "gen")}
    last = saved[-1][2]
    assert last["recipeId"] == "flow:f1:gen" and last["jobId"] == "job9"
    assert last["kind"] == "preview" and last["rows"] == 3
    assert len(last["dataset"]) == 30 and last["datasetTotal"] == 30
    assert last["analysis"] == {"num_records": 30, "column_statistics": []}
    assert last["artifact_path"] == "/x/recipes/recipe_sql" and last["finishedAt"]
    assert saved[1][2]["progress"]["done"] == 1 and saved[1][2]["current_column"] == "sql"
    ids = [r["lastEventId"] for _, _, r in saved]
    assert ids == sorted(ids) and len({r["id"] for _, _, r in saved}) == 1
    assert len(out["preview"]) == 20


def test_a_failed_run_is_kept_with_its_error(monkeypatch):
    saved = []
    from storage import flows_db

    monkeypatch.setattr(flows_db, "save_recipe_run", lambda f, n, record: saved.append(record))
    ctx = _FlowCtx(_job_routes("error"), params = {RECIPE_PARAM: _compiled(_flow())})
    with pytest.raises(FlowNodeError, match = "went away"):
        asyncio.run(data.recipe_group(ctx))
    assert saved[-1]["status"] == "error" and saved[-1]["error"] == "llama-server went away"
    assert saved[-1]["finishedAt"]


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(flows_routes.router, prefix = "/api/flows")
    app.dependency_overrides[get_current_subject] = lambda: "tester"
    with TestClient(app) as test_client:
        yield test_client


def _save_recipe(recipe_id: str, name: str, payload = SQL, updated = 2) -> None:
    data_recipes_db.upsert_recipe(
        {"id": recipe_id, "name": name, "payload": payload, "createdAt": 1, "updatedAt": updated}
    )


def test_import_a_saved_recipe_or_recipe_json(client):
    _save_recipe("r1", "My SQL")
    body = client.post("/api/flows/import/recipe", json = {"recipe_id": "r1"}).json()
    doc = body["flow"]["doc"]
    assert body["flow"]["name"] == "My SQL"
    assert doc["meta"]["from_recipe"] == {"id": "r1", "name": "My SQL", "updatedAt": 2}
    assert compiler.validate(doc)["ok"]
    loose = client.post("/api/flows/import/recipe", json = {"payload": SQL, "name": "Loose"}).json()
    assert "from_recipe" not in loose["flow"]["doc"]["meta"]
    again = client.post("/api/flows/import/recipe", json = {"recipe_id": "r1"}).json()
    assert again["flow"]["id"] == body["flow"]["id"] and again["report"]["reused"]
    _save_recipe("r1", "My SQL", updated = 3)
    changed = client.post("/api/flows/import/recipe", json = {"recipe_id": "r1"}).json()
    assert changed["flow"]["id"] != body["flow"]["id"]
    assert client.post("/api/flows/import/recipe", json = {"recipe_id": "nope"}).status_code == 404
    assert client.post("/api/flows/import/recipe", json = {}).status_code == 400
    assert data_recipes_db.get_recipe("r1")["payload"] == SQL


def test_bringing_every_recipe_in_twice_makes_one_flow_each(client):
    _save_recipe("r1", "One")
    _save_recipe("r2", "Two")
    _save_recipe("bad", "Broken", payload = {"recipe": {"columns": "nope"}})
    first = client.post("/api/flows/import/recipes").json()
    assert sorted(c["recipe_id"] for c in first["created"]) == ["r1", "r2"]
    assert [f["recipe_id"] for f in first["failed"]] == ["bad"]
    second = client.post("/api/flows/import/recipes").json()
    assert second["created"] == [] and len(second["skipped"]) == 2
    flows = client.get("/api/flows").json()["flows"]
    assert sorted(f["name"] for f in flows) == ["One", "Two"]
    assert len(data_recipes_db.list_recipes()) == 3


def test_recipe_runs_are_listed_per_step_and_go_with_the_flow(client):
    from storage import flows_db

    flow = client.post("/api/flows/import/recipe", json = {"payload": SQL, "name": "S"}).json()
    flow_id = flow["flow"]["id"]
    for i, created in enumerate((5, 9)):
        record = {"id": f"e{i}", "recipeId": f"flow:{flow_id}:gen", "createdAt": created}
        flows_db.save_recipe_run(flow_id, "gen", record)
    flows_db.save_recipe_run(flow_id, "other", {"id": "e9", "createdAt": 1})
    listed = client.get(f"/api/flows/{flow_id}/nodes/gen/recipe-runs").json()["executions"]
    assert [e["id"] for e in listed] == ["e1", "e0"]
    flows_db.save_recipe_run(flow_id, "gen", {"id": "e0", "createdAt": 5, "status": "completed"})
    listed = client.get(f"/api/flows/{flow_id}/nodes/gen/recipe-runs").json()["executions"]
    assert listed[1]["status"] == "completed"
    assert client.get(f"/api/flows/{flow_id}/nodes/bad id/recipe-runs").status_code == 400
    assert client.get("/api/flows/nope/nodes/gen/recipe-runs").status_code == 404
    client.delete(f"/api/flows/{flow_id}")
    assert flows_db.recipe_runs(flow_id, "gen") == []
