# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Recipe Studio parity for Flows recipe groups: seeds, tool profiles, recipe JSON in and out, run settings,
and the sharing rule (an exported flow or recipe never carries keys, local tool servers or upload paths)."""

import asyncio
import copy
import io
import json
import zipfile

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from auth.authentication import authenticated_via_api_key, get_current_subject
from core.flows import package
from core.flows.compiler import RECIPE_PARAM
from core.flows.nodes import data
from core.flows.recipe_compile import compile_recipe, recipe_graph
from core.flows.recipe_import import flow_from_recipe
from core.flows.registry import FlowNodeError
from routes import flows as flows_routes

from .test_flows_recipe_groups import LEARNING, SQL, _column, _compiled, _flow
from .test_flows_recipes import _Ctx

HTTP = {
    "name": "docs",
    "provider_type": "streamable_http",
    "endpoint": "https://user:pw@mcp.example/mcp?api_key=S3",
    "api_key": "mcp-secret",
    "headers": {"Authorization": "Bearer abc"},
}
STDIO = {
    "name": "fs",
    "provider_type": "stdio",
    "command": "npx",
    "args": ["-y", "server-fs", "/home/me"],
    "env": {"TOKEN": "t0"},
}


def _with_tools() -> dict:
    payload = copy.deepcopy(SQL)
    recipe = payload["recipe"]
    recipe["mcp_providers"] = [HTTP, STDIO]
    recipe["tool_configs"] = [
        {"tool_alias": "web", "providers": ["docs"], "max_tool_call_turns": 3},
        {"tool_alias": "files", "providers": ["fs"], "max_tool_call_turns": 5},
    ]
    recipe["model_providers"][0]["api_key"] = "sk-live-0123456789abcdefgh"
    recipe["columns"][4]["tool_alias"] = "web"
    recipe["columns"][5]["tool_alias"] = "files"
    payload["ui"] = {
        "nodes": [
            {"id": "web", "x": 0, "y": 0, "node_type": "tool_config", "tools_by_provider": {"docs": ["search"]}},
            {"id": "files", "x": 0, "y": 0, "node_type": "tool_config", "tools_by_provider": {"fs": ["read"]}},
            {"id": "sql", "x": 5, "y": 5},
        ]
    }
    return payload


def _routes(seen_jobs):
    def routes(method, path, params, body):
        if path == "/api/data-recipe/validate":
            return {"valid": True}
        if path == "/api/data-recipe/jobs":
            seen_jobs.append(body)
            return {"job_id": "j1"}
        if path.endswith("/status"):
            return {"status": "completed", "rows": 1, "artifact_path": "/x/recipes/r"}
        if path.endswith("/dataset"):
            return {"dataset": [{"a": 1}]}
        raise AssertionError(path)

    return routes


def _run(compiled, **params):
    jobs: list = []
    ctx = _Ctx(_routes(jobs), params = {RECIPE_PARAM: compiled, **params})
    asyncio.run(data.recipe_group(ctx))
    return jobs[0]


@pytest.fixture(autouse = True)
def _fast_poll(monkeypatch):
    monkeypatch.setattr(data, "POLL_S", 0)


# Seeds


@pytest.mark.parametrize(
    "name, seed_type",
    [
        ("github-support-bot.json", "github_repo"),
        ("books-and-documents.json", "unstructured"),
        ("instruction-from-answer.json", "hf"),
    ],
)
def test_every_seed_kind_reaches_the_job_as_saved(name, seed_type):
    payload = json.loads((LEARNING / name).read_text())
    doc = _flow(payload)
    group = doc["groups"][0]
    assert group["config"]["seed_config"] == payload["recipe"]["seed_config"]
    assert group["ui"].get("seed_source_type", seed_type) == seed_type
    sent = _run(_compiled(doc), mode = "preview", rows = 2)
    assert sent["recipe"]["seed_config"] == payload["recipe"]["seed_config"]
    assert sent["recipe"]["seed_config"]["source"]["seed_type"] == seed_type


def test_an_uploaded_csv_seed_and_its_preview_rows_survive_the_group():
    payload = copy.deepcopy(SQL)
    payload["recipe"]["seed_config"] = {
        "source": {"seed_type": "local", "path": "/srv/uploads/seed.csv"},
        "sampling_strategy": "shuffle",
        "selection_strategy": None,
    }
    payload["ui"] = {
        "seed_source_type": "local",
        "local_file_name": "seed.csv",
        "seed_columns": ["topic"],
        "seed_preview_rows": [{"topic": "joins"}],
        "seed_total_rows": 40,
    }
    doc = _flow(payload)
    ui = doc["groups"][0]["ui"]
    assert ui["seed_preview_rows"] == [{"topic": "joins"}] and ui["local_file_name"] == "seed.csv"
    sent = _run(_compiled(doc), mode = "full", rows = 3)
    assert sent["recipe"]["seed_config"]["source"] == {"seed_type": "local", "path": "/srv/uploads/seed.csv"}


# Tool profiles


def test_tool_profiles_ride_import_compile_and_run():
    doc = _flow(_with_tools())
    group = doc["groups"][0]
    assert [p["name"] for p in group["config"]["mcp_providers"]] == ["docs", "fs"]
    assert [n["id"] for n in group["ui"]["nodes"]] == ["web", "files"]
    for node in (n for n in doc["nodes"] if n["type"] == "recipe.column"):
        column = node["params"]["column"]
        if column["name"] == SQL["recipe"]["columns"][4]["name"]:
            assert column["tool_alias"] == "web"
    compiled = _compiled(doc)
    assert compiled["recipe"]["tool_configs"][0]["tool_alias"] == "web"
    assert compiled["recipe"]["model_providers"][0]["api_key"].startswith("sk-live")


def test_an_http_tool_server_runs_and_a_stdio_one_is_refused_before_any_call():
    payload = _with_tools()
    payload["recipe"]["mcp_providers"] = [HTTP]
    payload["recipe"]["tool_configs"] = payload["recipe"]["tool_configs"][:1]
    del payload["recipe"]["columns"][5]["tool_alias"]
    sent = _run(_compiled(_flow(payload)), mode = "preview")
    assert sent["recipe"]["mcp_providers"][0]["api_key"] == "mcp-secret"
    assert sent["recipe"]["tool_configs"][0]["providers"] == ["docs"]
    assert sent["recipe"]["model_providers"][0]["api_key"].startswith("sk-live")
    jobs: list = []
    ctx = _Ctx(_routes(jobs), params = {RECIPE_PARAM: _compiled(_flow(_with_tools()))})
    with pytest.raises(FlowNodeError, match = "fs is a local"):
        asyncio.run(data.recipe_group(ctx))
    assert ctx.calls == []


# Sharing rule


def _assert_shared(text: str) -> None:
    for secret in ("sk-live", "mcp-secret", "Bearer abc", "user:pw", "S3", "npx", "server-fs", "t0"):
        assert secret not in text, secret


def test_exporting_a_flow_drops_keys_stdio_servers_and_upload_paths():
    payload = _with_tools()
    payload["recipe"]["seed_config"] = {
        "source": {"seed_type": "github_repo", "repos": ["a/b"], "token": "ghp_" + "x" * 30},
    }
    doc = _flow(payload)

    async def no_media(_):
        raise AssertionError("no media")

    raw, _, _ = asyncio.run(package.export_flow(doc, "json", no_media))
    text = raw.decode()
    _assert_shared(text)
    assert "ghp_" not in text
    out = json.loads(text)
    config = out["groups"][0]["config"]
    assert [p["name"] for p in config["mcp_providers"]] == ["docs"]
    assert [t["tool_alias"] for t in config["tool_configs"]] == ["web"]
    assert out["groups"][0]["ui"]["nodes"][1]["tools_by_provider"] == {}
    aliases = [n["params"]["column"].get("tool_alias") for n in out["nodes"] if n["type"] == "recipe.column"]
    assert "files" not in aliases and "web" in aliases
    # The source flow on this computer still has everything.
    assert "npx" in json.dumps(doc)


def test_a_flow_package_from_elsewhere_cannot_bring_a_stdio_server():
    doc = _flow(_with_tools())
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        zf.writestr("flow.json", json.dumps(doc))
    imported, _, _ = package.read_package(buffer.getvalue())
    _assert_shared(json.dumps(imported))


# Recipe JSON in and out


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(flows_routes.router, prefix = "/api/flows")
    app.dependency_overrides[get_current_subject] = lambda: "tester"
    app.dependency_overrides[authenticated_via_api_key] = lambda: False
    with TestClient(app) as test_client:
        yield test_client


def test_recipe_json_round_trips_out_of_flows_and_back(client):
    doc = _flow()
    _column(doc, "sql")["params"]["column"]["prompt"] = "Edited in Flows {{ sql_prompt }}"
    out = client.post("/api/flows/export/recipe", json = {"doc": doc, "group_id": "recipe"})
    assert out.status_code == 200
    exported = out.json()
    assert set(exported) == {"recipe", "run", "ui"}
    expected = compile_recipe(recipe_graph(doc, "recipe"))["recipe"]
    for provider in expected["model_providers"]:
        provider.pop("extra_headers", None)  # headers can carry credentials; shared copies drop them
    assert exported["recipe"] == expected
    back = client.post("/api/flows/import/recipe", json = {"payload": exported, "name": "Back"}).json()
    again = compile_recipe(recipe_graph(back["flow"]["doc"], "recipe"))
    assert again["recipe"] == exported["recipe"]


def test_recipe_json_export_follows_the_sharing_rule(client):
    doc = _flow(_with_tools())
    exported = client.post("/api/flows/export/recipe", json = {"doc": doc, "group_id": "recipe"}).json()
    _assert_shared(json.dumps(exported))
    assert [t["tool_alias"] for t in exported["recipe"]["tool_configs"]] == ["web"]
    assert not any(c.get("tool_alias") == "files" for c in exported["recipe"]["columns"])
    bad = client.post("/api/flows/export/recipe", json = {"doc": doc, "group_id": "nope"})
    assert bad.status_code == 400


def test_pasting_recipe_json_with_a_stdio_server_needs_the_ui(client):
    body = {"payload": _with_tools(), "name": "Tools"}
    ok = client.post("/api/flows/import/recipe", json = body).json()
    assert any("fs" in n for n in ok["report"]["notices"])
    client.app.dependency_overrides[authenticated_via_api_key] = lambda: True
    assert client.post("/api/flows/import/recipe", json = body).status_code == 403
    assert client.post("/api/flows/import/recipe", json = {"payload": SQL}).status_code == 200


# Run settings


def test_run_settings_follow_recipe_studios_run_dialog():
    doc = _flow()
    doc["groups"][0]["run"] = {
        "rows": 5,
        "run_config": {
            "buffer_size": 250,
            "non_inference_max_parallel_workers": 8,
            "max_conversation_restarts": 2,
            "max_conversation_correction_steps": 1,
            "disable_early_shutdown": False,
            "shutdown_error_rate": 0.25,
            "shutdown_error_window": 20,
        },
        "merge_batches": True,
        "llm_parallel_requests": 16,
        "run_name": "nightly sql",
    }
    full = _run(_compiled(doc), mode = "full", rows = 1000)
    assert full["run"]["run_config"]["buffer_size"] == 250
    assert full["run"]["run_config"]["non_inference_max_parallel_workers"] == 8
    assert full["run"]["merge_batches"] is True and full["run"]["run_name"] == "nightly sql"
    assert full["run"]["rows"] == 1000 and full["run"]["execution_type"] == "full"
    assert all(
        m["inference_parameters"]["max_parallel_requests"] == 16 for m in full["recipe"]["model_configs"]
    )
    preview = _run(_compiled(doc), mode = "preview", rows = 4)
    assert preview["run"]["run_config"]["buffer_size"] == 4
    assert preview["run"]["merge_batches"] is False and "run_name" not in preview["run"]
    plain = _run(_compiled(_flow()), mode = "preview", rows = 2)
    assert plain["run"] == {"execution_type": "preview", "rows": 2}


def test_run_settings_change_the_cache_key_of_the_step():
    doc = _flow()
    before = _compiled(doc)
    doc["groups"][0]["run"] = {"llm_parallel_requests": 2}
    assert _compiled(doc) != before
