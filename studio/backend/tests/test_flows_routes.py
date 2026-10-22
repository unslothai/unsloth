# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The /api/flows routes over a bare app: CRUD and conflicts, validation, runs with SSE replay, sessions, import,
assets and examples, and account isolation."""

import hashlib
import json
import time

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from auth.authentication import get_current_subject
from core.flows import engine as engine_mod
from routes import flows as flows_routes
from utils.account_context import AccountContext, OWNER, bind_account, reset_account

from ._flows_helpers import doc, edge, node
from .test_flows_package import PNG, PNG_SHA

ALICE = AccountContext("alice-flows", "alice")
BOB = AccountContext("bob-flows", "bob")
ACCOUNTS = {"owner": OWNER, "alice": ALICE, "bob": BOB}


@pytest.fixture
def client(monkeypatch):
    eng = engine_mod.FlowEngine()
    eng.mint_key = lambda account: ("sk-unsloth-test", None)
    eng.revoke_key = lambda key_id: None

    async def text(ctx):
        return {"text": str(ctx.params.get("value") or "")}

    async def result(ctx):
        return {"value": ctx.inputs.get("value")}

    eng.handlers = {"input.text": text, "output.result": result}
    monkeypatch.setattr(engine_mod, "ENGINE", eng)

    app = FastAPI()

    @app.middleware("http")
    async def as_account(request: Request, call_next):
        token = bind_account(ACCOUNTS[request.headers.get("x-test-account", "owner")])
        try:
            return await call_next(request)
        finally:
            reset_account(token)

    app.include_router(flows_routes.router, prefix = "/api/flows")
    app.dependency_overrides[get_current_subject] = lambda: "tester"
    # One event loop for the whole test: without the context manager every request gets its own portal,
    # and a run task started by one request is stranded when that portal closes.
    with TestClient(app) as test_client:
        yield test_client


def _simple(flow_id = "flow1") -> dict:
    return doc(
        [node("a", "input.text", value = "hello"), node("o", "output.result", name = "greeting")],
        [edge("a", "text", "o", "value")],
        id = flow_id,
    )


def test_catalog_lists_every_spec(client):
    body = client.get("/api/flows/nodes").json()
    assert {n["type"] for n in body["nodes"]} >= {
        "llm.generate",
        "image.generate",
        "story.scene",
        "world.npc",
    }


def test_crud_conflict_and_tombstone(client):
    created = client.post("/api/flows", json = {"doc": _simple()}).json()
    assert created["id"] == "flow1"
    assert client.get("/api/flows").json()["flows"][0]["nodeCount"] == 2
    renamed = {**created["doc"], "name": "Renamed"}
    saved = client.put(
        "/api/flows/flow1", json = {"doc": renamed, "baseUpdatedAt": created["updatedAt"]}
    )
    assert saved.status_code == 200 and saved.json()["name"] == "Renamed"
    stale = client.put(
        "/api/flows/flow1", json = {"doc": renamed, "baseUpdatedAt": created["updatedAt"]}
    )
    assert stale.status_code == 409
    assert client.put("/api/flows/other", json = {"doc": renamed}).status_code == 400
    assert client.delete("/api/flows/flow1").status_code == 204
    assert client.get("/api/flows/flow1").status_code == 404
    assert client.put("/api/flows/flow1", json = {"doc": renamed}).status_code == 410
    # Creating with a deleted id gets a fresh one.
    assert client.post("/api/flows", json = {"doc": _simple()}).json()["id"] != "flow1"


def test_invalid_documents_are_refused(client):
    bad = client.post("/api/flows", json = {"doc": {**_simple(), "exec": "x"}})
    assert bad.status_code == 400
    result = client.post("/api/flows/validate", json = {"doc": doc([node("s", "text.split")])}).json()
    assert result["ok"] is False and result["errors"][0]["node_id"] == "s"


def _wait_terminal(
    client,
    run_id: str,
    headers = None,
) -> dict:
    for _ in range(100):
        snap = client.get(f"/api/flows/runs/{run_id}", headers = headers).json()
        if snap["run"]["status"] in ("completed", "failed", "cancelled"):
            return snap
        time.sleep(0.05)
    raise AssertionError("run did not finish")


def _events(text: str) -> list[dict]:
    return [json.loads(line[5:]) for line in text.splitlines() if line.startswith("data:")]


def test_run_snapshot_outputs_and_event_replay(client):
    client.post("/api/flows", json = {"doc": _simple()})
    run_id = client.post("/api/flows/flow1/runs", json = {}).json()["run_id"]
    snap = _wait_terminal(client, run_id)
    assert snap["run"]["outputs"] == {"greeting": "hello"}
    assert snap["nodes"]["a"]["status"] == "done"
    replay = _events(client.post(f"/api/flows/runs/{run_id}/events", params = {"after": 0}).text)
    assert replay[0]["seq"] == 1 and replay[-1]["type"] == "run.completed"
    tail = _events(
        client.get(
            f"/api/flows/runs/{run_id}/events", headers = {"Last-Event-ID": str(replay[-3]["seq"])}
        ).text
    )
    assert [e["seq"] for e in tail] == [e["seq"] for e in replay[-2:]]
    snapshot_only = _events(client.post(f"/api/flows/runs/{run_id}/events").text)
    assert snapshot_only[0]["type"] == "snapshot"
    runs = client.get("/api/flows/flow1/runs").json()["runs"]
    assert runs[0]["id"] == run_id and runs[0]["status"] == "completed"
    latest = client.get("/api/flows/flow1/latest-outputs").json()["nodes"]
    assert latest["a"]["outputs"] == {"text": "hello"}
    assert client.post("/api/flows/runs/nope/cancel").status_code == 404


def test_run_with_unsaved_editor_doc_and_compile_errors(client):
    client.post("/api/flows", json = {"doc": _simple()})
    edited = _simple()
    edited["nodes"][0]["params"]["value"] = "edited"
    run_id = client.post("/api/flows/flow1/runs", json = {"doc": edited}).json()["run_id"]
    assert _wait_terminal(client, run_id)["run"]["outputs"] == {"greeting": "edited"}
    broken = client.post(
        "/api/flows/flow1/runs", json = {"doc": doc([node("s", "text.split")], id = "flow1")}
    )
    assert broken.status_code == 400
    assert broken.json()["detail"]["errors"][0]["node_id"] == "s"


def test_flows_and_runs_are_per_account(client):
    alice = {"x-test-account": "alice"}
    bob = {"x-test-account": "bob"}
    client.post("/api/flows", json = {"doc": _simple("shared-id")}, headers = alice)
    assert client.get("/api/flows/shared-id", headers = bob).status_code == 404
    assert client.get("/api/flows", headers = bob).json()["flows"] == []
    run_id = client.post("/api/flows/shared-id/runs", json = {}, headers = alice).json()["run_id"]
    _wait_terminal(client, run_id, alice)
    assert client.get(f"/api/flows/runs/{run_id}", headers = bob).status_code == 404
    assert client.post(f"/api/flows/runs/{run_id}/events", headers = bob).status_code == 404
    assert client.post(f"/api/flows/runs/{run_id}/cancel", headers = bob).status_code == 404
    client.put("/api/flows/shared-id/sessions/slot1", json = {"state": {"scene": "a"}}, headers = alice)
    assert client.get("/api/flows/shared-id/sessions/slot1", headers = bob).status_code == 404


def test_sessions_round_trip(client):
    client.post("/api/flows", json = {"doc": _simple()})
    saved = client.put(
        "/api/flows/flow1/sessions/auto",
        json = {"state": {"scene": "s2", "vars": {"hp": 3}}, "label": "Cave"},
    )
    assert saved.status_code == 200
    assert client.get("/api/flows/flow1/sessions").json()["sessions"][0]["label"] == "Cave"
    assert client.get("/api/flows/flow1/sessions/auto").json()["state"]["vars"] == {"hp": 3}
    assert client.put("/api/flows/flow1/sessions/../x", json = {"state": {}}).status_code in (
        400,
        404,
    )
    assert client.delete("/api/flows/flow1/sessions/auto").status_code == 204
    assert client.get("/api/flows/flow1/sessions/auto").status_code == 404


def test_import_export_assets_round_trip(client, monkeypatch):
    async def no_resident(request):
        return {}

    monkeypatch.setattr(flows_routes, "_resident", no_resident)
    uploaded = client.post("/api/flows/assets", content = PNG).json()["media"]
    assert uploaded["ref"] == f"sha256:{PNG_SHA}" and uploaded["kind"] == "image"
    assert client.get(f"/api/flows/assets/sha256:{PNG_SHA}").content == PNG
    assert client.get(f"/api/flows/assets/sha256:{'0' * 64}").status_code == 404
    assert client.post("/api/flows/assets", content = b"#!/bin/sh").status_code == 415

    d = doc(
        [node("i", "input.image", asset = uploaded)], id = "withimg", baked = {"i": {"image": uploaded}}
    )
    client.post("/api/flows", json = {"doc": d})
    exported = client.get("/api/flows/withimg/export", params = {"mode": "baked"})
    assert exported.status_code == 200 and exported.content[:2] == b"PK"
    imported = client.post(
        "/api/flows/import", content = exported.content, params = {"name": "Copy"}
    ).json()
    assert imported["flow"]["id"] != "withimg" and imported["flow"]["name"] == "Copy"
    assert imported["report"]["baked_nodes"] == ["i"]
    assert client.post("/api/flows/import", content = b"PK\x03\x04garbage").status_code == 400
    assert client.get("/api/flows/withimg/export", params = {"mode": "evil"}).status_code == 400


def test_examples_list_and_copy(client, monkeypatch):
    hello = {**_simple("hello"), "kind": "pipeline"}
    monkeypatch.setattr(flows_routes.examples, "example_ids", lambda: ["hello"])
    monkeypatch.setattr(
        flows_routes.examples,
        "example",
        lambda i: json.loads(json.dumps(hello)) if i == "hello" else None,
    )
    examples = client.get("/api/flows/examples").json()["examples"]
    assert [e["id"] for e in examples] == ["hello"] and examples[0]["example"] is True
    copied = client.post("/api/flows/examples/hello/copy").json()
    assert copied["id"] != "hello"
    assert copied["doc"]["meta"]["derived_from"][0]["id"] == "hello"
    assert client.post("/api/flows/examples/missing/copy").status_code == 404


def test_bundled_examples_are_listed_and_copy(client):
    listed = client.get("/api/flows/examples").json()["examples"]
    assert len(listed) == 10 and "lantern-picnic" in {e["id"] for e in listed}
    copied = client.post("/api/flows/examples/lantern-picnic/copy").json()
    assert copied["doc"]["kind"] == "story"


def test_invoke_entrypoint_errors(client):
    client.post("/api/flows", json = {"doc": _simple()})
    refused = client.post("/api/flows/flow1/invoke", json = {"entry": "a"})
    assert refused.status_code == 422 and "entry point" in refused.json()["detail"]


def test_asset_ids_are_content_hashes():
    assert hashlib.sha256(PNG).hexdigest() == PNG_SHA


def test_event_replay_pages_past_one_stored_page(client, monkeypatch):
    client.post("/api/flows", json = {"doc": _simple()})
    run_id = client.post("/api/flows/flow1/runs", json = {}).json()["run_id"]
    _wait_terminal(client, run_id)
    monkeypatch.setattr(flows_routes, "REPLAY_PAGE", 2)
    real = flows_routes.flows_db.events_after
    monkeypatch.setattr(
        flows_routes.flows_db, "events_after", lambda rid, after: real(rid, after, limit = 2)
    )
    replay = _events(client.post(f"/api/flows/runs/{run_id}/events", params = {"after": 0}).text)
    assert [e["seq"] for e in replay] == list(range(1, len(replay) + 1))
    assert replay[-1]["type"] == "run.completed" and len(replay) > 2
