# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Automation blocks: who may run Python, HTTP and triggers (run origin x owner approval), the HTTP host guard,
the schedule (fires once per due time, never catches up, skips overlaps) and webhooks (secret, size, switched off)."""

import asyncio
import json
import time
from datetime import datetime

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from auth.authentication import get_current_subject
from core.flows import capabilities, triggers
from core.flows import engine as engine_mod
from core.flows.nodes import automation
from core.flows.registry import FlowNodeError
from routes import flows as flows_routes
from storage import flows_db
from utils.account_context import OWNER, AccountContext, bind_account, reset_account

from ._flows_helpers import doc, edge, node

OWNER_API = AccountContext(OWNER.account_id, OWNER.username, OWNER.role, via_api_key = True)
ALICE = AccountContext("alice-auto", "alice")
ACCOUNTS = {"owner": OWNER, "owner_api": OWNER_API, "alice": ALICE}
CODE = "result = 1"


@pytest.fixture
def client(monkeypatch):
    eng = engine_mod.FlowEngine()
    eng.mint_key = lambda account: ("sk-unsloth-test", None)
    eng.revoke_key = lambda key_id: None

    async def ran(ctx):
        return {"result": "ran", "log": ""}

    async def fired(ctx):
        return await automation.webhook(ctx)

    eng.handlers = {"code.python": ran, "trigger.webhook": fired, "trigger.schedule": ran}
    monkeypatch.setattr(engine_mod, "ENGINE", eng)
    app = FastAPI()

    @app.middleware("http")
    async def as_account(request: Request, call_next):
        token = bind_account(ACCOUNTS[request.headers.get("x-test-account", "owner")])
        try:
            return await call_next(request)
        finally:
            reset_account(token)

    app.include_router(flows_routes.hooks_router, prefix = "/api/flows/hooks")
    app.include_router(flows_routes.router, prefix = "/api/flows")
    app.dependency_overrides[get_current_subject] = lambda: "tester"
    with TestClient(app) as test_client:
        yield test_client


def _wait(
    client,
    run_id: str,
    account = "owner",
) -> dict:
    for _ in range(100):
        snap = client.get(f"/api/flows/runs/{run_id}", headers = {"x-test-account": account}).json()
        if snap["run"]["status"] in ("completed", "failed", "cancelled"):
            return snap
        time.sleep(0.05)
    raise AssertionError("run did not finish")


def _run(
    client,
    flow_id: str,
    account = "owner",
) -> dict:
    headers = {"x-test-account": account}
    run_id = client.post(f"/api/flows/{flow_id}/runs", json = {}, headers = headers).json()["run_id"]
    return _wait(client, run_id, account)


def _python_flow(
    client,
    flow_id = "py1",
    account = "owner",
    code = CODE,
) -> None:
    d = doc([node("p", "code.python", code = code)], id = flow_id)
    assert (
        client.post("/api/flows", json = {"doc": d}, headers = {"x-test-account": account}).status_code
        == 200
    )


def _approve(
    client,
    flow_id: str,
    account = "owner",
):
    pending = client.get(
        f"/api/flows/{flow_id}/capabilities", headers = {"x-test-account": account}
    ).json()["pending"]
    return client.post(
        f"/api/flows/{flow_id}/approve", json = pending, headers = {"x-test-account": account}
    )


def test_python_runs_only_for_the_owner_once_its_code_is_approved(client):
    _python_flow(client)
    snap = _run(client, "py1")
    assert snap["nodes"]["p"]["status"] == "failed"
    assert "not approved" in snap["nodes"]["p"]["error"]
    assert _approve(client, "py1").status_code == 200
    assert _run(client, "py1")["nodes"]["p"]["status"] == "done"
    # The owner's API key (and MCP, which forwards with a key) never runs it, approved or not.
    by_key = _run(client, "py1", "owner_api")
    assert by_key["nodes"]["p"]["status"] == "failed"
    assert "API key" in by_key["nodes"]["p"]["error"]


def test_another_account_never_runs_python_or_approves(client):
    _python_flow(client, "py2", account = "alice")
    assert _approve(client, "py2", account = "alice").status_code == 403
    snap = _run(client, "py2", "alice")
    assert snap["nodes"]["p"]["status"] == "failed"
    assert "Studio owner" in snap["nodes"]["p"]["error"]


def test_an_api_key_cannot_approve(client):
    _python_flow(client, "py3")
    assert _approve(client, "py3", account = "owner_api").status_code == 403


def test_changed_code_needs_approval_again(client):
    _python_flow(client, "py4")
    _approve(client, "py4")
    record = client.get("/api/flows/py4").json()
    edited = {**record["doc"], "nodes": [node("p", "code.python", code = "result = 2")]}
    client.put("/api/flows/py4", json = {"doc": edited, "baseUpdatedAt": record["updatedAt"]})
    assert client.get("/api/flows/py4/capabilities").json()["pending"]["python"]
    assert "not approved" in _run(client, "py4")["nodes"]["p"]["error"]
    # A doc sent with the run cannot smuggle in other code either.
    run_id = client.post(
        "/api/flows/py4/runs",
        json = {"doc": {**edited, "nodes": [node("p", "code.python", code = "result = 3")]}},
    ).json()["run_id"]
    assert "not approved" in _wait(client, run_id)["nodes"]["p"]["error"]


def test_approval_covers_only_what_the_owner_was_shown(client):
    _python_flow(client, "py5")
    resp = client.post("/api/flows/py5/approve", json = {"python": ["sha256:0000"], "triggers": True})
    assert resp.json()["pending"]["python"] == [capabilities.code_hash(CODE)]
    assert resp.json()["approved"]["triggers"] is False  # no trigger block, nothing to approve


def test_an_import_starts_unapproved(client):
    _python_flow(client, "py6")
    _approve(client, "py6")
    exported = client.get("/api/flows/py6/export", params = {"mode": "json"}).content
    imported = client.post("/api/flows/import", content = exported).json()["flow"]
    assert imported["id"] != "py6"
    assert client.get(f"/api/flows/{imported['id']}/capabilities").json()["pending"]["python"]


def test_doc_needs_and_hosts():
    d = doc(
        [
            node("p", "code.python", code = "x"),
            node("h", "http.request", url = "https://API.example.com/v1?q=1"),
            node("bad", "http.request", url = "ftp://example.com"),
            node("w", "trigger.webhook"),
        ]
    )
    need = capabilities.doc_needs(d)
    assert need["hosts"] == ["api.example.com"] and need["triggers"] is True
    assert need["python"] == [capabilities.code_hash("x")]
    assert capabilities.refusal(
        {"kind": "ui", "owner": True}, {}, "http", "", "HTTP Request"
    ).startswith("HTTP Request needs a full")


@pytest.mark.parametrize(
    "origin, ok",
    [
        ({"kind": "ui", "owner": True}, True),
        ({"kind": "trigger", "owner": True}, True),
        ({"kind": "player", "owner": True}, True),
        ({"kind": "api_key", "owner": False}, False),
        ({"kind": "ui", "owner": False}, False),
        ({}, False),
    ],
)
def test_origin_matrix(origin, ok):
    approved = {"mcp": True}
    assert (capabilities.refusal(origin, approved, "mcp", "mcp", "Agent") is None) is ok


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1/x",
        "http://localhost/",
        "http://10.0.0.5/",
        "http://169.254.169.254/latest",
    ],
)
def test_http_refuses_private_hosts(url):
    with pytest.raises(FlowNodeError):
        automation.guarded_request("GET", url, {}, None, 5)


def test_http_refuses_credentials_in_the_url():
    with pytest.raises(FlowNodeError, match = "header"):
        automation.guarded_request("GET", "https://user:pw@example.com/", {}, None, 5)


def test_http_block_checks_the_host_it_actually_calls():
    class Ctx:
        params = {"method": "GET", "url": "https://other.example.org/"}
        inputs = {}

        def capability_refusal(
            self,
            cap,
            detail = None,
        ):
            return None if detail == "api.example.com" else f"not approved: {detail}"

    with pytest.raises(FlowNodeError, match = "other.example.org"):
        asyncio.run(automation.http_request(Ctx()))


def test_python_output_parsing():
    result, log = automation.parse_python_output(
        "hello\n" + automation.PY_RESULT_MARKER + '{"a": 1}\n'
    )
    assert result == {"a": 1} and log == "hello"
    with pytest.raises(FlowNodeError, match = "did not finish"):
        automation.parse_python_output("Traceback: boom")


# Schedule


def test_cron_parsing_and_matching():
    sets = triggers.parse_cron("*/15 9-17 * * 1-5")
    assert triggers.cron_matches(sets, datetime(2026, 10, 22, 9, 30))  # a Thursday
    assert not triggers.cron_matches(sets, datetime(2026, 10, 25, 9, 30))  # a Sunday
    assert 0 in triggers.parse_cron("0 0 * * 7")[4]
    with pytest.raises(ValueError):
        triggers.parse_cron("61 * * * *")


def test_schedule_never_catches_up_and_fires_once_per_due_time():
    row = {"config": {"enabled": True, "every_minutes": 5}}
    start = int(datetime(2026, 10, 22, 9, 0).timestamp() * 1000)
    assert not triggers.due(row, datetime(2026, 10, 22, 9, 4), start)
    assert triggers.due(row, datetime(2026, 10, 22, 9, 5), start)
    cron = {"config": {"enabled": True, "cron": "30 9 * * *"}}
    fired = int(datetime(2026, 10, 22, 9, 30, 10).timestamp() * 1000)
    assert triggers.due(cron, datetime(2026, 10, 22, 9, 30, 5), start)
    assert not triggers.due(cron, datetime(2026, 10, 22, 9, 30, 40), fired)
    assert not triggers.due({"config": {"enabled": False, "every_minutes": 1}}, datetime.now(), 0)


def test_scheduler_fires_approved_flows_and_skips_overlaps(client, monkeypatch):
    d = doc(
        [
            node("s", "trigger.schedule", enabled = True, every_minutes = 1),
            node("p", "code.python", code = CODE),
        ],
        id = "sch1",
    )
    client.post("/api/flows", json = {"doc": d})
    started = []

    async def fake_start(**kwargs):
        started.append(kwargs)

        class RS:
            id = f"run-{len(started)}"

        return RS()

    monkeypatch.setattr(engine_mod.ENGINE, "start_run", fake_start)
    sched = triggers.Scheduler()
    sched.started_ms = int(datetime(2026, 10, 22, 9, 0).timestamp() * 1000)
    later = datetime(2026, 10, 22, 9, 2)

    def tick():
        token = bind_account(OWNER)
        try:
            return asyncio.run(sched.tick(None, later))
        finally:
            reset_account(token)

    assert tick() == []  # not approved yet
    _approve(client, "sch1")
    assert tick() == ["run-1"]
    assert started[0]["origin"] == "trigger"
    assert started[0]["options"]["params"]["__trigger"]["s"]["fired_at"]
    assert tick() == []  # already fired for this time
    monkeypatch.setattr(triggers, "_busy", lambda flow_id: True)
    flows_db_row = [r for r in _owner(flows_db.list_triggers) if r["flow_id"] == "sch1"][0]
    _owner(flows_db.mark_trigger_fired, "sch1", "s", 0)
    assert tick() == []  # its last run is still going
    assert flows_db_row["kind"] == "schedule"


def _owner(fn, *args):
    token = bind_account(OWNER)
    try:
        return fn(*args)
    finally:
        reset_account(token)


# Webhook


def _webhook_flow(client, enabled = True) -> str:
    d = doc([node("w", "trigger.webhook", enabled = enabled)], id = "hook1")
    client.post("/api/flows", json = {"doc": d})
    return client.get("/api/flows/hook1/triggers").json()["triggers"][0]["path"]


def test_webhook_needs_its_secret_approval_and_a_small_json_body(client):
    path = _webhook_flow(client)
    assert client.post("/api/flows/hooks/" + "x" * 32, json = {}).status_code == 404
    assert client.post(path, json = {"a": 1}).status_code == 409  # triggers not approved
    _approve(client, "hook1")
    run_id = client.post(path, json = {"a": 1}).json()["run_id"]
    snap = _wait(client, run_id)
    assert snap["nodes"]["w"]["outputs"]["payload"] == {"a": 1}
    assert client.post(path, content = b"not json").status_code == 400
    big = json.dumps({"x": "y" * (flows_routes.WEBHOOK_MAX_BYTES + 1)})
    assert client.post(path, content = big).status_code == 413


def test_webhook_secret_rotates_and_is_owner_only(client):
    path = _webhook_flow(client)
    assert (
        client.get("/api/flows/hook1/triggers", headers = {"x-test-account": "owner_api"}).json()[
            "triggers"
        ]
        == []
    )
    rotated = client.post("/api/flows/hook1/triggers/w/rotate").json()["triggers"][0]["path"]
    assert rotated != path
    _approve(client, "hook1")
    assert client.post(path, json = {}).status_code == 404
    assert client.post(rotated, json = {}).status_code == 200


def test_switched_off_webhook_refuses(client):
    path = _webhook_flow(client, enabled = False)
    _approve(client, "hook1")
    assert client.post(path, json = {}).status_code == 409


def test_webhook_secret_never_lands_in_the_document(client):
    path = _webhook_flow(client)
    exported = client.get("/api/flows/hook1/export", params = {"mode": "json"}).text
    assert path.rsplit("/", 1)[1] not in exported


def test_other_accounts_triggers_never_fire(client):
    d = doc([node("w", "trigger.webhook", enabled = True)], id = "hook2")
    client.post("/api/flows", json = {"doc": d}, headers = {"x-test-account": "alice"})
    assert (
        client.get("/api/flows/hook2/triggers", headers = {"x-test-account": "alice"}).json()[
            "triggers"
        ]
        == []
    )
