# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

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
    run_id = client.post(
        "/api/flows/py4/runs",
        json = {"doc": {**edited, "nodes": [node("p", "code.python", code = "result = 3")]}},
    ).json()["run_id"]
    assert "not approved" in _wait(client, run_id)["nodes"]["p"]["error"]


def test_approval_covers_only_what_the_owner_was_shown(client):
    _python_flow(client, "py5")
    resp = client.post("/api/flows/py5/approve", json = {"python": ["sha256:0000"], "triggers": True})
    assert resp.json()["pending"]["python"] == [capabilities.code_hash(CODE)]
    assert resp.json()["approved"]["triggers"] is False


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


def test_cron_parsing_and_matching():
    sets = triggers.parse_cron("*/15 9-17 * * 1-5")
    assert triggers.cron_matches(sets, datetime(2026, 10, 22, 9, 30))
    assert not triggers.cron_matches(sets, datetime(2026, 10, 25, 9, 30))
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

    assert tick() == []
    _approve(client, "sch1")
    assert tick() == ["run-1"]
    assert started[0]["origin"] == "trigger"
    assert started[0]["options"]["params"]["__trigger"]["s"]["fired_at"]
    assert tick() == []
    monkeypatch.setattr(triggers, "_busy", lambda flow_id: True)
    flows_db_row = [r for r in _owner(flows_db.list_triggers) if r["flow_id"] == "sch1"][0]
    _owner(flows_db.mark_trigger_fired, "sch1", "s", 0)
    assert tick() == []
    assert flows_db_row["kind"] == "schedule"


def _owner(fn, *args):
    token = bind_account(OWNER)
    try:
        return fn(*args)
    finally:
        reset_account(token)


def _webhook_flow(client, enabled = True) -> str:
    d = doc([node("w", "trigger.webhook", enabled = enabled)], id = "hook1")
    client.post("/api/flows", json = {"doc": d})
    return client.get("/api/flows/hook1/triggers").json()["triggers"][0]["path"]


def test_webhook_needs_its_secret_approval_and_a_small_json_body(client):
    path = _webhook_flow(client)
    assert client.post("/api/flows/hooks/" + "x" * 32, json = {}).status_code == 404
    assert client.post(path, json = {"a": 1}).status_code == 409
    _approve(client, "hook1")
    run_id = client.post(path, json = {"a": 1}).json()["run_id"]
    snap = _wait(client, run_id)
    assert snap["nodes"]["w"]["outputs"]["payload"] == {"a": 1}
    assert client.post(path, content = b"not json").status_code == 400
    assert client.post(path, content = b'{"a": NaN}').status_code == 400
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


def test_approving_a_new_capability_keeps_the_earlier_ones(client):
    d = doc(
        [
            node("p", "code.python", code = CODE),
            node("h", "http.request", url = "https://example.com/a"),
        ],
        id = "both1",
    )
    client.post("/api/flows", json = {"doc": d})
    _approve(client, "both1")
    d["nodes"][1]["params"]["url"] = "https://example.org/b"
    assert client.put("/api/flows/both1", json = {"doc": d}).status_code == 200
    assert _approve(client, "both1").status_code == 200
    report = client.get("/api/flows/both1/capabilities").json()
    assert report["approved"]["python"] == [capabilities.code_hash(CODE)]
    assert report["approved"]["hosts"] == ["example.org"]
    assert not capabilities.has_pending(report["pending"])


def test_a_trigger_run_skips_the_other_triggers_branches(client):
    d = doc(
        [
            node("w", "trigger.webhook", enabled = True),
            node("s", "trigger.schedule", enabled = True),
            node("p", "code.python", code = CODE),
            node("q", "code.python", code = CODE),
        ],
        [edge("s", "fired_at", "p", "input"), edge("w", "payload", "q", "input")],
        id = "two1",
    )
    client.post("/api/flows", json = {"doc": d})
    _approve(client, "two1")
    path = client.get("/api/flows/two1/triggers").json()["triggers"]
    path = [t for t in path if t["kind"] == "webhook"][0]["path"]
    snap = _wait(client, client.post(path, json = {"a": 1}).json()["run_id"])
    assert snap["nodes"]["q"]["status"] == "done"
    assert snap["nodes"]["s"]["status"] == "skipped"
    assert snap["nodes"]["p"]["status"] == "skipped"


def test_a_skipped_schedule_time_is_not_started_late(client, monkeypatch):
    d = doc([node("s", "trigger.schedule", enabled = True, every_minutes = 1)], id = "sch2")
    client.post("/api/flows", json = {"doc": d})
    _approve(client, "sch2")
    monkeypatch.setattr(triggers, "_busy", lambda flow_id: True)
    at = int(datetime(2026, 10, 22, 9, 2).timestamp() * 1000)
    assert _owner(asyncio.run, triggers.fire(None, "sch2", "s", {}, at)) is None
    row = [r for r in _owner(flows_db.list_triggers) if r["flow_id"] == "sch2"][0]
    assert row["last_fired_at"] == at


def test_a_trigger_off_in_the_saved_flow_never_fires_even_if_the_index_lags(client, monkeypatch):
    path = _webhook_flow(client)
    _approve(client, "hook1")
    d = client.get("/api/flows/hook1").json()
    d["doc"]["nodes"][0]["params"]["enabled"] = False
    monkeypatch.setattr(triggers, "sync", lambda *a: (_ for _ in ()).throw(RuntimeError("db")))
    assert client.put("/api/flows/hook1", json = {"doc": d["doc"]}).status_code == 200
    assert client.post(path, json = {}).status_code == 409


def test_cron_either_day_field_matches_when_both_are_set():
    sets = triggers.parse_cron("0 9 1 * 1")
    assert triggers.cron_matches(sets, datetime(2026, 10, 26, 9, 0))
    assert triggers.cron_matches(sets, datetime(2026, 10, 1, 9, 0))
    assert not triggers.cron_matches(sets, datetime(2026, 10, 27, 9, 0))
    weekdays = triggers.parse_cron("0 9 * * 1")
    assert not triggers.cron_matches(weekdays, datetime(2026, 10, 1, 9, 0))


def test_made_up_secrets_cannot_reset_a_real_webhooks_rate_limit(client, monkeypatch):
    monkeypatch.setattr(triggers, "_HITS", {})
    path = _webhook_flow(client)
    token = path.rsplit("/", 1)[1]
    for _ in range(triggers.WEBHOOK_PER_MINUTE):
        triggers.rate_limited(token)
    for i in range(3):
        assert client.post(f"/api/flows/hooks/{'z' * 30}{i:04d}", json = {}).status_code == 404
    assert len(triggers._HITS) == 1
    monkeypatch.setattr(
        triggers, "_HITS", {**triggers._HITS, **{f"k{i}": [0.0] for i in range(10001)}}
    )
    assert triggers.rate_limited(token)


def test_a_long_log_does_not_cut_the_python_result():
    import subprocess
    import sys

    program = automation._program("print('x' * 50000)\nresult = {'n': len(inputs)}", [1, 2])
    out = subprocess.run([sys.executable, "-c", program], capture_output = True, text = True).stdout
    result, log = automation.parse_python_output(out[:16000])
    assert result == {"n": 2} and log.startswith("xxx")


def test_http_body_read_stops_at_the_deadline(monkeypatch):
    import threading

    from core.inference import tools as web

    class Slow:
        status = 200
        headers = {"content-type": "text/plain"}

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self, n):
            time.sleep(0.4)
            return b"x"

    class Opener:
        def open(self, request, timeout):
            return Slow()

    monkeypatch.setattr(
        web, "_validate_and_resolve_host", lambda host, port: (True, "", ["93.184.216.34"])
    )
    import urllib.request

    monkeypatch.setattr(urllib.request, "build_opener", lambda *a: Opener())
    started = time.monotonic()
    with pytest.raises(FlowNodeError, match = "longer than"):
        automation.guarded_request("GET", "https://example.com", {}, None, 1, threading.Event())
    assert time.monotonic() - started < 3


def test_deleting_a_flow_removes_its_python_sandbox(client, monkeypatch):
    from core.inference import tools as tools_mod

    removed = []
    monkeypatch.setattr(
        tools_mod, "remove_session_sandbox", lambda sid, files = False: removed.append((sid, files))
    )
    _python_flow(client, "py9")
    assert client.delete("/api/flows/py9").status_code == 204
    assert removed == [(automation.python_session("py9"), True)]


def test_export_drops_request_header_values_and_url_secrets():
    from core.flows.package import scrub

    found = scrub(
        {
            "url": "https://u:pw@api.example.com/v1?q=cats&api_key=SECRET&page=2",
            "headers": {
                "Cookie": "session=abc",
                "X-Custom-Auth": "s3",
                "Accept": "application/json",
            },
        }
    )
    assert found["url"] == "https://api.example.com/v1?q=cats&api_key=&page=2"
    assert found["headers"] == {"Cookie": "", "X-Custom-Auth": "", "Accept": "application/json"}


def test_an_api_key_save_never_arms_triggers(client):
    path = _webhook_flow(client)
    _approve(client, "hook1")
    d = client.get("/api/flows/hook1").json()["doc"]
    d["nodes"].append(node("s", "trigger.schedule", enabled = True, every_minutes = 1))
    assert (
        client.put(
            "/api/flows/hook1", json = {"doc": d}, headers = {"x-test-account": "owner_api"}
        ).status_code
        == 200
    )
    assert [t["node_id"] for t in client.get("/api/flows/hook1/triggers").json()["triggers"]] == [
        "w"
    ]
    assert client.get("/api/flows/hook1/capabilities").json()["pending"]["triggers"] is True
    assert client.post(path, json = {}).status_code == 409


def test_python_saved_without_code_needs_the_default_codes_approval(client):
    from core.flows.specs import SPECS_BY_TYPE

    d = doc([node("p", "code.python")], id = "pydef")
    client.post("/api/flows", json = {"doc": d})
    default = next(
        p["default"] for p in SPECS_BY_TYPE["code.python"]["params"] if p["id"] == "code"
    )
    pending = client.get("/api/flows/pydef/capabilities").json()["pending"]["python"]
    assert pending == [capabilities.code_hash(default)]
    _approve(client, "pydef")
    assert _run(client, "pydef")["nodes"]["p"]["status"] == "done"


def test_python_nan_results_are_refused():
    with pytest.raises(FlowNodeError, match = "NaN"):
        automation.parse_python_output(automation.PY_RESULT_MARKER + "NaN\n")


def test_owner_approval_indexes_a_trigger_an_api_key_added(client):
    _python_flow(client, "keyed")
    d = client.get("/api/flows/keyed").json()["doc"]
    d["nodes"].append(node("w", "trigger.webhook", enabled = True))
    client.put("/api/flows/keyed", json = {"doc": d}, headers = {"x-test-account": "owner_api"})
    assert client.get("/api/flows/keyed/triggers").json()["triggers"] == []
    _approve(client, "keyed")
    assert [t["kind"] for t in client.get("/api/flows/keyed/triggers").json()["triggers"]] == [
        "webhook"
    ]


def test_a_trigger_fires_only_as_the_kind_the_saved_flow_has(client, monkeypatch):
    path = _webhook_flow(client)
    _approve(client, "hook1")
    d = client.get("/api/flows/hook1").json()["doc"]
    d["nodes"][0]["type"] = "trigger.schedule"
    monkeypatch.setattr(triggers, "sync", lambda *a: (_ for _ in ()).throw(RuntimeError("db")))
    client.put("/api/flows/hook1", json = {"doc": d})
    assert client.post(path, json = {}).status_code == 409


def test_http_json_with_nan_is_not_parsed():
    assert automation._no_constant
    import inspect
    assert "parse_constant = _no_constant" in inspect.getsource(automation.http_request)


def test_export_drops_signed_url_credentials_and_approvals_bind_the_full_digest():
    from core.flows.package import scrub

    signed = (
        "https://bucket.s3.amazonaws.com/a.csv?X-Amz-Algorithm=AWS4-HMAC-SHA256"
        "&X-Amz-Credential=AKIA%2F20261023&X-Amz-Signature=abc123&X-Goog-Signature=def&x-amz-security-token=tok"
        "&response-content-type=text%2Fcsv"
    )
    assert scrub({"url": signed})["url"] == (
        "https://bucket.s3.amazonaws.com/a.csv?X-Amz-Algorithm=AWS4-HMAC-SHA256"
        "&X-Amz-Credential=&X-Amz-Signature=&X-Goog-Signature=&x-amz-security-token="
        "&response-content-type=text%2Fcsv"
    )
    encoded = scrub({"url": "https://api.example/?api%5Fkey=S1&x-amz-security%2dtoken=S2&q=1"})[
        "url"
    ]
    assert encoded == "https://api.example/?api%5Fkey=&x-amz-security%2dtoken=&q=1"
    # A truncated digest would let a crafted program collide with an approved one.
    assert len(capabilities.code_hash("print(1)")) == len("sha256:") + 64


@pytest.mark.parametrize("expr", ["0, 9 * * *", "0,,30 * * * *", ",0 9 * * *", "0 9, * * *"])
def test_a_stray_comma_in_cron_is_an_error_not_every_minute(expr):
    with pytest.raises(ValueError):
        triggers.parse_cron(expr)


def test_run_flow_runs_a_called_flows_python_only_with_that_flows_approval(client):
    from core.flows.nodes import logic, output

    engine_mod.ENGINE.handlers.update(
        {"flow.call": logic.call_flow, "output.result": output.result}
    )
    _python_flow(client, flow_id = "child_py")
    parent = doc(
        [
            node("call", "flow.call", flow_id = "child_py", values = {}),
            node("out", "output.result", name = "got"),
        ],
        [edge("call", "outputs", "out", "value")],
        id = "parent_py",
    )
    assert client.post("/api/flows", json = {"doc": parent}).status_code == 200
    first = _run(client, "parent_py")
    assert (
        first["nodes"]["call"]["status"] == "failed"
        and "not approved" in first["nodes"]["call"]["error"]
    )
    assert _approve(client, "child_py").status_code == 200
    assert _run(client, "parent_py")["nodes"]["call"]["status"] == "done"
    by_key = _run(client, "parent_py", "owner_api")
    assert (
        by_key["nodes"]["call"]["status"] == "failed"
        and "API key" in by_key["nodes"]["call"]["error"]
    )


@pytest.mark.parametrize("stop_after", [None, 0.2])
def test_a_trickling_body_cannot_outlast_the_deadline_or_stop(monkeypatch, stop_after):
    import threading
    import urllib.request
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    from core.inference import tools as web

    class Trickle(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Length", "100")
            self.end_headers()
            try:
                for _ in range(100):
                    self.wfile.write(b"x")
                    self.wfile.flush()
                    time.sleep(0.1)
            except OSError:
                pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Trickle)
    threading.Thread(target = server.serve_forever, daemon = True).start()
    real = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    class Local:
        def open(self, request, timeout):
            return real.open(f"http://127.0.0.1:{server.server_port}/", timeout = timeout)

    monkeypatch.setattr(
        web, "_validate_and_resolve_host", lambda host, port: (True, "", ["127.0.0.1"])
    )
    monkeypatch.setattr(urllib.request, "build_opener", lambda *a: Local())
    stop = threading.Event()
    if stop_after is not None:
        threading.Timer(stop_after, stop.set).start()
    started = time.monotonic()
    try:
        with pytest.raises(FlowNodeError, match = "Stopped" if stop_after else "longer than"):
            automation.guarded_request("GET", "http://example.com/", {}, None, 1, stop)
        assert time.monotonic() - started < (0.8 if stop_after else 1.6)
    finally:
        server.shutdown()
        server.server_close()
