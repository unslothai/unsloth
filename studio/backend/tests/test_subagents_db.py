# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Subagent storage and routes (studio/SUBAGENTS.md)."""

import sys
import types

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from auth.authentication import get_current_subject
from routes.subagents import router
from storage import studio_db
from storage import subagents_db as db
from utils.account_context import OWNER, AccountContext, run_as

ALICE = AccountContext("11111111111111111111111111111111", "alice")

TRANSCRIPT = [
    {"kind": "text", "text": "héllo"},
    {
        "kind": "tool",
        "toolCallId": "c1",
        "toolName": "python",
        "arguments": {"code": "print(1)"},
        "status": "done",
        "approvalId": None,
        "result": "1",
    },
]


@pytest.fixture(autouse = True)
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    studio_db.close_wal_keeper()
    studio_db.reset_schema_state_for_tests()
    db.reset_schema_state_for_tests()
    yield tmp_path
    studio_db.close_wal_keeper()


def record(sub_id = "sa_1", **overrides) -> dict:
    base = {
        "id": sub_id,
        "parentThreadId": "parent-1",
        "childThreadId": "child-1",
        "sessionId": "sess-1",
        "description": "Inspect backend",
        "prompt": "Look at the backend.",
        "model": "openai/gpt-5.2",
        "tools": ["web_search", "python"],
        "status": "running",
        "error": None,
        "resultPreview": None,
        "toolCallCount": 0,
        "pendingApproval": None,
        "startedAt": 1000,
        "finishedAt": None,
        "transcript": [],
    }
    base.update(overrides)
    return base


def test_round_trip_keeps_every_field():
    db.upsert_subagent(record(transcript = TRANSCRIPT, toolCallCount = 1))
    got = db.get_subagent("sa_1")
    assert got == record(transcript = TRANSCRIPT, toolCallCount = 1)
    assert got["tools"] == ["web_search", "python"]
    assert got["pendingApproval"] is None


def test_unknown_id_is_none():
    assert db.get_subagent("missing") is None


def test_upsert_replaces_by_id():
    db.upsert_subagent(record())
    db.upsert_subagent(
        record(
            status = "completed",
            resultPreview = "done",
            toolCallCount = 3,
            finishedAt = 2000,
            transcript = TRANSCRIPT,
        )
    )
    got = db.get_subagent("sa_1")
    assert (got["status"], got["resultPreview"], got["toolCallCount"], got["finishedAt"]) == (
        "completed",
        "done",
        3,
        2000,
    )
    assert got["transcript"] == TRANSCRIPT
    assert len(db.list_subagents("parent-1")) == 1


def test_pending_approval_is_never_persisted():
    approval = {"approvalId": "a1", "toolName": "python", "toolCallId": "c1", "arguments": {}}
    db.upsert_subagent(record(pendingApproval = approval))
    assert db.get_subagent("sa_1")["pendingApproval"] is None
    assert db.list_subagents("parent-1")[0]["pendingApproval"] is None


def test_missing_optional_fields_get_defaults():
    db.upsert_subagent({"id": "sa_min", "startedAt": 5})
    got = db.get_subagent("sa_min")
    assert got["tools"] == [] and got["transcript"] == []
    assert got["parentThreadId"] is None and got["status"] == "running"


def test_list_omits_transcript_and_orders_oldest_first():
    db.upsert_subagent(record("sa_b", startedAt = 3000, transcript = TRANSCRIPT))
    db.upsert_subagent(record("sa_a", startedAt = 1000, transcript = TRANSCRIPT))
    db.upsert_subagent(record("sa_c", startedAt = 2000))
    db.upsert_subagent(record("sa_other", parentThreadId = "parent-2"))
    listed = db.list_subagents("parent-1")
    assert [item["id"] for item in listed] == ["sa_a", "sa_c", "sa_b"]
    assert all("transcript" not in item for item in listed)
    assert listed[0]["tools"] == ["web_search", "python"]
    assert db.list_subagents("nobody") == []


def test_delete_for_parent_removes_only_that_parent():
    db.upsert_subagent(record("sa_1"))
    db.upsert_subagent(record("sa_2"))
    db.upsert_subagent(record("sa_3", parentThreadId = "parent-2"))
    assert db.delete_subagents_for_parent("parent-1") == 2
    assert db.get_subagent("sa_1") is None
    assert db.get_subagent("sa_3") is not None
    assert db.delete_subagents_for_parent("parent-1") == 0


def test_records_are_per_account():
    db.upsert_subagent(record("sa_owner"))
    assert run_as(ALICE, db.get_subagent, "sa_owner") is None
    run_as(ALICE, db.upsert_subagent, record("sa_alice"))
    assert [r["id"] for r in run_as(ALICE, db.list_subagents, "parent-1")] == ["sa_alice"]
    assert [r["id"] for r in run_as(OWNER, db.list_subagents, "parent-1")] == ["sa_owner"]


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(router, prefix = "/api/subagents")
    app.dependency_overrides[get_current_subject] = lambda: "unsloth"
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def fake_live(monkeypatch):
    live = types.SimpleNamespace(runs = {}, cancelled = [])
    mod = types.ModuleType("core.inference.subagents")
    mod.live_subagent = lambda sid: live.runs.get(sid)
    mod.live_subagents_for_parent = lambda pid: [
        {k: v for k, v in r.items() if k != "transcript"}
        for r in live.runs.values()
        if r["parentThreadId"] == pid
    ]
    mod.cancel_subagent = lambda sid: live.cancelled.append(sid) or sid in live.runs
    import core.inference as pkg

    monkeypatch.setitem(sys.modules, "core.inference.subagents", mod)
    monkeypatch.setattr(pkg, "subagents", mod, raising = False)
    return live


@pytest.fixture
def no_live(monkeypatch):
    # None in sys.modules makes the import raise ImportError even once the real module exists.
    monkeypatch.setitem(sys.modules, "core.inference.subagents", None)


def test_routes_fall_back_to_stored_rows(client, no_live):
    db.upsert_subagent(record("sa_2", startedAt = 2000))
    db.upsert_subagent(record("sa_1", startedAt = 1000, transcript = TRANSCRIPT))
    body = client.get("/api/subagents", params = {"parent_thread_id": "parent-1"}).json()
    assert [r["id"] for r in body["subagents"]] == ["sa_1", "sa_2"]
    assert all("transcript" not in r for r in body["subagents"])
    detail = client.get("/api/subagents/sa_1").json()
    assert detail["transcript"] == TRANSCRIPT
    assert client.get("/api/subagents/missing").status_code == 404
    assert client.post("/api/subagents/sa_1/cancel").json() == {"cancelled": False}


def test_list_requires_parent_thread_id(client, no_live):
    assert client.get("/api/subagents").status_code == 422


def test_live_wins_and_sorts_oldest_first(client, fake_live):
    db.upsert_subagent(record("sa_1", startedAt = 1000, status = "running"))
    db.upsert_subagent(record("sa_3", startedAt = 3000, status = "completed"))
    fake_live.runs["sa_1"] = record(
        "sa_1", startedAt = 1000, status = "completed", transcript = [{"kind": "text", "text": "x"}]
    )
    fake_live.runs["sa_2"] = record("sa_2", startedAt = 2000)
    body = client.get("/api/subagents", params = {"parent_thread_id": "parent-1"}).json()
    assert [(r["id"], r["status"]) for r in body["subagents"]] == [
        ("sa_1", "completed"),
        ("sa_2", "running"),
        ("sa_3", "completed"),
    ]
    assert client.get("/api/subagents/sa_1").json()["transcript"] == [{"kind": "text", "text": "x"}]
    assert client.get("/api/subagents/sa_3").json()["transcript"] == []
    assert client.post("/api/subagents/sa_2/cancel").json() == {"cancelled": True}
    assert client.post("/api/subagents/nope/cancel").json() == {"cancelled": False}


STOPPED = "Studio stopped before this subagent finished."


def _running_with_tool(sub_id = "sa_1", **overrides):
    return record(
        sub_id,
        transcript = [
            {"kind": "text", "text": "partial"},
            {
                "kind": "tool",
                "toolCallId": "c1",
                "toolName": "python",
                "arguments": {},
                "status": "running",
                "approvalId": None,
                "result": None,
            },
            {"kind": "status", "text": "Running python"},
        ],
        **overrides,
    )


def test_stale_running_row_is_failed_in_the_list_and_written_back(client, no_live):
    db.upsert_subagent(_running_with_tool("sa_1", startedAt = 1000))
    db.upsert_subagent(record("sa_2", startedAt = 2000, status = "completed", finishedAt = 2500))
    body = client.get("/api/subagents", params = {"parent_thread_id": "parent-1"}).json()
    first, second = body["subagents"]
    assert (first["status"], first["error"]) == ("failed", STOPPED)
    assert first["finishedAt"] is not None and "transcript" not in first
    assert (second["status"], second["error"]) == ("completed", None)
    stored = db.get_subagent("sa_1")
    assert (stored["status"], stored["error"]) == ("failed", STOPPED)
    assert stored["finishedAt"] is not None
    # The correction kept the transcript, closed the open tool card and dropped the status badge.
    assert [item["kind"] for item in stored["transcript"]] == ["text", "tool"]
    assert stored["transcript"][1]["status"] == "done"
    assert db.get_subagent("sa_2")["status"] == "completed"


def test_stale_running_row_is_failed_in_the_detail(client, no_live):
    db.upsert_subagent(_running_with_tool())
    detail = client.get("/api/subagents/sa_1").json()
    assert (detail["status"], detail["error"]) == ("failed", STOPPED)
    assert detail["transcript"][0] == {"kind": "text", "text": "partial"}
    assert db.get_subagent("sa_1")["status"] == "failed"
    # Settled once: a later read returns the stored correction unchanged.
    first_finished = detail["finishedAt"]
    assert client.get("/api/subagents/sa_1").json()["finishedAt"] == first_finished


def test_cancel_on_a_stale_row_is_refused(client, no_live):
    db.upsert_subagent(record("sa_1"))
    assert client.post("/api/subagents/sa_1/cancel").json() == {"cancelled": False}


def test_a_running_row_with_a_live_record_is_left_alone(client, fake_live):
    db.upsert_subagent(record("sa_1", status = "running"))
    fake_live.runs["sa_1"] = record("sa_1", status = "running", toolCallCount = 2)
    listed = client.get("/api/subagents", params = {"parent_thread_id": "parent-1"}).json()
    assert [(r["status"], r["toolCallCount"], r["error"]) for r in listed["subagents"]] == [
        ("running", 2, None)
    ]
    assert client.get("/api/subagents/sa_1").json()["status"] == "running"
    assert db.get_subagent("sa_1")["status"] == "running"
    assert db.get_subagent("sa_1")["error"] is None


def test_stale_row_next_to_other_live_runs_is_still_failed(client, fake_live):
    db.upsert_subagent(record("sa_old", startedAt = 1000))
    fake_live.runs["sa_new"] = record("sa_new", startedAt = 2000)
    listed = client.get("/api/subagents", params = {"parent_thread_id": "parent-1"}).json()
    assert [(r["id"], r["status"]) for r in listed["subagents"]] == [
        ("sa_old", "failed"),
        ("sa_new", "running"),
    ]
