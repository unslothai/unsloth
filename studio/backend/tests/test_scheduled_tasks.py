# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Scheduled tasks: schedule math, storage, routes, and the runner that posts answers to chats."""

import asyncio
import sqlite3
import threading
from datetime import datetime
from types import SimpleNamespace

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient
from pydantic import ValidationError

from auth.authentication import authenticated_via_api_key, get_current_subject
from core.scheduled_tasks import scheduler as scheduler_module
from core.scheduled_tasks.schedule import InvalidSchedule, next_run_after, parse_schedule
from core.scheduled_tasks.scheduler import (
    Answer,
    RunFailed,
    ScheduledTaskScheduler,
    generate_answer,
    next_run_for,
)
from routes import inference as inference_routes
from routes.scheduled_tasks import router
from storage import scheduled_tasks_db as db
from storage import studio_db
from utils.account_context import OWNER

DAY = 24 * 60 * 60 * 1000


def ms(iso: str) -> int:
    return int(datetime.fromisoformat(iso).timestamp() * 1000)


def make_task(**overrides) -> dict:
    task = {
        "id": "task-1",
        "ownerSubject": "unsloth",
        "title": "Morning brief",
        "prompt": "Write a haiku about mornings.",
        "schedule": {"kind": "daily", "time": "09:00"},
        "timezone": "UTC",
        "enabled": True,
        "notify": "all",
        "named": True,
        "nextRunAt": ms("2026-10-14T09:00:00+00:00"),
        "createdAt": 1,
        "updatedAt": 1,
    }
    task.update(overrides)
    return db.create_task(task)


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(router, prefix = "/api/scheduled-tasks")
    app.dependency_overrides[get_current_subject] = lambda: "unsloth"
    app.dependency_overrides[authenticated_via_api_key] = lambda: False
    with TestClient(app) as test_client:
        yield test_client


# ── When a task runs ─────────────────────────────────────────────


@pytest.mark.parametrize(
    "schedule, zone, after, expected",
    [
        ({"kind": "daily", "time": "09:00"}, "UTC", "2026-10-14T08:59", "2026-10-14T09:00"),
        # Strictly after: a run at 09:00 schedules tomorrow's, not itself again.
        ({"kind": "daily", "time": "09:00"}, "UTC", "2026-10-14T09:00", "2026-10-15T09:00"),
        # Wall-clock time in the task's zone: 09:00 in Istanbul is 06:00 UTC.
        (
            {"kind": "daily", "time": "09:00"},
            "Europe/Istanbul",
            "2026-10-14T05:00",
            "2026-10-14T06:00",
        ),
        # 2026-10-14 is a Wednesday: Monday/Friday next lands on Friday.
        (
            {"kind": "weekly", "time": "09:00", "days": [1, 5]},
            "UTC",
            "2026-10-14T12:00",
            "2026-10-16T09:00",
        ),
        # A single weekday that already passed today comes round a week later.
        (
            {"kind": "weekly", "time": "09:00", "days": [3]},
            "UTC",
            "2026-10-14T12:00",
            "2026-10-21T09:00",
        ),
        # The 31st runs on the last day of a shorter month, leap years included.
        (
            {"kind": "monthly", "time": "09:00", "day": 31},
            "UTC",
            "2026-02-01T00:00",
            "2026-02-28T09:00",
        ),
        (
            {"kind": "monthly", "time": "09:00", "day": 31},
            "UTC",
            "2028-02-01T00:00",
            "2028-02-29T09:00",
        ),
        (
            {"kind": "monthly", "time": "09:00", "day": 15},
            "UTC",
            "2026-12-20T00:00",
            "2027-01-15T09:00",
        ),
        (
            {"kind": "once", "at": "2026-10-15T09:00"},
            "America/New_York",
            "2026-10-14T00:00",
            "2026-10-15T13:00",
        ),
    ],
)
def test_next_run_after(schedule, zone, after, expected):
    following = next_run_after(parse_schedule(schedule), zone, ms(f"{after}:00+00:00"))
    assert following == ms(f"{expected}:00+00:00")


def test_a_one_time_task_has_no_run_left_once_its_time_passes():
    schedule = parse_schedule({"kind": "once", "at": "2026-10-15T09:00"})
    assert next_run_after(schedule, "UTC", ms("2026-10-15T09:00:00+00:00")) is None


def test_a_time_the_clocks_skip_runs_just_past_the_change():
    # New York skips 02:00-03:00 on 2026-03-08; 02:30 runs at 03:30 EDT, 07:30 UTC.
    schedule = parse_schedule({"kind": "daily", "time": "02:30"})
    following = next_run_after(schedule, "America/New_York", ms("2026-03-08T05:00:00+00:00"))
    assert following == ms("2026-03-08T07:30:00+00:00")


def test_a_time_that_happens_twice_runs_once():
    # 01:30 happens twice in New York on 2026-11-01: the run takes the first and skips the second.
    schedule = parse_schedule({"kind": "daily", "time": "01:30"})
    first = next_run_after(schedule, "America/New_York", ms("2026-11-01T04:00:00+00:00"))
    assert first == ms("2026-11-01T05:30:00+00:00")
    assert next_run_after(schedule, "America/New_York", first) == ms("2026-11-02T06:30:00+00:00")


@pytest.mark.parametrize(
    "schedule",
    [
        {"kind": "daily", "time": "24:00"},
        {"kind": "daily", "time": "9:00"},
        {"kind": "weekly", "time": "09:00", "days": []},
        {"kind": "weekly", "time": "09:00", "days": [0]},
        {"kind": "monthly", "time": "09:00", "day": 32},
        {"kind": "once", "at": "2026-02-30T09:00"},
        {"kind": "daily", "time": "09:00", "extra": True},
        {"kind": "hourly", "time": "09:00"},
    ],
)
def test_malformed_schedules_are_refused(schedule):
    with pytest.raises(ValidationError):
        parse_schedule(schedule)


@pytest.mark.parametrize("zone", ["Mars/Olympus_Mons", "../../etc/passwd", "", "x" * 65])
def test_unknown_time_zones_are_refused(zone):
    with pytest.raises(InvalidSchedule):
        next_run_after(parse_schedule({"kind": "daily", "time": "09:00"}), zone, 0)


def test_weekdays_are_deduplicated_and_sorted():
    assert parse_schedule({"kind": "weekly", "time": "09:00", "days": [5, 1, 5]}).days == [1, 5]


# ── Claiming runs ────────────────────────────────────────────────


def test_a_claim_runs_an_occurrence_once_and_skips_the_ones_studio_missed():
    due = ms("2026-10-11T09:00:00+00:00")
    make_task(nextRunAt = due)
    now = ms("2026-10-14T12:00:00+00:00")

    task, run = db.claim_next(now, "worker", next_run_for)

    assert run["status"] == "running" and run["source"] == "schedule"
    assert run["scheduledFor"] == due
    # Three missed mornings collapse into this run; the next one is tomorrow, not the 12th.
    assert task["nextRunAt"] == ms("2026-10-15T09:00:00+00:00")
    assert db.claim_next(now, "worker", next_run_for) is None


def test_a_due_task_waits_while_it_already_has_a_run():
    make_task()
    db.request_run("task-1")
    now = ms("2026-10-14T12:00:00+00:00")
    _task, manual = db.claim_next(now, "worker", next_run_for)
    assert manual["source"] == "manual"
    # Due since 09:00, but the manual run is still going.
    assert db.claim_next(now, "worker", next_run_for) is None
    assert db.next_due_at() is None
    db.finish_run(manual["id"], "worker", status = "succeeded")
    _task, scheduled = db.claim_next(now, "worker", next_run_for)
    assert scheduled["source"] == "schedule"


def test_a_one_time_task_is_completed_once_claimed(client):
    make_task(schedule = {"kind": "once", "at": "2026-10-14T09:00"})
    db.claim_next(ms("2026-10-14T09:00:00+00:00"), "worker", next_run_for)
    task = client.get("/api/scheduled-tasks/task-1").json()
    assert task["status"] == "completed"
    assert task["nextRunAt"] is None
    assert task["lastRun"]["status"] == "running"


def test_paused_and_future_tasks_are_not_claimed():
    make_task(id = "paused", enabled = False, nextRunAt = None)
    make_task(id = "later", nextRunAt = ms("2026-10-14T18:00:00+00:00"))
    assert db.claim_next(ms("2026-10-14T12:00:00+00:00"), "worker", next_run_for) is None
    assert db.next_due_at() == ms("2026-10-14T18:00:00+00:00")


def test_runs_whose_heartbeat_stopped_are_failed_and_live_ones_kept():
    # Staggered, so the most overdue is claimed first and each worker gets a known task.
    make_task(id = "unsettled", nextRunAt = ms("2026-10-14T08:00:00+00:00"))
    make_task(id = "crashed", nextRunAt = ms("2026-10-14T08:30:00+00:00"))
    make_task(id = "fresh", nextRunAt = ms("2026-10-14T08:45:00+00:00"))
    claim_at = ms("2026-10-14T12:00:00+00:00")
    # The sweeping worker's own run whose result it could not record, then a stopped Studio's.
    db.claim_next(claim_at, "me", next_run_for)
    db.claim_next(claim_at, "crashed", next_run_for)
    db.claim_next(claim_at + 60_000, "alive", next_run_for)

    assert db.recover_stale_runs(claim_at + 1) == 2

    runs = {task["id"]: task["lastRun"] for task in db.list_tasks()}
    for stopped in ("unsettled", "crashed"):
        assert runs[stopped]["status"] == "failed"
        assert runs[stopped]["errorCode"] == "interrupted"
    assert runs["fresh"]["status"] == "running"


def test_a_run_the_scheduler_could_not_settle_does_not_hold_its_task(monkeypatch):
    monkeypatch.setattr(scheduler_module, "job_accounts", lambda: [OWNER])
    scheduler = ScheduledTaskScheduler(SimpleNamespace(state = SimpleNamespace()))
    # A one-time task, so nothing else falls due while the clock is moved on.
    make_task(schedule = {"kind": "once", "at": "2026-10-14T09:00"}, nextRunAt = db.now_ms() - DAY)
    assert scheduler._claim() is not None
    # Never settled, as when finish_run fails: the run stays marked running.
    later = db.now_ms() + scheduler_module._STALE_RUN_MS + 1
    monkeypatch.setattr(db, "now_ms", lambda: later)

    assert scheduler._claim() is None

    last_run = db.get_task("task-1")["lastRun"]
    assert (last_run["status"], last_run["errorCode"]) == ("failed", "interrupted")
    assert db.request_run("task-1")["status"] == "queued"


def test_finishing_settles_only_the_owners_run_and_bounds_the_history(monkeypatch):
    monkeypatch.setattr(db, "RUN_HISTORY_LIMIT", 2)
    make_task()
    for _ in range(3):
        db.request_run("task-1")
        _task, run = db.claim_next(db.now_ms(), "worker", next_run_for)
        assert not db.finish_run(run["id"], "someone-else", status = "succeeded")
        assert db.finish_run(run["id"], "worker", status = "failed", error_code = "empty")
    assert [run["errorCode"] for run in db.list_runs("task-1")] == ["empty", "empty"]


def test_the_sweep_keeps_a_task_s_history_bounded_too(monkeypatch):
    monkeypatch.setattr(db, "RUN_HISTORY_LIMIT", 2)
    make_task()
    for _ in range(3):
        db.request_run("task-1")
        db.claim_next(db.now_ms(), "crashed", next_run_for)
        assert db.recover_stale_runs(db.now_ms() + scheduler_module._STALE_RUN_MS + 1) == 1
    assert [run["errorCode"] for run in db.list_runs("task-1")] == ["interrupted", "interrupted"]


def test_a_deleted_chat_unlinks_from_its_run():
    make_task()
    studio_db.upsert_chat_thread(
        {"id": "chat-1", "title": "Morning brief", "modelType": "base", "createdAt": 1}
    )
    db.request_run("task-1")
    _task, run = db.claim_next(db.now_ms(), "worker", next_run_for)
    db.finish_run(run["id"], "worker", status = "succeeded", thread_id = "chat-1")
    assert db.get_task("task-1")["lastRun"]["threadId"] == "chat-1"

    studio_db.delete_chat_threads(["chat-1"])

    assert db.get_task("task-1")["lastRun"]["threadId"] is None


def test_a_run_whose_chat_is_already_gone_still_settles():
    make_task()
    db.request_run("task-1")
    _task, run = db.claim_next(db.now_ms(), "worker", next_run_for)
    assert db.finish_run(run["id"], "worker", status = "succeeded", thread_id = "missing")
    assert db.get_task("task-1")["lastRun"]["threadId"] is None


# ── Routes ───────────────────────────────────────────────────────


def create(client, **overrides):
    body = {
        "title": "Morning brief",
        "prompt": "Write a haiku about mornings.",
        "schedule": {"kind": "daily", "time": "09:00"},
        "timezone": "UTC",
    }
    body.update(overrides)
    return client.post("/api/scheduled-tasks", json = body)


def test_a_task_keeps_its_model_and_reasoning_until_cleared(client):
    connection = "external::conn-1::gpt-5-mini"
    task = create(client, model = connection, reasoningEffort = "highest").json()
    assert (task["model"], task["reasoningEffort"]) == (connection, "highest")
    # A rename leaves them alone; an explicit null goes back to the loaded model and its default.
    renamed = client.patch(f"/api/scheduled-tasks/{task['id']}", json = {"title": "Renamed"}).json()
    assert (renamed["model"], renamed["reasoningEffort"]) == (connection, "highest")
    cleared = client.patch(
        f"/api/scheduled-tasks/{task['id']}", json = {"model": None, "reasoningEffort": None}
    ).json()
    assert (cleared["model"], cleared["reasoningEffort"]) == (None, None)
    assert create(client, reasoningEffort = "extreme").status_code == 422
    assert create(client).json()["model"] is None


def test_run_now_can_carry_a_one_time_permission_to_load_the_model(client):
    task = create(client, model = "unsloth/Qwen3-1.7B-GGUF").json()
    assert task["allowModelLoad"] is False
    run = client.post(f"/api/scheduled-tasks/{task['id']}/run", json = {"loadModel": True})
    assert run.status_code == 202
    assert db.list_runs(task["id"])[0]["allowModelLoad"] is True
    # Always allowing is the task's own setting, and survives an unrelated edit.
    allowed = client.patch(f"/api/scheduled-tasks/{task['id']}", json = {"allowModelLoad": True})
    assert allowed.json()["allowModelLoad"] is True
    renamed = client.patch(f"/api/scheduled-tasks/{task['id']}", json = {"title": "Renamed"})
    assert renamed.json()["allowModelLoad"] is True


def make_project(project_id = "proj-1"):
    return studio_db.upsert_chat_project(
        {"id": project_id, "name": "Reports", "instructions": "", "createdAt": 1, "updatedAt": 1}
    )


def test_a_task_keeps_its_chat_options_and_refuses_a_missing_project(client):
    make_project()
    task = create(client, newChatEachRun = True, archiveSuccessful = True, projectId = "proj-1").json()
    assert (task["newChatEachRun"], task["archiveSuccessful"], task["projectId"]) == (
        True,
        True,
        "proj-1",
    )
    assert create(client, projectId = "nowhere").status_code == 400
    cleared = client.patch(f"/api/scheduled-tasks/{task['id']}", json = {"projectId": None}).json()
    assert cleared["projectId"] is None
    # New tasks start a chat per run, as before this option existed.
    assert create(client).json()["newChatEachRun"] is True


def test_create_list_and_read(client, monkeypatch):
    monkeypatch.setattr(db, "now_ms", lambda: ms("2026-10-14T12:00:00+00:00"))
    response = create(client, title = "  Morning brief  ")
    assert response.status_code == 201
    task = response.json()
    assert task["title"] == "Morning brief"
    assert task["status"] == "active"
    assert task["notify"] == "all"
    assert task["named"] is True
    assert task["nextRunAt"] == ms("2026-10-15T09:00:00+00:00")
    assert task["lastRun"] is None
    assert "ownerSubject" not in task
    listed = client.get("/api/scheduled-tasks").json()
    assert listed["tasks"] == [task]
    # The server's clock, for a client to time its refreshes by.
    assert listed["now"] == ms("2026-10-14T12:00:00+00:00")
    assert client.get(f"/api/scheduled-tasks/{task['id']}").json() == task


@pytest.mark.parametrize(
    "overrides, status",
    [
        ({"title": "   "}, 422),
        ({"prompt": ""}, 422),
        ({"title": "x" * 121}, 422),
        ({"schedule": {"kind": "daily", "time": "25:00"}}, 422),
        ({"timezone": "Nowhere/Special"}, 400),
        ({"schedule": {"kind": "once", "at": "2020-01-01T09:00"}}, 400),
        ({"model": "   "}, 422),
        ({"notify": "sometimes"}, 422),
    ],
)
def test_invalid_tasks_are_refused(client, overrides, status):
    assert create(client, **overrides).status_code == status
    assert client.get("/api/scheduled-tasks").json()["tasks"] == []


def test_the_task_limit(client, monkeypatch):
    monkeypatch.setattr(db, "MAX_TASKS", 1)
    assert create(client).status_code == 201
    response = create(client)
    assert response.status_code == 409
    assert "up to 1 scheduled tasks" in response.json()["detail"]


def test_pause_clears_the_next_run_and_resume_schedules_from_now(client, monkeypatch):
    now = ms("2026-10-14T12:00:00+00:00")
    monkeypatch.setattr(db, "now_ms", lambda: now)
    task = create(client).json()

    paused = client.patch(f"/api/scheduled-tasks/{task['id']}", json = {"enabled": False}).json()
    assert paused["status"] == "paused"
    assert paused["nextRunAt"] is None

    now = ms("2026-10-20T12:00:00+00:00")
    resumed = client.patch(f"/api/scheduled-tasks/{task['id']}", json = {"enabled": True}).json()
    # Nothing replays for the days it was paused.
    assert resumed["status"] == "active"
    assert resumed["nextRunAt"] == ms("2026-10-21T09:00:00+00:00")


def test_a_rename_or_resent_timing_keeps_an_overdue_run_due(client):
    make_task(nextRunAt = ms("2026-10-14T09:00:00+00:00"))
    renamed = client.patch("/api/scheduled-tasks/task-1", json = {"title": "Evening brief"}).json()
    assert renamed["title"] == "Evening brief"
    assert renamed["nextRunAt"] == ms("2026-10-14T09:00:00+00:00")
    # The values it already has, as a client saving a whole form sends them.
    resent = client.patch(
        "/api/scheduled-tasks/task-1",
        json = {"enabled": True, "schedule": {"kind": "daily", "time": "09:00"}, "timezone": "UTC"},
    ).json()
    assert resent["nextRunAt"] == ms("2026-10-14T09:00:00+00:00")


def test_a_notification_setting_is_kept_and_keeps_an_overdue_run_due(client):
    make_task(nextRunAt = ms("2026-10-14T09:00:00+00:00"))
    muted = client.patch("/api/scheduled-tasks/task-1", json = {"notify": "none"}).json()
    assert muted["notify"] == "none"
    assert muted["nextRunAt"] == ms("2026-10-14T09:00:00+00:00")
    assert client.get("/api/scheduled-tasks/task-1").json()["notify"] == "none"
    assert create(client, notify = "failures").json()["notify"] == "failures"


def test_whether_a_task_was_named_is_kept(client):
    unnamed = create(client, title = "Write a haiku about mornings.", named = False).json()
    assert unnamed["named"] is False
    named = client.patch(f"/api/scheduled-tasks/{unnamed['id']}", json = {"named": True}).json()
    assert named["named"] is True
    assert client.get(f"/api/scheduled-tasks/{unnamed['id']}").json()["named"] is True


def test_a_database_from_before_notification_settings_gains_them():
    conn = sqlite3.connect(":memory:")
    try:
        conn.execute(
            """CREATE TABLE scheduled_tasks (
                id TEXT NOT NULL PRIMARY KEY, owner_subject TEXT NOT NULL, title TEXT NOT NULL,
                prompt TEXT NOT NULL, schedule_json TEXT NOT NULL, timezone TEXT NOT NULL,
                enabled INTEGER NOT NULL DEFAULT 1, next_run_at INTEGER,
                created_at INTEGER NOT NULL, updated_at INTEGER NOT NULL
            )"""
        )
        conn.execute(
            "INSERT INTO scheduled_tasks VALUES ('t', 'u', 'T', 'P', '{}', 'UTC', 1, NULL, 1, 1)"
        )
        studio_db._ensure_schema(conn)
        studio_db._ensure_schema(conn)
        # Tasks made before either setting notify of every run and count as named.
        assert conn.execute("SELECT notify, named FROM scheduled_tasks").fetchall() == [("all", 1)]
    finally:
        conn.close()


def test_a_new_schedule_moves_the_next_run(client, monkeypatch):
    monkeypatch.setattr(db, "now_ms", lambda: ms("2026-10-14T12:00:00+00:00"))
    make_task()
    updated = client.patch(
        "/api/scheduled-tasks/task-1",
        json = {"schedule": {"kind": "weekly", "time": "18:30", "days": [3]}},
    ).json()
    assert updated["schedule"] == {"kind": "weekly", "time": "18:30", "days": [3]}
    assert updated["nextRunAt"] == ms("2026-10-14T18:30:00+00:00")


def test_a_refused_update_changes_nothing(client):
    make_task()
    before = client.get("/api/scheduled-tasks/task-1").json()
    response = client.patch(
        "/api/scheduled-tasks/task-1",
        json = {"title": "Changed", "schedule": {"kind": "once", "at": "2020-01-01T09:00"}},
    )
    assert response.status_code == 400
    assert client.get("/api/scheduled-tasks/task-1").json() == before


def test_unknown_tasks_are_404(client):
    assert client.get("/api/scheduled-tasks/nope").status_code == 404
    assert client.patch("/api/scheduled-tasks/nope", json = {"title": "x"}).status_code == 404
    assert client.delete("/api/scheduled-tasks/nope").status_code == 404
    assert client.post("/api/scheduled-tasks/nope/run").status_code == 404


def test_run_now_queues_one_run_at_a_time_and_delete_removes_it(client):
    task = create(client).json()
    response = client.post(f"/api/scheduled-tasks/{task['id']}/run")
    assert response.status_code == 202
    assert response.json()["status"] == "queued"
    assert response.json()["source"] == "manual"
    assert client.post(f"/api/scheduled-tasks/{task['id']}/run").status_code == 409
    assert client.get("/api/scheduled-tasks/runs").json()["activeRuns"] == 1

    assert client.delete(f"/api/scheduled-tasks/{task['id']}").status_code == 204
    assert client.get("/api/scheduled-tasks").json()["tasks"] == []
    assert client.get("/api/scheduled-tasks/runs").json()["activeRuns"] == 0


def test_activity_lists_runs_finished_since_a_cursor(client):
    make_task(notify = "failures")
    db.request_run("task-1")
    _task, run = db.claim_next(db.now_ms(), "worker", next_run_for)
    first = client.get("/api/scheduled-tasks/runs").json()
    assert first["runs"] == [] and first["activeRuns"] == 1
    db.finish_run(run["id"], "worker", status = "failed", error_code = "no_model")

    later = client.get("/api/scheduled-tasks/runs", params = {"since": first["now"]}).json()
    # With what its task announces, for the client to decide on a toast.
    assert [(r["taskTitle"], r["taskNotify"], r["errorCode"]) for r in later["runs"]] == [
        ("Morning brief", "failures", "no_model")
    ]
    assert later["activeRuns"] == 0
    assert later["nextRunAt"] == ms("2026-10-14T09:00:00+00:00")
    assert later["more"] is False


def test_activity_pages_through_more_runs_than_one_answer_holds(client, monkeypatch):
    monkeypatch.setattr(db, "ACTIVITY_LIMIT", 2)
    # A second apart, so no two runs share the page boundary's millisecond.
    clock = iter(range(1_000, 100_000, 1_000))
    monkeypatch.setattr(db, "now_ms", lambda: next(clock))
    make_task()
    finished = []
    for _ in range(3):
        db.request_run("task-1")
        _task, run = db.claim_next(db.now_ms(), "worker", next_run_for)
        db.finish_run(run["id"], "worker", status = "failed", error_code = "empty")
        finished.append(run["id"])

    first = client.get("/api/scheduled-tasks/runs", params = {"since": 0}).json()
    assert [run["id"] for run in first["runs"]] == finished[:2]
    assert first["more"] is True
    rest = client.get(
        "/api/scheduled-tasks/runs", params = {"since": first["runs"][-1]["finishedAt"]}
    ).json()
    # The boundary run comes again, then the one the first answer had no room for.
    assert [run["id"] for run in rest["runs"]] == finished[1:]
    assert rest["more"] is False


# ── Running a task ───────────────────────────────────────────────


def claim_manual_run(**task_overrides):
    task = make_task(**task_overrides)
    db.request_run(task["id"])
    return db.claim_next(db.now_ms(), "worker-under-test", next_run_for)


def run_scheduler(
    task,
    run,
    answer = None,
    failure = None,
    monkeypatch = None,
):
    async def fake_generate(
        app,
        task,
        run,
        cancel_event,
        holder = None,
    ):
        if failure is not None:
            raise failure
        return answer

    monkeypatch.setattr(scheduler_module, "generate_answer", fake_generate)
    scheduler = ScheduledTaskScheduler(SimpleNamespace(state = SimpleNamespace()))
    scheduler.worker_id = "worker-under-test"
    asyncio.run(scheduler._execute(OWNER, task, run))


def test_a_run_saves_the_exchange_as_a_new_chat(monkeypatch):
    task, run = claim_manual_run()
    answer = Answer(content = "Dawn light on the sill.", reasoning = "A haiku.", model = "gemma-3")

    run_scheduler(task, run, answer = answer, monkeypatch = monkeypatch)

    last_run = db.get_task(task["id"])["lastRun"]
    assert last_run["status"] == "succeeded"
    assert last_run["model"] == "gemma-3"
    thread = studio_db.get_chat_thread(last_run["threadId"])
    assert thread["title"] == "Morning brief"
    assert thread["modelId"] == "gemma-3"
    user, assistant = studio_db.list_chat_messages(last_run["threadId"])
    assert user["role"] == "user" and user["parentId"] is None
    assert user["content"] == [{"type": "text", "text": "Write a haiku about mornings."}]
    assert assistant["parentId"] == user["id"]
    assert assistant["createdAt"] > user["createdAt"]
    assert assistant["content"] == [
        {"type": "reasoning", "text": "A haiku."},
        {"type": "text", "text": "Dawn light on the sill."},
    ]
    assert assistant["metadata"]["scheduledTask"] == {"taskId": task["id"], "runId": run["id"]}
    assert assistant["metadata"]["responseDetails"]["modelId"] == "gemma-3"
    # The reasoning header's "Worked for", in whole seconds.
    duration = assistant["metadata"]["reasoningDuration"]
    assert isinstance(duration, int) and assistant["metadata"]["reasoningDurations"] == [duration]


def test_a_run_chat_is_marked_as_the_tasks_and_keeps_the_mark(monkeypatch):
    task, run = claim_manual_run()
    run_scheduler(
        task, run, answer = Answer(content = "Hi.", reasoning = "", model = "m"), monkeypatch = monkeypatch
    )

    thread_id = db.get_task(task["id"])["lastRun"]["threadId"]
    thread = studio_db.get_chat_thread(thread_id)
    assert thread["scheduledTaskId"] == task["id"]
    user, _assistant = studio_db.list_chat_messages(thread_id)
    assert user["metadata"]["scheduledTask"] == {"taskId": task["id"], "runId": run["id"]}
    (row,) = [r for r in studio_db.list_chat_sidebar_rows()["rows"] if thread_id in r["threadIds"]]
    assert row["isScheduled"] is True

    # The chat page writes the whole record back without the field: the mark stays.
    studio_db.upsert_chat_thread({**thread, "title": "Renamed", "scheduledTaskId": None})
    assert studio_db.get_chat_thread(thread_id)["scheduledTaskId"] == task["id"]


def test_an_answer_without_reasoning_carries_no_reasoning_time(monkeypatch):
    task, run = claim_manual_run()
    answer = Answer(content = "Dawn light on the sill.", reasoning = "", model = "gemma-3")

    run_scheduler(task, run, answer = answer, monkeypatch = monkeypatch)

    thread_id = db.get_task(task["id"])["lastRun"]["threadId"]
    _user, assistant = studio_db.list_chat_messages(thread_id)
    assert assistant["content"] == [{"type": "text", "text": "Dawn light on the sill."}]
    assert "reasoningDuration" not in assistant["metadata"]


@pytest.mark.parametrize(
    "failure, code, error",
    [
        (RunFailed("no_model"), "no_model", None),
        (RunFailed(None, "context window exceeded"), None, "context window exceeded"),
        (RuntimeError("llama-server exited"), None, "llama-server exited"),
    ],
)
def test_a_failed_run_records_why_and_writes_no_chat(monkeypatch, failure, code, error):
    task, run = claim_manual_run()

    run_scheduler(task, run, failure = failure, monkeypatch = monkeypatch)

    last_run = db.get_task(task["id"])["lastRun"]
    assert (last_run["status"], last_run["errorCode"], last_run["error"]) == (
        "failed",
        code,
        error,
    )
    assert studio_db.list_chat_threads() == []


def test_a_task_deleted_mid_run_writes_no_chat(monkeypatch):
    task, run = claim_manual_run()

    async def delete_then_answer(
        app,
        task,
        run,
        cancel_event,
        holder = None,
    ):
        await asyncio.to_thread(db.delete_task, task["id"])
        return Answer(content = "Too late.", reasoning = "", model = "gemma-3")

    monkeypatch.setattr(scheduler_module, "generate_answer", delete_then_answer)
    scheduler = ScheduledTaskScheduler(SimpleNamespace(state = SimpleNamespace()))
    scheduler.worker_id = "worker-under-test"
    asyncio.run(scheduler._execute(OWNER, task, run))

    assert studio_db.list_chat_threads() == []


def recording_generate(asked):
    async def generate(
        app,
        task,
        run,
        cancel_event,
        holder = None,
    ):
        asked.append(run["id"])
        return Answer(content = "Done.", reasoning = "", model = "gemma-3")

    return generate


def test_a_task_deleted_before_its_claimed_run_starts_is_not_run(monkeypatch):
    task, run = claim_manual_run()
    asked = []
    monkeypatch.setattr(scheduler_module, "generate_answer", recording_generate(asked))
    # After the claim, before the run is active: no cancel can reach it yet.
    db.delete_task(task["id"])
    scheduler = ScheduledTaskScheduler(SimpleNamespace(state = SimpleNamespace()))
    scheduler.worker_id = "worker-under-test"
    asyncio.run(scheduler._execute(OWNER, task, run))

    assert asked == []
    assert studio_db.list_chat_threads() == []


def test_a_run_claimed_for_an_account_disabled_since_is_not_run(monkeypatch):
    task, run = claim_manual_run()
    asked = []
    monkeypatch.setattr(scheduler_module, "generate_answer", recording_generate(asked))
    # Disabled accounts are out of job_accounts().
    monkeypatch.setattr(scheduler_module, "job_accounts", lambda: [])
    scheduler = ScheduledTaskScheduler(SimpleNamespace(state = SimpleNamespace()))
    scheduler.worker_id = "worker-under-test"
    asyncio.run(scheduler._execute(OWNER, task, run))

    assert asked == []
    last_run = db.get_task(task["id"])["lastRun"]
    assert (last_run["status"], last_run["errorCode"]) == ("failed", "cancelled")


def test_the_sweep_removes_the_chat_of_a_run_that_stopped_before_settling():
    task, run = claim_manual_run()
    # Studio stopped after the chat was written, before the run settled.
    answer = Answer(content = "Written, never settled.", reasoning = "", model = "gemma-3")
    scheduler_module.save_chat(task, run, answer, 1_000, 2_000)
    assert len(studio_db.list_chat_threads()) == 1

    assert db.recover_stale_runs(db.now_ms() + scheduler_module._STALE_RUN_MS + 1) == 1

    assert studio_db.list_chat_threads() == []
    last_run = db.get_task(task["id"])["lastRun"]
    assert (last_run["status"], last_run["errorCode"]) == ("failed", "interrupted")


def refuse_chat_deletion(conn, ids):
    raise sqlite3.OperationalError("disk I/O error")


def test_a_sweep_that_cannot_remove_a_chat_records_nothing(monkeypatch):
    task, run = claim_manual_run()
    answer = Answer(content = "Written, never settled.", reasoning = "", model = "gemma-3")
    scheduler_module.save_chat(task, run, answer, 1_000, 2_000)
    monkeypatch.setattr(db, "delete_chat_threads_in", refuse_chat_deletion)

    with pytest.raises(sqlite3.OperationalError):
        db.recover_stale_runs(db.now_ms() + scheduler_module._STALE_RUN_MS + 1)

    # One transaction: the run is not left failed with its chat still there, out of the sweep's
    # reach, but found again, chat and all, by the next sweep.
    assert db.get_task(task["id"])["lastRun"]["status"] == "running"
    assert len(studio_db.list_chat_threads()) == 1


def test_a_run_recorded_as_failed_takes_its_chat_in_the_same_transaction(monkeypatch):
    task, run = claim_manual_run()
    answer = Answer(content = "Written, then failed.", reasoning = "", model = "gemma-3")
    scheduler_module.save_chat(task, run, answer, 1_000, 2_000)
    delete_chat_threads_in = db.delete_chat_threads_in
    monkeypatch.setattr(db, "delete_chat_threads_in", refuse_chat_deletion)
    with pytest.raises(sqlite3.OperationalError):
        db.finish_run(run["id"], "worker-under-test", status = "failed", discard_chat = True)
    assert db.get_task(task["id"])["lastRun"]["status"] == "running"
    assert len(studio_db.list_chat_threads()) == 1

    monkeypatch.setattr(db, "delete_chat_threads_in", delete_chat_threads_in)
    assert db.finish_run(run["id"], "worker-under-test", status = "failed", discard_chat = True)
    assert db.get_task(task["id"])["lastRun"]["status"] == "failed"
    assert studio_db.list_chat_threads() == []


def test_a_run_that_fails_after_writing_its_chat_leaves_none_however_soon_studio_stops(monkeypatch):
    task, run = claim_manual_run()
    save_chat = scheduler_module.save_chat

    def save_then_fail(*args):
        save_chat(*args)
        raise sqlite3.OperationalError("disk I/O error")

    monkeypatch.setattr(scheduler_module, "generate_answer", answer_with("Half saved."))
    monkeypatch.setattr(scheduler_module, "save_chat", save_then_fail)
    # As if Studio stopped the moment the run was recorded: nothing after that removes the chat.
    monkeypatch.setattr(scheduler_module, "_discard_chat", lambda thread_id: None)
    scheduler = ScheduledTaskScheduler(SimpleNamespace(state = SimpleNamespace()))
    scheduler.worker_id = "worker-under-test"
    asyncio.run(scheduler._execute(OWNER, task, run))

    assert db.get_task(task["id"])["lastRun"]["status"] == "failed"
    assert studio_db.list_chat_threads() == []


def test_deleting_a_task_removes_only_the_chat_its_running_run_may_be_writing():
    task, run = claim_manual_run()
    kept = scheduler_module.save_chat(
        task, run, Answer(content = "Finished.", reasoning = "", model = "gemma-3"), 1_000, 2_000
    )
    assert db.finish_run(run["id"], "worker-under-test", status = "succeeded", thread_id = kept)
    db.request_run(task["id"])
    _task, running = db.claim_next(db.now_ms(), "worker-under-test", next_run_for)
    scheduler_module.save_chat(
        task, running, Answer(content = "Mid-run.", reasoning = "", model = "gemma-3"), 3_000, 4_000
    )
    assert len(studio_db.list_chat_threads()) == 2

    assert db.delete_task(task["id"])

    # In the deletion's own transaction, so no crash can leave it to a run that no longer exists.
    assert [thread["id"] for thread in studio_db.list_chat_threads()] == [kept]


def test_a_task_deleted_while_its_chat_is_saved_keeps_no_chat(monkeypatch):
    task, run = claim_manual_run()

    async def answer(
        app,
        task,
        run,
        cancel_event,
        holder = None,
    ):
        return Answer(content = "Just in time?", reasoning = "", model = "gemma-3")

    save_chat = scheduler_module.save_chat

    def save_then_delete(*args):
        thread_id = save_chat(*args)
        # Past the existence check and the save, but before the run settles.
        db.delete_task(task["id"])
        return thread_id

    monkeypatch.setattr(scheduler_module, "generate_answer", answer)
    monkeypatch.setattr(scheduler_module, "save_chat", save_then_delete)
    scheduler = ScheduledTaskScheduler(SimpleNamespace(state = SimpleNamespace()))
    scheduler.worker_id = "worker-under-test"
    asyncio.run(scheduler._execute(OWNER, task, run))

    assert db.get_task(task["id"]) is None
    assert studio_db.list_chat_threads() == []


def answer_with(content):
    async def generate(
        app,
        task,
        run,
        cancel_event,
        holder = None,
    ):
        return Answer(content = content, reasoning = "", model = "gemma-3")

    return generate


def test_settling_a_run_is_tried_again_over_a_locked_database(monkeypatch):
    task, run = claim_manual_run()
    finish_run = db.finish_run
    attempts = []

    def locked_once(*args, **kwargs):
        attempts.append(args)
        if len(attempts) == 1:
            raise sqlite3.OperationalError("database is locked")
        return finish_run(*args, **kwargs)

    monkeypatch.setattr(scheduler_module, "generate_answer", answer_with("Done."))
    monkeypatch.setattr(scheduler_module, "_SETTLE_RETRY_SECONDS", 0)
    monkeypatch.setattr(db, "finish_run", locked_once)
    scheduler = ScheduledTaskScheduler(SimpleNamespace(state = SimpleNamespace()))
    scheduler.worker_id = "worker-under-test"
    asyncio.run(scheduler._execute(OWNER, task, run))

    assert len(attempts) == 2
    last_run = db.get_task(task["id"])["lastRun"]
    assert last_run["status"] == "succeeded"
    assert [thread["id"] for thread in studio_db.list_chat_threads()] == [last_run["threadId"]]


def test_a_chat_whose_run_cannot_be_settled_is_removed(monkeypatch):
    task, run = claim_manual_run()

    def locked(*args, **kwargs):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(scheduler_module, "generate_answer", answer_with("Stranded."))
    monkeypatch.setattr(scheduler_module, "_SETTLE_RETRY_SECONDS", 0)
    monkeypatch.setattr(db, "finish_run", locked)
    scheduler = ScheduledTaskScheduler(SimpleNamespace(state = SimpleNamespace()))
    scheduler.worker_id = "worker-under-test"
    asyncio.run(scheduler._execute(OWNER, task, run))

    # No chat without a run recorded as succeeded; the stale-run sweep settles this one later.
    assert studio_db.list_chat_threads() == []
    assert db.get_task(task["id"])["lastRun"]["status"] == "running"


class FakeProducer:
    def __init__(
        self,
        response = None,
        error = None,
        hang = False,
    ):
        self.response, self.error, self.hang = response, error, hang
        self.payload = self.request = None
        self.cancelled = False

    async def __call__(self, payload, request, current_subject, *, cancel_on_disconnect):
        self.payload, self.request = payload, request
        assert cancel_on_disconnect is False
        if self.hang:
            try:
                await asyncio.sleep(3600)
            except asyncio.CancelledError:
                self.cancelled = True
                raise
        if self.error is not None:
            raise self.error
        return self.response


def completion(
    content,
    reasoning = None,
    model = "gemma-3",
):
    message = {"role": "assistant", "content": content}
    if reasoning is not None:
        message["reasoning_content"] = reasoning
    return JSONResponse(
        {"model": model, "choices": [{"index": 0, "message": message, "finish_reason": "stop"}]}
    )


class LoadsItsModel(FakeProducer):
    """A producer whose request loads the model it names, as the backend's switch does."""

    def __init__(self, monkeypatch, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.monkeypatch = monkeypatch

    async def __call__(self, payload, request, current_subject, *, cancel_on_disconnect):
        self.monkeypatch.setattr(
            inference_routes, "_loaded_satisfies", lambda model: model == payload.model
        )
        return await super().__call__(
            payload, request, current_subject, cancel_on_disconnect = cancel_on_disconnect
        )


def generate(
    monkeypatch,
    producer,
    task = None,
    cancel_event = None,
):
    monkeypatch.setattr(inference_routes, "produce_openai_chat_completions", producer)
    task = task or make_task(timezone = "Europe/Istanbul")
    run = {"id": "run-1"}
    app = SimpleNamespace(state = SimpleNamespace())
    return asyncio.run(generate_answer(app, task, run, cancel_event or threading.Event()))


def test_a_run_asks_the_loaded_model_for_text_only(monkeypatch):
    producer = FakeProducer(completion("Dawn light.", reasoning = "Short."))
    monkeypatch.setattr(db, "now_ms", lambda: ms("2026-10-14T06:00:00+00:00"))

    answer = generate(monkeypatch, producer)

    assert answer == Answer(content = "Dawn light.", reasoning = "Short.", model = "gemma-3")
    payload = producer.payload
    # No model named: the loaded one answers and nothing is switched.
    assert "model" not in payload.model_fields_set
    assert payload.stream is False
    assert payload.tool_choice == "none" and payload.enabled_tools == []
    assert payload.cancel_id == "run-1"
    system, user = payload.messages
    assert system.role == "system"
    # The time in the task's own zone: 06:00 UTC is 09:00 in Istanbul.
    assert "Wednesday, October 14, 2026 at 09:00 (Europe/Istanbul)" in system.content
    assert user.role == "user" and user.content == "Write a haiku about mornings."
    request = producer.request
    assert request.headers["x-unsloth-timezone"] == "Europe/Istanbul"
    assert request.headers["x-unsloth-require-text"] == "1"
    assert request.state.api_monitor_via_api_key is False
    assert request.state.keyless_api_admitted is False


def test_a_run_with_no_model_loaded_says_so(monkeypatch):
    producer = FakeProducer(error = HTTPException(status_code = 400, detail = "No model loaded."))
    monkeypatch.setattr(inference_routes, "_loaded_slot_ident", lambda: None)
    with pytest.raises(RunFailed) as failure:
        generate(monkeypatch, producer)
    assert failure.value.code == "no_model"


def test_a_refusal_from_a_loaded_model_keeps_its_reason(monkeypatch):
    producer = FakeProducer(
        error = HTTPException(status_code = 400, detail = {"error": {"message": "Too long."}})
    )
    monkeypatch.setattr(inference_routes, "_loaded_slot_ident", lambda: "gemma-3")
    with pytest.raises(RunFailed) as failure:
        generate(monkeypatch, producer)
    assert (failure.value.code, failure.value.message) == (None, "Too long.")


def test_a_task_can_pin_the_loaded_model_and_its_reasoning(monkeypatch):
    monkeypatch.setattr(inference_routes, "_loaded_satisfies", lambda model: True)
    producer = FakeProducer(completion("Dawn light."))
    generate(
        monkeypatch, producer, make_task(model = "unsloth/Qwen3-1.7B-GGUF", reasoningEffort = "lowest")
    )
    payload = producer.payload
    assert payload.model == "unsloth/Qwen3-1.7B-GGUF"
    assert (payload.enable_thinking, payload.reasoning_effort) == (True, "low")
    assert payload.provider_id is None


def test_reasoning_off_turns_thinking_off_on_a_local_model(monkeypatch):
    producer = FakeProducer(completion("Dawn light."))
    generate(monkeypatch, producer, make_task(reasoningEffort = "off"))
    assert (producer.payload.enable_thinking, producer.payload.reasoning_effort) == (False, None)
    assert "model" not in producer.payload.model_fields_set


def loaded_gguf(monkeypatch, **reasoning):
    backend = SimpleNamespace(
        is_loaded = True,
        supports_reasoning = True,
        reasoning_style = "enable_thinking",
        reasoning_effort_levels = [],
        reasoning_always_on = False,
    )
    for key, value in reasoning.items():
        setattr(backend, key, value)
    monkeypatch.setattr(inference_routes, "get_llama_cpp_backend", lambda: backend)


@pytest.mark.parametrize(
    "reasoning, effort, expected",
    [
        # An effort ladder: its own ends, not a fixed low and high.
        (
            {
                "reasoning_style": "reasoning_effort",
                "reasoning_effort_levels": ["minimal", "low", "medium", "high", "xhigh"],
            },
            "highest",
            {"reasoning_effort": "xhigh"},
        ),
        (
            {
                "reasoning_style": "reasoning_effort",
                "reasoning_effort_levels": ["minimal", "low", "medium", "high", "xhigh"],
            },
            "lowest",
            {"reasoning_effort": "minimal"},
        ),
        # An empty list is the plain ladder.
        ({"reasoning_style": "reasoning_effort"}, "lowest", {"reasoning_effort": "low"}),
        # gpt-oss style has no off: the model's default stands.
        ({"reasoning_style": "reasoning_effort"}, "off", {}),
        # On/off only: both ends mean on.
        ({}, "lowest", {"enable_thinking": True}),
        ({}, "off", {"enable_thinking": False}),
        ({"reasoning_always_on": True}, "off", {}),
        (
            {
                "reasoning_style": "enable_thinking_effort",
                "reasoning_effort_levels": ["low", "high", "max"],
            },
            "highest",
            {"enable_thinking": True, "reasoning_effort": "max"},
        ),
        # No thinking at all: nothing sent.
        ({"supports_reasoning": False}, "highest", {}),
    ],
)
def test_thinking_effort_resolves_to_the_loaded_model(monkeypatch, reasoning, effort, expected):
    loaded_gguf(monkeypatch, **reasoning)
    assert scheduler_module._reasoning_fields(effort, False, on_loaded_model = True) == expected


def test_thinking_effort_for_a_model_not_yet_loaded_or_a_connection(monkeypatch):
    loaded_gguf(monkeypatch, supports_reasoning = False)
    # The run loads its model, whose levels are unknown before the request: the shared ends.
    assert scheduler_module._reasoning_fields("highest", False, on_loaded_model = False) == {
        "enable_thinking": True,
        "reasoning_effort": "high",
    }
    assert scheduler_module._reasoning_fields("off", True, on_loaded_model = True) == {
        "reasoning_effort": "none"
    }
    assert scheduler_module._reasoning_fields(None, False, on_loaded_model = True) == {}


def test_a_connection_task_names_its_level_and_a_local_one_resolves_it(monkeypatch):
    loaded_gguf(monkeypatch, reasoning_style = "reasoning_effort")
    assert scheduler_module._reasoning_fields("xhigh", True, on_loaded_model = False) == {
        "reasoning_effort": "xhigh"
    }
    assert scheduler_module._reasoning_fields("medium", False, on_loaded_model = True) == {
        "enable_thinking": True,
        "reasoning_effort": "medium",
    }


def test_a_chosen_model_that_is_not_loaded_fails_rather_than_switching(monkeypatch):
    from utils import openai_auto_switch_settings

    monkeypatch.setattr(inference_routes, "_loaded_satisfies", lambda model: False)
    monkeypatch.setattr(inference_routes, "_loaded_identity_satisfies", lambda model: False)
    monkeypatch.setattr(
        openai_auto_switch_settings, "get_openai_auto_switch_enabled", lambda: False
    )
    producer = FakeProducer(completion("Dawn light."))
    with pytest.raises(RunFailed) as failure:
        generate(monkeypatch, producer, make_task(model = "unsloth/Qwen3-1.7B-GGUF"))
    assert failure.value.code == "model_not_loaded"
    assert producer.payload is None

    # With auto-switch on, the request names it and the backend loads it.
    monkeypatch.setattr(openai_auto_switch_settings, "get_openai_auto_switch_enabled", lambda: True)
    db.delete_task("task-1")
    producer = LoadsItsModel(monkeypatch, completion("Dawn light."))
    generate(monkeypatch, producer, make_task(model = "unsloth/Qwen3-1.7B-GGUF"))
    assert producer.payload.model == "unsloth/Qwen3-1.7B-GGUF"


def test_a_task_allowed_to_load_its_model_lets_only_its_own_request_switch(monkeypatch):
    from utils import openai_auto_switch_settings

    monkeypatch.setattr(inference_routes, "_loaded_satisfies", lambda model: False)
    monkeypatch.setattr(inference_routes, "_loaded_identity_satisfies", lambda model: False)
    monkeypatch.setattr(
        openai_auto_switch_settings, "get_openai_auto_switch_enabled", lambda: False
    )
    producer = LoadsItsModel(monkeypatch, completion("Dawn light."))
    generate(
        monkeypatch,
        producer,
        make_task(model = "unsloth/Qwen3-1.7B-GGUF", allowModelLoad = True),
    )
    assert producer.payload.model == "unsloth/Qwen3-1.7B-GGUF"
    # The opt-in rides on this run's in-process request, where no client can put it.
    assert inference_routes._model_switch_allowed_for_request(producer.request)


def test_a_one_time_yes_lets_that_run_alone_load_the_model(monkeypatch):
    from utils import openai_auto_switch_settings

    monkeypatch.setattr(inference_routes, "_loaded_satisfies", lambda model: False)
    monkeypatch.setattr(inference_routes, "_loaded_identity_satisfies", lambda model: False)
    monkeypatch.setattr(
        openai_auto_switch_settings, "get_openai_auto_switch_enabled", lambda: False
    )
    monkeypatch.setattr(
        inference_routes,
        "produce_openai_chat_completions",
        LoadsItsModel(monkeypatch, completion("Ok.")),
    )
    task = make_task(model = "unsloth/Qwen3-1.7B-GGUF")
    app = SimpleNamespace(state = SimpleNamespace())
    granted = {"id": "run-1", "allowModelLoad": True}
    asyncio.run(generate_answer(app, task, granted, threading.Event()))
    # Switched back since.
    monkeypatch.setattr(inference_routes, "_loaded_satisfies", lambda model: False)
    with pytest.raises(RunFailed) as failure:
        asyncio.run(generate_answer(app, task, {"id": "run-2"}, threading.Event()))
    assert failure.value.code == "model_not_loaded"


def test_a_permitted_load_that_still_finds_no_model_says_the_load_failed(monkeypatch):
    from utils import openai_auto_switch_settings

    monkeypatch.setattr(inference_routes, "_loaded_satisfies", lambda model: False)
    monkeypatch.setattr(inference_routes, "_loaded_identity_satisfies", lambda model: False)
    monkeypatch.setattr(inference_routes, "_loaded_slot_ident", lambda: None)
    monkeypatch.setattr(
        openai_auto_switch_settings, "get_openai_auto_switch_enabled", lambda: False
    )
    producer = FakeProducer(error = HTTPException(status_code = 400, detail = "No model loaded."))
    with pytest.raises(RunFailed) as failure:
        generate(
            monkeypatch,
            producer,
            make_task(model = "unsloth/Qwen3-1.7B-GGUF", allowModelLoad = True),
        )
    # Not "load a model in Chat": the run was let load one and could not.
    assert failure.value.code == "model_load_failed"


class SwapWorld:
    """The model slot around one run: what is loaded, and what restoring did to it."""

    EARLIER = ("/models/gemma.gguf", "Q4_K_M", "unsloth/gemma-3-GGUF")
    TASK_MODEL = "unsloth/Qwen3-1.7B-GGUF"

    def __init__(
        self,
        monkeypatch,
        *,
        idle = True,
        earlier = EARLIER,
    ):
        from core.inference import llama_keepwarm
        from utils import openai_auto_switch_settings

        self.loaded = earlier
        self.unloaded, self.stashed, self.reloads = [], [], []
        self.idle = idle
        monkeypatch.setattr(llama_keepwarm, "idle_for", lambda seconds: self.idle)
        monkeypatch.setattr(llama_keepwarm, "loaded_identity", lambda: self.loaded)
        monkeypatch.setattr(llama_keepwarm, "stash_for_reload", self.stashed.append)
        monkeypatch.setattr(
            inference_routes,
            "_loaded_satisfies",
            lambda model: self.loaded is not None and self.loaded[2] == model,
        )
        monkeypatch.setattr(inference_routes, "_loaded_identity_satisfies", lambda model: False)
        monkeypatch.setattr(inference_routes, "_resident_npu_model", lambda: None)
        self.training = False
        monkeypatch.setattr(inference_routes, "_training_is_active", lambda: self.training)
        monkeypatch.setattr(
            inference_routes,
            "get_inference_backend",
            lambda: SimpleNamespace(active_model_name = None),
        )
        monkeypatch.setattr(
            openai_auto_switch_settings, "get_openai_auto_switch_enabled", lambda: False
        )

        async def unload(request, subject):
            self.unloaded.append(request.model_path)
            self.loaded = None

        async def reload(model, request, subject, **kwargs):
            self.reloads.append(
                (model, inference_routes._model_switch_allowed_for_request(request))
            )
            self.loaded = self.stashed[-1]

        monkeypatch.setattr(inference_routes, "_unload_model_impl", unload)
        monkeypatch.setattr(inference_routes, "_maybe_auto_switch_model", reload)

    def producer(self):
        world = self

        class LoadsTheTaskModel(FakeProducer):
            async def __call__(self, payload, request, current_subject, *, cancel_on_disconnect):
                world.loaded = ("/models/qwen.gguf", "Q4_K_M", world.TASK_MODEL)
                return await super().__call__(
                    payload, request, current_subject, cancel_on_disconnect = cancel_on_disconnect
                )

        return LoadsTheTaskModel(completion("Dawn light."))

    def run(
        self,
        monkeypatch,
        run,
        producer = None,
    ):
        producer = producer or self.producer()
        monkeypatch.setattr(inference_routes, "produce_openai_chat_completions", producer)
        task = make_task(model = self.TASK_MODEL, allowModelLoad = True)
        app = SimpleNamespace(state = SimpleNamespace())
        asyncio.run(generate_answer(app, task, run, threading.Event()))
        return producer


def test_a_scheduled_run_leaves_the_model_alone_while_studio_is_in_use(monkeypatch):
    world = SwapWorld(monkeypatch, idle = False)
    producer = world.producer()
    with pytest.raises(RunFailed) as failure:
        world.run(monkeypatch, {"id": "run-1", "source": "schedule"}, producer)
    assert failure.value.code == "model_busy"
    assert producer.payload is None
    assert world.loaded == SwapWorld.EARLIER


def test_a_scheduled_run_while_away_switches_back_to_the_earlier_model(monkeypatch):
    world = SwapWorld(monkeypatch)
    world.run(monkeypatch, {"id": "run-1", "source": "schedule"})
    assert world.unloaded == ["/models/qwen.gguf"]
    assert world.stashed == [SwapWorld.EARLIER]
    # Brought back through the idle-reload path, so its saved launch settings come with it.
    assert world.reloads == [(inference_routes._RELOAD_ONLY_MODEL, True)]
    assert world.loaded == SwapWorld.EARLIER


def test_a_run_that_found_nothing_loaded_leaves_nothing_loaded(monkeypatch):
    world = SwapWorld(monkeypatch, earlier = None)
    world.run(monkeypatch, {"id": "run-1", "source": "schedule"})
    assert world.unloaded == ["/models/qwen.gguf"]
    assert world.reloads == [] and world.loaded is None


def test_a_run_now_swaps_even_while_studio_is_in_use_and_still_switches_back(monkeypatch):
    world = SwapWorld(monkeypatch, idle = False)
    # The user pressed Run now: they are there. Nothing restores while they are mid-request.
    world.run(monkeypatch, {"id": "run-1", "source": "manual"})
    assert world.loaded[2] == SwapWorld.TASK_MODEL and world.unloaded == []
    db.delete_task("task-1")
    world = SwapWorld(monkeypatch, idle = True)
    world.run(monkeypatch, {"id": "run-2", "source": "manual"})
    assert world.loaded == SwapWorld.EARLIER


def test_restoring_leaves_alone_a_model_the_user_loaded_meanwhile(monkeypatch):
    from core.inference import llama_keepwarm

    world = SwapWorld(monkeypatch)
    picked = ("/models/llama.gguf", "Q8_0", "unsloth/Llama-3.2-3B-GGUF")
    ended = llama_keepwarm.end_in_process_request

    def user_picks_another(scope, answered):
        ended(scope, answered)
        world.loaded = picked

    monkeypatch.setattr(llama_keepwarm, "end_in_process_request", user_picks_another)
    world.run(monkeypatch, {"id": "run-1", "source": "schedule"})
    assert world.unloaded == [] and world.reloads == [] and world.loaded == picked


def test_a_header_cannot_turn_on_the_switch_opt_in():
    from starlette.requests import Request

    request = Request(
        {"type": "http", "headers": [(b"unsloth.allow_model_switch", b"1")], "path": "/"}
    )
    assert not inference_routes._model_switch_allowed_for_request(request)
    inference_routes.allow_model_switch_for_request(request.scope)
    assert inference_routes._model_switch_allowed_for_request(request)


def test_a_scheduled_run_leaves_the_model_alone_while_training(monkeypatch):
    world = SwapWorld(monkeypatch)
    world.training = True
    producer = world.producer()
    with pytest.raises(RunFailed) as failure:
        world.run(monkeypatch, {"id": "run-1", "source": "schedule"}, producer)
    assert failure.value.code == "model_busy"
    assert producer.payload is None and world.loaded == SwapWorld.EARLIER


def test_a_scheduled_run_leaves_alone_another_accounts_model(monkeypatch):
    from hub.services.models import account_access

    world = SwapWorld(monkeypatch)
    monkeypatch.setattr(inference_routes, "_loaded_slot_ident", lambda: "gemma")
    monkeypatch.setattr(account_access, "resident_hidden", lambda modality, reference: True)
    with pytest.raises(RunFailed) as failure:
        world.run(monkeypatch, {"id": "run-1", "source": "schedule"})
    assert failure.value.code == "model_busy"
    assert world.loaded == SwapWorld.EARLIER


def keepwarm_module():
    from core.inference import llama_keepwarm
    return llama_keepwarm


def run_held(world, monkeypatch, task, run, holder):
    monkeypatch.setattr(inference_routes, "produce_openai_chat_completions", world.producer())
    app = SimpleNamespace(state = SimpleNamespace())
    return asyncio.run(generate_answer(app, task, run, threading.Event(), holder))


def test_back_to_back_runs_on_one_model_load_it_once_and_switch_back_once(monkeypatch):
    world = SwapWorld(monkeypatch)
    holder = scheduler_module.SwapHolder()
    first = make_task(model = SwapWorld.TASK_MODEL, allowModelLoad = True)
    second = make_task(id = "task-2", model = SwapWorld.TASK_MODEL, allowModelLoad = True)
    run_held(world, monkeypatch, first, {"id": "run-1", "source": "schedule"}, holder)
    # Left loaded for whatever runs next.
    assert world.loaded[2] == SwapWorld.TASK_MODEL and world.reloads == []
    run_held(world, monkeypatch, second, {"id": "run-2", "source": "schedule"}, holder)
    assert world.unloaded == [] and holder.pending.previous == SwapWorld.EARLIER
    asyncio.run(scheduler_module._restore_after_swap(holder.pending, None))
    assert world.unloaded == ["/models/qwen.gguf"]
    assert world.reloads == [(inference_routes._RELOAD_ONLY_MODEL, True)]
    assert world.loaded == SwapWorld.EARLIER


def test_a_run_on_the_loaded_model_gets_the_users_model_back_first(monkeypatch):
    world = SwapWorld(monkeypatch)
    holder = scheduler_module.SwapHolder()
    pinned = make_task(model = SwapWorld.TASK_MODEL, allowModelLoad = True)
    run_held(world, monkeypatch, pinned, {"id": "run-1", "source": "schedule"}, holder)
    unpinned = make_task(id = "task-2")
    producer = FakeProducer(completion("Ok."))
    monkeypatch.setattr(inference_routes, "produce_openai_chat_completions", producer)
    app = SimpleNamespace(state = SimpleNamespace())
    run = {"id": "run-2", "source": "schedule"}
    asyncio.run(generate_answer(app, unpinned, run, threading.Event(), holder))
    assert world.loaded == SwapWorld.EARLIER and holder.pending is None
    assert "model" not in producer.payload.model_fields_set


def test_the_scheduler_switches_back_after_a_timed_out_run_but_not_at_shutdown(monkeypatch):
    world = SwapWorld(monkeypatch)
    world.loaded = ("/models/qwen.gguf", "Q4_K_M", SwapWorld.TASK_MODEL)
    swap = scheduler_module._Swap(
        SwapWorld.EARLIER,
        model = SwapWorld.TASK_MODEL,
        subject = "unsloth",
        task_id = "task-1",
        attended = keepwarm_module().attended_stamp(),
    )
    scheduler = ScheduledTaskScheduler(SimpleNamespace(state = SimpleNamespace()))
    scheduler._swap.pending = swap
    scheduler._stopping = True
    asyncio.run(scheduler._restore_swap())
    assert world.loaded[2] == SwapWorld.TASK_MODEL and scheduler._swap.pending is None
    scheduler._stopping = False
    scheduler._swap.pending = swap
    asyncio.run(scheduler._restore_swap())
    assert world.loaded == SwapWorld.EARLIER


def test_switching_back_leaves_a_model_someone_used_since_the_swap(monkeypatch):
    keepwarm = keepwarm_module()
    world = SwapWorld(monkeypatch)
    holder = scheduler_module.SwapHolder()
    task = make_task(model = SwapWorld.TASK_MODEL, allowModelLoad = True)
    run_held(world, monkeypatch, task, {"id": "run-1", "source": "schedule"}, holder)
    # Someone's own request ends on the task's model.
    keepwarm._note_start()
    keepwarm._note_end()
    asyncio.run(scheduler_module._restore_after_swap(holder.pending, None))
    assert world.loaded[2] == SwapWorld.TASK_MODEL and world.reloads == []


def test_studios_own_runs_do_not_count_as_someone_using_it(monkeypatch):
    import time

    keepwarm = keepwarm_module()
    monkeypatch.setattr(keepwarm, "_last_attended", time.monotonic() - 3600)
    assert keepwarm.idle_for(300)
    with keepwarm.unattended_activity():
        keepwarm._note_start()
        keepwarm._note_end()
        keepwarm._note_activity()
    # Still away: only the run was active, though it keeps the model warm.
    assert keepwarm.idle_for(300)
    assert time.monotonic() - keepwarm._last_active < 60
    keepwarm._note_start()
    assert not keepwarm.idle_for(0)
    keepwarm._note_end()
    assert not keepwarm.idle_for(300)


def test_a_run_answered_by_another_model_than_its_own_fails(monkeypatch):
    from utils import openai_auto_switch_settings

    # The task's model is gone from disk: the request falls through to the loaded model.
    monkeypatch.setattr(inference_routes, "_loaded_satisfies", lambda model: False)
    monkeypatch.setattr(inference_routes, "_loaded_identity_satisfies", lambda model: False)
    monkeypatch.setattr(
        openai_auto_switch_settings, "get_openai_auto_switch_enabled", lambda: False
    )
    with pytest.raises(RunFailed) as failure:
        generate(
            monkeypatch,
            FakeProducer(completion("From gemma.")),
            make_task(model = "unsloth/Qwen3-1.7B-GGUF", allowModelLoad = True),
        )
    assert failure.value.code == "model_load_failed"


def test_a_run_never_starts_a_download(monkeypatch):
    from starlette.requests import Request

    from core.inference import openai_auto_download
    from utils import openai_auto_switch_settings

    producer = FakeProducer(completion("Ok."))
    generate(monkeypatch, producer)
    monkeypatch.setattr(
        openai_auto_switch_settings, "get_openai_auto_download_enabled", lambda: True
    )
    monkeypatch.setattr(openai_auto_download, "is_downloadable_ref", lambda model: True)
    monkeypatch.setattr(inference_routes, "_loaded_satisfies", lambda model: False)
    started = []

    async def download(*args, **kwargs):
        started.append(args)

    monkeypatch.setattr(openai_auto_download, "maybe_auto_download", download)
    model = "unsloth/Qwen3-1.7B-GGUF"
    asyncio.run(inference_routes._maybe_auto_download_model(model, producer.request))
    assert started == []
    # The same call from an ordinary request would download.
    plain = Request({"type": "http", "headers": [], "path": "/"})
    asyncio.run(inference_routes._maybe_auto_download_model(model, plain))
    assert len(started) == 1


def test_a_scheduled_run_waits_for_a_reply_being_generated(monkeypatch):
    keepwarm = keepwarm_module()
    busy = iter([True, True, False])
    monkeypatch.setattr(keepwarm, "inference_in_flight", lambda: next(busy))
    monkeypatch.setattr(scheduler_module, "_YIELD_POLL_SECONDS", 0.0)
    scheduler = ScheduledTaskScheduler(SimpleNamespace(state = SimpleNamespace()))
    asyncio.run(scheduler._yield_to_inference(threading.Event()))
    assert next(busy, "done") == "done"
    # Never past the bound.
    monkeypatch.setattr(keepwarm, "inference_in_flight", lambda: True)
    monkeypatch.setattr(scheduler_module, "_YIELD_SECONDS", 0.0)
    asyncio.run(scheduler._yield_to_inference(threading.Event()))


def test_only_a_signed_in_user_changes_tasks(client):
    task = create(client).json()
    client.app.dependency_overrides[authenticated_via_api_key] = lambda: True
    new_task = {
        "title": "x",
        "prompt": "y",
        "schedule": {"kind": "daily", "time": "09:00"},
        "timezone": "UTC",
    }
    for method, path, body in (
        ("post", "", new_task),
        ("patch", f"/{task['id']}", {"allowModelLoad": True}),
        ("post", f"/{task['id']}/run", {"loadModel": True}),
        ("delete", f"/{task['id']}", None),
    ):
        kwargs = {} if body is None else {"json": body}
        response = getattr(client, method)(f"/api/scheduled-tasks{path}", **kwargs)
        assert response.status_code == 403, (method, path)
    # Reading stays open, and nothing changed.
    listed = client.get("/api/scheduled-tasks").json()["tasks"]
    assert [entry["allowModelLoad"] for entry in listed] == [False]
    assert client.get("/api/scheduled-tasks/runs?since=0").status_code == 200


def test_a_missing_or_disabled_connection_fails_with_its_own_reason(monkeypatch):
    from storage import providers_db
    task = make_task(model = "external::conn-1::gpt-5-mini")
    for config in (None, {"id": "conn-1", "is_enabled": 0}):
        monkeypatch.setattr(providers_db, "get_provider", lambda provider_id, config = config: config)
        producer = FakeProducer(completion("unused"))
        with pytest.raises(RunFailed) as failure:
            generate(monkeypatch, producer, task)
        assert failure.value.code == "connection_unavailable"
        assert producer.payload is None


def test_a_connection_run_streams_and_records_which_connection_answered(monkeypatch):
    import json as jsonlib

    from fastapi.responses import StreamingResponse
    from storage import providers_db

    config = {"id": "conn-1", "is_enabled": 1, "display_name": "OpenAI", "provider_type": "openai"}
    monkeypatch.setattr(providers_db, "get_provider", lambda provider_id: config)

    async def chunks():
        for delta in ({"content": "Dawn "}, {"content": "light."}):
            chunk = {"id": "c1", "model": "gpt-5-mini", "choices": [{"index": 0, "delta": delta}]}
            yield f"data: {jsonlib.dumps(chunk)}\n\n".encode()
        yield b"data: [DONE]\n\n"

    producer = FakeProducer(StreamingResponse(chunks(), media_type = "text/event-stream"))
    answer = generate(
        monkeypatch,
        producer,
        make_task(model = "external::conn-1::gpt-5-mini", reasoningEffort = "off"),
    )
    payload = producer.payload
    assert (payload.provider_id, payload.external_model, payload.stream) == (
        "conn-1",
        "gpt-5-mini",
        True,
    )
    assert payload.reasoning_effort == "none" and payload.enable_thinking is None
    assert answer.content == "Dawn light."
    assert answer.provider == config


def test_a_new_chat_per_run_can_go_to_a_project_and_be_archived():
    make_project()
    task, run = claim_manual_run()
    db.update_task(task["id"], {"archiveSuccessful": True, "projectId": "proj-1"})
    task = db.get_task(task["id"])
    answer = Answer(content = "Done.", reasoning = "", model = "gemma-3")
    thread = studio_db.get_chat_thread(scheduler_module.save_chat(task, run, answer, 1_000, 2_000))
    assert (thread["projectId"], thread["archived"]) == ("proj-1", True)


def test_a_project_deleted_since_the_task_was_saved_is_skipped():
    task, run = claim_manual_run()
    db.update_task(task["id"], {"projectId": "gone"})
    answer = Answer(content = "Done.", reasoning = "", model = "gemma-3")
    thread_id = scheduler_module.save_chat(db.get_task(task["id"]), run, answer, 1_000, 2_000)
    assert studio_db.get_chat_thread(thread_id)["projectId"] is None


def test_without_a_new_chat_per_run_every_run_continues_one_chat():
    task, first = claim_manual_run()
    db.update_task(
        task["id"], {"newChatEachRun": False, "archiveSuccessful": True, "projectId": "x"}
    )
    task = db.get_task(task["id"])
    one = scheduler_module.save_chat(
        task, first, Answer("First answer.", "", "gemma-3"), 1_000, 2_000
    )
    # The run that started it is the one the stale-run sweep would remove it with.
    assert one == db.chat_id_for_run(first["id"])
    task = db.get_task(task["id"])
    assert task["chatThreadId"] == one
    # Those two options belong to a chat per run.
    assert studio_db.get_chat_thread(one)["archived"] is False

    assert scheduler_module.chat_history(one) == [
        {"role": "user", "content": "Write a haiku about mornings."},
        {"role": "assistant", "content": "First answer."},
    ]
    second = {**first, "id": "run-2"}
    two = scheduler_module.save_chat(
        task, second, Answer("Second answer.", "", "gemma-3"), 3_000, 4_000
    )
    assert two == one
    messages = studio_db.list_chat_messages(one)
    assert [m["role"] for m in messages] == ["user", "assistant", "user", "assistant"]
    # Threaded after the earlier answer, so the chat reads as one conversation.
    assert messages[2]["parentId"] == messages[1]["id"]
    assert len(studio_db.list_chat_threads()) == 1


def test_a_deleted_ongoing_chat_is_replaced_by_a_new_one():
    task, first = claim_manual_run()
    db.update_task(task["id"], {"newChatEachRun": False})
    one = scheduler_module.save_chat(
        db.get_task(task["id"]), first, Answer("A.", "", "m"), 1_000, 2_000
    )
    studio_db.delete_chat_threads([one])
    second = {**first, "id": "run-2"}
    two = scheduler_module.save_chat(
        db.get_task(task["id"]), second, Answer("B.", "", "m"), 3_000, 4_000
    )
    assert two != one and db.get_task(task["id"])["chatThreadId"] == two


def test_an_ongoing_chat_reaches_the_model_as_history(monkeypatch):
    task, first = claim_manual_run()
    db.update_task(task["id"], {"newChatEachRun": False})
    scheduler_module.save_chat(
        db.get_task(task["id"]), first, Answer("Yesterday.", "", "m"), 1_000, 2_000
    )
    producer = FakeProducer(completion("Today."))
    generate(monkeypatch, producer, db.get_task(task["id"]))
    roles = [message.role for message in producer.payload.messages]
    assert roles == ["system", "user", "assistant", "user"]
    assert producer.payload.messages[2].content == "Yesterday."


def test_an_empty_reply_is_a_failure(monkeypatch):
    with pytest.raises(RunFailed) as failure:
        generate(monkeypatch, FakeProducer(completion("  ", reasoning = "Thinking only.")))
    assert failure.value.code == "empty"


def test_a_run_that_never_answers_is_stopped(monkeypatch):
    monkeypatch.setattr(scheduler_module, "_RUN_TIMEOUT_SECONDS", 0.05)
    producer = FakeProducer(hang = True)
    cancel_event = threading.Event()
    with pytest.raises(RunFailed) as failure:
        generate(monkeypatch, producer, cancel_event = cancel_event)
    assert failure.value.code == "timeout"
    assert cancel_event.is_set() and producer.cancelled


class AdmittedProducer(FakeProducer):
    """Admits the request the way the auto-switch hook does for local chat."""

    async def __call__(self, payload, request, current_subject, *, cancel_on_disconnect):
        from core.inference import llama_keepwarm

        llama_keepwarm.note_admitted_inference(request.scope)
        self.admitted = llama_keepwarm.other_admitted_inference_count()
        return await super().__call__(
            payload, request, current_subject, cancel_on_disconnect = cancel_on_disconnect
        )


@pytest.mark.parametrize("answers", [True, False])
def test_a_run_gives_back_its_admission_and_adopts_the_model_only_once_answered(
    monkeypatch, answers
):
    from core.inference import llama_keepwarm

    claims = []
    monkeypatch.setattr(inference_routes, "_set_preview_resident", claims.append)
    before = llama_keepwarm.other_admitted_inference_count()
    producer = (
        AdmittedProducer(completion("Done."))
        if answers
        else AdmittedProducer(error = HTTPException(status_code = 500, detail = "boom"))
    )
    monkeypatch.setattr(inference_routes, "_loaded_slot_ident", lambda: "gemma-3")
    try:
        generate(monkeypatch, producer)
    except RunFailed:
        assert not answers

    assert producer.admitted == before + 1
    # No middleware sees this request end, so the run gives the count back itself, or preview
    # loads would stay refused as busy.
    assert llama_keepwarm.other_admitted_inference_count() == before
    assert claims == ([None] if answers else [])


def test_a_run_whose_model_load_never_lets_go_times_out(monkeypatch):
    from core.inference import llama_keepwarm

    async def wedged(self, cancel_event = None):
        await asyncio.sleep(3600)

    monkeypatch.setattr(llama_keepwarm.InferenceActivityReservation, "start", wedged)
    monkeypatch.setattr(scheduler_module, "_RUN_TIMEOUT_SECONDS", 0.05)
    producer = FakeProducer(completion("Never asked."))
    with pytest.raises(RunFailed) as failure:
        generate(monkeypatch, producer)
    assert failure.value.code == "timeout"
    assert producer.payload is None


def test_a_cancelled_run_discards_its_partial_answer(monkeypatch):
    cancel_event = threading.Event()

    class CancelledMidway(FakeProducer):
        async def __call__(self, *args, **kwargs):
            cancel_event.set()
            return completion("Half an ans")

    with pytest.raises(RunFailed) as failure:
        generate(monkeypatch, CancelledMidway(), cancel_event = cancel_event)
    assert failure.value.code == "cancelled"


def test_the_scheduler_runs_a_due_task_and_wakes_for_run_now(monkeypatch):
    monkeypatch.setattr(scheduler_module, "job_accounts", lambda: [OWNER])
    answered = []

    async def fake_generate(
        app,
        task,
        run,
        cancel_event,
        holder = None,
    ):
        answered.append(run["source"])
        return Answer(content = "Done.", reasoning = "", model = "gemma-3")

    monkeypatch.setattr(scheduler_module, "generate_answer", fake_generate)
    make_task(nextRunAt = db.now_ms() - DAY)

    async def settled(count: int) -> None:
        for _ in range(500):
            last_run = db.get_task("task-1")["lastRun"]
            if len(answered) == count and last_run and last_run["status"] == "succeeded":
                return
            await asyncio.sleep(0.01)
        raise AssertionError(f"run {count} never settled")

    async def scenario():
        scheduler = ScheduledTaskScheduler(SimpleNamespace(state = SimpleNamespace()))
        scheduler.start()
        try:
            await settled(1)
            db.request_run("task-1")
            scheduler.wake()
            await settled(2)
        finally:
            await scheduler.stop()

    asyncio.run(scenario())

    assert answered == ["schedule", "manual"]
    assert len(studio_db.list_chat_threads()) == 2
    # The overdue run moved the schedule on to the next 09:00 still ahead.
    assert db.get_task("task-1")["nextRunAt"] > db.now_ms()


def test_a_claim_given_back_is_due_again_or_queued_again():
    due_at = ms("2026-10-14T09:00:00+00:00")
    now = ms("2026-10-14T09:00:05+00:00")
    make_task(nextRunAt = due_at)
    task, run = db.claim_next(now, "worker", next_run_for)
    assert task["nextRunAt"] == ms("2026-10-15T09:00:00+00:00")

    assert db.release_claim(run["id"], "worker", task["nextRunAt"])
    returned = db.get_task("task-1")
    assert (returned["nextRunAt"], returned["lastRun"]) == (due_at, None)
    # Once given back it is no one's to give back again.
    assert not db.release_claim(run["id"], "worker", task["nextRunAt"])

    db.request_run("task-1")
    task, run = db.claim_next(now, "worker", next_run_for)
    assert run["source"] == "manual"
    assert db.release_claim(run["id"], "worker", task["nextRunAt"])
    assert db.get_task("task-1")["lastRun"]["status"] == "queued"


@pytest.mark.parametrize(
    "change",
    [
        {"schedule": {"kind": "daily", "time": "18:00"}},
        {"enabled": False},
    ],
)
def test_a_claim_given_back_keeps_a_schedule_changed_since(change):
    make_task(nextRunAt = ms("2026-10-14T09:00:00+00:00"))
    task, run = db.claim_next(ms("2026-10-14T09:00:05+00:00"), "worker", next_run_for)
    changed = db.update_task(
        "task-1", change, lambda t: None if not t["enabled"] else ms("2026-10-14T18:00:00+00:00")
    )

    assert db.release_claim(run["id"], "worker", task["nextRunAt"])
    assert db.get_task("task-1")["nextRunAt"] == changed["nextRunAt"]


def test_a_claim_made_as_shutdown_begins_is_given_back(monkeypatch):
    monkeypatch.setattr(scheduler_module, "job_accounts", lambda: [OWNER])
    due_at = db.now_ms() - DAY
    make_task(nextRunAt = due_at)
    scheduler = ScheduledTaskScheduler(SimpleNamespace(state = SimpleNamespace()))
    claim_next = db.claim_next

    def claim_as_stop_begins(*args):
        claimed = claim_next(*args)
        # stop() runs on the event loop while this thread commits; it cannot cancel the thread.
        scheduler._stopping = True
        return claimed

    monkeypatch.setattr(db, "claim_next", claim_as_stop_begins)

    assert scheduler._claim() is None
    task = db.get_task("task-1")
    assert (task["nextRunAt"], task["lastRun"]) == (due_at, None)


def test_the_loop_gives_back_a_claim_that_returns_after_shutdown_began(monkeypatch):
    monkeypatch.setattr(scheduler_module, "job_accounts", lambda: [OWNER])
    due_at = db.now_ms() - DAY
    make_task(nextRunAt = due_at)
    scheduler = ScheduledTaskScheduler(SimpleNamespace(state = SimpleNamespace()))
    claim = scheduler._claim
    started = []

    def claim_then_stop():
        claimed = claim()
        # The thread is done, and stop() begins before the loop picks the claim up.
        scheduler._stopping = True
        return claimed

    async def execute(*args):
        started.append(args)

    monkeypatch.setattr(scheduler, "_claim", claim_then_stop)
    monkeypatch.setattr(scheduler, "_execute", execute)

    async def scenario():
        scheduler.start()
        await asyncio.wait_for(scheduler._task, timeout = 10)

    asyncio.run(scenario())

    assert started == []
    task = db.get_task("task-1")
    assert (task["nextRunAt"], task["lastRun"]) == (due_at, None)


def test_a_cancellation_raised_for_a_run_ends_the_run_not_the_scheduler(monkeypatch):
    monkeypatch.setattr(scheduler_module, "job_accounts", lambda: [OWNER])
    calls = []

    async def fake_generate(
        app,
        task,
        run,
        cancel_event,
        holder = None,
    ):
        calls.append(run["id"])
        if len(calls) == 1:
            # What the model unload gate raises when the run's cancel event fires as it waits.
            raise asyncio.CancelledError()
        return Answer(content = "Done.", reasoning = "", model = "gemma-3")

    monkeypatch.setattr(scheduler_module, "generate_answer", fake_generate)
    # Not due, so only the Run now requests below are claimed.
    make_task(nextRunAt = db.now_ms() + DAY)
    db.request_run("task-1")

    async def settled(count: int) -> dict:
        for _ in range(500):
            runs = db.list_runs("task-1")
            if len(calls) == count and runs and runs[0]["status"] in ("succeeded", "failed"):
                return runs[0]
            await asyncio.sleep(0.01)
        raise AssertionError(f"run {count} never settled")

    async def scenario():
        scheduler = ScheduledTaskScheduler(SimpleNamespace(state = SimpleNamespace()))
        scheduler.start()
        try:
            first = await settled(1)
            assert (first["status"], first["errorCode"]) == ("failed", "cancelled")
            # The loop is still there to run the next one.
            db.request_run("task-1")
            scheduler.wake()
            assert (await settled(2))["status"] == "succeeded"
        finally:
            await scheduler.stop()

    asyncio.run(scenario())


def test_the_system_prompt_falls_back_to_utc_for_an_unknown_zone():
    prompt = scheduler_module.system_prompt(
        {"timezone": "Gone/Zone"}, ms("2026-10-14T06:00:00+00:00")
    )
    assert "Wednesday, October 14, 2026 at 06:00 (Gone/Zone)" in prompt
