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

from auth.authentication import get_current_subject
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


def test_create_list_and_read(client, monkeypatch):
    monkeypatch.setattr(db, "now_ms", lambda: ms("2026-10-14T12:00:00+00:00"))
    response = create(client, title = "  Morning brief  ")
    assert response.status_code == 201
    task = response.json()
    assert task["title"] == "Morning brief"
    assert task["status"] == "active"
    assert task["notify"] == "all"
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
        ({"model": "some-model"}, 422),
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
        assert conn.execute("SELECT notify FROM scheduled_tasks").fetchall() == [("all",)]
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
    async def fake_generate(app, task, run, cancel_event):
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

    async def delete_then_answer(app, task, run, cancel_event):
        await asyncio.to_thread(db.delete_task, task["id"])
        return Answer(content = "Too late.", reasoning = "", model = "gemma-3")

    monkeypatch.setattr(scheduler_module, "generate_answer", delete_then_answer)
    scheduler = ScheduledTaskScheduler(SimpleNamespace(state = SimpleNamespace()))
    scheduler.worker_id = "worker-under-test"
    asyncio.run(scheduler._execute(OWNER, task, run))

    assert studio_db.list_chat_threads() == []


def recording_generate(asked):
    async def generate(app, task, run, cancel_event):
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


def test_a_task_deleted_while_its_chat_is_saved_keeps_no_chat(monkeypatch):
    task, run = claim_manual_run()

    async def answer(app, task, run, cancel_event):
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
    async def generate(app, task, run, cancel_event):
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

    async def fake_generate(app, task, run, cancel_event):
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


def test_a_cancellation_raised_for_a_run_ends_the_run_not_the_scheduler(monkeypatch):
    monkeypatch.setattr(scheduler_module, "job_accounts", lambda: [OWNER])
    calls = []

    async def fake_generate(app, task, run, cancel_event):
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
