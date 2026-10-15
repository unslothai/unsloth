# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Scheduled tasks: schedule math, storage, routes, and the runner that posts answers to chats."""

import asyncio
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
    assert task["nextRunAt"] == ms("2026-10-15T09:00:00+00:00")
    assert task["lastRun"] is None
    assert "ownerSubject" not in task
    assert client.get("/api/scheduled-tasks").json()["tasks"] == [task]
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


def test_a_rename_keeps_an_overdue_run_due(client):
    make_task(nextRunAt = ms("2026-10-14T09:00:00+00:00"))
    renamed = client.patch("/api/scheduled-tasks/task-1", json = {"title": "Evening brief"}).json()
    assert renamed["title"] == "Evening brief"
    assert renamed["nextRunAt"] == ms("2026-10-14T09:00:00+00:00")


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
    make_task()
    db.request_run("task-1")
    _task, run = db.claim_next(db.now_ms(), "worker", next_run_for)
    first = client.get("/api/scheduled-tasks/runs").json()
    assert first["runs"] == [] and first["activeRuns"] == 1
    db.finish_run(run["id"], "worker", status = "failed", error_code = "no_model")

    later = client.get("/api/scheduled-tasks/runs", params = {"since": first["now"]}).json()
    assert [(r["taskTitle"], r["errorCode"]) for r in later["runs"]] == [
        ("Morning brief", "no_model")
    ]
    assert later["activeRuns"] == 0
    assert later["nextRunAt"] == ms("2026-10-14T09:00:00+00:00")


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


def test_the_system_prompt_falls_back_to_utc_for_an_unknown_zone():
    prompt = scheduler_module.system_prompt(
        {"timezone": "Gone/Zone"}, ms("2026-10-14T06:00:00+00:00")
    )
    assert "Wednesday, October 14, 2026 at 06:00 (Gone/Zone)" in prompt
