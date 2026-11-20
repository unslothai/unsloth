# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Scheduled training starts: time resolution, the saved request, and the scheduler's claim."""

import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from auth.authentication import authenticated_via_api_key, get_current_subject
from core.training import schedule_scheduler as scheduler_module
from core.training.schedule_scheduler import TrainingScheduleScheduler
from core.training.schedules import (
    InvalidStartTime,
    request_from_payload,
    request_payload,
    resolve_start_time,
)
from models.training import TrainingStartRequest
from routes import training as training_routes
from routes.training_schedules import router
from storage import credential_secrets
from storage import training_schedules_db as db
from utils.account_context import OWNER

MINUTE = 60 * 1000


def ms(iso: str) -> int:
    moment = datetime.fromisoformat(iso)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo = timezone.utc)
    return int(moment.timestamp() * 1000)


def make_request(**overrides) -> TrainingStartRequest:
    values = {
        "model_name": "unsloth/Llama-3.2-1B-Instruct",
        "training_type": "LoRA/QLoRA",
        "format_type": "alpaca",
        "hf_dataset": "unsloth/alpaca",
    }
    values.update(overrides)
    return TrainingStartRequest(**values)


def make_schedule(**overrides) -> dict:
    schedule = {
        "id": "schedule-1",
        "ownerSubject": "unsloth",
        "status": db.PENDING,
        "startAt": ms("2026-10-30T09:00:00+00:00"),
        "startAtLocal": "2026-10-30T09:00",
        "timezone": "UTC",
        "modelName": "unsloth/Llama-3.2-1B-Instruct",
        "projectName": None,
        "createdAt": 1,
        "updatedAt": 1,
    }
    schedule.update(overrides)
    return schedule


def store(**overrides) -> dict:
    return db.create_schedule(make_schedule(**overrides), request_payload(make_request()))


def payload_secret_exists(schedule_id: str) -> bool:
    return credential_secrets.secret_row_exists(db.PAYLOAD_KIND, schedule_id)


@pytest.fixture
def wakes():
    return []


@pytest.fixture
def client(monkeypatch, wakes):
    async def prepare(
        request,
        via_api_key,
        *,
        stop_token_preview = True,
    ):
        prepare.calls.append((via_api_key, stop_token_preview))

    async def admit(request, current_subject, via_api_key):
        return SimpleNamespace(training_kwargs = {}, hf_token = None, resume_run = None)

    prepare.calls = []
    monkeypatch.setattr(training_routes, "_prepare_training_request", prepare)
    monkeypatch.setattr(training_routes, "_admit_training_request", admit)
    app = FastAPI()
    app.state.training_schedule_scheduler = SimpleNamespace(wake = lambda: wakes.append(True))
    app.include_router(router, prefix = "/api/train/schedules")
    app.dependency_overrides[get_current_subject] = lambda: "unsloth"
    app.dependency_overrides[authenticated_via_api_key] = lambda: False
    with TestClient(app) as test_client:
        test_client.prepare_calls = prepare.calls
        yield test_client


@pytest.mark.parametrize(
    "start_at, zone, now, expected",
    [
        (
            "2026-10-30T09:00",
            "UTC",
            "2026-10-29T12:00",
            "2026-10-30T09:00",
        ),
        # Wall clock in the user's zone: 09:00 in Berlin is 08:00 UTC in late October.
        (
            "2026-10-30T09:00",
            "Europe/Berlin",
            "2026-10-29T12:00",
            "2026-10-30T08:00",
        ),
        # A time the clocks repeat runs at the first occurrence, as the chat schedules do.
        (
            "2026-10-25T02:30",
            "Europe/Berlin",
            "2026-10-24T12:00",
            "2026-10-25T00:30",
        ),
        # A start picked for the current minute is due now, not refused as the past.
        (
            "2026-10-29T12:00",
            "UTC",
            "2026-10-29T12:00:45",
            "2026-10-29T12:00",
        ),
    ],
)
def test_a_start_resolves_to_its_local_wall_clock(start_at, zone, now, expected):
    assert resolve_start_time(start_at, zone, ms(now)) == ms(expected)


@pytest.mark.parametrize(
    "start_at, zone",
    [
        ("2026-10-29T11:59", "UTC"),
        ("2026-13-01T09:00", "UTC"),
        ("tomorrow", "UTC"),
        ("2026-10-30T09:00", "Mars/Olympus"),
    ],
)
def test_a_start_that_cannot_happen_is_refused(start_at, zone):
    with pytest.raises(InvalidStartTime):
        resolve_start_time(start_at, zone, ms("2026-10-29T12:00"))


def test_a_saved_request_round_trips_through_its_payload():
    request = make_request(
        learning_rate = "3e-4",
        hf_token = "hf_secret",
        enable_wandb = True,
        wandb_token = "wandb_secret",
    )
    restored = request_from_payload(request_payload(request))
    assert restored == request


def test_a_schedule_stores_its_request_encrypted_and_lists_pending_first():
    schedule = store()
    assert schedule["status"] == db.PENDING
    assert payload_secret_exists(schedule["id"])
    assert request_from_payload(db.get_payload(schedule["id"])).model_name == schedule["modelName"]
    assert [row["id"] for row in db.list_schedules()] == [schedule["id"]]


def test_only_one_start_waits_per_account():
    store()
    with pytest.raises(db.TrainingScheduleExists):
        store(id = "schedule-2")


def test_cancelling_a_waiting_start_drops_its_request():
    schedule = store()
    cancelled, was_pending = db.cancel_pending(schedule["id"])
    assert was_pending
    assert cancelled["status"] == db.CANCELLED
    assert not payload_secret_exists(schedule["id"])
    # A second cancel still finds the row, so the route can say it already ended.
    again, was_pending = db.cancel_pending(schedule["id"])
    assert not was_pending and again["status"] == db.CANCELLED
    assert db.cancel_pending("missing") == (None, False)


def test_a_due_start_is_claimed_once():
    schedule = store()
    claimed, due = db.poll(schedule["startAt"])
    assert claimed["id"] == schedule["id"]
    assert claimed["status"] == db.STARTING
    assert due is None
    assert db.poll(schedule["startAt"])[0] is None


def test_a_claimed_start_that_never_began_is_given_back_at_startup():
    schedule = store()
    db.claim_due(schedule["startAt"])
    db.reconcile_startup(schedule["startAt"] + 1000)
    assert db.get_schedule(schedule["id"])["status"] == db.PENDING


def test_a_start_missed_while_studio_was_closed_is_not_run_late():
    schedule = store()
    late = schedule["startAt"] + db.START_GRACE_MS + MINUTE
    assert db.poll(late)[0] is None
    missed = db.get_schedule(schedule["id"])
    assert missed["status"] == db.MISSED
    assert missed["error"]
    assert not payload_secret_exists(schedule["id"])


def test_a_start_within_the_grace_still_runs():
    schedule = store()
    claimed, _ = db.poll(schedule["startAt"] + db.START_GRACE_MS - MINUTE)
    assert claimed["id"] == schedule["id"]


def test_releasing_a_claim_makes_the_start_due_again():
    schedule = store()
    db.claim_due(schedule["startAt"])
    assert db.release_claim(schedule["id"])
    assert not db.release_claim(schedule["id"])
    assert db.poll(schedule["startAt"])[0]["id"] == schedule["id"]


def test_settling_a_claim_records_the_outcome_and_drops_the_request():
    schedule = store()
    db.claim_due(schedule["startAt"])
    assert db.finish_claim(schedule["id"], db.STARTED, job_id = "job_1")
    settled = db.get_schedule(schedule["id"])
    assert settled["status"] == db.STARTED
    assert settled["jobId"] == "job_1"
    assert not payload_secret_exists(schedule["id"])
    assert not db.finish_claim(schedule["id"], db.FAILED)


def test_finished_schedules_keep_only_the_newest(monkeypatch):
    monkeypatch.setattr(db, "HISTORY_LIMIT", 2)
    for index in range(4):
        schedule = store(id = f"schedule-{index}")
        db.claim_due(schedule["startAt"])
        db.finish_claim(schedule["id"], db.STARTED)
    finished = [row["id"] for row in db.list_schedules() if row["status"] != db.PENDING]
    assert finished == ["schedule-3", "schedule-2"]


def test_a_retired_account_drops_its_waiting_start():
    schedule = store()
    assert db.cancel_all_active() == 1
    assert db.get_schedule(schedule["id"])["status"] == db.CANCELLED
    assert not payload_secret_exists(schedule["id"])


def test_create_list_and_cancel_over_http(client):
    response = client.post(
        "/api/train/schedules",
        json = {
            "start_at": "2099-01-01T09:00",
            "timezone": "UTC",
            "request": make_request().model_dump(mode = "json"),
        },
    )
    assert response.status_code == 201, response.text
    created = response.json()
    assert created["status"] == "pending"
    assert created["modelName"] == "unsloth/Llama-3.2-1B-Instruct"
    # Scheduling never stops the token preview; the run does that when it starts.
    assert client.prepare_calls == [(False, False)]

    listed = client.get("/api/train/schedules").json()
    assert [row["id"] for row in listed["schedules"]] == [created["id"]]
    assert listed["now"] > 0

    assert client.delete(f"/api/train/schedules/{created['id']}").status_code == 204
    assert client.delete(f"/api/train/schedules/{created['id']}").status_code == 409
    assert client.delete("/api/train/schedules/missing").status_code == 404


def test_a_second_schedule_is_refused(client):
    body = {
        "start_at": "2099-01-01T09:00",
        "timezone": "UTC",
        "request": make_request().model_dump(mode = "json"),
    }
    assert client.post("/api/train/schedules", json = body).status_code == 201
    second = client.post("/api/train/schedules", json = body)
    assert second.status_code == 409
    assert "already scheduled" in second.json()["detail"]


@pytest.mark.parametrize(
    "start_at, zone",
    [("2020-01-01T09:00", "UTC"), ("nonsense", "UTC"), ("2099-01-01T09:00", "Mars/Olympus")],
)
def test_a_bad_start_is_refused_over_http(client, start_at, zone):
    response = client.post(
        "/api/train/schedules",
        json = {
            "start_at": start_at,
            "timezone": zone,
            "request": make_request().model_dump(mode = "json"),
        },
    )
    assert response.status_code == 400


def test_an_api_key_cannot_change_schedules(monkeypatch):
    app = FastAPI()
    app.include_router(router, prefix = "/api/train/schedules")
    app.dependency_overrides[get_current_subject] = lambda: "unsloth"
    app.dependency_overrides[authenticated_via_api_key] = lambda: True
    with TestClient(app) as api_client:
        response = api_client.post(
            "/api/train/schedules",
            json = {
                "start_at": "2099-01-01T09:00",
                "timezone": "UTC",
                "request": make_request().model_dump(mode = "json"),
            },
        )
    assert response.status_code == 403


def test_a_waiting_schedule_refuses_a_manual_start():
    store()
    with pytest.raises(HTTPException) as raised:
        asyncio.run(
            training_routes.start_training(
                make_request(), current_subject = "unsloth", via_api_key = False
            )
        )
    assert raised.value.status_code == 409
    assert "scheduled" in raised.value.detail


def _claimed_schedule() -> dict:
    schedule = store()
    db.claim_due(schedule["startAt"])
    return db.get_schedule(schedule["id"])


def _run_execute(
    monkeypatch,
    response = None,
    error = None,
):
    async def fake_start(request, current_subject, via_api_key):
        if error is not None:
            raise error
        return response

    monkeypatch.setattr(training_routes, "start_training", fake_start)
    scheduler = TrainingScheduleScheduler(SimpleNamespace())
    asyncio.run(scheduler._execute(OWNER, _claimed_schedule()))
    return db.list_schedules()[0]


def test_the_scheduler_starts_a_due_run(monkeypatch):
    settled = _run_execute(
        monkeypatch,
        response = SimpleNamespace(status = "queued", job_id = "job_1", message = "queued"),
    )
    assert settled["status"] == db.STARTED
    assert settled["jobId"] == "job_1"
    assert not payload_secret_exists(settled["id"])


def test_the_scheduler_records_why_a_run_could_not_start(monkeypatch):
    settled = _run_execute(
        monkeypatch,
        response = SimpleNamespace(
            status = "error",
            job_id = "",
            message = "Training is already in progress.",
        ),
    )
    assert settled["status"] == db.FAILED
    assert settled["error"] == "Training is already in progress."


def test_the_scheduler_records_a_refused_start(monkeypatch):
    settled = _run_execute(
        monkeypatch,
        error = HTTPException(status_code = 400, detail = "Model is not trainable."),
    )
    assert settled["status"] == db.FAILED
    assert settled["error"] == "Model is not trainable."


def test_a_start_whose_request_is_gone_fails_clearly(monkeypatch):
    schedule = _claimed_schedule()
    credential_secrets.delete_secret(db.PAYLOAD_KIND, schedule["id"])
    scheduler = TrainingScheduleScheduler(SimpleNamespace())
    asyncio.run(scheduler._execute(OWNER, db.get_schedule(schedule["id"])))
    settled = db.get_schedule(schedule["id"])
    assert settled["status"] == db.FAILED
    assert "could not be read" in settled["error"]


def test_the_scheduler_claims_the_due_start_for_its_account(monkeypatch):
    schedule = store()
    monkeypatch.setattr(scheduler_module, "job_accounts", lambda: [OWNER])
    monkeypatch.setattr(scheduler_module.db, "now_ms", lambda: schedule["startAt"])
    scheduler = TrainingScheduleScheduler(SimpleNamespace())
    claimed = scheduler._claim()
    assert claimed[0].account_id == OWNER.account_id
    assert claimed[1]["id"] == schedule["id"]
    assert scheduler._due is None
