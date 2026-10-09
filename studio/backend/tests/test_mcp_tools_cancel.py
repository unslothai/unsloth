# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import asyncio
import json

import pytest
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from mcp_server import create_studio_mcp
from studio_mcp import export_jobs
from studio_mcp.export_jobs import ExportJob

from .mcp_harness import call_tool, fake_studio, served

PAYLOADS = {
    ("GET", "/api/train/status"): {
        "job_id": "job-7",
        "is_training_running": True,
        "phase": "training",
    },
    ("POST", "/api/train/stop"): {
        "status": "stopped",
        "message": "Stop requested. Training will stop at the next safe step.",
    },
    ("POST", "/api/train/start-requests/req-1/cancel"): {
        "start_request_id": "req-1",
        "job_id": "",
        "state": "rejected",
        "message": "Cancelled",
    },
    ("POST", "/api/train/diffusion/stop"): {"status": "stopping"},
    ("POST", "/api/export/cancel"): {"success": True, "message": "Export cancelled"},
    ("POST", "/api/data-recipe/jobs/rec-1/cancel"): {"job_id": "rec-1", "status": "cancelled"},
    ("POST", "/api/inference/images/generate/cancel"): {"cancelled": True},
    ("POST", "/api/inference/video/generate/cancel"): {"cancelled": True},
    ("POST", "/api/inference/cancel"): {"cancelled": 1},
    ("POST", "/api/hub/datasets/download/cancel"): {"repo_id": "a/b", "state": "cancelling"},
}


@pytest.fixture(autouse = True)
def fresh_registry():
    export_jobs._reset()
    yield
    export_jobs._reset()


def _studio(overrides = None):
    routes = {
        key: (lambda payload: lambda request, body: payload)(value)
        for key, value in PAYLOADS.items()
    }
    routes.update(overrides or {})
    return fake_studio(routes)


def _call(monkeypatch, studio, args):
    with TestClient(served(create_studio_mcp(), studio, monkeypatch = monkeypatch)) as http:
        return call_tool(http, "cancel", args)


@pytest.mark.parametrize(
    "args,route,body",
    [
        (
            {"kind": "training", "id": "job-7", "save": False},
            "/api/train/stop",
            {"save": False, "expected_job_id": "job-7"},
        ),
        ({"kind": "training_start", "id": "req-1"}, "/api/train/start-requests/req-1/cancel", None),
        ({"kind": "diffusion_training"}, "/api/train/diffusion/stop", None),
        ({"kind": "recipe", "id": "rec-1"}, "/api/data-recipe/jobs/rec-1/cancel", None),
        ({"kind": "image"}, "/api/inference/images/generate/cancel", None),
        ({"kind": "video"}, "/api/inference/video/generate/cancel", None),
        ({"kind": "chat", "id": "mcp-abc"}, "/api/inference/cancel", {"cancel_id": "mcp-abc"}),
        (
            {"kind": "dataset_download", "id": "a/b"},
            "/api/hub/datasets/download/cancel",
            {"repo_id": "a/b"},
        ),
    ],
)
def test_each_kind_reaches_its_route(monkeypatch, args, route, body):
    studio = _studio()
    result = _call(monkeypatch, studio, args)
    assert result["structuredContent"]["cancelled"] is True
    assert result["structuredContent"]["kind"] == args["kind"]
    (call,) = [c for c in studio.state.calls if c[0] == "POST"]
    assert call[1] == route
    assert (json.loads(call[3]) if call[3] else None) == body


def test_training_without_an_id_stops_the_running_job(monkeypatch):
    studio = _studio()
    result = _call(monkeypatch, studio, {"kind": "training"})
    assert result["structuredContent"] == {
        "kind": "training",
        "id": "job-7",
        "cancelled": True,
        "message": "Stop requested. Training will stop at the next safe step.",
    }
    assert [c[1] for c in studio.state.calls] == ["/api/train/status", "/api/train/stop"]
    assert json.loads(studio.state.calls[1][3]) == {"save": True, "expected_job_id": "job-7"}


def test_training_with_nothing_running_stops_nothing(monkeypatch):
    studio = _studio(
        {("GET", "/api/train/status"): lambda r, b: {"job_id": "", "is_training_running": False}}
    )
    result = _call(monkeypatch, studio, {"kind": "training"})
    assert result["structuredContent"]["cancelled"] is False
    assert [c[1] for c in studio.state.calls] == ["/api/train/status"]


def test_a_foreign_training_job_gives_the_routes_404(monkeypatch):
    def gone(request, body):
        return JSONResponse(
            {"detail": "The requested training job is no longer active."}, status_code = 404
        )

    studio = _studio({("POST", "/api/train/stop"): gone})
    result = _call(monkeypatch, studio, {"kind": "training", "id": "someone-elses"})
    assert result["isError"] is True
    assert (
        result["content"][0]["text"] == "The requested training job is no longer active. (HTTP 404)"
    )


def test_an_export_cancel_marks_the_job_and_says_the_checkpoint_was_unloaded(monkeypatch):
    job = ExportJob(job_id = "exp-1", account_id = "owner", format = "gguf")
    export_jobs._jobs["owner:exp-1"] = job
    studio = _studio()
    result = _call(monkeypatch, studio, {"kind": "export", "id": "exp-1"})
    assert (
        result["structuredContent"]["message"]
        == "The export was stopped and its checkpoint unloaded."
    )
    assert job.status == "cancelled"
    assert [c[1] for c in studio.state.calls] == ["/api/export/cancel"]


# What each route answers when there was nothing of the caller's to stop.
IDLE_ANSWERS = {
    "training": (
        "/api/train/stop",
        {"status": "idle", "message": "No training job is currently running"},
    ),
    "training_start": (
        "/api/train/start-requests/req-1/cancel",
        {"start_request_id": "req-1", "job_id": "job-7", "state": "accepted", "message": "Started"},
    ),
    "diffusion_training": ("/api/train/diffusion/stop", {"status": "idle"}),
    "export": ("/api/export/cancel", {"success": True, "message": "No active export to cancel"}),
    "recipe": ("/api/data-recipe/jobs/rec-1/cancel", {"job_id": "rec-1", "status": "completed"}),
    "image": ("/api/inference/images/generate/cancel", {"cancelled": False}),
    "video": ("/api/inference/video/generate/cancel", {"cancelled": False}),
    "chat": ("/api/inference/cancel", {"cancelled": 0}),
    "dataset_download": (
        "/api/hub/datasets/download/cancel",
        {"repo_id": "a/b", "state": "completed"},
    ),
}
IDS = {
    "training": "job-7",
    "training_start": "req-1",
    "recipe": "rec-1",
    "chat": "mcp-abc",
    "dataset_download": "a/b",
}


@pytest.mark.parametrize("kind", sorted(IDLE_ANSWERS))
def test_a_cancel_that_stopped_nothing_says_so(monkeypatch, kind):
    route, answer = IDLE_ANSWERS[kind]
    studio = _studio({("POST", route): lambda request, body: answer})
    args = {"kind": kind, **({"id": IDS[kind]} if kind in IDS else {})}
    result = _call(monkeypatch, studio, args)
    assert result["structuredContent"]["cancelled"] is False


def test_an_idle_export_cancel_leaves_the_job_running(monkeypatch):
    job = ExportJob(job_id = "exp-1", account_id = "owner", format = "gguf")
    export_jobs._jobs["owner:exp-1"] = job
    route, answer = IDLE_ANSWERS["export"]
    studio = _studio({("POST", route): lambda request, body: answer})
    result = _call(monkeypatch, studio, {"kind": "export", "id": "exp-1"})
    assert result["structuredContent"] == {
        "kind": "export",
        "id": "exp-1",
        "cancelled": False,
        "message": "No active export to cancel",
    }
    assert job.status == "running"


@pytest.mark.parametrize("status", ["completed", "failed", "cancelled"])
def test_a_finished_export_job_never_stops_the_export_running_now(monkeypatch, status):
    job = ExportJob(job_id = "exp-1", account_id = "owner", format = "gguf")
    job.status = status
    export_jobs._jobs["owner:exp-1"] = job
    studio = _studio()
    result = _call(monkeypatch, studio, {"kind": "export", "id": "exp-1"})
    assert result["structuredContent"]["cancelled"] is False
    assert result["structuredContent"]["message"] == f"The export already {status}."
    assert studio.state.calls == []


def test_another_accounts_export_is_not_cancelled(monkeypatch):
    export_jobs._jobs["alice:exp-1"] = ExportJob(job_id = "exp-1", account_id = "alice", format = "gguf")
    studio = _studio()
    result = _call(monkeypatch, studio, {"kind": "export", "id": "exp-1"})
    assert result["content"][0]["text"] == "No such export job"
    assert studio.state.calls == []


@pytest.mark.parametrize("kind", ["training_start", "recipe", "chat", "dataset_download"])
def test_kinds_that_need_an_id_say_so(monkeypatch, kind):
    studio = _studio()
    result = _call(monkeypatch, studio, {"kind": kind})
    assert result["isError"] is True
    assert "needs id" in result["content"][0]["text"]
    assert studio.state.calls == []


def test_cancel_is_destructive_and_offers_no_audio_kind():
    tool = {t.name: t for t in asyncio.run(create_studio_mcp().list_tools())}["cancel"]
    assert tool.annotations.destructiveHint is True
    assert tool.annotations.readOnlyHint is False
    assert tool.annotations.openWorldHint is False
    assert tool.output_schema is not None
    kinds = tool.parameters["properties"]["kind"]["enum"]
    assert "audio" not in kinds
    assert len(kinds) == 9
