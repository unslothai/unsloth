# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import asyncio
import types

import pytest

from studio_mcp import export_jobs

from .mcp_harness import bodies, register_export_job, run_tool, sequence

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
        "message": "Training start was cancelled",
        "error_code": "training_start_cancelled",
    },
    ("POST", "/api/train/diffusion/stop"): {"status": "stopping"},
    ("GET", "/api/export/status"): {"is_export_active": True, "active_op_kind": "export_gguf"},
    ("POST", "/api/export/cancel"): {"success": True, "message": "Export cancelled"},
    ("POST", "/api/data-recipe/jobs/rec-1/cancel"): {"job_id": "rec-1", "status": "cancelling"},
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


def _call(
    monkeypatch,
    args,
    overrides = None,
):
    return run_tool(monkeypatch, {**PAYLOADS, **(overrides or {})}, "cancel", args)


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
    result, studio = _call(monkeypatch, args)
    assert result["structuredContent"]["cancelled"] is True
    assert result["structuredContent"]["kind"] == args["kind"]
    (call,) = [c for c in studio.state.calls if c[0] == "POST"]
    assert call[1] == route
    assert (bodies(studio, route)[0] if call[3] else None) == body


def test_training_without_an_id_stops_the_running_job(monkeypatch):
    result, studio = _call(monkeypatch, {"kind": "training"})
    assert result["structuredContent"] == {
        "kind": "training",
        "id": "job-7",
        "cancelled": True,
        "message": "Stop requested. Training will stop at the next safe step.",
    }
    assert [c[1] for c in studio.state.calls] == ["/api/train/status", "/api/train/stop"]
    assert bodies(studio, "/api/train/stop") == [{"save": True, "expected_job_id": "job-7"}]


def test_training_with_nothing_running_stops_nothing(monkeypatch):
    result, studio = _call(
        monkeypatch,
        {"kind": "training"},
        {("GET", "/api/train/status"): {"job_id": "", "is_training_running": False}},
    )
    assert result["structuredContent"]["cancelled"] is False
    assert [c[1] for c in studio.state.calls] == ["/api/train/status"]


def test_an_export_cancel_marks_the_job_and_says_the_checkpoint_was_unloaded(monkeypatch):
    job = register_export_job(job_id = "exp-1", phase = "exporting")
    result, studio = _call(monkeypatch, {"kind": "export", "id": "exp-1"})
    assert (
        result["structuredContent"]["message"]
        == "The export was stopped and its checkpoint unloaded."
    )
    assert job.status == "cancelled"
    assert [c[1] for c in studio.state.calls] == ["/api/export/status", "/api/export/cancel"]


@pytest.mark.parametrize(
    "phase,op",
    [
        ("loading", "export_gguf"),
        ("starting", "load_checkpoint"),
        ("exporting", "export_merged"),
        ("exporting", "cleanup"),
    ],
)
def test_an_export_cancel_leaves_another_jobs_op_alone(monkeypatch, phase, op):
    job = register_export_job(job_id = "exp-1", phase = phase)
    result, studio = _call(
        monkeypatch,
        {"kind": "export", "id": "exp-1"},
        {("GET", "/api/export/status"): {"is_export_active": True, "active_op_kind": op}},
    )
    assert result["structuredContent"]["cancelled"] is False
    assert job.status == "running"
    assert [c[1] for c in studio.state.calls] == ["/api/export/status"]


IDLE_WORKER = {"is_export_active": False, "current_checkpoint": "ref:abc", "last_op_seq": 5}


def test_an_idle_export_worker_keeps_its_checkpoint(monkeypatch):
    # Between ops the worker is alive and holds a checkpoint; the cancel route would kill it.
    result, studio = _call(
        monkeypatch, {"kind": "export"}, {("GET", "/api/export/status"): IDLE_WORKER}
    )
    assert result["structuredContent"]["cancelled"] is False
    assert result["structuredContent"]["message"] == "No export is running."
    assert [c[1] for c in studio.state.calls] == ["/api/export/status"]


# "exporting" waits for the job's own answer first; see the direct tests below.
@pytest.mark.parametrize("phase", ["starting", "loading"])
def test_a_job_between_steps_is_stopped_without_touching_the_worker(monkeypatch, phase):
    job = register_export_job(job_id = "exp-1", phase = phase)
    result, studio = _call(
        monkeypatch, {"kind": "export", "id": "exp-1"}, {("GET", "/api/export/status"): IDLE_WORKER}
    )
    assert result["structuredContent"]["cancelled"] is True
    assert (
        result["structuredContent"]["message"] == "The export job was stopped before its next step."
    )
    assert job.status == "cancelled"
    assert [c[1] for c in studio.state.calls] == ["/api/export/status"]


def _cancel_directly(
    monkeypatch,
    job_run,
    statuses = (IDLE_WORKER,),
):
    """Run cancel on the loop the job's task lives on, which a served tool call cannot share."""
    from studio_mcp.tools import cancel as cancel_module

    status = sequence(*statuses)
    calls = []

    async def route_json(method, path, **_kwargs):
        calls.append((method, path))
        if path.endswith("/cancel"):
            return {"success": True, "message": "Export cancelled"}
        return status(None, None)

    monkeypatch.setattr(cancel_module, "route_json", route_json)
    # cancel reads only the account from its caller.
    monkeypatch.setattr(
        cancel_module, "current_caller", lambda: types.SimpleNamespace(account_id = "owner")
    )
    monkeypatch.setattr(cancel_module, "SETTLE_S", 0.2)

    async def run():
        job = export_jobs.start("owner", "gguf", job_run)
        await asyncio.sleep(0)
        result = await cancel_module.cancel("export", job.job_id)
        status = job.status
        # Clean up a job the cancel left running.
        export_jobs.mark_cancelled(job)
        await asyncio.gather(job.task, return_exceptions = True)
        job.status = status
        return job, result, calls

    return asyncio.run(run())


def test_an_export_whose_answer_is_on_its_way_is_reported_done(monkeypatch):
    # The worker already finished the export; the route is still writing its answer.
    async def answered_soon(job):
        job.phase = "exporting"
        await asyncio.sleep(0.05)

    job, result, _calls = _cancel_directly(monkeypatch, answered_soon)
    assert (result.cancelled, result.message) == (False, "The export already completed.")
    assert job.status == "completed"


async def _exports(job):
    job.phase = "exporting"
    await asyncio.Event().wait()


def test_an_export_that_starts_while_waiting_is_stopped_on_the_worker(monkeypatch):
    running = {"is_export_active": True, "active_op_kind": "export_gguf"}
    job, result, calls = _cancel_directly(monkeypatch, _exports, statuses = (IDLE_WORKER, running))
    assert result.cancelled is True
    assert ("POST", "/api/export/cancel") in calls
    assert job.status == "cancelled"


def test_an_export_queued_behind_someone_elses_op_is_left_alone(monkeypatch):
    theirs = {"is_export_active": True, "active_op_kind": "cleanup"}
    job, result, calls = _cancel_directly(monkeypatch, _exports, statuses = (IDLE_WORKER, theirs))
    assert result.cancelled is False
    assert result.message == "The export running now is not this job; it was left alone."
    assert ("POST", "/api/export/cancel") not in calls
    assert job.status == "running"


def test_a_job_between_steps_is_stopped_at_once(monkeypatch):
    started = []

    async def between(job):
        job.phase = "loading"
        await asyncio.sleep(0.05)
        # The next step must never start once the job was cancelled.
        started.append("export")

    job, result, calls = _cancel_directly(monkeypatch, between)
    assert result.cancelled is True
    assert job.status == "cancelled"
    assert started == []
    assert calls == [("GET", "/api/export/status")]


IDLE_EXPORT = (
    "export",
    "/api/export/cancel",
    {"success": True, "message": "No active export to cancel"},
)
# What each route answers when there was nothing of the caller's to stop.
IDLE_ANSWERS = [
    (
        "training",
        "/api/train/stop",
        {"status": "idle", "message": "No training job is currently running"},
    ),
    (
        "training_start",
        "/api/train/start-requests/req-1/cancel",
        {"start_request_id": "req-1", "job_id": "job-7", "state": "accepted", "message": "Started"},
    ),
    ("diffusion_training", "/api/train/diffusion/stop", {"status": "idle"}),
    IDLE_EXPORT,
    ("recipe", "/api/data-recipe/jobs/rec-1/cancel", {"job_id": "rec-1", "status": "completed"}),
    ("image", "/api/inference/images/generate/cancel", {"cancelled": False}),
    ("video", "/api/inference/video/generate/cancel", {"cancelled": False}),
    ("chat", "/api/inference/cancel", {"cancelled": 0}),
    (
        "dataset_download",
        "/api/hub/datasets/download/cancel",
        {"repo_id": "a/b", "state": "completed"},
    ),
    # A start request rejected for another reason was not cancelled.
    (
        "training_start",
        "/api/train/start-requests/req-1/cancel",
        {
            "start_request_id": "req-1",
            "job_id": "",
            "state": "rejected",
            "message": "Model not found",
            "error_code": "model_not_found",
        },
    ),
    # Cancelling an already cancelled job stops nothing.
    ("recipe", "/api/data-recipe/jobs/rec-1/cancel", {"job_id": "rec-1", "status": "cancelled"}),
    (
        "dataset_download",
        "/api/hub/datasets/download/cancel",
        {"repo_id": "a/b", "state": "cancelled"},
    ),
]
IDS = {
    "training": "job-7",
    "training_start": "req-1",
    "recipe": "rec-1",
    "chat": "mcp-abc",
    "dataset_download": "a/b",
}


@pytest.mark.parametrize("kind,route,answer", IDLE_ANSWERS)
def test_a_cancel_that_stopped_nothing_says_so(monkeypatch, kind, route, answer):
    args = {"kind": kind, **({"id": IDS[kind]} if kind in IDS else {})}
    result, _studio = _call(monkeypatch, args, {("POST", route): answer})
    assert result["structuredContent"]["cancelled"] is False


def test_an_export_the_route_found_gone_leaves_the_job_running(monkeypatch):
    # The op ended between the status read and the cancel.
    job = register_export_job(job_id = "exp-1", phase = "exporting")
    _kind, route, answer = IDLE_EXPORT
    result, _studio = _call(
        monkeypatch, {"kind": "export", "id": "exp-1"}, {("POST", route): answer}
    )
    assert result["structuredContent"] == {
        "kind": "export",
        "id": "exp-1",
        "cancelled": False,
        "message": "No active export to cancel",
    }
    assert job.status == "running"


@pytest.mark.parametrize("status", ["completed", "failed", "cancelled"])
def test_a_finished_export_job_never_stops_the_export_running_now(monkeypatch, status):
    register_export_job(job_id = "exp-1", status = status)
    result, studio = _call(monkeypatch, {"kind": "export", "id": "exp-1"})
    assert result["structuredContent"]["cancelled"] is False
    assert result["structuredContent"]["message"] == f"The export already {status}."
    assert studio.state.calls == []


def test_another_accounts_export_is_not_cancelled(monkeypatch):
    register_export_job(account_id = "alice", job_id = "exp-1")
    result, studio = _call(monkeypatch, {"kind": "export", "id": "exp-1"})
    assert result["content"][0]["text"] == "No such export job"
    assert studio.state.calls == []
