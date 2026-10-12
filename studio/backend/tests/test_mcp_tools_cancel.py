# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import asyncio
import types

import pytest

from studio_mcp import export_jobs

from .mcp_harness import bodies, register_export_job, run_tool, sequence

START = "/api/train/start-requests/req-1/cancel"
RECIPE = "/api/data-recipe/jobs/rec-1/cancel"
DOWNLOAD = "/api/hub/datasets/download/cancel"
STOPPING = "Stop requested. Training will stop at the next safe step."
PAYLOADS = {
    ("GET", "/api/train/status"): {"job_id": "job-7", "is_training_running": True},
    ("POST", "/api/train/stop"): {"status": "stopped", "message": STOPPING},
    ("POST", START): {
        "state": "rejected",
        "message": "Training start was cancelled",
        "error_code": "training_start_cancelled",
    },
    ("POST", "/api/train/diffusion/stop"): {"status": "stopping"},
    ("GET", "/api/export/status"): {"is_export_active": True, "active_op_kind": "export_gguf"},
    ("POST", "/api/export/cancel"): {"success": True, "message": "Export cancelled"},
    ("POST", RECIPE): {"job_id": "rec-1", "status": "cancelling"},
    ("POST", "/api/inference/images/generate/cancel"): {"cancelled": True},
    ("POST", "/api/inference/video/generate/cancel"): {"cancelled": True},
    ("POST", "/api/inference/cancel"): {"cancelled": 1},
    ("POST", DOWNLOAD): {"repo_id": "a/b", "state": "cancelling"},
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
        ({"kind": "training_start", "id": "req-1"}, START, None),
        ({"kind": "diffusion_training"}, "/api/train/diffusion/stop", {"save": True}),
        (
            {"kind": "diffusion_training", "save": False},
            "/api/train/diffusion/stop",
            {"save": False},
        ),
        ({"kind": "recipe", "id": "rec-1"}, RECIPE, None),
        ({"kind": "image"}, "/api/inference/images/generate/cancel", None),
        ({"kind": "video"}, "/api/inference/video/generate/cancel", None),
        ({"kind": "chat", "id": "mcp-abc"}, "/api/inference/cancel", {"cancel_id": "mcp-abc"}),
        ({"kind": "dataset_download", "id": "a/b"}, DOWNLOAD, {"repo_id": "a/b"}),
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
        "message": STOPPING,
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


# Between ops the worker is alive and holds a checkpoint; the cancel route would kill it. A job
# "exporting" waits for its own answer first (see the direct tests below).
@pytest.mark.parametrize(
    "phase,cancelled,message",
    [
        (None, False, "No export is running."),
        ("starting", True, "The export job was stopped before its next step."),
        ("loading", True, "The export job was stopped before its next step."),
    ],
)
def test_an_idle_export_worker_is_never_touched(monkeypatch, phase, cancelled, message):
    job = register_export_job(job_id = "exp-1", phase = phase) if phase else None
    args = {"kind": "export", **({"id": "exp-1"} if phase else {})}
    result, studio = _call(monkeypatch, args, {("GET", "/api/export/status"): IDLE_WORKER})
    assert (result["structuredContent"]["cancelled"], result["structuredContent"]["message"]) == (
        cancelled,
        message,
    )
    assert job is None or job.status == "cancelled"
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


@pytest.mark.parametrize(
    "op,cancelled,message",
    [
        ("export_gguf", True, None),
        ("cleanup", False, "The export running now is not this job; it was left alone."),
    ],
)
def test_an_export_starting_while_waiting_is_stopped_only_if_it_is_this_job(
    monkeypatch, op, cancelled, message
):
    then = {"is_export_active": True, "active_op_kind": op}
    job, result, calls = _cancel_directly(monkeypatch, _exports, statuses = (IDLE_WORKER, then))
    assert result.cancelled is cancelled
    assert message is None or result.message == message
    assert (("POST", "/api/export/cancel") in calls) is cancelled
    assert job.status == ("cancelled" if cancelled else "running")


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
    ("training", "/api/train/stop", {"status": "idle", "message": "No training job is running"}),
    ("training_start", START, {"state": "accepted", "message": "Started"}),
    # A start request rejected for another reason was not cancelled.
    ("training_start", START, {"state": "rejected", "error_code": "model_not_found"}),
    ("diffusion_training", "/api/train/diffusion/stop", {"status": "idle"}),
    IDLE_EXPORT,
    ("recipe", RECIPE, {"job_id": "rec-1", "status": "completed"}),
    ("recipe", RECIPE, {"job_id": "rec-1", "status": "cancelled"}),
    ("image", "/api/inference/images/generate/cancel", {"cancelled": False}),
    ("video", "/api/inference/video/generate/cancel", {"cancelled": False}),
    ("chat", "/api/inference/cancel", {"cancelled": 0}),
    ("dataset_download", DOWNLOAD, {"repo_id": "a/b", "state": "completed"}),
    ("dataset_download", DOWNLOAD, {"repo_id": "a/b", "state": "cancelled"}),
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
