# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""get_job(wait_s): one call follows a job until it finishes or moves on; and the training and load kinds."""

import asyncio
import threading
import time

import pytest
from fastapi.responses import Response

from studio_mcp import export_jobs, loading

from .mcp_harness import (
    OUT,
    RUN_CHECKPOINTS,
    call_with_progress,
    fake_studio,
    fast_polls,  # noqa: F401  (fixture)
    register_export_job,
    run_tool,
    sequence,
    session,
)

VIDEO = ("GET", "/v1/videos/vid-1")
TRAIN = ("GET", "/api/train/status")
CHECKPOINTS = ("GET", "/api/models/checkpoints")
DIFFUSION = ("GET", "/api/train/diffusion/status")


@pytest.fixture(autouse = True)
def _clean_stores():
    loading.LOADS.reset()
    export_jobs._reset()
    yield
    loading.LOADS.reset()
    export_jobs._reset()


def _video(status, progress):
    return {"id": "vid-1", "status": status, "progress": progress}


def _counted(*answers):
    answer, reads = sequence(*answers), []

    def handler(request, body):
        reads.append(1)
        return answer(request, body)

    return handler, reads


def test_a_wait_returns_once_the_job_finishes_and_reports_progress(monkeypatch, fast_polls):
    handler, reads = _counted(
        _video("in_progress", 10), _video("in_progress", 40), _video("completed", 100)
    )
    studio = fake_studio(
        {VIDEO: handler, ("GET", "/v1/videos/vid-1/content"): Response(status_code = 404)}
    )
    progress, result = call_with_progress(
        monkeypatch, studio, "get_job", {"kind": "video", "id": "vid-1", "wait_s": 30}
    )
    assert result["structuredContent"]["status"] == "completed"
    assert result["structuredContent"]["poll_after_s"] is None
    assert len(reads) == 3
    assert [p["progress"] for p in progress] == [0.4]


@pytest.mark.parametrize("wait_s,reads_at_least", [(0, 1), (1, 3)])
def test_a_running_job_says_when_to_look_again(monkeypatch, fast_polls, wait_s, reads_at_least):
    monkeypatch.setattr(loading, "MAX_WAIT_S", 0.1)
    handler, reads = _counted(_video("in_progress", 10))
    result, _studio = run_tool(
        monkeypatch, {VIDEO: handler}, "get_job", {"kind": "video", "id": "vid-1", "wait_s": wait_s}
    )
    assert result["structuredContent"]["status"] == "in_progress"
    assert result["structuredContent"]["poll_after_s"] == 15
    assert len(reads) >= reads_at_least
    assert wait_s or len(reads) == 1


def test_a_wait_returns_on_a_status_change(monkeypatch, fast_polls):
    handler, reads = _counted(_video("queued", 0), _video("in_progress", 0))
    result, _studio = run_tool(
        monkeypatch, {VIDEO: handler}, "get_job", {"kind": "video", "id": "vid-1", "wait_s": 30}
    )
    assert result["structuredContent"]["status"] == "in_progress"
    # It moved on, so the next wait can start at once.
    assert result["structuredContent"]["poll_after_s"] == 0
    assert len(reads) == 2


def test_an_export_wait_returns_on_a_phase_change(monkeypatch, fast_polls):
    # Export has no percent: a wait ends when the job moves to its next step.
    job = register_export_job(phase = "loading")
    threading.Timer(0.1, lambda: setattr(job, "phase", "exporting")).start()
    with session(monkeypatch, fake_studio({})) as call:
        result = call("get_job", {"kind": "export", "id": job.job_id, "wait_s": 30})
    assert result["structuredContent"]["export"]["phase"] == "exporting"
    assert result["structuredContent"]["status"] == "running"


TRAINING = {
    "job_id": "job-7",
    "phase": "training",
    "is_training_running": True,
    "message": "Training",
    "details": {"step": 10, "total_steps": 30, "loss": 1.2},
}


def _done(output_dir):
    details = {**TRAINING["details"], "step": 30, "output_dir": output_dir}
    return {**TRAINING, "phase": "completed", "is_training_running": False, "details": details}


# An API key sees output_dir as an opaque reference; a local caller may see the path.
@pytest.mark.parametrize("opaque", [True, False])
def test_a_training_wait_returns_the_run_folder_and_its_checkpoints(
    monkeypatch, fast_polls, opaque
):
    from hub.utils.host_paths import cache_reference

    path = f"{OUT}/qwen-lora"
    routes = {
        TRAIN: sequence(TRAINING, _done(cache_reference(path) if opaque else path)),
        CHECKPOINTS: RUN_CHECKPOINTS,
    }
    result, _studio = run_tool(monkeypatch, routes, "get_job", {"kind": "training", "wait_s": 30})
    job = result["structuredContent"]
    assert (job["kind"], job["id"], job["status"]) == ("training", "job-7", "completed")
    assert job["training"]["run_folder"] == "qwen-lora"
    assert job["training"]["checkpoints"] == ["qwen-lora", "qwen-lora/checkpoint-30"]
    assert job["training"]["loss_history"] == 'list_training_runs(run_id="job-7")'
    assert "/srv" not in str(result)


def test_a_diffusion_training_run_has_no_llm_history_pointer(monkeypatch):
    routes = {
        ("GET", "/api/train/diffusion/status"): {
            "active": False,
            "job_id": "d-1",
            "status": "completed",
            "step": 5,
            "total_steps": 5,
            "output_dir": f"{OUT}/my-lora",
        },
        CHECKPOINTS: RUN_CHECKPOINTS,
    }
    result, _studio = run_tool(monkeypatch, routes, "get_job", {"kind": "diffusion_training"})
    job = result["structuredContent"]
    assert (job["status"], job["training"]["run_folder"]) == ("completed", "my-lora")
    assert job["training"]["loss_history"] is None


@pytest.mark.parametrize("kind,route", [("training", TRAIN), ("diffusion_training", DIFFUSION)])
def test_a_training_job_that_is_not_current_points_at_its_history(monkeypatch, kind, route):
    current = (
        TRAINING if kind == "training" else {"active": True, "job_id": "d-2", "status": "running"}
    )
    result, _studio = run_tool(
        monkeypatch, {route: current}, "get_job", {"kind": kind, "id": "job-1"}
    )
    assert result["isError"] is True
    text = result["content"][0]["text"]
    assert "job-1 is not the current training job" in text
    # list_training_runs reads LLM runs only.
    assert ('list_training_runs(run_id="job-1")' in text) is (kind == "training")


def test_an_unknown_load_job_and_the_listing(monkeypatch):
    with session(monkeypatch, fake_studio({})) as call:
        missing = call("get_job", {"kind": "load", "id": "nope"})
        listing = call("get_job", {"kind": "load"})
    assert missing["isError"] is True
    assert "No such load job" in missing["content"][0]["text"]
    assert listing["structuredContent"]["jobs"] == []


SCHEMA = [
    ("parameters/properties/wait_s/default", 0),
    ("parameters/properties/wait_s/maximum", 240),
    (
        "parameters/properties/kind/enum",
        ["video", "recipe", "export", "training", "diffusion_training", "load", "download"],
    ),
    ("output_schema/type", "object"),
]


@pytest.mark.parametrize("path,expected", SCHEMA)
def test_get_job_schema(path, expected):
    from mcp_server import create_studio_mcp

    tool = asyncio.run(create_studio_mcp().get_tool("get_job"))
    node = {"parameters": tool.parameters, "output_schema": tool.output_schema}
    for part in path.split("/"):
        node = node[part]
    assert node == expected


def test_a_recipe_wait_reads_its_rows_once(monkeypatch, fast_polls):
    status = "/api/data-recipe/jobs/rec-1/status"
    running = {"job_id": "rec-1", "status": "running", "progress": {"percent": 10}}
    done = {"job_id": "rec-1", "status": "completed", "progress": {"percent": 100}}
    pages, page = [], {"dataset": [{"q": "a"}], "total": 1}

    def dataset(request, body):
        pages.append(1)
        return page

    studio = fake_studio(
        {
            ("GET", status): sequence(running, running, running, done),
            ("GET", "/api/data-recipe/jobs/rec-1/dataset"): dataset,
        }
    )
    result, _studio = run_tool(
        monkeypatch, studio, "get_job", {"kind": "recipe", "id": "rec-1", "rows": 5, "wait_s": 30}
    )
    assert result["structuredContent"]["status"] == "completed"
    # The first read and the answer; none of the polls between.
    assert len(pages) == 2


def test_a_status_read_that_stalls_never_holds_a_wait_past_wait_s(monkeypatch, fast_polls):
    reads = []

    async def handler(request, body):
        reads.append(1)
        if len(reads) > 1:
            await asyncio.sleep(10)
        return _video("in_progress", 10)

    studio = fake_studio(
        {VIDEO: handler, ("GET", "/v1/videos/vid-1/content"): Response(status_code = 404)}
    )
    started = time.monotonic()
    result, _studio = run_tool(
        monkeypatch, studio, "get_job", {"kind": "video", "id": "vid-1", "wait_s": 1}
    )
    assert time.monotonic() - started < 5
    assert result["structuredContent"]["status"] == "in_progress"
