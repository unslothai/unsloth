# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import asyncio
import json
from pathlib import Path

import pytest
from fastmcp.exceptions import ToolError

from studio_mcp import export_jobs, job_store

from .mcp_harness import (
    OUT,
    RUN_CHECKPOINTS,
    fake_studio,
    register_export_job,
    run_export_job,
    run_tool,
    sequence,
)


@pytest.fixture(autouse = True)
def fresh_registry():
    export_jobs._reset()
    yield
    export_jobs._reset()


def _get(
    monkeypatch,
    args,
    status = None,
):
    return run_tool(monkeypatch, {("GET", "/api/export/status"): status or {}}, "get_job", args)


@pytest.mark.parametrize("job_id", ["job-a", "nope"])
def test_another_accounts_or_an_unknown_job_is_unknown(monkeypatch, job_id):
    register_export_job(account_id = "alice-id", job_id = "job-a")
    result, _studio = _get(monkeypatch, {"kind": "export", "id": job_id})
    assert result["isError"] is True
    assert result["content"][0]["text"] == "No such export job"
    with pytest.raises(ToolError):
        export_jobs.lookup("owner", "job-a")


def test_a_running_job_is_not_settled_from_the_export_status(monkeypatch):
    # The status names whatever op ran last (here the Export page's); only the job's own call settles it.
    register_export_job(phase = "exporting")
    status = {
        "last_op_seq": 4,
        "last_op_kind": "export_gguf",
        "last_op_status": "success",
        "last_op_output_path": "theirs",
        "is_export_active": False,
    }
    result, studio = _get(monkeypatch, {"kind": "export", "id": "job-a"}, status)
    assert result["structuredContent"]["status"] == "running"
    assert result["structuredContent"]["export"]["output"] is None
    assert studio.state.calls == []


def test_a_failed_job_reports_its_error_scrubbed(monkeypatch):
    job_store.finish(register_export_job(), "failed", error = "Disk full writing /srv/exports/x.gguf")
    result, _studio = _get(monkeypatch, {"kind": "export", "id": "job-a"})
    assert result["structuredContent"]["status"] == "failed"
    assert result["structuredContent"]["error"].startswith("Disk full writing")
    assert "/srv" not in json.dumps(result)


def test_an_absolute_output_outside_exports_is_reduced_to_its_name():
    from utils.paths import exports_root

    name = export_jobs.output_name
    assert name(str(Path(exports_root()) / "my-model" / "gguf")) == "my-model/gguf"
    assert name("/srv/elsewhere/my-gguf") == "my-gguf (outside exports folder)"
    assert name("C:\\Users\\me\\out\\model") == "model (outside exports folder)"
    assert name("my-model/gguf") == "my-model/gguf"


def test_the_registry_runs_settles_fails_and_cancels_tasks():
    async def run():
        async def quick(job):
            job.phase = "exporting"
            await asyncio.sleep(0.01)

        async def fails(job):
            raise ToolError("Checkpoint not found")

        async def hangs(job):
            await asyncio.Event().wait()

        done = export_jobs.start("owner", "gguf", quick)
        assert done.task is not None and done.status == "running"
        failed = export_jobs.start("owner", "gguf", fails)
        await job_store.settle(done, 1.0)
        stuck = export_jobs.start("owner", "lora", hangs)
        await job_store.settle(stuck, 0.01)
        # settle never cancels: a job still busy after the wait is left running.
        assert stuck.status == "running"
        job_store.mark_cancelled(stuck)
        await asyncio.gather(failed.task, stuck.task, return_exceptions = True)
        return done, failed, stuck

    done, failed, stuck = asyncio.run(run())
    assert (done.status, done.phase) == ("completed", "done")
    assert len(done.job_id) >= 16
    assert (failed.status, failed.error) == ("failed", "Checkpoint not found")
    assert stuck.status == "cancelled"


def test_without_an_id_only_this_accounts_jobs_are_listed(monkeypatch):
    register_export_job(job_id = "mine", status = "completed")
    register_export_job(account_id = "alice-id", job_id = "theirs")
    result, _studio = _get(monkeypatch, {"kind": "export"})
    assert result["structuredContent"]["jobs"] == [
        {"id": "mine", "status": "completed", "progress_percent": None}
    ]


def test_capacity_drops_the_oldest_finished_jobs_first():
    for index in range(export_jobs.CAPACITY):
        register_export_job(job_id = f"done-{index}", status = "completed")
    running = register_export_job(job_id = "running")
    export_jobs._evict()
    assert len(export_jobs._jobs) == export_jobs.CAPACITY
    assert "owner:done-0" not in export_jobs._jobs
    assert export_jobs.lookup("owner", "running") is running


CHECKPOINTS = {**RUN_CHECKPOINTS, "models": RUN_CHECKPOINTS["models"][:1]}
ARGS = {"checkpoint": "qwen-lora", "format": "gguf", "save_directory": "out"}
DONE = {
    "last_op_seq": 6,
    "last_op_status": "success",
    "last_op_output_path": "my-gguf",
    "current_checkpoint": f"{OUT}/qwen-lora",
}
EXPORTED = {"success": True, "message": "Exported", "details": {"output_path": "my-gguf"}}


def _export_studio(
    *,
    export_answer = EXPORTED,
    statuses = ({"last_op_seq": 5}, DONE),
    calls = None,
    op_ids = None,
):
    # export_model first checks that no export is running.
    next_status = sequence({"is_export_active": False, "last_op_seq": 5}, *statuses)
    calls = [] if calls is None else calls
    op_ids = [] if op_ids is None else op_ids

    def record(name, answer):
        def handler(request, body):
            sent = json.loads(body) if body else None
            if isinstance(sent, dict) and "op_id" in sent:
                op_ids.append((name, sent.pop("op_id")))
            calls.append((name, sent))
            return answer

        return handler

    def status(request, body):
        answer = next_status(request, body)
        # The worker holds whatever the last load put there, unless a test says otherwise.
        loads = [body["checkpoint_path"] for name, body in calls if name == "load"]
        if loads and "current_checkpoint" not in answer:
            answer = {**answer, "current_checkpoint": loads[-1]}
        return answer

    routes = {
        ("GET", "/api/models/checkpoints"): CHECKPOINTS,
        ("GET", "/api/export/status"): status,
        ("POST", "/api/export/load-checkpoint"): record("load", {"success": True, "message": "Ok"}),
    }
    for fmt in ("gguf", "merged", "lora", "base"):
        routes[("POST", f"/api/export/export/{fmt}")] = record(fmt, export_answer)
    return fake_studio(routes)


def test_an_export_finishing_after_this_one_does_not_become_its_result(monkeypatch):
    # The Export page's own export ended before the status was read again; the job keeps its answer.
    theirs = {
        "last_op_seq": 7,
        "last_op_kind": "export_gguf",
        "last_op_status": "error",
        "last_op_error": "theirs failed",
        "current_checkpoint": f"{OUT}/qwen-lora",
    }
    studio = _export_studio(statuses = [{"last_op_seq": 5}, theirs])
    _started, job = run_export_job(monkeypatch, studio, {**ARGS, "save_directory": "my-gguf"})
    out = job["structuredContent"]
    assert (out["status"], out["error"], out["export"]["output"]) == ("completed", None, "my-gguf")


def test_export_returns_a_job_at_once_and_the_job_completes(monkeypatch):
    calls = []
    path = f"{OUT}/qwen-lora/checkpoint-30"
    args = {**ARGS, "checkpoint": "qwen-lora/checkpoint-30", "save_directory": "my-gguf"}
    args["quantization_method"] = ["Q4_K_M", "Q8_0"]
    started, job = run_export_job(monkeypatch, _export_studio(calls = calls), args)
    assert started["structuredContent"]["status"] == "running"
    assert job["structuredContent"]["status"] == "completed"
    assert job["structuredContent"]["export"] == {
        "format": "gguf",
        "output": "my-gguf",
        "phase": "done",
    }
    assert calls == [
        ("load", {"checkpoint_path": path, "max_seq_length": 2048}),
        (
            "gguf",
            {
                "save_directory": "my-gguf",
                "push_to_hub": False,
                "private": False,
                "quantization_method": ["Q4_K_M", "Q8_0"],
                "expected_checkpoint": path,
            },
        ),
    ]
    assert OUT not in json.dumps([started, job])


def test_an_export_is_refused_while_another_one_runs(monkeypatch):
    routes = {
        ("GET", "/api/export/status"): {"is_export_active": True, "active_op_kind": "export_gguf"},
        ("GET", "/api/models/checkpoints"): CHECKPOINTS,
    }
    result, studio = run_tool(monkeypatch, routes, "export_model", ARGS)
    assert result["isError"] is True
    assert result["content"][0]["text"].startswith("Another export is running.")
    assert [c[0] for c in studio.state.calls] == ["GET"]


@pytest.mark.parametrize("race", [False, True])
def test_an_export_is_refused_while_an_mcp_export_job_runs(monkeypatch, race):
    from studio_mcp.tools import export as export_tool

    real = export_tool.checkpoints.resolve

    async def racing(caller, name):
        # Another call registers its job while this one awaits the checkpoint lookup.
        found = await real(caller, name)
        register_export_job(account_id = "owner", job_id = "first")
        return found

    if race:
        monkeypatch.setattr(export_tool.checkpoints, "resolve", racing)
    else:
        register_export_job(account_id = "alice-id", job_id = "first")
    calls = []
    result, _job = run_export_job(monkeypatch, _export_studio(calls = calls), ARGS)
    assert result["content"][0]["text"].startswith("Another export is running.")
    assert calls == []
    assert [job.job_id for job in export_jobs._jobs.values()] == ["first"]


@pytest.mark.parametrize(
    "studio_kw,error,names",
    [
        # A checkpoint loaded by someone else in between is never exported.
        (
            {"statuses": [{"last_op_seq": 5, "current_checkpoint": f"{OUT}/someone-else"}]},
            "nothing was exported",
            ["load"],
        ),
        (
            {
                "export_answer": {
                    "success": False,
                    "message": "llama.cpp converter missing at /srv/x",
                }
            },
            "llama.cpp converter missing at",
            ["load", "gguf"],
        ),
    ],
)
def test_a_failed_step_fails_the_job_without_leaking_paths(monkeypatch, studio_kw, error, names):
    calls = []
    started, job = run_export_job(monkeypatch, _export_studio(calls = calls, **studio_kw), ARGS)
    assert started["isError"] is False
    assert job["structuredContent"]["status"] == "failed"
    assert error in job["structuredContent"]["error"]
    assert [name for name, _ in calls] == names
    assert "/srv" not in json.dumps([started, job])


@pytest.mark.parametrize(
    "extra,expected",
    [
        ({"format": "merged"}, {}),
        ({"format": "lora"}, {}),
        ({"format": "base"}, {}),
        ({"imatrix": True}, {"imatrix": True}),
        (
            {"format": "merged", "push_to_hub": True, "repo_id": "me/m", "hf_token": "hf_x"},
            {"hf_token": "hf_x", "repo_id": "me/m"},
        ),
    ],
)
def test_each_format_reaches_its_route_with_its_body(monkeypatch, extra, expected):
    calls = []
    run_export_job(
        monkeypatch, _export_studio(calls = calls), {**ARGS, **extra, "load_in_4bit": False}
    )
    fmt = extra.get("format", "gguf")
    assert [name for name, _ in calls] == ["load", fmt]
    load = {"checkpoint_path": f"{OUT}/qwen-lora", "max_seq_length": 2048, "load_in_4bit": False}
    assert calls[0][1] == {**load, **({"hf_token": "hf_x"} if "hf_token" in extra else {})}
    assert calls[1][1].items() >= expected.items()
    assert ("quantization_method" in calls[1][1]) == (fmt == "gguf")


def test_an_unknown_checkpoint_lists_the_names(monkeypatch):
    started, _job = run_export_job(monkeypatch, _export_studio(), {**ARGS, "checkpoint": "nope"})
    assert started["isError"] is True
    assert (
        started["content"][0]["text"]
        == "No checkpoint named nope. Available: qwen-lora, qwen-lora/checkpoint-30"
    )


def test_each_step_carries_its_own_op_id(monkeypatch):
    # cancel(kind="export", id=...) sends the step's op id, so only this job's own op is ever stopped.
    op_ids = []
    started, _job = run_export_job(monkeypatch, _export_studio(op_ids = op_ids), ARGS)
    job_id = started["structuredContent"]["job_id"]
    assert op_ids == [("load", f"mcp-{job_id}-load"), ("gguf", f"mcp-{job_id}-export")]
