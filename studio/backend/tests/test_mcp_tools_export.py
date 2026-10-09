# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import asyncio
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from fastmcp.exceptions import ToolError

from mcp_server import create_studio_mcp
from studio_mcp import export_jobs
from studio_mcp.export_jobs import ExportJob

from .mcp_harness import call_tool, fake_studio, served


@pytest.fixture(autouse = True)
def fresh_registry():
    export_jobs._reset()
    yield
    export_jobs._reset()


def _register(
    account_id = "owner",
    job_id = "job-a",
    **fields,
):
    job = ExportJob(job_id = job_id, account_id = account_id, format = "gguf", **fields)
    export_jobs._jobs[f"{account_id}:{job_id}"] = job
    return job


def _get(monkeypatch, studio, args):
    with TestClient(served(create_studio_mcp(), studio, monkeypatch = monkeypatch)) as http:
        return call_tool(http, "get_job", args)


def _status(payload):
    return fake_studio({("GET", "/api/export/status"): lambda request, body: payload})


def test_another_accounts_job_is_unknown(monkeypatch):
    _register(account_id = "alice-id", job_id = "job-a")
    result = _get(monkeypatch, _status({}), {"kind": "export", "id": "job-a"})
    assert result["isError"] is True
    assert result["content"][0]["text"] == "No such export job"
    with pytest.raises(ToolError):
        export_jobs.lookup("owner", "job-a")


def test_an_unknown_id_is_refused(monkeypatch):
    result = _get(monkeypatch, _status({}), {"kind": "export", "id": "nope"})
    assert result["content"][0]["text"] == "No such export job"


def test_a_running_job_is_not_settled_from_the_export_status(monkeypatch):
    # The status names whatever op ran last, here an export from the Export page; only the
    # job's own export call settles it.
    _register(phase = "exporting")
    status = {
        "last_op_seq": 4,
        "last_op_kind": "export_gguf",
        "last_op_status": "success",
        "last_op_output_path": "theirs",
        "is_export_active": False,
    }
    studio = _status(status)
    result = _get(monkeypatch, studio, {"kind": "export", "id": "job-a"})
    assert result["structuredContent"]["status"] == "running"
    assert result["structuredContent"]["export"]["output"] is None
    assert studio.state.calls == []


def test_a_failed_job_reports_its_error_scrubbed(monkeypatch):
    job = _register()
    export_jobs.finish(job, "failed", error = "Disk full writing /srv/exports/x.gguf")
    result = _get(monkeypatch, _status({}), {"kind": "export", "id": "job-a"})
    out = result["structuredContent"]
    assert out["status"] == "failed"
    assert out["error"].startswith("Disk full writing")
    assert "/srv" not in json.dumps(result)


def test_an_absolute_output_outside_exports_is_reduced_to_its_name():
    from utils.paths import exports_root

    assert (
        export_jobs.output_name(str(Path(exports_root()) / "my-model" / "gguf")) == "my-model/gguf"
    )
    assert export_jobs.output_name("/srv/elsewhere/my-gguf") == "my-gguf (outside exports folder)"
    assert export_jobs.output_name("C:\\Users\\me\\out\\model") == "model (outside exports folder)"
    assert export_jobs.output_name("my-model/gguf") == "my-model/gguf"


def test_settle_waits_for_the_job_to_finish_on_its_own():
    async def run():
        async def quick(job):
            await asyncio.sleep(0.01)

        async def hangs(job):
            await asyncio.Event().wait()

        done = export_jobs.start("owner", "gguf", quick)
        await export_jobs.settle(done, 1.0)
        stuck = export_jobs.start("owner", "lora", hangs)
        await export_jobs.settle(stuck, 0.01)
        status = stuck.status
        export_jobs.mark_cancelled(stuck)
        await asyncio.gather(stuck.task, return_exceptions = True)
        return done.status, status

    # settle never cancels: a job still busy after the wait is left running.
    assert asyncio.run(run()) == ("completed", "running")


def test_without_an_id_only_this_accounts_jobs_are_listed(monkeypatch):
    _register(job_id = "mine", status = "completed")
    _register(account_id = "alice-id", job_id = "theirs")
    result = _get(monkeypatch, _status({}), {"kind": "export"})
    assert result["structuredContent"]["jobs"] == [
        {"id": "mine", "status": "completed", "progress_percent": None}
    ]


def test_the_registry_holds_the_task_and_settles_it():
    async def run():
        async def work(job):
            job.phase = "exporting"
            await asyncio.sleep(0)

        job = export_jobs.start("owner", "gguf", work)
        assert job.task is not None and job.status == "running"
        await job.task
        return job

    job = asyncio.run(run())
    assert job.status == "completed"
    assert job.phase == "done"
    assert len(job.job_id) >= 16


def test_a_failing_task_is_failed_and_a_cancelled_one_cancelled():
    async def run():
        async def fails(job):
            raise ToolError("Checkpoint not found")

        async def hangs(job):
            await asyncio.sleep(3600)

        failed = export_jobs.start("owner", "gguf", fails)
        stuck = export_jobs.start("owner", "lora", hangs)
        await asyncio.sleep(0)
        export_jobs.mark_cancelled(stuck)
        await asyncio.gather(failed.task, stuck.task, return_exceptions = True)
        return failed, stuck

    failed, stuck = asyncio.run(run())
    assert (failed.status, failed.error) == ("failed", "Checkpoint not found")
    assert stuck.status == "cancelled"


def test_capacity_drops_the_oldest_finished_jobs_first():
    for index in range(export_jobs.CAPACITY):
        _register(job_id = f"done-{index}", status = "completed")
    running = _register(job_id = "running")
    export_jobs._evict()
    assert len(export_jobs._jobs) == export_jobs.CAPACITY
    assert "owner:done-0" not in export_jobs._jobs
    assert export_jobs.lookup("owner", "running") is running


# ---------------------------------------------------------------- export_model

OUT = "/srv/unsloth/outputs"
CHECKPOINTS = {
    "outputs_dir": OUT,
    "models": [
        {
            "name": "qwen-lora",
            "checkpoints": [
                {"display_name": "qwen-lora", "path": f"{OUT}/qwen-lora", "loss": 0.9},
                {
                    "display_name": "checkpoint-30",
                    "path": f"{OUT}/qwen-lora/checkpoint-30",
                    "loss": 1.1,
                },
            ],
        }
    ],
}


def _export_studio(
    *,
    export_answer = None,
    statuses = None,
    recorder = None,
):
    statuses = list(
        statuses
        or [
            {"last_op_seq": 5},
            {
                "last_op_seq": 6,
                "last_op_status": "success",
                "last_op_output_path": "my-gguf",
                "current_checkpoint": f"{OUT}/qwen-lora",
            },
        ]
    )
    # export_model first checks that no export is running.
    statuses.insert(0, {"is_export_active": False, "last_op_seq": 5})
    loaded = []

    def record(name, answer):
        def handler(request, body):
            if recorder is not None:
                recorder.append((name, json.loads(body) if body else None))
            if name == "load":
                loaded.append(json.loads(body)["checkpoint_path"])
            return answer

        return handler

    def status(request, body):
        answer = statuses.pop(0) if len(statuses) > 1 else statuses[0]
        # The status names whatever the last load put in the worker, unless a test says otherwise.
        if loaded and "current_checkpoint" not in answer:
            answer = {**answer, "current_checkpoint": loaded[-1]}
        return answer

    routes = {
        ("GET", "/api/models/checkpoints"): lambda request, body: CHECKPOINTS,
        ("GET", "/api/export/status"): status,
        ("POST", "/api/export/load-checkpoint"): record(
            "load", {"success": True, "message": "Loaded"}
        ),
    }
    for fmt in ("gguf", "merged", "lora", "base"):
        routes[("POST", f"/api/export/export/{fmt}")] = record(
            fmt,
            export_answer
            or {"success": True, "message": "Exported", "details": {"output_path": "my-gguf"}},
        )
    return fake_studio(routes)


def _run_export(
    monkeypatch,
    studio,
    args,
    polls = 20,
):
    with TestClient(served(create_studio_mcp(), studio, monkeypatch = monkeypatch)) as http:
        started = call_tool(http, "export_model", args)
        if started.get("isError"):
            return started, None
        job_id = started["structuredContent"]["job_id"]
        for _ in range(polls):
            job = call_tool(http, "get_job", {"kind": "export", "id": job_id})
            if job["structuredContent"]["status"] != "running":
                break
    return started, job


def test_an_export_finishing_after_this_one_does_not_become_its_result(monkeypatch):
    # The Export page's own export finished before the status was read again; this job keeps
    # the answer its own export call returned.
    studio = _export_studio(
        statuses = [
            {"last_op_seq": 5},
            {
                "last_op_seq": 7,
                "last_op_kind": "export_gguf",
                "last_op_status": "error",
                "last_op_error": "theirs failed",
                "current_checkpoint": f"{OUT}/qwen-lora",
            },
        ]
    )
    _started, job = _run_export(
        monkeypatch,
        studio,
        {"checkpoint": "qwen-lora", "format": "gguf", "save_directory": "my-gguf"},
    )
    assert job["structuredContent"]["status"] == "completed"
    assert job["structuredContent"]["error"] is None
    assert job["structuredContent"]["export"]["output"] == "my-gguf"


def test_export_returns_a_job_at_once_and_the_job_completes(monkeypatch):
    calls = []
    studio = _export_studio(recorder = calls)
    started, job = _run_export(
        monkeypatch,
        studio,
        {
            "checkpoint": "qwen-lora/checkpoint-30",
            "format": "gguf",
            "save_directory": "my-gguf",
            "quantization_method": ["Q4_K_M", "Q8_0"],
        },
    )
    assert started["structuredContent"]["status"] == "running"
    assert job["structuredContent"]["status"] == "completed"
    assert job["structuredContent"]["export"] == {
        "format": "gguf",
        "output": "my-gguf",
        "phase": "done",
    }
    assert calls == [
        ("load", {"checkpoint_path": f"{OUT}/qwen-lora/checkpoint-30", "max_seq_length": 2048}),
        (
            "gguf",
            {
                "save_directory": "my-gguf",
                "push_to_hub": False,
                "private": False,
                "quantization_method": ["Q4_K_M", "Q8_0"],
                "expected_checkpoint": f"{OUT}/qwen-lora/checkpoint-30",
            },
        ),
    ]
    assert OUT not in json.dumps([started, job])


def test_an_export_is_refused_while_another_one_runs(monkeypatch):
    studio = fake_studio(
        {
            ("GET", "/api/export/status"): lambda request, body: {
                "is_export_active": True,
                "active_op_kind": "export_gguf",
            },
            ("GET", "/api/models/checkpoints"): lambda request, body: CHECKPOINTS,
        }
    )
    with TestClient(served(create_studio_mcp(), studio, monkeypatch = monkeypatch)) as http:
        result = call_tool(
            http,
            "export_model",
            {"checkpoint": "qwen-lora", "format": "gguf", "save_directory": "out"},
        )
    assert result["isError"] is True
    assert result["content"][0]["text"].startswith("Another export is running.")
    assert [c[0] for c in studio.state.calls] == ["GET"]


def test_an_export_is_refused_while_an_mcp_export_job_runs(monkeypatch):
    _register(account_id = "alice-id", job_id = "job-a")
    calls = []
    result, _job = _run_export(
        monkeypatch,
        _export_studio(recorder = calls),
        {"checkpoint": "qwen-lora", "format": "gguf", "save_directory": "out"},
    )
    assert result["isError"] is True
    assert calls == []


def test_two_exports_at_once_cannot_both_start(monkeypatch):
    # Another call registers its job while this one awaits the checkpoint lookup.
    from studio_mcp.tools import export as export_tool

    real = export_tool.checkpoints.resolve

    async def racing(caller, name):
        found = await real(caller, name)
        _register(account_id = "owner", job_id = "first")
        return found

    monkeypatch.setattr(export_tool.checkpoints, "resolve", racing)
    calls = []
    result, _job = _run_export(
        monkeypatch,
        _export_studio(recorder = calls),
        {"checkpoint": "qwen-lora", "format": "gguf", "save_directory": "out"},
    )
    assert result["isError"] is True
    assert result["content"][0]["text"].startswith("Another export is running.")
    assert calls == []
    assert [job.job_id for job in export_jobs._jobs.values()] == ["first"]


def test_a_checkpoint_loaded_by_someone_else_in_between_is_never_exported(monkeypatch):
    calls = []
    swapped = [
        {"last_op_seq": 5, "current_checkpoint": f"{OUT}/someone-else"},
    ]
    started, job = _run_export(
        monkeypatch,
        _export_studio(recorder = calls, statuses = swapped),
        {"checkpoint": "qwen-lora", "format": "gguf", "save_directory": "out"},
    )
    assert started["isError"] is False
    assert job["structuredContent"]["status"] == "failed"
    assert "nothing was exported" in job["structuredContent"]["error"]
    assert [name for name, _ in calls] == ["load"]
    assert OUT not in json.dumps([started, job])


def test_each_format_reaches_its_route(monkeypatch):
    for fmt in ("merged", "lora", "base"):
        calls = []
        _run_export(
            monkeypatch,
            _export_studio(recorder = calls),
            {"checkpoint": "qwen-lora", "format": fmt, "save_directory": "out"},
        )
        assert [name for name, _ in calls] == ["load", fmt]
        assert "quantization_method" not in calls[1][1]


def test_hf_token_and_load_in_4bit_go_in_the_bodies(monkeypatch):
    calls = []
    _run_export(
        monkeypatch,
        _export_studio(recorder = calls),
        {
            "checkpoint": "qwen-lora",
            "format": "merged",
            "save_directory": "out",
            "push_to_hub": True,
            "repo_id": "me/m",
            "hf_token": "hf_x",
            "load_in_4bit": False,
        },
    )
    assert calls[0][1] == {
        "checkpoint_path": f"{OUT}/qwen-lora",
        "max_seq_length": 2048,
        "load_in_4bit": False,
        "hf_token": "hf_x",
    }
    assert calls[1][1]["hf_token"] == "hf_x"
    assert calls[1][1]["repo_id"] == "me/m"


def test_an_unknown_checkpoint_lists_the_names(monkeypatch):
    started, _job = _run_export(
        monkeypatch,
        _export_studio(),
        {"checkpoint": "nope", "format": "gguf", "save_directory": "out"},
    )
    assert started["isError"] is True
    assert (
        started["content"][0]["text"]
        == "No checkpoint named nope. Available: qwen-lora, qwen-lora/checkpoint-30"
    )


@pytest.mark.parametrize("directory", ["/srv/out", "C:\\out", "~/out", "a/../../etc", "..", "  "])
def test_save_directory_must_be_relative_and_inside(monkeypatch, directory):
    calls = []
    studio = _export_studio(recorder = calls)
    started, _job = _run_export(
        monkeypatch,
        studio,
        {"checkpoint": "qwen-lora", "format": "gguf", "save_directory": directory},
    )
    assert started["isError"] is True
    assert calls == []
    assert studio.state.calls == []


def test_a_failed_export_step_fails_the_job(monkeypatch):
    studio = _export_studio(
        export_answer = {"success": False, "message": "llama.cpp converter missing at /srv/llama"}
    )
    _started, job = _run_export(
        monkeypatch, studio, {"checkpoint": "qwen-lora", "format": "gguf", "save_directory": "out"}
    )
    assert job["structuredContent"]["status"] == "failed"
    assert job["structuredContent"]["error"].startswith("llama.cpp converter missing at")
    assert "/srv" not in json.dumps(job)


def test_export_model_annotations():
    tool = {t.name: t for t in asyncio.run(create_studio_mcp().list_tools())}["export_model"]
    assert tool.annotations.readOnlyHint is False
    assert tool.annotations.destructiveHint is False
    assert tool.annotations.openWorldHint is False
    assert tool.output_schema is not None


PAYLOADS = {
    ("GET", "/api/models/checkpoints"): CHECKPOINTS,
    ("GET", "/api/export/status"): {
        "last_op_seq": 1,
        "current_checkpoint": f"{OUT}/qwen-lora",
        "last_op_output_path": f"{OUT}/x",
    },
    ("POST", "/api/export/load-checkpoint"): {
        "success": True,
        "message": f"Loaded {OUT}/qwen-lora",
    },
    ("POST", "/api/export/export/gguf"): {"success": True, "message": f"Saved to {OUT}/x"},
}
