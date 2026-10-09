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


def test_a_running_job_settles_from_the_export_status(monkeypatch):
    from utils.paths import exports_root

    inside = Path(exports_root()) / "my-model" / "gguf"
    _register(started_seq = 3, phase = "exporting")
    status = {
        "last_op_seq": 4,
        "last_op_status": "success",
        "last_op_output_path": str(inside),
        "is_export_active": False,
    }
    result = _get(monkeypatch, _status(status), {"kind": "export", "id": "job-a"})
    assert result["structuredContent"]["status"] == "completed"
    assert result["structuredContent"]["export"] == {
        "format": "gguf",
        "output": "my-model/gguf",
        "phase": "done",
    }


def test_an_op_from_before_the_job_does_not_settle_it(monkeypatch):
    _register(started_seq = 4, phase = "exporting")
    status = {"last_op_seq": 4, "last_op_status": "error", "last_op_error": "old failure"}
    result = _get(monkeypatch, _status(status), {"kind": "export", "id": "job-a"})
    assert result["structuredContent"]["status"] == "running"
    assert result["structuredContent"]["error"] is None


def test_a_failed_op_reports_its_error_scrubbed(monkeypatch):
    _register(started_seq = 1)
    status = {
        "last_op_seq": 2,
        "last_op_status": "error",
        "last_op_error": "Disk full writing /srv/exports/x.gguf",
    }
    result = _get(monkeypatch, _status(status), {"kind": "export", "id": "job-a"})
    out = result["structuredContent"]
    assert out["status"] == "failed"
    assert out["error"].startswith("Disk full writing")
    assert "/srv" not in json.dumps(result)


def test_an_absolute_output_outside_exports_is_reduced_to_its_name(monkeypatch):
    _register(started_seq = 1)
    status = {
        "last_op_seq": 2,
        "last_op_status": "success",
        "last_op_output_path": "/srv/elsewhere/my-gguf",
    }
    result = _get(monkeypatch, _status(status), {"kind": "export", "id": "job-a"})
    assert result["structuredContent"]["export"]["output"] == "my-gguf (outside exports folder)"
    assert "/srv" not in json.dumps(result)
    assert export_jobs.output_name("C:\\Users\\me\\out\\model") == "model (outside exports folder)"
    assert export_jobs.output_name("my-model/gguf") == "my-model/gguf"


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
