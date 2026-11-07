# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import asyncio
import types

import pytest
from fastmcp.exceptions import ToolError

from studio_mcp import export_jobs, jobs
from studio_mcp.media import media_result
from studio_mcp.outputs import JobStatus
from studio_mcp.tools import audio
from studio_mcp.tools import cancel as cancel_tool
from studio_mcp.tools import jobs as jobs_tool

from .mcp_harness import fake_studio, register_export_job, run_tool, session

CANCEL = "/api/inference/audio/cancel"


def _reset():
    audio._RUNS.clear()
    audio._CANCELLED.clear()
    export_jobs._reset()


@pytest.fixture(autouse = True)
def fresh_jobs():
    _reset()
    yield
    _reset()


def test_a_cancelled_audio_id_is_refused_for_a_new_run(monkeypatch):
    studio = fake_studio({("POST", CANCEL): {"cancelled": 0}})
    with session(monkeypatch, studio) as call:
        call("cancel", {"kind": "audio", "id": "run-old-1"})
        args = {"workflow": "speak", "text": "hi", "cancel_id": "run-old-1"}
        again = call("generate_audio", args)
    assert again["isError"] is True and "pass a new cancel_id" in again["content"][0]["text"]
    assert [c[1] for c in studio.state.calls] == [CANCEL]


@pytest.mark.parametrize("status, poll", [("completed", None), ("running", 15)])
def test_an_export_job_reads_with_a_poll_hint_only_while_running(monkeypatch, status, poll):
    register_export_job(job_id = "exp-1", status = status)
    result, _studio = run_tool(monkeypatch, {}, "get_job", {"kind": "export", "id": "exp-1"})
    job = result["structuredContent"]
    assert (job["status"], job["poll_after_s"]) == (status, poll)


def test_a_registered_kind_is_reached_by_get_job_and_cancel(monkeypatch):
    seen = []

    async def read(caller, job_id, **_options):
        seen.append(("read", caller.account_id, job_id))
        return media_result([], JobStatus(kind = "video", id = job_id, status = "running"))

    async def stop(caller, job_id, save):
        seen.append(("cancel", caller.account_id, job_id, save))
        return jobs.Stop(True, "stopped", job_id)

    monkeypatch.setattr(jobs, "KINDS", dict(jobs.KINDS))
    jobs.register(jobs.Kind("fake", read = read, cancel = stop, poll_after_s = 7, needs_id = "id"))
    owner = types.SimpleNamespace(account_id = "owner")
    for module in (jobs_tool, cancel_tool):
        monkeypatch.setattr(module, "current_caller", lambda: owner)

    assert asyncio.run(jobs_tool.get_job("fake", "f-1")).structured_content["poll_after_s"] == 7
    stopped = asyncio.run(cancel_tool.cancel("fake", "f-1", save = False))
    assert (stopped.kind, stopped.id, stopped.cancelled) == ("fake", "f-1", True)
    assert seen == [("read", "owner", "f-1"), ("cancel", "owner", "f-1", False)]
    with pytest.raises(ToolError, match = "needs id: id"):
        asyncio.run(cancel_tool.cancel("fake"))
    for tool in (jobs_tool.get_job("nope", "x"), cancel_tool.cancel("nope", "x")):
        with pytest.raises(ToolError, match = "Unknown job kind"):
            asyncio.run(tool)
