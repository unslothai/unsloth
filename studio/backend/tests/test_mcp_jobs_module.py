# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""get_job and cancel across kinds: what every kind shares, pinned through the tools."""

import pytest

from studio_mcp import export_jobs, job_store, loading
from studio_mcp.tools import audio

from .mcp_harness import fake_studio, register_export_job, run_tool, session

CANCEL = "/api/inference/audio/cancel"


@pytest.fixture(autouse = True)
def fresh_jobs():
    audio._RUNS.clear()
    audio._CANCELLED.clear()
    export_jobs._reset()
    loading.LOADS.reset()
    yield
    audio._RUNS.clear()
    audio._CANCELLED.clear()
    export_jobs._reset()
    loading.LOADS.reset()


def test_a_cancelled_audio_id_is_remembered_and_refused_for_a_new_run(monkeypatch):
    studio = fake_studio({("POST", CANCEL): {"cancelled": 0}})
    with session(monkeypatch, studio) as call:
        cancelled = call("cancel", {"kind": "audio", "id": "run-old-1"})
        again = call(
            "generate_audio", {"workflow": "speak", "text": "hi", "cancel_id": "run-old-1"}
        )
    assert cancelled["structuredContent"]["cancelled"] is False
    assert again["isError"] is True
    assert "was just cancelled; pass a new cancel_id" in again["content"][0]["text"]
    assert [c[1] for c in studio.state.calls] == [CANCEL]


def test_a_full_store_drops_the_oldest_finished_job_and_keeps_running_ones():
    store = job_store.JobStore(capacity = 2, unknown = "No such job", failure = "failed")
    running = store.add(job_store.Job(job_id = "a", account_id = "owner"))
    done = store.add(job_store.Job(job_id = "b", account_id = "owner", status = "completed"))
    later = store.add(job_store.Job(job_id = "c", account_id = "owner", status = "completed"))
    assert list(store.jobs.values()) == [running, later]
    store.add(job_store.Job(job_id = "d", account_id = "owner"))
    store.add(job_store.Job(job_id = "e", account_id = "owner"))
    # Nothing finished is left to drop, so the store grows past capacity rather than lose a running job.
    assert [job.job_id for job in store.jobs.values()] == ["a", "d", "e"]
    assert done not in store.jobs.values()


def test_an_export_job_of_another_account_reads_as_unknown(monkeypatch):
    register_export_job(account_id = "someone-else", job_id = "exp-x")
    result, _studio = run_tool(monkeypatch, {}, "get_job", {"kind": "export", "id": "exp-x"})
    assert result["isError"] is True
    assert result["content"][0]["text"] == "No such export job"


def test_a_cancel_of_an_unknown_export_job_reads_as_unknown(monkeypatch):
    result, studio = run_tool(monkeypatch, {}, "cancel", {"kind": "export", "id": "nope"})
    assert result["isError"] is True
    assert result["content"][0]["text"] == "No such export job"
    assert studio.state.calls == []


def test_a_finished_export_job_reads_with_no_poll_hint(monkeypatch):
    register_export_job(job_id = "exp-1", status = "completed", phase = "done", output = "out")
    result, _studio = run_tool(monkeypatch, {}, "get_job", {"kind": "export", "id": "exp-1"})
    job = result["structuredContent"]
    assert (job["kind"], job["id"], job["status"]) == ("export", "exp-1", "completed")
    assert job["export"] == {"format": "gguf", "output": "out", "phase": "done"}
    assert job["poll_after_s"] is None


def test_a_running_export_job_says_when_to_look_again(monkeypatch):
    register_export_job(job_id = "exp-1", phase = "exporting")
    result, _studio = run_tool(monkeypatch, {}, "get_job", {"kind": "export", "id": "exp-1"})
    assert result["structuredContent"]["poll_after_s"] == 15


def test_training_with_nothing_running_names_no_job(monkeypatch):
    routes = {("GET", "/api/train/status"): {"job_id": "old", "is_training_running": False}}
    result, studio = run_tool(monkeypatch, routes, "cancel", {"kind": "training"})
    assert result["structuredContent"] == {
        "kind": "training",
        "id": None,
        "cancelled": False,
        "message": "No training job is running.",
    }
    assert [c[1] for c in studio.state.calls] == ["/api/train/status"]


def test_a_registered_kind_is_reached_by_get_job_and_cancel(monkeypatch):
    import asyncio
    import types

    from studio_mcp import jobs
    from studio_mcp.media import media_result
    from studio_mcp.outputs import JobStatus
    from studio_mcp.tools import cancel as cancel_tool
    from studio_mcp.tools import jobs as jobs_tool

    seen = []

    async def read(caller, job_id, **_options):
        seen.append(("read", caller.account_id, job_id))
        return media_result([], JobStatus(kind = "video", id = job_id, status = "running"))

    async def stop(caller, job_id, save):
        seen.append(("cancel", caller.account_id, job_id, save))
        return jobs.Stop(True, "stopped", job_id)

    monkeypatch.setattr(jobs, "KINDS", dict(jobs.KINDS))
    jobs.register(jobs.Kind("fake", read = read, cancel = stop, poll_after_s = 7, needs_id = "its id"))
    owner = types.SimpleNamespace(account_id = "owner")
    monkeypatch.setattr(jobs_tool, "current_caller", lambda: owner)
    monkeypatch.setattr(cancel_tool, "current_caller", lambda: owner)

    read_back = asyncio.run(jobs_tool.get_job("fake", "f-1"))
    stopped = asyncio.run(cancel_tool.cancel("fake", "f-1", save = False))

    assert read_back.structured_content["poll_after_s"] == 7
    assert stopped.model_dump() == {
        "kind": "fake",
        "id": "f-1",
        "cancelled": True,
        "message": "stopped",
    }
    assert seen == [("read", "owner", "f-1"), ("cancel", "owner", "f-1", False)]
    with pytest.raises(Exception, match = "needs id: its id"):
        asyncio.run(cancel_tool.cancel("fake"))
