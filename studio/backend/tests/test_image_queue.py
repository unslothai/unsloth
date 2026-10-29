# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Image generation queue: run order, reorder, cancel and clear, with the generator replaced by a gated fake."""

from __future__ import annotations

import asyncio
import types

import pytest

from core.inference import image_queue
from core.inference.image_queue import ImageGenerationQueue, JobCancelled

OWNER = types.SimpleNamespace(account_id = "owner")
OTHER = types.SimpleNamespace(account_id = "other")


class _GatedRunner:
    """Each job runs until the test releases it, so the queue can be edited mid-run."""

    def __init__(self) -> None:
        self.started: list[str] = []
        self.gates: dict[str, asyncio.Event] = {}
        self.outcomes: dict[str, Exception] = {}

    def gate(self, prompt: str) -> asyncio.Event:
        return self.gates.setdefault(prompt, asyncio.Event())

    async def __call__(self, job) -> list[str]:
        prompt = job.request.prompt
        self.started.append(prompt)
        await self.gate(prompt).wait()
        if prompt in self.outcomes:
            raise self.outcomes[prompt]
        return [f"img-{prompt}"]


def _request(prompt: str):
    return types.SimpleNamespace(prompt = prompt)


async def _settle():
    for _ in range(5):
        await asyncio.sleep(0)


def _statuses(queue, account = OWNER):
    return {job.request.prompt: job.status for job in queue.jobs(account.account_id)}


def test_jobs_run_one_at_a_time_in_order():
    async def scenario():
        runner = _GatedRunner()
        queue = ImageGenerationQueue(runner)
        for prompt in ("a", "b", "c"):
            queue.enqueue(OWNER, _request(prompt))
        await _settle()
        assert runner.started == ["a"]
        assert _statuses(queue) == {"a": "running", "b": "queued", "c": "queued"}
        for prompt in ("a", "b", "c"):
            runner.gate(prompt).set()
            await _settle()
        assert runner.started == ["a", "b", "c"]
        assert _statuses(queue) == {"a": "done", "b": "done", "c": "done"}
        assert [job.image_ids for job in queue.jobs("owner")] == [["img-a"], ["img-b"], ["img-c"]]

    asyncio.run(scenario())


def test_reorder_and_cancel_change_what_runs_next():
    async def scenario():
        runner = _GatedRunner()
        queue = ImageGenerationQueue(runner)
        jobs = {p: queue.enqueue(OWNER, _request(p)) for p in ("a", "b", "c", "d")}
        await _settle()
        queue.move("owner", jobs["d"].id, 0)
        assert [queue.position(jobs[p]) for p in ("d", "b", "c")] == [0, 1, 2]
        assert queue.cancel("owner", jobs["b"].id).status == "cancelled"
        for prompt in ("a", "d", "c"):
            runner.gate(prompt).set()
            await _settle()
        assert runner.started == ["a", "d", "c"]
        assert _statuses(queue) == {"a": "done", "b": "cancelled", "c": "done", "d": "done"}

    asyncio.run(scenario())


def test_move_clamps_and_refuses_a_running_job():
    async def scenario():
        runner = _GatedRunner()
        queue = ImageGenerationQueue(runner)
        a, b, c = (queue.enqueue(OWNER, _request(p)) for p in ("a", "b", "c"))
        await _settle()
        assert queue.move("owner", a.id, 0) is None
        queue.move("owner", b.id, 99)
        assert [queue.position(j) for j in (c, b)] == [0, 1]
        for gate in ("a", "b", "c"):
            runner.gate(gate).set()
        await _settle()

    asyncio.run(scenario())


def test_cancelling_the_running_job_only_flags_it_and_the_queue_moves_on():
    async def scenario():
        runner = _GatedRunner()
        queue = ImageGenerationQueue(runner)
        a = queue.enqueue(OWNER, _request("a"))
        queue.enqueue(OWNER, _request("b"))
        await _settle()
        assert queue.cancel("owner", a.id).status == "running"
        assert a.cancel_requested
        # The engine's stop surfaces as the run raising.
        runner.outcomes["a"] = RuntimeError("Generation cancelled.")
        runner.gate("a").set()
        await _settle()
        assert _statuses(queue) == {"a": "cancelled", "b": "running"}
        runner.gate("b").set()
        await _settle()

    asyncio.run(scenario())


def test_a_failed_job_records_its_error_and_the_next_one_still_runs():
    async def scenario():
        runner = _GatedRunner()
        queue = ImageGenerationQueue(runner)
        a = queue.enqueue(OWNER, _request("a"))
        queue.enqueue(OWNER, _request("b"))
        runner.outcomes["a"] = RuntimeError("out of memory")
        runner.outcomes["b"] = JobCancelled()
        runner.gate("a").set()
        runner.gate("b").set()
        await _settle()
        assert a.status == "failed" and a.error == "out of memory"
        assert _statuses(queue)["b"] == "cancelled"

    asyncio.run(scenario())


def test_accounts_see_and_edit_only_their_own_jobs():
    async def scenario():
        runner = _GatedRunner()
        queue = ImageGenerationQueue(runner)
        mine = queue.enqueue(OWNER, _request("a"))
        theirs = queue.enqueue(OTHER, _request("b"))
        await _settle()
        assert [j.id for j in queue.jobs("owner")] == [mine.id]
        assert queue.cancel("owner", theirs.id) is None
        assert queue.move("owner", theirs.id, 0) is None
        assert queue.clear("owner") == 0  # its only job is running
        assert theirs.status == "queued"
        runner.gate("a").set()
        runner.gate("b").set()
        await _settle()

    asyncio.run(scenario())


def test_clear_keeps_the_running_job_and_drops_the_rest():
    async def scenario():
        runner = _GatedRunner()
        queue = ImageGenerationQueue(runner)
        a = queue.enqueue(OWNER, _request("a"))
        queue.enqueue(OWNER, _request("b"))
        queue.enqueue(OWNER, _request("c"))
        await _settle()
        assert queue.clear("owner") == 2
        assert queue.jobs("owner") == [a]
        runner.gate("a").set()
        await _settle()
        assert runner.started == ["a"]

    asyncio.run(scenario())


def test_the_worker_waits_for_a_direct_run_before_starting(monkeypatch):
    async def scenario():
        runner = _GatedRunner()
        idle = asyncio.Event()

        async def wait_idle():
            await idle.wait()

        queue = ImageGenerationQueue(runner, wait_idle)
        a = queue.enqueue(OWNER, _request("a"))
        await _settle()
        assert runner.started == [] and a.status == "queued"
        idle.set()
        runner.gate("a").set()
        await _settle()
        assert a.status == "done"

    asyncio.run(scenario())


def test_queue_is_capped_and_finished_history_trimmed(monkeypatch):
    monkeypatch.setattr(image_queue, "MAX_QUEUED_PER_ACCOUNT", 2)
    monkeypatch.setattr(image_queue, "MAX_FINISHED_PER_ACCOUNT", 1)

    async def scenario():
        runner = _GatedRunner()
        queue = ImageGenerationQueue(runner)
        queue.enqueue(OWNER, _request("a"))
        await _settle()
        queue.enqueue(OWNER, _request("b"))
        queue.enqueue(OWNER, _request("c"))
        with pytest.raises(image_queue.QueueFull):
            queue.enqueue(OWNER, _request("d"))
        for gate in ("a", "b", "c"):
            runner.gate(gate).set()
        await _settle()
        assert [j.request.prompt for j in queue.jobs("owner")] == ["c"]

    asyncio.run(scenario())
