# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A run whose every step gathered nothing must not be delivered as a finished report."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from storage import research_runs_db as research_db
from storage import studio_db


REPORT = "## Findings\n\nWritten from memory, because nothing came back."
THROTTLED = "Search failed: the search engines are rate limiting this machine."
FOUND = (
    "Title: What happened\nURL: https://example.test/what-happened\n"
    "Snippet: It happened on a Tuesday."
)


@pytest.fixture
def research_home(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    monkeypatch.setattr(studio_db, "_schema_ready", set())
    studio_db.upsert_chat_thread(
        {
            "id": "thread-1",
            "title": "Research",
            "modelType": "base",
            "modelId": "local-model",
            "createdAt": 1,
        }
    )
    studio_db.upsert_chat_message(
        {
            "id": "user-1",
            "threadId": "thread-1",
            "role": "user",
            "content": [{"type": "text", "text": "what happened?"}],
            "createdAt": 2,
        }
    )
    return tmp_path


def _claimed_run(
    supervisor, plan_steps: list[dict], max_steps: int, website_policy: dict | None
) -> dict:
    research_db.create_run(
        run_id = "run-1",
        owner_subject = "alice",
        thread_id = "thread-1",
        user_message_id = "user-1",
        assistant_message_id = None,
        config = {
            "model": "local-model",
            "inferenceRequest": {"model": "local-model"},
            "ragScope": None,
            "instructions": "",
            "question": "what happened?",
            "websitePolicy": website_policy,
            "budgets": {
                "maxSteps": max_steps,
                "maxSources": 5,
                "modelTimeoutSeconds": 900,
                "toolTimeoutSeconds": 10,
            },
        },
    )
    planned = research_db.set_plan("run-1", {"title": "Plan", "steps": plan_steps})
    research_db.approve("run-1", planned["planRevision"], planned["planHash"])
    return research_db.claim_next(supervisor.worker_id)


def _run(
    monkeypatch,
    tool_results: list[str],
    website_policy: dict | None = None,
) -> dict:
    from core import research_runs as worker

    supervisor = worker.ResearchSupervisor(SimpleNamespace(state = SimpleNamespace(server_port = 1)))
    plan_steps = [
        {"title": f"Step {index}", "query": f"what happened {index}"}
        for index in range(len(tool_results))
    ]
    claimed = _claimed_run(supervisor, plan_steps, len(tool_results), website_policy)
    calls = {"n": 0}

    def fake_execute_tool(name, arguments, **kwargs):
        result = tool_results[min(calls["n"], len(tool_results) - 1)]
        calls["n"] += 1
        return result

    async def fake_stream_completion(run, messages, **kwargs):
        if kwargs.get("phase") == "synthesis":
            return REPORT, "", "stop", None
        # Unparseable, so every step falls back to the next unused plan seed.
        return "not json", "", "stop", None

    monkeypatch.setattr(worker, "execute_tool", fake_execute_tool)
    monkeypatch.setattr(supervisor, "_stream_completion", fake_stream_completion)
    asyncio.run(supervisor._process(claimed))
    return research_db.get_run("run-1")


def test_a_run_whose_every_search_was_throttled_fails_with_the_search_error(
    research_home, monkeypatch
):
    finished = _run(monkeypatch, [THROTTLED, THROTTLED])

    assert finished["status"] == "failed"
    assert "rate limiting this machine" in (finished["error"] or "")
    assert not finished["report"]
    assert not finished["sources"]


def test_a_run_whose_every_search_matched_nothing_fails(research_home, monkeypatch):
    finished = _run(monkeypatch, ["No results found."])

    assert finished["status"] == "failed"
    assert not finished["report"]


def test_the_failure_reaches_the_chat_message(research_home, monkeypatch):
    _run(monkeypatch, [THROTTLED])

    messages = studio_db.list_chat_messages("thread-1")
    assistant = [message for message in messages if message["role"] == "assistant"][-1]
    assert assistant["metadata"]["researchStatus"] == "failed"
    assert "rate limiting this machine" in str(assistant["content"])


def test_a_partially_failed_run_still_completes(research_home, monkeypatch):
    finished = _run(monkeypatch, [FOUND, THROTTLED])

    assert finished["status"] == "completed"
    assert REPORT in finished["report"]
    assert len(finished["sources"]) == 1


def test_a_step_that_found_results_completes_the_run_even_with_no_citable_source(
    research_home, monkeypatch
):
    """A sweep the website policy filters down to nothing catalogable still gathered the
    evidence in its snippets, and its step is recorded completed."""
    finished = _run(monkeypatch, [FOUND], {"allowedDomains": ["allowed.test"]})

    assert finished["status"] == "completed"
    assert not finished["sources"]
    assert REPORT in finished["report"]


@pytest.mark.parametrize(
    ("restored_result", "expected_status"),
    [
        ({"excerpt": "It happened on a Tuesday."}, "completed"),
        ({}, "failed"),
    ],
)
def test_a_resumed_step_counts_only_if_its_evidence_survived(
    research_home, monkeypatch, restored_result, expected_status
):
    from core import research_runs as worker

    supervisor = worker.ResearchSupervisor(SimpleNamespace(state = SimpleNamespace(server_port = 1)))
    plan_steps = [
        {"title": "One", "query": "what happened 0"},
        {"title": "Two", "query": "what happened 1"},
    ]
    _claimed_run(supervisor, plan_steps, 2, None)
    research_db.upsert_execution_step(
        "run-1",
        0,
        "One",
        "what happened 0",
        "completed",
        {"action": "search", "input": "what happened 0", "sourceCount": 0, **restored_result},
        supervisor.worker_id,
    )
    conn = studio_db.get_connection()
    conn.execute(
        "UPDATE research_runs SET lease_owner = NULL, lease_expires_at = 0 WHERE id = 'run-1'"
    )
    conn.commit()
    conn.close()
    resumed = research_db.claim_next(supervisor.worker_id)
    assert resumed["claimedFromStatus"] == "running"

    async def fake_stream_completion(run, messages, **kwargs):
        if kwargs.get("phase") == "synthesis":
            return REPORT, "", "stop", None
        return "not json", "", "stop", None

    monkeypatch.setattr(worker, "execute_tool", lambda *args, **kwargs: THROTTLED)
    monkeypatch.setattr(supervisor, "_stream_completion", fake_stream_completion)
    asyncio.run(supervisor._process(resumed))

    assert research_db.get_run("run-1")["status"] == expected_status
