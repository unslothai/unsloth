# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The synthesis recovery pass must not destroy the report it exists to rescue."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from storage import research_runs_db as research_db
from storage import studio_db


FIRST_DRAFT = "## Findings\n\n" + ("The evidence says a great deal. " * 120).strip()
SHORTER_DRAFT = "## Findings\n\nToo little."


@pytest.fixture
def research_home(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    monkeypatch.setattr(studio_db, "_schema_ready", False)
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


def _claimed_run(supervisor) -> dict:
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
            "budgets": {
                "maxSteps": 1,
                "maxSources": 5,
                "modelTimeoutSeconds": 30,
                "toolTimeoutSeconds": 10,
            },
        },
    )
    planned = research_db.set_plan("run-1", {"title": "Plan", "steps": []})
    research_db.approve("run-1", planned["planRevision"], planned["planHash"])
    return research_db.claim_next(supervisor.worker_id)


def _run_synthesis(monkeypatch, *, synthesis, recovery) -> dict:
    from core import research_runs as worker

    supervisor = worker.ResearchSupervisor(SimpleNamespace(state = SimpleNamespace(server_port = 1)))
    claimed = _claimed_run(supervisor)
    phases: list[str] = []

    async def fake_stream_completion(run, messages, **kwargs):
        phase = kwargs.get("phase")
        phases.append(phase)
        if phase == "synthesis":
            return synthesis
        if phase == "synthesis_recovery":
            return recovery
        # Unparseable, and an empty plan has no seed action, so the step loop breaks.
        return "not json", "", "stop", None

    monkeypatch.setattr(supervisor, "_stream_completion", fake_stream_completion)
    asyncio.run(supervisor._research(claimed))
    finished = research_db.get_run("run-1")
    finished["phases"] = phases
    return finished


def test_an_empty_recovery_does_not_discard_the_report_it_was_rescuing(research_home, monkeypatch):
    finished = _run_synthesis(
        monkeypatch,
        synthesis = (FIRST_DRAFT, "", "length", {"completion_tokens": 16384}),
        recovery = ("", "", "stop", None),
    )

    assert "synthesis_recovery" in finished["phases"]
    assert finished["status"] == "completed"
    assert FIRST_DRAFT in finished["report"]


def test_two_length_stops_deliver_the_longer_draft_with_a_notice(research_home, monkeypatch):
    finished = _run_synthesis(
        monkeypatch,
        synthesis = (FIRST_DRAFT, "", "length", {"completion_tokens": 16384}),
        recovery = (SHORTER_DRAFT, "", "length", {"completion_tokens": 16384}),
    )

    assert finished["status"] == "completed"
    assert FIRST_DRAFT in finished["report"]
    assert "Incomplete report." in finished["report"]


def test_a_longer_recovery_still_wins(research_home, monkeypatch):
    recovered = FIRST_DRAFT + "\n\nAnd the conclusion."

    finished = _run_synthesis(
        monkeypatch,
        synthesis = (SHORTER_DRAFT, "", "length", {"completion_tokens": 16384}),
        recovery = (recovered, "", "stop", None),
    )

    assert finished["status"] == "completed"
    assert finished["report"].strip() == recovered
    assert "Incomplete report." not in finished["report"]


def test_a_complete_report_never_runs_the_recovery_pass(research_home, monkeypatch):
    finished = _run_synthesis(
        monkeypatch,
        synthesis = (FIRST_DRAFT, "", "stop", None),
        recovery = ("", "", "stop", None),
    )

    assert "synthesis_recovery" not in finished["phases"]
    assert finished["status"] == "completed"
    assert finished["report"].strip() == FIRST_DRAFT.strip()


def test_two_empty_attempts_still_fail_the_run(research_home, monkeypatch):
    with pytest.raises(ValueError, match = "no safely identifiable final report"):
        _run_synthesis(
            monkeypatch,
            synthesis = ("", "", "stop", None),
            recovery = ("", "", "stop", None),
        )


def test_a_complete_recovery_beats_a_longer_truncated_first_draft(research_home, monkeypatch):
    """`length` is the one finish reason that means the text is unfinished, so size is
    the wrong tiebreak: picking the longer draft delivered a truncated report, and
    labelled it incomplete, while a finished one was in hand."""
    complete = "## Findings\n\nDemand outran supply.\n\n## Conclusion\n\nDone."

    finished = _run_synthesis(
        monkeypatch,
        synthesis = (FIRST_DRAFT, "", "length", {"completion_tokens": 16384}),
        recovery = (complete, "", "stop", None),
    )

    assert finished["status"] == "completed"
    assert finished["report"].strip() == complete
    assert "Incomplete report." not in finished["report"]


def test_the_notice_survives_a_fence_the_truncation_left_open(research_home, monkeypatch):
    """Running out of budget mid-code-block leaves an unterminated fence, and in CommonMark
    that fence runs to the end of the document -- so an appended notice renders as code."""
    cut_off_in_a_fence = "## Findings\n\n```python\nctx = 32768,\nrope_scaling ="

    finished = _run_synthesis(
        monkeypatch,
        synthesis = (cut_off_in_a_fence, "", "length", {"completion_tokens": 16384}),
        recovery = ("", "", "stop", None),
    )

    report = finished["report"]
    assert "Incomplete report." in report
    assert report.count("```") % 2 == 0
    _, _, after_the_last_fence = report.rpartition("```")
    assert "> **Incomplete report.**" in after_the_last_fence


def test_a_report_that_closed_its_own_fence_gains_no_stray_one(research_home, monkeypatch):
    whole_fence = "## Findings\n\n```python\nctx = 32768\n```\n\nAnd it was cut off here"

    finished = _run_synthesis(
        monkeypatch,
        synthesis = (whole_fence, "", "length", {"completion_tokens": 16384}),
        recovery = ("", "", "stop", None),
    )

    assert finished["report"].count("```") == 2
    assert "Incomplete report." in finished["report"]
