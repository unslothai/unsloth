# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A run whose owner chose "stop when I close the tab" ends once no tab follows it."""

import asyncio
import time
from types import SimpleNamespace

import pytest

from routes import chat_generation_runs as chat_routes
from routes import research_runs as research_routes
from state import run_subscribers, unattended_runs
from storage import chat_generation_runs_db as chat_db
from storage import research_runs_db as research_db
from storage import studio_db
from utils.account_context import current_account_id


@pytest.fixture(autouse = True)
def _clean_watchdog():
    unattended_runs.reset_for_tests()
    run_subscribers.reset_for_tests()
    yield
    unattended_runs.reset_for_tests()
    run_subscribers.reset_for_tests()


@pytest.fixture
def studio_home(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    monkeypatch.setattr(studio_db, "_schema_ready", set())
    studio_db.upsert_chat_thread(
        {
            "id": "thread-1",
            "title": "Chat",
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
            "content": [{"type": "text", "text": "What changed?"}],
            "createdAt": 2,
        }
    )
    studio_db.upsert_chat_message(
        {
            "id": "assistant-1",
            "threadId": "thread-1",
            "parentId": "user-1",
            "role": "assistant",
            "content": [],
            "createdAt": 3,
        }
    )
    return tmp_path


def _past_grace() -> float:
    return time.monotonic() + unattended_runs._GRACE_S + 1


def _watch(attendance_id = "run-1", *, settled = False):
    cancelled: list[str] = []

    async def cancel():
        cancelled.append(attendance_id)

    unattended_runs.watch(attendance_id, settled = lambda: settled, cancel = cancel)
    return cancelled


def test_an_unwatched_run_is_stopped_once_the_grace_runs_out():
    cancelled = _watch()
    assert asyncio.run(unattended_runs.check_once(time.monotonic())) == []
    assert asyncio.run(unattended_runs.check_once(_past_grace())) == ["run-1"]
    assert cancelled == ["run-1"]
    assert not unattended_runs.watched_for_tests("run-1")


def test_a_followed_run_keeps_going_and_its_grace_restarts_when_the_tab_goes():
    grace = unattended_runs._GRACE_S
    cancelled = _watch()
    run_subscribers.mark_subscriber_seen("run-1", "tab-a")
    seen_at = _past_grace()
    assert asyncio.run(unattended_runs.check_once(seen_at)) == []
    run_subscribers.subscriber_departed("run-1", "tab-a")
    # The grace counts from when the tab was last seen, not from when the run was watched.
    assert asyncio.run(unattended_runs.check_once(seen_at + grace / 2)) == []
    assert asyncio.run(unattended_runs.check_once(seen_at + grace + 1)) == ["run-1"]
    assert cancelled == ["run-1"]


def test_another_tab_still_following_keeps_the_run():
    cancelled = _watch()
    run_subscribers.mark_subscriber_seen("run-1", "tab-a")
    run_subscribers.mark_subscriber_seen("run-1", "tab-b")
    run_subscribers.subscriber_departed("run-1", "tab-a")
    assert asyncio.run(unattended_runs.check_once(_past_grace())) == []
    assert cancelled == []


def test_a_settled_run_is_dropped_without_a_cancel():
    cancelled = _watch(settled = True)
    assert asyncio.run(unattended_runs.check_once(_past_grace())) == []
    assert cancelled == []
    assert not unattended_runs.watched_for_tests("run-1")


def test_rewatching_keeps_the_original_grace():
    first = _watch()
    second = _watch()
    assert asyncio.run(unattended_runs.check_once(_past_grace())) == ["run-1"]
    assert first == ["run-1"] and second == []


def test_watches_are_scoped_to_their_account():
    cancelled = []

    async def cancel():
        cancelled.append("alice")

    unattended_runs.watch("run-1", "alice", settled = lambda: False, cancel = cancel)
    # Bob following his own run-1 says nothing about Alice's.
    run_subscribers.mark_subscriber_seen("run-1", "tab-b", "bob")
    assert asyncio.run(unattended_runs.check_once(_past_grace())) == ["run-1"]
    assert cancelled == ["alice"]


def test_a_failing_check_keeps_the_watch():
    def settled():
        raise RuntimeError("database is locked")

    async def cancel():
        raise AssertionError("must not cancel on a failed check")

    unattended_runs.watch("run-1", settled = settled, cancel = cancel)
    assert asyncio.run(unattended_runs.check_once(_past_grace())) == []
    assert unattended_runs.watched_for_tests("run-1")


def _research_payload(**overrides):
    payload = {
        "threadId": "thread-1",
        "userMessageId": "user-1",
        "inferenceRequest": {"model": "m"},
    }
    payload.update(overrides)
    return research_routes.CreateResearchRun(**payload)


def test_research_config_records_the_choice_only_when_made():
    thread = {"modelId": "m"}
    assert "stopWhenUnattended" not in research_routes._sanitize_config(_research_payload(), thread)
    config = research_routes._sanitize_config(_research_payload(stopWhenUnattended = True), thread)
    assert config["stopWhenUnattended"] is True


def _research_run(*, stop_when_unattended: bool):
    config = {
        "model": "local-model",
        "inferenceRequest": {"model": "local-model"},
        "budgets": {"maxSteps": 5, "maxSources": 15, "modelTimeoutSeconds": 30},
    }
    if stop_when_unattended:
        config["stopWhenUnattended"] = True
    return research_db.create_run(
        run_id = "run-1",
        owner_subject = "alice",
        thread_id = "thread-1",
        user_message_id = "user-1",
        assistant_message_id = "assistant-1",
        config = config,
        created_at = 10,
    )


def _research_request():
    cancelled: list[str] = []
    request = SimpleNamespace(
        app = SimpleNamespace(
            state = SimpleNamespace(research_supervisor = SimpleNamespace(cancel = cancelled.append))
        )
    )
    return cancelled, request


def test_research_left_running_is_not_watched(studio_home):
    _cancelled, request = _research_request()
    research_routes._watch_if_unattended_stop(request, _research_run(stop_when_unattended = False))
    assert not unattended_runs.watched_for_tests("research:run-1", current_account_id() or "")


def test_research_nobody_follows_is_cancelled(studio_home):
    cancelled, request = _research_request()
    research_routes._watch_if_unattended_stop(request, _research_run(stop_when_unattended = True))
    assert unattended_runs.watched_for_tests("research:run-1", current_account_id() or "")
    assert asyncio.run(unattended_runs.check_once(_past_grace())) == ["research:run-1"]
    run = research_db.get_run("run-1")
    assert run["status"] in {"cancelling", "cancelled"}
    assert cancelled == (["run-1"] if run["status"] == "cancelling" else [])


def test_followed_research_keeps_going(studio_home):
    cancelled, request = _research_request()
    research_routes._watch_if_unattended_stop(request, _research_run(stop_when_unattended = True))
    # Stamped the way the events stream stamps it, under the follower's account.
    run_subscribers.mark_subscriber_seen("research:run-1", "tab-a", current_account_id() or "")
    assert asyncio.run(unattended_runs.check_once(_past_grace())) == []
    assert research_db.get_run("run-1")["status"] not in research_db.TERMINAL_STATUSES
    assert cancelled == []


def test_chat_admission_takes_the_choice_and_defaults_to_running_on():
    base = {
        "runId": "run-1",
        "threadId": "thread-1",
        "userMessageId": "user-1",
        "assistantMessageId": "assistant-1",
        "requestPayload": {"model": "local-model", "messages": []},
    }
    assert chat_routes.CreateChatGenerationRun(**base).stopWhenUnattended is False
    assert chat_routes.CreateChatGenerationRun(**base, stopWhenUnattended = True).stopWhenUnattended


def test_chat_run_nobody_follows_is_cancelled(studio_home):
    run, _created = chat_db.create_run(
        run_id = "run-1",
        owner_subject = "alice",
        thread_id = "thread-1",
        user_message_id = "user-1",
        assistant_message_id = "assistant-1",
        request_payload = {"model": "local-model", "messages": [], "stream": True},
    )
    assert run["status"] == "queued"
    cancelled: list[str] = []
    request = SimpleNamespace(
        app = SimpleNamespace(
            state = SimpleNamespace(
                chat_generation_supervisor = SimpleNamespace(cancel = cancelled.append)
            )
        )
    )
    chat_routes._watch_unattended(request, "run-1")
    assert asyncio.run(unattended_runs.check_once(_past_grace())) == ["run-1"]
    assert chat_db.get_run("run-1")["status"] == "cancelled"
    assert cancelled == ["run-1"]
    # Settled, so a second pass has nothing left to stop.
    assert asyncio.run(unattended_runs.check_once(_past_grace())) == []


def test_the_research_event_stream_counts_as_a_tab_following_the_run(studio_home):
    _cancelled, request = _research_request()
    run = _research_run(stop_when_unattended = True)

    async def is_disconnected():
        return False

    request.is_disconnected = is_disconnected
    account = current_account_id() or ""

    async def follow_once():
        response = await research_routes.research_events(
            run["id"], request, after = 0, last_event_id = None, current_subject = "alice"
        )
        frames = response.body_iterator
        await frames.__anext__()
        attended = run_subscribers.attendance_for_tests("research:run-1", account)
        await frames.aclose()
        return attended

    assert asyncio.run(follow_once()) == 1
    # Closing the stream lets the grace start now rather than when the stamp ages out.
    assert run_subscribers.attendance_for_tests("research:run-1", account) == 0
    # Attaching also arms the watch, which is how a run that outlived a restart gets one back.
    assert unattended_runs.watched_for_tests("research:run-1", account)
