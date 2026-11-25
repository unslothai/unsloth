# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A parked follower wakes only for its own run's commits."""

import threading
import time

import pytest

from storage import chat_generation_runs_db as runs_db
from storage import studio_db


def _running(index):
    thread_id, user_id, run_id = f"thread-{index}", f"user-{index}", f"run-{index}"
    studio_db.upsert_chat_thread(
        {"id": thread_id, "title": "Chat", "modelType": "base", "modelId": "local", "createdAt": 1}
    )
    studio_db.upsert_chat_message(
        {
            "id": user_id,
            "threadId": thread_id,
            "role": "user",
            "content": [{"type": "text", "text": "Hello"}],
            "createdAt": 2,
        }
    )
    runs_db.create_run(
        run_id = run_id,
        owner_subject = "alice",
        thread_id = thread_id,
        user_message_id = user_id,
        assistant_message_id = f"assistant-{index}",
        request_payload = {"model": "local", "messages": [], "stream": True},
    )
    token = runs_db.get_worker_token(run_id)
    assert runs_db.mark_running(run_id, token) is True
    return run_id, token


@pytest.fixture
def two_runs(monkeypatch):
    streaming, parked = _running(1), _running(2)
    reads = []
    original = runs_db.list_events

    def counting(run_id, *args, **kwargs):
        reads.append(run_id)
        return original(run_id, *args, **kwargs)

    monkeypatch.setattr(runs_db, "list_events", counting)
    return streaming, parked, reads


def _park(
    run_id,
    reads,
    timeout = 10,
):
    cursor = int(runs_db.get_run_state(run_id)["lastEventSeq"])
    before = reads.count(run_id)
    result = {}

    def wait():
        started = time.monotonic()
        result["events"] = runs_db.wait_for_events(run_id, cursor, timeout, True)
        result["elapsed"] = time.monotonic() - started

    waiter = threading.Thread(target = wait, daemon = True)
    waiter.start()
    deadline = time.monotonic() + 5
    while reads.count(run_id) == before and time.monotonic() < deadline:
        time.sleep(0.005)
    time.sleep(0.05)
    return waiter, result


def test_other_runs_commits_do_not_wake_a_parked_follower(two_runs):
    (streaming, streaming_token), (parked, parked_token), reads = two_runs
    waiter, result = _park(parked, reads)
    parked_reads = reads.count(parked)

    for index in range(30):
        runs_db.append_events(streaming, streaming_token, [("chunk", {"n": index})])
    time.sleep(0.1)
    assert waiter.is_alive()
    assert reads.count(parked) == parked_reads

    runs_db.append_events(parked, parked_token, [("chunk", {"own": True})])
    waiter.join(5)
    assert not waiter.is_alive()
    assert [event["payloadJson"] for event in result["events"]] == ['{"own":true}']
    assert result["elapsed"] < 5


@pytest.mark.parametrize("transition", ["finish", "cancel", "reconcile"])
def test_own_state_transition_wakes_a_parked_follower_promptly(two_runs, transition):
    (streaming, streaming_token), (parked, parked_token), reads = two_runs
    waiter, result = _park(parked, reads)
    runs_db.append_events(streaming, streaming_token, [("chunk", {"n": 0})])
    time.sleep(0.05)
    assert waiter.is_alive()

    if transition == "finish":
        runs_db.finish_run(parked, worker_token = parked_token, status = "completed")
        expected = "run.completed"
    elif transition == "cancel":
        runs_db.request_cancel(parked)
        expected = "run.cancelling"
    else:
        runs_db.reconcile_runs()
        expected = "run.failed"
    waiter.join(5)
    assert not waiter.is_alive()
    assert [event["type"] for event in result["events"]] == [expected]
    assert result["elapsed"] < 5


def test_parked_waiters_leave_no_registration_behind(two_runs):
    (_, _), (parked, parked_token), reads = two_runs
    waiters = [_park(parked, reads) for _ in range(3)]
    runs_db.append_events(parked, parked_token, [("chunk", {"n": 1})])
    for waiter, result in waiters:
        waiter.join(5)
        assert [event["type"] for event in result["events"]] == ["chunk"]
    assert runs_db.wait_for_events(parked, 0, 0.01, True)
    assert runs_db.wait_for_events(parked, 10**6, 0.01, True) == []
    assert runs_db._RUN_WAITERS == {}
