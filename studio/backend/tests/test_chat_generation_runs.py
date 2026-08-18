# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import json

import pytest

from routes.chat_generation_runs import (
    CreateChatGenerationRun,
    _contains_sensitive_key,
    _event_cursor,
    _sanitize_request,
)
from storage import chat_generation_runs_db as runs_db
from storage import studio_db


@pytest.fixture
def chat_home(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    monkeypatch.setattr(studio_db, "_schema_ready", False)
    studio_db.upsert_chat_thread(
        {
            "id": "thread-1",
            "title": "Chat",
            "modelType": "base",
            "modelId": "local",
            "createdAt": 1,
        }
    )
    studio_db.upsert_chat_message(
        {
            "id": "user-1",
            "threadId": "thread-1",
            "role": "user",
            "content": [{"type": "text", "text": "Hello"}],
            "createdAt": 2,
        }
    )


def _request(**overrides):
    request = {
        "model": "local",
        "messages": [{"role": "user", "content": "Hello"}],
        "stream": True,
    }
    request.update(overrides)
    return request


def _create(run_id = "run-1", owner = "alice", request = None):
    return runs_db.create_run(
        run_id = run_id,
        owner_subject = owner,
        thread_id = "thread-1",
        user_message_id = "user-1",
        assistant_message_id = "assistant-1" if run_id == "run-1" else f"assistant-{run_id}",
        request_payload = request or _request(),
    )


def test_create_is_owner_scoped_idempotent_and_binds_placeholder(chat_home):
    run, created = _create()
    replay, replay_created = _create()
    assert created is True and replay_created is False
    assert replay == run
    assert runs_db.get_run("run-1", "bob") is None
    assert runs_db.list_active("bob", "thread-1") == []
    message = studio_db.get_chat_message("thread-1", "assistant-1")
    assert message["metadata"] == {
        "generationRunId": "run-1",
        "generationSeq": 0,
        "generationStatus": "queued",
        "serverManaged": True,
    }
    with pytest.raises(studio_db.ChatMessageProtectedError):
        studio_db.upsert_chat_message(
            {
                **message,
                "content": [{"type": "text", "text": "stale overwrite"}],
            }
        )
    synced = studio_db.sync_chat_messages("thread-1", [], prune_missing = True)
    assert {message["id"] for message in synced} == {"user-1", "assistant-1"}
    assert runs_db.get_run("run-1", "alice") is not None
    with pytest.raises(runs_db.ChatGenerationConflictError):
        _create(owner = "bob")
    with pytest.raises(runs_db.ChatGenerationConflictError):
        _create(request = _request(max_tokens = 9))


def test_batched_events_have_gapless_cursor_and_terminal_flush(chat_home):
    run, _created = _create()
    worker_token = runs_db.get_worker_token("run-1")
    assert runs_db.mark_running("run-1", worker_token) is True
    assert runs_db.append_events(
        "run-1",
        worker_token,
        [("chunk", {"i": 1}), ("chunk", {"i": 2})],
    ) == [3, 4]
    cancelling = runs_db.request_cancel("run-1", "alice")
    assert cancelling["status"] == "cancelling"
    terminal = runs_db.finish_run(
        "run-1",
        worker_token = worker_token,
        status = "completed",
        finish_reason = "stop",
        pending_events = [("chunk", {"i": 3})],
    )
    assert terminal["status"] == "cancelled"
    events = runs_db.list_events("run-1")
    assert [event["seq"] for event in events] == list(range(1, 8))
    assert [event["payload"].get("i") for event in events if event["type"] == "chunk"] == [
        1,
        2,
        3,
    ]
    assert runs_db.list_events("run-1", after = 4)[0]["seq"] == 5
    assert runs_db.request_cancel("run-1", "alice")["lastEventSeq"] == 7


def test_cancel_before_registration_and_startup_orphan_reconciliation(chat_home):
    queued, _created = _create("queued")
    cancelled = runs_db.request_cancel("queued", "alice")
    assert cancelled["status"] == "cancelled"
    assert runs_db.mark_running("queued", runs_db.get_worker_token("queued")) is False

    orphan, _created = _create("orphan", request = _request(seed = 2))
    assert runs_db.mark_running("orphan", runs_db.get_worker_token("orphan")) is True
    assert runs_db.reconcile_orphaned_runs() == 1
    orphan = runs_db.get_run("orphan", "alice")
    assert (orphan["status"], orphan["finishReason"]) == ("failed", "interrupted")
    assert runs_db.list_events("orphan")[-1]["payload"]["interrupted"] is True


def test_deleted_run_id_is_tombstoned_against_stale_tabs(chat_home):
    _original, _created = _create()
    studio_db.delete_chat_threads(["thread-1"])
    studio_db.upsert_chat_thread(
        {
            "id": "thread-2",
            "title": "Next",
            "modelType": "base",
            "modelId": "local",
            "createdAt": 3,
        }
    )
    studio_db.upsert_chat_message(
        {
            "id": "user-2",
            "threadId": "thread-2",
            "role": "user",
            "content": [{"type": "text", "text": "Next"}],
            "createdAt": 4,
        }
    )
    with pytest.raises(runs_db.ChatGenerationConflictError, match = "already been used"):
        runs_db.create_run(
            run_id = "run-1",
            owner_subject = "alice",
            thread_id = "thread-2",
            user_message_id = "user-2",
            assistant_message_id = "assistant-2",
            request_payload = _request(seed = 2),
        )
    assert runs_db.get_run("run-1", "alice") is None


@pytest.mark.parametrize(
    "override,detail",
    [
        ({"provider_id": "external"}, "only for local"),
        ({"tools": [{"type": "function"}]}, "legacy streaming"),
        ({"rag_scope": {"access_token": "secret"}}, "Credentials"),
        ({"rag_scope": {"signing_key": "secret"}}, "Credentials"),
        ({"rag_scope": {"ssh_key": "secret"}}, "Credentials"),
        ({"rag_scope": {"encryption_key": "secret"}}, "Credentials"),
        ({"rag_scope": {"secret_key": "secret"}}, "Credentials"),
        ({"rag_scope": {"api_token": "secret"}}, "Credentials"),
        (
            {
                "messages": [
                    {
                        "role": "user",
                        "content": "Hello",
                        "extra_content": {"api_key": "secret"},
                    }
                ]
            },
            "Credentials",
        ),
        (
            {
                "messages": [
                    {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "call-1",
                                "type": "function",
                                "function": {
                                    "name": "lookup",
                                    "arguments": '{"api_key":"secret"}',
                                },
                            }
                        ],
                    }
                ]
            },
            "Credentials",
        ),
    ],
)
def test_request_sanitization_rejects_nonlocal_or_sensitive_payloads(override, detail):
    model = CreateChatGenerationRun(
        runId = "run-1",
        threadId = "thread-1",
        userMessageId = "user-1",
        assistantMessageId = "assistant-1",
        requestPayload = _request(**override),
    )
    with pytest.raises(Exception, match = detail):
        _sanitize_request(model)


def test_request_sanitization_pins_server_owned_fields():
    model = CreateChatGenerationRun(
        runId = "run-1",
        threadId = "thread-1",
        userMessageId = "user-1",
        assistantMessageId = "assistant-1",
        requestPayload = _request(stream = False, cancel_id = "legacy", thread_id = "wrong"),
    )
    sanitized = _sanitize_request(model)
    assert sanitized["stream"] is True
    assert sanitized["cancel_id"] == "run-1"
    assert sanitized["thread_id"] == "thread-1"


def test_request_sanitization_treats_message_text_as_data():
    model = CreateChatGenerationRun(
        runId = "run-1",
        threadId = "thread-1",
        userMessageId = "user-1",
        assistantMessageId = "assistant-1",
        requestPayload = _request(
            messages = [{"role": "user", "content": '{"api_key":"example"}'}]
        ),
    )
    assert _sanitize_request(model)["messages"][0]["content"] == '{"api_key":"example"}'


@pytest.mark.parametrize(
    "value", ['{"api_key":"secret"', '"{\\"api_key\\":\\"secret\\"}"']
)
def test_request_sanitization_scans_json_string_envelopes(value):
    assert _contains_sensitive_key(value) is True


def test_request_sanitization_bounds_nested_envelopes():
    nested = {"value": None}
    for _ in range(100):
        nested = {"value": nested}
    assert _contains_sensitive_key(nested) is True
    assert _contains_sensitive_key("[" * 5000 + "0" + "]" * 5000) is True
    model = CreateChatGenerationRun(
        runId = "run-1",
        threadId = "thread-1",
        userMessageId = "user-1",
        assistantMessageId = "assistant-1",
        requestPayload = _request(
            messages = [{"role": "user", "content": "hello", "extra_content": nested}]
        ),
    )
    with pytest.raises(Exception, match = "Credentials"):
        _sanitize_request(model)


@pytest.mark.parametrize("messages", [None, 1, [{"role": "user", "content": None}]])
def test_request_sanitization_returns_json_safe_validation_errors(messages):
    model = CreateChatGenerationRun(
        runId = "run-1",
        threadId = "thread-1",
        userMessageId = "user-1",
        assistantMessageId = "assistant-1",
        requestPayload = _request(messages = messages),
    )
    with pytest.raises(Exception) as exc_info:
        _sanitize_request(model)
    assert exc_info.value.status_code == 422
    json.dumps(exc_info.value.detail)


@pytest.mark.parametrize(
    "overrides",
    [
        {"provider_id": ""},
        {"provider_id": None, "encrypted_api_key": None},
        {"tools": []},
    ],
)
def test_request_sanitization_accepts_empty_optional_routing(overrides):
    model = CreateChatGenerationRun(
        runId = "run-1",
        threadId = "thread-1",
        userMessageId = "user-1",
        assistantMessageId = "assistant-1",
        requestPayload = _request(**overrides),
    )
    assert _sanitize_request(model)["stream"] is True


def test_event_cursor_rejects_values_outside_sqlite_integer_range():
    with pytest.raises(Exception, match = "cursor is too large"):
        _event_cursor(10**30, None)
    with pytest.raises(Exception, match = "cursor is too large"):
        _event_cursor(None, str(10**30))
    with pytest.raises(Exception, match = "must be an integer"):
        _event_cursor(None, "²")
    with pytest.raises(Exception, match = "cursor is too large"):
        _event_cursor(None, "9" * 4301)
