# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Account isolation at chat, settings, cancellation and worker boundaries."""

import asyncio
import hashlib
import json
import threading
from types import SimpleNamespace

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from auth import policy
from auth.authentication import authenticated_via_api_key, get_current_subject
from routes import (
    chat_generation_runs,
    chat_history,
    profile_stats,
    prompts,
    research_runs,
    settings,
)
from state import active_generations
from storage import (
    api_usage_db,
    chat_generation_runs_db,
    profile_stats_db,
    research_runs_db,
    studio_db,
)
from utils import keyless_api_access
from utils.account_context import (
    OWNER,
    AccountContext,
    arun_as,
    bind_account,
    current_account_id,
    reset_account,
    run_as,
)
from utils.paths import studio_db_path

ALICE = AccountContext("a" * 32, "alice")
BOB = AccountContext("b" * 32, "bob")
ACCOUNTS = {account.username: account for account in (OWNER, ALICE, BOB)}


@pytest.fixture(autouse = True)
def account_databases(monkeypatch):
    monkeypatch.setattr(policy, "installation_is_multi_user", lambda: True)
    # Worker 02 owns schema-cache partitioning. Initialize each database through
    # existing public entry points so these route tests do not depend on its merge.
    for account in ACCOUNTS.values():
        monkeypatch.setattr(studio_db, "_schema_ready", False)
        monkeypatch.setattr(chat_generation_runs_db, "_schema_ready", False)
        run_as(account, chat_generation_runs_db.get_run, "initialize-schema")
    active_generations.reset_for_tests()
    keyless_api_access._reset_scope_cache()
    profile_stats_db.invalidate_profile_stats_cache()
    yield
    active_generations.reset_for_tests()
    keyless_api_access._reset_scope_cache()
    profile_stats_db.invalidate_profile_stats_cache()


@pytest.fixture
def client(monkeypatch):
    app = FastAPI()

    async def subject(request: Request):
        account = ACCOUNTS[request.headers.get("x-test-account", "alice")]
        marker = bind_account(account)
        try:
            yield account.username
        finally:
            reset_account(marker)

    app.dependency_overrides[get_current_subject] = subject
    app.dependency_overrides[authenticated_via_api_key] = lambda: False
    app.include_router(chat_history.router, prefix = "/chat")
    app.include_router(chat_generation_runs.router, prefix = "/runs")
    app.include_router(settings.router, prefix = "/settings")
    app.include_router(prompts.router, prefix = "/prompts")
    app.include_router(profile_stats.router, prefix = "/profile")
    app.include_router(research_runs.router, prefix = "/research")

    # Filesystem sandboxes and RAG cleanup belong to the other workers. These
    # tests exercise real account databases and the shared generation registry.
    async def remove_sandboxes(*args):
        return 0, []

    monkeypatch.setattr(chat_history, "_remove_sandboxes", remove_sandboxes)
    monkeypatch.setattr(chat_history, "_remove_conversation_archives", lambda *args, **kwargs: None)
    with TestClient(app) as test_client:
        yield test_client


def seed_chat(account, thread_id = "private", text = "private text"):
    def seed():
        studio_db.upsert_chat_thread(
            {
                "id": thread_id,
                "title": text,
                "modelType": "base",
                "modelId": text,
                "createdAt": 1,
            }
        )
        studio_db.upsert_chat_message(
            {
                "id": "message",
                "threadId": thread_id,
                "role": "user",
                "createdAt": 2,
                "content": [{"type": "text", "text": text}],
                "attachments": [
                    {
                        "id": "attachment",
                        "name": "private.txt",
                        "type": "file",
                        "contentType": "text/plain",
                        "content": [{"type": "text", "text": text}],
                    }
                ],
            }
        )

    run_as(account, seed)


def seed_run(account, run_id = "run", thread_id = "private"):
    return run_as(
        account,
        chat_generation_runs_db.create_run,
        run_id = run_id,
        owner_subject = account.username,
        thread_id = thread_id,
        user_message_id = "message",
        assistant_message_id = "reply",
        request_payload = {"model": "local", "messages": [{"role": "user", "content": "Hello"}]},
    )[0]


@pytest.mark.parametrize(
    "method,path,body",
    [
        ("GET", "/chat/threads/private", None),
        ("PATCH", "/chat/threads/private", {"title": "stolen"}),
        ("GET", "/chat/threads/private/messages", None),
        ("GET", "/chat/threads/private/messages/message", None),
        (
            "PUT",
            "/chat/threads/private/messages/message",
            {
                "id": "message",
                "threadId": "private",
                "role": "user",
                "createdAt": 2,
                "attachments": [{"id": "attachment", "name": "renamed.txt"}],
            },
        ),
        (
            "POST",
            "/chat/threads/private/fork",
            {
                "messageId": "message",
                "newThreadId": "stolen",
                "createdAt": 3,
            },
        ),
        ("GET", "/chat/attachments/message/attachment/file", None),
        ("DELETE", "/chat/attachments/message/attachment", None),
        ("GET", "/runs/run", None),
        ("POST", "/runs/run/cancel", None),
        ("POST", "/runs/run/events", None),
    ],
)
def test_foreign_ids_do_not_authorize_reads_or_writes(client, method, path, body):
    seed_chat(BOB)
    seed_run(BOB)
    response = client.request(method, path, json = body)
    assert response.status_code == 404, response.text
    assert run_as(BOB, studio_db.get_chat_thread, "private")["title"] == "private text"
    assert run_as(BOB, chat_generation_runs_db.get_run, "run")["status"] == "queued"


def test_list_export_and_delete_stay_in_the_account(client):
    seed_chat(BOB)
    seed_run(BOB)
    assert client.get("/chat/threads").json() == {"threads": []}
    assert client.get("/runs/active?threadId=private").json() == {"runs": []}
    assert client.get("/chat/attachments").json()["attachments"] == []
    assert client.get("/chat/export").json()["threadCount"] == 0
    bob_event = threading.Event()
    with run_as(
        BOB, active_generations.ActiveGeneration, bob_event, thread_id = "private", run_id = "run"
    ):
        assert (
            client.request("DELETE", "/chat/threads", json = {"ids": ["private"]}).status_code == 200
        )
        assert not bob_event.is_set()
    assert run_as(BOB, studio_db.get_chat_thread, "private") is not None
    assert run_as(BOB, studio_db.get_chat_attachment, "message", "attachment") is not None


@pytest.mark.parametrize("supervisor_name", ["chat_generation_supervisor", "research_supervisor"])
def test_cancellation_never_uses_an_unscoped_supervisor(supervisor_name):
    calls = []
    request = SimpleNamespace(
        app = SimpleNamespace(
            state = SimpleNamespace(
                **{
                    supervisor_name: SimpleNamespace(cancel = calls.append),
                }
            )
        )
    )
    alice_event, bob_event = threading.Event(), threading.Event()
    with (
        run_as(ALICE, active_generations.ActiveGeneration, alice_event, run_id = "same"),
        run_as(BOB, active_generations.ActiveGeneration, bob_event, run_id = "same"),
    ):
        run_as(
            ALICE,
            chat_generation_runs.cancel_account_run,
            request,
            "same",
            supervisor_name = supervisor_name,
        )
    assert alice_event.is_set() and not bob_event.is_set()
    assert calls == []


@pytest.mark.parametrize(
    "cleanup",
    [
        chat_history._cancel_deleted_research_runs,
        chat_history._cancel_research_runs,
        chat_history._cancel_chat_generation_runs,
    ],
)
@pytest.mark.parametrize("supervisor_present", [True, False])
def test_thread_cleanup_scopes_run_ids(cleanup, supervisor_present):
    calls = []
    supervisor = SimpleNamespace(cancel = calls.append) if supervisor_present else None
    request = SimpleNamespace(
        app = SimpleNamespace(
            state = SimpleNamespace(
                research_supervisor = supervisor,
                chat_generation_supervisor = supervisor,
            )
        )
    )
    alice_event, bob_event = threading.Event(), threading.Event()
    with (
        run_as(ALICE, active_generations.ActiveGeneration, alice_event, run_id = "same"),
        run_as(BOB, active_generations.ActiveGeneration, bob_event, run_id = "same"),
    ):
        run_as(ALICE, cleanup, request, ["same"])
    assert alice_event.is_set() and not bob_event.is_set()
    assert calls == []


def test_cancel_route_scopes_same_id_in_two_accounts(client):
    for account in (ALICE, BOB):
        seed_chat(account)
        seed_run(account)
    alice_event, bob_event = threading.Event(), threading.Event()
    with (
        run_as(ALICE, active_generations.ActiveGeneration, alice_event, run_id = "run"),
        run_as(BOB, active_generations.ActiveGeneration, bob_event, run_id = "run"),
    ):
        assert client.post("/runs/run/cancel").status_code == 200
    assert alice_event.is_set() and not bob_event.is_set()
    assert run_as(BOB, chat_generation_runs_db.get_run, "run")["status"] == "queued"


def test_create_refuses_a_foreign_supervisor_slot_before_writing(client, monkeypatch):
    seed_chat(ALICE)
    monkeypatch.setattr(
        chat_generation_runs, "_sanitize_request", lambda payload: payload.requestPayload
    )
    body = {
        "runId": "same",
        "threadId": "private",
        "userMessageId": "message",
        "assistantMessageId": "reply",
        "requestPayload": {"model": "local"},
    }
    with run_as(BOB, active_generations.ActiveGeneration, threading.Event(), run_id = "same"):
        assert client.post("/runs", json = body).status_code == 404
    assert run_as(ALICE, chat_generation_runs_db.get_run, "same") is None


def test_event_wait_executor_retains_account_context(client, monkeypatch):
    for account in (OWNER, ALICE):
        seed_chat(account, text = account.username)
        seed_run(account)
        run_as(account, chat_generation_runs_db.request_cancel, "run")
    seen = []
    original = chat_generation_runs_db.wait_for_events

    def wait(*args):
        seen.append(current_account_id())
        return original(*args)

    monkeypatch.setattr(chat_generation_runs_db, "wait_for_events", wait)
    response = client.post("/runs/run/events")
    assert response.status_code == 200
    assert seen == [ALICE.account_id]
    events = [
        json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")
    ]
    expected = run_as(ALICE, chat_generation_runs_db.list_events, "run")
    assert [(event["seq"], event["createdAt"]) for event in events] == [
        (event["seq"], event["createdAt"]) for event in expected
    ]


def test_clear_history_cannot_reap_global_images_or_foreign_runs(client, monkeypatch):
    from core.inference import search_images

    calls = []
    monkeypatch.setattr(
        search_images, "snapshot_and_fence_registrations", lambda: calls.append("snapshot")
    )
    monkeypatch.setattr(search_images, "clear_cache", lambda *args: calls.append("clear"))
    for account in (ALICE, BOB):
        seed_chat(account)
        seed_run(account)
    bob_event = threading.Event()
    with run_as(
        BOB, active_generations.ActiveGeneration, bob_event, thread_id = "private", run_id = "run"
    ):
        for _ in range(2):
            response = client.request(
                "DELETE", "/chat", json = {"ids": ["private"], "operationId": "same-clear"}
            )
            assert response.status_code == 200, response.text
        assert not bob_event.is_set()
    assert calls == []
    assert run_as(BOB, studio_db.get_chat_thread, "private") is not None


# An explicit contract: moving a route to the wrong policy group must fail.
OWNER_PATHS = [
    (method, path)
    for path, methods in {
        "/hugging-face-cache": ("GET", "PUT"),
        "/llama-cpp-path": ("GET", "PUT"),
        "/upload-limit": ("PUT",),
        "/helper-precache": ("PUT",),
        "/download-transport": ("PUT",),
        "/xet-notice/reserve": ("POST",),
        "/model-memory": ("GET", "PUT"),
        "/vram-budget": ("GET", "PUT"),
        "/coding-agents": ("GET",),
        "/openai-auto-switch": ("GET", "PUT"),
        "/openai-auto-switch/overrides": ("GET", "PUT"),
        "/embedding-model": ("GET", "PUT", "DELETE"),
        "/embedding-model/resolve": ("GET",),
        "/embedding-model/unload": ("POST",),
        "/preview-links/rotate": ("POST",),
        "/remote-access": ("GET",),
        "/remote-access/start": ("POST",),
        "/remote-access/stop": ("POST",),
        "/remote-access/auto-start": ("PUT",),
        "/lan-access": ("GET",),
        "/lan-access/start": ("POST",),
        "/lan-access/stop": ("POST",),
        "/lan-access/auto-start": ("PUT",),
        "/lan-access/port": ("PUT",),
        "/preview-sharing": ("GET", "PUT"),
        "/keyless-api-access": ("GET", "PUT"),
        "/debug/logs/sources": ("GET",),
        "/debug/logs": ("GET",),
    }.items()
    for method in methods
]


@pytest.mark.parametrize("method,path", OWNER_PATHS)
def test_every_owner_setting_rejects_managed_accounts(client, method, path):
    response = client.request(method, "/settings" + path, json = {})
    assert response.status_code == 403, response.text


@pytest.mark.parametrize(
    "path,body",
    [
        ("/personalization", {"profile": {"displayName": "Alice"}}),
        ("/chat-preferences", {"show_model_disclaimer": True}),
        ("/current-date-prompt", {"enabled": False}),
    ],
)
def test_personal_settings_do_not_change_owner_rows(client, path, body):
    owner_record = {
        "version": 1,
        "profile": {"displayName": "Original owner"},
        "future": {"keep": True},
    }
    studio_db.upsert_app_settings({"personalization": owner_record})
    with studio_db.get_connection() as conn:
        before = tuple(
            conn.execute("SELECT * FROM app_settings WHERE key = 'personalization'").fetchone()
        )
    assert client.put("/settings" + path, json = body).status_code == 200
    with studio_db.get_connection() as conn:
        after = tuple(
            conn.execute("SELECT * FROM app_settings WHERE key = 'personalization'").fetchone()
        )
    assert after == before
    assert studio_db.get_app_setting("personalization") == owner_record


def test_shared_policy_reads_owner_storage_and_restores_context(client, monkeypatch):
    seen = []
    monkeypatch.setattr(
        settings, "get_upload_limit_mb", lambda: seen.append(current_account_id()) or 100
    )
    assert client.get("/settings/upload-limit").status_code == 200
    assert seen == [OWNER.account_id]
    assert (
        client.put("/settings/personalization", json = {"profile": {"nickname": "alice"}}).status_code
        == 200
    )
    assert (
        run_as(ALICE, studio_db.get_app_setting, "personalization")["profile"]["nickname"]
        == "alice"
    )
    assert studio_db.get_app_setting("personalization", None) is None


def test_owner_setting_keeps_200_and_single_account_policy_is_inert(client, monkeypatch):
    monkeypatch.setattr(
        settings,
        "_llama_cpp_path_response",
        lambda: settings.LlamaCppPathResponse(source = "default", editable = True, available = False),
    )
    response = client.get("/settings/llama-cpp-path", headers = {"x-test-account": "unsloth"})
    assert response.status_code == 200, response.text
    monkeypatch.setattr(policy, "installation_is_multi_user", lambda: False)
    assert client.get("/settings/llama-cpp-path").status_code == 200


def test_managed_last_model_key_survives_username_rename(client):
    payload = {"id": "alice-model", "kind": "gguf"}
    assert client.put("/settings/last-local-model", json = payload).status_code == 200
    renamed = AccountContext(ALICE.account_id, "new-name")
    result = run_as(renamed, settings.get_last_local_model, current_subject = renamed.username)
    assert result.id == "alice-model"
    digest = hashlib.sha256(b"unsloth").hexdigest()[:32]
    assert settings._last_local_model_key("unsloth") == f"last_local_model_load:{digest}"


def test_owner_chats_prompts_and_settings_survive_alice(client):
    seed_chat(OWNER, text = "owner chat")
    seed_chat(ALICE, text = "alice chat")
    studio_db.upsert_chat_settings_merge({"autoTitle": False})
    assert client.put("/chat/settings", json = {"autoTitle": True}).status_code == 200
    prompt = {"id": "same", "name": "Owner", "text": "owner prompt", "createdAt": 1, "updatedAt": 1}
    studio_db.upsert_prompt_entry(prompt)
    assert (
        client.put("/prompts/entries/same", json = {**prompt, "text": "alice prompt"}).status_code
        == 200
    )
    assert client.delete("/prompts/entries/same").status_code == 204
    assert client.get("/chat/export").json()["threads"][0]["title"] == "alice chat"
    assert studio_db.list_prompt_entries()[0]["text"] == "owner prompt"
    assert studio_db.list_chat_settings()["autoTitle"] is False
    assert studio_db.get_chat_thread("private")["title"] == "owner chat"
    assert studio_db_path().name == "studio.db" and "accounts" not in str(studio_db_path())


def test_keyless_cache_is_always_populated_from_the_owner():
    studio_db.upsert_app_settings(
        {"keyless_api_access_scope": "off", "keyless_api_access_tools": False}
    )
    run_as(
        ALICE,
        studio_db.upsert_app_settings,
        {"keyless_api_access_scope": "full", "keyless_api_access_tools": True},
    )
    assert run_as(ALICE, keyless_api_access.get_keyless_api_access_settings) == ("off", False)
    assert keyless_api_access.get_keyless_api_access_settings() == ("off", False)
    with pytest.raises(ValueError, match = "owner"):
        run_as(ALICE, keyless_api_access.set_keyless_api_access, "full", tools = True)
    assert keyless_api_access.get_keyless_api_access_settings() == ("off", False)


def test_profile_cache_cannot_follow_a_reused_username(monkeypatch):
    seed_chat(ALICE, text = "Alice model")
    seed_chat(BOB, text = "Bob model")
    for account in (ALICE, BOB):
        run_as(
            account,
            studio_db.upsert_chat_message,
            {
                "id": "reply",
                "threadId": "private",
                "role": "assistant",
                "createdAt": 3,
                "content": [{"type": "text", "text": "reply"}],
            },
        )
    renamed_bob = AccountContext(BOB.account_id, "alice")

    async def stats(account):
        return await arun_as(
            account,
            profile_stats.get_profile_stats(
                days = 30,
                tz_offset_minutes = 0,
                tz = "",
                current_subject = "alice",
            ),
        )

    alice = asyncio.run(stats(ALICE))
    bob = asyncio.run(stats(renamed_bob))
    assert "Alice model" in str(alice)
    assert "Alice model" not in str(bob)
    assert "Bob model" in str(bob)


def test_api_usage_statistics_read_only_the_account_database(client):
    receipt = api_usage_db.ApiUsageReceipt(
        id = "receipt",
        subject = "bob",
        endpoint = "/v1/chat/completions",
        model = "bob-model",
        status = "completed",
        prompt_tokens = 10,
        completion_tokens = 20,
        total_tokens = 30,
        created_at = 1,
        via_api_key = True,
    )
    run_as(BOB, api_usage_db.record_api_usage, receipt)
    response = client.get("/profile/stats")
    assert response.status_code == 200, response.text
    assert "bob-model" not in response.text


def test_single_account_cancel_still_calls_the_original_supervisor(monkeypatch):
    monkeypatch.setattr(policy, "installation_is_multi_user", lambda: False)
    calls = []
    request = SimpleNamespace(
        app = SimpleNamespace(
            state = SimpleNamespace(
                chat_generation_supervisor = SimpleNamespace(cancel = calls.append),
            )
        )
    )
    chat_generation_runs.cancel_account_run(
        request, "old-run", supervisor_name = "chat_generation_supervisor"
    )
    assert calls == ["old-run"]


def test_single_account_cancel_without_supervisor_keeps_existing_behavior(client, monkeypatch):
    monkeypatch.setattr(policy, "installation_is_multi_user", lambda: False)
    seed_chat(OWNER)
    seed_run(OWNER)
    calls = []
    monkeypatch.setattr(
        chat_generation_runs, "cancel_account_run", lambda *args, **kwargs: calls.append(args)
    )
    response = client.post("/runs/run/cancel", headers = {"x-test-account": "unsloth"})
    assert response.status_code == 200
    assert calls == []


def test_deep_research_foreign_run_is_not_visible_or_cancellable(client):
    seed_chat(BOB)
    run_as(
        BOB,
        research_runs_db.create_run,
        run_id = "research",
        owner_subject = "bob",
        thread_id = "private",
        user_message_id = "message",
        assistant_message_id = None,
        config = {},
    )
    assert client.get("/research/active?threadId=private").json() == {"runs": [], "hasRun": False}
    assert client.get("/research/research").status_code == 404
    assert client.post("/research/research/cancel").status_code == 404
    assert run_as(BOB, research_runs_db.get_run, "research")["status"] == "planning"
