# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from .factory_base import Factory, seeder

CHAT_SENTINEL = "chat-domain resource: café / 日本語"
ATTACHMENT_TEXT = "chat-domain attachment body: café / 日本語"

RUN_THREAD_ID = "chat-run-thread"
RUN_USER_MESSAGE_ID = "chat-run-user-message"
RUN_ASSISTANT_MESSAGE_ID = "chat-run-assistant-message"
RUN_ID = "chat-run-queued"
DONE_RUN_ID = "chat-run-completed"

ATTACHMENT_THREAD_ID = "chat-attachment-thread"
ATTACHMENT_MESSAGE_ID = "chat-attachment-message"
ATTACHMENT_ID = "chat-attachment-id"

PROJECT_ID = "chat-project"

FORK_THREAD_ID = "chat-fork-source-thread"
FORK_MESSAGE_ID = "chat-fork-source-message"
FORK_NEW_THREAD_ID = "chat-fork-new-thread"
FORKED_CHILD_THREAD_ID = "chat-fork-existing-child"

RUN_REQUEST_PAYLOAD = {
    "model": "local/model",
    "messages": [{"role": "user", "content": CHAT_SENTINEL}],
    "stream": True,
    "thread_id": RUN_THREAD_ID,
    "generation_run_id": RUN_ID,
}


def _thread(thread_id: str, **extra) -> dict:
    return {
        "id": thread_id,
        "title": CHAT_SENTINEL,
        "modelType": "base",
        "modelId": "local/model",
        "createdAt": 1000,
        "updatedAt": 1000,
        **extra,
    }


def _user_message(message_id: str, thread_id: str, **extra) -> dict:
    return {
        "id": message_id,
        "threadId": thread_id,
        "role": "user",
        "content": [{"type": "text", "text": CHAT_SENTINEL}],
        "createdAt": 1000,
        **extra,
    }


def _seed_run(account, run_id: str):
    from storage import chat_generation_runs_db, studio_db
    from utils.account_context import run_as

    run_as(account, studio_db.upsert_chat_thread, _thread(RUN_THREAD_ID))
    message = _user_message(RUN_USER_MESSAGE_ID, RUN_THREAD_ID)
    run_as(account, studio_db.upsert_chat_message, message)
    run, _ = run_as(
        account,
        chat_generation_runs_db.create_run,
        run_id = run_id,
        owner_subject = account.username,
        thread_id = RUN_THREAD_ID,
        user_message_id = RUN_USER_MESSAGE_ID,
        assistant_message_id = RUN_ASSISTANT_MESSAGE_ID,
        request_payload = {**RUN_REQUEST_PAYLOAD, "generation_run_id": run_id},
    )
    return run


@seeder("chat-generation-run")
def seed_chat_generation_run(account) -> dict[str, str]:
    _seed_run(account, RUN_ID)
    return {"run_id": RUN_ID}


@seeder("chat-generation-run-events")
def seed_chat_generation_run_events(account) -> dict[str, str]:
    from storage import chat_generation_runs_db
    from utils.account_context import run_as

    _seed_run(account, DONE_RUN_ID)
    worker_token = run_as(account, chat_generation_runs_db.get_worker_token, DONE_RUN_ID)
    run_as(
        account,
        chat_generation_runs_db.finish_run,
        DONE_RUN_ID,
        worker_token = worker_token,
        status = "completed",
        finish_reason = "stop",
        pending_events = [("chunk", {"delta": CHAT_SENTINEL})],
    )
    return {"run_id": DONE_RUN_ID}


@seeder("chat-attachment")
def seed_chat_attachment(account) -> dict[str, str]:
    from storage import studio_db
    from utils.account_context import run_as

    run_as(account, studio_db.upsert_chat_thread, _thread(ATTACHMENT_THREAD_ID))
    run_as(
        account,
        studio_db.upsert_chat_message,
        _user_message(
            ATTACHMENT_MESSAGE_ID,
            ATTACHMENT_THREAD_ID,
            attachments = [
                {
                    "id": ATTACHMENT_ID,
                    "name": "notes.txt",
                    "type": "file",
                    "contentType": "text/plain",
                    "content": [{"type": "text", "text": ATTACHMENT_TEXT}],
                }
            ],
        ),
    )
    return {"message_id": ATTACHMENT_MESSAGE_ID, "attachment_id": ATTACHMENT_ID}


@seeder("chat-project")
def seed_chat_project_for_delete(account) -> dict[str, str]:
    from storage import studio_db
    from utils.account_context import run_as

    run_as(
        account,
        studio_db.upsert_chat_project,
        {"id": PROJECT_ID, "name": CHAT_SENTINEL, "createdAt": 1000, "updatedAt": 1000},
    )
    return {"project_id": PROJECT_ID}


@seeder("chat-fork-source")
def seed_chat_fork_source(account) -> dict[str, str]:
    from storage import studio_db
    from utils.account_context import run_as

    run_as(account, studio_db.upsert_chat_thread, _thread(FORK_THREAD_ID))
    run_as(account, studio_db.upsert_chat_message, _user_message(FORK_MESSAGE_ID, FORK_THREAD_ID))
    return {"thread_id": FORK_THREAD_ID, "message_id": FORK_MESSAGE_ID}


@seeder("chat-fork-tree")
def seed_chat_fork_tree(account) -> dict[str, str]:
    from storage import studio_db
    from utils.account_context import run_as

    seed_chat_fork_source(account)
    run_as(
        account,
        studio_db.fork_chat_thread,
        source_thread_id = FORK_THREAD_ID,
        branch_message_id = FORK_MESSAGE_ID,
        new_thread_id = FORKED_CHILD_THREAD_ID,
        new_title = f"fork of {CHAT_SENTINEL}",
        created_at = 1001,
        id_factory = lambda: FORKED_CHILD_THREAD_ID + "-message",
    )
    return {"thread_id": FORK_THREAD_ID, "message_id": FORK_MESSAGE_ID}


# Both count routes answer 200 for any thread id, so the foreign-account expectation is 200.
_FORK_COUNT_REASON = (
    "Fork counts aggregate per account: another account counts its own empty database, not alice's."
)

FACTORIES = {
    "routes.chat_generation_runs:GET:/{run_id}": Factory("chat-generation-run", fragment = RUN_ID),
    "routes.chat_generation_runs:POST:/{run_id}/cancel": Factory(
        "chat-generation-run", fragment = '"status":"cancelled"'
    ),
    "routes.chat_generation_runs:POST:/{run_id}/events": Factory(
        "chat-generation-run-events", fragment = "event: run.completed"
    ),
    "routes.chat_history:GET:/attachments/{message_id}/{attachment_id}/file": Factory(
        "chat-attachment", fragment = ATTACHMENT_TEXT
    ),
    "routes.chat_history:DELETE:/attachments/{message_id}/{attachment_id}": Factory(
        "chat-attachment", fragment = '"ok":true'
    ),
    "routes.chat_history:DELETE:/projects/{project_id}": Factory(
        "chat-project", fragment = CHAT_SENTINEL
    ),
    "routes.chat_history:POST:/threads/{thread_id}/fork": Factory(
        "chat-fork-source",
        {"messageId": FORK_MESSAGE_ID, "newThreadId": FORK_NEW_THREAD_ID, "createdAt": 2000},
        fragment = CHAT_SENTINEL,
    ),
    "routes.chat_history:GET:/threads/{thread_id}/forks": Factory(
        "chat-fork-tree",
        fragment = f'"{FORK_MESSAGE_ID}":1',
        absent = f'"{FORK_MESSAGE_ID}"',
        owner = (200,),
        wrong = (200,),
        reason = _FORK_COUNT_REASON,
    ),
    "routes.chat_history:GET:/threads/{thread_id}/messages/{message_id}/forks": Factory(
        "chat-fork-tree",
        fragment = '"count":1',
        absent = '"count":1',
        owner = (200,),
        wrong = (200,),
        reason = _FORK_COUNT_REASON,
    ),
}

SKIPPED: dict = {}
