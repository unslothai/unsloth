# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Focused storage, service, and route contract tests for chat memory."""

import asyncio
import sqlite3
import importlib.util
import sys
from pathlib import Path

import pytest
from fastapi import HTTPException

from core.inference import memory
from storage import studio_db

_ROUTE_SPEC = importlib.util.spec_from_file_location(
    "chat_memory_route", Path(__file__).resolve().parents[1] / "routes" / "chat_memory.py"
)
assert _ROUTE_SPEC is not None and _ROUTE_SPEC.loader is not None
chat_memory = importlib.util.module_from_spec(_ROUTE_SPEC)
sys.modules[_ROUTE_SPEC.name] = chat_memory
_ROUTE_SPEC.loader.exec_module(chat_memory)


def _setup_source(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    monkeypatch.setenv("UNSLOTH_STUDIO_PROJECTS_HOME", str(tmp_path / "Projects"))
    monkeypatch.setattr(studio_db, "_schema_ready", False)
    studio_db.upsert_chat_project(
        {
            "id": "project",
            "name": "Project",
            "instructions": "",
            "archived": False,
            "createdAt": 1,
            "updatedAt": 1,
        }
    )
    studio_db.upsert_chat_thread(
        {
            "id": "thread",
            "title": "Thread",
            "modelType": "base",
            "modelId": "m",
            "pairId": None,
            "projectId": "project",
            "archived": False,
            "createdAt": 1,
        }
    )
    studio_db.upsert_chat_message(
        {
            "id": "message",
            "threadId": "thread",
            "parentId": None,
            "role": "user",
            "content": [{"type": "text", "text": "I prefer dark mode"}],
            "createdAt": 2,
        }
    )


def test_storage_is_installation_global_and_project_clear_is_unambiguous(tmp_path, monkeypatch):
    _setup_source(tmp_path, monkeypatch)
    saved = memory.create_memory(content = "Use tabs", scope = "project", project_id = "project")
    assert studio_db.get_chat_memory(saved["id"]) == saved
    assert memory.clear_scope("project", "project") == 1
    assert studio_db.list_chat_memories("project", "project") == []
    with pytest.raises(memory.MemoryValidationError):
        memory.clear_scope("project", None)


def test_capture_validates_targets_and_is_idempotent(tmp_path, monkeypatch):
    _setup_source(tmp_path, monkeypatch)
    old = memory.create_memory(content = "I prefer light mode", scope = "global")
    output = (
        '{"operations":[{"action":"replace","scope":"global","memory_id":"%s","content":"I prefer dark mode"}]}'
        % old["id"]
    )
    first = memory.apply_capture(thread_id = "thread", source_message_id = "message", raw_output = output)
    second = memory.apply_capture(
        thread_id = "thread", source_message_id = "message", raw_output = output
    )
    assert [item["content"] for item in first] == ["I prefer dark mode"]
    assert second == []
    assert studio_db.get_chat_memory(old["id"])["content"] == "I prefer dark mode"


def test_model_forget_requires_user_correction_evidence(tmp_path, monkeypatch):
    _setup_source(tmp_path, monkeypatch)
    saved = memory.create_memory(content = "I prefer dark mode", scope = "global")
    output = '{"operations":[{"action":"forget","scope":"global","memory_id":"%s"}]}' % saved["id"]

    assert (
        memory.apply_capture(thread_id = "thread", source_message_id = "message", raw_output = output)
        == []
    )
    assert studio_db.get_chat_memory(saved["id"]) is not None

    monkeypatch.setattr(
        memory,
        "get_chat_message",
        lambda *_: {
            "threadId": "thread",
            "role": "user",
            "content": [{"type": "text", "text": "I no longer prefer dark mode"}],
        },
    )
    forgotten = memory.apply_capture(
        thread_id = "thread", source_message_id = "message", raw_output = output
    )
    assert forgotten == [saved]
    assert studio_db.get_chat_memory(saved["id"]) is None


@pytest.mark.parametrize(
    "content",
    (
        "My email is me@example.com",
        "My phone number is +1 (415) 555-0199",
        "Text me at 415-555-0199",
    ),
)
def test_automatic_capture_rejects_contact_pii(content):
    with pytest.raises(memory.MemoryValidationError):
        memory.create_memory(content = content, scope = "global", source_type = "model")


@pytest.mark.parametrize(
    "content",
    (
        "I live at 123 Main Street",
        "My home is 45 Park Ave",
    ),
)
def test_automatic_capture_rejects_street_addresses(content):
    with pytest.raises(memory.MemoryValidationError):
        memory.create_memory(content = content, scope = "global", source_type = "model")


@pytest.mark.parametrize(
    "content",
    (
        "My password is hunter2",
        "My API key is abc123secret",
        "My private key is hidden-value",
    ),
)
def test_automatic_capture_rejects_natural_language_secrets(content):
    with pytest.raises(memory.MemoryValidationError):
        memory.create_memory(content = content, scope = "global", source_type = "model")


@pytest.mark.parametrize(
    "content",
    (
        "My SSN is 123-45-6789",
        "Use account 4111 1111 1111 1111",
    ),
)
def test_automatic_capture_rejects_structured_identifiers(content):
    with pytest.raises(memory.MemoryValidationError):
        memory.create_memory(content = content, scope = "global", source_type = "model")


def test_direct_statement_skips_transient_and_profile_claims(tmp_path, monkeypatch):
    _setup_source(tmp_path, monkeypatch)

    for text in ("I am getting an error", "My name is Wasim"):
        monkeypatch.setattr(
            memory,
            "get_chat_message",
            lambda *_, text = text: {
                "threadId": "thread",
                "role": "user",
                "content": [{"type": "text", "text": text}],
            },
        )
        assert memory.direct_statement("thread", "message") == []

    assert studio_db.list_chat_memories() == []


def test_direct_statement_skips_question_shaped_claims(tmp_path, monkeypatch):
    _setup_source(tmp_path, monkeypatch)
    monkeypatch.setattr(
        memory,
        "get_chat_message",
        lambda *_: {
            "threadId": "thread",
            "role": "user",
            "content": [{"type": "text", "text": "I prefer dark mode?"}],
        },
    )

    assert memory.direct_statement("thread", "message") == []
    assert studio_db.list_chat_memories() == []


def test_explicit_commands_accept_optional_please(tmp_path, monkeypatch):
    _setup_source(tmp_path, monkeypatch)
    saved = memory.create_memory(content = "Use dark mode", scope = "global")
    monkeypatch.setattr(
        memory,
        "get_chat_message",
        lambda *_: {
            "role": "user",
            "content": [{"type": "text", "text": "Please forget that Use dark mode"}],
        },
    )
    assert memory.explicit_command("thread", "message") == [saved]
    monkeypatch.setattr(
        memory,
        "get_chat_message",
        lambda *_: {
            "role": "user",
            "content": [{"type": "text", "text": "Please remember that I prefer dark mode"}],
        },
    )
    captured = memory.explicit_command("thread", "message")
    assert captured and captured[0]["sourceType"] == "explicit"


@pytest.mark.parametrize(
    "command",
    (
        "forget my phone number",
        "Can you forget my phone number?",
        "remove my phone number from memory",
        "delete the memory about my phone number",
    ),
)
def test_explicit_forget_matches_partial_targets_and_aliases(tmp_path, monkeypatch, command):
    _setup_source(tmp_path, monkeypatch)
    saved = memory.create_memory(content = "My phone number is 415-555-0199", scope = "global")
    monkeypatch.setattr(
        memory,
        "get_chat_message",
        lambda *_: {
            "threadId": "thread",
            "role": "user",
            "content": [{"type": "text", "text": command}],
        },
    )

    assert memory.explicit_command("thread", "message") == [saved]
    assert studio_db.get_chat_memory(saved["id"]) is None


def test_generic_delete_prompt_does_not_remove_memory(tmp_path, monkeypatch):
    _setup_source(tmp_path, monkeypatch)
    saved = memory.create_memory(content = "Use the dark mode CSS class", scope = "global")
    monkeypatch.setattr(
        memory,
        "get_chat_message",
        lambda *_: {
            "threadId": "thread",
            "role": "user",
            "content": [{"type": "text", "text": "delete the dark mode CSS class"}],
        },
    )

    assert memory.explicit_command("thread", "message") == []
    assert studio_db.get_chat_memory(saved["id"]) == saved


@pytest.mark.parametrize(
    "command",
    (
        "remove my phone number from memory",
        "delete the memory about my phone number",
        "Can you forget my phone number?",
    ),
)
def test_memory_removal_aliases_skip_recall(tmp_path, monkeypatch, command):
    _setup_source(tmp_path, monkeypatch)
    memory.create_memory(content = "My phone number is 415-555-0199", scope = "global")
    monkeypatch.setattr(
        memory,
        "get_chat_message",
        lambda *_: {
            "threadId": "thread",
            "role": "user",
            "content": [{"type": "text", "text": command}],
        },
    )

    assert memory.recall_context("thread", "message") is None


def test_forget_command_skips_recall(tmp_path, monkeypatch):
    _setup_source(tmp_path, monkeypatch)
    memory.create_memory(content = "My phone number is 415-555-0199", scope = "global")
    monkeypatch.setattr(
        memory,
        "get_chat_message",
        lambda *_: {
            "threadId": "thread",
            "role": "user",
            "content": [
                {"type": "text", "text": "Please forget that My phone number is 415-555-0199"}
            ],
        },
    )

    assert memory.recall_context("thread", "message") is None


def test_model_forget_rejects_ambiguous_evidence(tmp_path, monkeypatch):
    _setup_source(tmp_path, monkeypatch)
    saved = memory.create_memory(content = "I prefer dark mode", scope = "global")
    output = '{"operations":[{"action":"forget","scope":"global","memory_id":"%s"}]}' % saved["id"]
    monkeypatch.setattr(
        memory,
        "get_chat_message",
        lambda *_: {
            "threadId": "thread",
            "role": "user",
            "content": [{"type": "text", "text": "Actually, how do I configure dark mode?"}],
        },
    )

    assert (
        memory.apply_capture(thread_id = "thread", source_message_id = "message", raw_output = output)
        == []
    )
    assert studio_db.get_chat_memory(saved["id"]) == saved


def test_recall_requires_relevant_content(tmp_path, monkeypatch):
    _setup_source(tmp_path, monkeypatch)
    memory.create_memory(content = "My favorite editor is Helix", scope = "global")
    monkeypatch.setattr(
        memory,
        "get_chat_message",
        lambda *_: {
            "threadId": "thread",
            "role": "user",
            "content": [{"type": "text", "text": "Help me debug this error"}],
        },
    )

    assert memory.recall_context("thread", "message") is None


@pytest.mark.parametrize(
    ("saved_content", "query"),
    (
        ("I am HIV positive", "I am debugging a TypeError"),
        ("You can reach me in Berlin", "Please help me debug this"),
        ("I'm allergic to latex", "I'm seeing a TypeError"),
    ),
)
def test_recall_ignores_first_person_filler_overlap(tmp_path, monkeypatch, saved_content, query):
    _setup_source(tmp_path, monkeypatch)
    memory.create_memory(content = saved_content, scope = "global")
    monkeypatch.setattr(
        memory,
        "get_chat_message",
        lambda *_: {
            "threadId": "thread",
            "role": "user",
            "content": [{"type": "text", "text": query}],
        },
    )

    assert memory.recall_context("thread", "message") is None


def test_memory_settings_are_global(tmp_path, monkeypatch):
    _setup_source(tmp_path, monkeypatch)
    settings = studio_db.upsert_chat_settings_merge(
        {"referenceMemories": False, "autoSaveMemories": False}
    )
    assert settings["referenceMemories"] is False
    assert studio_db.list_chat_settings()["autoSaveMemories"] is False

    assert memory.get_memory_settings() == (False, False)


def test_schema_enforces_normalized_scope_uniqueness(tmp_path, monkeypatch):
    _setup_source(tmp_path, monkeypatch)
    saved = memory.create_memory(content = "Use tabs", scope = "global")
    conn = studio_db.get_connection()
    try:
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                """
                INSERT INTO chat_memories (
                    id, scope, project_id, content, normalized_content, source_type,
                    source_thread_id, source_message_id, created_at, updated_at
                ) VALUES (?, 'global', NULL, ?, ?, 'manual', NULL, NULL, 3, 3)
                """,
                ("duplicate", "Use tabs", saved["content"].casefold()),
            )
    finally:
        conn.rollback()
        conn.close()


def test_apply_capture_respects_global_auto_save_setting(tmp_path, monkeypatch):
    _setup_source(tmp_path, monkeypatch)
    studio_db.upsert_chat_settings_merge({"autoSaveMemories": False})
    monkeypatch.setattr(
        memory,
        "apply_capture",
        lambda **_: pytest.fail("capture must not run while automatic saving is disabled"),
    )
    payload = chat_memory.CapturePayload(
        threadId = "thread",
        sourceMessageId = "message",
        rawOutput = '{"operations":[]}',
    )
    assert asyncio.run(chat_memory.apply_capture(payload)) == {"memories": []}


def test_route_rejects_ambiguous_project_filters(tmp_path, monkeypatch):
    _setup_source(tmp_path, monkeypatch)
    with pytest.raises(HTTPException) as exc:
        asyncio.run(chat_memory.list_memories(project_id = "project"))
    assert exc.value.status_code == 400
    with pytest.raises(HTTPException) as exc:
        asyncio.run(chat_memory.clear_memories(scope = "project", project_id = None))
    assert exc.value.status_code == 400
