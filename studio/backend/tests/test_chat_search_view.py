# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import json
import os
import sys

import pytest

_backend = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, _backend)

from routes import chat_history
from storage import studio_db

PASTED = "<pasted_text name=notes bytes=9>\nPASTEBODY\n</pasted_text>"
IMAGE = "data:image/png;base64," + "QUJD" * 25_000
TEXT_PARTS = [
    {"type": "reasoning", "text": "why"},
    {"type": "thinking", "thinking": "deep"},
    {
        "type": "tool-call",
        "toolName": "python",
        "args": {"code": "print(1)"},
        "result": {"stdout": "1"},
    },
    {"type": "source", "title": "Docs", "url": "https://example.com"},
    {"type": "text", "text": "hello"},
]


@pytest.fixture
def history(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    monkeypatch.setenv("UNSLOTH_STUDIO_PROJECTS_HOME", str(tmp_path / "Projects"))
    monkeypatch.setattr(studio_db, "_schema_ready", set())
    studio_db.upsert_chat_thread(
        {
            "id": "t1",
            "title": "Chat",
            "modelType": "base",
            "modelId": "m",
            "pairId": None,
            "archived": False,
            "createdAt": 1_700_000_000_000,
        }
    )
    messages = [
        {
            "id": "u1",
            "threadId": "t1",
            "parentId": None,
            "role": "user",
            "content": [{"type": "text", "text": "look"}, {"type": "image", "image": IMAGE}],
            "attachments": [
                {
                    "id": "a1",
                    "type": "document",
                    "name": "notes.txt",
                    "content": [{"type": "text", "text": PASTED}],
                },
                {
                    "id": "a2",
                    "type": "document",
                    "name": "paper.pdf",
                    "content": [{"type": "text", "text": "<attachment name=paper.pdf>\nDOCBODY"}],
                },
                {
                    "id": "a3",
                    "type": "image",
                    "name": "shot.png",
                    "content": [{"type": "image", "image": IMAGE}],
                },
            ],
            "createdAt": 1_700_000_000_001,
        },
        {
            "id": "a1",
            "threadId": "t1",
            "parentId": "u1",
            "role": "assistant",
            "content": TEXT_PARTS,
            "metadata": {"custom": {"timings": {"predicted_n": 3}}},
            "createdAt": 1_700_000_000_002,
        },
    ]
    studio_db.sync_chat_messages("t1", messages)
    return messages


def _batch(view = None):
    payload = {"threadIds": ["t1", "missing"]}
    if view is not None:
        payload["view"] = view
    return chat_history.batch_thread_messages(
        chat_history.ChatMessagesBatchRequest(**payload), current_subject = "test-user"
    ).model_dump(mode = "json")["messagesByThreadId"]


def test_batch_without_a_view_still_returns_full_messages(history):
    full = _batch()
    assert full == _batch("full")
    assert full["missing"] == []
    user, assistant = full["t1"]
    assert user["content"] == history[0]["content"]
    assert [a["id"] for a in user["attachments"]] == ["a1", "a2", "a3"]
    assert assistant["metadata"] == history[1]["metadata"]
    assert assistant["content"] == TEXT_PARTS


def test_search_view_keeps_what_the_index_reads_and_drops_the_rest(history):
    full = _batch()
    view = _batch("search")
    assert view["missing"] == []
    user, assistant = view["t1"]
    for kept, source in zip(view["t1"], full["t1"]):
        for key in ("id", "threadId", "parentId", "role", "createdAt"):
            assert kept[key] == source[key]
        assert kept["metadata"] is None
    assert user["content"] == [{"type": "text", "text": "look"}]
    # Only the pasted-text body is searchable; the document body and image never are.
    assert user["attachments"] == [{"content": [{"type": "text", "text": PASTED}]}]
    assert assistant["content"] == TEXT_PARTS

    full_bytes = len(json.dumps(full))
    view_bytes = len(json.dumps(view))
    assert full_bytes > 150_000
    assert view_bytes < 5_000


@pytest.mark.parametrize(
    "attachments",
    [
        [None],
        [3],
        ["s"],
        [{"content": "string"}],
        [{"content": {"type": "text"}}],
        [{"content": [None]}],
        [{"content": [{"type": "text", "text": 5}]}],
        {"not": "a list"},
        "a string",
    ],
)
def test_search_view_passes_malformed_attachments_through(attachments):
    """The index reads these as stored (some make it throw), so the view must not normalise them."""
    message = {
        "id": "m",
        "threadId": "t",
        "role": "user",
        "content": [],
        "attachments": attachments,
        "createdAt": 1,
    }
    assert chat_history._search_view_message(message)["attachments"] == attachments


def test_search_view_drops_parts_the_index_skips():
    content = [
        None,
        "str",
        ["text"],
        {"type": ["text"], "text": "x"},
        {"type": "Text", "text": "x"},
        {"type": "file", "data": "QUJD"},
        {"type": "audio", "audio": {"data": "QUJD"}},
        {"type": "text", "text": "kept"},
    ]
    message = {"id": "m", "threadId": "t", "role": "user", "content": content, "createdAt": 1}
    assert chat_history._search_view_message(message)["content"] == [
        {"type": "text", "text": "kept"}
    ]
    # A non-list content is read as stored.
    for value in ("plain", None, {"type": "text", "text": "x"}):
        message["content"] = value
        assert chat_history._search_view_message(message)["content"] == value


def test_search_view_leaves_absent_content_to_the_model_default():
    """Without content the full view serialises [], and the index collapses pasted text only then."""
    message = {
        "id": "m",
        "threadId": "t",
        "role": "user",
        "attachments": [{"content": [{"type": "text", "text": PASTED}]}],
        "createdAt": 1,
    }
    view = chat_history.ChatMessage(**chat_history._search_view_message(message))
    assert view.content == chat_history.ChatMessage(**message).content == []
