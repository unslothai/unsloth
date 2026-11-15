# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import json
import sys
from pathlib import Path

import pytest

import storage.studio_db as studio_db
from core.external_import import claude, run_import, thread_id_for


def _write(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents = True, exist_ok = True)
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding = "utf-8")


def _user(
    uuid,
    content,
    parent = None,
):
    return {
        "type": "user",
        "uuid": uuid,
        "parentUuid": parent,
        "timestamp": "2026-08-01T10:00:00Z",
        "message": {"role": "user", "content": content},
    }


def _assistant(uuid, content, parent):
    return {
        "type": "assistant",
        "uuid": uuid,
        "parentUuid": parent,
        "timestamp": "2026-08-01T10:00:01Z",
        "message": {"id": f"msg-{uuid}", "role": "assistant", "content": content},
    }


def _count_importer_reads(monkeypatch, reads):
    """Counts reads made by the importer itself; studio_db calls the same function internally."""
    real = studio_db.list_chat_messages

    def counted(thread_id):
        if sys._getframe(1).f_globals.get("__name__") == "core.external_import":
            reads.append(thread_id)
        return real(thread_id)

    monkeypatch.setattr(studio_db, "list_chat_messages", counted)
    return real


@pytest.fixture
def claude_home(tmp_path, monkeypatch):
    home = tmp_path / "claude"
    monkeypatch.setenv(claude.SOURCE.home_env, str(home))
    return home


def _open_call_session():
    return [
        _user("u1", "Read the file"),
        _assistant(
            "a1",
            [{"type": "tool_use", "id": "t1", "name": "Read", "input": {"path": "x"}}],
            "u1",
        ),
    ]


def _closed_call_tail():
    return [
        _user("r1", [{"type": "tool_result", "tool_use_id": "t1", "content": "body"}], "a1"),
        _assistant("a2", [{"type": "text", "text": "Done."}], "r1"),
    ]


def test_reimport_reads_each_existing_thread_once(claude_home, monkeypatch):
    sessions = ["s1", "s2", "s3"]
    for sid in sessions:
        _write(claude_home / "projects" / "-Users-me-app" / f"{sid}.jsonl", _open_call_session())
    first = run_import(claude.SOURCE)
    assert (first.new_chats, first.messages) == (3, 6)

    for sid in sessions:
        _write(
            claude_home / "projects" / "-Users-me-app" / f"{sid}.jsonl",
            _open_call_session() + _closed_call_tail(),
        )
    reads = []
    real = _count_importer_reads(monkeypatch, reads)
    second = run_import(claude.SOURCE)
    monkeypatch.setattr(studio_db, "list_chat_messages", real)

    assert sorted(reads) == sorted(thread_id_for(claude.SOURCE, sid) for sid in sessions)
    # One patched tool call plus the new assistant turn per session.
    assert (second.new_chats, second.messages) == (0, 6)
    for sid in sessions:
        rows = real(thread_id_for(claude.SOURCE, sid))
        calls = [p for m in rows for p in m["content"] if p.get("type") == "tool-call"]
        assert [c.get("result") for c in calls] == ["body"]
        texts = [p.get("text") for m in rows for p in m["content"] if p.get("type") == "text"]
        assert texts.count("Done.") == 1


def test_first_import_does_not_read_message_rows(claude_home, monkeypatch):
    _write(claude_home / "projects" / "-Users-me-app" / "s1.jsonl", _open_call_session())
    reads = []
    real = _count_importer_reads(monkeypatch, reads)
    summary = run_import(claude.SOURCE)
    assert (summary.new_chats, summary.messages) == (1, 2)
    assert reads == []
