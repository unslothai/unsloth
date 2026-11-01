# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import json
import sqlite3
from contextlib import contextmanager

import pytest

from core.external_import import claude, run_import, thread_id_for
from storage import studio_db


def _turn(
    role,
    uuid,
    text,
    parent = None,
    second = 0,
):
    return {
        "type": role,
        "uuid": uuid,
        "parentUuid": parent,
        "timestamp": f"2026-11-01T10:00:{second:02d}Z",
        "message": {
            "role": role,
            "content": text if role == "user" else [{"type": "text", "text": text}],
        },
    }


def _write(
    path,
    records,
    mode = "w",
):
    path.parent.mkdir(parents = True, exist_ok = True)
    with path.open(mode, encoding = "utf-8") as handle:
        handle.write("".join(json.dumps(record) + "\n" for record in records))


def _execute(sql):
    conn = studio_db.get_connection()
    try:
        conn.execute(sql)
        conn.commit()
    finally:
        conn.close()


@contextmanager
def _reject_checkpoint():
    _execute(
        "CREATE TRIGGER reject_external_import_checkpoint "
        "BEFORE INSERT ON external_import_sessions "
        "BEGIN SELECT RAISE(ABORT, 'checkpoint write failed'); END"
    )
    try:
        yield
    finally:
        _execute("DROP TRIGGER reject_external_import_checkpoint")


@pytest.fixture
def source_path(tmp_path, monkeypatch):
    home = tmp_path / "claude"
    monkeypatch.setenv(claude.SOURCE.home_env, str(home))
    path = home / "projects" / "-Users-me-app" / "s1.jsonl"
    _write(path, [_turn("user", "u1", "first")])
    return path


def _messages():
    return studio_db.list_chat_messages(thread_id_for(claude.SOURCE, "s1"))


def _mark():
    return studio_db.get_external_import_mark(claude.SOURCE.key, "s1")


def _texts(messages):
    return [message["content"][0]["text"] for message in messages]


def test_first_import_rolls_back_messages_when_checkpoint_fails(source_path):
    with (
        _reject_checkpoint(),
        pytest.raises(sqlite3.IntegrityError, match = "checkpoint write failed"),
    ):
        run_import(claude.SOURCE)

    assert _messages() == []
    assert _mark() is None
    _write(source_path, [_turn("user", "u2", "second", parent = "u1", second = 1)], mode = "a")

    assert run_import(claude.SOURCE).messages == 2
    rows = _messages()
    assert _texts(rows) == ["first", "second"]
    assert rows[1]["parentId"] == rows[0]["id"]
    assert _mark() == (2, "")
    assert run_import(claude.SOURCE).messages == 0
    assert _messages() == rows


def test_incremental_import_rolls_back_and_preserves_studio_changes(source_path):
    _write(source_path, [_turn("assistant", "a1", "reply", parent = "u1", second = 1)], mode = "a")
    assert run_import(claude.SOURCE).messages == 2
    first, reply = _messages()
    studio_db.sync_chat_messages(
        thread_id_for(claude.SOURCE, "s1"),
        [{**first, "content": [{"type": "text", "text": "edited in Studio"}]}],
        prune_missing = True,
    )
    before = _messages()
    assert _texts(before) == ["edited in Studio"]
    assert _mark() == (2, "")
    _write(source_path, [_turn("user", "u2", "second", parent = "a1", second = 2)], mode = "a")

    with (
        _reject_checkpoint(),
        pytest.raises(sqlite3.IntegrityError, match = "checkpoint write failed"),
    ):
        run_import(claude.SOURCE)

    assert _messages() == before
    assert _mark() == (2, "")
    _write(source_path, [_turn("user", "u3", "third", parent = "u2", second = 3)], mode = "a")

    assert run_import(claude.SOURCE).messages == 2
    rows = _messages()
    assert _texts(rows) == ["edited in Studio", "second", "third"]
    assert reply["id"] not in {message["id"] for message in rows}
    assert rows[1]["parentId"] == rows[0]["id"]
    assert rows[2]["parentId"] == rows[1]["id"]
    assert _mark() == (4, "")
    assert run_import(claude.SOURCE).messages == 0
    assert _messages() == rows


def test_checkpoint_only_failure_keeps_existing_import_intact(source_path):
    assert run_import(claude.SOURCE).messages == 1
    before = _messages()
    assert _texts(before) == ["first"]
    assert _mark() == (1, "")

    with (
        _reject_checkpoint(),
        pytest.raises(sqlite3.IntegrityError, match = "checkpoint write failed"),
    ):
        run_import(claude.SOURCE)

    assert _messages() == before
    assert _mark() == (1, "")
    assert run_import(claude.SOURCE).messages == 0
    assert _messages() == before
    assert _mark() == (1, "")
