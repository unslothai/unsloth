# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A re-import with nothing new writes nothing, so open tabs see no revision move."""

import itertools
import json
from pathlib import Path

import pytest

import storage.studio_db as studio_db
from core.external_import import claude, codex, run_import, thread_id_for


def _write(path: Path, records: list[dict]) -> Path:
    path.parent.mkdir(parents = True, exist_ok = True)
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding = "utf-8")
    return path


def _turn(
    uuid,
    role,
    text,
    parent = None,
):
    return {
        "type": role,
        "uuid": uuid,
        "parentUuid": parent,
        "timestamp": "2026-08-01T10:00:00Z",
        "message": {"role": role, "content": [{"type": "text", "text": text}]},
    }


@pytest.fixture
def studio(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "studio"))
    monkeypatch.setenv("UNSLOTH_STUDIO_PROJECTS_HOME", str(tmp_path / "Projects"))
    monkeypatch.setattr(studio_db, "_schema_ready", set())
    home = tmp_path / "claude"
    monkeypatch.setenv(claude.SOURCE.home_env, str(home))
    return home


def _session(
    home,
    sid = "s1",
    extra = (),
):
    records = [_turn("u1", "user", "Fix the header"), _turn("a1", "assistant", "Done.", "u1")]
    return _write(home / "projects" / "-Users-me-app" / f"{sid}.jsonl", [*records, *extra])


def _revisions(thread_ids):
    return studio_db.get_chat_history_revisions(thread_ids)


def test_unchanged_reimport_moves_no_revision(studio):
    _session(studio, "s1")
    _session(studio, "s2")
    assert run_import(claude.SOURCE).new_chats == 2
    tids = [thread_id_for(claude.SOURCE, sid) for sid in ("s1", "s2")]
    before = _revisions(tids)

    again = run_import(claude.SOURCE)

    assert (again.chats, again.new_chats, again.messages) == (2, 0, 0)
    assert _revisions(tids) == before


def test_unchanged_reimport_skips_the_thread_and_ledger_writes(studio, monkeypatch):
    _session(studio)
    run_import(claude.SOURCE)
    for name in ("upsert_chat_thread", "record_external_import_mark", "sync_chat_messages"):
        monkeypatch.setattr(
            studio_db, name, lambda *a, _name = name, **k: pytest.fail(f"{_name} on a no-op")
        )
    assert run_import(claude.SOURCE).chats == 1


def test_a_grown_session_still_lands_and_moves_its_revision(studio):
    _session(studio)
    run_import(claude.SOURCE)
    tid = thread_id_for(claude.SOURCE, "s1")
    before = _revisions([tid])
    _session(studio, extra = [_turn("u2", "user", "And the footer", "a1")])

    assert run_import(claude.SOURCE).messages == 1

    after = _revisions([tid])
    assert after[0] > before[0] and after[1][tid] > before[1][tid]
    texts = [p.get("text") for m in studio_db.list_chat_messages(tid) for p in m["content"]]
    assert "And the footer" in texts


def test_a_row_the_upsert_would_change_is_still_written(studio):
    # A NULL updated_at (pre-migration row) reads back as created_at but the upsert fills it in.
    _session(studio)
    run_import(claude.SOURCE)
    tid = thread_id_for(claude.SOURCE, "s1")
    conn = studio_db.get_connection()
    try:
        conn.execute("UPDATE chat_threads SET updated_at = NULL WHERE id = ?", (tid,))
        conn.commit()
    finally:
        conn.close()

    run_import(claude.SOURCE)

    conn = studio_db.get_connection()
    try:
        row = conn.execute("SELECT updated_at FROM chat_threads WHERE id = ?", (tid,)).fetchone()
    finally:
        conn.close()
    assert row["updated_at"] is not None


def test_a_stale_ledger_mark_is_still_rewritten(studio):
    _session(studio)
    run_import(claude.SOURCE)
    studio_db.record_external_import_mark("claude", "s1", 2, "older")

    run_import(claude.SOURCE)

    assert studio_db.get_external_import_mark("claude", "s1") == (2, claude._REVISION)


_STORED_VARIANTS = {
    "model_id": ["", "m1", None],
    "model_gguf_variant": [None, "", "Q4"],
    "pair_id": [None, "", "p1"],
    "updated_at": [None, 5, 5.0],
    "fork_title_base": [None, "Base"],
    "settings_json": [None, '{"a": 1}', '{"a":1}', "not json"],
    "scheduled_task_id": [None, "", "t1"],
}


def _raw(conn, tid):
    row = conn.execute("SELECT * FROM chat_threads WHERE id = ?", (tid,)).fetchone()
    return {k: (type(v), v) for k, v in dict(row).items()}


@pytest.mark.parametrize("title", ["Chat", ""])
def test_the_noop_check_never_skips_a_write_that_changes_the_row(studio, title):
    """Every stored-row shape: the predicate says no-op only when the real upsert changes nothing."""
    studio_db.upsert_chat_project(
        {
            "id": "p",
            "name": "P",
            "instructions": "",
            "archived": False,
            "createdAt": 1,
            "updatedAt": 1,
        }
    )
    columns = list(_STORED_VARIANTS)
    skipped = 0
    for index, values in enumerate(itertools.product(*_STORED_VARIANTS.values())):
        tid = f"t{index}"
        conn = studio_db.get_connection()
        try:
            conn.execute(
                f"INSERT INTO chat_threads (id, title, model_type, project_id, archived, created_at, "
                f"{', '.join(columns)}) VALUES (?, ?, 'base', 'p', 0, 3, "
                f"{', '.join('?' for _ in columns)})",
                (tid, title, *values),
            )
            conn.commit()
            existing, stored = studio_db.get_chat_thread_with_row(tid)
            thread = {**existing, "updatedAt": max(4, int(existing.get("updatedAt") or 0))}
            noop = studio_db.chat_thread_upsert_is_noop(thread, stored)
            before = _raw(conn, tid)
            studio_db.upsert_chat_thread(thread)
            after = _raw(conn, tid)
        finally:
            conn.close()
        if noop:
            skipped += 1
            assert after == before, dict(zip(columns, values))
    # The shape the importer itself writes is recognised; an empty title is always rewritten.
    assert (skipped > 0) == bool(title)


def test_sync_can_skip_reading_the_rows_back(studio, monkeypatch):
    studio_db.upsert_chat_thread({"id": "t", "modelType": "base", "createdAt": 1, "updatedAt": 1})
    message = {
        "id": "m",
        "threadId": "t",
        "parentId": None,
        "role": "user",
        "content": [{"type": "text", "text": "hi"}],
        "createdAt": 2,
    }
    monkeypatch.setattr(studio_db, "list_chat_messages", lambda _t: pytest.fail("read back"))
    assert studio_db.sync_chat_messages("t", [message], return_rows = False) is None


def _rollout(home, name, meta, lines):
    path = home / "sessions" / "2026" / "09" / "22" / f"rollout-2026-09-22T12-00-00-{name}.jsonl"
    event = lambda kind, msg, n: {  # noqa: E731
        "timestamp": "2026-09-22T12:00:00Z",
        "type": "event_msg",
        "payload": {"type": kind, "message": msg},
        "ordinal": n,
    }
    body = [
        event("user_message" if i % 2 == 0 else "agent_message", t, n)
        for i, (t, n) in enumerate(lines)
    ]
    head = {"timestamp": "2026-09-22T12:00:00Z", "type": "session_meta", "payload": meta}
    return _write(path, [{**head, "ordinal": lines[0][1] - 1}, *body])


def test_codex_forks_share_one_rollout_scan(tmp_path, monkeypatch, studio):
    home = tmp_path / "codex"
    monkeypatch.setenv(codex.SOURCE.home_env, str(home))
    _rollout(home, "base", {"id": "base", "cwd": "/w"}, [("Hello", 1), ("Hi", 2)])
    for k in range(4):
        meta = {
            "id": f"f{k}",
            "cwd": "/w",
            "history_base": {"thread_id": "base", "end_ordinal_exclusive": 3},
        }
        _rollout(home, f"f{k}", meta, [(f"Fork {k}", 4), ("Ok", 5)])
    scans = []
    real = codex._rollouts
    monkeypatch.setattr(codex, "_rollouts", lambda h: scans.append(h) or real(h))

    assert run_import(codex.SOURCE).new_chats == 5

    assert len(scans) == 2
    fork = studio_db.list_chat_messages(thread_id_for(codex.SOURCE, "f0"))
    assert sorted(m["content"][0]["text"] for m in fork) == ["Fork 0", "Hello", "Hi", "Ok"]


def test_codex_fork_base_moved_since_the_scan_is_found_again(tmp_path, monkeypatch, studio):
    home = tmp_path / "codex"
    monkeypatch.setenv(codex.SOURCE.home_env, str(home))
    base = _rollout(home, "base", {"id": "base", "cwd": "/w"}, [("Hello", 1), ("Hi", 2)])
    meta = {
        "id": "f",
        "cwd": "/w",
        "history_base": {"thread_id": "base", "end_ordinal_exclusive": 3},
    }
    fork = _rollout(home, "f", meta, [("Fork", 4), ("Ok", 5)])
    codex.list_projects(home)
    codex.read_transcript(fork, "t", "f")  # builds the map
    moved = home / "archived_sessions" / base.name
    moved.parent.mkdir(parents = True)
    base.rename(moved)

    texts = [m["content"][0]["text"] for m in codex.read_transcript(fork, "t", "f").messages]

    assert texts == ["Hello", "Hi", "Fork", "Ok"]
