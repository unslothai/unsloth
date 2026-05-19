"""
Comprehensive simulation tests for PR #5272 — POST-FIX edition.

After subject scoping applies (A1 subject scoping, A2 hijack guard, A3 confirm flag,
A4 atomic settings merge), every assertion that previously confirmed a bug
is inverted to confirm the FIX. Each test is self-contained (own tmpdir +
fresh studio_db module) so we can isolate state and migration semantics.
"""

from __future__ import annotations

import json
import os
import sqlite3
import sys
import threading
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import sim_harness

SUB = "test-subject"
OTHER = "other-subject"


# ---------- Fixtures -------------------------------------------------- #


@pytest.fixture()
def env():
    home, db, ch = sim_harness.mount()
    yield home, db, ch
    sim_harness.remove_tmp(home)


@pytest.fixture()
def http_env():
    app, db, ch = sim_harness.fresh_app()
    client = TestClient(app)
    yield client, db, ch
    home = Path(os.environ["UNSLOTH_STUDIO_HOME"])
    sim_harness.remove_tmp(home)


def _thread(thread_id="t1", title="T", model_type="base", model_id="m", pair_id=None, archived=False, created_at=1_700_000_000_000):
    return {
        "id": thread_id,
        "title": title,
        "modelType": model_type,
        "modelId": model_id,
        "pairId": pair_id,
        "archived": archived,
        "createdAt": created_at,
    }


def _msg(message_id, thread_id="t1", role="user", content="hi", created_at=1, parent_id=None, metadata=None, attachments=None):
    return {
        "id": message_id,
        "threadId": thread_id,
        "parentId": parent_id,
        "role": role,
        "content": [{"type": "text", "text": content}] if isinstance(content, str) else content,
        "attachments": attachments,
        "metadata": metadata,
        "createdAt": created_at,
    }


# ==========================================================================
# A) SCHEMA + MIGRATION
# ==========================================================================


def test_schema_idempotent_called_twice(env):
    _, db, _ = env
    conn = db.get_connection()
    db._ensure_schema(conn)
    conn.close()
    tables = {
        r[0]
        for r in db.get_connection()
        .execute("select name from sqlite_master where type='table'")
        .fetchall()
    }
    assert {"chat_threads", "chat_messages", "chat_settings"} <= tables


def test_migration_from_pre_pr_db(env):
    """An older studio.db lacking `subject` and the two container columns is
    upgraded in place: existing rows survive under the legacy sentinel
    subject."""
    home, db, _ = env
    db_path = db.studio_db_path()
    conn = sqlite3.connect(str(db_path))
    conn.execute("DROP TABLE IF EXISTS chat_threads")
    conn.execute("DROP TABLE IF EXISTS chat_messages")
    conn.execute("DROP TABLE IF EXISTS chat_settings")
    conn.execute(
        """CREATE TABLE chat_threads (
            id TEXT NOT NULL PRIMARY KEY,
            title TEXT NOT NULL,
            model_type TEXT NOT NULL,
            model_id TEXT,
            pair_id TEXT,
            archived INTEGER NOT NULL DEFAULT 0,
            created_at INTEGER NOT NULL
        )"""
    )
    conn.execute(
        "INSERT INTO chat_threads (id, title, model_type, created_at) "
        "VALUES ('old-1', 'Pre-fix thread', 'base', 1)"
    )
    conn.commit()
    conn.close()
    db._schema_ready = False
    # The pre-A1 row was claimed under the legacy sentinel subject
    got = db.get_chat_thread("old-1", subject=db._LEGACY_UNSCOPED_SUBJECT)
    assert got is not None
    assert got["title"] == "Pre-fix thread"
    cols = {
        r[1]
        for r in db.get_connection()
        .execute("PRAGMA table_info(chat_threads)")
        .fetchall()
    }
    assert "openai_code_exec_container_id" in cols
    assert "anthropic_code_exec_container_id" in cols
    assert "subject" in cols


def test_migration_legacy_subject_invisible_to_real_user(env):
    """Pre-fix rows migrated under `__legacy_unscoped__` should NOT appear to
    a real authenticated subject. The frontend can expose an admin path to
    reassign them, but a fresh login must not see another user's chats."""
    home, db, _ = env
    db_path = db.studio_db_path()
    conn = sqlite3.connect(str(db_path))
    conn.execute("DROP TABLE IF EXISTS chat_threads")
    conn.execute("DROP TABLE IF EXISTS chat_messages")
    conn.execute("DROP TABLE IF EXISTS chat_settings")
    conn.execute(
        """CREATE TABLE chat_threads (
            id TEXT NOT NULL PRIMARY KEY,
            title TEXT NOT NULL,
            model_type TEXT NOT NULL,
            model_id TEXT,
            pair_id TEXT,
            archived INTEGER NOT NULL DEFAULT 0,
            created_at INTEGER NOT NULL
        )"""
    )
    conn.execute(
        "INSERT INTO chat_threads (id, title, model_type, created_at) "
        "VALUES ('migrated', 'Legacy chat', 'base', 1)"
    )
    conn.commit()
    conn.close()
    db._schema_ready = False
    assert db.get_chat_thread("migrated", subject=SUB) is None
    assert db.get_chat_thread("migrated", subject=db._LEGACY_UNSCOPED_SUBJECT) is not None


# ==========================================================================
# B) CRUD ROUND-TRIPS (all scoped by subject)
# ==========================================================================


def test_basic_thread_crud(env):
    _, db, _ = env
    db.upsert_chat_thread(_thread("t1", title="hi"), subject=SUB)
    g = db.get_chat_thread("t1", subject=SUB)
    assert g["title"] == "hi"
    db.update_chat_thread("t1", {"title": "renamed"}, subject=SUB)
    assert db.get_chat_thread("t1", subject=SUB)["title"] == "renamed"
    n = db.delete_chat_threads(["t1"], subject=SUB)
    assert n == 1
    assert db.get_chat_thread("t1", subject=SUB) is None


def test_message_round_trip_with_metadata(env):
    _, db, _ = env
    db.upsert_chat_thread(_thread("t1"), subject=SUB)
    db.upsert_chat_message(
        _msg("m1", metadata={"timing": {"ms": 42}, "custom": {"k": "v"}}),
        subject=SUB,
    )
    got = db.list_chat_messages("t1", subject=SUB)
    assert len(got) == 1
    assert got[0]["metadata"]["timing"]["ms"] == 42


def test_cascade_delete_messages_when_thread_deleted(env):
    _, db, _ = env
    db.upsert_chat_thread(_thread("t1"), subject=SUB)
    db.upsert_chat_message(_msg("m1"), subject=SUB)
    db.delete_chat_threads(["t1"], subject=SUB)
    assert db.list_chat_messages("t1", subject=SUB) == []


# ==========================================================================
# C) sync_chat_messages PRUNE SEMANTICS + CHUNKING
# ==========================================================================


def test_sync_prune_missing(env):
    _, db, _ = env
    db.upsert_chat_thread(_thread("t1"), subject=SUB)
    db.sync_chat_messages(
        "t1",
        [_msg("m1", created_at=1), _msg("m2", created_at=2), _msg("m3", created_at=3)],
        subject=SUB,
        prune_missing=True,
    )
    got = db.sync_chat_messages(
        "t1", [_msg("m2", created_at=2)], subject=SUB, prune_missing=True
    )
    assert [m["id"] for m in got] == ["m2"]


def test_chunking_boundary_999_threads(env):
    _, db, _ = env
    n = 1500
    for i in range(n):
        db.upsert_chat_thread(_thread(f"t{i}", model_id="m", created_at=i), subject=SUB)
    for i in range(n):
        db.upsert_chat_message(_msg(f"m{i}", thread_id=f"t{i}", created_at=i), subject=SUB)
    ids = [f"t{i}" for i in range(n)]
    got = db.list_chat_messages_for_threads(ids, subject=SUB)
    assert len(got) == n
    cas = [m["createdAt"] for m in got]
    assert cas == sorted(cas)


def test_chunking_boundary_at_exactly_900_and_901(env):
    _, db, _ = env
    for n in (900, 901):
        for i in range(n):
            db.upsert_chat_thread(_thread(f"x{n}-{i}", created_at=i), subject=SUB)
            db.upsert_chat_message(_msg(f"xm{n}-{i}", thread_id=f"x{n}-{i}", created_at=i), subject=SUB)
        ids = [f"x{n}-{i}" for i in range(n)]
        got = db.list_chat_messages_for_threads(ids, subject=SUB)
        assert len(got) == n, f"failed at n={n}"


# ==========================================================================
# D) HIJACK FIXES (A2)
# ==========================================================================


def test_cross_thread_message_hijack_via_upsert_rejected(env):
    """Same id under different threads (same subject) must NOT re-parent."""
    _, db, _ = env
    db.upsert_chat_thread(_thread("t_owner"), subject=SUB)
    db.upsert_chat_thread(_thread("t_attacker"), subject=SUB)
    db.upsert_chat_message(_msg("shared", thread_id="t_owner", content="legit"), subject=SUB)
    with pytest.raises(db.ChatMessageThreadMismatch):
        db.upsert_chat_message(_msg("shared", thread_id="t_attacker", content="hijack"), subject=SUB)
    # Original row preserved
    owner_msgs = db.list_chat_messages("t_owner", subject=SUB)
    assert len(owner_msgs) == 1
    assert "legit" in json.dumps(owner_msgs[0]["content"])
    assert db.list_chat_messages("t_attacker", subject=SUB) == []


def test_cross_thread_message_hijack_via_sync_rejected(env):
    _, db, _ = env
    db.upsert_chat_thread(_thread("a"), subject=SUB)
    db.upsert_chat_thread(_thread("b"), subject=SUB)
    db.upsert_chat_message(_msg("victim", thread_id="a", content="orig"), subject=SUB)
    # Body says threadId="a" but URL is "b" — this is the silent-rewrite bug
    with pytest.raises(db.ChatMessageThreadMismatch):
        db.sync_chat_messages(
            "b",
            [_msg("victim", thread_id="a", content="hijack")],
            subject=SUB,
        )
    # Or: same id but body claims threadId="b" — pre-existing same-id row
    # under thread "a" must reject the move.
    with pytest.raises(db.ChatMessageThreadMismatch):
        db.sync_chat_messages(
            "b",
            [_msg("victim", thread_id="b", content="hijack2")],
            subject=SUB,
        )
    # Original survives
    a_msgs = db.list_chat_messages("a", subject=SUB)
    assert len(a_msgs) == 1


def test_same_message_id_allowed_across_subjects(env):
    """Two subjects can independently use the same client-generated id."""
    _, db, _ = env
    db.upsert_chat_thread(_thread("t1"), subject=SUB)
    db.upsert_chat_thread(_thread("t1"), subject=OTHER)
    db.upsert_chat_message(_msg("m1", thread_id="t1", content="alice"), subject=SUB)
    db.upsert_chat_message(_msg("m1", thread_id="t1", content="bob"), subject=OTHER)
    alice_msgs = db.list_chat_messages("t1", subject=SUB)
    bob_msgs = db.list_chat_messages("t1", subject=OTHER)
    assert len(alice_msgs) == 1
    assert len(bob_msgs) == 1
    assert "alice" in json.dumps(alice_msgs[0]["content"])
    assert "bob" in json.dumps(bob_msgs[0]["content"])


# ==========================================================================
# E) clear_chat_history scoped by subject
# ==========================================================================


def test_clear_chat_history_only_wipes_caller_subject(env):
    _, db, _ = env
    for i in range(3):
        db.upsert_chat_thread(_thread(f"a{i}"), subject=SUB)
    for i in range(2):
        db.upsert_chat_thread(_thread(f"b{i}"), subject=OTHER)
    n = db.clear_chat_history(subject=SUB)
    assert n == 3
    assert db.count_chat_threads(subject=SUB) == 0
    assert db.count_chat_threads(subject=OTHER) == 2


# ==========================================================================
# F) EDGE CASES
# ==========================================================================


def test_unicode_emoji_thread_and_message(env):
    _, db, _ = env
    db.upsert_chat_thread(_thread("t-✨", title="Café 中文 🍜"), subject=SUB)
    db.upsert_chat_message(
        _msg("m-🦄", thread_id="t-✨", content="Hello 世界! 😀"),
        subject=SUB,
    )
    g = db.get_chat_thread("t-✨", subject=SUB)
    assert g["title"] == "Café 中文 🍜"
    msgs = db.list_chat_messages("t-✨", subject=SUB)
    assert "世界" in json.dumps(msgs[0]["content"], ensure_ascii=False)


def test_large_content_2mb(env):
    _, db, _ = env
    db.upsert_chat_thread(_thread("big"), subject=SUB)
    big_text = "A" * (2 * 1024 * 1024)
    db.upsert_chat_message(_msg("bm", thread_id="big", content=big_text), subject=SUB)
    msgs = db.list_chat_messages("big", subject=SUB)
    assert msgs[0]["content"][0]["text"] == big_text


def test_null_metadata_round_trips(env):
    _, db, _ = env
    db.upsert_chat_thread(_thread("t1"), subject=SUB)
    db.upsert_chat_message(_msg("m1", metadata=None, attachments=None), subject=SUB)
    got = db.list_chat_messages("t1", subject=SUB)
    assert got[0].get("metadata") in (None, {}, [])
    assert got[0].get("attachments") in (None, [])


def test_sql_injection_in_id_and_title(env):
    _, db, _ = env
    nasty_id = "x'); DROP TABLE chat_threads; --"
    db.upsert_chat_thread(
        _thread(nasty_id, title="'); DROP TABLE chat_threads; --"), subject=SUB
    )
    tables = {
        r[0]
        for r in db.get_connection()
        .execute("select name from sqlite_master where type='table'")
        .fetchall()
    }
    assert "chat_threads" in tables
    got = db.get_chat_thread(nasty_id, subject=SUB)
    assert got is not None


def test_message_id_with_slashes_and_paths(env):
    _, db, _ = env
    for bad in (
        "id/with/slashes",
        "id\\with\\backslashes",
        "id\nwith\nnewlines",
        "id\rwith\rcr",
        "../etc/passwd",
        "C:\\Windows\\System32",
        "id with spaces and 'quotes'",
    ):
        db.upsert_chat_thread(_thread(bad, title=f"title for {bad!r}"), subject=SUB)
        db.upsert_chat_message(_msg(f"m-{hash(bad)}", thread_id=bad), subject=SUB)
        assert db.get_chat_thread(bad, subject=SUB) is not None


# ==========================================================================
# G) ATOMIC SETTINGS MERGE (A4)
# ==========================================================================


def test_concurrent_settings_writers_no_lost_update(env):
    """With upsert_chat_settings_merge inside BEGIN IMMEDIATE, two concurrent
    writers must NOT drop one another's update."""
    _, db, _ = env
    db.upsert_chat_settings(
        {"autoTitle": False, "reasoningEffort": "low"}, subject=SUB
    )

    barrier = threading.Barrier(2)

    def worker(updates):
        barrier.wait()
        time.sleep(0.01)
        # Each thread retries on SQLITE_BUSY for up to ~1s, mirroring the
        # server's behavior under contention.
        for _ in range(50):
            try:
                db.upsert_chat_settings_merge(updates, subject=SUB)
                return
            except sqlite3.OperationalError as e:
                if "locked" in str(e) or "busy" in str(e):
                    time.sleep(0.02)
                    continue
                raise
        raise RuntimeError("worker gave up waiting for lock")

    t1 = threading.Thread(target=worker, args=({"autoTitle": True},))
    t2 = threading.Thread(target=worker, args=({"reasoningEffort": "high"},))
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    result = db.list_chat_settings(subject=SUB)
    # BOTH updates must be present
    assert result.get("autoTitle") is True
    assert result.get("reasoningEffort") == "high"


def test_settings_isolated_by_subject(env):
    _, db, _ = env
    db.upsert_chat_settings({"autoTitle": True}, subject=SUB)
    db.upsert_chat_settings({"autoTitle": False}, subject=OTHER)
    assert db.list_chat_settings(subject=SUB)["autoTitle"] is True
    assert db.list_chat_settings(subject=OTHER)["autoTitle"] is False


# ==========================================================================
# H) HTTP — A1 SUBJECT SCOPING
# ==========================================================================


def test_two_subjects_isolated_via_http(http_env):
    """Alice creates a thread; Bob must NOT see it via GET /threads. subject scoping."""
    _, _, ch = http_env
    app, _, _ = sim_harness.fresh_app()
    from auth.authentication import get_current_subject

    current = {"sub": "alice"}

    def _dep():
        return current["sub"]

    app.dependency_overrides[get_current_subject] = _dep
    client = TestClient(app)

    r = client.post(
        "/api/chat/threads",
        json={
            "id": "alice-private",
            "title": "Alice's diary",
            "modelType": "base",
            "modelId": "m",
            "pairId": None,
            "archived": False,
            "createdAt": 1,
        },
    )
    assert r.status_code == 200, r.text

    current["sub"] = "bob"
    r = client.get("/api/chat/threads")
    assert r.status_code == 200
    threads = r.json()["threads"]
    assert all(t["id"] != "alice-private" for t in threads), (
        "subject scoping failed: Bob still sees Alice's threads"
    )

    # Bob's clear (with confirm) does NOT wipe Alice's data
    r = client.delete("/api/chat?confirm=true")
    assert r.status_code == 200

    current["sub"] = "alice"
    r = client.get("/api/chat/threads")
    assert r.status_code == 200
    assert any(t["id"] == "alice-private" for t in r.json()["threads"])


# ==========================================================================
# I) HTTP — A3 CONFIRM FLAG, A2 BULK MISMATCH
# ==========================================================================


def test_http_404_on_unknown_thread(http_env):
    client, *_ = http_env
    r = client.get("/api/chat/threads/does-not-exist")
    assert r.status_code == 404


def test_http_400_on_message_id_mismatch(http_env):
    client, *_ = http_env
    client.post("/api/chat/threads", json=_thread("t1"))
    r = client.put(
        "/api/chat/threads/t1/messages/m1",
        json=_msg("DIFFERENT_ID"),
    )
    assert r.status_code == 400


def test_http_clear_requires_confirm(http_env):
    """clear-confirm: DELETE /api/chat without ?confirm=true must 400."""
    client, *_ = http_env
    for i in range(3):
        client.post("/api/chat/threads", json=_thread(f"x{i}"))
    r = client.delete("/api/chat")
    assert r.status_code == 400
    # Threads still there
    assert client.get("/api/chat/count").json()["count"] == 3
    # With confirm flag, succeeds and returns count
    r = client.delete("/api/chat?confirm=true")
    assert r.status_code == 200
    assert r.json()["count"] == 3
    assert client.get("/api/chat/count").json()["count"] == 0


def test_http_bulk_replace_rejects_mismatched_threadId(http_env):
    """hijack guard part 2: replace_thread_messages must reject body messages whose
    threadId doesn't match the URL, instead of silently rewriting them."""
    client, *_ = http_env
    client.post("/api/chat/threads", json=_thread("a"))
    client.post("/api/chat/threads", json=_thread("b"))
    # URL says 'a' but body says 'b'
    r = client.put(
        "/api/chat/threads/a/messages",
        json={"messages": [_msg("m1", thread_id="b")], "pruneMissing": False},
    )
    assert r.status_code == 400


def test_http_upsert_message_rejects_cross_thread_reparent(http_env):
    """hijack guard: PUT /threads/B/messages/m1 when m1 already exists under A → 409."""
    client, *_ = http_env
    client.post("/api/chat/threads", json=_thread("a"))
    client.post("/api/chat/threads", json=_thread("b"))
    # Create m1 under a
    r = client.put(
        "/api/chat/threads/a/messages/m1", json=_msg("m1", thread_id="a")
    )
    assert r.status_code == 200
    # Attempt to PUT m1 under b — should 409 because b's body says threadId=b
    # but m1 already exists under a
    r = client.put(
        "/api/chat/threads/b/messages/m1", json=_msg("m1", thread_id="b")
    )
    assert r.status_code == 409


# ==========================================================================
# J) EXPORT BEHAVIOR
# ==========================================================================


def test_export_returns_only_caller_threads(http_env):
    client, db, ch = http_env
    for i in range(20):
        client.post("/api/chat/threads", json=_thread(f"e{i}", created_at=i))
        client.put(
            f"/api/chat/threads/e{i}/messages/em{i}",
            json=_msg(f"em{i}", thread_id=f"e{i}", created_at=i),
        )
    r = client.get("/api/chat/export")
    assert r.status_code == 200
    body = r.json()
    assert body["threadCount"] == 20
    assert len(body["threads"]) == 20
    assert len(body["messages"]) == 20


# ==========================================================================
# K) FRONTEND COMPAT
# ==========================================================================


def test_thread_response_shape_matches_old_dexie_record(http_env):
    client, *_ = http_env
    payload = {
        "id": "compat-1",
        "title": "compat",
        "modelType": "base",
        "modelId": "m",
        "pairId": None,
        "archived": False,
        "createdAt": 1,
        "openaiCodeExecContainerId": "oai-c-1",
        "anthropicCodeExecContainerId": "ant-c-1",
    }
    r = client.post("/api/chat/threads", json=payload)
    assert r.status_code == 200
    got = r.json()
    for k in payload:
        assert k in got, f"field {k} missing in response"
    assert got["openaiCodeExecContainerId"] == "oai-c-1"
    assert got["anthropicCodeExecContainerId"] == "ant-c-1"


# ==========================================================================
# L) CROSS-PLATFORM
# ==========================================================================


def test_studio_db_path_uses_pathlib(env):
    import os as _os

    _, db, _ = env
    p = db.studio_db_path()
    assert isinstance(p, Path)
    assert _os.sep in str(p)
    assert Path(str(p)) == p
    assert p.name == "studio.db"


def test_no_hardcoded_posix_paths_in_chat_history_routes():
    bad = []
    for f in (
        sim_harness.PR_ROOT / "routes" / "chat_history.py",
        sim_harness.PR_ROOT / "storage" / "studio_db.py",
    ):
        text = f.read_text()
        for lineno, line in enumerate(text.splitlines(), 1):
            stripped = line.strip()
            if (
                stripped.startswith("#")
                or stripped.startswith('"""')
                or stripped.startswith("@router")
            ):
                continue
            if 'open("' in line and '"/' in line:
                bad.append((str(f), lineno, line))
    assert not bad


# ==========================================================================
# M) FRONTEND STATIC CONTRACT
# ==========================================================================


def test_frontend_legacy_dexie_is_never_cleared_implicitly():
    storage = (
        sim_harness.PR_ROOT.parent
        / "frontend"
        / "src"
        / "features"
        / "chat"
        / "utils"
        / "chat-history-storage.ts"
    ).read_text()
    import re

    fn_start = storage.find("async function importLegacyChatsIfNeeded")
    assert fn_start >= 0
    rest = storage[fn_start + 1 :]
    m = re.search(r"\n(?:async function |function |export )", rest)
    fn_end = fn_start + 1 + m.start() if m else len(storage)
    body = storage[fn_start:fn_end]
    assert "db.threads.clear" not in body
    assert "db.messages.clear" not in body
    clear_idx = storage.find("export async function clearStoredChats")
    assert clear_idx >= 0
    # partial-clear toast grew the function body; widen the search window.
    assert "db.threads.clear" in storage[clear_idx : clear_idx + 4000]
    assert "db.messages.clear" in storage[clear_idx : clear_idx + 4000]


def test_frontend_legacy_import_writes_via_upsert_so_idempotent_on_retry():
    storage = (
        sim_harness.PR_ROOT.parent
        / "frontend"
        / "src"
        / "features"
        / "chat"
        / "utils"
        / "chat-history-storage.ts"
    ).read_text()
    assert "saveChatThread(thread)" in storage
