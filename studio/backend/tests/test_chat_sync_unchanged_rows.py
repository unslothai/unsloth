# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""sync_chat_messages skips rows the thread already stores unchanged.

The client resends the whole thread on every run start and end, so the stored state after any
sequence of syncs must match a sync that rewrites every row, while unchanged rows cost no writes
and no attachment inventory rebuild.
"""

import base64
import copy
import hashlib
import random
import sqlite3
import threading

import pytest

from storage import studio_db
from utils.paths import studio_db_path

_THREAD = "thread-sync"
# Wall-clock columns that differ between two runs of the same operations.
_CLOCK_COLUMNS = {
    "chat_attachment_inventory_state": {"backfilled_at"},
    "chat_attachment_tombstones": {"deleted_at"},
}


def _use_home(path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(path))
    monkeypatch.setenv("UNSLOTH_STUDIO_PROJECTS_HOME", str(path / "Projects"))
    monkeypatch.setattr(studio_db, "_schema_ready", set())


def _thread(thread_id = _THREAD) -> dict:
    return {
        "id": thread_id,
        "title": "Sync",
        "modelType": "base",
        "modelId": "test-model",
        "pairId": None,
        "archived": False,
        "createdAt": 1_700_000_000_000,
    }


def _image(rng: random.Random) -> str:
    return "data:image/png;base64," + base64.b64encode(rng.randbytes(48)).decode()


def _document(rng: random.Random, message_id: str) -> dict:
    attachment_id = f"doc-{message_id}-{rng.randrange(1000)}"
    return {
        "id": attachment_id,
        "type": "document",
        "name": "notes.txt",
        "contentType": "text/plain",
        "content": [{"type": "text", "text": f"doc {rng.random()}"}],
        "original": {
            "sha256": hashlib.sha256(attachment_id.encode()).hexdigest(),
            "sizeBytes": rng.randrange(10, 10_000),
        },
    }


def _new_message(rng: random.Random, index: int, parent_id) -> dict:
    message_id = f"m{index:03d}"
    role = "user" if parent_id is None or rng.random() < 0.5 else "assistant"
    content = [{"type": "text", "text": f"text {index} {rng.random()}"}]
    message = {
        "id": message_id,
        "threadId": _THREAD,
        "parentId": parent_id,
        "role": role,
        "content": content,
        "createdAt": 1_700_000_001_000 + index * 10,
    }
    if rng.random() < 0.3:
        content.append({"type": "image", "image": _image(rng)})
    if role == "user" and rng.random() < 0.3:
        message["attachments"] = [_document(rng, message_id)]
    if rng.random() < 0.4:
        message["metadata"] = {"model": "m", "n": rng.randrange(5)}
    return message


def _edit(rng: random.Random, message: dict) -> dict:
    edited = copy.deepcopy(message)
    kind = rng.randrange(6)
    if kind == 0:
        edited["content"] = [{"type": "text", "text": f"edited {rng.random()}"}]
    elif kind == 1:
        edited["content"] = [*edited["content"], {"type": "image", "image": _image(rng)}]
    elif kind == 2:
        edited["attachments"] = [*(edited.get("attachments") or []), _document(rng, edited["id"])]
    elif kind == 3:
        edited.pop("attachments", None)
    elif kind == 4:
        edited["metadata"] = {"edited": rng.randrange(100)}
    else:
        edited["createdAt"] += rng.choice([-5, 5])
    return edited


def _ops(seed: int, steps: int) -> list[tuple]:
    """A deterministic client session: syncs with edits, branches, prunes, duplicates, uploads deleted
    through the Data tab, single-message saves and a dirty inventory."""
    rng = random.Random(seed)
    client: dict[str, dict] = {}
    ops: list[tuple] = []
    index = 0
    for _ in range(steps):
        roll = rng.random()
        if roll < 0.08 and client:
            ops.append(("delete_attachment", rng.choice(sorted(client))))
            continue
        if roll < 0.13:
            ops.append(("dirty",))
            continue
        if roll < 0.23 and client:
            message = _edit(rng, client[rng.choice(sorted(client))])
            client[message["id"]] = message
            ops.append(("upsert", copy.deepcopy(message)))
            continue
        for _ in range(rng.randrange(0, 3)):
            # A parent other than the newest message is a new branch.
            parent = rng.choice([None, *sorted(client)]) if client else None
            message = _new_message(rng, index, parent)
            index += 1
            client[message["id"]] = message
        for message_id in rng.sample(sorted(client), k = min(len(client), rng.randrange(0, 3))):
            client[message_id] = _edit(rng, client[message_id])
        snapshot = [copy.deepcopy(client[message_id]) for message_id in sorted(client)]
        rng.shuffle(snapshot)
        if snapshot and rng.random() < 0.15:
            # A duplicate id whose first copy is stale, then the current one.
            stale = _edit(rng, rng.choice(snapshot))
            snapshot.insert(0, stale)
        prune = False
        deleted: list[str] = []
        if snapshot and rng.random() < 0.2:
            dropped = rng.choice(sorted(client))
            client.pop(dropped)
            snapshot = [m for m in snapshot if m["id"] != dropped]
            prune = True
            if rng.random() < 0.5:
                deleted = [dropped]
        ops.append(("sync", snapshot, prune, deleted))
    return ops


def _apply(op: tuple) -> None:
    kind = op[0]
    if kind == "sync":
        _, snapshot, prune, deleted = op
        studio_db.sync_chat_messages(
            _THREAD,
            copy.deepcopy(snapshot),
            prune_missing = prune,
            deleted_message_ids = deleted,
        )
    elif kind == "upsert":
        studio_db.upsert_chat_message(copy.deepcopy(op[1]))
    elif kind == "delete_attachment":
        stored = studio_db.get_chat_message(_THREAD, op[1])
        if stored is None:
            return
        ids = [a["id"] for a in stored.get("attachments") or [] if isinstance(a, dict)]
        ids += [
            part_id
            for part in stored.get("content") or []
            if isinstance(part, dict)
            for part_id in [studio_db._content_part_id(part)]
            if part_id is not None
        ]
        if ids:
            studio_db.delete_chat_attachment(op[1], sorted(ids)[0])
    elif kind == "dirty":
        conn = studio_db.get_connection()
        try:
            conn.execute("UPDATE chat_attachment_inventory_state SET dirty = 1")
            conn.commit()
        finally:
            conn.close()


def _dump() -> dict:
    conn = sqlite3.connect(studio_db_path())
    try:
        tables = [
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%' "
                "ORDER BY name"
            )
        ]
        dump = {}
        for table in tables:
            columns = [row[1] for row in conn.execute(f"PRAGMA table_info({table})")]
            kept = [c for c in columns if c not in _CLOCK_COLUMNS.get(table, set())]
            select = ", ".join(f'"{c}"' for c in kept)
            # rowid too where there is one, so a delete + reinsert would show.
            try:
                rows = conn.execute(f"SELECT rowid, {select} FROM {table}").fetchall()
            except sqlite3.OperationalError:
                rows = conn.execute(f"SELECT {select} FROM {table}").fetchall()
            dump[table] = sorted(rows, key = repr)
        return dump
    finally:
        conn.close()


def _run(path, monkeypatch, ops, *, rewrite_all: bool) -> dict:
    path.mkdir()
    _use_home(path, monkeypatch)
    if rewrite_all:
        # Nothing looks stored, so every row is upserted and its inventory rebuilt: the old behaviour.
        monkeypatch.setattr(
            studio_db, "_stored_chat_message_values", lambda *args: {}, raising = False
        )
    studio_db.upsert_chat_thread(_thread())
    for op in ops:
        try:
            _apply(op)
        except (studio_db.ChatMessageConflictError, studio_db.ChatMessageProtectedError):
            pass
    return _dump()


@pytest.mark.parametrize("seed", range(12))
def test_stored_state_matches_rewriting_every_row(tmp_path, monkeypatch, seed):
    ops = _ops(seed, steps = 30)
    assert any(op[0] == "sync" for op in ops)
    rewritten = _run(tmp_path / "rewrite_all", monkeypatch, ops, rewrite_all = True)
    monkeypatch.undo()
    skipped = _run(tmp_path / "skip_unchanged", monkeypatch, ops, rewrite_all = False)
    assert skipped == rewritten
    assert skipped["chat_messages"], "the session must leave messages behind"


def test_inventory_matches_full_rebuild_after_session(tmp_path, monkeypatch):
    _use_home(tmp_path, monkeypatch)
    studio_db.upsert_chat_thread(_thread())
    for op in _ops(99, steps = 40):
        try:
            _apply(op)
        except (studio_db.ChatMessageConflictError, studio_db.ChatMessageProtectedError):
            pass
    before = _dump()["chat_attachment_inventory"]
    assert before, "the session must leave attachments behind"
    _apply(("dirty",))
    studio_db.list_chat_messages(_THREAD)
    conn = studio_db.get_connection()
    try:
        studio_db._ensure_chat_attachment_inventory_current(conn)
    finally:
        conn.close()
    assert _dump()["chat_attachment_inventory"] == before


def _thread_messages(count: int) -> list[dict]:
    rng = random.Random(7)
    messages = []
    parent = None
    for index in range(count):
        message = _new_message(rng, index, parent)
        message["content"].append({"type": "image", "image": _image(rng)})
        messages.append(message)
        parent = message["id"]
    return messages


def _row_write(sql: str) -> bool:
    """A statement writing message or inventory rows (thread bump and inventory state excluded)."""
    words = " ".join(sql.split())
    return any(
        marker in words
        for marker in (
            "INSERT INTO chat_messages",
            "UPDATE chat_messages",
            "DELETE FROM chat_messages",
            "INSERT INTO chat_attachment_inventory (",
            "DELETE FROM chat_attachment_inventory WHERE",
        )
    )


def test_resending_unchanged_thread_writes_nothing(tmp_path, monkeypatch):
    _use_home(tmp_path, monkeypatch)
    studio_db.upsert_chat_thread(_thread())
    messages = _thread_messages(40)
    studio_db.sync_chat_messages(_THREAD, copy.deepcopy(messages))

    rebuilt: list[str] = []
    real_replace = studio_db._replace_chat_attachment_inventory

    def spy(conn, message_id, *args, **kwargs):
        rebuilt.append(message_id)
        return real_replace(conn, message_id, *args, **kwargs)

    monkeypatch.setattr(studio_db, "_replace_chat_attachment_inventory", spy)
    hashed = []
    real_sha256 = hashlib.sha256

    class _Hashlib:
        def __getattr__(self, name):
            return getattr(hashlib, name)

        @staticmethod
        def sha256(
            data = b"",
            *args,
            **kwargs,
        ):
            hashed.append(len(data))
            return real_sha256(data, *args, **kwargs)

    monkeypatch.setattr(studio_db, "hashlib", _Hashlib())
    writes: list[int] = []
    real_get_connection = studio_db.get_connection

    def counting_connection(*args, **kwargs):
        conn = real_get_connection(*args, **kwargs)
        conn.set_trace_callback(lambda sql: writes.append(sql) if _row_write(sql) else None)
        return conn

    monkeypatch.setattr(studio_db, "get_connection", counting_connection)

    synced = studio_db.sync_chat_messages(_THREAD, copy.deepcopy(messages))
    assert [m["id"] for m in synced] == [m["id"] for m in messages]
    assert rebuilt == []
    assert hashed == []
    assert writes == []

    # One edited and one new message: only those two are written and re-inventoried.
    edited = copy.deepcopy(messages)
    edited[5]["content"] = [{"type": "text", "text": "edited"}]
    new = _new_message(random.Random(1), 40, messages[-1]["id"])
    studio_db.sync_chat_messages(_THREAD, [*edited, new])
    assert sorted(rebuilt) == sorted([edited[5]["id"], new["id"]])
    stored = {m["id"]: m for m in studio_db.list_chat_messages(_THREAD)}
    assert stored[edited[5]["id"]]["content"] == [{"type": "text", "text": "edited"}]
    assert new["id"] in stored


def test_unchanged_rows_still_bump_thread_and_reconcile_tombstones(tmp_path, monkeypatch):
    _use_home(tmp_path, monkeypatch)
    studio_db.upsert_chat_thread(_thread())
    rng = random.Random(3)
    message = _new_message(rng, 0, None)
    message["attachments"] = [_document(rng, message["id"])]
    studio_db.sync_chat_messages(_THREAD, [copy.deepcopy(message)])
    assert studio_db.delete_chat_attachment(message["id"], message["attachments"][0]["id"])
    after_delete = studio_db.get_chat_message(_THREAD, message["id"])
    # A stale tab resends the deleted upload: the stored row already equals the reconciled one.
    studio_db.sync_chat_messages(_THREAD, [copy.deepcopy(message)])
    assert studio_db.get_chat_message(_THREAD, message["id"]) == after_delete
    assert _dump()["chat_attachment_inventory"] == []


def test_two_tabs_syncing_concurrently_keep_inventory_consistent(tmp_path, monkeypatch):
    _use_home(tmp_path, monkeypatch)
    studio_db.upsert_chat_thread(_thread())
    base_messages = _thread_messages(20)
    studio_db.sync_chat_messages(_THREAD, copy.deepcopy(base_messages))
    rng = random.Random(11)
    tab_a = copy.deepcopy(base_messages)
    tab_b = copy.deepcopy(base_messages)
    tab_a[3]["content"] = [{"type": "image", "image": _image(rng)}]
    tab_b[3]["content"] = [{"type": "text", "text": "tab b"}]
    tab_b[3]["attachments"] = [_document(rng, tab_b[3]["id"])]
    errors: list[BaseException] = []

    def tab(snapshot):
        try:
            for _ in range(15):
                studio_db.sync_chat_messages(_THREAD, copy.deepcopy(snapshot))
        except BaseException as exc:  # surfaced below
            errors.append(exc)

    threads = [threading.Thread(target = tab, args = (s,)) for s in (tab_a, tab_b)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []

    stored = studio_db.get_chat_message(_THREAD, tab_a[3]["id"])
    assert stored["content"] in (tab_a[3]["content"], tab_b[3]["content"])
    inventory = _dump()["chat_attachment_inventory"]
    _apply(("dirty",))
    conn = studio_db.get_connection()
    try:
        studio_db._ensure_chat_attachment_inventory_current(conn)
    finally:
        conn.close()
    assert _dump()["chat_attachment_inventory"] == inventory
