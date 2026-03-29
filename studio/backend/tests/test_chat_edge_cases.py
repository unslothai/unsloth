# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""
Edge case tests for chat persistence layer (PR #4637).

Tests concurrency, large payloads, Unicode, and cross-platform concerns.
"""

from __future__ import annotations

import json
import os
import sqlite3
import sys
import tempfile
import threading
import time

_backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _backend_dir not in sys.path:
    sys.path.insert(0, _backend_dir)

_tmpdir = tempfile.mkdtemp(prefix="chat_edge_test_")
_test_db = os.path.join(_tmpdir, "test_studio.db")


def _patched_studio_db_path():
    from pathlib import Path
    return Path(_test_db)


import utils.paths as _paths_mod
_paths_mod.studio_db_path = _patched_studio_db_path

import storage.studio_db as sdb
sdb._schema_ready = False


def reset_db():
    if os.path.exists(_test_db):
        os.remove(_test_db)
    for ext in ["-wal", "-shm"]:
        p = _test_db + ext
        if os.path.exists(p):
            os.remove(p)
    sdb._schema_ready = False


def test_unicode_content():
    """Unicode in thread titles and message content should round-trip."""
    reset_db()
    now = int(time.time() * 1000)

    sdb.create_chat_thread(
        thread_id="t1", title="CJK test", model_type="base",
        model_id="", pair_id=None, created_at=now,
    )
    sdb.upsert_chat_message(
        message_id="m1", thread_id="t1", role="user",
        content=json.dumps([{"type": "text", "text": "Bonjour le monde"}]),
        attachments=None, metadata=None, created_at=now,
    )

    msgs = sdb.get_chat_thread_messages("t1")
    parsed = json.loads(msgs[0]["content"])
    assert parsed[0]["text"] == "Bonjour le monde"

    # Emoji
    sdb.update_chat_thread("t1", title="Chat with emojis")
    threads = sdb.list_chat_threads()
    assert threads[0]["title"] == "Chat with emojis"

    print("  PASS: test_unicode_content")


def test_large_message_content():
    """Large JSON content (simulating many tool-call parts) should persist."""
    reset_db()
    now = int(time.time() * 1000)

    sdb.create_chat_thread(
        thread_id="t1", title="Large", model_type="base",
        model_id="", pair_id=None, created_at=now,
    )

    # ~500 KB of content
    parts = [{"type": "text", "text": "x" * 1000} for _ in range(500)]
    content = json.dumps(parts)

    sdb.upsert_chat_message(
        message_id="m1", thread_id="t1", role="assistant",
        content=content, attachments=None, metadata=None, created_at=now,
    )

    msgs = sdb.get_chat_thread_messages("t1")
    assert len(msgs) == 1
    assert len(msgs[0]["content"]) == len(content)

    print("  PASS: test_large_message_content")


def test_concurrent_thread_creation():
    """Multiple threads creating threads concurrently should not corrupt the DB."""
    reset_db()
    now = int(time.time() * 1000)
    errors = []

    def create_thread(i):
        try:
            sdb.create_chat_thread(
                thread_id=f"t{i}", title=f"Thread {i}", model_type="base",
                model_id="", pair_id=None, created_at=now + i,
            )
        except Exception as e:
            errors.append((i, str(e)))

    threads = [threading.Thread(target=create_thread, args=(i,)) for i in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(errors) == 0, f"Concurrent creation errors: {errors}"

    all_threads = sdb.list_chat_threads()
    assert len(all_threads) == 20, f"Expected 20 threads, got {len(all_threads)}"

    print("  PASS: test_concurrent_thread_creation")


def test_concurrent_message_upsert():
    """Multiple threads upserting messages concurrently should not lose data."""
    reset_db()
    now = int(time.time() * 1000)
    errors = []

    sdb.create_chat_thread(
        thread_id="t1", title="Thread", model_type="base",
        model_id="", pair_id=None, created_at=now,
    )

    def upsert_msg(i):
        try:
            sdb.upsert_chat_message(
                message_id=f"m{i}", thread_id="t1", role="user",
                content=f'[{{"type":"text","text":"msg{i}"}}]',
                attachments=None, metadata=None, created_at=now + i,
            )
        except Exception as e:
            errors.append((i, str(e)))

    threads = [threading.Thread(target=upsert_msg, args=(i,)) for i in range(50)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(errors) == 0, f"Concurrent upsert errors: {errors}"

    msgs = sdb.get_chat_thread_messages("t1")
    assert len(msgs) == 50, f"Expected 50 messages, got {len(msgs)}"

    print("  PASS: test_concurrent_message_upsert")


def test_empty_content_fields():
    """Empty strings and None values for optional fields."""
    reset_db()
    now = int(time.time() * 1000)

    sdb.create_chat_thread(
        thread_id="t1", title="Thread", model_type="base",
        model_id="", pair_id=None, created_at=now,
    )

    # Empty content string (valid)
    sdb.upsert_chat_message(
        message_id="m1", thread_id="t1", role="user",
        content="[]", attachments=None, metadata=None, created_at=now,
    )

    msgs = sdb.get_chat_thread_messages("t1")
    assert msgs[0]["content"] == "[]"
    assert msgs[0]["attachments"] is None
    assert msgs[0]["metadata"] is None

    # Update with metadata
    sdb.upsert_chat_message(
        message_id="m1", thread_id="t1", role="user",
        content="[]", attachments='[]', metadata='{}', created_at=now,
    )

    msgs = sdb.get_chat_thread_messages("t1")
    assert msgs[0]["attachments"] == "[]"
    assert msgs[0]["metadata"] == "{}"

    print("  PASS: test_empty_content_fields")


def test_pair_id_threading():
    """Threads with the same pair_id can be created and queried."""
    reset_db()
    now = int(time.time() * 1000)

    sdb.create_chat_thread(
        thread_id="t1", title="Base Thread", model_type="base",
        model_id="llama", pair_id="pair-abc", created_at=now,
    )
    sdb.create_chat_thread(
        thread_id="t2", title="LoRA Thread", model_type="lora",
        model_id="adapter", pair_id="pair-abc", created_at=now + 1,
    )

    # Both threads exist
    all_threads = sdb.list_chat_threads()
    assert len(all_threads) == 2

    # Filter by model_type
    base_threads = sdb.list_chat_threads(model_type="base")
    assert len(base_threads) == 1
    assert base_threads[0]["id"] == "t1"

    lora_threads = sdb.list_chat_threads(model_type="lora")
    assert len(lora_threads) == 1
    assert lora_threads[0]["id"] == "t2"

    # Both share pair_id
    assert all_threads[0]["pair_id"] == "pair-abc"
    assert all_threads[1]["pair_id"] == "pair-abc"

    print("  PASS: test_pair_id_threading")


def test_updated_at_is_set_on_update():
    """update_chat_thread should set updated_at to current timestamp."""
    reset_db()
    now = int(time.time() * 1000)

    sdb.create_chat_thread(
        thread_id="t1", title="Thread", model_type="base",
        model_id="", pair_id=None, created_at=now,
    )

    threads = sdb.list_chat_threads()
    original_updated = threads[0]["updated_at"]
    assert original_updated == now  # created_at == updated_at on creation

    time.sleep(0.01)  # Ensure timestamp difference
    sdb.update_chat_thread("t1", title="New Title")

    threads = sdb.list_chat_threads()
    new_updated = threads[0]["updated_at"]
    assert new_updated > original_updated, (
        f"updated_at should increase after update: {new_updated} <= {original_updated}"
    )

    print("  PASS: test_updated_at_is_set_on_update")


def test_hydrate_empty_db():
    """get_all_chat_data on an empty DB returns empty lists."""
    reset_db()

    data = sdb.get_all_chat_data()
    assert data["threads"] == []
    assert data["messages"] == []

    print("  PASS: test_hydrate_empty_db")


def main():
    print("Running chat edge case tests...")
    test_unicode_content()
    test_large_message_content()
    test_concurrent_thread_creation()
    test_concurrent_message_upsert()
    test_empty_content_fields()
    test_pair_id_threading()
    test_updated_at_is_set_on_update()
    test_hydrate_empty_db()
    print(f"\nAll 8 edge case tests passed!")

    import shutil
    shutil.rmtree(_tmpdir, ignore_errors=True)


if __name__ == "__main__":
    main()
