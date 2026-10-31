# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Ordered sidebar pins persist across clients through the shared settings API."""

import json
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi import HTTPException
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError

from routes import chat_history
from storage import studio_db


def _save_if_current(
    expected,
    pins,
    *,
    expected_absent = False,
    operation_id = None,
    unpinned_ids = (),
    cancelled_operation_ids = (),
):
    return chat_history.compare_and_set_settings(
        {
            "expected": expected,
            "expectedAbsent": ["pinnedChatIds"] if expected_absent else [],
            "patch": {"pinnedChatIds": pins},
            "operationId": operation_id,
            "unpinnedChatIds": list(unpinned_ids),
            "cancelledOperationIds": list(cancelled_operation_ids),
        },
        current_subject = "test-user",
    )


def _seed(
    *ids,
    pair_id = None,
    archived = False,
    project_id = None,
):
    for id in ids:
        studio_db.upsert_chat_thread(
            {
                "id": id,
                "title": id,
                "modelType": "base",
                "createdAt": 1,
                "pairId": pair_id,
                "archived": archived,
                "projectId": project_id,
            }
        )


def _pins():
    return chat_history.get_pins(current_subject = "test-user").model_dump(exclude_unset = True)


def _migrate(ids):
    return chat_history.migrate_pins(
        chat_history.ChatPinsMigrationRequest(ids = ids), current_subject = "test-user"
    ).model_dump(exclude_unset = True)


def test_pin_payload_distinguishes_omitted_from_explicitly_empty():
    assert chat_history.ChatSettingsPayload().model_dump(exclude_unset = True) == {}
    assert chat_history.ChatSettingsPayload(pinnedChatIds = []).model_dump(exclude_unset = True) == {
        "pinnedChatIds": []
    }


def test_new_pin_waits_for_a_visible_legacy_thread_to_finish_importing():
    first = _save_if_current({}, ["late-import"], expected_absent = True, operation_id = "pin-late")
    assert not first.applied
    conn = studio_db.get_connection()
    try:
        assert (
            conn.execute("SELECT 1 FROM chat_pin_operations WHERE id = 'pin-late'").fetchone()
            is None
        )
    finally:
        conn.close()
    _seed("late-import")
    retry = _save_if_current({}, ["late-import"], expected_absent = True, operation_id = "pin-late")
    assert retry.applied
    assert retry.settings["pinnedChatIds"] == ["late-import"]


def test_a_deleted_comparison_pin_is_acknowledged_without_blocking_later_pins():
    _seed("left", "right", pair_id = "deleted-pair")
    _seed("keep")
    studio_db.delete_chat_threads_with_active_runs(["left", "right"])
    deleted = _save_if_current(
        {}, ["deleted-pair"], expected_absent = True, operation_id = "stale-pair"
    )
    assert deleted.applied
    assert deleted.settings["pinnedChatIds"] == []
    later = _save_if_current({"pinnedChatIds": []}, ["keep"], operation_id = "next-pin")
    assert later.applied
    assert later.settings["pinnedChatIds"] == ["keep"]


def test_pin_intent_for_a_chat_deleted_before_upgrade_does_not_block_the_queue():
    _seed("deleted-before-upgrade")
    studio_db.delete_chat_threads_with_active_runs(["deleted-before-upgrade"])
    conn = studio_db.get_connection()
    try:
        conn.execute("DELETE FROM chat_pin_deletions")
        conn.commit()
    finally:
        conn.close()
    saved = _save_if_current(
        {}, ["deleted-before-upgrade"], expected_absent = True, operation_id = "old-pin"
    )
    assert saved.applied
    assert saved.settings["pinnedChatIds"] == []


@pytest.mark.parametrize(
    "pins",
    [
        None,
        "chat-1",
        [None],
        [True],
        [42],
        [""],
        ["x" * 513],
        [chr(0xD800)],
        ["a", "a"],
        [str(i) for i in range(10_001)],
    ],
    ids = [
        "null",
        "not-list",
        "null-id",
        "boolean-id",
        "number-id",
        "empty-id",
        "long-id",
        "invalid-unicode",
        "duplicate",
        "too-many",
    ],
)
def test_invalid_pins_do_not_overwrite_saved_state(pins):
    _seed("keep")
    chat_history.put_settings({"pinnedChatIds": ["keep"]}, current_subject = "test-user")

    with pytest.raises(ValidationError):
        chat_history.ChatSettingsPayload.model_validate({"pinnedChatIds": pins})
    with pytest.raises(HTTPException) as error:
        chat_history.put_settings({"pinnedChatIds": pins}, current_subject = "test-user")

    assert error.value.status_code == 400
    assert studio_db.list_chat_settings()["pinnedChatIds"] == ["keep"]


def test_ordered_chat_and_comparison_pins_round_trip_through_sqlite(monkeypatch):
    # A comparison's sidebar ID is its pair ID, not either underlying thread ID.
    pins = ["comparison-pair", "ordinary-thread", "another-thread"]
    _seed("left", "right", pair_id = "comparison-pair")
    _seed("ordinary-thread", "another-thread")
    chat_history.put_settings({"pinnedChatIds": pins}, current_subject = "test-user")
    chat_history.put_settings({"autoTitle": False}, current_subject = "test-user")

    conn = studio_db.get_connection()
    try:
        row = conn.execute(
            "SELECT value_json FROM chat_settings WHERE key = ?", ("pinnedChatIds",)
        ).fetchone()
        assert json.loads(row["value_json"]) == pins
    finally:
        conn.close()

    # Reinitialize storage and read through a separate request, as another browser does.
    studio_db.reset_schema_state_for_tests()
    saved = chat_history.get_settings(current_subject = "another-client").settings
    assert saved == {"autoTitle": False, "pinnedChatIds": pins}


def test_legacy_migration_only_claims_an_absent_pin_setting():
    _seed("legacy-pin", "different-browser-pin")
    chat_history.put_settings({"autoTitle": False}, current_subject = "test-user")
    assert "pinnedChatIds" not in chat_history.get_settings(current_subject = "test-user").settings

    migrated = _save_if_current({}, ["legacy-pin"], expected_absent = True)
    assert migrated.applied is True
    assert migrated.settings == {"autoTitle": False, "pinnedChatIds": ["legacy-pin"]}

    stale_migration = _save_if_current({}, ["different-browser-pin"], expected_absent = True)
    assert stale_migration.applied is False
    assert stale_migration.settings["pinnedChatIds"] == ["legacy-pin"]


def test_explicit_empty_pins_prevent_legacy_browser_resurrection():
    _seed("legacy-pin")
    migrated = _save_if_current({}, ["legacy-pin"], expected_absent = True)
    cleared = _save_if_current({"pinnedChatIds": migrated.settings["pinnedChatIds"]}, [])
    assert cleared.applied is True

    stale_migration = _save_if_current({}, ["legacy-pin"], expected_absent = True)
    assert stale_migration.applied is False
    assert stale_migration.settings["pinnedChatIds"] == []
    assert chat_history.get_settings(current_subject = "test-user").settings["pinnedChatIds"] == []


@pytest.mark.parametrize("field", ["expected", "patch"])
def test_compare_and_set_rejects_null_pins(field):
    _seed("keep")
    chat_history.put_settings({"pinnedChatIds": ["keep"]}, current_subject = "test-user")
    payload = {"expected": {"pinnedChatIds": ["keep"]}, "patch": {"pinnedChatIds": []}}
    payload[field] = {"pinnedChatIds": None}
    with pytest.raises(HTTPException) as error:
        chat_history.compare_and_set_settings(payload, current_subject = "test-user")
    assert error.value.status_code == 400
    assert studio_db.list_chat_settings()["pinnedChatIds"] == ["keep"]


def test_stale_pin_write_rebases_without_resurrecting_an_unpinned_chat():
    _seed("a", "b", "c")
    chat_history.put_settings({"pinnedChatIds": ["a", "b"]}, current_subject = "test-user")
    old_snapshot = {"pinnedChatIds": ["a", "b"]}
    assert _save_if_current(old_snapshot, ["b"]).applied is True

    # Another browser pins c using the state it read before a was unpinned.
    stale = _save_if_current(old_snapshot, ["c", "a", "b"])
    assert stale.applied is False
    assert stale.settings["pinnedChatIds"] == ["b"]

    current = stale.settings["pinnedChatIds"]
    rebased = _save_if_current({"pinnedChatIds": current}, ["c", *current])
    assert rebased.applied is True
    assert studio_db.list_chat_settings()["pinnedChatIds"] == ["c", "b"]


def test_concurrent_pin_writers_conflict_then_rebase_without_lost_pins():
    _seed("a", "b")
    chat_history.put_settings({"pinnedChatIds": []}, current_subject = "test-user")
    barrier = threading.Barrier(2)

    def pin_from_empty_snapshot(chat_id):
        barrier.wait(timeout = 5)
        return chat_id, _save_if_current({"pinnedChatIds": []}, [chat_id])

    with ThreadPoolExecutor(max_workers = 2) as pool:
        results = list(pool.map(pin_from_empty_snapshot, ["a", "b"]))

    assert sum(result.applied for _, result in results) == 1
    winner = next(chat_id for chat_id, result in results if result.applied)
    loser, rejected = next((chat_id, result) for chat_id, result in results if not result.applied)
    assert rejected.settings["pinnedChatIds"] == [winner]

    current = rejected.settings["pinnedChatIds"]
    rebased = _save_if_current({"pinnedChatIds": current}, [loser, *current])
    assert rebased.applied is True
    assert studio_db.list_chat_settings()["pinnedChatIds"] == [loser, winner]


def test_pin_read_and_empty_migration_leave_absent_state_absent():
    assert _pins() == {}
    assert _migrate([]) == {}
    assert _migrate(["not-yet-imported"]) == {}
    assert "pinnedChatIds" not in studio_db.list_chat_settings()
    _seed("not-yet-imported")
    assert _migrate(["not-yet-imported"]) == {"pinnedIds": ["not-yet-imported"]}


def test_distinct_browser_migrations_merge_without_reordering_existing_pins():
    _seed("a", "b", "c")
    assert _migrate(["a", "b"]) == {"pinnedIds": ["a", "b"]}
    assert _migrate(["c", "b"]) == {"pinnedIds": ["a", "b", "c"]}
    assert _migrate(["b", "a"]) == {"pinnedIds": ["a", "b", "c"]}


@pytest.mark.parametrize("method", ["put", "cas"])
def test_migration_cannot_resurrect_unpins_and_explicit_repin_is_allowed(method, monkeypatch):
    _seed("a", "b", "c")
    _migrate(["a", "b"])
    if method == "put":
        chat_history.put_settings({"pinnedChatIds": ["a"]}, current_subject = "test-user")
    else:
        assert _save_if_current({"pinnedChatIds": ["a", "b"]}, ["a"]).applied
    studio_db.reset_schema_state_for_tests()
    assert _migrate(["b", "c"]) == {"pinnedIds": ["a", "c"]}
    assert _save_if_current({"pinnedChatIds": ["a", "c"]}, ["b", "a", "c"]).applied
    assert _pins() == {"pinnedIds": ["b", "a", "c"]}
    conn = studio_db.get_connection()
    try:
        assert conn.execute("SELECT id FROM chat_pin_tombstones WHERE id = 'b'").fetchone() is None
    finally:
        conn.close()


def test_noop_and_migration_preserve_unpin_tombstones():
    _seed("a", "b")
    _migrate(["a", "b"])
    assert _save_if_current({"pinnedChatIds": ["a", "b"]}, ["a"]).applied
    assert _save_if_current({"pinnedChatIds": ["a"]}, ["a"]).applied
    assert _migrate(["b"]) == {"pinnedIds": ["a"]}


def test_deleting_threads_prunes_single_and_comparison_pins_atomically():
    _seed("single")
    _seed("left", "right", pair_id = "pair")
    _seed("archived", archived = True)
    assert _migrate(["single", "pair", "archived", "left", "missing"]) == {
        "pinnedIds": ["single", "pair", "archived"]
    }
    studio_db.delete_chat_threads(["single", "left"])
    # Read raw settings to prove deletion itself pruned, without relying on GET repair.
    assert studio_db.list_chat_settings()["pinnedChatIds"] == ["pair", "archived"]
    studio_db.delete_chat_threads(["right"])
    assert studio_db.list_chat_settings()["pinnedChatIds"] == ["archived"]
    assert _migrate(["pair", "single"]) == {"pinnedIds": ["archived"]}


def test_project_deletion_and_clear_all_prune_pin_state():
    studio_db.upsert_chat_project(
        {"id": "project", "name": "Project", "createdAt": 1, "updatedAt": 1}
    )
    _seed("project-chat", project_id = "project")
    _seed("other-chat")
    _migrate(["project-chat", "other-chat"])
    studio_db.delete_chat_project("project")
    assert studio_db.list_chat_settings()["pinnedChatIds"] == ["other-chat"]
    studio_db.clear_chat_history()
    assert studio_db.list_chat_settings()["pinnedChatIds"] == []


def test_orphan_pins_do_not_exhaust_capacity_and_stale_cas_can_rebase():
    _seed("new-chat")
    orphaned = [f"deleted-{i}" for i in range(studio_db.MAX_PINNED_CHAT_IDS)]
    studio_db.upsert_chat_settings({"pinnedChatIds": orphaned})
    # Expected state can still name old IDs: normalization returns a conflict, not a lost update.
    stale = _save_if_current({"pinnedChatIds": orphaned}, ["new-chat"])
    assert not stale.applied
    assert stale.settings["pinnedChatIds"] == []
    assert _save_if_current({"pinnedChatIds": []}, ["new-chat"]).applied
    assert _pins() == {"pinnedIds": ["new-chat"]}


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("{bad-json", []),
        ("null", []),
        ('"wrong-shape"', []),
        ('{"a":true}', []),
        ('["a",null,true,"a","missing",""]', ["a"]),
        (json.dumps(["a", chr(0xD800)]), ["a"]),
    ],
)
def test_targeted_get_repairs_corrupt_pin_state_and_preserves_unrelated_settings(raw, expected):
    _seed("a", "b")
    studio_db.upsert_chat_settings({"autoTitle": False, "pinnedChatIds": []})
    conn = studio_db.get_connection()
    try:
        conn.execute("UPDATE chat_settings SET value_json = ? WHERE key = 'pinnedChatIds'", (raw,))
        conn.commit()
    finally:
        conn.close()
    assert _pins() == {"pinnedIds": expected}
    assert studio_db.list_chat_settings() == {"autoTitle": False, "pinnedChatIds": expected}
    assert _save_if_current({"pinnedChatIds": expected}, ["b", *expected]).applied
    conn = studio_db.get_connection()
    try:
        assert (
            conn.execute(
                "SELECT value_json FROM chat_settings_quarantine WHERE key = 'pinnedChatIds'"
            ).fetchone()[0]
            == raw
        )
    finally:
        conn.close()


def test_healthy_targeted_get_neither_writes_nor_loads_unrelated_settings(monkeypatch):
    _seed("a")
    _migrate(["a"])
    studio_db.upsert_chat_settings({"customPresets": ["x" * 100_000]})
    statements = []
    original = studio_db.get_connection

    def traced_connection(*args, **kwargs):
        conn = original(*args, **kwargs)
        conn.set_trace_callback(statements.append)
        return conn

    monkeypatch.setattr(studio_db, "get_connection", traced_connection)
    assert _pins() == {"pinnedIds": ["a"]}
    assert all(sql.lstrip().upper().startswith("SELECT") for sql in statements)
    settings_reads = [sql for sql in statements if "FROM chat_settings" in sql]
    assert len(settings_reads) == 1 and "WHERE key = 'pinnedChatIds'" in settings_reads[0]


def test_operation_replay_acknowledges_without_reversing_a_later_unpin(monkeypatch):
    _seed("a")
    assert _save_if_current({}, ["a"], expected_absent = True, operation_id = "pin-1").applied
    assert _save_if_current({"pinnedChatIds": ["a"]}, [], operation_id = "unpin-2").applied
    studio_db.reset_schema_state_for_tests()
    replay = _save_if_current({}, ["a"], expected_absent = True, operation_id = "pin-1")
    assert replay.applied
    assert replay.settings["pinnedChatIds"] == []
    assert _migrate(["a"]) == {"pinnedIds": []}
    assert set(studio_db.list_chat_settings()) == {"pinnedChatIds"}


def test_conflicting_operation_is_not_acknowledged_until_rebased():
    _seed("a", "b")
    _migrate(["a"])
    stale = _save_if_current({"pinnedChatIds": []}, ["b"], operation_id = "pin-b")
    assert not stale.applied
    saved = _save_if_current({"pinnedChatIds": ["a"]}, ["b", "a"], operation_id = "pin-b")
    assert saved.applied and saved.settings["pinnedChatIds"] == ["b", "a"]


@pytest.mark.parametrize(
    "patch", [{}, {"autoTitle": False}, {"pinnedChatIds": [], "autoTitle": False}]
)
def test_operation_ids_cannot_acknowledge_unrelated_settings(patch):
    with pytest.raises(HTTPException) as error:
        chat_history.compare_and_set_settings(
            {"expected": {}, "patch": patch, "operationId": "op"}, current_subject = "test-user"
        )
    assert error.value.status_code == 400


def test_concurrent_legacy_migration_and_unpin_never_resurrects_the_unpin():
    _seed("a", "b")
    _migrate(["a"])
    barrier = threading.Barrier(2)

    def migrate():
        barrier.wait(timeout = 5)
        return _migrate(["a", "b"])

    def unpin():
        barrier.wait(timeout = 5)
        snapshot = ["a"]
        for _ in range(3):
            reply = _save_if_current(
                {"pinnedChatIds": snapshot}, [id for id in snapshot if id != "a"]
            )
            if reply.applied:
                return
            snapshot = reply.settings["pinnedChatIds"]
        raise AssertionError("Unpin could not rebase")

    with ThreadPoolExecutor(max_workers = 2) as pool:
        a, b = pool.submit(migrate), pool.submit(unpin)
        a.result()
        b.result()
    assert _pins() == {"pinnedIds": ["b"]}
    assert _migrate(["a"]) == {"pinnedIds": ["b"]}


def test_pin_endpoints_serialize_absence_and_never_expose_other_settings():
    app = FastAPI()
    app.include_router(chat_history.router, prefix = "/api/chat")
    app.dependency_overrides[chat_history.get_current_subject] = lambda: "test-user"
    _seed("a")
    studio_db.upsert_chat_settings({"autoTitle": False})
    with TestClient(app) as client:
        assert client.get("/api/chat/pins").json() == {}
        assert client.post("/api/chat/pins/migrate", json = {"ids": ["a"]}).json() == {
            "pinnedIds": ["a"]
        }
        assert client.get("/api/chat/pins").json() == {"pinnedIds": ["a"]}
        assert client.post("/api/chat/pins/migrate", json = {"ids": None}).status_code == 422


def test_concurrent_browser_migrations_preserve_both_pin_lists():
    _seed("a", "b")
    barrier = threading.Barrier(2)

    def migrate(id):
        barrier.wait(timeout = 5)
        return _migrate([id])

    with ThreadPoolExecutor(max_workers = 2) as pool:
        list(pool.map(migrate, ["a", "b"]))
    assert set(_pins()["pinnedIds"]) == {"a", "b"}


def test_failed_pin_write_rolls_back_tombstone_and_operation_ack():
    _seed("a")
    _migrate(["a"])
    conn = studio_db.get_connection()
    try:
        conn.execute("""CREATE TRIGGER fail_pin_write BEFORE UPDATE ON chat_settings
            WHEN NEW.key = 'pinnedChatIds'
            BEGIN SELECT RAISE(ABORT, 'test write failure'); END""")
        conn.commit()
    finally:
        conn.close()
    with pytest.raises(sqlite3.IntegrityError):
        _save_if_current(
            {"pinnedChatIds": ["a"]},
            [],
            operation_id = "failed-unpin",
            unpinned_ids = ["not-yet-imported"],
        )
    assert _pins() == {"pinnedIds": ["a"]}
    conn = studio_db.get_connection()
    try:
        assert conn.execute("SELECT id FROM chat_pin_tombstones").fetchall() == []
        assert conn.execute("SELECT id FROM chat_pin_operations").fetchall() == []
    finally:
        conn.close()


def test_bad_unicode_pin_does_not_prevent_deleting_a_chat():
    _seed("a")
    studio_db.upsert_chat_settings({"pinnedChatIds": ["a", chr(0xD800)]})
    studio_db.delete_chat_threads(["a"])
    assert studio_db.get_chat_thread("a") is None
    assert studio_db.list_chat_settings()["pinnedChatIds"] == []


def test_explicit_unpin_before_thread_import_blocks_legacy_resurrection():
    assert _migrate(["late-thread"]) == {}
    unpinned = _save_if_current(
        {},
        [],
        expected_absent = True,
        operation_id = "unpin-before-import",
        unpinned_ids = ["late-thread"],
    )
    assert unpinned.applied and unpinned.settings["pinnedChatIds"] == []
    _seed("late-thread")
    assert _migrate(["late-thread"]) == {"pinnedIds": []}

    # A deliberate new pin can undo the unpin. Replaying the old acknowledgement
    # cannot add its tombstone again or undo the later explicit choice.
    assert _save_if_current(
        {"pinnedChatIds": []}, ["late-thread"], operation_id = "repin-later"
    ).applied
    replay = _save_if_current(
        {},
        [],
        expected_absent = True,
        operation_id = "unpin-before-import",
        unpinned_ids = ["late-thread"],
    )
    assert replay.applied and replay.settings["pinnedChatIds"] == ["late-thread"]
    conn = studio_db.get_connection()
    try:
        assert (
            conn.execute("SELECT id FROM chat_pin_tombstones WHERE id = 'late-thread'").fetchone()
            is None
        )
    finally:
        conn.close()


def test_conflicting_explicit_unpin_does_not_leave_a_tombstone():
    _seed("existing")
    _migrate(["existing"])
    rejected = _save_if_current({"pinnedChatIds": []}, [], unpinned_ids = ["late-thread"])
    assert not rejected.applied
    _seed("late-thread")
    assert _migrate(["late-thread"]) == {"pinnedIds": ["existing", "late-thread"]}


@pytest.mark.parametrize(
    "patch,unpinned",
    [
        ({"pinnedChatIds": ["a"]}, ["a"]),
        ({"autoTitle": False}, ["a"]),
        ({"autoTitle": False}, []),
        ({"pinnedChatIds": []}, None),
        ({"pinnedChatIds": []}, "a"),
        ({"pinnedChatIds": []}, ["a", "a"]),
        ({"pinnedChatIds": []}, [chr(0xD800)]),
    ],
)
def test_invalid_or_contradictory_explicit_unpins_are_rejected(patch, unpinned):
    with pytest.raises(HTTPException) as error:
        chat_history.compare_and_set_settings(
            {"expected": {}, "patch": patch, "unpinnedChatIds": unpinned},
            current_subject = "test-user",
        )
    assert error.value.status_code == 400
    assert studio_db.list_chat_settings() == {}


@pytest.mark.parametrize("operation_id", [chr(0xD800), "op-" + chr(0xDFFF)])
def test_invalid_unicode_operation_id_is_rejected_before_sqlite(operation_id):
    with pytest.raises(HTTPException) as error:
        _save_if_current({}, [], expected_absent = True, operation_id = operation_id)
    assert error.value.status_code == 400
    assert studio_db.list_chat_settings() == {}


def test_a_delayed_pin_an_unpin_cancelled_is_acknowledged_without_applying():
    _seed("chat")
    _save_if_current({}, [], expected_absent = True)
    unpin = _save_if_current(
        {"pinnedChatIds": []},
        [],
        operation_id = "unpin-op",
        unpinned_ids = ["chat"],
        cancelled_operation_ids = ["pin-op"],
    )
    assert unpin.applied
    delayed = _save_if_current({"pinnedChatIds": []}, ["chat"], operation_id = "pin-op")
    assert delayed.applied
    assert _pins() == {"pinnedIds": []}


def test_cancelled_operation_ids_require_a_pin_only_patch():
    with pytest.raises(HTTPException) as error:
        chat_history.compare_and_set_settings(
            {"expected": {}, "patch": {"autoTitle": False}, "cancelledOperationIds": ["op"]},
            current_subject = "test-user",
        )
    assert error.value.status_code == 400
