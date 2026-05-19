"""
Extra simulation tests for PR #5272 driving coverage above 99% of branches.

Adds:
 - DELETE /threads with empty ids (no-op)
 - pruneMissing via HTTP layer
 - settings deep-merge: nested dicts vs scalar overwrite
 - list_chat_threads filters (model_type, pair_id, include_archived)
 - count_chat_threads accuracy after delete
 - Repeated upsert of same thread (idempotency)
 - Concurrent thread upserts (no exceptions, last-writer-wins)
 - Settings extra-field rejection (model_config = extra="forbid")
 - Settings type validation (negative integers etc.)
 - PATCH thread can't NULL required fields
 - Empty messages list with prune_missing wipes thread cleanly
 - Auth refresh: dependency_overrides simulating an empty subject (anonymous)
 - Pydantic schema: route GET shape stable under attachments=None vs []
 - sync_chat_messages preserves prune_missing=True under empty list
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import sim_harness


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


def _thread(thread_id="t1", **kw):
    base = {
        "id": thread_id,
        "title": "T",
        "modelType": "base",
        "modelId": "m",
        "pairId": None,
        "archived": False,
        "createdAt": 1,
    }
    base.update(kw)
    return base


def _msg(message_id, thread_id="t1", **kw):
    base = {
        "id": message_id,
        "threadId": thread_id,
        "parentId": None,
        "role": "user",
        "content": [{"type": "text", "text": "x"}],
        "attachments": None,
        "metadata": None,
        "createdAt": 1,
    }
    base.update(kw)
    return base


# ==========================================================================
# Extra HTTP-layer behavior
# ==========================================================================


def test_delete_threads_empty_ids_is_noop(http_env):
    client, db, _ = http_env
    client.post("/api/chat/threads", json=_thread("a"))
    r = client.request("DELETE", "/api/chat/threads", json={"ids": []})
    assert r.status_code == 200
    assert client.get("/api/chat/count").json()["count"] == 1


def test_replace_thread_messages_prune_missing(http_env):
    client, *_ = http_env
    client.post("/api/chat/threads", json=_thread("t1"))
    # seed two
    r = client.put(
        "/api/chat/threads/t1/messages",
        json={"messages": [_msg("a", created_at=1), _msg("b", created_at=2)], "pruneMissing": True},
    )
    assert r.status_code == 200
    # now replace with just one
    r = client.put(
        "/api/chat/threads/t1/messages",
        json={"messages": [_msg("c", created_at=3)], "pruneMissing": True},
    )
    msgs = r.json()["messages"]
    assert [m["id"] for m in msgs] == ["c"]


def test_settings_deep_merge_preserves_nested_keys(http_env):
    client, *_ = http_env
    r = client.put(
        "/api/chat/settings",
        json={"inferenceParams": {"temperature": 0.5, "topP": 0.9}},
    )
    assert r.status_code == 200
    # Patch ONE nested key — others should remain
    r = client.put(
        "/api/chat/settings",
        json={"inferenceParams": {"topP": 0.95}},
    )
    settings = r.json()["settings"]
    assert settings["inferenceParams"]["temperature"] == 0.5
    assert settings["inferenceParams"]["topP"] == 0.95


def test_settings_rejects_extra_fields(http_env):
    client, *_ = http_env
    r = client.put(
        "/api/chat/settings",
        json={"unexpectedField": True},
    )
    assert r.status_code == 400


def test_settings_validation_negative_max_tool_calls(http_env):
    client, *_ = http_env
    r = client.put(
        "/api/chat/settings",
        json={"maxToolCallsPerMessage": -1},
    )
    assert r.status_code == 400


def test_patch_thread_cannot_null_required_fields(http_env):
    client, *_ = http_env
    client.post("/api/chat/threads", json=_thread("t1"))
    # Title is required and cannot be NULL'd
    r = client.patch("/api/chat/threads/t1", json={"title": None})
    assert r.status_code == 400


def test_patch_thread_archive_then_unarchive(http_env):
    client, *_ = http_env
    client.post("/api/chat/threads", json=_thread("t1"))
    r = client.patch("/api/chat/threads/t1", json={"archived": True})
    assert r.status_code == 200
    assert r.json()["archived"] is True
    r = client.patch("/api/chat/threads/t1", json={"archived": False})
    assert r.json()["archived"] is False


def test_list_threads_filters(http_env):
    client, *_ = http_env
    client.post("/api/chat/threads", json=_thread("a", modelType="base", pairId="p1"))
    client.post("/api/chat/threads", json=_thread("b", modelType="lora", pairId="p1"))
    client.post("/api/chat/threads", json=_thread("c", modelType="base", pairId="p2", archived=True))

    # by model_type
    r = client.get("/api/chat/threads?model_type=lora")
    assert {t["id"] for t in r.json()["threads"]} == {"b"}

    # by pair_id
    r = client.get("/api/chat/threads?pair_id=p1")
    assert {t["id"] for t in r.json()["threads"]} == {"a", "b"}

    # without archived
    r = client.get("/api/chat/threads?include_archived=false")
    assert {t["id"] for t in r.json()["threads"]} == {"a", "b"}


def test_count_threads_accuracy(http_env):
    client, *_ = http_env
    for i in range(7):
        client.post("/api/chat/threads", json=_thread(f"t{i}"))
    assert client.get("/api/chat/count").json()["count"] == 7
    client.request("DELETE", "/api/chat/threads", json={"ids": ["t0", "t1", "t2"]})
    assert client.get("/api/chat/count").json()["count"] == 4


# ==========================================================================
# Concurrent writes
# ==========================================================================


SUB = "test-subject"


def test_concurrent_thread_upserts_dont_throw(env):
    _, db, _ = env

    def w(i):
        for _ in range(50):
            db.upsert_chat_thread(_thread(f"t{i}", title=f"v{_}"), subject=SUB)

    threads = [threading.Thread(target=w, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert db.count_chat_threads(subject=SUB) == 8


def test_concurrent_message_upserts_no_duplicate_rows(env):
    """Same id from many writers: should produce exactly one row each."""
    _, db, _ = env
    db.upsert_chat_thread(_thread("t1"), subject=SUB)

    def w(i):
        for _ in range(50):
            db.upsert_chat_message(_msg(f"m{i}", thread_id="t1", content=f"r{_}"), subject=SUB)

    threads = [threading.Thread(target=w, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(db.list_chat_messages("t1", subject=SUB)) == 8


# ==========================================================================
# Empty edge cases
# ==========================================================================


def test_sync_empty_with_prune_wipes_thread(env):
    _, db, _ = env
    db.upsert_chat_thread(_thread("t1"), subject=SUB)
    db.sync_chat_messages("t1", [_msg("a"), _msg("b")], subject=SUB, prune_missing=True)
    out = db.sync_chat_messages("t1", [], subject=SUB, prune_missing=True)
    assert out == []


def test_sync_empty_no_prune_no_op(env):
    _, db, _ = env
    db.upsert_chat_thread(_thread("t1"), subject=SUB)
    db.sync_chat_messages("t1", [_msg("a"), _msg("b")], subject=SUB, prune_missing=True)
    out = db.sync_chat_messages("t1", [], subject=SUB)
    assert {m["id"] for m in out} == {"a", "b"}


def test_export_when_empty(http_env):
    client, *_ = http_env
    r = client.get("/api/chat/export")
    assert r.status_code == 200
    body = r.json()
    assert body["threadCount"] == 0
    assert body["threads"] == []
    assert body["messages"] == []


def test_export_ordering_stable(http_env):
    client, *_ = http_env
    for i in range(10):
        client.post("/api/chat/threads", json=_thread(f"e{i}", created_at=10 - i))
    r = client.get("/api/chat/export")
    assert r.status_code == 200
    # list_chat_threads orders by created_at DESC
    created = [t["createdAt"] for t in r.json()["threads"]]
    assert created == sorted(created, reverse=True)


# ==========================================================================
# Schema robustness — call from a fresh process simulation
# ==========================================================================


def test_schema_survives_drop_and_recreate(env):
    home, db, _ = env
    db.upsert_chat_thread(_thread("t1"), subject=SUB)
    conn = db.get_connection()
    conn.execute("DROP TABLE chat_threads")
    conn.execute("DROP TABLE chat_messages")
    conn.execute("DROP TABLE chat_settings")
    conn.commit()
    conn.close()
    db._schema_ready = False  # force re-create
    # Should re-create on next connection
    db.get_connection().close()
    # And we can write again
    db.upsert_chat_thread(_thread("t2"), subject=SUB)
    assert db.get_chat_thread("t2", subject=SUB) is not None


# ==========================================================================
# Pydantic schema parity (frontend compat)
# ==========================================================================


def test_message_with_explicit_empty_attachments_array(http_env):
    """Frontend Dexie used to allow attachments: []. PR must accept both
    None and [] without 422."""
    client, *_ = http_env
    client.post("/api/chat/threads", json=_thread("t1"))
    for atts in (None, [], [{"name": "f.png"}]):
        body = _msg("m1", thread_id="t1", attachments=atts)
        r = client.put("/api/chat/threads/t1/messages/m1", json=body)
        assert r.status_code == 200, (atts, r.text)


# ==========================================================================
# Auth bypass simulation
# ==========================================================================


def test_unauthenticated_subject_blocked_by_stub():
    """The PR uses get_current_subject as a Depends. Verify that overriding
    the dep to raise 401 blocks the request (proves auth path is wired in)."""
    from fastapi import HTTPException

    app, _, ch = sim_harness.fresh_app()

    def unauth():
        raise HTTPException(401, "unauth")

    from auth.authentication import get_current_subject

    app.dependency_overrides[get_current_subject] = unauth
    client = TestClient(app)
    r = client.get("/api/chat/threads")
    assert r.status_code == 401

# ==========================================================================
# batch endpoint: batched messages endpoint
# ==========================================================================


def test_batch_messages_returns_one_per_thread(http_env):
    client, *_ = http_env
    for i in range(3):
        client.post("/api/chat/threads", json=_thread(f"t{i}", created_at=i))
        client.put(f"/api/chat/threads/t{i}/messages/m{i}",
                   json=_msg(f"m{i}", thread_id=f"t{i}", created_at=i))
    r = client.post("/api/chat/messages:batch",
                    json={"thread_ids": ["t0", "t1", "t2"]})
    assert r.status_code == 200
    body = r.json()
    assert set(body["threads"].keys()) == {"t0", "t1", "t2"}
    for k in ("t0", "t1", "t2"):
        assert len(body["threads"][k]) == 1
        assert body["threads"][k][0]["id"] == f"m{k[-1:]}"


def test_batch_messages_unknown_id_returns_empty_list(http_env):
    """Endpoint must not 404 on unknown ids — the typical caller is
    rebuilding a UI index and partial failure is the wrong default."""
    client, *_ = http_env
    client.post("/api/chat/threads", json=_thread("known"))
    client.put("/api/chat/threads/known/messages/m1", json=_msg("m1", thread_id="known"))
    r = client.post("/api/chat/messages:batch",
                    json={"thread_ids": ["known", "ghost"]})
    assert r.status_code == 200
    body = r.json()
    assert len(body["threads"]["known"]) == 1
    assert body["threads"]["ghost"] == []


def test_batch_messages_empty_request(http_env):
    client, *_ = http_env
    r = client.post("/api/chat/messages:batch", json={"thread_ids": []})
    assert r.status_code == 200
    assert r.json()["threads"] == {}


def test_batch_messages_subject_scoped(http_env):
    """subject scoping + batch endpoint interaction: a caller cannot see another subject's
    messages via the batch endpoint."""
    app, _, ch = sim_harness.fresh_app()
    from auth.authentication import get_current_subject

    current = {"sub": "alice"}
    app.dependency_overrides[get_current_subject] = lambda: current["sub"]
    client = TestClient(app)

    client.post("/api/chat/threads", json=_thread("alice-t1"))
    client.put("/api/chat/threads/alice-t1/messages/am1",
               json=_msg("am1", thread_id="alice-t1"))

    current["sub"] = "bob"
    r = client.post("/api/chat/messages:batch",
                    json={"thread_ids": ["alice-t1"]})
    assert r.status_code == 200
    assert r.json()["threads"]["alice-t1"] == []


def test_batch_messages_chunks_over_900_ids(http_env):
    """The endpoint must handle id lists larger than SQLITE_MAX_VARIABLE_NUMBER
    (defaults to 999 on older builds; the storage layer chunks at 900)."""
    client, *_ = http_env
    n = 1200
    for i in range(n):
        client.post("/api/chat/threads", json=_thread(f"b{i}", created_at=i))
        client.put(f"/api/chat/threads/b{i}/messages/bm{i}",
                   json=_msg(f"bm{i}", thread_id=f"b{i}", created_at=i))
    r = client.post("/api/chat/messages:batch",
                    json={"thread_ids": [f"b{i}" for i in range(n)]})
    assert r.status_code == 200
    body = r.json()
    assert len(body["threads"]) == n
    for i in range(n):
        assert len(body["threads"][f"b{i}"]) == 1


def test_batch_messages_preserves_per_thread_order(http_env):
    """Within each thread the messages must be in created_at ASC order."""
    client, *_ = http_env
    client.post("/api/chat/threads", json=_thread("t"))
    for i in (3, 1, 2):  # insert out of order
        body = _msg(f"m{i}", thread_id="t")
        body["createdAt"] = i
        client.put(f"/api/chat/threads/t/messages/m{i}", json=body)
    r = client.post("/api/chat/messages:batch", json={"thread_ids": ["t"]})
    assert r.status_code == 200
    msgs = r.json()["threads"]["t"]
    assert [m["createdAt"] for m in msgs] == [1, 2, 3]


# ==========================================================================
# correctness static contracts
# ==========================================================================


def _read(rel: str) -> str:
    return (sim_harness.PR_ROOT.parent / "frontend" / "src" / rel).read_text()


def test_B1_hydrate_failure_sets_hydrated_true():
    """hydrate-as-defaults: catch block in hydratePersistedSettings must set
    settingsHydrated:true so writes resume after a transient failure."""
    src = _read("features/chat/stores/chat-runtime-store.ts")
    # Locate the catch in hydratePersistedSettings (the one near
    # `warnSettingsPersistenceFailure()` in the hydration body) and confirm
    # it sets settingsHydrated: true.
    idx = src.find("hydratePersistedSettings: async")
    assert idx >= 0
    end = src.find("\n  setModelLoading:", idx)
    body = src[idx : end if end > 0 else len(src)]
    assert "warnSettingsPersistenceFailure()" in body
    assert "set({ settingsHydrated: true })" in body


def test_B2_setparams_bumps_versions_unconditionally():
    """version bump: setParams must call getChangedInferenceParams (which bumps
    inferenceParamMutationVersions) even when not hydrated, so late
    hydration's version check doesn't clobber the user's pre-hydrate edit."""
    src = _read("features/chat/stores/chat-runtime-store.ts")
    # Locate setParams handler
    idx = src.find("setParams: (params)")
    assert idx >= 0
    end = src.find("\n  setCustomPresets:", idx)
    body = src[idx : end if end > 0 else len(src)]
    # The buggy version had `if (!state.settingsHydrated) { return { params }; }`
    # BEFORE the getChangedInferenceParams call. The fixed version no longer
    # short-circuits before computing changed params.
    assert "getChangedInferenceParams" in body
    # The early-return-skip-bump pattern must be gone
    assert "if (!state.settingsHydrated)\n        return { params };" not in body
    # Save is now conditional on settingsHydrated, version-bump is not
    assert "state.settingsHydrated && hasKeys(changedParams)" in body or (
        "if (state.settingsHydrated" in body and "saveSettingsPatch" in body
    )


def test_B5_optimistic_delete_tombstone_before_await():
    """optimistic delete: deleteChatItem must tombstone synchronously before awaiting
    the backend round-trip, with rollback on failure."""
    src = _read("features/chat/hooks/use-chat-sidebar-items.ts")
    idx = src.find("export async function deleteChatItem")
    assert idx >= 0
    body = src[idx : idx + 2000]
    assert "markChatThreadsDeleted(threadIds)" in body
    # Tombstone call must come BEFORE the await
    tombstone_at = body.find("markChatThreadsDeleted")
    await_at = body.find("await deleteStoredChatThreads")
    assert 0 <= tombstone_at < await_at, (
        "tombstone must run before the backend await"
    )
    # Rollback path must exist
    assert "removeChatThreadTombstones" in body


def test_B6_tombstones_carry_deletedAt_and_have_gc():
    """tombstone GC: tombstones store {id, deletedAt} tuples and GC after 90 days."""
    src = _read("features/chat/utils/chat-thread-tombstones.ts")
    assert "deletedAt" in src
    assert "TOMBSTONE_MAX_AGE_MS" in src
    assert "90 * 24 * 60 * 60 * 1000" in src
    assert "removeChatThreadTombstones" in src
    assert "clearAllChatThreadTombstones" in src
    # Legacy plain-string format must still load (back-compat)
    assert 'typeof item === "string"' in src


def test_B4_clearStoredChats_returns_partial_result():
    """partial-clear toast: clearStoredChats must return a result object distinguishing
    backend/legacy outcomes instead of swallowing failures."""
    src = _read("features/chat/utils/chat-history-storage.ts")
    assert "ClearStoredChatsResult" in src
    assert "result.backend = " in src and "result.legacy = " in src
    assert '"cleared" | "failed" | "skipped"' in src


# ==========================================================================
# perf static contracts
# ==========================================================================


def test_C1_sidebar_debounce_and_seq_guard():
    """sidebar debounce: useChatSidebarItems must debounce the event handler and
    discard stale responses."""
    src = _read("features/chat/hooks/use-chat-sidebar-items.ts")
    assert "SIDEBAR_REFRESH_DEBOUNCE_MS" in src
    assert "300" in src
    assert "requestSeq" in src or "request-id" in src.lower()


def test_C2_frontend_batchListChatMessages_exported():
    """batch endpoint frontend half: batchListChatMessages must exist and fall back
    to per-thread listChatMessages on 404/405."""
    src = _read("features/chat/api/chat-api.ts")
    assert "export async function batchListChatMessages" in src
    assert "/api/chat/messages:batch" in src
    assert 'response.status === 404 || response.status === 405' in src
    # Consumer: listStoredChatThreadsWithMessages must use it
    storage = _read("features/chat/utils/chat-history-storage.ts")
    assert "batchListChatMessages(threadIds)" in storage


def test_C3_settings_write_debounce_with_coalesce():
    """settings debounce: saveSettingsPatch must coalesce into pendingPatch and flush
    on a trailing-edge timer (default 400ms)."""
    src = _read("features/chat/stores/chat-runtime-store.ts")
    assert "SETTINGS_DEBOUNCE_MS" in src
    assert "pendingPatch" in src
    assert "mergePatch" in src
    assert "beforeunload" in src
    # The old "serial chain" pattern must be gone
    assert "settingsSaveQueue = settingsSaveQueue" not in src
