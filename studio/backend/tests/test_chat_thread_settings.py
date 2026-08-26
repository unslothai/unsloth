# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The per-thread settings snapshot that makes a chat keep its own modes.

Covers the contract PATCH /api/chat/threads/{id} accepts and the storage rules the
snapshot depends on: writers that rebuild a thread record must not clear it, the thread
listing must not carry it, and a fork must inherit it.
"""

import sqlite3
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from routes.chat_history import (  # noqa: E402
    ChatThread,
    ChatThreadPatch,
    ChatThreadSettings,
    _settings_write_from_patch,
    thread_from_row,
)
from storage import studio_db  # noqa: E402
from utils.paths import studio_db_path  # noqa: E402

SETTINGS = {
    "toolsEnabled": True,
    "codeToolsEnabled": False,
    "deepResearchEnabled": False,
    "permissionMode": "auto",
    "ragEnabled": True,
    "ragSource": {"type": "kb", "kbId": "notes"},
    "ragMode": "dense",
    "ragTopK": 12,
    "ragAutoInject": "on",
    "ragAutoInjectMinScore": 0.42,
    "reasoningEffort": "high",
}


def _reset_studio_db(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    monkeypatch.setenv("UNSLOTH_STUDIO_PROJECTS_HOME", str(tmp_path / "Projects"))
    monkeypatch.setattr(studio_db, "_schema_ready", False)


def _thread(thread_id: str = "thread-1", **extra) -> dict:
    return {
        "id": thread_id,
        "title": "Test Chat",
        "modelType": "base",
        "modelId": "test-model",
        "pairId": None,
        "archived": False,
        "createdAt": 1_700_000_000_000,
        **extra,
    }


def test_thread_settings_round_trip():
    settings = ChatThreadSettings.model_validate(SETTINGS)
    assert settings.model_dump(exclude_unset = True) == SETTINGS


@pytest.mark.parametrize(
    "field, value",
    [
        # full access disables the sandbox, so it stays session-only per thread too.
        ("permissionMode", "full"),
        ("ragTopK", 51),
        ("ragAutoInjectMinScore", 1.5),
        ("ragMode", "vector"),
        # global-only settings must not reach a thread and silently become per-chat.
        ("gpuMemoryMode", "manual"),
        ("showCanvasMenuItem", True),
    ],
)
def test_thread_settings_rejects_out_of_contract(field, value):
    with pytest.raises(ValidationError):
        ChatThreadSettings.model_validate({field: value})


def test_thread_settings_survive_a_record_rewrite(tmp_path, monkeypatch):
    _reset_studio_db(tmp_path, monkeypatch)
    studio_db.upsert_chat_thread(_thread(settings = SETTINGS))

    # the title autosave and every other writer rebuild the record without the snapshot.
    studio_db.upsert_chat_thread(_thread(title = "Renamed"))

    stored = studio_db.get_chat_thread("thread-1")
    assert stored["title"] == "Renamed"
    assert stored["settings"] == SETTINGS


def test_thread_settings_patch_replaces_the_snapshot(tmp_path, monkeypatch):
    _reset_studio_db(tmp_path, monkeypatch)
    studio_db.upsert_chat_thread(_thread(settings = SETTINGS))

    # ragSource is a discriminated union, so the write replaces: no kb id survives the switch.
    replacement = {"toolsEnabled": False, "ragSource": {"type": "thread"}}
    updated = studio_db.update_chat_thread("thread-1", {"settings": replacement})

    assert updated["settings"] == replacement
    assert studio_db.get_chat_thread("thread-1")["settings"] == replacement


def test_thread_listing_leaves_out_the_snapshot(tmp_path, monkeypatch):
    _reset_studio_db(tmp_path, monkeypatch)
    studio_db.upsert_chat_thread(_thread(settings = SETTINGS))

    listed = studio_db.list_chat_threads()

    assert [row["id"] for row in listed] == ["thread-1"]
    assert "settings" not in listed[0]


def test_thread_without_a_snapshot_reads_back_null(tmp_path, monkeypatch):
    _reset_studio_db(tmp_path, monkeypatch)
    studio_db.upsert_chat_thread(_thread())

    assert studio_db.get_chat_thread("thread-1")["settings"] is None


def test_fork_inherits_the_snapshot(tmp_path, monkeypatch):
    _reset_studio_db(tmp_path, monkeypatch)
    studio_db.upsert_chat_thread(_thread(settings = SETTINGS))
    studio_db.upsert_chat_message(
        {
            "id": "message-1",
            "threadId": "thread-1",
            "parentId": None,
            "role": "user",
            "content": "hello",
            "createdAt": 1_700_000_000_001,
        }
    )

    forked = studio_db.fork_chat_thread(
        source_thread_id = "thread-1",
        branch_message_id = "message-1",
        new_thread_id = "thread-2",
        new_title = "Fork",
        created_at = 1_700_000_000_002,
        id_factory = lambda: "message-2",
    )

    assert forked is not None
    assert studio_db.get_chat_thread("thread-2")["settings"] == SETTINGS


def test_settings_column_is_added_to_an_existing_database(tmp_path, monkeypatch):
    _reset_studio_db(tmp_path, monkeypatch)
    # a database created before this change has no settings_json, and the CREATE TABLE in
    # _ensure_schema never runs for it, so only the ALTER keeps thread writes working.
    db_path = Path(studio_db_path())
    db_path.parent.mkdir(parents = True, exist_ok = True)
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute(
            """
            CREATE TABLE chat_threads (
                id TEXT NOT NULL PRIMARY KEY,
                title TEXT NOT NULL,
                model_type TEXT NOT NULL,
                model_id TEXT,
                pair_id TEXT,
                archived INTEGER NOT NULL DEFAULT 0,
                created_at INTEGER NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE chat_messages (
                id TEXT NOT NULL PRIMARY KEY,
                thread_id TEXT NOT NULL,
                parent_id TEXT,
                role TEXT NOT NULL,
                content_json TEXT NOT NULL,
                attachments_json TEXT,
                metadata_json TEXT,
                created_at INTEGER NOT NULL
            )
            """
        )
        conn.execute(
            "INSERT INTO chat_threads (id, title, model_type, created_at) VALUES (?, ?, ?, ?)",
            ("thread-1", "Old", "base", 1_700_000_000_000),
        )
        conn.commit()
    finally:
        conn.close()

    assert studio_db.get_chat_thread("thread-1")["settings"] is None
    studio_db.upsert_chat_thread(_thread(settings = SETTINGS))
    assert studio_db.get_chat_thread("thread-1")["settings"] == SETTINGS


# A snapshot on disk outlives the build that wrote it. A newer Unsloth adding a
# setting, widening an enum or raising a bound writes a blob this build has never
# seen, and it reaches the response model rather than the request one, so refusing
# it 500s the chat on open and takes the whole history export with it. The wire
# contract stays strict; only the read is forgiving.
@pytest.mark.parametrize(
    "stored, expected",
    [
        ({"toolsEnabled": True, "voiceModeEnabled": True}, {"toolsEnabled": True}),
        ({"reasoningEffort": "ultra", "toolsEnabled": True}, {"toolsEnabled": True}),
        ({"ragTopK": 999, "toolsEnabled": True}, {"toolsEnabled": True}),
        # several at once, which is what a version gap actually looks like
        (
            {"ragTopK": 999, "reasoningEffort": "ultra", "futureThing": 1, "toolsEnabled": True},
            {"toolsEnabled": True},
        ),
        ({"ragSource": {"type": "web", "url": "x"}, "toolsEnabled": True}, {"toolsEnabled": True}),
        # nothing salvageable, and non-objects
        ({"quantumMode": True}, {}),
        ("hello", None),
        ([1, 2, 3], None),
    ],
)
def test_a_snapshot_from_a_newer_build_still_reads(stored, expected):
    thread = thread_from_row(
        {"id": "t", "title": "T", "modelType": "base", "createdAt": 1, "settings": stored},
    )
    if expected is None:
        assert thread.settings is None
    else:
        assert thread.settings is not None
        assert thread.settings.model_dump(exclude_none = True) == expected


def test_the_wire_contract_stays_strict():
    # Only rows off disk are forgiven. A client may not invent a setting on either
    # the patch or the full-record POST, which shares the ChatThread model.
    with pytest.raises(ValidationError):
        ChatThreadPatch(settings = {"toolsEnabled": True, "voiceModeEnabled": True})
    with pytest.raises(ValidationError):
        ChatThreadPatch(settings = {"ragTopK": 999})
    with pytest.raises(ValidationError):
        ChatThread(
            id = "t",
            title = "T",
            modelType = "base",
            createdAt = 1,
            settings = {"toolsEnabled": True, "voiceModeEnabled": True},
        )
    with pytest.raises(ValidationError):
        ChatThread(
            id = "t",
            title = "T",
            modelType = "base",
            createdAt = 1,
            settings = {"ragTopK": 999},
        )


def test_an_unreadable_key_does_not_disturb_the_stored_row(tmp_path, monkeypatch):
    # the row keeps what it had, so upgrading again gets the setting back.
    _reset_studio_db(tmp_path, monkeypatch)
    studio_db.upsert_chat_thread(_thread())
    studio_db.update_chat_thread("thread-1", {"settings": {"toolsEnabled": True}})
    conn = sqlite3.connect(studio_db_path())
    try:
        conn.execute(
            "UPDATE chat_threads SET settings_json = ? WHERE id = ?",
            ('{"toolsEnabled": true, "voiceModeEnabled": true}', "thread-1"),
        )
        conn.commit()
    finally:
        conn.close()

    row = studio_db.get_chat_thread("thread-1")
    assert thread_from_row(row).settings.toolsEnabled is True

    studio_db.update_chat_thread("thread-1", {"title": "renamed"})
    conn = sqlite3.connect(studio_db_path())
    try:
        raw = conn.execute(
            "SELECT settings_json FROM chat_threads WHERE id = ?", ("thread-1",)
        ).fetchone()[0]
    finally:
        conn.close()
    assert "voiceModeEnabled" in raw


def _raw_settings(thread_id: str = "thread-1") -> str:
    conn = sqlite3.connect(studio_db_path())
    try:
        return conn.execute(
            "SELECT settings_json FROM chat_threads WHERE id = ?", (thread_id,)
        ).fetchone()[0]
    finally:
        conn.close()


def _store_raw(payload: str, thread_id: str = "thread-1") -> None:
    conn = sqlite3.connect(studio_db_path())
    try:
        conn.execute("UPDATE chat_threads SET settings_json = ? WHERE id = ?", (payload, thread_id))
        conn.commit()
    finally:
        conn.close()


def test_a_downgraded_client_cannot_delete_what_it_could_not_read(tmp_path, monkeypatch):
    # The whole point of the lenient read is that upgrading gets the setting back, which
    # only holds if writing in the meantime leaves the unreadable part alone.
    _reset_studio_db(tmp_path, monkeypatch)
    studio_db.upsert_chat_thread(_thread())
    _store_raw('{"toolsEnabled": true, "voiceModeEnabled": true, "ragTopK": 999}')

    # what this build serves the client: neither the unknown key nor the out-of-range one
    served = thread_from_row(studio_db.get_chat_thread("thread-1")).settings
    assert served.toolsEnabled is True
    assert served.ragTopK is None

    # the client writes back everything it knows about
    patch = {"settings": {"toolsEnabled": False}}
    settings_write = _settings_write_from_patch(patch)
    studio_db.update_chat_thread("thread-1", patch, settings_write = settings_write)

    raw = _raw_settings()
    assert "voiceModeEnabled" in raw
    assert "999" in raw
    assert thread_from_row(studio_db.get_chat_thread("thread-1")).settings.toolsEnabled is False


def test_a_merge_touches_only_the_fields_it_names(tmp_path, monkeypatch):
    # The unload path knows one pill changed and nothing about the rest of the row.
    _reset_studio_db(tmp_path, monkeypatch)
    studio_db.upsert_chat_thread(_thread())
    studio_db.update_chat_thread(
        "thread-1", {"settings": {"toolsEnabled": True, "permissionMode": "ask"}}
    )

    payload = ChatThreadPatch(settingsPatch = {"codeToolsEnabled": True})
    patch = payload.model_dump(exclude_unset = True)
    settings_write = _settings_write_from_patch(patch)
    studio_db.update_chat_thread("thread-1", patch, settings_write = settings_write)

    got = thread_from_row(studio_db.get_chat_thread("thread-1")).settings
    assert got.codeToolsEnabled is True
    assert got.toolsEnabled is True
    assert got.permissionMode == "ask"


def test_a_merge_also_spares_an_unreadable_key(tmp_path, monkeypatch):
    _reset_studio_db(tmp_path, monkeypatch)
    studio_db.upsert_chat_thread(_thread())
    _store_raw('{"toolsEnabled": true, "voiceModeEnabled": true}')

    payload = ChatThreadPatch(settingsPatch = {"toolsEnabled": False})
    patch = payload.model_dump(exclude_unset = True)
    settings_write = _settings_write_from_patch(patch)
    studio_db.update_chat_thread("thread-1", patch, settings_write = settings_write)

    assert "voiceModeEnabled" in _raw_settings()
    assert thread_from_row(studio_db.get_chat_thread("thread-1")).settings.toolsEnabled is False


def test_clearing_still_clears_the_whole_column(tmp_path, monkeypatch):
    # An explicit null is the one instruction that means all of it, unreadable part included.
    _reset_studio_db(tmp_path, monkeypatch)
    studio_db.upsert_chat_thread(_thread())
    _store_raw('{"toolsEnabled": true, "voiceModeEnabled": true}')

    patch = {"settings": None}
    settings_write = _settings_write_from_patch(patch)
    studio_db.update_chat_thread("thread-1", patch, settings_write = settings_write)

    assert thread_from_row(studio_db.get_chat_thread("thread-1")).settings is None
    assert _raw_settings() in (None, "", "null")


def test_a_merge_of_nothing_leaves_the_row_alone(tmp_path, monkeypatch):
    _reset_studio_db(tmp_path, monkeypatch)
    studio_db.upsert_chat_thread(_thread())
    studio_db.update_chat_thread("thread-1", {"settings": {"toolsEnabled": True}})

    patch = {"title": "renamed", "settingsPatch": None}
    settings_write = _settings_write_from_patch(patch)
    studio_db.update_chat_thread("thread-1", patch, settings_write = settings_write)
    assert "settings" not in patch
    studio_db.update_chat_thread("thread-1", patch)

    got = thread_from_row(studio_db.get_chat_thread("thread-1"))
    assert got.title == "renamed"
    assert got.settings.toolsEnabled is True


def test_a_merge_is_still_held_to_the_contract():
    with pytest.raises(ValidationError):
        ChatThreadPatch(settingsPatch = {"voiceModeEnabled": True})
    with pytest.raises(ValidationError):
        ChatThreadPatch(settingsPatch = {"ragTopK": 999})


def test_an_older_write_cannot_overtake_a_newer_one(tmp_path, monkeypatch):
    # The tab-close beacon can pass a PATCH the server has already accepted, and no
    # client-side abort reaches a handler that is already running.
    _reset_studio_db(tmp_path, monkeypatch)
    studio_db.upsert_chat_thread(_thread())

    studio_db.write_chat_thread_settings(
        "thread-1", replace = {"toolsEnabled": True}, seq = 200, writer = "tab-a"
    )
    # the straggler, carrying what the user had moved away from
    studio_db.write_chat_thread_settings(
        "thread-1", replace = {"toolsEnabled": False}, seq = 100, writer = "tab-a"
    )

    got = thread_from_row(studio_db.get_chat_thread("thread-1")).settings
    assert got.toolsEnabled is True


def test_the_same_seq_is_not_applied_twice(tmp_path, monkeypatch):
    # keepalive retries can duplicate a request; the second must be a no-op, not a revert.
    _reset_studio_db(tmp_path, monkeypatch)
    studio_db.upsert_chat_thread(_thread())

    studio_db.write_chat_thread_settings(
        "thread-1", replace = {"toolsEnabled": True}, seq = 500, writer = "tab-a"
    )
    studio_db.write_chat_thread_settings(
        "thread-1", replace = {"toolsEnabled": False}, seq = 500, writer = "tab-a"
    )

    assert thread_from_row(studio_db.get_chat_thread("thread-1")).settings.toolsEnabled is True


def test_a_write_without_a_seq_still_applies(tmp_path, monkeypatch):
    # Old clients send none, and an unordered write is still better than a dropped one.
    _reset_studio_db(tmp_path, monkeypatch)
    studio_db.upsert_chat_thread(_thread())

    studio_db.write_chat_thread_settings(
        "thread-1", replace = {"toolsEnabled": True}, seq = 900, writer = "tab-a"
    )
    studio_db.write_chat_thread_settings("thread-1", replace = {"toolsEnabled": False})

    assert thread_from_row(studio_db.get_chat_thread("thread-1")).settings.toolsEnabled is False


def test_two_merges_of_different_fields_both_survive(tmp_path, monkeypatch):
    # Two tabs, each knowing only its own field. Read-merge-write has to be one
    # transaction or the second one's stale read erases the first one's field.
    _reset_studio_db(tmp_path, monkeypatch)
    studio_db.upsert_chat_thread(_thread())

    studio_db.write_chat_thread_settings("thread-1", merge = {"toolsEnabled": True})
    studio_db.write_chat_thread_settings("thread-1", merge = {"codeToolsEnabled": True})

    got = thread_from_row(studio_db.get_chat_thread("thread-1")).settings
    assert got.toolsEnabled is True
    assert got.codeToolsEnabled is True


def test_the_watermark_column_is_added_to_an_existing_database(tmp_path, monkeypatch):
    # Same upgrade path as settings_json: an install that predates the column.
    _reset_studio_db(tmp_path, monkeypatch)
    studio_db.upsert_chat_thread(_thread())
    conn = sqlite3.connect(studio_db_path())
    try:
        conn.execute("ALTER TABLE chat_threads DROP COLUMN settings_seqs")
        conn.commit()
    finally:
        conn.close()
    monkeypatch.setattr(studio_db, "_schema_ready", False)

    studio_db.write_chat_thread_settings(
        "thread-1", replace = {"toolsEnabled": True}, seq = 1, writer = "tab-a"
    )
    assert thread_from_row(studio_db.get_chat_thread("thread-1")).settings.toolsEnabled is True


def test_two_browsers_are_never_ordered_against_each_other(tmp_path, monkeypatch):
    # The seq is a client's own counter. Comparing one machine's against another's means
    # the browser whose clock or counter is behind has every edit silently refused, while
    # the server still answers 200 and the user believes it saved.
    _reset_studio_db(tmp_path, monkeypatch)
    studio_db.upsert_chat_thread(_thread())

    studio_db.write_chat_thread_settings(
        "thread-1", replace = {"toolsEnabled": True}, seq = 9_000, writer = "laptop"
    )
    # a second machine, far behind on the same counter, still gets its edit
    studio_db.write_chat_thread_settings(
        "thread-1", replace = {"toolsEnabled": False}, seq = 12, writer = "desktop"
    )

    assert thread_from_row(studio_db.get_chat_thread("thread-1")).settings.toolsEnabled is False


def test_a_failed_precondition_does_not_commit_the_settings(tmp_path, monkeypatch):
    # PATCH allows settings and a guarded rename together; a 409 must leave neither.
    _reset_studio_db(tmp_path, monkeypatch)
    studio_db.upsert_chat_thread(_thread())
    studio_db.update_chat_thread("thread-1", {"settings": {"toolsEnabled": True}})

    with pytest.raises(studio_db.ChatThreadPreconditionFailed):
        studio_db.update_chat_thread(
            "thread-1",
            {"title": "renamed"},
            expected_title = "Not The Current Title",
            settings_write = {"replace": {"toolsEnabled": False}},
        )

    got = thread_from_row(studio_db.get_chat_thread("thread-1"))
    assert got.title == "Test Chat"
    assert got.settings.toolsEnabled is True


def test_settings_and_a_passing_guard_both_land(tmp_path, monkeypatch):
    _reset_studio_db(tmp_path, monkeypatch)
    studio_db.upsert_chat_thread(_thread())

    studio_db.update_chat_thread(
        "thread-1",
        {"title": "renamed"},
        expected_title = "Test Chat",
        settings_write = {"replace": {"toolsEnabled": True}},
    )

    got = thread_from_row(studio_db.get_chat_thread("thread-1"))
    assert got.title == "renamed"
    assert got.settings.toolsEnabled is True


def test_another_tab_writing_does_not_clear_a_writer_watermark(tmp_path, monkeypatch):
    # The race the ordering exists for: A's newer keepalive lands, B writes, then A's
    # older request finally arrives. With one writer column, B's write has replaced the
    # watermark and A's straggler is no longer compared against its own.
    _reset_studio_db(tmp_path, monkeypatch)
    studio_db.upsert_chat_thread(_thread())

    studio_db.write_chat_thread_settings(
        "thread-1", merge = {"toolsEnabled": True}, seq = 5, writer = "tab-a"
    )
    studio_db.write_chat_thread_settings(
        "thread-1", merge = {"codeToolsEnabled": True}, seq = 1, writer = "tab-b"
    )
    # tab A's straggler, older than what A already had stored
    studio_db.write_chat_thread_settings(
        "thread-1", merge = {"toolsEnabled": False}, seq = 2, writer = "tab-a"
    )

    got = thread_from_row(studio_db.get_chat_thread("thread-1")).settings
    assert got.toolsEnabled is True
    assert got.codeToolsEnabled is True


def test_watermarks_do_not_grow_without_bound(tmp_path, monkeypatch):
    _reset_studio_db(tmp_path, monkeypatch)
    studio_db.upsert_chat_thread(_thread())
    for i in range(studio_db._MAX_SETTINGS_WRITERS + 10):
        studio_db.write_chat_thread_settings(
            "thread-1", merge = {"toolsEnabled": True}, seq = i + 1, writer = f"tab-{i}"
        )

    conn = sqlite3.connect(studio_db_path())
    try:
        raw = conn.execute(
            "SELECT settings_seqs FROM chat_threads WHERE id = ?", ("thread-1",)
        ).fetchone()[0]
    finally:
        conn.close()
    import json as _json

    assert len(_json.loads(raw)) <= studio_db._MAX_SETTINGS_WRITERS


def test_the_newest_writer_is_not_the_first_evicted(tmp_path, monkeypatch):
    # Every session starts its counter at 1, so evicting by counter throws out the tab
    # that just arrived and keeps long-dead ones, leaving the active writer with no
    # watermark for its own stragglers to be refused by.
    _reset_studio_db(tmp_path, monkeypatch)
    studio_db.upsert_chat_thread(_thread())
    for i in range(studio_db._MAX_SETTINGS_WRITERS):
        studio_db.write_chat_thread_settings(
            "thread-1", merge = {"toolsEnabled": True}, seq = 500 + i, writer = f"old-{i}"
        )
    # a brand new tab, counter starting at 1
    studio_db.write_chat_thread_settings(
        "thread-1", merge = {"toolsEnabled": False}, seq = 1, writer = "fresh"
    )
    # its own straggler must still be refused
    studio_db.write_chat_thread_settings(
        "thread-1", merge = {"toolsEnabled": True}, seq = 1, writer = "fresh"
    )

    got = thread_from_row(studio_db.get_chat_thread("thread-1")).settings
    assert got.toolsEnabled is False


def test_sampling_params_are_part_of_the_snapshot():
    """The reported gap: a chat's system prompt and sampling did not travel with it."""
    settings = ChatThreadSettings.model_validate(
        {
            "temperature": 0.2,
            "topP": 0.85,
            "topK": 40,
            "minP": 0.02,
            "repetitionPenalty": 1.1,
            "presencePenalty": 0.5,
            "systemPrompt": "You are a terse reviewer.",
            "systemVariables": "name=Ada",
        }
    )
    assert settings.temperature == 0.2
    assert settings.systemPrompt == "You are a terse reviewer."
    assert settings.systemVariables == "name=Ada"


@pytest.mark.parametrize(
    "field, inside, outside",
    [
        ("temperature", 2, 2.5),
        ("temperature", 0, -0.1),
        ("topP", 1, 1.5),
        ("topK", 100, 101),
        # -1 disables top-k; the floor is below it, not at zero.
        ("topK", -1, -2),
        ("minP", 0, -1),
        ("repetitionPenalty", 1, 0.5),
        ("presencePenalty", 2, 3),
    ],
)
def test_sampling_params_take_the_slider_range_and_no_more(field, inside, outside):
    """Both halves: extra="forbid" alone would refuse the out-of-range value too."""
    assert getattr(ChatThreadSettings.model_validate({field: inside}), field) == inside
    with pytest.raises(ValidationError):
        ChatThreadSettings.model_validate({field: outside})


def test_the_disabled_top_k_value_round_trips():
    """ChatCompletionRequest allows -1 and default.yaml falls back to it, so a chat
    running with top-k off has to be able to store that."""
    assert ChatThreadSettings.model_validate({"topK": -1}).topK == -1


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_sampling_params_refuse_nan_and_infinity(bad):
    """Stored bare, these parse back in Python but are not valid JSON for anything else."""
    assert ChatThreadSettings.model_validate({"temperature": 0.5}).temperature == 0.5
    with pytest.raises(ValidationError):
        ChatThreadSettings.model_validate({"temperature": bad})


def test_a_long_system_prompt_is_stored_whole():
    """Truncating would silently change what the chat runs with."""
    prompt = "x" * 20_000
    settings = ChatThreadSettings.model_validate({"systemPrompt": prompt})
    assert settings.systemPrompt == prompt


def test_an_older_build_drops_only_the_sampling_it_cannot_read():
    """readable_thread_settings keeps the rest of a snapshot a newer build wrote."""
    from routes.chat_history import readable_thread_settings

    kept = readable_thread_settings(
        {"toolsEnabled": True, "temperature": 0.3, "somethingNewer": "?"}
    )
    assert kept == {"toolsEnabled": True, "temperature": 0.3}
