# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A user turn is identified by its id, never by its text (#9984).

These run the real studio_db, so they cover the behaviour rather than the source. The frontend
guards in studio/frontend/tests/chat-user-turn-identity.test.ts only read source and say so.
"""

import itertools
import random
import sys
from pathlib import Path

import pytest

from storage import studio_db

_TESTS_DIR = Path(__file__).resolve().parent
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))

from test_chat_history_storage import _reset_studio_db, _thread  # noqa: E402

THREAD = "thread-1"
BIG_DOC = "x" * 20_000


@pytest.fixture
def db(tmp_path, monkeypatch):
    _reset_studio_db(tmp_path, monkeypatch)
    studio_db.upsert_chat_thread(_thread(THREAD))
    return studio_db


def _msg(
    message_id,
    role,
    parent,
    text,
    created,
    attachments = None,
    thread_id = THREAD,
):
    record = {
        "id": message_id,
        "threadId": thread_id,
        "parentId": parent,
        "role": role,
        "content": [{"type": "text", "text": text}],
        "createdAt": created,
    }
    if attachments is not None:
        record["attachments"] = attachments
    return record


def _doc(name = "PRF_PART_II-THEORY_(v3.1).md"):
    return [{"id": f"att-{name}", "name": name, "type": "document", "text": BIG_DOC}]


def _stored(thread_id = THREAD):
    return {m["id"]: m for m in studio_db.list_chat_messages(thread_id)}


def _walk(stored, message_id):
    """Raises unless every ancestor of message_id is stored, without looping."""
    seen = set()
    current = message_id
    while current is not None and current != "":
        assert (
            current in stored
        ), f"{message_id} has ancestor {current!r}, which is not a stored row: {sorted(stored)}"
        assert current not in seen, f"parent chain from {message_id} loops at {current!r}"
        seen.add(current)
        current = stored[current]["parentId"]


def _assert_no_dangling_parents(thread_id = THREAD):
    stored = _stored(thread_id)
    for message_id in stored:
        _walk(stored, message_id)
    return stored


@pytest.mark.parametrize("parented", [False, True], ids = ["flat_thread", "branched_thread"])
def test_sending_the_same_text_twice_keeps_both_turns(db, parented):
    """Legacy threads store parent_id NULL throughout, so both turns share one bucket."""
    db.upsert_chat_message(_msg("u1", "user", None, "continue", 1000))
    db.upsert_chat_message(_msg("a1", "assistant", "u1" if parented else None, "ok", 1100))
    db.upsert_chat_message(_msg("u2", "user", "a1" if parented else None, "continue", 1200))

    assert sorted(_stored()) == ["a1", "u1", "u2"]


def test_the_same_attachment_sent_twice_keeps_both_turns(db):
    db.upsert_chat_message(_msg("u1", "user", None, "review this", 1000, _doc()))
    db.upsert_chat_message(_msg("a1", "assistant", "u1", "ok", 1100))
    db.upsert_chat_message(_msg("u2", "user", "a1", "review this", 1200, _doc()))

    assert sorted(_stored()) == ["a1", "u1", "u2"]


def test_an_edit_that_lands_on_the_same_text_keeps_both_branches(db):
    """Two user siblings under one parent is what an edit-resend legitimately builds."""
    db.upsert_chat_message(_msg("root", "assistant", None, "hi", 900))
    db.upsert_chat_message(_msg("u1", "user", "root", "same", 1000))
    db.upsert_chat_message(_msg("u2", "user", "root", "same", 1100))

    assert sorted(_stored()) == ["root", "u1", "u2"]


def test_upsert_never_rewrites_an_id(db):
    db.upsert_chat_message(_msg("u1", "user", None, "hi", 1000))
    assert db.upsert_chat_message(_msg("u2", "user", None, "hi", 2000))["id"] == "u2"


def test_a_child_saved_after_a_twin_still_has_its_parent(db):
    """A remapped id would leave this assistant pointing at a row that was never written."""
    db.upsert_chat_message(_msg("u1", "user", None, "hi", 1000))
    db.upsert_chat_message(_msg("u2", "user", None, "hi", 2000))
    db.upsert_chat_message(_msg("a2", "assistant", "u2", "answer", 2100))

    _assert_no_dangling_parents()


def test_sync_keeps_every_message_and_its_links(db):
    db.sync_chat_messages(
        THREAD,
        [
            _msg("u1", "user", None, "hi", 1000),
            _msg("a1", "assistant", "u1", "first", 1100),
            _msg("u2", "user", "a1", "hi", 2000),
            _msg("a2", "assistant", "u2", "second", 2100),
        ],
    )

    assert sorted(_assert_no_dangling_parents()) == ["a1", "a2", "u1", "u2"]


def _regenerate_sequence(regenerations = 3):
    """The rows a regenerate writes: one assistant sibling per attempt, no new user turn.

    reload calls startRun({ parentId }), so there is no user append to replay, and nothing in
    this repo drives that path end to end: chat-adapter.ts will not import under node --test.
    """
    yield _msg("u-doc", "user", None, "summarise the attached spec", 1000, _doc())
    for attempt in range(regenerations + 1):
        yield _msg(f"a-{attempt}", "assistant", "u-doc", f"attempt {attempt}", 1100 + attempt)


def test_storing_a_regenerate_sequence_keeps_one_user_turn(db):
    for record in _regenerate_sequence():
        db.upsert_chat_message(record)

    stored = _assert_no_dangling_parents()
    user_rows = [m for m in stored.values() if m["role"] == "user"]
    assert len(user_rows) == 1, f"user turn was multiplied: {[m['id'] for m in user_rows]}"
    assert len(user_rows[0]["attachments"]) == 1


def test_storing_a_regenerate_sequence_keeps_one_copy_of_the_document(db):
    for record in _regenerate_sequence(regenerations = 5):
        db.upsert_chat_message(record)

    assert len(studio_db.list_chat_attachments()) == 1


def test_syncing_a_regenerate_sequence_keeps_one_user_turn(db):
    """The whole thread is rewritten on each sync."""
    records = list(_regenerate_sequence())
    for end in range(1, len(records) + 1):
        db.sync_chat_messages(THREAD, records[:end])
        _assert_no_dangling_parents()

    assert sum(1 for m in _stored().values() if m["role"] == "user") == 1


def test_importing_a_conversation_with_repeated_turns_keeps_them_all(db):
    """chat-import.ts syncs with pruneMissing off; a transcript may repeat a prompt verbatim."""
    records = []
    previous = None
    for index in range(6):
        user_id = f"iu{index}"
        assistant_id = f"ia{index}"
        records.append(_msg(user_id, "user", previous, "continue", 1000 + index * 10))
        records.append(_msg(assistant_id, "assistant", user_id, f"part {index}", 1005 + index * 10))
        previous = assistant_id

    db.sync_chat_messages(THREAD, records, prune_missing = False)

    assert len(_assert_no_dangling_parents()) == len(records)


# delete-thread-message.ts also goes through sync_chat_messages.
def test_deleting_one_message_leaves_the_rest_of_the_tree_linked(db):
    records = [
        _msg("u1", "user", None, "hi", 1000),
        _msg("a1", "assistant", "u1", "one", 1100),
        _msg("u2", "user", "a1", "hi", 1200),
        _msg("a2", "assistant", "u2", "two", 1300),
    ]
    db.sync_chat_messages(THREAD, records)
    survivors = [records[0], records[1], {**records[3], "parentId": "a1"}]

    db.sync_chat_messages(THREAD, survivors, prune_missing = True)

    assert sorted(_assert_no_dangling_parents()) == ["a1", "a2", "u1"]


_OPERATIONS = ("send", "regenerate", "stop_regenerate", "edit_resend", "resync")


def _apply(operation, state, counter):
    """One runtime action, written straight through."""
    if operation == "send" or not state["records"]:
        user_id = f"u{next(counter)}"
        record = _msg(user_id, "user", state["head"], "continue", next(counter))
        state["records"].append(record)
        studio_db.upsert_chat_message(record)
        state["head"] = user_id
        state["last_user"] = user_id
        return
    if operation in ("regenerate", "stop_regenerate") and state["last_user"]:
        assistant_id = f"a{next(counter)}"
        record = _msg(assistant_id, "assistant", state["last_user"], "reply", next(counter))
        state["records"].append(record)
        studio_db.upsert_chat_message(record)
        # A stopped reply is still persisted, and the head only advances on a finished one.
        if operation == "regenerate":
            state["head"] = assistant_id
        return
    if operation == "edit_resend" and state["last_user"]:
        parent = _stored()[state["last_user"]]["parentId"]
        sibling_id = f"e{next(counter)}"
        record = _msg(sibling_id, "user", parent, "continue", next(counter))
        state["records"].append(record)
        studio_db.upsert_chat_message(record)
        state["head"] = sibling_id
        state["last_user"] = sibling_id
        return
    if operation == "resync":
        studio_db.sync_chat_messages(THREAD, list(state["records"]))


@pytest.mark.parametrize("seed", range(60))
def test_no_sequence_of_operations_strands_a_message(db, seed):
    """parent_id has no foreign key, so only this stops a write stranding a subtree."""
    rng = random.Random(seed)
    counter = itertools.count(1)
    state = {"records": [], "head": None, "last_user": None}

    for _ in range(25):
        _apply(rng.choice(_OPERATIONS), state, counter)
        _assert_no_dangling_parents()

    written = {record["id"] for record in state["records"]}
    assert set(_stored()) == written, "a message vanished without an explicit delete"


def test_a_legacy_flat_thread_still_accepts_new_turns(db):
    """Pre-branching Studio wrote parent_id NULL throughout, and those DBs are still opened."""
    for index in range(4):
        db.upsert_chat_message(_msg(f"legacy-u{index}", "user", None, "continue", 1000 + index))
        db.upsert_chat_message(_msg(f"legacy-a{index}", "assistant", None, "ok", 1050 + index))

    db.upsert_chat_message(_msg("new-u", "user", "legacy-a3", "continue", 2000))

    assert len(_assert_no_dangling_parents()) == 9


def test_a_mixed_legacy_and_branched_thread_keeps_both_shapes(db):
    db.upsert_chat_message(_msg("legacy-u", "user", None, "hi", 1000))
    db.upsert_chat_message(_msg("legacy-a", "assistant", None, "ok", 1100))
    db.upsert_chat_message(_msg("new-u", "user", "legacy-a", "hi", 1200))
    db.upsert_chat_message(_msg("new-a", "assistant", "new-u", "ok", 1300))

    stored = _assert_no_dangling_parents()
    assert stored["legacy-a"]["parentId"] is None
    assert stored["new-a"]["parentId"] == "new-u"


def test_an_empty_string_parent_reads_as_the_root(db):
    """Some legacy rows hold '' rather than NULL; both mean the root."""
    db.upsert_chat_message(_msg("u1", "user", "", "hi", 1000))

    _assert_no_dangling_parents()


# The #9984 thread id for id: two user rows under cOfdER0, 26.12 hours apart, each with
# replies. A fix must not merge them, since four assistant rows hang off the pair.
_REPORTED_THREAD = [
    ("4dwSP7r", "user", None, 1787854341631),
    ("Nmi02kB", "assistant", "4dwSP7r", 1787854341640),
    ("1GW3S79", "user", "Nmi02kB", 1787856918464),
    ("d7YROpZ", "assistant", "1GW3S79", 1787856918469),
    ("zLNf9Wp", "user", "d7YROpZ", 1787858909474),
    ("cOfdER0", "assistant", "zLNf9Wp", 1787858909480),
    ("oHXbD51", "user", "cOfdER0", 1787861739724),
    ("EenXxCU", "assistant", "oHXbD51", 1787861739732),
    ("MAVhZII", "assistant", "oHXbD51", 1787862065697),
    ("SaKf868", "user", "cOfdER0", 1787955784827),
    ("i59wGIe", "assistant", "SaKf868", 1787955784827),
    ("toVAdjZ", "assistant", "SaKf868", 1788232805365),
]


def _seed_reported_thread(db):
    for message_id, role, parent, created in _REPORTED_THREAD:
        attachments = _doc() if role == "user" and parent == "cOfdER0" else None
        db.upsert_chat_message(
            _msg(
                message_id,
                role,
                parent,
                "improved version of the document v3.1",
                created,
                attachments,
            )
        )


def test_the_reported_duplicate_pair_survives_a_reload(db):
    _seed_reported_thread(db)

    stored = _assert_no_dangling_parents()
    assert len(stored) == len(_REPORTED_THREAD)
    assert stored["oHXbD51"]["parentId"] == stored["SaKf868"]["parentId"] == "cOfdER0"


def test_the_reported_duplicate_pair_survives_a_whole_thread_sync(db):
    _seed_reported_thread(db)
    records = [
        _msg(
            m,
            r,
            p,
            "improved version of the document v3.1",
            c,
            _doc() if r == "user" and p == "cOfdER0" else None,
        )
        for m, r, p, c in _REPORTED_THREAD
    ]

    db.sync_chat_messages(THREAD, records, prune_missing = True)

    assert len(_assert_no_dangling_parents()) == len(_REPORTED_THREAD)


def test_collapsing_the_reported_pair_would_strand_four_replies(db):
    """Why a fix must key on identity: the duplicate is not a leaf."""
    _seed_reported_thread(db)
    stored = _stored()

    children = [m for m in stored.values() if m["parentId"] in ("oHXbD51", "SaKf868")]
    assert len(children) == 4
