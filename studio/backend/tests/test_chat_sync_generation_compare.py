# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The generation-row guard answers exactly as before when it compares stored JSON by string first."""

import copy
import json

import pytest

from storage import chat_generation_runs_db as runs_db
from storage import studio_db

UNICODE = 'quote " back \\ nul \x00 \u2028 \u00e9 \u6f22\u5b57 \U0001f600 \ud7ff'


def _old_safe_generation_assistant_update(conn, thread_id, message):
    """The guard before this change, verbatim."""
    row = conn.execute(
        """SELECT r.id AS run_id, r.status AS run_status, r.last_event_seq,
                  m.parent_id, m.role, m.content_json, m.metadata_json,
                  m.attachments_json, m.created_at
           FROM chat_generation_runs r
           JOIN chat_messages m ON m.id = r.assistant_message_id
           WHERE r.thread_id = ? AND r.assistant_message_id = ?""",
        (thread_id, str(message["id"])),
    ).fetchone()
    if row is None or str(message.get("role")) != "assistant":
        return False
    if (message.get("parentId") or None) != (row["parent_id"] or None):
        return False
    if int(message.get("createdAt", row["created_at"])) != int(row["created_at"]):
        return False

    stored_attachments = json.loads(row["attachments_json"]) if row["attachments_json"] else None
    if json.dumps(message.get("attachments"), sort_keys = True) != json.dumps(
        stored_attachments, sort_keys = True
    ):
        return False

    incoming = message.get("metadata")
    stored = json.loads(row["metadata_json"]) if row["metadata_json"] else {}
    if not isinstance(incoming, dict) or not isinstance(stored, dict):
        return False
    if incoming.get("serverManaged") is not True:
        return False
    if incoming.get("generationRunId") != row["run_id"]:
        return False

    incoming_seq = incoming.get("generationSeq")
    stored_seq = stored.get("generationSeq", 0)
    if (
        isinstance(incoming_seq, bool)
        or not isinstance(incoming_seq, int)
        or isinstance(stored_seq, bool)
        or not isinstance(stored_seq, int)
        or incoming_seq < stored_seq
        or incoming_seq > int(row["last_event_seq"])
    ):
        return False
    if incoming_seq == stored_seq and json.dumps(
        message.get("content", []), sort_keys = True
    ) != json.dumps(json.loads(row["content_json"] or "[]"), sort_keys = True):
        return False

    if stored.get("generationSettled") is True:
        return incoming == stored

    incoming_status = incoming.get("generationStatus")
    stored_status = stored.get("generationStatus")
    run_status = str(row["run_status"])
    active = studio_db._GENERATION_ACTIVE_STATUSES
    terminal = studio_db._GENERATION_TERMINAL_STATUSES
    rank = studio_db._GENERATION_STATUS_RANK
    if incoming_status not in active | terminal:
        return False
    if run_status in terminal and incoming_status != run_status:
        return False
    if run_status in terminal:
        for key in stored:
            if key not in incoming:
                return False
        if incoming.get("incomplete") != stored.get("incomplete"):
            return False
    if run_status in active and incoming_status in terminal:
        return False
    if incoming.get("generationSettled") is True and run_status not in terminal:
        return False
    if incoming.get("generationSettled") is True and incoming_seq != int(row["last_event_seq"]):
        return False
    if stored_status in terminal and incoming_status != stored_status:
        return False
    if stored.get("generationSettled") is True and incoming.get("generationSettled") is not True:
        return False
    if (
        stored_status in active
        and incoming_status in active
        and rank[incoming_status] < rank[stored_status]
    ):
        return False
    return True


def _thread():
    studio_db.upsert_chat_thread(
        {"id": "t", "title": "Chat", "modelType": "base", "modelId": "local", "createdAt": 1}
    )


def _turn(i, *, settle):
    user_id, assistant_id, run_id = f"u{i}", f"a{i}", f"r{i}"
    studio_db.upsert_chat_message(
        {
            "id": user_id,
            "threadId": "t",
            "parentId": f"a{i - 1}" if i else None,
            "role": "user",
            "content": [{"type": "text", "text": f"q{i}"}],
            "createdAt": 10 * i + 2,
        }
    )
    runs_db.create_run(
        run_id = run_id,
        owner_subject = "alice",
        thread_id = "t",
        user_message_id = user_id,
        assistant_message_id = assistant_id,
        request_payload = {
            "model": "local",
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": True,
        },
    )
    token = runs_db.get_worker_token(run_id)
    assert runs_db.mark_running(run_id, token)
    runs_db.append_events(run_id, token, [("chunk", {"text": "A"}), ("chunk", {"text": "B"})])
    message = studio_db.get_chat_message("t", assistant_id)
    message["content"] = [
        {"type": "reasoning", "text": "think " + UNICODE},
        {
            "type": "text",
            "text": f"answer {i} " + UNICODE,
            "meta": {"z": 1, "a": [1.5, None, True]},
        },
    ]
    if settle:
        runs_db.finish_run(run_id, worker_token = token, status = "completed", finish_reason = "stop")
        seq = runs_db.get_run(run_id, "alice")["lastEventSeq"]
        message["metadata"].update(
            {"generationSeq": seq, "generationStatus": "completed", "generationSettled": True}
        )
    else:
        message["metadata"].update({"generationSeq": 1, "generationStatus": "running"})
    studio_db.upsert_chat_message(message)
    return studio_db.get_chat_message("t", assistant_id)


def _reorder(value):
    if isinstance(value, dict):
        return {k: _reorder(value[k]) for k in reversed(list(value))}
    if isinstance(value, list):
        return [_reorder(v) for v in value]
    return value


def _variants(stored):
    yield "unchanged", copy.deepcopy(stored)
    yield "key_order", _reorder(copy.deepcopy(stored))
    yield "content_tuple", {**copy.deepcopy(stored), "content": tuple(stored["content"])}
    for name, change in [
        ("content_text", lambda m: m["content"][1].update(text = "changed")),
        ("content_unicode", lambda m: m["content"][1].update(text = m["content"][1]["text"] + "é")),
        ("content_int_float", lambda m: m["content"][1]["meta"].update(z = 1.0)),
        ("content_extra_part", lambda m: m["content"].append({"type": "text", "text": "x"})),
        ("content_dropped", lambda m: m.pop("content")),
        ("content_empty", lambda m: m.update(content = [])),
        ("content_string", lambda m: m.update(content = "answer")),
        ("attachments_none", lambda m: m.update(attachments = None)),
        ("attachments_empty", lambda m: m.update(attachments = [])),
        ("attachments_dropped", lambda m: m.pop("attachments", None)),
        (
            "attachments_new",
            lambda m: m.update(attachments = [{"id": "x", "type": "image", "name": "n"}]),
        ),
        ("attachments_reordered", lambda m: m.update(attachments = _reorder(m.get("attachments")))),
        ("metadata_extra", lambda m: m["metadata"].update(extra = 1)),
        ("metadata_reordered", lambda m: m.update(metadata = _reorder(m["metadata"]))),
        ("metadata_seq_back", lambda m: m["metadata"].update(generationSeq = 0)),
        ("metadata_seq_ahead", lambda m: m["metadata"].update(generationSeq = 99)),
        ("metadata_status", lambda m: m["metadata"].update(generationStatus = "running")),
        ("metadata_unsettled", lambda m: m["metadata"].pop("generationSettled", None)),
        (
            "metadata_settled",
            lambda m: m["metadata"].update(generationSettled = True, generationStatus = "completed"),
        ),
        ("metadata_run", lambda m: m["metadata"].update(generationRunId = "other")),
        ("metadata_none", lambda m: m.update(metadata = None)),
        ("role_user", lambda m: m.update(role = "user")),
        ("parent", lambda m: m.update(parentId = "elsewhere")),
        ("created_at", lambda m: m.update(createdAt = m["createdAt"] + 1)),
    ]:
        message = copy.deepcopy(stored)
        change(message)
        yield name, message


@pytest.mark.parametrize("settle", [True, False], ids = ["settled", "streaming"])
@pytest.mark.parametrize(
    "attachments",
    [
        None,
        [],
        [
            {
                "id": "a",
                "type": "document",
                "name": UNICODE,
                "content": [{"type": "text", "text": "B"}],
            }
        ],
    ],
    ids = ["no-attachments", "empty-attachments", "attachments"],
)
def test_guard_decisions_match_the_old_comparison(settle, attachments):
    _thread()
    _turn(0, settle = settle)
    conn = studio_db.get_connection()
    try:
        if attachments is not None:
            conn.execute(
                "UPDATE chat_messages SET attachments_json = ? WHERE id = 'a0'",
                (json.dumps(attachments),),
            )
            conn.commit()
        stored = studio_db.get_chat_message("t", "a0")
        assert stored.get("attachments") == attachments
        for compact in (False, True):
            if compact:
                # Another serializer's row takes the full comparison.
                row = conn.execute(
                    "SELECT content_json, attachments_json FROM chat_messages WHERE id = 'a0'"
                ).fetchone()
                conn.execute(
                    "UPDATE chat_messages SET content_json = ?, attachments_json = ? WHERE id = 'a0'",
                    (
                        json.dumps(
                            json.loads(row["content_json"]),
                            separators = (",", ":"),
                            ensure_ascii = False,
                        ),
                        None
                        if row["attachments_json"] is None
                        else json.dumps(json.loads(row["attachments_json"]), separators = (",", ":")),
                    ),
                )
            decisions = {}
            for name, message in _variants(stored):
                old = _old_safe_generation_assistant_update(conn, "t", message)
                new = studio_db._safe_generation_assistant_update(conn, "t", message)
                assert new == old, (name, compact)
                decisions[name] = new
            assert decisions["unchanged"] and decisions["key_order"]
            assert not decisions["content_text"] and not decisions["role_user"]
            assert not decisions["attachments_new"]
        user = studio_db.get_chat_message("t", "u0")
        assert _old_safe_generation_assistant_update(conn, "t", user) is False
        assert studio_db._safe_generation_assistant_update(conn, "t", user) is False
    finally:
        conn.close()


def test_whole_chat_resave_keeps_every_row_and_skips_none():
    _thread()
    for i in range(6):
        _turn(i, settle = i != 5)
    before = studio_db.list_chat_messages("t")
    studio_db.sync_chat_messages("t", copy.deepcopy(before), return_rows = False)
    studio_db.sync_chat_messages("t", _reorder(copy.deepcopy(before)), return_rows = False)
    assert studio_db.list_chat_messages("t") == before
    drifted = copy.deepcopy(before)
    drifted[1]["content"] = [{"type": "text", "text": "stale tab"}]
    studio_db.sync_chat_messages("t", drifted, return_rows = False)
    assert studio_db.list_chat_messages("t") == before
