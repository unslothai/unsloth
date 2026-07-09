# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import asyncio
import base64
import json
import os
import sqlite3
import sys

import pytest
from fastapi import HTTPException

_backend = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, _backend)

from routes import chat_history
from storage import studio_db
from utils.paths import studio_db_path

PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)
PNG_DATA_URL = "data:image/png;base64," + base64.b64encode(PNG_BYTES).decode("ascii")


def _reset_studio_db(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    monkeypatch.setenv("UNSLOTH_STUDIO_PROJECTS_HOME", str(tmp_path / "Projects"))
    monkeypatch.setattr(studio_db, "_schema_ready", False)


def _thread(thread_id: str = "thread-1", title: str = "Test Chat") -> dict:
    return {
        "id": thread_id,
        "title": title,
        "modelType": "base",
        "modelId": "test-model",
        "pairId": None,
        "archived": False,
        "createdAt": 1_700_000_000_000,
    }


def _message(
    message_id: str,
    created_at: int = 1_700_000_000_000,
    attachments = None,
    thread_id: str = "thread-1",
) -> dict:
    message = {
        "id": message_id,
        "threadId": thread_id,
        "parentId": None,
        "role": "user",
        "content": [{"type": "text", "text": "hello"}],
        "createdAt": created_at,
    }
    if attachments is not None:
        message["attachments"] = attachments
    return message


def _image_attachment(attachment_id: str = "att-1", name: str = "photo.png") -> dict:
    return {
        "id": attachment_id,
        "type": "image",
        "name": name,
        "contentType": "image/png",
        "content": [{"type": "image", "image": PNG_DATA_URL}],
        "status": {"type": "complete"},
    }


def _seed(
    tmp_path,
    monkeypatch,
    attachments,
    message_id: str = "msg-1",
):
    _reset_studio_db(tmp_path, monkeypatch)
    studio_db.upsert_chat_thread(_thread())
    studio_db.upsert_chat_message(_message(message_id, attachments = attachments))


def _set_raw_attachments_json(message_id: str, raw: str) -> None:
    conn = sqlite3.connect(studio_db_path())
    try:
        conn.execute(
            "UPDATE chat_messages SET attachments_json = ? WHERE id = ?",
            (raw, message_id),
        )
        conn.commit()
    finally:
        conn.close()


def _raw_attachments_json(message_id: str):
    conn = sqlite3.connect(studio_db_path())
    try:
        row = conn.execute(
            "SELECT attachments_json FROM chat_messages WHERE id = ?",
            (message_id,),
        ).fetchone()
        return row[0] if row is not None else None
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Storage: list_chat_attachments
# ---------------------------------------------------------------------------


def test_list_chat_attachments_empty_db(tmp_path, monkeypatch):
    _reset_studio_db(tmp_path, monkeypatch)
    assert studio_db.list_chat_attachments() == []


def test_list_chat_attachments_round_trip(tmp_path, monkeypatch):
    _seed(tmp_path, monkeypatch, [_image_attachment()])
    records = studio_db.list_chat_attachments()
    assert len(records) == 1
    record = records[0]
    assert record["id"] == "att-1"
    assert record["messageId"] == "msg-1"
    assert record["threadId"] == "thread-1"
    assert record["threadTitle"] == "Test Chat"
    assert record["name"] == "photo.png"
    assert record["type"] == "image"
    assert record["contentType"] == "image/png"
    assert record["createdAt"] == 1_700_000_000_000
    # Base64 length estimate is within padding error of the decoded size.
    assert abs(record["sizeBytes"] - len(PNG_BYTES)) <= 2


def test_list_chat_attachments_counts_text_utf8(tmp_path, monkeypatch):
    text = "héllo wörld é世界"
    attachment = {
        "id": "att-txt",
        "type": "document",
        "name": "notes.txt",
        "content": [{"type": "text", "text": text}],
    }
    _seed(tmp_path, monkeypatch, [attachment])
    records = studio_db.list_chat_attachments()
    assert records[0]["sizeBytes"] == len(text.encode("utf-8"))


def test_list_chat_attachments_no_content_size_is_none(tmp_path, monkeypatch):
    attachment = {"id": "att-empty", "name": "ghost.bin", "content": []}
    _seed(tmp_path, monkeypatch, [attachment])
    records = studio_db.list_chat_attachments()
    assert records[0]["sizeBytes"] is None
    assert records[0]["name"] == "ghost.bin"


def test_list_chat_attachments_defaults_missing_name(tmp_path, monkeypatch):
    attachment = {"id": "att-noname", "content": []}
    _seed(tmp_path, monkeypatch, [attachment])
    assert studio_db.list_chat_attachments()[0]["name"] == "attachment"


def test_list_chat_attachments_skips_malformed_rows(tmp_path, monkeypatch):
    _reset_studio_db(tmp_path, monkeypatch)
    studio_db.upsert_chat_thread(_thread())
    for i, raw in enumerate(
        [
            "not json at all",
            '{"id": "att-obj"}',
            "null",
            "[]",
            '[{"noid": true}, "just a string", 42]',
            '[{"id": ""}]',
        ]
    ):
        message_id = f"msg-bad-{i}"
        studio_db.upsert_chat_message(_message(message_id))
        _set_raw_attachments_json(message_id, raw)
    studio_db.upsert_chat_message(_message("msg-good", attachments = [_image_attachment("att-ok")]))
    records = studio_db.list_chat_attachments()
    assert [r["id"] for r in records] == ["att-ok"]


def test_list_chat_attachments_orders_newest_first(tmp_path, monkeypatch):
    _reset_studio_db(tmp_path, monkeypatch)
    studio_db.upsert_chat_thread(_thread())
    studio_db.upsert_chat_message(
        _message("msg-old", 1_700_000_000_000, [_image_attachment("att-old")])
    )
    studio_db.upsert_chat_message(
        _message("msg-new", 1_700_000_100_000, [_image_attachment("att-new")])
    )
    assert [r["id"] for r in studio_db.list_chat_attachments()] == ["att-new", "att-old"]


def test_list_chat_attachments_survives_missing_thread_row(tmp_path, monkeypatch):
    _reset_studio_db(tmp_path, monkeypatch)
    studio_db.upsert_chat_thread(_thread())
    studio_db.upsert_chat_message(_message("msg-1", attachments = [_image_attachment()]))
    conn = sqlite3.connect(studio_db_path())
    try:
        conn.execute("DELETE FROM chat_threads WHERE id = 'thread-1'")
        conn.commit()
    finally:
        conn.close()
    records = studio_db.list_chat_attachments()
    assert len(records) == 1
    assert records[0]["threadTitle"] is None


def test_list_chat_attachments_gone_after_thread_delete(tmp_path, monkeypatch):
    _seed(tmp_path, monkeypatch, [_image_attachment()])
    studio_db.delete_chat_threads(["thread-1"])
    assert studio_db.list_chat_attachments() == []


# ---------------------------------------------------------------------------
# Storage: get_chat_attachment / delete_chat_attachment
# ---------------------------------------------------------------------------


def test_get_chat_attachment_found_and_missing(tmp_path, monkeypatch):
    _seed(tmp_path, monkeypatch, [_image_attachment()])
    attachment = studio_db.get_chat_attachment("msg-1", "att-1")
    assert attachment is not None
    assert attachment["content"][0]["image"] == PNG_DATA_URL
    assert studio_db.get_chat_attachment("msg-1", "att-missing") is None
    assert studio_db.get_chat_attachment("msg-missing", "att-1") is None


def test_delete_chat_attachment_keeps_others(tmp_path, monkeypatch):
    _seed(
        tmp_path,
        monkeypatch,
        [_image_attachment("att-1"), _image_attachment("att-2", "other.png")],
    )
    assert studio_db.delete_chat_attachment("msg-1", "att-1") is True
    assert studio_db.get_chat_attachment("msg-1", "att-1") is None
    assert studio_db.get_chat_attachment("msg-1", "att-2") is not None
    assert [r["id"] for r in studio_db.list_chat_attachments()] == ["att-2"]


def test_delete_last_chat_attachment_stores_null(tmp_path, monkeypatch):
    _seed(tmp_path, monkeypatch, [_image_attachment()])
    assert studio_db.delete_chat_attachment("msg-1", "att-1") is True
    assert _raw_attachments_json("msg-1") is None
    assert studio_db.list_chat_attachments() == []
    # The message itself must survive with its content intact.
    message = studio_db.get_chat_message("thread-1", "msg-1")
    assert message is not None
    assert message["content"] == [{"type": "text", "text": "hello"}]


def test_delete_chat_attachment_missing_targets(tmp_path, monkeypatch):
    _seed(tmp_path, monkeypatch, [_image_attachment()])
    assert studio_db.delete_chat_attachment("msg-missing", "att-1") is False
    assert studio_db.delete_chat_attachment("msg-1", "att-missing") is False
    _set_raw_attachments_json("msg-1", "not json")
    assert studio_db.delete_chat_attachment("msg-1", "att-1") is False


# ---------------------------------------------------------------------------
# Routes: /attachments endpoints (real storage, direct calls)
# ---------------------------------------------------------------------------


def test_list_attachments_route(tmp_path, monkeypatch):
    _seed(tmp_path, monkeypatch, [_image_attachment()])
    result = asyncio.run(chat_history.list_attachments(current_subject = "unsloth"))
    assert [a["id"] for a in result["attachments"]] == ["att-1"]


def test_attachment_file_serves_image_bytes(tmp_path, monkeypatch):
    _seed(tmp_path, monkeypatch, [_image_attachment()])
    response = asyncio.run(
        chat_history.get_attachment_file("msg-1", "att-1", current_subject = "unsloth")
    )
    assert response.body == PNG_BYTES
    assert response.media_type == "image/png"


def test_attachment_file_tolerates_whitespace_in_base64(tmp_path, monkeypatch):
    encoded = base64.b64encode(PNG_BYTES).decode("ascii")
    wrapped = "\n".join(encoded[i : i + 8] for i in range(0, len(encoded), 8))
    attachment = _image_attachment()
    attachment["content"] = [{"type": "image", "image": "data:image/png;base64," + wrapped}]
    _seed(tmp_path, monkeypatch, [attachment])
    response = asyncio.run(
        chat_history.get_attachment_file("msg-1", "att-1", current_subject = "unsloth")
    )
    assert response.body == PNG_BYTES


def test_attachment_file_corrupt_base64_is_422(tmp_path, monkeypatch):
    attachment = _image_attachment()
    attachment["content"] = [{"type": "image", "image": "data:image/png;base64,%%%"}]
    _seed(tmp_path, monkeypatch, [attachment])
    with pytest.raises(HTTPException) as excinfo:
        asyncio.run(chat_history.get_attachment_file("msg-1", "att-1", current_subject = "unsloth"))
    assert excinfo.value.status_code == 422


def test_attachment_file_accepts_urlsafe_base64(tmp_path, monkeypatch):
    data = bytes(range(251, 256)) * 3  # encodes to characters remapped by urlsafe
    payload = base64.urlsafe_b64encode(data).decode("ascii")
    assert "-" in payload or "_" in payload
    attachment = _image_attachment()
    attachment["content"] = [{"type": "image", "image": "data:image/png;base64," + payload}]
    _seed(tmp_path, monkeypatch, [attachment])
    response = asyncio.run(
        chat_history.get_attachment_file("msg-1", "att-1", current_subject = "unsloth")
    )
    assert response.body == data


def test_attachment_file_accepts_missing_padding(tmp_path, monkeypatch):
    payload = base64.b64encode(PNG_BYTES).decode("ascii").rstrip("=")
    attachment = _image_attachment()
    attachment["content"] = [{"type": "image", "image": "data:image/png;base64," + payload}]
    _seed(tmp_path, monkeypatch, [attachment])
    response = asyncio.run(
        chat_history.get_attachment_file("msg-1", "att-1", current_subject = "unsloth")
    )
    assert response.body == PNG_BYTES


def test_attachment_file_serves_percent_encoded_data_url(tmp_path, monkeypatch):
    attachment = _image_attachment()
    attachment["content"] = [{"type": "image", "image": "data:text/plain,hello%20world"}]
    _seed(tmp_path, monkeypatch, [attachment])
    response = asyncio.run(
        chat_history.get_attachment_file("msg-1", "att-1", current_subject = "unsloth")
    )
    assert response.body == b"hello world"
    assert response.media_type == "text/plain"


def test_attachment_file_serves_text_parts(tmp_path, monkeypatch):
    attachment = {
        "id": "att-txt",
        "type": "document",
        "name": "notes.txt",
        "content": [
            {"type": "text", "text": "first"},
            {"type": "text", "text": "second"},
        ],
    }
    _seed(tmp_path, monkeypatch, [attachment])
    response = asyncio.run(
        chat_history.get_attachment_file("msg-1", "att-txt", current_subject = "unsloth")
    )
    assert response.body.decode("utf-8") == "first\nsecond"
    assert response.media_type.startswith("text/plain")


def test_attachment_file_no_content_is_404(tmp_path, monkeypatch):
    _seed(tmp_path, monkeypatch, [{"id": "att-empty", "name": "ghost", "content": []}])
    with pytest.raises(HTTPException) as excinfo:
        asyncio.run(
            chat_history.get_attachment_file("msg-1", "att-empty", current_subject = "unsloth")
        )
    assert excinfo.value.status_code == 404


def test_attachment_file_missing_message_is_404(tmp_path, monkeypatch):
    _reset_studio_db(tmp_path, monkeypatch)
    with pytest.raises(HTTPException) as excinfo:
        asyncio.run(chat_history.get_attachment_file("nope", "att-1", current_subject = "unsloth"))
    assert excinfo.value.status_code == 404


def test_attachment_file_non_data_url_image_is_404(tmp_path, monkeypatch):
    attachment = _image_attachment()
    attachment["content"] = [{"type": "image", "image": "https://example.com/a.png"}]
    _seed(tmp_path, monkeypatch, [attachment])
    with pytest.raises(HTTPException) as excinfo:
        asyncio.run(chat_history.get_attachment_file("msg-1", "att-1", current_subject = "unsloth"))
    assert excinfo.value.status_code == 404


def test_attachment_file_defaults_media_type(tmp_path, monkeypatch):
    payload = base64.b64encode(b"raw-bytes").decode("ascii")
    attachment = _image_attachment()
    attachment["content"] = [{"type": "image", "image": "data:;base64," + payload}]
    _seed(tmp_path, monkeypatch, [attachment])
    response = asyncio.run(
        chat_history.get_attachment_file("msg-1", "att-1", current_subject = "unsloth")
    )
    assert response.body == b"raw-bytes"
    assert response.media_type == "application/octet-stream"


def test_attachment_file_svg_media_type(tmp_path, monkeypatch):
    svg = b"<svg xmlns='http://www.w3.org/2000/svg'/>"
    payload = base64.b64encode(svg).decode("ascii")
    attachment = _image_attachment()
    attachment["content"] = [{"type": "image", "image": "data:image/svg+xml;base64," + payload}]
    _seed(tmp_path, monkeypatch, [attachment])
    response = asyncio.run(
        chat_history.get_attachment_file("msg-1", "att-1", current_subject = "unsloth")
    )
    assert response.body == svg
    assert response.media_type == "image/svg+xml"


def test_delete_attachment_route_then_404(tmp_path, monkeypatch):
    _seed(tmp_path, monkeypatch, [_image_attachment()])
    result = asyncio.run(
        chat_history.delete_attachment("msg-1", "att-1", current_subject = "unsloth")
    )
    assert result == {"ok": True}
    with pytest.raises(HTTPException) as excinfo:
        asyncio.run(chat_history.delete_attachment("msg-1", "att-1", current_subject = "unsloth"))
    assert excinfo.value.status_code == 404
