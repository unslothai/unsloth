# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import base64
import json
import os
import sqlite3
import sys
import time

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
    monkeypatch.setattr(studio_db, "_schema_ready", set())


def _thread(
    thread_id: str = "thread-1",
    title: str = "Test Chat",
    pair_id: str | None = None,
) -> dict:
    return {
        "id": thread_id,
        "title": title,
        "modelType": "base",
        "modelId": "test-model",
        "pairId": pair_id,
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


def test_list_chat_attachments_sanitizes_structured_metadata(tmp_path, monkeypatch):
    attachment = {
        "id": "att-weird",
        "name": {"nested": "name"},
        "type": ["image"],
        "contentType": {"mime": "image/png"},
        "content": [],
    }
    _seed(tmp_path, monkeypatch, [attachment])
    record = studio_db.list_chat_attachments()[0]
    assert record["name"] == "attachment"
    assert record["type"] is None
    assert record["contentType"] is None


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


def test_list_chat_attachments_includes_compare_pair_id(tmp_path, monkeypatch):
    _reset_studio_db(tmp_path, monkeypatch)
    studio_db.upsert_chat_thread(_thread(pair_id = "pair-1"))
    studio_db.upsert_chat_message(_message("msg-compare", attachments = [_image_attachment()]))
    record = studio_db.list_chat_attachments()[0]
    assert record["threadId"] == "thread-1"
    assert record["pairId"] == "pair-1"


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


def test_delete_last_chat_attachment_stores_empty_list(tmp_path, monkeypatch):
    _seed(tmp_path, monkeypatch, [_image_attachment()])
    assert studio_db.delete_chat_attachment("msg-1", "att-1") is True
    # '[]' rather than NULL: a NULL attachments field reads back as missing
    # and triggers the legacy IndexedDB backfill, resurrecting the deleted
    # attachment on the next chat load.
    assert _raw_attachments_json("msg-1") == "[]"
    assert studio_db.list_chat_attachments() == []
    # The message itself must survive with its content intact.
    message = studio_db.get_chat_message("thread-1", "msg-1")
    assert message is not None
    assert message["content"] == [{"type": "text", "text": "hello"}]
    assert message["attachments"] == []


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
    result = chat_history.list_attachments(current_subject = "unsloth")
    assert [a["id"] for a in result["attachments"]] == ["att-1"]


def test_attachment_file_serves_image_bytes(tmp_path, monkeypatch):
    _seed(tmp_path, monkeypatch, [_image_attachment()])
    response = chat_history.get_attachment_file("msg-1", "att-1", current_subject = "unsloth")
    assert response.body == PNG_BYTES
    assert response.media_type == "image/png"


def test_attachment_file_tolerates_whitespace_in_base64(tmp_path, monkeypatch):
    encoded = base64.b64encode(PNG_BYTES).decode("ascii")
    wrapped = "\n".join(encoded[i : i + 8] for i in range(0, len(encoded), 8))
    attachment = _image_attachment()
    attachment["content"] = [{"type": "image", "image": "data:image/png;base64," + wrapped}]
    _seed(tmp_path, monkeypatch, [attachment])
    response = chat_history.get_attachment_file("msg-1", "att-1", current_subject = "unsloth")
    assert response.body == PNG_BYTES


def test_attachment_file_corrupt_base64_is_422(tmp_path, monkeypatch):
    attachment = _image_attachment()
    attachment["content"] = [{"type": "image", "image": "data:image/png;base64,%%%"}]
    _seed(tmp_path, monkeypatch, [attachment])
    with pytest.raises(HTTPException) as excinfo:
        chat_history.get_attachment_file("msg-1", "att-1", current_subject = "unsloth")
    assert excinfo.value.status_code == 422


def test_attachment_file_accepts_urlsafe_base64(tmp_path, monkeypatch):
    data = bytes(range(251, 256)) * 3  # encodes to characters remapped by urlsafe
    payload = base64.urlsafe_b64encode(data).decode("ascii")
    assert "-" in payload or "_" in payload
    attachment = _image_attachment()
    attachment["content"] = [{"type": "image", "image": "data:image/png;base64," + payload}]
    _seed(tmp_path, monkeypatch, [attachment])
    response = chat_history.get_attachment_file("msg-1", "att-1", current_subject = "unsloth")
    assert response.body == data


def test_attachment_file_accepts_missing_padding(tmp_path, monkeypatch):
    payload = base64.b64encode(PNG_BYTES).decode("ascii").rstrip("=")
    attachment = _image_attachment()
    attachment["content"] = [{"type": "image", "image": "data:image/png;base64," + payload}]
    _seed(tmp_path, monkeypatch, [attachment])
    response = chat_history.get_attachment_file("msg-1", "att-1", current_subject = "unsloth")
    assert response.body == PNG_BYTES


def test_attachment_file_serves_percent_encoded_data_url(tmp_path, monkeypatch):
    attachment = _image_attachment()
    attachment["content"] = [{"type": "image", "image": "data:text/plain,hello%20world"}]
    _seed(tmp_path, monkeypatch, [attachment])
    response = chat_history.get_attachment_file("msg-1", "att-1", current_subject = "unsloth")
    assert response.body == b"hello world"
    # Non-image data URL types are clamped so markup never renders same-origin.
    assert response.media_type == "application/octet-stream"


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
    response = chat_history.get_attachment_file("msg-1", "att-txt", current_subject = "unsloth")
    assert response.body.decode("utf-8") == "first\nsecond"
    assert response.media_type.startswith("text/plain")


def test_attachment_file_no_content_is_404(tmp_path, monkeypatch):
    _seed(tmp_path, monkeypatch, [{"id": "att-empty", "name": "ghost", "content": []}])
    with pytest.raises(HTTPException) as excinfo:
        chat_history.get_attachment_file("msg-1", "att-empty", current_subject = "unsloth")
    assert excinfo.value.status_code == 404


def test_attachment_file_missing_message_is_404(tmp_path, monkeypatch):
    _reset_studio_db(tmp_path, monkeypatch)
    with pytest.raises(HTTPException) as excinfo:
        chat_history.get_attachment_file("nope", "att-1", current_subject = "unsloth")
    assert excinfo.value.status_code == 404


def test_attachment_file_non_data_url_image_is_404(tmp_path, monkeypatch):
    attachment = _image_attachment()
    attachment["content"] = [{"type": "image", "image": "https://example.com/a.png"}]
    _seed(tmp_path, monkeypatch, [attachment])
    with pytest.raises(HTTPException) as excinfo:
        chat_history.get_attachment_file("msg-1", "att-1", current_subject = "unsloth")
    assert excinfo.value.status_code == 404


def test_attachment_file_defaults_media_type(tmp_path, monkeypatch):
    payload = base64.b64encode(b"raw-bytes").decode("ascii")
    attachment = _image_attachment()
    attachment["content"] = [{"type": "image", "image": "data:;base64," + payload}]
    _seed(tmp_path, monkeypatch, [attachment])
    response = chat_history.get_attachment_file("msg-1", "att-1", current_subject = "unsloth")
    assert response.body == b"raw-bytes"
    assert response.media_type == "application/octet-stream"


def test_attachment_file_svg_media_type(tmp_path, monkeypatch):
    svg = b"<svg xmlns='http://www.w3.org/2000/svg'/>"
    payload = base64.b64encode(svg).decode("ascii")
    attachment = _image_attachment()
    attachment["content"] = [{"type": "image", "image": "data:image/svg+xml;base64," + payload}]
    _seed(tmp_path, monkeypatch, [attachment])
    response = chat_history.get_attachment_file("msg-1", "att-1", current_subject = "unsloth")
    assert response.body == svg
    # SVG can carry scripts, so it downloads as bytes instead of rendering.
    assert response.media_type == "application/octet-stream"


def test_delete_attachment_route_then_404(tmp_path, monkeypatch):
    _seed(tmp_path, monkeypatch, [_image_attachment()])
    result = chat_history.delete_attachment("msg-1", "att-1", current_subject = "unsloth")
    assert result == {"ok": True}
    with pytest.raises(HTTPException) as excinfo:
        chat_history.delete_attachment("msg-1", "att-1", current_subject = "unsloth")
    assert excinfo.value.status_code == 404


# ---------------------------------------------------------------------------
# Stored original files
# ---------------------------------------------------------------------------


def _upload(data: bytes, filename: str = "book.xlsx") -> dict:
    import io

    from fastapi import UploadFile
    return chat_history.upload_attachment_file(
        UploadFile(io.BytesIO(data), filename = filename), current_subject = "unsloth"
    )


def test_tool_only_upload_returns_a_preview(tmp_path, monkeypatch):
    import io
    import zipfile

    _reset_studio_db(tmp_path, monkeypatch)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("big.bin", b"x" * 5000)
        archive.writestr("notes/readme.txt", "codeword: OTTER")
    preview = _upload(buffer.getvalue(), "cities.zip")["preview"]
    assert preview == {
        "kind": "outline",
        "text": "2 entries:\n  big.bin (5000 bytes)\n  notes/readme.txt (15 bytes)\n"
        "--- notes/readme.txt ---\ncodeword: OTTER",
    }
    # A corrupt file is still stored; it just has no preview.
    broken = _upload(b"PK\x03\x04 not a zip", "t.zip")
    assert "preview" not in broken and broken["id"]

    # A reader that runs past the deadline is killed: MuPDF takes about 18 s to lay this out.
    from core import chat_attachment_preview as previews

    monkeypatch.setattr(previews, "PREVIEW_TIMEOUT_SECONDS", 1.0)
    started = time.monotonic()
    slow = f"<FictionBook><body><section><p>{'a' * 128_000}</p></section></body></FictionBook>"
    assert "preview" not in _upload(slow.encode(), "slow.fb2") and time.monotonic() - started < 5


def test_previews_convert_images_and_read_drawings(tmp_path):
    import io
    import zipfile

    from PIL import Image

    from core.chat_attachment_preview import build_preview

    # 16-bit grey is stretched to 8 bits rather than clipped to white.
    Image.frombytes("I;16", (2, 1), bytes([0, 0, 128, 0])).save(tmp_path / "depth.pgm")
    preview = build_preview(tmp_path / "depth.pgm", "depth.pgm")
    png = Image.open(io.BytesIO(base64.b64decode(preview["image"].split(",", 1)[1])))
    assert preview["description"].endswith("2x1, mode I")
    assert (png.mode, list(png.getdata())) == ("L", [0, 255])

    with zipfile.ZipFile(tmp_path / "deck.odp", "w") as archive:
        archive.writestr(
            "content.xml",
            '<o xmlns:d="urn:d" xmlns:t="urn:t"><d:page><t:p>Roadmap</t:p></d:page>'
            "<d:page><t:h>Beta</t:h><t:p>April</t:p></d:page></o>",
        )
    # The upload stores an attachment under its hash, so the reader cannot go by the path.
    xps = "http://schemas.openxps.org/oxps/v1.0"
    with zipfile.ZipFile(tmp_path / "3f9a", "w") as archive:
        archive.writestr(
            "_rels/.rels",
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            f'<Relationship Id="r1" Type="{xps}/fixedrepresentation" Target="/s.fdseq"/>'
            "</Relationships>",
        )
        archive.writestr(
            "s.fdseq",
            f'<FixedDocumentSequence xmlns="{xps}"><DocumentReference Source="/d.fdoc"/></FixedDocumentSequence>',
        )
        archive.writestr(
            "d.fdoc",
            f'<FixedDocument xmlns="{xps}"><PageContent Source="/p.fpage"/></FixedDocument>',
        )
        archive.writestr(
            "p.fpage", f'<FixedPage xmlns="{xps}" Width="816" Height="1056" xml:lang="en-US"/>'
        )
    assert build_preview(tmp_path / "3f9a", "manifest.oxps") == {
        "kind": "outline",
        "text": "1 pages with no text (images only)",
    }

    assert build_preview(tmp_path / "deck.odp", "Deck.ODP") == {
        "kind": "text",
        "label": "ODP",
        "text": "[Slide 1]\nRoadmap\n\n[Slide 2]\nBeta\nApril",
    }


def test_previews_stop_at_their_budgets(tmp_path, monkeypatch):
    import gzip
    import io
    import struct
    import tarfile
    import zipfile
    from xml.etree import ElementTree

    import fitz
    from PIL import Image

    from core import chat_attachment_preview as previews

    def text(name: str) -> str:
        return previews.build_preview(tmp_path / name, name)["text"]

    # tarfile reads extended headers whole, so the scan cap has to hold below it.
    monkeypatch.setattr(previews, "MAX_SCANNED_BYTES", 4096)
    with tarfile.open(tmp_path / "logs.tar.gz", "w:gz") as archive:
        for name, pax in (("a.txt", {}), ("b.txt", {"comment": "x" * 8192}), ("c.txt", {})):
            info = tarfile.TarInfo(name)
            info.size, info.pax_headers = 5, pax
            archive.addfile(info, io.BytesIO(b"first"))
    outline = text("logs.tar.gz")
    assert outline.startswith("1+ entries:") and "first" in outline and "c.txt" not in outline
    with tarfile.open(
        tmp_path / "latin.tar", "w", format = tarfile.GNU_FORMAT, encoding = "latin-1"
    ) as tar:
        tar.addfile(tarfile.TarInfo("café.txt"))
    assert "caf\ufffd.txt" in text("latin.tar")

    monkeypatch.setattr(previews, "MAX_XML_BYTES", 250)
    with zipfile.ZipFile(tmp_path / "net.vsdx", "w") as archive:
        for n in (10, 2, 1):
            archive.writestr(f"visio/pages/page{n}.xml", f"<P><Text>{'n' * 80}{n}</Text></P>")
    assert text("net.vsdx") == (
        f"[Page 1]\n{'n' * 80}1\n\n[Page 2]\n{'n' * 80}2\n[Truncated: 1 more pages over the read budget]"
    )

    monkeypatch.setattr(previews, "MAX_DOCUMENT_PAGES", 1)
    (tmp_path / "book.fb2").write_text(
        f"<FictionBook><body><section>{'<p>tide and harbour</p>' * 400}</section></body></FictionBook>"
    )
    assert text("book.fb2").endswith("more pages not read]")

    # The read cap counts bytes, so multi-byte text is cut short of the character cap.
    monkeypatch.setattr(previews, "MAX_TEXT_CHARS", 10)
    assert text("book.fb2").endswith(
        "[Truncated: the document has more text than one attachment carries]"
    )
    (tmp_path / "poem.txt.gz").write_bytes(gzip.compress("la marée monte".encode()))
    assert (
        text("poem.txt.gz")
        == "la marée \n[Truncated: poem.txt is longer than one attachment carries]"
    )

    # Nested elements repeat their descendants' text, so the budget bounds what is built at all.
    nested = ("<p>" * 2000) + ("body " * 100) + ("</p>" * 2000)
    root = ElementTree.fromstring(f"<o>{nested}</o>")
    assert len(previews._xml_text(root, {"p"}, 50)) <= 50
    siblings = ElementTree.fromstring("<o>" + "<p>ten chars</p>" * 50 + "</o>")
    assert len(previews._xml_text(siblings, {"p"}, 50)) <= 50
    two = '<o xmlns:d="urn:d"><d:page><p>{}</p></d:page><d:page><p>{}</p></d:page></o>'
    with zipfile.ZipFile(tmp_path / "two.odp", "w") as archive:
        archive.writestr("content.xml", two.format("a" * 50, "b" * 50))
    assert "b" not in previews._drawing_text(tmp_path / "two.odp", ".odp")

    # ZipFile does not cap what one read of an LZMA or bzip2 member inflates, so those are not read.
    with zipfile.ZipFile(tmp_path / "packed.zip", "w", zipfile.ZIP_LZMA) as archive:
        archive.writestr("notes.txt", "packed")
    assert "packed" not in text("packed.zip")
    with zipfile.ZipFile(tmp_path / "packed.odp", "w", zipfile.ZIP_BZIP2) as archive:
        archive.writestr("content.xml", "<o><p>packed</p></o>")
    assert previews.build_preview(tmp_path / "packed.odp", "packed.odp") is None

    monkeypatch.setattr(previews, "LISTED_MEMBERS", 2)
    with zipfile.ZipFile(tmp_path / "bin.zip", "w") as archive:
        for name, data in (("a.bin", b"\x80"), ("b.bin", b"\x81"), ("c.txt", b"late")):
            archive.writestr(name, data)
    assert "late" not in text("bin.zip")
    monkeypatch.setattr(previews, "MAX_ZIP_DIRECTORY_BYTES", 100)
    assert previews.build_preview(tmp_path / "bin.zip", "bin.zip") == {
        "kind": "outline",
        "text": "central directory of 153 bytes, too large to list",
    }
    # Documents that are zips get the same directory check, and image-only pages end the walk too.
    with zipfile.ZipFile(tmp_path / "comic.cbz", "w") as archive:
        archive.writestr("ComicInfo.xml", "<ComicInfo/>")
        for number in range(3):
            page = io.BytesIO()
            Image.new("RGB", (1, 1)).save(page, format = "PNG")
            archive.writestr(f"{number}.png", page.getvalue())
    assert text("comic.cbz").startswith("central directory of")
    monkeypatch.setattr(previews, "MAX_ZIP_DIRECTORY_BYTES", 4096)
    monkeypatch.setattr(previews, "MAX_DOCUMENT_MEMBER_BYTES", 60)
    assert text("comic.cbz") == "0.png is 69 bytes, too large to read"
    monkeypatch.setattr(previews, "MAX_DOCUMENT_MEMBER_BYTES", 4096)
    monkeypatch.setattr(previews, "MAX_DOCUMENT_PAGES", 2)
    assert text("comic.cbz") == "3 pages, no text in the first 2 (images only)"

    # An icon's PNG or BMP frame is checked before Pillow decodes it on open.
    monkeypatch.setattr(previews, "MAX_IMAGE_PIXELS", 100)
    for frames in ("png", "bmp"):
        Image.new("RGB", (16, 16)).save(tmp_path / f"{frames}.ico", bitmap_format = frames)
    # An icon named as another format opens as that format only, so it cannot skip the check.
    (tmp_path / "icon.tga").write_bytes((tmp_path / "png.ico").read_bytes())
    assert previews.build_preview(tmp_path / "icon.tga", "icon.tga") is None
    monkeypatch.setattr(Image, "open", lambda *args, **kwargs: pytest.fail("icon was decoded"))
    siz = b"\xff\x4f\xff\x51" + struct.pack(">HHII", 41, 0, 16, 16)
    (tmp_path / "raw.icns").write_bytes(
        b"icns" + struct.pack(">I", 24 + len(siz)) + b"ic09" + struct.pack(">I", 8 + len(siz)) + siz
    )
    assert previews.build_preview(tmp_path / "raw.icns", "raw.icns")["kind"] == "outline"
    for frames in ("png", "bmp"):
        assert previews.build_preview(tmp_path / f"{frames}.ico", "app.ico")["kind"] == "outline"

    seen = {}

    def opened(path, **kwargs):
        seen.update(kwargs)
        raise ValueError("stop")

    monkeypatch.setattr(fitz, "open", opened)
    zipfile.ZipFile(tmp_path / "deck.ppsm", "w").close()
    assert previews.build_preview(tmp_path / "deck.ppsm", "deck.ppsm") is None
    assert seen == {"filetype": "pptx"}


def test_attachment_upload_dedupes_and_renews(tmp_path, monkeypatch):
    import hashlib

    from storage import chat_attachment_store as store

    _reset_studio_db(tmp_path, monkeypatch)
    first = _upload(b"sheet bytes")
    digest = hashlib.sha256(b"sheet bytes").hexdigest()
    assert first == {
        "id": digest,
        "sizeBytes": 11,
        "sandboxPath": f".unsloth_attachments/{digest[:12]}/book.xlsx",
    }
    path = store.attachment_path(first["id"])
    assert path.read_bytes() == b"sheet bytes"
    os.utime(path, (1, 1))
    assert _upload(b"sheet bytes") == first
    assert path.stat().st_mtime > 1
    assert [p.name for p in path.parent.iterdir()] == [first["id"]]

    monkeypatch.setattr(store, "MAX_ATTACHMENT_BYTES", 4)
    for data, status in ((b"", 400), (b"too big", 413)):
        with pytest.raises(HTTPException) as excinfo:
            _upload(data)
        assert excinfo.value.status_code == status
    assert [p.name for p in path.parent.iterdir()] == [first["id"]]


def test_attachment_sweep_keeps_referenced_and_recent_files(tmp_path, monkeypatch):
    from storage import chat_attachment_store as store

    _reset_studio_db(tmp_path, monkeypatch)
    kept, orphan, recent = (_upload(data)["id"] for data in (b"kept", b"orphan", b"recent"))
    attachment = {**_image_attachment(), "storedFileId": kept}
    studio_db.upsert_chat_thread(_thread())
    studio_db.upsert_chat_message(_message("msg-1", attachments = [attachment]))
    root = store.attachment_path(kept).parent
    (root / ".upload-abandoned").write_bytes(b"partial")
    for name in (kept, orphan, ".upload-abandoned"):
        os.utime(root / name, (1, 1))

    assert store.sweep_attachments() == 2
    assert sorted(p.name for p in root.iterdir()) == sorted([kept, recent])

    chat_history.delete_attachment("msg-1", "att-1", current_subject = "unsloth")
    assert sorted(p.name for p in root.iterdir()) == [recent]
    # A server that deletes nothing still reclaims, on an upload, at most once an interval.
    os.utime(root / recent, (1, 1))
    _upload(b"fresh")
    assert recent in [p.name for p in root.iterdir()]
    monkeypatch.setattr(store, "_swept_at", {})
    _upload(b"newer")
    assert recent not in [p.name for p in root.iterdir()]
    # Keyed per store, so a busy account's uploads cannot hold off the sweep of a quiet one.
    monkeypatch.setattr(store, "_root", lambda: tmp_path / "other")
    store.sweep_attachments_if_due()
    assert sorted(store._swept_at) == sorted([root, tmp_path / "other"])


def test_attachment_file_serves_the_stored_original(tmp_path, monkeypatch):
    _reset_studio_db(tmp_path, monkeypatch)
    stored = _upload(b"PK original workbook")
    attachment = {
        "id": "att-xlsx",
        "type": "document",
        "name": "book.xlsx",
        "content": [{"type": "text", "text": "[XLSX: book.xlsx]"}],
        "storedFile": stored,
    }
    studio_db.upsert_chat_thread(_thread())
    studio_db.upsert_chat_message(_message("msg-1", attachments = [attachment]))
    response = chat_history.get_attachment_file("msg-1", "att-xlsx", current_subject = "unsloth")
    assert open(response.path, "rb").read() == b"PK original workbook"
    assert response.media_type == "application/octet-stream"

    os.remove(response.path)
    response = chat_history.get_attachment_file("msg-1", "att-xlsx", current_subject = "unsloth")
    assert response.body.decode("utf-8") == "[XLSX: book.xlsx]"


# ---------------------------------------------------------------------------
# Audio attachments (adapter {data, format} and compare-chat bare base64)
# ---------------------------------------------------------------------------

WAV_BYTES = b"RIFF$\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00"
WAV_B64 = base64.b64encode(WAV_BYTES).decode("ascii")


def _audio_attachment(attachment_id: str = "att-audio") -> dict:
    return {
        "id": attachment_id,
        "type": "file",
        "name": "clip.wav",
        "contentType": "audio/wav",
        "content": [{"type": "audio", "audio": {"data": WAV_B64, "format": "wav"}}],
        "status": {"type": "complete"},
    }


def test_audio_attachment_lists_with_size(tmp_path, monkeypatch):
    _seed(tmp_path, monkeypatch, [_audio_attachment()])
    records = studio_db.list_chat_attachments()
    assert len(records) == 1
    assert records[0]["id"] == "att-audio"
    assert abs(records[0]["sizeBytes"] - len(WAV_BYTES)) <= 2


def test_audio_attachment_file_serves_bytes(tmp_path, monkeypatch):
    _seed(tmp_path, monkeypatch, [_audio_attachment()])
    response = chat_history.get_attachment_file("msg-1", "att-audio", current_subject = "unsloth")
    assert response.body == WAV_BYTES
    assert response.media_type == "audio/wav"


def test_audio_attachment_media_type_from_format(tmp_path, monkeypatch):
    attachment = _audio_attachment()
    attachment["contentType"] = None
    attachment["content"] = [{"type": "audio", "audio": {"data": WAV_B64, "format": "mp3"}}]
    _seed(tmp_path, monkeypatch, [attachment])
    response = chat_history.get_attachment_file("msg-1", "att-audio", current_subject = "unsloth")
    assert response.media_type == "audio/mpeg"


def test_audio_attachment_corrupt_payload_is_422(tmp_path, monkeypatch):
    attachment = _audio_attachment()
    attachment["content"] = [{"type": "audio", "audio": {"data": "%%%", "format": "wav"}}]
    _seed(tmp_path, monkeypatch, [attachment])
    with pytest.raises(HTTPException) as excinfo:
        chat_history.get_attachment_file("msg-1", "att-audio", current_subject = "unsloth")
    assert excinfo.value.status_code == 422


# ---------------------------------------------------------------------------
# Compare-chat uploads stored as message content parts
# ---------------------------------------------------------------------------


def _compare_message(message_id: str = "msg-cmp") -> dict:
    return {
        "id": message_id,
        "threadId": "thread-1",
        "parentId": None,
        "role": "user",
        "content": [
            {"type": "image", "image": PNG_DATA_URL},
            {"type": "audio", "audio": WAV_B64},
            {"type": "text", "text": "compare these"},
        ],
        "createdAt": 1_700_000_000_000,
    }


def _seed_compare(tmp_path, monkeypatch):
    _reset_studio_db(tmp_path, monkeypatch)
    studio_db.upsert_chat_thread(_thread())
    studio_db.upsert_chat_message(_compare_message())


_CONTENT_PART_PREFIX = "content-part-sha256-"


def _content_part_id_for(message_id: str, kind: str) -> str:
    """Resolve the stable content-hash id for a message's stored blob.

    Content-part ids are SHA-256 hashes of the blob payload, not array
    indices, so tests look them up from the listing instead of hardcoding an
    index that would shift when an earlier part is deleted.
    """
    for record in studio_db.list_chat_attachments():
        if record["messageId"] == message_id and record["type"] == kind:
            return record["id"]
    raise AssertionError(f"no {kind} content-part upload for {message_id}")


def test_content_part_uploads_are_listed(tmp_path, monkeypatch):
    _seed_compare(tmp_path, monkeypatch)
    records = studio_db.list_chat_attachments()
    # Ids are stable content hashes, not array indices.
    assert all(r["id"].startswith(_CONTENT_PART_PREFIX) for r in records)
    assert {r["type"] for r in records} == {"image", "audio"}
    image = next(r for r in records if r["type"] == "image")
    assert image["contentType"] == "image/png"
    assert abs(image["sizeBytes"] - len(PNG_BYTES)) <= 2
    audio = next(r for r in records if r["type"] == "audio")
    assert audio["type"] == "audio"


def test_content_part_file_serves_image_bytes(tmp_path, monkeypatch):
    _seed_compare(tmp_path, monkeypatch)
    image_id = _content_part_id_for("msg-cmp", "image")
    response = chat_history.get_attachment_file("msg-cmp", image_id, current_subject = "unsloth")
    assert response.body == PNG_BYTES
    assert response.media_type == "image/png"


def test_content_part_delete_keeps_text(tmp_path, monkeypatch):
    _seed_compare(tmp_path, monkeypatch)
    image_id = _content_part_id_for("msg-cmp", "image")
    assert studio_db.delete_chat_attachment("msg-cmp", image_id) is True
    message = studio_db.get_chat_message("thread-1", "msg-cmp")
    types = [p["type"] for p in message["content"]]
    assert types == ["audio", "text"]
    # The surviving audio blob keeps its own stable hash id after the delete.
    remaining = studio_db.list_chat_attachments()
    assert [r["type"] for r in remaining] == ["audio"]
    assert remaining[0]["id"].startswith(_CONTENT_PART_PREFIX)
    assert remaining[0]["id"] != image_id


def test_content_part_delete_rejects_non_blob(tmp_path, monkeypatch):
    _seed_compare(tmp_path, monkeypatch)
    # The text part is not a stored upload, so it never gets an id: only the
    # image and audio blobs are addressable.
    assert len(studio_db.list_chat_attachments()) == 2
    # A well-formed but unknown content-hash id, and malformed ids, all no-op.
    assert studio_db.delete_chat_attachment("msg-cmp", _CONTENT_PART_PREFIX + "0" * 64) is False
    assert studio_db.delete_chat_attachment("msg-cmp", "content-part-99") is False
    assert studio_db.delete_chat_attachment("msg-cmp", "content-part-x") is False


def test_text_only_messages_not_listed_as_uploads(tmp_path, monkeypatch):
    _reset_studio_db(tmp_path, monkeypatch)
    studio_db.upsert_chat_thread(_thread())
    # The word "image" inside text must not create phantom upload rows.
    message = _message("msg-txt")
    message["content"] = [{"type": "text", "text": 'discussing an "image" and "audio" here'}]
    studio_db.upsert_chat_message(message)
    assert studio_db.list_chat_attachments() == []


def test_remote_image_urls_are_not_listed_as_uploads(tmp_path, monkeypatch):
    _reset_studio_db(tmp_path, monkeypatch)
    studio_db.upsert_chat_thread(_thread())
    message = _message("msg-remote")
    message["content"] = [
        {"type": "image", "image": "https://example.com/cat.png"},
        {"type": "text", "text": "look at this"},
    ]
    studio_db.upsert_chat_message(message)
    # No stored bytes: nothing to list, open, or delete.
    assert studio_db.list_chat_attachments() == []
    assert studio_db.get_chat_attachment("msg-remote", "content-part-0") is None
    assert studio_db.delete_chat_attachment("msg-remote", "content-part-0") is False
    stored = studio_db.get_chat_message("thread-1", "msg-remote")
    assert [p["type"] for p in stored["content"]] == ["image", "text"]


def test_html_data_url_serves_as_octet_stream(tmp_path, monkeypatch):
    _reset_studio_db(tmp_path, monkeypatch)
    studio_db.upsert_chat_thread(_thread())
    html_b64 = base64.b64encode(b"<script>alert(1)</script>").decode()
    message = _message("msg-html")
    message["content"] = [
        {"type": "image", "image": f"data:text/html;base64,{html_b64}"},
    ]
    studio_db.upsert_chat_message(message)
    attachment_id = _content_part_id_for("msg-html", "image")
    response = chat_history.get_attachment_file(
        "msg-html", attachment_id, current_subject = "unsloth"
    )
    # Never echo a script-capable media type back under the app origin.
    assert response.media_type == "application/octet-stream"
    assert response.body == b"<script>alert(1)</script>"


def test_svg_data_url_serves_as_octet_stream(tmp_path, monkeypatch):
    _reset_studio_db(tmp_path, monkeypatch)
    studio_db.upsert_chat_thread(_thread())
    svg_b64 = base64.b64encode(b"<svg onload='x'/>").decode()
    message = _message("msg-svg")
    message["content"] = [
        {"type": "image", "image": f"data:image/svg+xml;base64,{svg_b64}"},
    ]
    studio_db.upsert_chat_message(message)
    attachment_id = _content_part_id_for("msg-svg", "image")
    response = chat_history.get_attachment_file("msg-svg", attachment_id, current_subject = "unsloth")
    assert response.media_type == "application/octet-stream"


def test_png_data_url_keeps_its_media_type(tmp_path, monkeypatch):
    _seed_compare(tmp_path, monkeypatch)
    image_id = _content_part_id_for("msg-cmp", "image")
    response = chat_history.get_attachment_file("msg-cmp", image_id, current_subject = "unsloth")
    assert response.media_type == "image/png"
