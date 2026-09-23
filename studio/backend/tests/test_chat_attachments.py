# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import base64
import json
import os
import pathlib
import sqlite3
import struct
import sys
import time
import zipfile

import pytest
from fastapi import HTTPException
from starlette.responses import JSONResponse

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
    siblings = ElementTree.fromstring("<o>" + "<p>ten chars</p>" * 50 + "</o>")
    for built in (previews._xml_text(root, {"p"}, 50), previews._xml_text(siblings, {"p"}, 50)):
        # The marker sits past the budget, as every truncation marker in this module does.
        assert 40 <= len(built.split("\n[Truncated")[0]) <= 50 and built.endswith("shows]")
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


def _tables(tmp_path) -> None:
    import numpy as np
    import pandas as pd
    import pyarrow as pa
    import pyarrow.orc as orc

    cities = ["Oslo", "Lima", "Hanoi", "Perth", "Cairo", "Kyoto", "Quito"]
    frame = pd.DataFrame({"city": cities, "temp": [1.5, 22.0, 33.0, 18.5, 27.0, 9.0, 14.5]})
    frame.to_parquet(tmp_path / "w.parquet")
    frame.to_feather(tmp_path / "w.feather")
    frame.to_stata(tmp_path / "w.dta", write_index = False)
    orc.write_table(pa.Table.from_pandas(frame, preserve_index = False), str(tmp_path / "w.orc"))
    np.savez(
        tmp_path / "e.npz",
        signal = np.arange(4.0),
        labels = np.zeros(3, dtype = np.int8),
        counts = np.arange(20),
        blank = np.zeros(0),
    )
    np.save(tmp_path / "m.npy", np.arange(6).reshape(2, 3))


def test_previews_outline_tables_and_arrays(tmp_path, monkeypatch):
    import pandas as pd

    from core import chat_attachment_preview as previews
    from core.chat_attachment_preview import build_preview

    _tables(tmp_path)
    rows = "first 5 rows:\n  Oslo\t1.5\n  Lima\t22.0\n  Hanoi\t33.0\n  Perth\t18.5\n  Cairo\t27.0"
    for index, source in enumerate(("w.parquet", "w.feather", "w.orc", "w.dta")):
        (tmp_path / f"7c2{index}").write_bytes((tmp_path / source).read_bytes())
    for name, stored, count in (
        ("weather.parquet", "7c20", "7"),
        ("weather.feather", "7c21", "7"),
        ("weather.orc", "7c22", "7"),
        # StataReader counts observations only privately.
        ("survey.dta", "7c23", "unknown"),
    ):
        text = build_preview(tmp_path / stored, name)["text"]
        assert text.startswith(f"{count} rows x 2 columns\ncolumns: city (")
        assert text.endswith(rows), name

    assert build_preview(tmp_path / "m.npy", "m.npy") == {
        "kind": "outline",
        "text": "array: shape (2, 3), dtype int64, min 0, max 5, mean 2.5, "
        "values [[0, 1, 2], [3, 4, 5]]",
    }
    assert build_preview(tmp_path / "e.npz", "e.npz")["text"] == (
        "4 arrays\n"
        "signal: shape (4,), dtype float64, min 0, max 3, mean 1.5, values [0.0, 1.0, 2.0, 3.0]\n"
        "labels: shape (3,), dtype int8, min 0, max 0, mean 0, values [0, 0, 0]\n"
        "counts: shape (20,), dtype int64, min 0, max 19, mean 9.5\n"
        "blank: shape (0,), dtype float64, values []"
    )

    header = json.dumps(
        {
            "__metadata__": {"r": "16"},
            "w": {"dtype": "F32", "shape": [2, 3], "data_offsets": [0, 24]},
        }
    ).encode()
    (tmp_path / "a.safetensors").write_bytes(struct.pack("<Q", len(header)) + header + b"\0" * 24)
    assert build_preview(tmp_path / "a.safetensors", "a.safetensors")["text"] == (
        "1 tensors, 6 parameters, metadata {'r': '16'}\nw: shape (2, 3), dtype F32"
    )

    # A name out of a file can hold a lone surrogate, which the JSON response cannot encode.
    lone = json.dumps({"bad\ud800": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]}})
    raw = lone.encode("utf-8", "surrogatepass")
    (tmp_path / "u.safetensors").write_bytes(struct.pack("<Q", len(raw)) + raw + b"\0" * 8)
    text = build_preview(tmp_path / "u.safetensors", "u.safetensors")["text"]
    assert text.endswith("bad?: shape (2,), dtype F32")
    assert JSONResponse({"preview": text}).body

    # A .sas7bdat is opened as its own layout, and states its row count under another name.
    class _Sas7bdat:
        row_count = 9
        nobs = 4
        # pandas names the encoding a file states even when Python has no codec by that name.
        inferred_encoding = "unknown (code=0)"

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def read(self, rows):
            # SAS7BDATReader leaves character columns as bytes in the file's own encoding.
            return pd.DataFrame(
                {
                    "year": [1948, 1949, 1950],
                    "city": [b"caf\xe9", ("a" + "\u00e9" * 1000).encode(), b"\xe9" * 2000],
                }
            )

    opened = {}
    monkeypatch.setattr(pd, "read_sas", lambda path, **kwargs: opened.update(kwargs) or _Sas7bdat())
    (tmp_path / "a.sas7bdat").touch()
    text = build_preview(tmp_path / "a.sas7bdat", "airline.sas7bdat")["text"]
    assert text.startswith("9 rows x 2 columns") and opened["format"] == "sas7bdat"
    # Python reserves the buffer for the lengths a sas7bdat claims, filled or not.
    magic = bytes.fromhex("0" * 24 + "c2ea8160b31411cfbd92080009c7318c181f1011")

    def sas7bdat(
        header,
        page,
        order = "<",
        aligned = False,
        pad = b"",
    ):
        head = bytearray(magic + b"\x00" * 256)
        head[35] = 0x33 if aligned else 0
        head[37] = 1 if order == "<" else 0
        at = 196 + (4 if aligned else 0)
        head[at : at + 8] = struct.pack(f"{order}2I", header, page)
        (tmp_path / "claim.sas7bdat").write_bytes(bytes(head) + pad)
        return build_preview(tmp_path / "claim.sas7bdat", "big.sas7bdat")["text"]

    claim = "67108864 bytes to read through before the first row"
    assert sas7bdat(4096, 64 * 1024 * 1024) == claim
    assert sas7bdat(64 * 1024 * 1024, 4096, order = ">", aligned = True) == claim

    (tmp_path / "t.xpt").touch()
    assert build_preview(tmp_path / "t.xpt", "trial.xpt")["text"].startswith("4 rows x 2 columns")
    assert opened["format"] == "xport"

    _Sas7bdat.inferred_encoding = "cp1251"
    _Sas7bdat.read = lambda self, rows: pd.DataFrame({"city": [b"\xcc\xee\xf1\xea\xe2\xe0"]})
    assert build_preview(tmp_path / "a.sas7bdat", "airline.sas7bdat")["text"].endswith(
        "\n  \u041c\u043e\u0441\u043a\u0432\u0430"
    )

    # Below the 288 bytes already read, the next read runs to the end and the claim states nothing.
    monkeypatch.setattr(previews, "MAX_SAMPLED_BYTES", 100)
    assert sas7bdat(287, 4096, pad = b"\x00" * 712) == (
        "1000 bytes to read through before the first row"
    )
    # A value cut mid-character is not a single-byte encoding; a real one reads once UTF-8 fails.
    assert text.endswith(
        "\n  1948\tcaf\u00e9\n  1949\ta" + "\u00e9" * 79 + "...\n  1950\t" + "\u00e9" * 80 + "..."
    )
    monkeypatch.undo()


def test_table_previews_stop_at_their_budgets(tmp_path, monkeypatch):
    import zipfile

    import numpy as np
    import pandas as pd

    from core import chat_attachment_preview as previews

    _tables(tmp_path)

    def preview(name):
        return previews.build_preview(tmp_path / name, name)["text"]

    # Parquet decodes a page at a time, checked in every row group the sample reaches.
    monkeypatch.setattr(previews, "MAX_SAMPLED_BYTES", 10)
    text = preview("w.parquet")
    assert (
        text.startswith("7 rows x 2 columns\n") and "[First rows not read: a row group of " in text
    )
    monkeypatch.undo()

    import pyarrow as pa
    import pyarrow.parquet as pq

    pq.write_table(pa.table({"s": ["x", "z" * 400]}), tmp_path / "two.parquet", row_group_size = 1)
    monkeypatch.setattr(previews, "MAX_SAMPLED_BYTES", 300)
    assert "[First rows not read: a row group of " in (preview("two.parquet"))
    # The group is the bound, not the file: 267 bytes under the cap in a file over it.
    assert (tmp_path / "w.parquet").stat().st_size > 300
    assert "first 5 rows:\n  Oslo\t1.5" in (preview("w.parquet"))
    monkeypatch.undo()

    # Labels are decoded with the first row, so the map's size gates them, the codes surviving.
    def dta(
        name,
        frame,
        version = 118,
        **kwargs,
    ):
        frame.to_stata(tmp_path / name, version = version, write_index = False, **kwargs)
        return tmp_path / name

    for version, name in ((118, "coded.dta"), (114, "old.dta")):
        dta(name, pd.DataFrame({"c": pd.Categorical(["yes", "no"])}), version = version)
        assert preview(name).endswith("\n  yes\n  no"), name
        monkeypatch.setattr(previews, "MAX_SAMPLED_BYTES", 40)
        text = preview(name)
        assert text.endswith(" bytes to read through]") and "\n  1\n  0\n" in text, name
        monkeypatch.undo()
    # The stated table gates them, not the file: over the cap, its stated table under it.
    monkeypatch.setattr(previews, "MAX_SAMPLED_BYTES", 1000)
    assert (tmp_path / "coded.dta").stat().st_size > 1000
    assert preview("coded.dta").endswith("\n  yes\n  no")
    monkeypatch.undo()

    # Stata decodes its long-string table whole on the first row, so its map's size gates it.
    dta("long.dta", pd.DataFrame({"s": ["x" * 2100]}))
    monkeypatch.setattr(previews, "MAX_SAMPLED_BYTES", 1000)
    assert preview("long.dta") == ("long-string table of 2136 bytes, too large to read")

    # Fixed-width strings state 15 bytes of long strings, so a file over the cap still samples.
    dta("fixed.dta", pd.DataFrame({"s": ["y" * 244] * 40}))
    assert (tmp_path / "fixed.dta").stat().st_size > 1000
    assert preview("fixed.dta").startswith("unknown rows x 1 columns")
    # A Stata file states the byte order its map is written in, and MSF is not the host's.
    dta("msf.dta", pd.DataFrame({"n": [1, 2]}), byteorder = ">")
    assert b"<byteorder>MSF" in (tmp_path / "msf.dta").read_bytes()[:200]
    assert preview("msf.dta").startswith("unknown rows x 1 columns")
    # A label and a value hold the same tags, so neither the first <map> nor the last is the real one.
    dta(
        "spoof.dta",
        pd.DataFrame({"s": ["<map>" + "x" * 2100]}),
        data_label = "<byteorder>MSF<map>" + "x" * 60,
    )
    raw = (tmp_path / "spoof.dta").read_bytes()
    assert raw.count(b"<map>") == 3 and b"<byteorder>MSF" in raw[:200]
    assert preview("spoof.dta") == ("long-string table of 2141 bytes, too large to read")
    # pandas skips those tags unchecked, so a header this cannot follow states nothing.
    for tag in (b"</header><map>", b"<stata_dta>"):
        (tmp_path / "odd.dta").write_bytes(raw.replace(tag, tag.upper(), 1))
        text = preview("odd.dta")
        assert text.startswith("header in no known layout, and "), tag
    # A release opening with its own number has no long-string table, so it samples over the cap.
    dta("plain.dta", pd.DataFrame({"s": ["y" * 200] * 20}), version = 114)
    assert (tmp_path / "plain.dta").read_bytes()[0] == 114
    assert (tmp_path / "plain.dta").stat().st_size > 1000
    assert "first 5 rows:" in preview("plain.dta")
    monkeypatch.undo()

    # ORC and Arrow materialise a whole stripe or batch, so size and stated rows both gate it.
    for cap, value in (("MAX_SAMPLED_CELLS", 3), ("MAX_SAMPLED_BYTES", 10)):
        monkeypatch.setattr(previews, cap, value)
        for name in ("w.orc", "w.feather"):
            text = preview(name)
            assert "columns: city (" in text and "[First rows not read: " in text, (name, cap)
        monkeypatch.undo()

    # A column of lists states one cell a row: 500 of them take 200 MB in Feather, 492 MB in ORC.
    import pyarrow.feather as feather
    import pyarrow.orc as orc

    nested = pa.table(
        {"xs": pa.array([[0] * 400] * 4, type = pa.list_(pa.int64())), "n": pa.array([1, 2, 3, 4])}
    )
    orc.write_table(nested, str(tmp_path / "n.orc"))
    feather.write_feather(nested, tmp_path / "n.feather")
    for name in ("n.orc", "n.feather"):
        text = preview(name)
        assert (
            text.startswith("4 rows x 2 columns") and "[First rows not read: 4 rows]" in text
        ), name

    # Compression hides a batch's size, so the shape the footer states refuses it instead.
    wide = pa.table({f"c{index}": pa.array([0.0] * 5000) for index in range(4)})
    feather.write_feather(wide, tmp_path / "z.feather", compression = "zstd")
    monkeypatch.setattr(previews, "MAX_SAMPLED_CELLS", 1000)
    text = preview("z.feather")
    assert text.startswith("5000 rows x 4 columns") and "[First rows not read: 5000 rows]" in text
    monkeypatch.undo()

    # Feather v1 predates the IPC reader; an IPC stream has no footer, and is left to the tool.
    feather.write_feather(pa.table({"n": [1, 2]}), tmp_path / "v1.feather", version = 1)
    assert preview("v1.feather") == (
        "2 rows x 1 columns\ncolumns: n (int64)\nfirst 2 rows:\n  1\n  2"
    )
    # Nothing states its schema separately, so a large one is left alone entirely.
    monkeypatch.setattr(previews, "MAX_SAMPLED_BYTES", 10)
    assert previews.build_preview(tmp_path / "v1.feather", "v1.feather") is None
    monkeypatch.undo()
    with pa.OSFile(str(tmp_path / "s.arrow"), "wb") as sink:
        with pa.ipc.new_stream(sink, wide.schema) as out:
            out.write_table(wide)
    assert previews.build_preview(tmp_path / "s.arrow", "s.arrow") is None

    np.save(tmp_path / "big.npy", np.arange(10_000.0))
    np.savez(tmp_path / "mixed.npz", wide = np.arange(10_000.0), small = np.zeros(3, np.int8))
    monkeypatch.setattr(previews, "MAX_SAMPLED_BYTES", 1000)
    assert preview("big.npy") == ("array: shape (10000,), dtype float64")
    assert preview("mixed.npz") == (
        "2 arrays\nwide: shape (10000,), dtype float64\n"
        "small: shape (3,), dtype int8, min 0, max 0, mean 0, values [0, 0, 0]"
    )
    monkeypatch.undo()
    # The version decides the encoding: 3.0 is UTF-8, written for a name outside Latin-1.
    monkeypatch.setattr(previews, "MAX_SAMPLED_BYTES", 10)
    for field, version in (("caf\u00e9", b"\x01\x00"), ("\u6e29\u5ea6", b"\x03\x00")):
        np.save(tmp_path / f"{field}.npy", np.zeros(4000, dtype = [(field, "f8")]))
        assert (tmp_path / f"{field}.npy").read_bytes()[6:8] == version, field
        assert preview(f"{field}.npy") == (
            f"array: shape (4000,), dtype [('{field}', '<f8')]"
        ), field
    monkeypatch.undo()

    # A structured dtype of a thousand fields passes numpy's 10,000-byte refusal in a 21 KB file.
    np.save(tmp_path / "wide.npy", np.zeros(1, dtype = [(f"f{i}", "f4") for i in range(1000)]))
    np.savez(tmp_path / "wide.npz", w = np.zeros(1, dtype = [(f"f{i}", "f4") for i in range(1000)]))
    for name in ("wide.npy", "wide.npz"):
        assert "[('f0', '<f4')" in preview(name), name

    with zipfile.ZipFile(tmp_path / "z.npz", "w", zipfile.ZIP_LZMA) as archive:
        archive.writestr("packed.npy", np.arange(3).tobytes())
    assert preview("z.npz") == ("1 arrays\npacked: packed.npy uses zip compression method 14")

    # Unchecked, the two bytes after a header the file cannot hold read as an empty one.
    (tmp_path / "b.safetensors").write_bytes(struct.pack("<Q", 4 << 30) + b"{}")
    assert preview("b.safetensors") == ("header of 4294967296 bytes, too large to read")
    (tmp_path / "b.npy").write_bytes(b"\x93NUMPY\x02\x00" + struct.pack("<I", 1 << 30) + b"{}")
    monkeypatch.setattr(previews, "MAX_SAMPLED_BYTES", 10)
    assert preview("b.npy") == ("array: header of 1073741824 bytes, too large to read")
    monkeypatch.undo()

    # A long cell is cut where the row is built, not where the outline is clipped.
    long_frame = pd.DataFrame({"note": ["z" * 200]})
    long_frame.to_parquet(tmp_path / "long.parquet")
    text = preview("long.parquet")
    assert text.endswith("\n  " + "z" * 80 + "...")


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


def _models(tmp_path):
    import matplotlib

    database = sqlite3.connect(tmp_path / "shelf.sqlite")
    database.executescript(  # the view is a query no preview may run: it counts to 200 million
        'create table "my books" (id integer, title text);'
        "insert into \"my books\" values (1, 'Solaris'), (2, 'Ubik');"
        "create view slow as with recursive c(x) as (select 1 union all select x + 1 from c "
        "where x < 200000000) select count(*) from c"
    )
    database.close()
    (tmp_path / "b.stl").write_bytes(b"solid".ljust(80, b"\0") + struct.pack("<I", 2) + bytes(100))
    (tmp_path / "b.ply").write_bytes(
        b"ply\ncomment end_header?\nelement vertex 8\nend_header\n" + b"1 2 3\n" * 400
    )
    meshes = ", ".join('{"name": "m%d"}' % index for index in range(13))
    scene = ('{"asset": {"generator": "mk"}, "scenes": {"n": 0}, "meshes": [%s]}' % meshes).encode()
    head = struct.pack("<4sII", b"glTF", 2, 0) + struct.pack("<I4s", len(scene), b"JSON")
    (tmp_path / "c.glb").write_bytes(head + scene)
    padded = "<model><metadata>Plate</metadata><object/><!--" + "p" * 1500 + "--></model>"
    rels = '<R><Relationship Type="t/thumbnail" Target="/3D/3dmodel.model"/><Relationship '
    rels += 'Type="t/3dmodel" Target="/3D/p.model"/><!--' + "r" * 1500 + "--></R>"
    # One archive answers for both container formats, so its listing carries members of each.
    with zipfile.ZipFile(tmp_path / "p.3mf", "w") as archive:
        archive.writestr("_rels/.rels", rels)
        archive.writestr("3D/3dmodel.model", padded.replace("Plate", "Decoy"))
        archive.writestr("3D/p.model", padded)
        kml = padded.replace("model>", "kml>")
        archive.writestr("a.kml", kml.replace("Plate", "Decoy"))
        archive.writestr("doc.kml", kml.replace("Plate", "Trail"))
    font = pathlib.Path(matplotlib.get_data_path()) / "fonts/ttf/DejaVuSansMono.ttf"
    (tmp_path / "s.ttf").write_bytes(font.read_bytes())


def test_previews_outline_databases_models_and_fonts(tmp_path):
    from core import chat_attachment_preview as previews
    from core.chat_attachment_preview import build_preview

    _models(tmp_path)
    expected = {
        "shelf.sqlite": "1 tables:\n  my books\nmy books (2 rows): id INTEGER, title TEXT"
        "\n  1\tSolaris\n  2\tUbik",
        "b.stl": "binary STL, 2 triangles, header 'solid'",
        "b.ply": "PLY header:\nply\ncomment end_header?\nelement vertex 8\nend_header",
        "c.glb": "glTF 2 binary, 13 meshes\ngenerator: mk\nmeshes: "
        "m0, m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, m11, ... 1 more",
        "p.3mf": "3D/p.model: 1 model, 1 metadata, 1 object\nnamed: Plate\n5 entries:"
        "\n  _rels/.rels (1628 bytes)\n  3D/3dmodel.model (1557 bytes)\n  3D/p.model (1557 bytes)"
        "\n  a.kml (1553 bytes)\n  doc.kml (1553 bytes)",
        "s.ttf": "DejaVu Sans Mono Book, 3369 glyphs, 340240 bytes, monospaced",
    }
    if not previews.SQLITE_MEASURES_UNREAD:  # an older SQLite reads no rows to sample them
        expected.pop("shelf.sqlite")
    for index, (name, text) in enumerate(expected.items()):
        stored = tmp_path / f"9e4{index}"
        stored.write_bytes((tmp_path / name).read_bytes())
        assert build_preview(stored, name)["text"] == text, name


def test_model_previews_stop_at_their_budgets(tmp_path, monkeypatch):
    from core import chat_attachment_preview as previews

    def preview(name):
        return previews.build_preview(tmp_path / name, name)["text"]

    _models(tmp_path)
    database = sqlite3.connect(tmp_path / "shelf.sqlite")
    rows = "(3, '%s'), (4, 'x'), (5, 'x'), (6, 'x')" % ("\u00e9" * 751)
    database.executescript(
        "create virtual table notes using fts5(body);" 'insert into "my books" values ' + rows
    )
    database.close()
    (tmp_path / "claim.stl").write_bytes(b"solid".ljust(80, b"\0") + struct.pack("<I", 100_000_000))
    (tmp_path / "up.stl").write_bytes(b"SOLID " + b"x" * 100 + b"\nfacet normal 0 0 1\n" * 200)
    bomb = struct.pack("<4sII", b"glTF", 2, 12) + struct.pack("<I4s", 3 << 30, b"JSON")
    (tmp_path / "bomb.glb").write_bytes(bomb)
    # octet_length measures from the record: 751 two-byte characters are 1502 bytes.
    if previews.SQLITE_MEASURES_UNREAD:
        sampled = preview("shelf.sqlite")
        assert "\n  3\t<text of 1502 bytes>" in sampled and "\n  6\t" not in sampled
        assert "\nnotes: a virtual table, not read" in sampled
    # 100 million triangles are not 84 bytes, and a GLB states the size of the scene it opens with.
    assert preview("claim.stl") == "STL of 84 bytes, in neither layout"
    assert preview("up.stl") == "ASCII STL, 'SOLID %s'" % ("x" * 100)
    assert preview("bomb.glb") == "glTF 2 binary, scene of 3221225472 bytes, too large to read"
    # A KMZ declares no start part, so the conventional name answers for it.
    assert previews.build_preview(tmp_path / "p.3mf", "t.kmz")["text"].startswith("doc.kml: 1 kml")
    monkeypatch.setattr(previews, "MAX_XML_BYTES", 10)
    assert preview("p.3mf").startswith("[_rels/.rels not read: 1628 bytes of XML]\n[3D/3dmodel")
    # Without octet_length, sizing a cell would read it, so no row is read at all.
    monkeypatch.setattr(previews, "SQLITE_MEASURES_UNREAD", False)
    assert "[First rows not read: SQLite " in preview("shelf.sqlite")
    monkeypatch.setattr(previews, "MAX_OUTLINE_CHARS", 270)
    assert preview("shelf.sqlite").endswith("\n... 5 more tables, not detailed")
    monkeypatch.setattr(previews, "MAX_SAMPLED_BYTES", 10)
    assert preview("s.ttf") == "font of 340240 bytes, too large to read"
    assert preview("shelf.sqlite").endswith("nothing here bounds what its schema costs")
    with zipfile.ZipFile(tmp_path / "bad.3mf", "w") as archive:
        archive.writestr("_rels/.rels", "<Relations/>")
        archive.writestr("3D/3dmodel.model", "<not-xml")
    assert preview("bad.3mf").startswith("[_rels/.rels not read: 12 bytes of XML]")
    (tmp_path / "pad.ply").write_bytes(b"ply\ncomment " + b" " * 2100 + b"\nend_header\n")
    (tmp_path / "pad.stl").write_bytes(b"solid part" + b" " * 2100 + b"name\nfacet\n")
    assert preview("pad.ply").endswith("[Truncated: the header runs past what an outline shows]")
    assert preview("pad.stl").endswith("[Truncated: the name runs past what an outline shows]")
    big = "create table t (v text default '%s')" % ("x" * (20 << 20))
    sqlite3.connect(tmp_path / "big.sqlite").executescript(big).connection.close()
    spawned = previews.preview_attachment(tmp_path / "big.sqlite", "b.sqlite")["text"]
    assert spawned.endswith(f"more than {previews.MAX_SQLITE_BYTES} bytes to read")
