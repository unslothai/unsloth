# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Real document bytes through browser/native persistence, parsing and indexing."""

import base64
import codecs
from pathlib import Path

import pytest
from fastapi import HTTPException, UploadFile

from core.rag import captioner, config, ingestion, parsers, store
from routes.rag import _resolve_document_upload
from storage import rag_db
from .test_rag_native_drop_upload import SECRET, _sign
import utils.native_path_leases as leases

EXTENSIONS = (".pdf", ".txt", ".md", ".markdown", ".docx", ".html", ".htm")
TEXT_EXTENSIONS = (".txt", ".md", ".markdown", ".html", ".htm")


def write_document(path, *, large = False):
    text = "Revenue increased substantially. Café operations expanded."
    paragraphs = [text]
    if large:
        paragraphs += [f"Quarter {i}: revenue increased across all markets." for i in range(400)]
    paragraphs.append("Zebramarker concludes the report.")
    if path.suffix == ".pdf":
        pymupdf = pytest.importorskip("pymupdf")
        with pymupdf.open() as doc:
            for i in range(0, len(paragraphs), 30):
                page = doc.new_page()
                assert (
                    page.insert_textbox(
                        pymupdf.Rect(40, 40, 550, 800),
                        "\n".join(paragraphs[i : i + 30]),
                        fontsize = 10,
                    )
                    >= 0
                )
            doc.save(path)
    elif path.suffix == ".docx":
        docx = pytest.importorskip("docx")
        doc = docx.Document()
        for paragraph in paragraphs[:-1]:
            doc.add_paragraph(paragraph)
        table = doc.add_table(rows = 1, cols = 2)
        table.cell(0, 0).text = "Conclusion"
        table.cell(0, 1).text = paragraphs[-1]
        doc.save(path)
    elif path.suffix in (".html", ".htm"):
        path.write_text(
            "<html><style>hiddenstylemarker</style><script>hiddenscriptmarker</script>"
            + "".join(f"<p>{p}</p>" for p in paragraphs)
            + "</html>",
            encoding = "utf-8",
        )
    else:
        path.write_text("\n\n".join(paragraphs), encoding = "utf-8")
    return path


@pytest.fixture
def persist_document(monkeypatch):
    monkeypatch.setenv(
        leases.LEASE_SECRET_ENV, base64.urlsafe_b64encode(SECRET).decode("ascii").rstrip("=")
    )
    monkeypatch.setattr(leases, "_CACHED_LEASE_SECRET", None)

    def persist(path, transport):
        if transport == "desktop":
            return _resolve_document_upload(None, _sign(path))
        with path.open("rb") as handle:
            return _resolve_document_upload(UploadFile(file = handle, filename = path.name), None)

    return persist


@pytest.mark.parametrize("extension", EXTENSIONS)
@pytest.mark.parametrize("transport", ["browser", "desktop"])
def test_formats_finish_and_remain_searchable(
    rag_home, stub_embeddings, tmp_path, monkeypatch, persist_document, extension, transport
):
    assert set(EXTENSIONS) == config.UPLOAD_EXTS
    source = write_document(tmp_path / f"report{extension}", large = True)
    stored, filename = persist_document(source, transport)
    assert Path(stored).read_bytes() == source.read_bytes()
    vision_calls = []
    monkeypatch.setattr(captioner, "vision_endpoint", lambda: ("http://local", "local"))
    monkeypatch.setattr(captioner, "_caption_one", lambda *args: vision_calls.append("caption"))
    monkeypatch.setattr(captioner, "_ocr_one", lambda *args: vision_calls.append("ocr"))
    scope = store.thread_scope("formats")
    doc_id, job = ingestion.start_ingestion(
        scope, None, "formats", filename, stored, caption = True, ocr = True, background = False
    )
    events = list(ingestion.job_events(job))
    assert events[-1]["type"] == "complete"
    progress = [e["progress"] for e in events if e["type"] == "progress"]
    assert progress == sorted(progress)
    assert not vision_calls  # Text-only fixtures need no vision.
    conn = rag_db.get_connection()
    try:
        doc = store.get_document(conn, doc_id)
        assert doc["status"] == "completed" and doc["num_chunks"] > 1
        assert store.search_lexical(conn, scope, "revenue", 5)
        assert store.search_lexical(
            conn, scope, "zebramarker", 5
        )  # Preserve the final paragraph or table cell.
        assert not store.search_lexical(conn, scope, "hiddenscriptmarker", 5)
    finally:
        conn.close()
    duplicate_path, filename = persist_document(source, transport)
    duplicate_id, duplicate_job = ingestion.start_ingestion(
        scope, None, "formats", filename, duplicate_path, background = False
    )
    assert duplicate_id == doc_id
    assert ingestion.get_job_status(duplicate_job)["status"] == "completed"
    assert Path(stored).exists() and source.exists()


@pytest.mark.parametrize("extension", TEXT_EXTENSIONS)
@pytest.mark.parametrize(
    "encoding,bom",
    [
        ("utf-8", codecs.BOM_UTF8),
        ("utf-16-le", codecs.BOM_UTF16_LE),
        ("utf-16-be", codecs.BOM_UTF16_BE),
        ("utf-32-le", codecs.BOM_UTF32_LE),
        ("utf-32-be", codecs.BOM_UTF32_BE),
    ],
)
def test_unicode_text_upload_is_decoded_before_indexing(
    rag_home, stub_embeddings, tmp_path, extension, encoding, bom
):
    text = "Café revenue increased. 東京 operations expanded. Zebramarker concludes the report."
    raw = f"<p>{text}</p>" if extension in (".html", ".htm") else text
    path = tmp_path / f"unicode{extension}"
    path.write_bytes(bom + raw.encode(encoding))
    pages = parsers.parse(str(path))
    assert pages[0].text == text
    scope = store.thread_scope("unicode")
    _, job = ingestion.start_ingestion(
        scope, None, "unicode", path.name, str(path), background = False
    )
    assert ingestion.get_job_status(job)["status"] == "completed"
    conn = rag_db.get_connection()
    try:
        assert store.search_lexical(conn, scope, "zebramarker", 5)
    finally:
        conn.close()


@pytest.mark.parametrize("extension", EXTENSIONS)
@pytest.mark.parametrize("transport", ["browser", "desktop"])
def test_zero_byte_formats_are_rejected_before_indexing(
    rag_home, tmp_path, persist_document, extension, transport
):
    path = tmp_path / f"empty{extension}"
    path.write_bytes(b"")
    with pytest.raises(HTTPException) as error:
        persist_document(path, transport)
    assert error.value.status_code == 400


@pytest.mark.parametrize("extension", [".pdf", ".docx"])
def test_corrupt_documents_fail_then_valid_upload_succeeds(
    rag_home, stub_embeddings, tmp_path, extension
):
    path = tmp_path / f"broken{extension}"
    path.write_bytes(b"not a document archive")
    scope = store.thread_scope("corrupt")
    _, job = ingestion.start_ingestion(
        scope, None, "corrupt", path.name, str(path), background = False
    )
    assert ingestion.get_job_status(job)["status"] == "failed"
    assert list(ingestion.job_events(job))[-1]["type"] == "error"
    write_document(path)
    _, retry = ingestion.start_ingestion(
        scope, None, "corrupt", path.name, str(path), background = False
    )
    assert ingestion.get_job_status(retry)["status"] == "completed"
