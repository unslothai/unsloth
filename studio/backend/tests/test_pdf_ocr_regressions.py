# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import asyncio
from pathlib import Path
import time
import sys

import pymupdf
import pytest

from core.rag import captioner, chunking, config, ingestion, job_leases, parsers, pdf_ocr, store
from storage import rag_db
from .test_data_recipe_seed import _FakeUpload, _block_files, _load_seed_route
from .test_pdf_local_ocr import local_ocr, scanned_pdf
from .test_rag_ocr_fallback import _ingest


def test_short_digital_title_with_logo_needs_no_ocr(tmp_path):
    path = tmp_path / "logo.pdf"
    with pymupdf.open() as doc:
        page = doc.new_page()
        page.insert_text((40, 35), "Report")
        pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 20, 20))
        pix.clear_with(150)
        page.insert_image(pymupdf.Rect(40, 70, 60, 90), pixmap = pix)
        doc.save(path)
    page = parsers.parse(str(path))[0]
    assert not page.needs_ocr
    assert "Report" in page.text


def test_selectable_overlay_header_does_not_hide_scan(tmp_path):
    path = tmp_path / "overlay.pdf"
    scanned_pdf(path)
    with pymupdf.open(path) as doc:
        doc[0].insert_text((40, 75), "A selectable header inside the image rectangle")
        doc.saveIncr()
    assert parsers.parse(str(path))[0].needs_ocr


def test_blank_page_does_not_displace_scan_from_budget(
    rag_conn, stub_embeddings, tmp_path, monkeypatch, local_ocr
):
    path = tmp_path / "blank-first.pdf"
    scanned_pdf(path)
    with pymupdf.open(path) as doc:
        doc.new_page(pno = 0)
        doc.saveIncr()
    monkeypatch.setattr(config, "OCR_MAX_PAGES", 1)
    result = _ingest(rag_conn, "blank-first", path.name, path)
    assert result["status"] == "completed", result["error"]
    assert local_ocr == [2]


@pytest.mark.parametrize(
    "fullpage, cap, failed_image, expected",
    [
        (False, 1, None, "failed"),
        (False, 4, None, "completed"),
        (False, 4, 2, "failed"),
        (True, 1, None, "completed"),
        (True, 2, 1, "failed"),
    ],
)
def test_only_complete_caption_coverage_resolves_a_scan(
    rag_conn, stub_embeddings, tmp_path, monkeypatch, fullpage, cap, failed_image, expected
):
    path = tmp_path / "captions.pdf"
    scanned_pdf(path)
    monkeypatch.setattr(config, "OCR_SCANNED", False)
    monkeypatch.setattr(config, "CAPTION_IMAGES", True)
    monkeypatch.setattr(config, "FIGURE_FULLPAGE", fullpage)
    monkeypatch.setattr(config, "FIGURE_TILE_ROWS", 2)
    monkeypatch.setattr(config, "FIGURE_TILE_COLS", 2)
    monkeypatch.setattr(config, "CAPTION_MAX_IMAGES", cap)
    monkeypatch.setattr(captioner, "vision_endpoint", lambda: ("http://unused", "vision"))
    calls = []

    def caption(*args):
        calls.append(1)
        return None if len(calls) == failed_image else "A transcribed page region"

    monkeypatch.setattr(captioner, "_caption_one", caption)
    result = _ingest(rag_conn, "captions", path.name, path)
    assert result["status"] == expected, result["error"]
    if expected == "failed":
        assert "scanned PDF pages: 1" in result["error"]
        assert result["num_chunks"] == 0


@pytest.mark.parametrize("phase", ["ocr", "chunking"])
def test_empty_result_cannot_fail_job_after_lease_reclaim(
    rag_conn, stub_embeddings, tmp_path, monkeypatch, phase
):
    path = tmp_path / "reclaimed.pdf"
    scanned_pdf(path)
    monkeypatch.setattr(config, "CAPTION_IMAGES", False)
    monkeypatch.setattr(config, "OCR_SCANNED", True)
    monkeypatch.setattr(captioner, "vision_endpoint", lambda: None)

    def reclaim():
        conn = rag_db.get_connection()
        try:
            conn.execute(
                "UPDATE rag_job_leases SET owner_id=? WHERE job_id=?", ("successor", job_id)
            )
            conn.commit()
        finally:
            conn.close()

    def empty_ocr(*args):
        reclaim()
        return {}

    def empty_chunks(*args, **kwargs):
        reclaim()
        return []

    if phase == "ocr":
        monkeypatch.setattr(pdf_ocr, "ocr_pages", empty_ocr)
    else:
        monkeypatch.setattr(pdf_ocr, "ocr_pages", lambda *a: {1: "Readable page text"})
        monkeypatch.setattr(chunking, "chunk_pages", empty_chunks)
    scope = store.thread_scope("reclaim")
    doc_id = store.create_document(
        rag_conn,
        scope = scope,
        thread_id = "reclaim",
        filename = path.name,
        sha256 = phase,
        stored_path = str(path),
    )
    job_id = ingestion._new_job(rag_conn, doc_id, scope)
    events = []
    monkeypatch.setattr(ingestion, "_emit", lambda job, event: events.append(event))
    ingestion._run(job_id, doc_id, scope, str(path), None)
    assert store.get_document(rag_conn, doc_id)["status"] == "pending"
    assert ingestion.get_job_status(job_id)["status"] == "running"
    assert (
        rag_conn.execute(
            "SELECT owner_id FROM rag_job_leases WHERE job_id=?", (job_id,)
        ).fetchone()[0]
        == "successor"
    )
    assert any(event and event["type"] == "progress" for event in events)
    assert not any(event and event["type"] in {"error", "complete"} for event in events)


@pytest.mark.parametrize("module_launcher", [False, True])
def test_recipe_pdf_process_keeps_event_loop_responsive(monkeypatch, tmp_path, module_launcher):
    if module_launcher:
        monkeypatch.setattr(
            sys.modules["__main__"],
            "__file__",
            str(Path(__file__).resolve().parents[3] / "unsloth_cli/__main__.py"),
            raising = False,
        )
    route = _load_seed_route(monkeypatch, tmp_path, inline_extraction = False)
    path = tmp_path / "digital.pdf"
    with pymupdf.open() as doc:
        for _ in range(10):
            doc.new_page().insert_text((50, 100), "Worker process extracts readable text")
        doc.save(path)

    async def run():
        upload = asyncio.create_task(
            route.upload_unstructured_file(_FakeUpload(path.name, path.read_bytes()), "block")
        )
        ticks = 0
        while not upload.done():
            await asyncio.sleep(0.01)
            if not upload.done():
                ticks += 1
        return await upload, ticks

    result, ticks = asyncio.run(run())
    assert result.status == "ok", result.error
    assert ticks > 0
    assert (
        "Worker process extracts"
        in next(route.UNSTRUCTURED_UPLOAD_ROOT.rglob("*.extracted.txt")).read_text()
    )


def _slow_extraction(path, *args):
    Path(path).with_suffix(".started").write_text("worker started")
    time.sleep(60)
    return "late output"


def test_cancelled_pdf_upload_kills_worker_before_cleanup(monkeypatch, tmp_path):
    route = _load_seed_route(monkeypatch, tmp_path, inline_extraction = False)
    monkeypatch.setattr(pdf_ocr, "extract_text", _slow_extraction)

    async def run():
        upload = asyncio.create_task(
            route.upload_unstructured_file(_FakeUpload("slow.pdf", b"%PDF-1.7"), "block")
        )
        for _ in range(500):
            if list(route.UNSTRUCTURED_UPLOAD_ROOT.rglob("*.started")):
                break
            if upload.done():
                pytest.fail(f"worker did not start: {upload.result()}")
            await asyncio.sleep(0.01)
        else:
            upload.cancel()
            pytest.fail("worker did not signal start")
        upload.cancel()
        with pytest.raises(asyncio.CancelledError):
            await upload

    asyncio.run(run())
    assert all(name.endswith(".started") for name in _block_files(route))
