# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""PDFs indexed at the same time take turns in PyMuPDF, which crashes the process when two threads use it."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
import threading
from pathlib import Path

import pytest

pymupdf = pytest.importorskip("pymupdf")

from core.rag import folder_sync
from storage import rag_db

requires_sqlite_vec = pytest.mark.skipif(
    not rag_db.RAG_AVAILABLE, reason = "sqlite-vec is not installed"
)


def _pdf(
    path,
    *,
    pages: int,
    damaged: bool = False,
):
    doc = pymupdf.open()
    for number in range(pages):
        page = doc.new_page()
        page.insert_text((50, 40), f"Report {path.stem} page {number + 1}", fontname = "helv")
        if not damaged:
            continue
        # A number as a dictionary key and a garbage colour profile: the two MuPDF errors in #17981.
        content = []
        for column in range(4):
            pixmap = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 16, 16))
            pixmap.clear_with((number * 4 + column) % 256)
            image = page.insert_image(pymupdf.Rect(0, 0, 16, 16), pixmap = pixmap)
            profile = doc.get_new_xref()
            doc.update_object(profile, "<< /N 3 /Alternate /DeviceRGB >>")
            doc.update_stream(profile, bytes(range(256)))
            doc.xref_set_key(image, "ColorSpace", f"[/ICCBased {profile} 0 R]")
            name = page.get_images()[-1][7]
            content.append(f"q 120 0 0 120 {40 + 135 * column} 600 cm /{name} Do Q")
        for line in range(30):
            content.append(
                f"/Span << 12 (oops) /MCID {line} >> BDC BT /helv 9 Tf 50 {560 - 14 * line} Td"
                f" (Page {number + 1} line {line}: alpha beta gamma delta engine valve) Tj ET EMC"
            )
        doc.update_stream(page.get_contents()[0], "\n".join(content).encode())
    doc.save(str(path))
    doc.close()
    return path


_INDEX_PDFS_ON_THREADS = textwrap.dedent(
    """
    import sys
    import threading

    from core.rag import locators, parsers, pdf_ocr
    from core.rag.chunking import chunk_pages


    def index(path):
        pages = parsers.parse(path)
        with open(path, "rb") as f:
            parsers.parse_pdf_bytes(f.read(), max_pages = 5)
        parsers.render_pdf_pages(path, [1], dpi = 36)
        parsers.render_pdf_figure_tiles(path, [1], dpi = 36)
        parsers.pages_with_figures(path)
        pdf_ocr.ocr_pages(path, [1])
        chunks = chunk_pages(pages, max_tokens = 64, overlap = 8, count = lambda t: len(t.split()))
        locators.pdf_regions_for_chunks(path, pages, chunks)


    threads = [threading.Thread(target = index, args = (path,)) for path in sys.argv[1:]]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    print("INDEXED")
    """
)


def test_pdfs_with_mupdf_errors_indexed_together_do_not_crash_the_process(tmp_path):
    paths = [str(_pdf(tmp_path / f"damaged-{i}.pdf", pages = 20, damaged = True)) for i in range(4)]
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1])}
    proc = subprocess.run(
        [sys.executable, "-c", _INDEX_PDFS_ON_THREADS, *paths],
        capture_output = True,
        text = True,
        env = env,
        timeout = 300,
    )
    assert "MuPDF error" in proc.stdout + proc.stderr
    assert proc.returncode == 0, f"exit {proc.returncode}: {proc.stderr[-2000:]}"
    assert "INDEXED" in proc.stdout


@requires_sqlite_vec
def test_linked_folder_workers_never_have_two_pdfs_open_at_once(
    rag_home, stub_embeddings, monkeypatch
):
    import fitz

    source = rag_home / "source"
    source.mkdir()
    for i in range(3):
        _pdf(source / f"manual-{i}.pdf", pages = 2)

    guard = threading.Lock()
    opened: list[int] = []
    second_open = threading.Event()
    real_open = pymupdf.open

    def tracked_open(*args, **kwargs):
        doc = real_open(*args, **kwargs)
        real_close = doc.close

        def close():
            with guard:
                opened.pop()
            real_close()

        doc.close = close
        with guard:
            opened.append(1)
            together = len(opened)
        if together > 1:
            second_open.set()
        else:
            second_open.wait(0.5)
        return doc

    monkeypatch.setattr(pymupdf, "open", tracked_open)
    monkeypatch.setattr(fitz, "open", tracked_open)

    folder = folder_sync.create_folder(
        scope_type = "knowledge_base", scope_id = "kb-1", path = str(source), name = "Manuals"
    )
    job_id = folder_sync.request_sync(folder["id"])
    folder_sync.reconcile_folder(job_id)

    job = folder_sync.get_job(job_id)
    assert (job["status"], job["added"]) == ("completed", 3)
    assert not second_open.is_set()
