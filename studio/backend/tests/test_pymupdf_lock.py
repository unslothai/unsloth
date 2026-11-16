# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Concurrent PDF parse must serialize on PYMUPDF_LOCK (#17981)."""

from __future__ import annotations

import threading
import time
import unittest
from email.message import Message
from unittest import mock

import pytest

from core.inference import tools
from core.rag import locators, parsers, pdf_ocr


class PyMuPdfLockTests(unittest.TestCase):
    def test_pdf_parse_serializes_across_threads(self):
        active = 0
        max_active = 0
        counter = threading.Lock()

        def fake_unlocked(
            source,
            want_images,
            max_pages = None,
        ):
            nonlocal active, max_active
            with counter:
                active += 1
                max_active = max(max_active, active)
            time.sleep(0.08)
            with counter:
                active -= 1
            return [], [], 0

        errors: list[BaseException] = []

        def worker():
            try:
                parsers._pdf(b"%PDF", want_images = False)
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        with mock.patch.object(parsers, "_pdf_unlocked", fake_unlocked):
            threads = [threading.Thread(target = worker) for _ in range(2)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout = 5)
                self.assertFalse(thread.is_alive())

        self.assertEqual(errors, [])
        self.assertEqual(max_active, 1)

    def test_pymupdf_lock_is_reentrant(self):
        with parsers.PYMUPDF_LOCK:
            with parsers.PYMUPDF_LOCK:
                held = True
        self.assertTrue(held)


def _one_page_pdf(tmp_path) -> str:
    pymupdf = pytest.importorskip("pymupdf")
    doc = pymupdf.open()
    doc.new_page().insert_text((50, 72), "lock probe text for the locator pass")
    path = str(tmp_path / "probe.pdf")
    doc.save(path)
    doc.close()
    return path


class _Chunk:
    source_page_index = 0
    page_char_start = 0
    page_char_end = 20
    page_number = 1


_ENTRY_POINTS = {
    "parse": lambda path: parsers.parse(path),
    "mupdf_document_text": lambda path: parsers.mupdf_document_text(path),
    "pages_with_figures": lambda path: parsers.pages_with_figures(path),
    "render_pdf_figure_tiles": lambda path: parsers.render_pdf_figure_tiles(path, [1]),
    "render_pdf_pages": lambda path: parsers.render_pdf_pages(path, [1]),
    "pdf_regions_for_chunks": lambda path: locators.pdf_regions_for_chunks(
        path, [parsers.Page(text = "lock probe text for the locator pass", page_number = 1)], [_Chunk()]
    ),
    "ocr_pages": lambda path: pdf_ocr.ocr_pages(path, [1]),
}


@pytest.mark.parametrize("name", sorted(_ENTRY_POINTS))
def test_entry_point_waits_for_pymupdf_lock(tmp_path, monkeypatch, name):
    pymupdf = pytest.importorskip("pymupdf")
    import fitz

    path = _one_page_pdf(tmp_path)
    opened = threading.Event()
    real_open = pymupdf.open

    def tracking_open(*args, **kwargs):
        opened.set()
        return real_open(*args, **kwargs)

    monkeypatch.setattr(pymupdf, "open", tracking_open)
    monkeypatch.setattr(fitz, "open", tracking_open)
    errors: list[BaseException] = []

    def worker():
        try:
            _ENTRY_POINTS[name](path)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    thread = threading.Thread(target = worker)
    with parsers.PYMUPDF_LOCK:
        thread.start()
        assert not opened.wait(0.3), f"{name} opened a document while another thread held the lock"
    thread.join(timeout = 10)
    assert not thread.is_alive()
    assert opened.is_set()
    assert errors == []


def test_web_pdf_fetch_cancels_while_waiting_for_lock(monkeypatch):
    pymupdf = pytest.importorskip("pymupdf")
    doc = pymupdf.open()
    doc.new_page().insert_text((50, 72), "web pdf body")
    body = doc.tobytes()
    doc.close()

    class Resp:
        def __init__(self):
            self.headers = Message()
            self.headers["Content-Type"] = "application/pdf"
            self._pos = 0

        def read(self, n = None):
            chunk = body[self._pos :] if n is None else body[self._pos : self._pos + n]
            self._pos += len(chunk)
            return chunk

    class Opener:
        def open(
            self,
            req,
            timeout = None,
        ):
            return Resp()

    monkeypatch.setattr(
        tools, "_validate_and_resolve_host", lambda host, port: (True, "", ["93.184.216.34"])
    )
    monkeypatch.setattr(tools.urllib.request, "build_opener", lambda *a, **k: Opener())
    cancel = threading.Event()
    result: list[str] = []
    thread = threading.Thread(
        target = lambda: result.append(
            tools._fetch_page_text("https://example.com/a.pdf", timeout = 5, cancel_event = cancel)
        )
    )
    with parsers.PYMUPDF_LOCK:
        thread.start()
        time.sleep(0.3)
        assert thread.is_alive()
        cancel.set()
        thread.join(timeout = 5)
        assert not thread.is_alive()
    assert "cancelled" in result[0]


if __name__ == "__main__":
    unittest.main()
