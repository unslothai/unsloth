# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Concurrent PDF parse must serialize on PYMUPDF_LOCK (#17981)."""

from __future__ import annotations

import threading
import time
import unittest
from unittest import mock

from core.rag import parsers


class PyMuPdfLockTests(unittest.TestCase):
    def test_pdf_parse_serializes_across_threads(self):
        active = 0
        max_active = 0
        counter = threading.Lock()

        def fake_unlocked(source, want_images, max_pages = None):
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
        # Web fetch holds the lock around parse_pdf_bytes, which locks again in _pdf.
        with parsers.PYMUPDF_LOCK:
            with parsers.PYMUPDF_LOCK:
                held = True
        self.assertTrue(held)


if __name__ == "__main__":
    unittest.main()
