# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
"""Null-password PDFs (Acrobat-distilled /Encrypt dicts, the Orimi test
file) must not be rejected as encrypted by the preflight or the extractor.

Pinned regression: both paths checked ``is_encrypted`` instead of
``needs_pass`` and 422'd files that open without a password prompt.
"""

from __future__ import annotations

import pytest


def _make_pseudo_encrypted_pdf() -> bytes:
    """Mint a tiny PDF with an empty user password (Orimi-style)."""
    pymupdf = pytest.importorskip("pymupdf")
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text(
        (72, 100),
        "pseudo-encrypted PDF: null user password, opens without prompt",
        fontsize = 12,
    )
    out = doc.tobytes(
        encryption = pymupdf.PDF_ENCRYPT_AES_256,
        owner_pw = "owner-pw",
        user_pw = "",
    )
    doc.close()
    return out


def test_extract_pdf_accepts_null_password(monkeypatch):
    """No DocumentExtractionEncrypted for an empty-password PDF. PyMuPDF's
    ``needs_pass`` is the canonical signal; ``is_encrypted`` is too
    aggressive."""
    from core.chat import document_extractor as mod

    file_bytes = _make_pseudo_encrypted_pdf()

    md, figures, page_count, truncated, seen = mod._extract_pdf(
        file_bytes,
        max_figures = 0,
        use_vlm_ocr = False,
        max_visual_payloads = 0,
    )

    assert page_count == 1
    assert "pseudo-encrypted PDF" in md
    assert figures == []


def test_preflight_pdf_page_count_accepts_null_password():
    """_preflight_pdf_page_count must accept null-password PDFs."""
    from routes.inference import _preflight_pdf_page_count

    file_bytes = _make_pseudo_encrypted_pdf()
    n = _preflight_pdf_page_count(
        file_bytes,
        filename = "pseudo_encrypted.pdf",
        content_type = "application/pdf",
    )
    assert n == 1


def test_extract_pdf_still_rejects_password_required(monkeypatch):
    """A PDF that truly requires a password must still raise
    DocumentExtractionEncrypted."""
    pymupdf = pytest.importorskip("pymupdf")
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 100), "this one needs a password", fontsize = 12)
    encrypted = doc.tobytes(
        encryption = pymupdf.PDF_ENCRYPT_AES_256,
        owner_pw = "owner",
        user_pw = "real-password",
    )
    doc.close()

    from core.chat import document_extractor as mod

    with pytest.raises(mod.DocumentExtractionEncrypted):
        mod._extract_pdf(
            encrypted,
            max_figures = 0,
            use_vlm_ocr = False,
            max_visual_payloads = 0,
        )
