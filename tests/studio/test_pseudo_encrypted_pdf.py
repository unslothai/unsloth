# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
"""
Tests that PDFs with a null/empty user password (very common; Acrobat
distillation often writes /Encrypt dicts with no password) are NOT
falsely rejected as "encrypted" by either the preflight or the
extractor.

Failure mode the test pins:
    The classic Orimi PDF Test File (and many scanner-output PDFs)
    carry "Standard V2 R3 128-bit RC4" encryption with an empty user
    password -- the file opens without prompting in any reader.
    Pre-fix, both ``routes.inference._preflight_pdf_page_count`` and
    ``core.chat.document_extractor._extract_pdf`` returned HTTP 422
    "Encrypted PDFs are not supported" because they checked
    ``is_encrypted`` rather than ``needs_pass``. After the fix the
    file is accepted and its text is extracted.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest


_BACKEND = Path(__file__).resolve().parents[2] / "studio" / "backend"
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))


def _make_pseudo_encrypted_pdf() -> bytes:
    """Mint a tiny PDF with an empty user password (mirrors what
    Orimi's test file and many distiller pipelines produce)."""
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
    """The extractor must not raise DocumentExtractionEncrypted for a
    PDF whose user password is the empty string. PyMuPDF's
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
    """The pre-extraction preflight at
    ``routes.inference._preflight_pdf_page_count`` must accept
    null-password PDFs."""
    from routes.inference import _preflight_pdf_page_count

    file_bytes = _make_pseudo_encrypted_pdf()
    n = _preflight_pdf_page_count(
        file_bytes,
        filename = "pseudo_encrypted.pdf",
        content_type = "application/pdf",
    )
    assert n == 1


def test_extract_pdf_still_rejects_password_required(monkeypatch):
    """Sanity-check the other direction: a PDF that actually requires
    a non-empty user password must still raise
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
