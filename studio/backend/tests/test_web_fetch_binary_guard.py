# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Regression for unslothai/unsloth#7084: the web_search fetcher must not decode
a binary body (PDF, image, octet-stream) into a flood of U+FFFD replacement
chars that poison the model context. It rejects non-text Content-Types up front
and, for binary mislabeled as text/* or sent unlabeled, falls back to a
replacement-char ratio check. Real HTML pages are unaffected.
"""

from __future__ import annotations

import sys
from email.message import Message
from pathlib import Path

import pytest

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from core.inference import tools


class _FakeResp:
    def __init__(self, body: bytes, content_type: str | None):
        self._body = body
        self.headers = Message()
        if content_type is not None:
            self.headers["Content-Type"] = content_type

    def read(self, n: int | None = None) -> bytes:
        return self._body if n is None else self._body[:n]


class _FakeOpener:
    def __init__(self, resp):
        self._resp = resp

    def open(
        self,
        req,
        timeout = None,
    ):
        return self._resp


def _fetch_with(monkeypatch, body: bytes, content_type: str | None) -> str:
    # Pass SSRF validation and skip real DNS/network.
    monkeypatch.setattr(
        tools, "_validate_and_resolve_host", lambda host, port: (True, "", "93.184.216.34")
    )
    monkeypatch.setattr(
        tools.urllib.request,
        "build_opener",
        lambda *a, **k: _FakeOpener(_FakeResp(body, content_type)),
    )
    return tools._fetch_page_text("https://example.com/thing", timeout = 5)


# ── content-type classifier ──


@pytest.mark.parametrize(
    "content_type,expected",
    [
        ("text/html", True),
        ("text/plain; charset=utf-8", True),
        ("application/json", True),
        ("application/xml", True),
        ("application/xhtml+xml", True),
        ("application/ld+json", True),
        ("application/yaml", True),  # raw configs / API specs
        ("application/x-yaml", True),
        ("application/x-ndjson", True),  # newline-delimited JSON is text
        ("application/ndjson", True),
        ("application/pdf", False),
        ("image/png", False),
        ("image/svg+xml", False),  # SVG source isn't extracted downstream; reject
        ("application/octet-stream", False),
        ("application/zip", False),
        # A .docx ZIP must not pass just because "xml" is inside "openxmlformats".
        ("application/vnd.openxmlformats-officedocument.wordprocessingml.document", False),
        ("", True),  # unlabeled: defer to the binary-char fallback
        (None, True),
    ],
)
def test_is_texty_content_type(content_type, expected):
    assert tools._is_texty_content_type(content_type) is expected


# ── fetcher end-to-end (mocked network) ──


def test_pdf_rejected_by_content_type(monkeypatch):
    out = _fetch_with(monkeypatch, b"%PDF-1.7\n\xff\xd8\xff\x00\x89PNG" * 200, "application/pdf")
    assert "�" not in out
    assert "non-text content" in out and "application/pdf" in out


def test_image_rejected_by_content_type(monkeypatch):
    out = _fetch_with(monkeypatch, b"\x89PNG\r\n\x1a\n" + bytes(range(256)) * 4, "image/png")
    assert "�" not in out
    assert "non-text content" in out and "image/png" in out


def test_binary_mislabeled_as_text_caught_by_fallback(monkeypatch):
    # A server sends binary but labels it text/plain -> the type check passes,
    # so the binary-char fallback must catch it.
    out = _fetch_with(monkeypatch, bytes(range(256)) * 20, "text/plain")
    assert "�" not in out
    assert "binary content" in out


def test_valid_utf8_binary_caught_by_control_chars(monkeypatch):
    # NUL/control-heavy binary is valid UTF-8, so it decodes with no U+FFFD;
    # the fallback must still catch it via control-char density (codex #2).
    body = bytes([0, 1, 2, 3, 4, 5, 6, 7]) * 400  # all valid UTF-8, 0 replacement chars
    out = _fetch_with(monkeypatch, body, "text/plain")
    assert "binary content" in out


def test_pdf_mislabeled_as_text_caught_by_magic(monkeypatch):
    # A PDF served as text/plain: the content-type gate passes, so the magic-byte
    # sniff must catch it (its ASCII-heavy head can stay under the ratio).
    pdf = b"%PDF-1.7\n" + b"1 0 obj<</Type/Catalog>>endobj\n" * 300
    out = _fetch_with(monkeypatch, pdf, "text/plain")
    assert "binary content" in out


def test_zip_mislabeled_as_text_caught_by_magic(monkeypatch):
    out = _fetch_with(monkeypatch, b"PK\x03\x04" + b"filename.txt content " * 200, "text/plain")
    assert "binary content" in out


@pytest.mark.parametrize(
    "magic",
    [b"\x1f\x8b", b"BZh", b"\xfd7zXZ\x00", b"\x28\xb5\x2f\xfd"],  # gzip, bzip2, xz, zstd
)
def test_compression_mislabeled_as_text_caught_by_magic(monkeypatch, magic):
    # A small compressed body mislabeled as text can slip past the density check;
    # the signature must reject it.
    out = _fetch_with(monkeypatch, magic + b"some short compressed-looking body", "text/plain")
    assert "binary content" in out


def test_latin1_text_without_charset_kept(monkeypatch):
    # Accent-dense cp1252 text with no declared charset decodes to many U+FFFD as
    # UTF-8; the cp1252 retry must keep it instead of dropping it as binary.
    body = (
        "Muller lauft uber die Strasse: schoene, groesse. MARKERWORD ".replace("ue", "ü")
        + "äöüß éèà "
    ) * 30
    out = _fetch_with(monkeypatch, body.encode("cp1252"), "text/plain")
    assert "binary content" not in out
    assert "MARKERWORD" in out


def test_control_heavy_binary_survives_cp1252_retry(monkeypatch):
    # Genuine control-heavy binary (no magic) must stay rejected even after the
    # cp1252 retry, which cannot rescue real binary.
    out = _fetch_with(monkeypatch, bytes([0, 1, 2, 3, 4, 5, 6, 7]) * 400, "text/plain")
    assert "binary content" in out


def test_ansi_colored_text_log_kept(monkeypatch):
    # A text log with per-token ANSI color codes is text; ESC is excluded from
    # the binary-char set so it isn't dropped as binary.
    line = "".join(f"\x1b[32m+{i}\x1b[0m\n" for i in range(300)).encode()
    out = _fetch_with(monkeypatch, line, "text/plain")
    assert "binary content" not in out


def test_binary_unlabeled_caught_by_fallback(monkeypatch):
    # No Content-Type coerces to text/plain upstream; the fallback still catches it.
    out = _fetch_with(monkeypatch, bytes(range(256)) * 20, None)
    assert "�" not in out
    assert "binary content" in out


def test_html_page_unaffected(monkeypatch):
    html = b"<html><body><h1>Hello</h1><p>Real text content here.</p></body></html>"
    out = _fetch_with(monkeypatch, html, "text/html; charset=utf-8")
    assert "Hello" in out
    assert "non-text content" not in out and "binary content" not in out


def test_content_type_sanitized_in_message(monkeypatch):
    # An obs-folded Content-Type can smuggle control chars into get_content_type();
    # the returned message must be trimmed to a clean MIME token.
    out = _fetch_with(
        monkeypatch, b"\x00\x01\x02" * 500, "application/octet-stream\r\n data: injected"
    )
    assert "\n" not in out and "\r" not in out
    assert "injected" not in out
    assert "application/octet-stream" in out


@pytest.mark.parametrize(
    "n_bad,n_total,expect_binary",
    [
        # Straddle the 12.5% ratio (well above the 16-char floor): just under vs
        # just over len//8. Locks the divisor so it can't silently drift.
        (120, 1000, False),  # 120 <= 1000//8 (125) -> kept
        (130, 1000, True),  # 130 >  1000//8 (125) -> binary
    ],
)
def test_replacement_ratio_boundary(monkeypatch, n_bad, n_total, expect_binary):
    # Body of n_total chars: n_bad NUL control bytes + ASCII filler. NUL stays
    # binary through the cp1252 retry, so only the ratio threshold decides here.
    body = b"\x00" * n_bad + b"a" * (n_total - n_bad)
    out = _fetch_with(monkeypatch, body, "text/plain")
    assert ("binary content" in out) is expect_binary


def test_text_with_a_few_stray_replacement_chars_kept(monkeypatch):
    # A mostly-clean page with a handful of bad bytes stays (below the floor),
    # so we don't drop legitimate pages over minor encoding glitches.
    body = ("Real article text. " * 200).encode() + b"\xff\xfe\xff"
    out = _fetch_with(monkeypatch, body, "text/html")
    assert "Real article text." in out
    assert "binary content" not in out
