# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Regression tests for binary bodies poisoning web_search model context (#7084)."""

from __future__ import annotations

import codecs
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
        self._pos = 0
        self.headers = Message()
        if content_type is not None:
            self.headers["Content-Type"] = content_type

    def read(self, n: int | None = None) -> bytes:
        # Advance a cursor like a real stream so the chunked reader reaches EOF.
        chunk = self._body[self._pos :] if n is None else self._body[self._pos : self._pos + n]
        self._pos += len(chunk)
        return chunk


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


@pytest.mark.parametrize(
    "content_type,expected",
    [
        ("text/html", True),
        ("text/plain; charset=utf-8", True),
        ("application/json", True),
        ("application/json; charset=utf-8", True),
        ("application/xml", True),
        ("application/xhtml+xml", True),
        ("application/ld+json", True),
        ("application/yaml", True),
        ("application/x-yaml", True),
        ("application/x-ndjson", True),
        ("application/ndjson", True),
        ("application/sql", True),
        ("application/x-www-form-urlencoded", True),
        ("application/pdf", False),
        ("image/png", False),
        ("image/svg+xml", False),
        ("application/octet-stream", True),
        ("application/zip", False),
        ("application/vnd.ms-excel", True),
        ("application/vnd.openxmlformats-officedocument.wordprocessingml.document", True),
        ("", True),
        (None, True),
    ],
)
def test_is_text_candidate_content_type(content_type, expected):
    assert tools._is_text_candidate_content_type(content_type) is expected


def test_pdf_rejected_by_content_type(monkeypatch):
    out = _fetch_with(monkeypatch, b"%PDF-1.7\n\xff\xd8\xff\x00\x89PNG" * 200, "application/pdf")
    assert "�" not in out
    assert "non-text content" in out and "application/pdf" in out


def test_text_octet_stream_kept_after_sniffing(monkeypatch):
    body = b"level=info\nmessage=plain text artifact\n" * 100
    out = _fetch_with(monkeypatch, body, "application/octet-stream")
    assert "plain text artifact" in out
    assert "non-text content" not in out and "binary content" not in out


@pytest.mark.parametrize(
    "content_type",
    ["application/octet-stream", "application/x-custom-binary", "text/plain", None],
)
def test_binary_candidates_rejected_after_sniffing(monkeypatch, content_type):
    out = _fetch_with(monkeypatch, bytes(range(256)) * 20, content_type)
    assert "�" not in out
    assert "binary content" in out


@pytest.mark.parametrize("content_type", ["application/sql", "application/x-www-form-urlencoded"])
def test_unknown_application_text_kept_after_sniffing(monkeypatch, content_type):
    out = _fetch_with(monkeypatch, b"select readable_text from artifacts;\n" * 100, content_type)
    assert "readable_text" in out
    assert "non-text content" not in out and "binary content" not in out


def test_excel_labeled_csv_kept_after_sniffing(monkeypatch):
    body = b"name,value\nreadable,42\n" * 100
    out = _fetch_with(monkeypatch, body, "application/vnd.ms-excel")
    assert "readable" in out
    assert "binary content" not in out


@pytest.mark.parametrize(
    "bom,encoding",
    [
        (codecs.BOM_UTF16_LE, "utf-16-le"),
        (codecs.BOM_UTF16_BE, "utf-16-be"),
        (codecs.BOM_UTF32_LE, "utf-32-le"),
        (codecs.BOM_UTF32_BE, "utf-32-be"),
    ],
)
@pytest.mark.parametrize("content_type", ["text/plain", "application/vnd.ms-excel"])
def test_bom_unicode_text_without_charset_kept(monkeypatch, bom, encoding, content_type):
    body = bom + ("name,value\nreadable,42\n" * 100).encode(encoding)
    out = _fetch_with(monkeypatch, body, content_type)
    assert "readable" in out
    assert "binary content" not in out


def test_valid_utf8_binary_caught_by_control_chars(monkeypatch):
    # These controls are valid UTF-8 and therefore produce no replacement chars.
    body = bytes([0, 1, 2, 3, 4, 5, 6, 7]) * 400
    out = _fetch_with(monkeypatch, body, "text/plain")
    assert "binary content" in out


@pytest.mark.parametrize(
    "magic",
    [
        b"%PDF-",
        b"PK\x03\x04",
        b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1",
        b"\x1f\x8b",
        b"BZh",
        b"\xfd7zXZ\x00",
        b"\x28\xb5\x2f\xfd",
    ],
)
def test_text_labeled_binary_caught_by_magic(monkeypatch, magic):
    out = _fetch_with(monkeypatch, magic + b" printable text-heavy body" * 100, "text/plain")
    assert "binary content" in out


@pytest.mark.parametrize(
    "prefix",
    [
        codecs.BOM_UTF8,
        codecs.BOM_UTF16_LE,
        codecs.BOM_UTF16_BE,
        codecs.BOM_UTF32_LE,
        codecs.BOM_UTF32_BE,
        b" \r\n",
        b"\t\xef\xbb\xbf ",
    ],
)
def test_pdf_magic_after_harmless_prefix(monkeypatch, prefix):
    body = prefix + b"%PDF-1.7\n" + b"1 0 obj<</Type/Catalog>>endobj\n" * 100
    out = _fetch_with(monkeypatch, body, "text/plain")
    assert "binary content" in out


@pytest.mark.parametrize(
    "content_type,magic",
    [
        ("application/vnd.ms-excel", b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"),
        (
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            b"PK\x03\x04",
        ),
    ],
)
def test_office_labeled_binary_caught_by_magic(monkeypatch, content_type, magic):
    out = _fetch_with(monkeypatch, magic + b" printable text-heavy body" * 100, content_type)
    assert "binary content" in out


def test_latin1_text_without_charset_kept(monkeypatch):
    # The cp1252 retry should rescue accent-heavy text with ASCII structure.
    body = (
        "Muller lauft uber die Strasse: schoene, groesse. MARKERWORD ".replace("ue", "ü")
        + "äöüß éèà "
    ) * 30
    out = _fetch_with(monkeypatch, body.encode("cp1252"), "text/plain")
    assert "binary content" not in out
    assert "MARKERWORD" in out


@pytest.mark.parametrize("charset", ["iso-8859-1", "latin-1", "latin1"])
def test_declared_latin1_cp1252_punctuation_kept(monkeypatch, charset):
    body = ("“quoted” " * 100).encode("cp1252")
    out = _fetch_with(monkeypatch, body, f"text/plain; charset={charset}")
    assert "quoted" in out
    assert "binary content" not in out


def test_high_byte_binary_not_rescued_as_cp1252(monkeypatch):
    # cp1252 maps these bytes to printable characters, but they lack ASCII structure.
    body = bytes(range(0xA0, 0x100)) * 40
    out = _fetch_with(monkeypatch, body, "text/plain")
    assert "binary content" in out


def test_ansi_colored_text_log_kept(monkeypatch):
    # ESC is excluded from the binary set so ANSI logs remain readable.
    line = "".join(f"\x1b[32m+{i}\x1b[0m\n" for i in range(300)).encode()
    out = _fetch_with(monkeypatch, line, "text/plain")
    assert "binary content" not in out


def test_html_page_unaffected(monkeypatch):
    html = b"<html><body><h1>Hello</h1><p>Real text content here.</p></body></html>"
    out = _fetch_with(monkeypatch, html, "text/html; charset=utf-8")
    assert "Hello" in out
    assert "non-text content" not in out and "binary content" not in out


def test_content_type_sanitized_in_message(monkeypatch):
    # Do not echo obs-folded header content into the model response.
    out = _fetch_with(monkeypatch, b"\x00\x01\x02" * 500, "application/pdf\r\n data: injected")
    assert "\n" not in out and "\r" not in out
    assert "injected" not in out
    assert "application/pdf" in out


@pytest.mark.parametrize(
    "n_bad,n_total,expect_binary",
    [
        (120, 1000, False),
        (130, 1000, True),
    ],
)
def test_binary_char_ratio_boundary(monkeypatch, n_bad, n_total, expect_binary):
    body = b"\x00" * n_bad + b"a" * (n_total - n_bad)
    out = _fetch_with(monkeypatch, body, "text/plain")
    assert ("binary content" in out) is expect_binary


def test_text_with_a_few_stray_replacement_chars_kept(monkeypatch):
    # Minor encoding glitches below the floor should not drop a real page.
    body = ("Real article text. " * 200).encode() + b"\xff\xfe\xff"
    out = _fetch_with(monkeypatch, body, "text/html")
    assert "Real article text." in out
    assert "binary content" not in out
