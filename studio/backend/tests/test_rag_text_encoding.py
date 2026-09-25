# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Text, Markdown and HTML uploads are read in the encoding they were written in.

Decoded as UTF-8 with ``errors="replace"``, a Windows-1252 file ("ANSI", what
Notepad and Excel write in Western Europe) loses every accented letter to
U+FFFD, and an HTML page in Shift_JIS or GBK loses all of its text.
"""

from __future__ import annotations

from core.rag import parsers

GREETING = "Sehr geehrte Frau Müller,\nviele Grüße aus Köln – 5 €.\n"


def _text(tmp_path, name: str, data: bytes) -> str:
    path = tmp_path / name
    path.write_bytes(data)
    return "\n".join(page.text for page in parsers.parse(str(path)))


def test_a_windows_1252_text_file_keeps_its_accented_letters(tmp_path):
    assert _text(tmp_path, "letter.txt", GREETING.encode("cp1252")) == GREETING


def test_a_latin_1_markdown_file_keeps_its_accented_letters(tmp_path):
    markdown = "# Überblick\nCafé und Crème brûlée\n"
    assert _text(tmp_path, "notes.md", markdown.encode("latin-1")) == markdown


def test_utf8_is_read_as_before(tmp_path):
    assert _text(tmp_path, "plain.txt", GREETING.encode("utf-8")) == GREETING
    assert _text(tmp_path, "bom.txt", GREETING.encode("utf-8-sig")) == GREETING


def test_a_damaged_byte_in_utf8_costs_only_that_byte(tmp_path):
    data = GREETING.encode("utf-8") + b"\xff"
    assert _text(tmp_path, "damaged.txt", data) == GREETING + "�"


def test_a_utf16_file_is_read_by_its_byte_order_mark(tmp_path):
    assert _text(tmp_path, "wide.txt", GREETING.encode("utf-16")) == GREETING


def test_an_html_page_is_read_in_the_charset_it_declares(tmp_path):
    for charset, text in (("shift_jis", "日本語のページ"), ("gbk", "中文网页")):
        page = f'<html><head><meta charset="{charset}"></head><body><p>{text}</p></body></html>'
        assert _text(tmp_path, f"{charset}.html", page.encode(charset)) == text


def test_an_http_equiv_content_type_declares_the_charset_too(tmp_path):
    page = (
        '<html><head><meta http-equiv="Content-Type" content="text/html; charset=iso-8859-2">'
        "</head><body><p>Dobrý den, Łódź</p></body></html>"
    )
    assert _text(tmp_path, "latin2.html", page.encode("iso-8859-2")) == "Dobrý den, Łódź"


def test_an_undeclared_windows_1252_page_keeps_its_accented_letters(tmp_path):
    page = "<html><body><p>Grüße aus Köln</p></body></html>"
    assert _text(tmp_path, "page.html", page.encode("cp1252")) == "Grüße aus Köln"


def test_a_declared_charset_that_cannot_have_written_the_page_is_ignored(tmp_path):
    for charset in ("no-such-charset", "utf-16", "punycode", "idna"):
        page = f'<html><head><meta charset="{charset}"></head><body><p>Grüße</p></body></html>'
        assert _text(tmp_path, "odd.html", page.encode("cp1252")) == "Grüße", charset
