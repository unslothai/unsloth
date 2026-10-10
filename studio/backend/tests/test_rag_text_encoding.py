# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import codecs

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


def test_a_short_utf8_file_with_a_truncated_character_stays_utf8(tmp_path):
    assert _text(tmp_path, "cut.txt", b"caf\xc3\xa9\xc3") == "caf\u00e9\ufffd"
    page = b"<meta charset=utf-8><p>caf\xc3\xa9\xff</p>"
    assert _text(tmp_path, "cut.html", page) == "caf\u00e9\ufffd"


def test_a_utf8_byte_order_mark_wins_over_a_damaged_byte_and_a_declared_charset(tmp_path):
    assert _text(tmp_path, "bom.txt", b"\xef\xbb\xbfhello\xff") == "hello\ufffd"
    page = '<meta charset="shift_jis"><p>café \xff</p>'.encode("utf-8").replace(
        b"\xc3\xbf", b"\xff"
    )
    assert _text(tmp_path, "bom.html", codecs.BOM_UTF8 + page) == "café \ufffd"


def test_a_utf16_file_is_read_by_its_byte_order_mark(tmp_path):
    assert _text(tmp_path, "wide.txt", GREETING.encode("utf-16")) == GREETING


def test_an_html_page_is_read_in_the_charset_it_declares(tmp_path):
    for charset, text in (("shift_jis", "日本語のページ"), ("gbk", "中文网页")):
        page = f'<html><head><meta charset="{charset}"></head><body><p>{text}</p></body></html>'
        assert _text(tmp_path, f"{charset}.html", page.encode(charset)) == text


def test_a_declared_iso_2022_jp_page_is_not_read_as_its_escape_sequences(tmp_path):
    page = (
        '<html><head><meta charset="iso-2022-jp"></head><body><p>日本語のページ</p></body></html>'
    )
    assert _text(tmp_path, "jis.html", page.encode("iso2022_jp")) == "日本語のページ"


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


def test_crlf_and_cr_line_endings_become_newlines(tmp_path):
    windows = "Para one.\r\nStill one.\r\n\r\nPara two.\r\n"
    expected = "Para one.\nStill one.\n\nPara two.\n"
    assert _text(tmp_path, "crlf.txt", windows.encode("utf-8")) == expected
    assert _text(tmp_path, "crlf_ansi.txt", windows.encode("cp1252")) == expected
    assert _text(tmp_path, "crlf_wide.txt", windows.encode("utf-16")) == expected
    assert _text(tmp_path, "cr.md", b"a\rb\r") == "a\nb\n"


def test_a_stray_valid_utf8_sequence_does_not_keep_a_windows_1252_file_as_utf8(tmp_path):
    # Word puts a no-break space before "»" in French; "à\xa0»" is also valid UTF-8.
    french = "Il a dit «\xa0voilà\xa0» à l'été, café, crème.\n"
    assert _text(tmp_path, "fr.txt", french.encode("cp1252")) == french


def test_a_declared_charset_decodes_as_its_browser_superset(tmp_path):
    for charset, codec, text in (
        ("iso-8859-1", "cp1252", "“smart” quotes cost €5"),
        ("shift_jis", "cp932", "① 髙橋"),
        ("gb2312", "gbk", "镕"),
    ):
        page = f'<html><head><meta charset="{charset}"></head><body><p>{text}</p></body></html>'
        assert _text(tmp_path, f"{charset}.html", page.encode(codec)) == text, charset


def test_a_chinese_japanese_korean_or_russian_text_file_is_read_in_its_own_encoding(tmp_path):
    for codec, text in (
        ("gbk", "退货政策：收到商品后三十天内可以退货。请保留原始包装和发票。\n"),
        ("shift_jis", "返品ポリシー：商品到着後三十日以内に返品できます。\n"),
        ("euc_kr", "반품 정책: 상품 수령 후 30일 이내에 반품할 수 있습니다.\n"),
        ("big5", "退貨政策：收到商品後三十天內可以退貨。請保留原始包裝和發票。\n"),
        ("cp1251", "Политика возврата: товар можно вернуть в течение тридцати дней.\n"),
        (
            "cp1253",
            "Πολιτική επιστροφών: μπορείτε να επιστρέψετε το προϊόν εντός τριάντα ημερών.\n",
        ),
        ("cp1255", "מדיניות החזרות: ניתן להחזיר את המוצר תוך שלושים יום.\n"),
        ("cp1256", "سياسة الإرجاع: يمكنك إرجاع المنتج خلال ثلاثين يومًا.\n"),
    ):
        assert _text(tmp_path, f"{codec}.txt", text.encode(codec)) == text, codec


def test_a_chinese_japanese_korean_or_russian_file_with_code_in_it_is_read_in_its_own_encoding(
    tmp_path,
):
    for codec, text in (
        (
            "gbk",
            "# Unsloth 微调指南\n\n本文档介绍如何使用 Unsloth 在单张 GPU 上微调 Llama 3 模型。\n\n```bash\npip install unsloth\npip install --upgrade transformers datasets\n```\n\n注意：如果显存不足，请将 `max_seq_length` 调小，或者使用 `load_in_4bit = True`。\n",
        ),
        (
            "shift_jis",
            "このドキュメントでは、Docker を使って PostgreSQL サーバーを起動する方法を説明します。\n```\ndocker run -d --name pg -e POSTGRES_PASSWORD=secret postgres:16\n```\n",
        ),
        (
            "euc_kr",
            "이 문서는 Kubernetes 클러스터에서 nginx ingress controller 를 설정하는 방법을 설명합니다.\n```yaml\napiVersion: networking.k8s.io/v1\nkind: Ingress\n```\n",
        ),
        (
            "cp1251",
            "Установите пакет командой `pip install requests` и выполните скрипт `python main.py --config config.yaml`.\n",
        ),
    ):
        assert _text(tmp_path, f"{codec}.md", text.encode(codec)) == text, codec


def test_a_short_windows_1252_line_is_not_mistaken_for_another_language(tmp_path):
    for line in (
        "Copyright © 2024 Acme Inc. All rights reserved ®",
        "Price: 25 € – shipping included.",
        "Temperature: 25°C ± 2°C",
        "naïve résumé",
        "Jürgen Weiß",
        "São Paulo",
        "café",
        "é",
        "€ 12,50",
        "£50",
        "½ ¼ ¾",
        "“”",
        "1. Ä\n2. Ö\n3. Ü",
        "é" * 8,
        "Ü Ö Ä ß ü ö ä é",
        "° ± µ ½ ¼ ¾ © ®",
        "é è ê ë à â ä ô ö û ü ç",
        "Æ Ø Å æ ø å Æ Ø",
        "élève, für, está, bênção, égalité, âgé, forêt, café",
        "à é è ù â ê î ô û ë ï ü ç œ æ",
        "À É È Ê Ë Ù Û Ü Ô Ö Î Ï Â Ä Ç Œ Æ à é è ê ë ù û ü ô ö î ï â ä ç œ æ",
    ):
        assert _text(tmp_path, "short.txt", line.encode("cp1252")) == line, line
