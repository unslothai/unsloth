"""Document parser tests — each format skipped if its lib is unavailable."""

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
STUDIO_BACKEND = REPO_ROOT / "studio" / "backend"
if str(STUDIO_BACKEND) not in sys.path:
    sys.path.insert(0, str(STUDIO_BACKEND))


def test_text_parser_utf8(tmp_path):
    from core.rag.parsers import parse

    file = tmp_path / "sample.txt"
    file.write_text("hello world\n\nsecond paragraph", encoding = "utf-8")
    pages = parse(file)
    assert len(pages) == 1
    assert "hello world" in pages[0].text
    assert "second paragraph" in pages[0].text


def test_markdown_parser_treated_as_text(tmp_path):
    from core.rag.parsers import parse

    file = tmp_path / "sample.md"
    file.write_text("# Title\n\nBody text with **emphasis**.", encoding = "utf-8")
    pages = parse(file)
    assert pages and "Title" in pages[0].text


def test_unsupported_format_raises(tmp_path):
    from core.rag.parsers import UnsupportedFormatError, parse

    file = tmp_path / "weird.xyz"
    file.write_text("nope")
    with pytest.raises(UnsupportedFormatError):
        parse(file)


def test_html_parser_strips_scripts(tmp_path):
    pytest.importorskip("bs4")
    pytest.importorskip("lxml")
    from core.rag.parsers import parse

    file = tmp_path / "sample.html"
    file.write_text(
        "<html><body><script>alert(1)</script><p>visible text</p></body></html>",
        encoding = "utf-8",
    )
    pages = parse(file)
    assert pages
    assert "visible text" in pages[0].text
    assert "alert" not in pages[0].text


def test_pdf_parser_extracts_pages(tmp_path):
    pypdf = pytest.importorskip("pypdf")
    from pypdf import PdfWriter

    file = tmp_path / "tiny.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width = 72, height = 72)
    with open(file, "wb") as f:
        writer.write(f)
    from core.rag.parsers import parse

    # blank page yields no extractable text — should return [] without error
    pages = parse(file)
    assert isinstance(pages, list)


def test_docx_parser_extracts_paragraphs(tmp_path):
    docx = pytest.importorskip("docx")
    from docx import Document

    file = tmp_path / "sample.docx"
    doc = Document()
    doc.add_paragraph("First paragraph here.")
    doc.add_paragraph("Second paragraph here.")
    doc.save(str(file))
    from core.rag.parsers import parse

    pages = parse(file)
    assert pages
    assert "First paragraph" in pages[0].text
    assert "Second paragraph" in pages[0].text
