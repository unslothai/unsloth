# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""PDF text extraction: layout-aware Markdown (pymupdf4llm) with plain-text fallback."""

from __future__ import annotations

import re

import pytest


def _shared_setup_1():
    pytest.importorskip("docx")
    import docx

    from core.rag import parsers

    document = docx.Document()
    return document, docx, parsers


pytest.importorskip("pymupdf")


def _table_pdf(path):
    import pymupdf

    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_textbox(pymupdf.Rect(40, 40, 550, 70), "Quarterly Results", fontsize = 16)
    rows = [("Quarter", "Revenue", "Growth"), ("Q1", "$1.2M", "12%"), ("Q2", "$1.5M", "25%")]
    y = 90
    for r in rows:
        page.insert_textbox(pymupdf.Rect(40, y, 250, y + 20), r[0], fontsize = 11)
        page.insert_textbox(pymupdf.Rect(250, y, 400, y + 20), r[1], fontsize = 11)
        page.insert_textbox(pymupdf.Rect(400, y, 540, y + 20), r[2], fontsize = 11)
        y += 24
    doc.save(str(path))
    doc.close()


def test_pdf_extracts_markdown_table(tmp_path, monkeypatch):
    pytest.importorskip("pymupdf4llm")
    from core.rag import config, parsers

    monkeypatch.setattr(config, "PDF_MARKDOWN", True)
    pdf = tmp_path / "table.pdf"
    _table_pdf(pdf)
    text = "\n".join(p.text for p in parsers.parse(str(pdf)))
    assert "Q2" in text and "$1.5M" in text
    assert "#" in text or "|" in text


def test_pdf_markdown_keeps_text_drawn_over_a_picture(tmp_path, monkeypatch):
    pytest.importorskip("pymupdf4llm")
    import pymupdf

    from core.rag import config, parsers

    monkeypatch.setattr(config, "PDF_MARKDOWN", True)
    doc = pymupdf.open()
    page = doc.new_page()
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 60, 300), False)
    pix.set_rect(pix.irect, (40, 70, 120))
    page.insert_image(pymupdf.Rect(0, 0, 170, page.rect.height), pixmap = pix)
    for i, line in enumerate(["jane.doe@example.com", "Kubernetes", "Spanish"]):
        page.insert_text((15, 80 + i * 22), line, fontsize = 11, color = (1, 1, 1))
    page.insert_text((190, 60), "Experience", fontsize = 16)
    for i in range(30):
        page.insert_text((190, 90 + i * 22), f"Led project {i} for customers.", fontsize = 10)
    pdf = tmp_path / "resume.pdf"
    doc.save(str(pdf))
    doc.close()

    text = "\n".join(p.text for p in parsers.parse(str(pdf)))
    assert "# Experience" in text
    assert "Led project 29" in text
    for sidebar in ("jane.doe@example.com", "Kubernetes", "Spanish"):
        assert sidebar in text


def test_pdf_markdown_does_not_render_scanned_pages(tmp_path, monkeypatch):
    # pymupdf4llm's background-colour probe renders page corners; under a full-page scan that
    # decodes the whole image while holding the GIL, which starved the event loop until the
    # desktop watchdog killed the backend (#13094). Text pages keep the probe.
    pytest.importorskip("pymupdf4llm")
    import pymupdf

    from core.rag import config, parsers

    monkeypatch.setattr(config, "PDF_MARKDOWN", True)
    doc = pymupdf.open()
    scan = doc.new_page()
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 1000, 1300), False)
    pix.set_rect(pix.irect, (250, 250, 240))
    scan.insert_image(scan.rect, pixmap = pix, keep_proportion = False)
    scan.insert_text((60, 120), "Chapter one covers shelter.", fontsize = 10)
    text_page = doc.new_page()
    text_page.insert_text((60, 80), "Survival Training", fontsize = 16)
    text_page.insert_text((60, 120), "Chapter two covers water.", fontsize = 10)
    pdf = tmp_path / "scan.pdf"
    doc.save(str(pdf))
    doc.close()

    rendered = []
    real_get_pixmap = pymupdf.Page.get_pixmap

    def recording_get_pixmap(self, *args, **kwargs):
        rendered.append(self.number)
        return real_get_pixmap(self, *args, **kwargs)

    monkeypatch.setattr(pymupdf.Page, "get_pixmap", recording_get_pixmap)
    pages = parsers.parse(str(pdf))
    assert [p.page_number for p in pages] == [1, 2]
    assert "Chapter one covers shelter." in pages[0].text
    assert "Chapter two covers water." in pages[1].text
    assert "# Survival Training" in pages[1].text
    assert 0 not in rendered
    assert 1 in rendered


@pytest.mark.parametrize("rotation", [0, 90, 180, 270])
def test_scan_corner_check_follows_page_rotation(rotation):
    import pymupdf

    from core.rag import parsers

    doc = pymupdf.open()
    page = doc.new_page()
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 2000, 500), False)
    pix.set_rect(pix.irect, (250, 250, 240))
    # A strip along the unrotated bottom edge covers two corners at every rotation.
    page.insert_image(pymupdf.Rect(0, 700, 595, 842), pixmap = pix, keep_proportion = False)
    inset = doc.new_page()
    inset.insert_image(pymupdf.Rect(100, 100, 400, 600), pixmap = pix, keep_proportion = False)
    for number in range(2):
        doc[number].set_rotation(rotation)
    assert parsers._image_covers_a_corner(doc, 0)
    assert not parsers._image_covers_a_corner(doc, 1)


def test_scan_corner_check_ignores_small_corner_logos():
    # A logo is cheap to decode, so its page keeps pymupdf4llm's background probe.
    import pymupdf

    from core.rag import parsers

    doc = pymupdf.open()
    page = doc.new_page()
    logo = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 200, 200), False)
    logo.set_rect(logo.irect, (200, 30, 30))
    page.insert_image(pymupdf.Rect(0, 0, 60, 60), pixmap = logo)
    assert not parsers._image_covers_a_corner(doc, 0)


def test_pdf_markdown_off_uses_plain_text(tmp_path, monkeypatch):
    from core.rag import config, parsers

    monkeypatch.setattr(config, "PDF_MARKDOWN", False)
    pdf = tmp_path / "table.pdf"
    _table_pdf(pdf)
    text = "\n".join(p.text for p in parsers.parse(str(pdf)))
    assert "Q2" in text and "$1.5M" in text
    assert "#" not in text and "|" not in text


def test_pdf_bytes_use_same_extraction_path(tmp_path, monkeypatch):
    from core.rag import config, parsers

    monkeypatch.setattr(config, "PDF_MARKDOWN", False)
    pdf = tmp_path / "table.pdf"
    _table_pdf(pdf)
    from_file = parsers.parse(str(pdf))
    from_bytes, total_pages = parsers.parse_pdf_bytes(pdf.read_bytes())
    assert [page.text for page in from_bytes] == [page.text for page in from_file]
    assert total_pages == len(from_file)


def test_pdf_bytes_limit_pages_before_extraction(monkeypatch):
    import pymupdf

    from core.rag import config, parsers

    monkeypatch.setattr(config, "PDF_MARKDOWN", False)
    doc = pymupdf.open()
    for marker in ("page one", "page two", "page three"):
        page = doc.new_page()
        page.insert_text((40, 40), marker)
    data = doc.tobytes()
    doc.close()

    pages, total_pages = parsers.parse_pdf_bytes(data, max_pages = 2)
    assert len(pages) == 2
    assert "page two" in pages[-1].text
    assert total_pages == 3


def test_pdf_markdown_receives_page_limit(monkeypatch):
    from core.rag import parsers

    captured = {}

    class _FakePymupdf4llm:
        @staticmethod
        def to_markdown(doc, **kwargs):
            captured.update(kwargs)
            return [{"text": "page"} for _ in kwargs["pages"]]

    class _Doc:
        page_count = 100

    monkeypatch.setitem(__import__("sys").modules, "pymupdf4llm", _FakePymupdf4llm)
    assert parsers._pdf_markdown(_Doc(), range(2)) == ["page", "page"]
    assert captured == {
        "page_chunks": True,
        "show_progress": False,
        "ignore_images": True,
        "pages": [0, 1],
    }


def test_pdf_markdown_passes_only_supported_legacy_kwargs(monkeypatch):
    # The pinned PyMuPDF4LLM legacy path ignores unknown kwargs; do not pass layout-only OCR knobs.
    from core.rag import parsers

    captured = {}

    class _FakePymupdf4llm:
        @staticmethod
        def to_markdown(doc, **kwargs):
            captured.update(kwargs)
            return [{"text": "plain markdown"}]

    class _Doc:
        page_count = 1

    monkeypatch.setitem(__import__("sys").modules, "pymupdf4llm", _FakePymupdf4llm)
    assert parsers._pdf_markdown(_Doc()) == ["plain markdown"]
    assert captured == {"page_chunks": True, "show_progress": False, "ignore_images": True}


def test_pdf_markdown_falls_back_when_lib_missing(tmp_path, monkeypatch):
    from core.rag import config, parsers

    monkeypatch.setattr(config, "PDF_MARKDOWN", True)
    monkeypatch.setattr(parsers, "_pdf_markdown", lambda doc: None)
    pdf = tmp_path / "table.pdf"
    _table_pdf(pdf)
    pages = parsers.parse(str(pdf))
    assert pages and "Quarter" in pages[0].text


def _long_text_pdf(path):
    import pymupdf

    doc = pymupdf.open()
    page = doc.new_page()
    body = "The quick brown fox jumps over the lazy dog. " * 12
    page.insert_textbox(pymupdf.Rect(40, 40, 550, 750), body, fontsize = 11)
    doc.save(str(path))
    doc.close()


def test_pdf_markdown_corruption_falls_back_to_plain(tmp_path, monkeypatch):
    # pymupdf4llm can emit shaped RTL Presentation Forms; the parser uses logical-order text.
    from core.rag import config, parsers

    monkeypatch.setattr(config, "PDF_MARKDOWN", True)
    shaped = "".join(chr(c) for c in range(0xFE8D, 0xFEA0)) * 20
    monkeypatch.setattr(parsers, "_pdf_markdown", lambda doc: [shaped] * doc.page_count)
    pdf = tmp_path / "table.pdf"
    _table_pdf(pdf)
    text = "\n".join(p.text for p in parsers.parse(str(pdf)))
    assert "Quarter" in text
    assert not parsers._markdown_corrupted(text)


def test_pdf_markdown_incomplete_falls_back_to_plain(tmp_path, monkeypatch):
    from core.rag import config, parsers

    monkeypatch.setattr(config, "PDF_MARKDOWN", True)
    monkeypatch.setattr(parsers, "_pdf_markdown", lambda doc: ["x"] * doc.page_count)
    pdf = tmp_path / "long.pdf"
    _long_text_pdf(pdf)
    text = "\n".join(p.text for p in parsers.parse(str(pdf)))
    assert "quick brown fox" in text


def _docx_with_table(path):
    import docx

    document = docx.Document()
    document.add_paragraph("Intro before table.")
    table = document.add_table(rows = 2, cols = 2)
    table.cell(0, 0).text = "NAME"
    table.cell(0, 1).text = "SCORE"
    table.cell(1, 0).text = "Alice"
    table.cell(1, 1).text = "97pts"
    document.add_paragraph("Outro after table.")
    document.save(str(path))


def test_docx_extracts_table_cells(tmp_path):
    # document.paragraphs drops tables; cells are pipe-joined, which the preview locator anchors on.
    pytest.importorskip("docx")
    from core.rag import parsers

    docx_path = tmp_path / "t.docx"
    _docx_with_table(docx_path)
    text = "\n".join(p.text for p in parsers.parse(str(docx_path)))
    assert all(v in text for v in ("NAME", "SCORE", "Alice", "97pts"))
    assert "Alice | 97pts" in text
    assert text.index("Intro") < text.index("NAME") < text.index("Outro")


def test_docx_table_keeps_columns_and_collapses_cell_newlines(tmp_path):
    document, docx, parsers = _shared_setup_1()
    table = document.add_table(rows = 2, cols = 3)
    table.cell(0, 0).text = "A"
    table.cell(0, 1).text = ""
    table.cell(0, 2).text = "C"
    multiline = table.cell(1, 0)
    multiline.text = "line1"
    multiline.add_paragraph("line2")
    table.cell(1, 1).text = "mid"
    table.cell(1, 2).text = "end"
    path = tmp_path / "aligned.docx"
    document.save(str(path))

    text = "\n".join(p.text for p in parsers.parse(str(path)))
    assert "A |  | C" in text
    assert "line1 line2 | mid | end" in text


def test_docx_table_merged_cell_keeps_grid_alignment(tmp_path):
    document, docx, parsers = _shared_setup_1()
    table = document.add_table(rows = 2, cols = 3)
    table.cell(0, 0).text = "WIDE"
    table.cell(0, 2).text = "END"
    table.cell(0, 0).merge(table.cell(0, 1))
    table.cell(1, 0).text = "a"
    table.cell(1, 1).text = "b"
    table.cell(1, 2).text = "c"
    path = tmp_path / "merged.docx"
    document.save(str(path))

    text = "\n".join(p.text for p in parsers.parse(str(path)))
    assert text.count("WIDE") == 1
    assert "WIDE |  | END" in text
    assert "a | b | c" in text


def test_docx_table_pads_omitted_grid_columns(tmp_path):
    pytest.importorskip("docx")
    import docx
    from docx.oxml.ns import qn

    from core.rag import parsers

    document = docx.Document()
    table = document.add_table(rows = 2, cols = 3)
    table.cell(0, 0).text = "H1"
    table.cell(0, 1).text = "H2"
    table.cell(0, 2).text = "H3"
    tr = table.rows[1]._tr
    tr.remove(tr.tc_lst[0])
    trPr = tr.get_or_add_trPr()
    trPr.insert(0, trPr.makeelement(qn("w:gridBefore"), {qn("w:val"): "1"}))
    table.rows[1].cells[0].text = "X"
    path = tmp_path / "gap.docx"
    document.save(str(path))

    text = "\n".join(p.text for p in parsers.parse(str(path)))
    assert " | X | " in text


def test_docx_flattens_nested_table(tmp_path):
    document, docx, parsers = _shared_setup_1()
    outer = document.add_table(rows = 1, cols = 1).cell(0, 0)
    outer.text = "outer"
    nested = outer.add_table(rows = 1, cols = 2)
    nested.cell(0, 0).text = "NESTED-A"
    nested.cell(0, 1).text = "NESTED-B"
    path = tmp_path / "nested.docx"
    document.save(str(path))

    text = "\n".join(p.text for p in parsers.parse(str(path)))
    assert "NESTED-A | NESTED-B" in text


def test_docx_nested_table_keeps_in_cell_order(tmp_path):
    document, docx, parsers = _shared_setup_1()
    cell = document.add_table(rows = 1, cols = 1).cell(0, 0)
    cell.text = "before"
    nested = cell.add_table(rows = 1, cols = 2)
    nested.cell(0, 0).text = "NESTED-A"
    nested.cell(0, 1).text = "NESTED-B"
    cell.add_paragraph("after")
    path = tmp_path / "nested_order.docx"
    document.save(str(path))

    text = "\n".join(p.text for p in parsers.parse(str(path)))
    assert text.index("before") < text.index("NESTED-A") < text.index("after")


def test_docx_table_vertical_merge_emitted_once(tmp_path):
    document, docx, parsers = _shared_setup_1()
    table = document.add_table(rows = 3, cols = 2)
    table.cell(0, 0).merge(table.cell(1, 0)).merge(table.cell(2, 0)).text = "SECTION"
    table.cell(0, 1).text = "r0"
    table.cell(1, 1).text = "r1"
    table.cell(2, 1).text = "r2"
    path = tmp_path / "vmerge.docx"
    document.save(str(path))

    text = "\n".join(p.text for p in parsers.parse(str(path)))
    assert text.count("SECTION") == 1
    assert "SECTION | r0" in text and " | r1" in text and " | r2" in text


_DOCX_XMLNS = (
    'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" '
    'xmlns:mc="http://schemas.openxmlformats.org/markup-compatibility/2006" '
    'xmlns:wps="http://schemas.microsoft.com/office/word/2010/wordprocessingShape" '
    'xmlns:v="urn:schemas-microsoft-com:vml" '
    'xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math"'
)


def _docx_from_xml(tmp_path, *fragments):
    document, docx, parsers = _shared_setup_1()
    from docx.oxml import parse_xml

    section = document.element.body[-1]
    for element in list(parse_xml(f"<w:body {_DOCX_XMLNS}>{''.join(fragments)}</w:body>")):
        section.addprevious(element)
    path = tmp_path / "xml.docx"
    document.save(str(path))
    return "\n".join(p.text for p in parsers.parse(str(path)))


def _r(text):
    return f'<w:r><w:t xml:space="preserve">{text}</w:t></w:r>'


def test_docx_reads_tracked_insertions_not_deletions(tmp_path):
    text = _docx_from_xml(
        tmp_path,
        "<w:p>"
        + _r("The fee is ")
        + '<w:del w:id="1" w:author="a"><w:r><w:delText>ten</w:delText><w:tab/></w:r></w:del>'
        + f'<w:ins w:id="2" w:author="a">{_r("twelve")}</w:ins>'
        + f'<w:moveFrom w:id="3" w:author="a">{_r(" moved away")}</w:moveFrom>'
        + _r(" euros.")
        + "</w:p>",
    )
    assert text == "The fee is twelve euros."


def test_docx_reads_content_controls_fields_and_smart_tags(tmp_path):
    text = _docx_from_xml(
        tmp_path,
        f"<w:p>{_r('Client: ')}<w:sdt><w:sdtPr/><w:sdtContent>{_r('Acme Corp')}</w:sdtContent></w:sdt></w:p>",
        f"<w:sdt><w:sdtPr/><w:sdtContent><w:p>{_r('Block control')}</w:p></w:sdtContent></w:sdt>",
        f'<w:p><w:fldSimple w:instr=" DOCPROPERTY Company ">{_r("Field result")}</w:fldSimple></w:p>',
        f'<w:p><w:smartTag w:uri="urn:x" w:element="place">{_r("Smart tag")}</w:smartTag></w:p>',
        f'<w:customXml w:element="clause"><w:p>{_r("Custom XML")}</w:p></w:customXml>',
    )
    assert text == "Client: Acme Corp\nBlock control\nField result\nSmart tag\nCustom XML"


def test_docx_table_cells_read_content_controls_and_insertions(tmp_path):
    text = _docx_from_xml(
        tmp_path,
        "<w:tbl><w:tr>"
        f"<w:tc><w:p>{_r('Owner')}</w:p></w:tc>"
        f"<w:tc><w:sdt><w:sdtPr/><w:sdtContent><w:p>{_r('Ada')}</w:p></w:sdtContent></w:sdt></w:tc>"
        f'<w:tc><w:p><w:ins w:id="1" w:author="a">{_r("Engineer")}</w:ins></w:p></w:tc>'
        "</w:tr></w:tbl>",
    )
    assert text == "Owner | Ada | Engineer"


def test_docx_reads_text_box_once_after_its_paragraph(tmp_path):
    # Word writes a text box twice: the DrawingML shape and a VML fallback copy.
    box = f"<w:txbxContent><w:p>{_r('Callout')}</w:p></w:txbxContent>"
    text = _docx_from_xml(
        tmp_path,
        "<w:p>"
        + _r("Host line")
        + "<w:r><mc:AlternateContent>"
        + f'<mc:Choice Requires="wps"><w:drawing><wps:wsp><wps:txbx>{box}</wps:txbx></wps:wsp></w:drawing></mc:Choice>'
        + f"<mc:Fallback><w:pict><v:shape><v:textbox>{box}</v:textbox></v:shape></w:pict></mc:Fallback>"
        + "</mc:AlternateContent></w:r></w:p>",
    )
    assert text == "Host line\nCallout"


def test_docx_drops_text_box_inside_tracked_deletion_or_move(tmp_path):
    def box(text):
        shape = f"<w:txbxContent><w:p>{_r(text)}</w:p></w:txbxContent>"
        return f"<w:r><w:pict><v:shape><v:textbox>{shape}</v:textbox></v:shape></w:pict></w:r>"

    text = _docx_from_xml(
        tmp_path,
        "<w:p>"
        + _r("Kept")
        + f'<w:del w:id="1" w:author="a">{box("Deleted box")}</w:del>'
        + f'<w:moveFrom w:id="2" w:author="a">{box("Moved-away box")}</w:moveFrom>'
        + "</w:p>",
    )
    assert text == "Kept"


def test_docx_reads_ruby_base_without_its_guide(tmp_path):
    text = _docx_from_xml(
        tmp_path,
        f"<w:p><w:r><w:ruby><w:rubyPr/><w:rt>{_r('kanji')}</w:rt><w:rubyBase>{_r('漢字')}</w:rubyBase></w:ruby></w:r></w:p>",
    )
    assert text == "漢字"


def test_docx_reads_one_branch_of_alternate_content(tmp_path):
    text = _docx_from_xml(
        tmp_path,
        "<w:p><mc:AlternateContent>"
        f'<mc:Choice Requires="w14">{_r("Preferred")}</mc:Choice>'
        f"<mc:Fallback>{_r('Fallback')}</mc:Fallback>"
        "</mc:AlternateContent></w:p>",
    )
    assert text == "Preferred"


def _m(text):
    return f'<m:r><w:rPr><w:rFonts w:ascii="Cambria Math"/></w:rPr><m:t>{text}</m:t></m:r>'


def test_docx_keeps_equations_where_they_appear(tmp_path):
    half = f"<m:f><m:fPr><m:ctrlPr><w:rPr><w:i/></w:rPr></m:ctrlPr></m:fPr><m:num>{_m('1')}</m:num><m:den>{_m('2')}</m:den></m:f>"
    squared = f"<m:sSup><m:e>{_m('v')}</m:e><m:sup>{_m('2')}</m:sup></m:sSup>"
    text = _docx_from_xml(
        tmp_path,
        f"<w:p>{_r('The kinetic energy is ')}<m:oMath>{_m('E=')}{half}{_m('m')}{squared}</m:oMath>{_r(' joules.')}</w:p>",
        f"<w:p><m:oMathPara><m:oMathParaPr/><m:oMath>{_m('F=ma')}</m:oMath><m:oMath>{_m('p=mv')}</m:oMath></m:oMathPara></w:p>",
    )
    assert text == "The kinetic energy is E=\\frac{1}{2}mv^{2} joules.\nF=ma\np=mv"


def test_docx_table_cells_keep_equations_without_deleted_parts(tmp_path):
    root = f"<m:rad><m:radPr><m:degHide m:val=\"1\"/></m:radPr><m:deg/><m:e>{_m('x')}</m:e></m:rad>"
    text = _docx_from_xml(
        tmp_path,
        "<w:tbl><w:tr>"
        f"<w:tc><w:p>{_r('Work')}</w:p></w:tc>"
        f"<w:tc><w:p><m:oMath>{_m('W=F')}<m:d><m:e>{_m('a+b')}</m:e></m:d>"
        f'<w:del w:id="1" w:author="a">{_m("+c")}</w:del>{root}</m:oMath></w:p></w:tc>'
        "</w:tr></w:tbl>",
    )
    assert text == "Work | W=F(a+b)\\sqrt{x}"


def test_docx_equations_keep_binomials_and_skip_hidden_phantoms(tmp_path):
    binom = f'<m:d><m:e><m:f><m:fPr><m:type m:val="noBar"/></m:fPr><m:num>{_m("n")}</m:num><m:den>{_m("k")}</m:den></m:f></m:e></m:d>'
    phantom = (
        f'<m:phant><m:phantPr><m:show m:val="off"/></m:phantPr><m:e>{_m("xyz")}</m:e></m:phant>'
    )
    text = _docx_from_xml(
        tmp_path, f"<w:p><m:oMath>{binom}{_m('a')}{phantom}{_m('b')}</m:oMath></w:p>"
    )
    assert text == "({n \\atop k})ab"


def test_docx_equations_keep_bars_and_group_characters(tmp_path):
    mean = f'<m:bar><m:barPr><m:pos m:val="top"/></m:barPr><m:e>{_m("x")}</m:e></m:bar>'
    brace = f"<m:limLow><m:e><m:groupChr><m:e>{_m('a+b')}</m:e></m:groupChr></m:e><m:lim>{_m('n')}</m:lim></m:limLow>"
    arrow = f'<m:groupChr><m:groupChrPr><m:chr m:val="→"/><m:pos m:val="top"/></m:groupChrPr><m:e>{_m("Δ")}</m:e></m:groupChr>'
    text = _docx_from_xml(
        tmp_path, f"<w:p><m:oMath>{mean}{_m('=')}{brace}{_m(',')}{arrow}</m:oMath></w:p>"
    )
    assert text == "\\overline{x}=\\underbrace{a+b}_{n},\\overset{→}{Δ}"


# Same cases and expected text as the chat attachment reader's test (attachment-preview-text.test.ts).
_OMML_CASES = [
    (
        '<m:nary><m:naryPr><m:chr m:val="∑"/></m:naryPr><m:sub>{i=1}</m:sub><m:sup>{n}</m:sup><m:e>{i}</m:e></m:nary>',
        "∑_{i=1}^{n}i",
    ),
    ("<m:nary><m:sub>{0}</m:sub><m:sup>{1}</m:sup><m:e>{x}</m:e></m:nary>", "∫_{0}^{1}x"),
    (
        "<m:sSubSup><m:e>{x}</m:e><m:sub>{i}</m:sub><m:sup>{2}</m:sup></m:sSubSup><m:sSub><m:e>{a}</m:e><m:sub>{0}</m:sub></m:sSub>",
        "x_{i}^{2}a_{0}",
    ),
    ("<m:sPre><m:sub>{6}</m:sub><m:sup>{14}</m:sup><m:e>{C}</m:e></m:sPre>", "{}_{6}^{14}C"),
    ("<m:limUpp><m:e>{x}</m:e><m:lim>{def}</m:lim></m:limUpp>", "x^{def}"),
    ("<m:limLow><m:e>{lim}</m:e><m:lim>{n→∞}</m:lim></m:limLow>", "lim_{n→∞}"),
    (
        '<m:f><m:fPr><m:type m:val="noBar"/></m:fPr><m:num>{n}</m:num><m:den>{k}</m:den></m:f>'
        '<m:phant><m:phantPr><m:show m:val="off"/></m:phantPr><m:e>{xyz}</m:e></m:phant>',
        "{n \\atop k}",
    ),
    (
        "<m:acc><m:e>{θ}</m:e></m:acc><m:rad><m:deg>{3}</m:deg><m:e>{y}</m:e></m:rad>",
        "θ̂\\sqrt[3]{y}",
    ),
    (
        '<m:rad><m:radPr><m:degHide m:val="1"/></m:radPr><m:deg>{3}</m:deg><m:e>{x}</m:e></m:rad>'
        '<m:nary><m:naryPr><m:chr m:val="∑"/><m:subHide m:val="0"/><m:supHide/></m:naryPr><m:sub>{k}</m:sub><m:sup>{n}</m:sup><m:e>{a}</m:e></m:nary>',
        "\\sqrt{x}∑_{k}a",
    ),
    ("<m:func><m:fName>{sin}</m:fName><m:e>{x}</m:e></m:func>", "sin x"),
    (
        "<m:m><m:mr><m:e>{a}</m:e><m:e>{b}</m:e></m:mr><m:mr><m:e>{c}</m:e><m:e>{d}</m:e></m:mr></m:m>",
        "a & b \\\\ c & d",
    ),
    ("<m:eqArr><m:e>{x=1}</m:e><m:e>{y=2}</m:e></m:eqArr>", "x=1\ny=2"),
    (
        '<m:d><m:e>{a}</m:e><m:e>{b}</m:e></m:d><m:d><m:dPr><m:begChr m:val="["/><m:endChr m:val=""/></m:dPr><m:e>{c}</m:e></m:d>',
        "(a|b)[c",
    ),
    (
        '<m:bar><m:e>{x}</m:e></m:bar><m:groupChr><m:groupChrPr><m:chr m:val="⏞"/><m:pos m:val="top"/></m:groupChrPr><m:e>{y}</m:e></m:groupChr>'
        '<m:groupChr><m:groupChrPr><m:chr m:val="←"/></m:groupChrPr><m:e>{z}</m:e></m:groupChr>',
        "\\underline{x}\\overbrace{y}\\underset{←}{z}",
    ),
    (
        '{a}<w:r><w:t xml:space="preserve"> if </w:t></w:r>'
        "<w:sdt><w:sdtPr><w:showingPlcHdr/></w:sdtPr><w:sdtContent>{prompt}</w:sdtContent></w:sdt>{b}",
        "a if b",
    ),
]


@pytest.mark.parametrize(("omml", "expected"), _OMML_CASES)
def test_docx_equation_structures(tmp_path, omml, expected):
    filled = re.sub(r"\{([^{}]*)\}", lambda match: _m(match.group(1)), omml)
    text = _docx_from_xml(tmp_path, f"<w:p><m:oMath>{filled}</m:oMath></w:p>")
    assert text == expected


def test_docx_keeps_rows_and_cells_wrapped_in_content_controls(tmp_path):
    document, docx, parsers = _shared_setup_1()
    from docx.oxml import parse_xml
    from docx.oxml.ns import nsdecls

    ns = nsdecls("w")
    table = document.add_table(rows = 1, cols = 2)
    table.cell(0, 0).text = "Name"
    tr = table.rows[0]._tr
    tc = table.cell(0, 1)._tc
    tr.remove(tc)
    tr.append(
        parse_xml(
            f"<w:sdt {ns}><w:sdtContent><w:tc><w:p><w:r><w:t>CELL-WRAPPED</w:t></w:r></w:p></w:tc>"
            "</w:sdtContent></w:sdt>"
        )
    )
    table._tbl.append(
        parse_xml(
            f"<w:sdt {ns}><w:sdtContent><w:sdt><w:sdtContent><w:tr>"
            "<w:tc><w:p><w:r><w:t>ROW-A</w:t></w:r></w:p></w:tc>"
            "<w:tc><w:p><w:r><w:t>ROW-B</w:t></w:r></w:p></w:tc>"
            "</w:tr></w:sdtContent></w:sdt></w:sdtContent></w:sdt>"
        )
    )
    path = tmp_path / "wrapped.docx"
    document.save(str(path))

    text = "\n".join(pg.text for pg in parsers.parse(str(path)))
    assert "Name | CELL-WRAPPED" in text
    assert "ROW-A | ROW-B" in text


def test_docx_skips_placeholder_text_and_keeps_field_and_bidi_runs(tmp_path):
    document, docx, parsers = _shared_setup_1()
    from docx.oxml import parse_xml
    from docx.oxml.ns import nsdecls

    ns = nsdecls("w")

    def sdt(props, inner):
        return parse_xml(
            f"<w:sdt {ns}><w:sdtPr>{props}</w:sdtPr><w:sdtContent>{inner}</w:sdtContent></w:sdt>"
        )

    body = document.element.body
    body.insert(
        len(body) - 1,
        sdt("<w:showingPlcHdr/>", "<w:p><w:r><w:t>BLOCK-PROMPT</w:t></w:r></w:p>"),
    )
    p = document.add_paragraph("Name: ")._p
    p.append(sdt("<w:showingPlcHdr/>", "<w:r><w:t>Click or tap here to enter text.</w:t></w:r>"))
    p = document.add_paragraph("Client: ")._p
    p.append(sdt('<w:showingPlcHdr w:val="0"/>', "<w:r><w:t>ACME</w:t></w:r>"))
    p = document.add_paragraph("Ref ")._p
    p.append(
        parse_xml(
            f'<w:fldSimple {ns} w:instr=" MERGEFIELD Name "><w:r><w:t>FIELD</w:t></w:r></w:fldSimple>'
        )
    )
    p.append(parse_xml(f'<w:dir {ns} w:val="rtl"><w:r><w:t> RTL</w:t></w:r></w:dir>'))
    table = document.add_table(rows = 1, cols = 2)
    table.cell(0, 0).text = "Owner"
    table.cell(0, 1)._tc.append(
        sdt("<w:showingPlcHdr/>", "<w:p><w:r><w:t>CELL-PROMPT</w:t></w:r></w:p>")
    )
    path = tmp_path / "placeholders.docx"
    document.save(str(path))

    text = "\n".join(pg.text for pg in parsers.parse(str(path)))
    assert "PROMPT" not in text and "Click or tap" not in text
    assert "Client: ACME" in text
    assert "Ref FIELD RTL" in text
    assert "Owner | " in text


def test_docx_skips_placeholder_rows_and_cells_but_keeps_columns(tmp_path):
    document, docx, parsers = _shared_setup_1()
    from docx.oxml import parse_xml
    from docx.oxml.ns import nsdecls

    ns = nsdecls("w")
    table = document.add_table(rows = 1, cols = 3)
    table.cell(0, 0).text = "Name"
    table.cell(0, 2).text = "END"
    tr = table.rows[0]._tr
    tc = table.cell(0, 1)._tc
    idx = tr.index(tc)
    tr.remove(tc)
    tr.insert(
        idx,
        parse_xml(
            f"<w:sdt {ns}><w:sdtPr><w:showingPlcHdr/></w:sdtPr><w:sdtContent>"
            "<w:tc><w:p><w:r><w:t>CELL-PROMPT</w:t></w:r></w:p></w:tc></w:sdtContent></w:sdt>"
        ),
    )
    table._tbl.append(
        parse_xml(
            f"<w:sdt {ns}><w:sdtPr><w:showingPlcHdr/></w:sdtPr><w:sdtContent><w:tr>"
            "<w:tc><w:p><w:r><w:t>ROW-PROMPT</w:t></w:r></w:p></w:tc>"
            "<w:tc><w:p/></w:tc><w:tc><w:p/></w:tc></w:tr></w:sdtContent></w:sdt>"
        )
    )
    path = tmp_path / "placeholder_cells.docx"
    document.save(str(path))

    text = "\n".join(pg.text for pg in parsers.parse(str(path)))
    assert "PROMPT" not in text
    assert "Name |  | END" in text


def test_docx_numbers_visible_note_references_and_marks_the_body(tmp_path):
    document, docx, parsers = _shared_setup_1()
    from docx.opc.constants import CONTENT_TYPE as CT, RELATIONSHIP_TYPE as RT
    from docx.opc.packuri import PackURI
    from docx.opc.part import Part
    from docx.oxml import parse_xml

    def note(kind, note_id, text):
        return f'<w:{kind} w:id="{note_id}"><w:p><w:r><w:{kind}Ref/></w:r>{_r(" " + text)}</w:p></w:{kind}>'

    def notes(kind, body):
        return (
            f"<w:{kind}s {_DOCX_XMLNS}>"
            f'<w:{kind} w:type="separator" w:id="-1"><w:p><w:r><w:separator/></w:r></w:p></w:{kind}>'
            f'<w:{kind} w:type="continuationSeparator" w:id="0"><w:p><w:r><w:continuationSeparator/></w:r></w:p></w:{kind}>'
            f"{body}</w:{kind}s>"
        ).encode()

    def ref(kind, note_id):
        return f'<w:r><w:{kind}Reference w:id="{note_id}"/></w:r>'

    section = document.element.body[-1]
    section.addprevious(
        parse_xml(
            f"<w:p {_DOCX_XMLNS}><w:del w:id=\"9\">{ref('footnote', 4)}</w:del>"
            f"{_r('First.')}{ref('footnote', 2)}"
            f"{_r(' Second.')}{ref('footnote', 1)}{ref('endnote', 1)}"
            f"{_r(' ' + chr(0xE000) + '7' + chr(0xE001))}</w:p>"
        )
    )
    for kind, content_type, reltype, body in (
        (
            "footnote",
            CT.WML_FOOTNOTES,
            RT.FOOTNOTES,
            note("footnote", 1, "Source: LATER")
            + note("footnote", 2, "Source: EARLIER")
            + note("footnote", 3, "Source: UNREFERENCED")
            + note("footnote", 4, "Source: DELETED"),
        ),
        ("endnote", CT.WML_ENDNOTES, RT.ENDNOTES, note("endnote", 1, "Source: ENDNOTEBODY")),
    ):
        part = Part(
            PackURI(f"/word/{kind}s.xml"), content_type, notes(kind, body), document.part.package
        )
        document.part.relate_to(part, reltype)
    path = tmp_path / "notes.docx"
    document.save(str(path))

    text = "\n".join(pg.text for pg in parsers.parse(str(path)))
    assert text == (
        "First.[1] Second.[2][i] \ue0007\ue001\n"
        "Footnotes\n[1] Source: EARLIER\n[2] Source: LATER\n[3] Source: UNREFERENCED\n"
        "Endnotes\n[i] Source: ENDNOTEBODY"
    )


def _parse_html(tmp_path, body):
    from core.rag import parsers

    path = tmp_path / "page.html"
    path.write_text(f"<html><body>{body}</body></html>", encoding = "utf-8")
    return "\n".join(p.text for p in parsers.parse(str(path)))


def test_html_keeps_inline_elements_in_their_line(tmp_path):
    text = _parse_html(
        tmp_path,
        '<p>The <b>quick</b> brown fox jumps over the <a href="#">lazy</a> dog.</p>'
        "<p>It is un<b>believ</b>able.</p>",
    )
    assert text == "The quick brown fox jumps over the lazy dog.\nIt is unbelievable."


def test_html_adjacent_buttons_stay_separate_words(tmp_path):
    text = _parse_html(
        tmp_path, "<p>Click <button>Accept</button><button>Decline</button> to go on.</p>"
    )
    assert text == "Click Accept Decline to go on."


def test_html_block_elements_start_new_lines(tmp_path):
    text = _parse_html(
        tmp_path,
        "<h1>Install <em>guide</em></h1><ul><li>One</li><li>Two <i>items</i></li></ul>"
        "<p>line one<br>line two</p><table><tr><td>cell a</td><td>cell b</td></tr></table>",
    )
    assert text == "Install guide\nOne\nTwo items\nline one\nline two\ncell a | cell b"


def test_html_table_rows_keep_their_columns(tmp_path):
    text = _parse_html(
        tmp_path,
        "<table><tr><th>Plan</th><th>Price</th><th>Support</th><th>Seats</th></tr>"
        "<tr><td>Starter</td><td>$9</td><td><p>Email</p><p>Chat</p></td><td>1</td></tr>"
        "<tr><td>Team<td>$49<td><td>10</table><p>After</p>",
    )
    assert text == (
        "Plan | Price | Support | Seats\nStarter | $9 | Email Chat | 1\nTeam | $49 |  | 10\nAfter"
    )


def test_html_table_spans_keep_columns_aligned(tmp_path):
    text = _parse_html(
        tmp_path,
        "<table><tr><th>Plan</th><th colspan=2>Price</th></tr>"
        "<tr><td rowspan=2>Pro</td><td>Monthly</td><td>$20</td></tr>"
        "<tr><td>Annual</td><td>$200</td></tr></table>",
    )
    assert text == "Plan | Price | \nPro | Monthly | $20\n | Annual | $200"


def test_html_trailing_rowspan_keeps_columns_aligned(tmp_path):
    text = _parse_html(
        tmp_path,
        "<table><tr><td>Item</td><td rowspan=2>Notes</td></tr><tr><td>Next</td></tr></table>",
    )
    assert text == "Item | Notes\nNext | "


def test_html_table_preserves_preformatted_cell_whitespace(tmp_path):
    text = _parse_html(
        tmp_path,
        "<table><tr><td><pre>a\n b<br>  c</pre></td><td>d</td></tr></table>",
    )
    assert text == "a\n b\n  c | d"


def test_html_layout_and_nested_tables_keep_their_lines(tmp_path):
    text = _parse_html(
        tmp_path,
        "<table><tr><td><h1>Title</h1><p>Intro</p>"
        "<table><tr><td>Name<table><tr><td>x</td><td>y</td></tr></table></td><td>Value</td></tr></table>"
        "<p>Outro</p></td></tr></table>",
    )
    assert text == "Title\nIntro\nName | Value\nx | y\nOutro"


def test_html_text_after_a_nested_table_stays_in_its_parent_cell(tmp_path):
    text = _parse_html(
        tmp_path,
        "<table><tr><td>Before<table><tr><td>x</td><td>y</td></tr></table>After</td>"
        "<td>Peer</td></tr></table>",
    )
    assert text == "Before After | Peer\nx | y"


def test_html_table_spans_are_capped_by_the_file_size(tmp_path):
    html = "<table><tr><td colspan=1000 rowspan=65534>x" + "<tr><td>a" * 2000
    text = _parse_html(tmp_path, html)
    assert text.count("a") == 2000 and len(text) < 4 * len(html)


def test_html_empty_rows_spend_the_span_budget(tmp_path):
    text = _parse_html(
        tmp_path,
        "<table><tr><td colspan=50 rowspan=65534>x" + "<tr>" * 100 + "<tr><td>a<td>b</table>",
    )
    assert text.splitlines()[-1] == "a | b"


def test_html_rowspan_stops_at_its_row_group(tmp_path):
    text = _parse_html(
        tmp_path,
        "<table><thead><tr><th rowspan=3>Plan</th><th>Price</th></tr></thead>"
        "<tbody><tr><td rowspan=0>Team</td><td>$49</td></tr><tr><td>$490</td></tr></tbody>"
        "<tbody><tr><td>Pro</td><td>$99</td></tr></tbody></table>",
    )
    assert text == "Plan | Price\nTeam | $49\n | $490\nPro | $99"


def test_html_tables_nested_past_the_depth_cap_read_as_blocks(tmp_path):
    text = _parse_html(tmp_path, "<table><tr><td>" * 32 + "<table><tr><td>a<td>b" + "<p>end")
    assert text == "a\nb\nend"


def test_html_stray_cell_end_keeps_words_apart(tmp_path):
    text = _parse_html(tmp_path, "<table><tr><td>Total</td>Note</td>Extra</tr></table>")
    assert text == "Note\nExtra\nTotal"


def test_html_legend_and_options_stay_separate_words(tmp_path):
    text = _parse_html(
        tmp_path,
        "<fieldset><legend>Size</legend>Pick one</fieldset>"
        "<select><option>Small</option><option>Large</option></select>",
    )
    assert text == "Size\nPick one\nSmall\nLarge"


def test_html_keeps_text_after_the_last_block(tmp_path):
    text = _parse_html(tmp_path, "<p>First</p>Trailing <b>text</b>")
    assert text == "First\nTrailing text"


def test_html_pre_keeps_its_layout(tmp_path):
    text = _parse_html(tmp_path, "<p>Code:</p><pre>def f():\n    return 1</pre>")
    assert text == "Code:\ndef f():\n    return 1"


def test_html_textarea_keeps_its_layout_and_svg_labels_stay_apart(tmp_path):
    text = _parse_html(
        tmp_path,
        "<textarea>line one\n    indented</textarea>"
        '<svg><text x="0">Revenue</text><text x="0" y="20">Cost</text></svg>',
    )
    assert text == "line one\n    indented\nRevenue\nCost"


def test_html_positioned_svg_tspans_are_separate_labels(tmp_path):
    text = _parse_html(
        tmp_path,
        '<svg><text><tspan x="0" y="0">Revenue</tspan><tspan x="0" dy="20">Cost</tspan>'
        "</text><text>Bold<tspan>er</tspan></text></svg>",
    )
    assert text == "Revenue\nCost\nBolder"


def test_html_skips_script_style_and_template(tmp_path):
    text = _parse_html(
        tmp_path,
        "<p>Visible</p><script>var x = 1;</script><style>p { color: red }</style>"
        "<template><p>Inert until cloned</p></template>",
    )
    assert text == "Visible"


def test_html_keeps_declarative_shadow_root_text(tmp_path):
    text = _parse_html(
        tmp_path,
        '<my-card><template shadowrootmode="open"><p>Shadow text</p>'
        "<template><p>inert</p></template></template></my-card><p>After</p>",
    )
    assert text == "Shadow text\nAfter"


def test_html_template_blocks_do_not_split_visible_text(tmp_path):
    text = _parse_html(tmp_path, "<p>Hello <template><div>hidden</div></template>world</p>")
    assert text == "Hello world"
