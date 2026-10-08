# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Office, OpenDocument, e-book, email and RTF files: parsing, upload and indexing."""

import datetime as dt
import io
import re
import struct
import zipfile
from email.message import EmailMessage
from pathlib import Path

import pytest
from fastapi import UploadFile

from core.rag import cfb, config, ingestion, parsers, store
from routes.rag import _save_upload
from storage import rag_db

from ._compound_file import compound_file

S = 'xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"'
P = 'xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"'
REL = 'xmlns="http://schemas.openxmlformats.org/package/2006/relationships"'
ODF = (
    'xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0" '
    'xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0" '
    'xmlns:table="urn:oasis:names:tc:opendocument:xmlns:table:1.0" '
    'xmlns:draw="urn:oasis:names:tc:opendocument:xmlns:drawing:1.0" '
    'xmlns:presentation="urn:oasis:names:tc:opendocument:xmlns:presentation:1.0"'
)


def _zip(path, members):
    with zipfile.ZipFile(path, "w") as z:
        for name, data in members.items():
            z.writestr(name, data)
    return path


def _text(path):
    return "\n".join(page.text for page in parsers.parse(str(path)))


# ---------------------------------------------------------------- builders


def build_xlsx(path):
    serial = (dt.date(2026, 3, 31) - dt.date(1899, 12, 30)).days
    return _zip(
        path,
        {
            "xl/workbook.xml": f'<workbook {S}><sheets><sheet name="Revenue" sheetId="1" r:id="rId1"/></sheets></workbook>',
            "xl/_rels/workbook.xml.rels": f'<Relationships {REL}><Relationship Id="rId1" Target="worksheets/sheet1.xml"/></Relationships>',
            "xl/sharedStrings.xml": f"<sst {S}><si><t>Region</t></si>"
            "<si><r><t>Zebra</t></r><r><t>marker</t></r><rPh><t>ignored</t></rPh></si></sst>",
            "xl/styles.xml": f'<styleSheet {S}><cellXfs><xf numFmtId="0"/><xf numFmtId="14"/></cellXfs></styleSheet>',
            "xl/worksheets/sheet1.xml": f"<worksheet {S}><sheetData>"
            '<row r="1"><c r="A1" t="s"><v>0</v></c><c r="B1" t="inlineStr"><is><t>Q1</t></is></c></row>'
            f'<row r="2"><c r="A2" t="s"><v>1</v></c><c r="B2"><v>1200.5</v></c><c r="C2" s="1"><v>{serial}</v></c>'
            '<c r="D2" t="b"><v>1</v></c><c r="F2" t="str"><v>formula text</v></c></row>'
            "</sheetData></worksheet>",
        },
    )


def build_pptx(path):
    run = lambda text: f"<a:p><a:r><a:t>{text}</a:t></a:r></a:p>"
    return _zip(
        path,
        {
            "ppt/presentation.xml": f'<p:presentation {P}><p:sldIdLst><p:sldId id="256" r:id="rId2"/></p:sldIdLst></p:presentation>',
            "ppt/_rels/presentation.xml.rels": f'<Relationships {REL}><Relationship Id="rId2" Target="slides/slide1.xml"/></Relationships>',
            "ppt/slides/slide1.xml": f"<p:sld {P}><p:cSld><p:spTree>"
            f"<p:sp><p:txBody>{run('Quarterly review')}</p:txBody></p:sp>"
            "<p:graphicFrame><a:graphic><a:graphicData><a:tbl>"
            f"<a:tr><a:tc><a:txBody>{run('Region')}</a:txBody></a:tc><a:tc><a:txBody>{run('Q1')}</a:txBody></a:tc></a:tr>"
            f"<a:tr><a:tc><a:txBody>{run('North')}</a:txBody></a:tc><a:tc><a:txBody>{run('1200')}</a:txBody></a:tc></a:tr>"
            "</a:tbl></a:graphicData></a:graphic></p:graphicFrame>"
            f"<p:sp><p:txBody>{run('Revenue increased')}</p:txBody></p:sp>"
            "</p:spTree></p:cSld></p:sld>",
            "ppt/slides/_rels/slide1.xml.rels": f'<Relationships {REL}><Relationship Id="rId1" Target="../notesSlides/notesSlide1.xml"/></Relationships>',
            "ppt/notesSlides/notesSlide1.xml": f"<p:notes {P}><p:cSld><p:spTree><p:sp><p:txBody>"
            f"{run('1')}{run('Speaker notes zebramarker')}</p:txBody></p:sp></p:spTree></p:cSld></p:notes>",
        },
    )


def _odf(path, body):
    return _zip(
        path,
        {
            "mimetype": "application/vnd.oasis.opendocument",
            "content.xml": f"<office:document-content {ODF}><office:body>{body}</office:body></office:document-content>",
        },
    )


def build_odt(path):
    return _odf(
        path,
        "<office:text><text:h>Quarterly report</text:h>"
        '<text:p>Revenue<text:s text:c="2"/>increased<text:note><text:note-body><text:p>see appendix</text:p></text:note-body></text:note>.</text:p>'
        "<table:table><table:table-row><table:table-cell><text:p>Zebramarker</text:p></table:table-cell>"
        "<table:table-cell><text:p>1200</text:p></table:table-cell></table:table-row></table:table></office:text>",
    )


def build_ods(path):
    return _odf(
        path,
        '<office:spreadsheet><table:table table:name="Revenue">'
        "<table:table-row><table:table-cell><text:p>Region</text:p></table:table-cell>"
        '<table:table-cell table:number-columns-repeated="2"><text:p>Q</text:p></table:table-cell>'
        '<table:table-cell table:number-columns-repeated="16000"/></table:table-row>'
        '<table:table-row table:number-rows-repeated="1048000"><table:table-cell table:number-columns-repeated="16000"/></table:table-row>'
        "<table:table-row><table:table-cell><text:p>Zebramarker</text:p></table:table-cell></table:table-row>"
        "</table:table></office:spreadsheet>",
    )


def build_odp(path):
    return _odf(
        path,
        "<office:presentation>"
        "<draw:page><draw:frame><draw:text-box><text:p>First slide</text:p></draw:text-box></draw:frame>"
        "<presentation:notes><draw:frame><draw:text-box><text:p>zebramarker notes</text:p></draw:text-box></draw:frame></presentation:notes></draw:page>"
        "<draw:page><draw:frame><draw:text-box><text:p>Second slide revenue</text:p></draw:text-box></draw:frame></draw:page>"
        "</office:presentation>",
    )


def build_epub(path):
    page = (
        lambda title,
        text: f"<html><head><title>{title}</title></head><body><p>{text}</p></body></html>"
    )
    return _zip(
        path,
        {
            "mimetype": "application/epub+zip",
            "META-INF/container.xml": '<container xmlns="urn:oasis:names:tc:opendocument:xmlns:container">'
            '<rootfiles><rootfile full-path="OEBPS/content.opf"/></rootfiles></container>',
            "OEBPS/content.opf": '<package xmlns="http://www.idpf.org/2007/opf"><manifest>'
            '<item id="nav" href="nav.xhtml" media-type="application/xhtml+xml" properties="nav"/>'
            '<item id="c1" href="text/c1.xhtml" media-type="application/xhtml+xml"/>'
            '<item id="c2" href="text/c2.xhtml" media-type="application/xhtml+xml"/>'
            '</manifest><spine><itemref idref="nav"/><itemref idref="c2"/><itemref idref="c1"/></spine></package>',
            "OEBPS/nav.xhtml": page("Contents", "navmarker"),
            "OEBPS/text/c1.xhtml": page("One", "Zebramarker ends the book."),
            "OEBPS/text/c2.xhtml": page("Two", "Revenue increased."),
        },
    )


def build_eml(path):
    message = EmailMessage()
    message["From"] = "Ann <ann@example.com>"
    message["To"] = "bob@example.com"
    message["Subject"] = "Q1 numbers résumé"
    message.set_content("Revenue increased.\nZebramarker in the body.", cte = "base64")
    message.add_alternative("<p>html copy</p>", subtype = "html")
    message.add_attachment(b"x,y\n", maintype = "text", subtype = "csv", filename = "data.csv")
    path.write_bytes(bytes(message))
    return path


def build_rtf(path):
    path.write_bytes(
        rb"{\rtf1\ansi\ansicpg1252\uc1{\fonttbl{\f0 Arial;}}{\colortbl;\red0\green0\blue0;}"
        rb"{\*\generator Writer;}\f0 Quarterly report\par Caf\'e9 \u8364? revenue increased\par"
        rb"{\field{\*\fldinst HYPERLINK hiddenmarker}{\fldrslt link text}}\par"
        rb"\trowd Zebramarker\cell 1200\cell\row}"
    )
    return path


def build_doc(path, *, encrypted = False):
    compressed = "Quarterly report\r".encode("cp1252")
    wide = ("Café 東京 \x13 HYPERLINK x \x14link\x15 Zebramarker\x071200\x07\x07").encode(
        "utf-16-le"
    )
    word = bytearray(0x1800)
    flags = 0x0200 | (0x0100 if encrypted else 0)
    struct.pack_into("<HH", word, 0, 0xA5EC, 0xC1)
    struct.pack_into("<H", word, 0x0A, flags)
    struct.pack_into("<H", word, 32, 14)  # csw
    struct.pack_into("<H", word, 62, 22)  # cslw
    ccp = len(compressed) + len(wide) // 2
    struct.pack_into("<i", word, 64 + 3 * 4, ccp)  # ccpText
    struct.pack_into("<H", word, 152, 93)  # cbRgFcLcb
    word[0x1000 : 0x1000 + len(compressed)] = compressed
    word[0x1400 : 0x1400 + len(wide)] = wide
    cps = [0, len(compressed), ccp]
    pcds = [(0x2000 | 0x40000000), 0x1400]
    plc = struct.pack("<3I", *cps) + b"".join(struct.pack("<HIH", 0, fc, 0) for fc in pcds)
    clx = b"\x01" + struct.pack("<H", 2) + b"\0\0" + b"\x02" + struct.pack("<I", len(plc)) + plc
    struct.pack_into("<II", word, 154 + 33 * 8, 0, len(clx))  # fcClx, lcbClx
    path.write_bytes(compound_file({("WordDocument",): bytes(word), ("1Table",): clx}))
    return path


def _record(kind, body):
    return struct.pack("<HH", kind, len(body)) + body


def build_xls(path):
    def short(text, size = 1):
        return struct.pack("<B" if size == 1 else "<H", len(text)) + b"\0" + text.encode("latin-1")

    # SST: "Region", then a string whose characters continue, wide, in a CONTINUE record.
    sst = struct.pack("<II", 2, 2) + struct.pack("<HB", 6, 0) + b"Region"
    sst += struct.pack("<HB", 12, 0) + b"Zebra"
    cont = b"\x01" + "marker€".encode("utf-16-le")
    globals_ = (
        _record(0x0809, b"\0" * 16)
        + _record(0x0085, b"\0" * 4 + b"\0\0" + short("Revenue"))
        + _record(0x00FC, sst)
        + _record(0x003C, cont)
        + _record(0x000A, b"")
    )
    rk_int = lambda n: ((n << 2) | 2) & 0xFFFFFFFF
    sheet = (
        _record(0x0809, b"\0" * 16)
        + _record(0x00FD, struct.pack("<HHHI", 0, 0, 0, 0))
        + _record(0x0203, struct.pack("<HHHd", 0, 1, 0, 1200.5))
        + _record(0x00FD, struct.pack("<HHHI", 1, 0, 0, 1))
        + _record(0x027E, struct.pack("<HHHI", 1, 1, 0, rk_int(-7)))
        + _record(
            0x00BD,
            struct.pack("<HH", 2, 0)
            + struct.pack("<HI", 0, rk_int(3))
            + struct.pack("<HI", 0, rk_int(4))
            + struct.pack("<H", 1),
        )
        + _record(0x0809, b"\0" * 16)  # embedded chart
        + _record(0x000A, b"")
        + _record(0x0006, struct.pack("<HHH", 3, 0, 0) + b"\0" * 6 + b"\xff\xff" + b"\0" * 6)
        + _record(0x0207, short("formula text", 2))
        + _record(0x0205, struct.pack("<HHHBB", 3, 1, 0, 1, 0))
        + _record(0x000A, b"")
    )
    stream = bytearray(globals_ + sheet)
    struct.pack_into("<I", stream, 4 + 16 + 4, len(globals_))  # BOUNDSHEET lbPlyPos
    path.write_bytes(compound_file({("Workbook",): bytes(stream)}))
    return path


def _ppt_atom(
    kind,
    body,
    inst = 0,
):
    return struct.pack("<HHI", inst << 4, kind, len(body)) + body


def _ppt_container(
    kind,
    children,
    inst = 0,
):
    body = b"".join(children)
    return struct.pack("<HHI", (inst << 4) | 0xF, kind, len(body)) + body


def build_ppt(path, *, from_slides = False):
    chars = lambda text: _ppt_atom(0x0FA0, text.encode("utf-16-le"))
    persist = _ppt_atom(0x03F3, b"\0" * 20)
    if from_slides:
        slides = [
            _ppt_container(
                0x03EE, [_ppt_container(0x040C, [_ppt_container(0xF00D, [chars(text)])])]
            )
            for text in ("Drawn slide", "Zebramarker drawn")
        ]
        document = _ppt_container(0x03E8, slides)
    else:
        document = _ppt_container(
            0x03E8,
            [
                _ppt_container(
                    0x0FF0,
                    [
                        persist,
                        chars("Quarterly review\rRevenue up"),
                        persist,
                        _ppt_atom(0x0FA8, b"Zebramarker slide"),
                    ],
                    inst = 0,
                ),
                _ppt_container(
                    0x0FF0, [persist, chars("Click to edit Master title style")], inst = 1
                ),
            ],
        )
    path.write_bytes(compound_file({("PowerPoint Document",): document}))
    return path


def build_msg(path):
    utf16 = lambda text: text.encode("utf-16-le")
    filetime = int(
        (dt.datetime(2026, 10, 6, 10, 0) - dt.datetime(1601, 1, 1)).total_seconds() * 10**7
    )
    props = b"\0" * 32 + struct.pack("<IIQ", (0x0039 << 16) | 0x0040, 0, filetime)
    streams = {
        ("__substg1.0_0037001F",): utf16("Q1 numbers"),
        ("__substg1.0_0C1A001F",): utf16("Ann"),
        ("__substg1.0_0C1F001F",): utf16("ann@example.com"),
        ("__substg1.0_0E04001F",): utf16("Bob"),
        ("__substg1.0_1000001F",): utf16("Revenue increased.\r\nZebramarker in the body."),
        ("__properties_version1.0",): props,
        ("__attach_version1.0_#00000000", "__substg1.0_3707001F"): utf16("data.csv"),
    }
    path.write_bytes(compound_file(streams))
    return path


BUILDERS = {
    ".doc": build_doc,
    ".xls": build_xls,
    ".xlsx": build_xlsx,
    ".xlsm": build_xlsx,
    ".ppt": build_ppt,
    ".pptx": build_pptx,
    ".msg": build_msg,
    ".eml": build_eml,
    ".rtf": build_rtf,
    ".odt": build_odt,
    ".ods": build_ods,
    ".odp": build_odp,
    ".epub": build_epub,
}


# ---------------------------------------------------------------- parsing


def test_every_document_type_has_a_fixture():
    assert set(BUILDERS) == config.DOCUMENT_UPLOAD_EXTS
    assert config.DOCUMENT_UPLOAD_EXTS <= config.UPLOAD_EXTS


def test_document_exts_mirror_the_frontend_accept_list():
    source = (
        Path(__file__).resolve().parents[2] / "frontend/src/features/rag/types/rag.ts"
    ).read_text(encoding = "utf-8")
    accept = re.search(r'RAG_DOCUMENT_UPLOAD_ACCEPT =\s*"([^"]+)"', source).group(1)
    assert set(accept.split(",")) == config.DOCUMENT_UPLOAD_EXTS


def test_xlsx_reads_shared_inline_dates_and_gaps(tmp_path):
    text = _text(build_xlsx(tmp_path / "book.xlsx"))
    assert (
        text
        == "Sheet: Revenue\nRegion | Q1\nZebramarker | 1200.5 | 2026-03-31 | TRUE |  | formula text"
    )


def test_pptx_reads_slides_in_order_with_tables_and_notes(tmp_path):
    pages = parsers.parse(str(build_pptx(tmp_path / "deck.pptx")))
    assert [(p.page_number, p.text) for p in pages] == [
        (
            1,
            "Quarterly review\nRegion | Q1\nNorth | 1200\nRevenue increased\nNotes:\nSpeaker notes zebramarker",
        )
    ]


def test_odt_reads_paragraphs_spaces_notes_and_tables(tmp_path):
    assert _text(build_odt(tmp_path / "memo.odt")) == (
        "Quarterly report\nRevenue  increased [see appendix].\nZebramarker | 1200"
    )


def test_ods_does_not_expand_empty_repeats(tmp_path):
    assert _text(build_ods(tmp_path / "sheet.ods")) == "Sheet: Revenue\nRegion | Q | Q\nZebramarker"


def test_odp_reads_one_page_per_slide(tmp_path):
    pages = parsers.parse(str(build_odp(tmp_path / "slides.odp")))
    assert [(p.page_number, p.text) for p in pages] == [
        (1, "First slide\nNotes:\nzebramarker notes"),
        (2, "Second slide revenue"),
    ]


def test_epub_follows_the_spine_and_skips_navigation(tmp_path):
    text = _text(build_epub(tmp_path / "book.epub"))
    assert "navmarker" not in text
    assert text.index("Revenue increased") < text.index("Zebramarker ends the book")


def test_eml_decodes_headers_and_body(tmp_path):
    text = _text(build_eml(tmp_path / "mail.eml"))
    assert "Subject: Q1 numbers résumé" in text
    assert "From: Ann <ann@example.com>" in text
    assert "Attachments: data.csv" in text
    assert "Zebramarker in the body." in text
    assert "html copy" not in text


def test_rtf_keeps_text_and_drops_control_groups(tmp_path):
    assert _text(build_rtf(tmp_path / "memo.rtf")) == (
        "Quarterly report\nCafé € revenue increased\nlink text\nZebramarker | 1200"
    )


def test_doc_reads_compressed_and_unicode_pieces(tmp_path):
    assert _text(build_doc(tmp_path / "memo.doc")) == (
        "Quarterly report\nCafé 東京 link Zebramarker | 1200"
    )


def test_xls_reads_continued_strings_numbers_and_formulas(tmp_path):
    assert _text(build_xls(tmp_path / "book.xls")) == (
        "Sheet: Revenue\nRegion | 1200.5\nZebramarker€ | -7\n3 | 4\nformula text | TRUE"
    )


def test_ppt_reads_slide_text_and_skips_masters(tmp_path):
    pages = parsers.parse(str(build_ppt(tmp_path / "deck.ppt")))
    assert [(p.page_number, p.text) for p in pages] == [
        (1, "Quarterly review\nRevenue up"),
        (2, "Zebramarker slide"),
    ]


def test_ppt_falls_back_to_slide_drawings(tmp_path):
    pages = parsers.parse(str(build_ppt(tmp_path / "deck.ppt", from_slides = True)))
    assert [p.text for p in pages] == ["Drawn slide", "Zebramarker drawn"]


def test_msg_reads_headers_body_and_attachment_names(tmp_path):
    assert _text(build_msg(tmp_path / "mail.msg")) == (
        "From: Ann <ann@example.com>\nTo: Bob\nDate: 2026-10-06 10:00 UTC\nSubject: Q1 numbers\n"
        "Attachments: data.csv\n\nRevenue increased.\nZebramarker in the body."
    )


def test_compound_file_reads_mini_and_regular_streams():
    small, large = b"a" * 100, bytes(range(256)) * 40
    reader = cfb.CompoundFile(compound_file({("small",): small, ("dir", "large"): large}))
    assert reader.open("small") == small
    assert reader.open("DIR", "Large") == large
    assert sorted(reader.listdir()) == ["dir", "small"]


# ---------------------------------------------------------------- failures


def test_encrypted_doc_is_refused(tmp_path):
    with pytest.raises(ValueError, match = "password"):
        parsers.parse(str(build_doc(tmp_path / "secret.doc", encrypted = True)))


@pytest.mark.parametrize("extension", [".doc", ".xls", ".ppt", ".msg"])
def test_non_compound_file_is_refused(tmp_path, extension):
    path = tmp_path / f"fake{extension}"
    path.write_bytes(b"not a compound file" * 50)
    with pytest.raises(ValueError):
        parsers.parse(str(path))


@pytest.mark.parametrize("extension", [".xlsx", ".pptx", ".odt", ".epub"])
def test_damaged_archive_is_refused(tmp_path, extension):
    path = tmp_path / f"broken{extension}"
    path.write_bytes(b"PK\x03\x04 truncated")
    with pytest.raises(ValueError):
        parsers.parse(str(path))


def test_xml_entities_are_refused(tmp_path):
    path = _zip(
        tmp_path / "bomb.odt", {"content.xml": '<!DOCTYPE x [<!ENTITY a "aaaa">]><x>&a;</x>'}
    )
    with pytest.raises(ValueError):
        parsers.parse(str(path))


# ---------------------------------------------------------------- upload and indexing


@pytest.mark.parametrize("extension", sorted(BUILDERS))
def test_documents_upload_index_and_stay_searchable(rag_home, stub_embeddings, tmp_path, extension):
    source = BUILDERS[extension](tmp_path / f"report{extension}")
    stored, filename, _hash = _save_upload(
        UploadFile(file = io.BytesIO(source.read_bytes()), filename = source.name)
    )
    assert filename == source.name
    scope = store.thread_scope("office")
    doc_id, job = ingestion.start_ingestion(
        scope, None, "office", filename, stored, background = False
    )
    assert ingestion.get_job_status(job)["status"] == "completed"
    conn = rag_db.get_connection()
    try:
        assert store.get_document(conn, doc_id)["status"] == "completed"
        assert store.search_lexical(conn, scope, "zebramarker", 5)
    finally:
        conn.close()
