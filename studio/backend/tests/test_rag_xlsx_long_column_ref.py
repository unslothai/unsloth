# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A cell reference with a huge column part must not cost quadratic time per cell."""

import time
import zipfile

from core.rag import office_formats

S = 'xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"'
REL = 'xmlns="http://schemas.openxmlformats.org/package/2006/relationships"'


def _book(path, ref):
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(
            "xl/workbook.xml",
            f'<workbook {S}><sheets><sheet name="S" sheetId="1" r:id="rId1"/></sheets></workbook>',
        )
        z.writestr(
            "xl/_rels/workbook.xml.rels",
            f'<Relationships {REL}><Relationship Id="rId1" Target="worksheets/sheet1.xml"/></Relationships>',
        )
        cells = "".join(f'<c r="{ref}" t="inlineStr"><is><t>far</t></is></c>' for _ in range(4))
        z.writestr(
            "xl/worksheets/sheet1.xml",
            f"<worksheet {S}><sheetData>"
            f'<row r="1"><c r="A1" t="inlineStr"><is><t>kept</t></is></c>'
            f'<c r="B1" t="inlineStr"><is><t>also kept</t></is></c>{cells}</row>'
            "</sheetData></worksheet>",
        )
    return path


def test_normal_references_are_unchanged():
    assert office_formats._column_index("A1") == 0
    assert office_formats._column_index("AB12") == 27
    assert office_formats._column_index("AMJ1") == office_formats._MAX_COLUMNS - 1


def test_a_huge_column_reference_is_skipped_quickly(tmp_path):
    ref = "A" * 200_000 + "1"
    start = time.monotonic()
    assert office_formats._column_index(ref) >= office_formats._MAX_COLUMNS
    sections = office_formats.xlsx(str(_book(tmp_path / "book.xlsx", ref)))
    elapsed = time.monotonic() - start
    text = "\n".join(section[0] for section in sections)
    assert "kept" in text and "also kept" in text
    assert "far" not in text
    assert elapsed < 2.0, f"a long column reference took {elapsed:.1f} s"
