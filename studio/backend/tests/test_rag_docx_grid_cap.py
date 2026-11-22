# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A few bytes of declared Word table grid (gridSpan / gridBefore / gridAfter) must not expand into a giant grid."""

from __future__ import annotations

import pytest


def _doc_with(tmp_path, tag, value, name):
    pytest.importorskip("docx")
    import docx
    from docx.oxml.ns import qn

    document = docx.Document()
    table = document.add_table(rows = 1, cols = 2)
    table.cell(0, 0).text = "LEFT"
    table.cell(0, 1).text = "RIGHT"
    tr = table.rows[0]._tr
    if tag == "w:gridSpan":
        tc_pr = tr.tc_lst[0].get_or_add_tcPr()
        tc_pr.append(tc_pr.makeelement(qn(tag), {qn("w:val"): str(value)}))
    else:
        tr_pr = tr.get_or_add_trPr()
        tr_pr.insert(0, tr_pr.makeelement(qn(tag), {qn("w:val"): str(value)}))
    path = tmp_path / name
    document.save(str(path))
    return path


@pytest.mark.parametrize("tag", ["w:gridSpan", "w:gridBefore", "w:gridAfter"])
def test_declared_grid_is_capped(tmp_path, tag):
    from core.rag import parsers

    path = _doc_with(tmp_path, tag, 5_000_000, "huge.docx")
    text = "\n".join(p.text for p in parsers.parse(str(path)))
    assert "LEFT" in text and "RIGHT" in text
    assert len(text) < 100_000, len(text)


def test_normal_grid_unchanged(tmp_path):
    from core.rag import parsers

    path = _doc_with(tmp_path, "w:gridBefore", 2, "normal.docx")
    text = "\n".join(p.text for p in parsers.parse(str(path)))
    assert " |  | LEFT | RIGHT" in text
