# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""PDF Markdown skips pymupdf4llm's lines_strict table search only where it cannot find a table."""

from __future__ import annotations

import random

import pytest

pymupdf = pytest.importorskip("pymupdf")
pytest.importorskip("pymupdf4llm")

from core.rag import config, parsers  # noqa: E402


def _guard_on() -> bool:
    # Environment facts, not the parser's own check, so these tests also run (and fail) without it.
    import pymupdf4llm
    return (
        pymupdf4llm.to_markdown.__module__ == "pymupdf4llm.helpers.pymupdf_rag"
        and pymupdf.__version__ in ("1.27.2.3", "1.28.2")
        and getattr(pymupdf, "_get_layout", None) is None
    )


requires_guard = pytest.mark.skipif(
    not _guard_on(), reason = "table guard is off for this PyMuPDF / pymupdf4llm build"
)


def _text(page, y, line):
    page.insert_text((60, y), line, fontsize = 11)


def _lined_table(
    page,
    x0 = 60,
    y0 = 300,
    rows = 3,
    cols = 3,
    w = 120,
    h = 24,
):
    shape = page.new_shape()
    for r in range(rows + 1):
        shape.draw_line((x0, y0 + r * h), (x0 + cols * w, y0 + r * h))
    for c in range(cols + 1):
        shape.draw_line((x0 + c * w, y0), (x0 + c * w, y0 + rows * h))
    shape.finish(color = (0, 0, 0), width = 0.8)
    shape.commit()
    for r in range(rows):
        for c in range(cols):
            page.insert_text((x0 + c * w + 6, y0 + r * h + 16), f"r{r}c{c} {r * c}", fontsize = 10)


def _sample_pdf(path):
    """Pages: plain text, lined table, text-aligned table without lines, one rule,
    a large background fill, a rotated lined table, a boxed note."""
    doc = pymupdf.open()
    p = doc.new_page()
    _text(p, 80, "Chapter One")
    for i in range(12):
        _text(p, 110 + 16 * i, f"Plain body line {i} with words to read.")
    p = doc.new_page()
    _text(p, 80, "Quarterly results by region")
    _lined_table(p)
    p = doc.new_page()
    for r, row in enumerate(
        [("Quarter", "Revenue", "Growth"), ("Q1", "$1.2M", "12%"), ("Q2", "$1.5M", "25%")]
    ):
        for c, cell in enumerate(row):
            p.insert_text((60 + 150 * c, 120 + 24 * r), cell, fontsize = 11)
    p = doc.new_page()
    _text(p, 80, "Body above a footnote rule.")
    p.draw_line((60, 700), (300, 700), width = 0.5)
    _text(p, 720, "1. A footnote.")
    p = doc.new_page()
    p.draw_rect(p.rect, color = None, fill = (0.95, 0.95, 0.9))
    _text(p, 80, "Text on a tinted page.")
    p = doc.new_page()
    _text(p, 80, "Rotated table")
    _lined_table(p, rows = 4, cols = 2)
    p.set_rotation(90)
    p = doc.new_page()
    p.draw_rect(pymupdf.Rect(50, 100, 400, 160), color = (0, 0, 0), width = 1)
    _text(p, 70, "Note box")
    _text(p, 130, "Inside a single outlined box.")
    doc.save(str(path))
    doc.close()


def _parse(path, monkeypatch, guard: bool):
    monkeypatch.setattr(config, "PDF_MARKDOWN", True)
    if not guard:
        monkeypatch.setattr(parsers, "_table_guard_enabled", lambda _module: False)
    pages = parsers.parse(str(path))
    monkeypatch.undo()
    return [(p.text, p.page_number, p.char_count, p.needs_ocr) for p in pages]


def _count_find_tables(monkeypatch):
    calls = []
    real = pymupdf.Page.find_tables

    def counting(self, *args, **kwargs):
        calls.append(self.number)
        return real(self, *args, **kwargs)

    monkeypatch.setattr(pymupdf.Page, "find_tables", counting)
    return calls


@requires_guard
def test_markdown_identical_with_and_without_guard(tmp_path, monkeypatch):
    pdf = tmp_path / "mixed.pdf"
    _sample_pdf(pdf)
    with_guard = _parse(pdf, monkeypatch, guard = True)
    without = _parse(pdf, monkeypatch, guard = False)
    assert with_guard == without
    assert "|" in with_guard[1][0] and "r2c2 4" in with_guard[1][0]
    assert "|" in with_guard[5][0]
    assert "Plain body line 11" in with_guard[0][0]


@requires_guard
def test_guard_skips_table_search_on_pages_without_line_art(tmp_path, monkeypatch):
    pdf = tmp_path / "mixed.pdf"
    _sample_pdf(pdf)
    calls = _count_find_tables(monkeypatch)
    monkeypatch.setattr(config, "PDF_MARKDOWN", True)
    parsers.parse(str(pdf))
    # Only the lined table, the rotated page and the outlined box (4 edges) keep the search.
    assert sorted(calls) == [1, 5, 6]


@requires_guard
def test_guard_engages_on_a_text_only_book(tmp_path, monkeypatch):
    doc = pymupdf.open()
    for n in range(6):
        p = doc.new_page()
        for i in range(30):
            _text(p, 60 + 20 * i, f"Page {n} line {i} of a book without any drawings.")
    pdf = tmp_path / "book.pdf"
    doc.save(str(pdf))
    doc.close()
    calls = _count_find_tables(monkeypatch)
    monkeypatch.setattr(config, "PDF_MARKDOWN", True)
    pages = parsers.parse(str(pdf))
    assert len(pages) == 6 and "Page 5 line 29" in pages[5].text
    assert calls == []


@requires_guard
def test_guard_keeps_form_and_annotation_output(tmp_path, monkeypatch):
    doc = pymupdf.open()
    p = doc.new_page()
    _text(p, 80, "Applicant details")
    widget = pymupdf.Widget()
    widget.field_name = "full_name"
    widget.field_type = pymupdf.PDF_WIDGET_TYPE_TEXT
    widget.field_value = "Ada Lovelace"
    widget.rect = pymupdf.Rect(60, 100, 300, 120)
    p.add_widget(widget)
    p = doc.new_page()
    _text(p, 80, "Reviewed page")
    p.add_highlight_annot(pymupdf.Rect(58, 68, 200, 84))
    p.add_text_annot((300, 300), "check this")
    _lined_table(p, rows = 2, cols = 2)
    pdf = tmp_path / "form.pdf"
    doc.save(str(pdf))
    doc.close()
    assert _parse(pdf, monkeypatch, guard = True) == _parse(pdf, monkeypatch, guard = False)


def _kept_tables(page) -> int:
    tabs = page.find_tables(clip = page.rect, strategy = "lines_strict")
    return sum(1 for t in tabs.tables if t.row_count >= 2 and t.col_count >= 2)


def test_guard_is_sound_on_random_line_art():
    # The guard may only say "no table" where find_tables keeps none.
    rng = random.Random(1234)
    doc = pymupdf.open()
    for _ in range(80):
        p = doc.new_page()
        _text(p, 40, "random line art")
        shape = p.new_shape()
        for _ in range(rng.randint(0, 6)):
            kind = rng.random()
            x, y = rng.uniform(40, 400), rng.uniform(60, 700)
            if kind < 0.4:
                dx, dy = (
                    (rng.uniform(5, 200), rng.choice([0, 0, 0.5, 2]))
                    if rng.random() < 0.5
                    else (rng.choice([0, 0, 0.5, 2]), rng.uniform(5, 200))
                )
                shape.draw_line((x, y), (x + dx, y + dy))
            elif kind < 0.7:
                shape.draw_rect(
                    pymupdf.Rect(x, y, x + rng.uniform(0.5, 150), y + rng.uniform(0.5, 80))
                )
            elif kind < 0.85:
                shape.draw_quad(
                    pymupdf.Rect(x, y, x + rng.uniform(2, 90), y + rng.uniform(2, 60)).quad
                )
            else:
                shape.draw_circle((x, y), rng.uniform(2, 30))
            shape.finish(color = (0, 0, 0), fill = (0.5, 0.5, 0.5) if rng.random() < 0.3 else None)
        shape.commit()
        for i in range(rng.randint(0, 6)):
            p.insert_text((rng.uniform(40, 400), rng.uniform(60, 760)), f"cell {i}", fontsize = 9)
        if rng.random() < 0.3:
            _lined_table(
                p,
                x0 = rng.uniform(40, 200),
                y0 = rng.uniform(100, 500),
                rows = rng.randint(1, 3),
                cols = rng.randint(1, 3),
            )
    skipped = 0
    for page in doc:
        if parsers._no_lined_table_possible(page):
            skipped += 1
            assert _kept_tables(page) == 0, page.number
    assert 0 < skipped < doc.page_count


def test_guard_never_skips_a_page_with_a_kept_table():
    doc = pymupdf.open()
    for rows, cols in [(2, 2), (3, 2), (2, 5), (6, 6)]:
        p = doc.new_page()
        _lined_table(p, rows = rows, cols = cols, w = 80)
        assert _kept_tables(p) >= 1
        assert not parsers._no_lined_table_possible(p)
    p = doc.new_page()
    _text(p, 80, "No drawings")
    assert parsers._no_lined_table_possible(p)


def test_guard_off_for_other_markdown_backends():
    class _Layout:
        @staticmethod
        def to_markdown(doc, **kwargs):
            return []

    assert not parsers._table_guard_enabled(_Layout)
