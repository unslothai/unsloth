# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Citation regions follow the displayed PDF page orientation."""

from __future__ import annotations

import pytest

pymupdf = pytest.importorskip("pymupdf")


@pytest.mark.parametrize("rotation", [0, 90, 180, 270])
@pytest.mark.parametrize("cropped", [False, True])
def test_pdf_citation_regions_cover_rotated_text(tmp_path, rotation, cropped):
    from core.rag.chunking import chunk_pages
    from core.rag.locators import pdf_regions_for_chunks
    from core.rag.parsers import parse

    path = tmp_path / "rotated.pdf"
    with pymupdf.open() as doc:
        page = doc.new_page(width = 360, height = 540)
        page.insert_text((60, 100), "alpha beta gamma delta", fontsize = 12)
        if cropped:
            page.set_cropbox(pymupdf.Rect(20, 30, 320, 510))
        page.set_rotation(rotation)
        doc.save(path)

    pages = parse(str(path))
    chunks = chunk_pages(pages, max_tokens = 500, overlap = 0, count = len)
    regions = pdf_regions_for_chunks(path, pages, chunks)
    assert len(regions) == 1 and len(regions[0]) == 1
    region = regions[0][0]
    assert region["pageIndex"] == 0 and region["pageNumber"] == 1

    # The preview places these percentages over the rendered page. Compare with
    # actual dark pixels so extraction coordinates cannot supply the same error.
    with pymupdf.open(path) as doc:
        pix = doc[0].get_pixmap(colorspace = pymupdf.csGRAY)
        ink = [
            (i % pix.width, i // pix.width) for i, value in enumerate(pix.samples) if value < 128
        ]
    assert ink
    ink_x0, ink_y0 = min(x for x, _ in ink), min(y for _, y in ink)
    ink_x1, ink_y1 = max(x for x, _ in ink) + 1, max(y for _, y in ink) + 1
    x0, y0 = region["x"] * pix.width, region["y"] * pix.height
    x1 = (region["x"] + region["width"]) * pix.width
    y1 = (region["y"] + region["height"]) * pix.height

    # Font ascenders/descenders allow a small margin around visible glyphs;
    # this also rejects a page-sized rectangle that trivially covers the text.
    assert 0 <= ink_x0 - x0 <= 6
    assert 0 <= ink_y0 - y0 <= 6
    assert 0 <= x1 - ink_x1 <= 6
    assert 0 <= y1 - ink_y1 <= 6
