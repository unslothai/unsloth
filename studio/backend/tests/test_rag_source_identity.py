# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for durable source identity in tool XML output (PLAN.md T3, contracts §3).

Acceptance criteria:

- _format_hits_for_llm emits document_id and chunk_id attributes on <chunk> elements.
- The visible citation id (id="N") is a per-call counter, NOT the backend chunk UUID.
- Same-filename documents in different KB slots remain distinguishable by document_id.
- Hits without a matching DB row (lookup miss) are silently dropped — not emitted
  with wrong IDs.
- Legacy hits (no document_id in hit dict) still render without crashing.
"""

from __future__ import annotations

import re
import uuid
from xml.etree import ElementTree

import pytest

from core.rag.tool import _format_hits_for_llm


# ── Helpers ───────────────────────────────────────────────────────────


def _uid() -> str:
    return str(uuid.uuid4())


def _hit(
    *,
    chunk_id: str,
    document_id: str,
    filename: str = "report.pdf",
    text: str = "some text",
    page_number: int | None = 3,
    chunk_index: int = 0,
    score: float = 0.85,
) -> dict:
    """Build a flat hit dict as _format_hits_for_llm expects."""
    return {
        "chunk_id": chunk_id,
        "document_id": document_id,
        "filename": filename,
        "text": text,
        "page_number": page_number,
        "chunk_index": chunk_index,
        "score": score,
        "dense_score": score,
        "token_count": 20,
        "kind": "text",
        "image_path": None,
    }


def _parse_chunks(xml_output: str) -> list[dict]:
    """Parse <chunk ...> elements from the multi-block tool output."""
    chunks = []
    # Each block is wrapped in <chunk ...>\n...\n</chunk>; parse directly.
    for match in re.finditer(r"<chunk\s([^>]*)>", xml_output):
        attrs_raw = match.group(1)
        # Quick attribute parser for "key="value"" pairs.
        attrs: dict = {}
        for m in re.finditer(r'(\w+)="([^"]*)"', attrs_raw):
            attrs[m.group(1)] = m.group(2)
        chunks.append(attrs)
    return chunks


# ── Tests: durable IDs present in XML ────────────────────────────────


def test_format_hits_emits_document_id_and_chunk_id():
    """T3: tool XML <chunk> must carry document_id and chunk_id attributes."""
    chunk_id, doc_id = _uid(), _uid()
    hits = [_hit(chunk_id = chunk_id, document_id = doc_id)]
    output = _format_hits_for_llm(hits)
    chunks = _parse_chunks(output)
    assert len(chunks) == 1, output
    assert chunks[0]["document_id"] == doc_id
    assert chunks[0]["chunk_id"] == chunk_id


def test_citation_id_is_sequential_counter_not_uuid():
    """Visible id='N' is a 1-based counter — never equal to the backend chunk UUID."""
    chunk_id, doc_id = _uid(), _uid()
    hits = [_hit(chunk_id = chunk_id, document_id = doc_id)]
    output = _format_hits_for_llm(hits, start_id = 0)
    chunks = _parse_chunks(output)
    visible_id = chunks[0]["id"]
    # Must be a small integer string, NOT the UUID
    assert visible_id == "1", f"expected '1' got {visible_id!r}"
    assert visible_id != chunk_id


def test_citation_ids_are_globally_sequential_across_calls():
    """start_id offset ensures IDs stay unique across multiple tool calls per turn."""
    hits_call1 = [_hit(chunk_id = _uid(), document_id = _uid(), filename = "a.pdf")]
    hits_call2 = [
        _hit(chunk_id = _uid(), document_id = _uid(), filename = "b.pdf"),
        _hit(chunk_id = _uid(), document_id = _uid(), filename = "c.pdf"),
    ]
    out1 = _format_hits_for_llm(hits_call1, start_id = 0)
    out2 = _format_hits_for_llm(hits_call2, start_id = 1)

    chunks1 = _parse_chunks(out1)
    chunks2 = _parse_chunks(out2)

    assert chunks1[0]["id"] == "1"
    assert chunks2[0]["id"] == "2"
    assert chunks2[1]["id"] == "3"

    # No id overlap
    all_ids = {c["id"] for c in chunks1 + chunks2}
    assert len(all_ids) == 3


def test_same_filename_docs_have_distinct_document_ids():
    """Two docs with the same filename route to distinct document_id values (Risk #4)."""
    filename = "annual-report.pdf"
    chunk_a, doc_a = _uid(), _uid()
    chunk_b, doc_b = _uid(), _uid()
    hits = [
        _hit(chunk_id = chunk_a, document_id = doc_a, filename = filename),
        _hit(chunk_id = chunk_b, document_id = doc_b, filename = filename),
    ]
    output = _format_hits_for_llm(hits)
    chunks = _parse_chunks(output)
    assert len(chunks) == 2
    # Both use the same filename but MUST have distinct document_id values
    assert chunks[0]["document_id"] != chunks[1]["document_id"]
    assert chunks[0]["document_id"] == doc_a
    assert chunks[1]["document_id"] == doc_b


def test_same_filename_docs_have_distinct_citation_ids():
    """Same-filename docs in the same turn still get distinct visible [N] ids."""
    filename = "notes.pdf"
    chunk_a, doc_a = _uid(), _uid()
    chunk_b, doc_b = _uid(), _uid()
    hits = [
        _hit(chunk_id = chunk_a, document_id = doc_a, filename = filename),
        _hit(chunk_id = chunk_b, document_id = doc_b, filename = filename),
    ]
    output = _format_hits_for_llm(hits)
    chunks = _parse_chunks(output)
    citation_ids = {c["id"] for c in chunks}
    assert len(citation_ids) == 2, f"citation IDs not unique: {chunks}"


def test_empty_hits_returns_no_chunks_message():
    """Empty hit list returns the 'no matching chunks' message, not broken XML."""
    output = _format_hits_for_llm([])
    chunks = _parse_chunks(output)
    assert len(chunks) == 0
    assert "no matching chunks" in output.lower() or "no matching" in output.lower()


def test_page_number_attribute_present_when_page_exists():
    """page attribute is emitted when page_number is not None."""
    chunk_id, doc_id = _uid(), _uid()
    hits = [_hit(chunk_id = chunk_id, document_id = doc_id, page_number = 5)]
    output = _format_hits_for_llm(hits)
    chunks = _parse_chunks(output)
    assert chunks[0].get("page") == "5"


def test_page_number_attribute_absent_when_null():
    """page attribute is omitted when page_number is None."""
    chunk_id, doc_id = _uid(), _uid()
    hits = [_hit(chunk_id = chunk_id, document_id = doc_id, page_number = None)]
    output = _format_hits_for_llm(hits)
    chunks = _parse_chunks(output)
    assert "page" not in chunks[0], f"unexpected page attr: {chunks[0]}"


def test_locator_attributes_are_additive_when_present():
    """T10: tool XML carries nullable locator metadata without changing visible ids."""
    chunk_id, doc_id = _uid(), _uid()
    hit = _hit(chunk_id = chunk_id, document_id = doc_id, page_number = 5)
    hit.update(
        {
            "source_page_index": 4,
            "page_char_start": 11,
            "page_char_end": 42,
            "line_start": 2,
            "line_end": 3,
        }
    )
    output = _format_hits_for_llm([hit])
    chunk = _parse_chunks(output)[0]
    assert chunk["id"] == "1"
    assert chunk["chunk_id"] == chunk_id
    assert chunk["source_page_index"] == "4"
    assert chunk["page_char_start"] == "11"
    assert chunk["page_char_end"] == "42"
    assert chunk["line_start"] == "2"
    assert chunk["line_end"] == "3"


def test_xml_special_chars_in_filename_escaped():
    """Filename with XML special chars does not break the chunk element."""
    chunk_id, doc_id = _uid(), _uid()
    hits = [
        _hit(
            chunk_id = chunk_id,
            document_id = doc_id,
            filename = 'report <2025> "final" & draft.pdf',
        )
    ]
    output = _format_hits_for_llm(hits)
    # The output must parse cleanly (no unescaped < or " in attrs)
    chunks = _parse_chunks(output)
    assert len(chunks) == 1
    # source attribute should have the filename escaped
    source_attr = chunks[0].get("source", "")
    assert "<" not in source_attr and '"' not in source_attr


def test_multiple_hits_carry_independent_ids():
    """Three hits each carry their own distinct chunk_id and document_id."""
    hit_data = [
        (_uid(), _uid()),
        (_uid(), _uid()),
        (_uid(), _uid()),
    ]
    hits = [
        _hit(chunk_id = cid, document_id = did, filename = f"doc{i}.pdf")
        for i, (cid, did) in enumerate(hit_data)
    ]
    output = _format_hits_for_llm(hits)
    chunks = _parse_chunks(output)
    assert len(chunks) == 3
    emitted_chunk_ids = {c["chunk_id"] for c in chunks}
    emitted_doc_ids = {c["document_id"] for c in chunks}
    expected_chunk_ids = {cid for cid, _ in hit_data}
    expected_doc_ids = {did for _, did in hit_data}
    assert emitted_chunk_ids == expected_chunk_ids
    assert emitted_doc_ids == expected_doc_ids
