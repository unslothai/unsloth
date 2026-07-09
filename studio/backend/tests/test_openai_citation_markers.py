# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for the OpenAI Responses-API citation marker rewriter.

The stream interleaves text deltas with ``\\ue200cite\\ue202SOURCE_ID\\ue201``
markers. The rewriter resolves each to `[N](URL)` once the annotation arrives
and drops it otherwise; the URL list still flows to Sources via
`_record_url_citation`.

Reference: https://developers.openai.com/api/docs/guides/citation-formatting
"""

import pytest

from core.inference.external_provider import (
    _replace_openai_citation_markers,
    _rewrite_citation_markers_partial,
)


# Citation marker control codepoints (private-use area):
CITE_START = ""
CITE_STOP = ""
CITE_DELIM = ""


def _marker(source_id: str, locator: str | None = None) -> str:
    payload = f"{CITE_START}cite{CITE_DELIM}{source_id}"
    if locator:
        payload = f"{payload}{CITE_DELIM}{locator}"
    return f"{payload}{CITE_STOP}"


def _has_marker_codepoints(text: str) -> bool:
    return any(c in text for c in (CITE_START, CITE_STOP, CITE_DELIM))


def test_passthrough_when_no_marker_present():
    text = "Plain text with no citation markers."
    assert _replace_openai_citation_markers(text, []) == text


def test_marker_rewritten_to_link_when_annotation_known():
    text = f"The capital is Paris {_marker('turn0view0')}."
    citations = [
        {
            "source_id": "turn0view0",
            "url": "https://example.com/paris",
            "title": "Paris",
        },
    ]
    out = _replace_openai_citation_markers(text, citations)
    assert not _has_marker_codepoints(out)
    assert "[[1]](https://example.com/paris)" in out


def test_unknown_source_marker_dropped_silently():
    text = f"Foo {_marker('turn9view9')} bar."
    out = _replace_openai_citation_markers(text, [])
    # Marker stripped, no garbled "E202" glyph leaks, surrounding text intact.
    assert not _has_marker_codepoints(out)
    assert "E202" not in out
    assert "turn9view9" not in out
    assert "Foo" in out and "bar" in out


def test_multiple_concatenated_markers_resolved_in_order():
    """Real-world wire shape: markers butted together after a sentence (user-reported bug)."""
    markers = "".join(_marker(f"turn{i}view{j}") for i, j in [(1, 0), (1, 1), (3, 0)])
    text = f"All animals ranked. {markers}"
    citations = [
        {"source_id": "turn1view0", "url": "https://a.example/dog", "title": "Dog"},
        {"source_id": "turn1view1", "url": "https://a.example/cat", "title": "Cat"},
        {"source_id": "turn3view0", "url": "https://a.example/tiger", "title": "Tiger"},
    ]
    out = _replace_openai_citation_markers(text, citations)
    assert "[[1]](https://a.example/dog)" in out
    assert "[[2]](https://a.example/cat)" in out
    assert "[[3]](https://a.example/tiger)" in out
    assert not _has_marker_codepoints(out)


def test_marker_with_locator_resolves():
    text = f"See {_marker('turn2file0', 'L8-L13')}."
    citations = [
        {"source_id": "turn2file0", "url": "https://example.com/doc.txt"},
    ]
    out = _replace_openai_citation_markers(text, citations)
    assert "[[1]](https://example.com/doc.txt)" in out
    assert "L8-L13" not in out  # locator detail dropped; we just link.
    assert not _has_marker_codepoints(out)


def test_mixed_known_and_unknown_markers():
    known = _marker("turn0view0")
    unknown = _marker("turn0view99")
    text = f"Known {known} and unknown {unknown}."
    citations = [
        {"source_id": "turn0view0", "url": "https://example.com/known"},
    ]
    out = _replace_openai_citation_markers(text, citations)
    assert "[[1]](https://example.com/known)" in out
    # Unknown markers leave no trace, but surrounding prose stays.
    assert "Known" in out and "unknown" in out
    assert not _has_marker_codepoints(out)
    assert "E202" not in out


def test_empty_text_returns_verbatim():
    assert _replace_openai_citation_markers("", []) == ""


def test_idempotent_on_pre_stripped_text():
    """Pre-stripped text (no private-use codepoints) returns verbatim."""
    text = "citeturn1view0 plain"
    assert _replace_openai_citation_markers(text, []) == text


@pytest.mark.parametrize(
    "citation",
    [
        {"url": "https://example.com/a"},  # no source_id at all
        {"source_id": None, "url": "https://example.com/b"},
        {"source_id": "", "url": "https://example.com/c"},
    ],
)
def test_citation_without_source_id_does_not_crash(citation):
    text = f"X {_marker('turnXviewY')} Y"
    out = _replace_openai_citation_markers(text, [citation])
    # No mapping, marker stripped. Crash-free is the contract.
    assert not _has_marker_codepoints(out)
    assert "turnXviewY" not in out


def test_multiple_source_id_aliases_resolve_to_same_url():
    """Every alias for the same URL must resolve, not just the first (Codex P1 regression)."""
    a = _marker("turn0view0")
    b = _marker("turn0view0_span_1")
    c = _marker("turn0view0_span_2")
    text = f"Triple {a}{b}{c} cite."
    citations = [
        {
            "source_ids": ["turn0view0", "turn0view0_span_1", "turn0view0_span_2"],
            "url": "https://example.com/paris",
            "title": "Paris",
        },
    ]
    out = _replace_openai_citation_markers(text, citations)
    # All three aliases collapse onto citation [1] -- same URL, so showing
    # three different numbers would mislead.
    assert out.count("[[1]](https://example.com/paris)") == 3
    assert not _has_marker_codepoints(out)


def test_source_ids_list_and_legacy_source_id_both_resolve():
    """Mixed-shape citation: legacy ``source_id`` plus newer ``source_ids`` aliases both resolve."""
    legacy = _marker("legacy_id")
    alias = _marker("alias_id")
    text = f"Both {legacy} and {alias} work."
    citations = [
        {
            "source_id": "legacy_id",
            "source_ids": ["alias_id"],
            "url": "https://example.com/doc",
        },
    ]
    out = _replace_openai_citation_markers(text, citations)
    assert out.count("[[1]](https://example.com/doc)") == 2
    assert not _has_marker_codepoints(out)


# _rewrite_citation_markers_partial: deferred-annotation tests. OpenAI emits
# url_citation annotations on a later SSE event; this helper reports
# `has_unresolved` so the stream loop defers emission. See PR #5713 audit.


def test_partial_known_marker_resolves_and_clears_unresolved():
    text = f"Foo {_marker('s1')} bar."
    out, unresolved = _rewrite_citation_markers_partial(
        text,
        [{"source_id": "s1", "url": "https://example.com/a"}],
    )
    assert "[[1]](https://example.com/a)" in out
    assert unresolved is False
    assert not _has_marker_codepoints(out)


def test_partial_unknown_marker_preserves_verbatim_and_flags():
    text = f"Foo {_marker('s1')} bar."
    out, unresolved = _rewrite_citation_markers_partial(text, [])
    assert unresolved is True
    # Codepoints must remain so a follow-up pass can re-parse.
    assert _has_marker_codepoints(out)
    assert "Foo" in out and "bar." in out


def test_partial_resolves_after_late_annotation():
    """Two-pass: first call sees no citations; second resolves after annotation."""
    text = f"See {_marker('s1')} for details."
    out1, unresolved1 = _rewrite_citation_markers_partial(text, [])
    assert unresolved1 is True
    citations = [{"source_id": "s1", "url": "https://example.com/x"}]
    out2, unresolved2 = _rewrite_citation_markers_partial(out1, citations)
    assert unresolved2 is False
    assert "[[1]](https://example.com/x)" in out2
    assert not _has_marker_codepoints(out2)


def test_partial_multi_source_partial_resolution_keeps_marker_pending():
    """Any unresolved token in a multi-source marker leaves the whole marker verbatim with ``unresolved`` True until every id resolves or end-of-stream flushes."""
    cite = f"{CITE_START}cite{CITE_DELIM}known{CITE_DELIM}locator{CITE_STOP}"
    text = f"Pre {cite} post."
    citations = [{"source_id": "known", "url": "https://example.com/y"}]
    out, unresolved = _rewrite_citation_markers_partial(text, citations)
    assert unresolved is True
    assert cite in out
    # End-of-stream force flush: drop the unresolved token, keep the resolved
    # link. The streamer routes pending segments through
    # `_replace_openai_citation_markers` at force=True for this.
    forced = _replace_openai_citation_markers(out, citations)
    assert "[[1]](https://example.com/y)" in forced
    assert "locator" not in forced
    assert not _has_marker_codepoints(forced)


def test_partial_idempotent_on_marker_free_text():
    text = "Plain text."
    out, unresolved = _rewrite_citation_markers_partial(text, [])
    assert out == text
    assert unresolved is False


def test_partial_mixed_known_and_pending_markers_flags_unresolved():
    known = _marker("known")
    pending = _marker("pending")
    text = f"{known} {pending}"
    citations = [{"source_id": "known", "url": "https://example.com/k"}]
    out, unresolved = _rewrite_citation_markers_partial(text, citations)
    assert unresolved is True  # the pending marker drives the flag
    assert "[[1]](https://example.com/k)" in out
    # The pending marker stays verbatim for the next pass.
    assert CITE_START in out and "pending" in out
