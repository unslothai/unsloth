# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for the OpenAI Responses-API citation marker rewriter.

The Responses stream interleaves text deltas with citation markers
shaped like ``\\ue200cite\\ue202SOURCE_ID\\ue201``. The private-use
code points either render as garbled "E202" glyphs in most fonts or
get stripped by the markdown layer, leaving run-on text like
"citeturn1view0turn1view1...". The rewriter translates each marker
into a markdown link `[N](URL)` when the matching annotation has
arrived, and drops the marker silently otherwise so the prose stays
clean. The full URL list still flows through to the Sources panel via
`_record_url_citation`.

Reference: https://developers.openai.com/api/docs/guides/citation-formatting
"""

import pytest

from core.inference.external_provider import _replace_openai_citation_markers


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
        {"source_id": "turn0view0", "url": "https://example.com/paris", "title": "Paris"},
    ]
    out = _replace_openai_citation_markers(text, citations)
    assert not _has_marker_codepoints(out)
    assert "[[1]](https://example.com/paris)" in out


def test_unknown_source_marker_dropped_silently():
    text = f"Foo {_marker('turn9view9')} bar."
    out = _replace_openai_citation_markers(text, [])
    # Marker stripped, no garbled "E202" glyph leaks through, and the
    # surrounding text stays intact.
    assert not _has_marker_codepoints(out)
    assert "E202" not in out
    assert "turn9view9" not in out
    assert "Foo" in out and "bar" in out


def test_multiple_concatenated_markers_resolved_in_order():
    """Real-world wire shape: a string of markers butted up against each other
    after a sentence, as in the user-reported bug."""
    markers = "".join(
        _marker(f"turn{i}view{j}") for i, j in [(1, 0), (1, 1), (3, 0)]
    )
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
    """If something upstream already stripped the private-use codepoints
    (leaving readable run-on text like "citeturn1view0"), don't try to
    parse them again -- return verbatim."""
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
