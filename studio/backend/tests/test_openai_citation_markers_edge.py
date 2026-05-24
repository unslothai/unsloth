# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Edge-case tests for the OpenAI Responses citation marker rewriter.

These cover:

  * Multi-source markers (one ``cite`` element referencing several
    ``source_id`` aliases at once -- per the OpenAI doc, the wire shape
    is ``\\ue200cite\\ue202id1\\ue202id2\\ue201``).
  * Source id + locator markers (``\\ue200cite\\ue202sid\\ue202L8-L13
    \\ue201`` -- the locator is dropped, the link still resolves).
  * Marker SPLIT across two streaming ``response.output_text.delta``
    events. OpenAI's stream chunks text on byte-buffer boundaries
    that have no awareness of the marker grammar, so a marker can be
    cut anywhere. The rewriter must buffer the unterminated tail and
    prepend it onto the next delta so the user never sees
    ``\\ue200citetu`` followed by ``rn0view0\\ue201`` (i.e. a garbled
    glyph then a stray "rn0view0" word).
  * Unterminated marker at the end of a stream (truncation): the
    held-over tail is stripped of private-use bytes on flush so no
    glyph leaks.
  * Multiple unrelated markers within a single delta still resolve.
  * Annotation-before-marker ordering (the annotation arrives on the
    same delta as the marker -- the rewriter must see it in the
    lookup table).
  * Idempotency: running the rewriter twice produces the same output.

Reference: https://developers.openai.com/api/docs/guides/citation-formatting
"""

import importlib


# Import the module-level helpers under test. The streaming integration
# is exercised by ``_simulate_delta_stream`` further down which mirrors
# the head/buffer/flush dance from ``_stream_openai_responses``.
_module = importlib.import_module("core.inference.external_provider")
_replace_openai_citation_markers = _module._replace_openai_citation_markers
_split_pending_citation_tail = _module._split_pending_citation_tail


CITE_START = ""
CITE_STOP = ""
CITE_DELIM = ""


def _marker(*source_ids: str, locator: str | None = None) -> str:
    """Build a ``\\ue200cite\\ue202<sid>[\\ue202<sid>...][\\ue202<loc>]\\ue201``
    marker. ``source_ids`` may contain one entry (single-source) or
    several (multi-source). ``locator`` adds a trailing locator token
    after all source ids."""
    payload = f"{CITE_START}cite{CITE_DELIM}" + CITE_DELIM.join(source_ids)
    if locator:
        payload = f"{payload}{CITE_DELIM}{locator}"
    return f"{payload}{CITE_STOP}"


def _no_private_use(text: str) -> bool:
    return all(c not in text for c in (CITE_START, CITE_STOP, CITE_DELIM))


# A tiny harness that mirrors the head / pending-tail / flush dance
# that lives in `_stream_openai_responses`. Used by the streaming
# tests below so we don't have to spin up an httpx mock per case.
def _simulate_delta_stream(
    deltas: list[str],
    citations: list[dict],
    *,
    flush: bool = True,
) -> str:
    pending = ""
    emitted: list[str] = []
    for delta in deltas:
        combined = pending + delta
        head, pending = _split_pending_citation_tail(combined)
        if head:
            head = _replace_openai_citation_markers(head, citations)
            if head:
                emitted.append(head)
    if flush and pending:
        # Mirror `_flush_pending_marker_tail`: try once more in case a
        # late annotation resolved it, then strip private-use bytes.
        rendered = _replace_openai_citation_markers(pending, citations)
        for ch in (CITE_START, CITE_STOP, CITE_DELIM):
            rendered = rendered.replace(ch, "")
        if rendered == "cite":
            rendered = ""
        if rendered:
            emitted.append(rendered)
    return "".join(emitted)


# ---------------------------------------------------------------------------
# 1. Multi-source markers per the OpenAI docs.
# ---------------------------------------------------------------------------


def test_multi_source_marker_all_resolve():
    """\\ue200cite\\ue202id1\\ue202id2\\ue202id3\\ue201 expands to three
    bracket links when every source id is known. The previous regex
    captured only the first source id and silently dropped id2/id3."""
    text = f"All three: {_marker('id1', 'id2', 'id3')}"
    citations = [
        {"source_id": "id1", "url": "https://example.com/1"},
        {"source_id": "id2", "url": "https://example.com/2"},
        {"source_id": "id3", "url": "https://example.com/3"},
    ]
    out = _replace_openai_citation_markers(text, citations)
    assert "[[1]](https://example.com/1)" in out
    assert "[[2]](https://example.com/2)" in out
    assert "[[3]](https://example.com/3)" in out
    assert _no_private_use(out)


def test_multi_source_marker_partial_resolution():
    """Some ids known, some not. Known ids render; unknown ids drop
    silently. No garbled glyph leaks."""
    text = f"Mixed: {_marker('known', 'unknown', 'also_known')}"
    citations = [
        {"source_id": "known", "url": "https://k.example"},
        {"source_id": "also_known", "url": "https://ak.example"},
    ]
    out = _replace_openai_citation_markers(text, citations)
    assert "[[1]](https://k.example)" in out
    assert "[[2]](https://ak.example)" in out
    assert "unknown" not in out
    assert _no_private_use(out)


# ---------------------------------------------------------------------------
# 2. Source + locator: locator is dropped, link still resolves.
# ---------------------------------------------------------------------------


def test_marker_with_numeric_locator():
    text = f"See {_marker('tu0', locator='42')}."
    citations = [{"source_id": "tu0", "url": "https://example.com/doc"}]
    out = _replace_openai_citation_markers(text, citations)
    assert "[[1]](https://example.com/doc)" in out
    assert "42" not in out
    assert _no_private_use(out)


def test_marker_with_range_locator():
    text = f"See {_marker('tu0', locator='L8-L13')}."
    citations = [{"source_id": "tu0", "url": "https://example.com/code"}]
    out = _replace_openai_citation_markers(text, citations)
    assert "[[1]](https://example.com/code)" in out
    assert "L8-L13" not in out
    assert _no_private_use(out)


# ---------------------------------------------------------------------------
# 3. Marker SPLIT across two SSE deltas -- the codex-flagged P1.
# ---------------------------------------------------------------------------


def test_marker_split_in_source_id():
    """Delta-1 ends mid-source-id (``\\ue200cite\\ue202tu``); delta-2
    starts with the rest (``rn0view0\\ue201``). WITHOUT the buffer,
    the rewriter sees each delta in isolation, leaks
    ``\\ue200cite\\ue202tu`` to the user, then leaks ``rn0view0`` and
    the stray ``\\ue201``. WITH the buffer the two halves are
    stitched back together and resolve to one link."""
    full = f"See {_marker('turn0view0')} now."
    # Cut right after the second delim + "tu" inside the source id.
    cut = full.index("tu", full.index(CITE_START)) + len("tu")
    d1, d2 = full[:cut], full[cut:]
    # Sanity check: delta-1 actually contains a partial marker.
    assert CITE_START in d1 and CITE_STOP not in d1
    assert CITE_STOP in d2
    citations = [{"source_id": "turn0view0", "url": "https://x"}]
    out = _simulate_delta_stream([d1, d2], citations)
    assert out == "See [[1]](https://x) now."
    assert _no_private_use(out)


def test_marker_split_at_start_byte():
    """The split is exactly at the opening ``\\ue200`` byte -- the
    next delta starts a fresh marker. The buffer must hold the lone
    open byte until the rest arrives."""
    full = f"Text {_marker('sid')} done"
    cut = full.index(CITE_START) + 1  # right AFTER the open byte
    d1, d2 = full[:cut], full[cut:]
    citations = [{"source_id": "sid", "url": "https://y"}]
    out = _simulate_delta_stream([d1, d2], citations)
    assert out == "Text [[1]](https://y) done"
    assert _no_private_use(out)


def test_marker_split_across_three_deltas():
    """Worst-case split: the marker is chopped into three pieces and
    arrives over three separate deltas with arbitrary text in
    between."""
    full = f"A {_marker('threesplit')} B"
    # cut at two points inside the marker
    open_pos = full.index(CITE_START)
    stop_pos = full.index(CITE_STOP)
    cut1 = open_pos + 4
    cut2 = stop_pos - 2
    parts = [full[:cut1], full[cut1:cut2], full[cut2:]]
    citations = [{"source_id": "threesplit", "url": "https://z"}]
    out = _simulate_delta_stream(parts, citations)
    assert out == "A [[1]](https://z) B"
    assert _no_private_use(out)


def test_marker_split_with_trailing_text_after_close():
    """The second delta closes the marker AND carries more prose
    after the close byte. Both pieces must emit cleanly."""
    full = f"X {_marker('sid')} after"
    cut = full.index("cite") + len("ci")
    d1, d2 = full[:cut], full[cut:]
    citations = [{"source_id": "sid", "url": "https://a"}]
    out = _simulate_delta_stream([d1, d2], citations)
    assert out == "X [[1]](https://a) after"
    assert _no_private_use(out)


def test_split_marker_unknown_source_is_dropped_cleanly():
    """Split marker for an unknown source -- the whole marker drops
    silently on flush, no garbled glyph leaks."""
    full = f"Pre {_marker('never_seen')} post"
    cut = full.index(CITE_START) + 3
    d1, d2 = full[:cut], full[cut:]
    out = _simulate_delta_stream([d1, d2], [])
    assert out == "Pre  post"
    assert _no_private_use(out)


# ---------------------------------------------------------------------------
# 4. Unterminated marker at end-of-stream -- truncation safety.
# ---------------------------------------------------------------------------


def test_unterminated_marker_at_stream_end_dropped_on_flush():
    """Stream ends mid-marker (e.g. response.incomplete). The pending
    tail is flushed: private-use bytes are stripped so no garbled
    glyph or `E202` text leaks."""
    deltas = ["Some text ", f"{CITE_START}citetu", "rn0view0"]  # no STOP ever
    out = _simulate_delta_stream(deltas, [], flush = True)
    assert _no_private_use(out)
    assert "E200" not in out and "E202" not in out
    # Surrounding prose stays. We don't make hard claims on the exact
    # remainder of the unterminated marker -- only that no glyph leaks.
    assert "Some text " in out


def test_flush_resolves_marker_when_late_annotation_arrives():
    """The marker arrives in a delta, but the matching annotation
    only arrives later (e.g. on response.output_text.annotation.added
    after the final delta). Today, the rewriter looks at
    ``all_url_citations`` LIVE at flush time, so a late annotation
    still resolves a buffered marker -- the SAME annotation_first
    re-order applies on flush."""
    deltas = ["Look ", f"{CITE_START}cite{CITE_DELIM}late_sid"]
    pending = ""
    citations: list[dict] = []
    emitted: list[str] = []
    for d in deltas:
        combined = pending + d
        head, pending = _split_pending_citation_tail(combined)
        if head:
            emitted.append(_replace_openai_citation_markers(head, citations))
    # Annotation arrives AFTER all deltas but BEFORE flush.
    citations.append({"source_id": "late_sid", "url": "https://late.example"})
    # Append the STOP byte that closed the marker in a later delta.
    pending = pending + CITE_STOP
    flushed = _replace_openai_citation_markers(pending, citations)
    for ch in (CITE_START, CITE_STOP, CITE_DELIM):
        flushed = flushed.replace(ch, "")
    emitted.append(flushed)
    out = "".join(emitted)
    assert "[[1]](https://late.example)" in out
    assert _no_private_use(out)


# ---------------------------------------------------------------------------
# 5. Multiple unrelated markers in a single delta.
# ---------------------------------------------------------------------------


def test_three_markers_in_one_delta_resolve_independently():
    text = f"alpha {_marker('a')} beta {_marker('b')} gamma {_marker('c')} end"
    citations = [
        {"source_id": "a", "url": "https://example.com/a"},
        {"source_id": "b", "url": "https://example.com/b"},
        {"source_id": "c", "url": "https://example.com/c"},
    ]
    out = _replace_openai_citation_markers(text, citations)
    assert out == (
        "alpha [[1]](https://example.com/a) beta "
        "[[2]](https://example.com/b) gamma "
        "[[3]](https://example.com/c) end"
    )


# ---------------------------------------------------------------------------
# 6. Idempotency.
# ---------------------------------------------------------------------------


def test_rewriter_idempotent_on_already_rewritten_text():
    """Running the rewriter twice doesn't double-link or corrupt the
    markdown brackets it just emitted."""
    text = f"alpha {_marker('a')} omega"
    citations = [{"source_id": "a", "url": "https://example.com/a"}]
    once = _replace_openai_citation_markers(text, citations)
    twice = _replace_openai_citation_markers(once, citations)
    assert once == twice
    assert _no_private_use(once)


def test_rewriter_idempotent_on_marker_free_text():
    """No-op when there is nothing to rewrite."""
    text = "Plain prose with no citations and no private-use bytes."
    out = _replace_openai_citation_markers(text, [])
    assert out is text or out == text


# ---------------------------------------------------------------------------
# 7. Edge / robustness.
# ---------------------------------------------------------------------------


def test_only_marker_no_surrounding_text():
    """A delta that is JUST a marker (no surrounding prose) must
    still render correctly. Without the empty-string short-circuit
    in _split_pending_citation_tail this used to leak."""
    text = _marker("solo")
    citations = [{"source_id": "solo", "url": "https://solo.example"}]
    out = _replace_openai_citation_markers(text, citations)
    assert out == "[[1]](https://solo.example)"


def test_back_to_back_markers_with_no_separator():
    """Two markers butted up against each other resolve to two
    bracket links concatenated, no joining whitespace."""
    text = f"{_marker('x')}{_marker('y')}"
    citations = [
        {"source_id": "x", "url": "https://x.example"},
        {"source_id": "y", "url": "https://y.example"},
    ]
    out = _replace_openai_citation_markers(text, citations)
    assert out == "[[1]](https://x.example)[[2]](https://y.example)"


def test_split_helper_buffers_only_after_last_open_byte():
    """If a delta contains a complete marker followed by an
    unterminated marker, the head must include the complete marker
    and the buffer must hold only the trailing partial."""
    complete = _marker("done")
    partial = f"{CITE_START}cite{CITE_DELIM}half"  # no STOP
    text = f"pre {complete} mid {partial}"
    head, tail = _split_pending_citation_tail(text)
    assert head == f"pre {complete} mid "
    assert tail == partial
    # And the head, once rewritten, drops every private-use byte.
    rewritten = _replace_openai_citation_markers(
        head, [{"source_id": "done", "url": "https://d"}]
    )
    assert rewritten == "pre [[1]](https://d) mid "


def test_split_helper_empty_input():
    head, tail = _split_pending_citation_tail("")
    assert head == "" and tail == ""


def test_split_helper_no_open_byte():
    head, tail = _split_pending_citation_tail("nothing to see here")
    assert head == "nothing to see here" and tail == ""


def test_split_helper_complete_marker_only():
    """A delta that ends with a complete (closed) marker should NOT
    leave anything in the buffer."""
    text = f"alpha {_marker('a')}"
    head, tail = _split_pending_citation_tail(text)
    assert head == text and tail == ""


# ---------------------------------------------------------------------------
# 8. Sources-panel behaviour: marker drop must not affect citation
# aggregation. We re-derive the citation index list from the same
# url_citations list the rewriter consumes.
# ---------------------------------------------------------------------------


def test_unknown_marker_does_not_perturb_citation_indexing():
    """An unknown source_id marker is dropped, but the OTHER citations
    in the list keep their original 1-based numbering."""
    text = f"A {_marker('unknown')} B {_marker('real_a')} C {_marker('real_b')}"
    citations = [
        {"source_id": "real_a", "url": "https://example.com/a"},
        {"source_id": "real_b", "url": "https://example.com/b"},
    ]
    out = _replace_openai_citation_markers(text, citations)
    # real_a is index 1 (NOT 0 -- the unknown marker doesn't take a
    # slot, because indices come from the citation list, not the
    # marker stream).
    assert "[[1]](https://example.com/a)" in out
    assert "[[2]](https://example.com/b)" in out
    assert _no_private_use(out)
