# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Edge-case tests for the OpenAI Responses citation marker rewriter.

Covers multi-source markers, source+locator, marker SPLIT across SSE deltas,
unterminated tails at end-of-stream, multiple markers per delta, late
annotation ordering, and idempotency.

Reference: https://developers.openai.com/api/docs/guides/citation-formatting
"""

import importlib


# Streaming is exercised by ``_simulate_delta_stream`` below, mirroring the
# head/buffer/flush dance from ``_stream_openai_responses``.
_module = importlib.import_module("core.inference.external_provider")
_replace_openai_citation_markers = _module._replace_openai_citation_markers
_split_pending_citation_tail = _module._split_pending_citation_tail


CITE_START = ""
CITE_STOP = ""
CITE_DELIM = ""


def _marker(*source_ids: str, locator: str | None = None) -> str:
    """Build a ``\\ue200cite\\ue202<sid>[\\ue202<sid>...][\\ue202<loc>]\\ue201``
    marker from one or more ``source_ids`` plus an optional ``locator``."""
    payload = f"{CITE_START}cite{CITE_DELIM}" + CITE_DELIM.join(source_ids)
    if locator:
        payload = f"{payload}{CITE_DELIM}{locator}"
    return f"{payload}{CITE_STOP}"


def _no_private_use(text: str) -> bool:
    return all(c not in text for c in (CITE_START, CITE_STOP, CITE_DELIM))


# Harness mirroring the head/pending-tail/flush dance in
# `_stream_openai_responses` so streaming tests skip the httpx mock.
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
        # Mirror `_flush_pending_marker_tail`: drop the tail if no closing stop
        # byte arrived; the literal ``cite<sid>`` would leak otherwise.
        if CITE_STOP not in pending:
            rendered = ""
        else:
            rendered = _replace_openai_citation_markers(pending, citations)
            for ch in (CITE_START, CITE_STOP, CITE_DELIM):
                rendered = rendered.replace(ch, "")
            import re as _re

            rendered = _re.sub(r"^cite\S*", "", rendered)
        if rendered:
            emitted.append(rendered)
    return "".join(emitted)


# ---------------------------------------------------------------------------
# 1. Multi-source markers per the OpenAI docs.
# ---------------------------------------------------------------------------


def test_multi_source_marker_all_resolve():
    """\\ue200cite\\ue202id1\\ue202id2\\ue202id3\\ue201 expands to three links
    when every id is known. Earlier regex captured only id1."""
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
    """Known ids render, unknown ids drop silently, no glyph leaks."""
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
    text = f"See {_marker('tu0', locator = '42')}."
    citations = [{"source_id": "tu0", "url": "https://example.com/doc"}]
    out = _replace_openai_citation_markers(text, citations)
    assert "[[1]](https://example.com/doc)" in out
    assert "42" not in out
    assert _no_private_use(out)


def test_marker_with_range_locator():
    text = f"See {_marker('tu0', locator = 'L8-L13')}."
    citations = [{"source_id": "tu0", "url": "https://example.com/code"}]
    out = _replace_openai_citation_markers(text, citations)
    assert "[[1]](https://example.com/code)" in out
    assert "L8-L13" not in out
    assert _no_private_use(out)


# ---------------------------------------------------------------------------
# 3. Marker SPLIT across two SSE deltas -- the codex-flagged P1.
# ---------------------------------------------------------------------------


def test_marker_split_in_source_id():
    """Delta-1 ends mid-source-id (``\\ue200cite\\ue202tu``), delta-2 has the
    rest (``rn0view0\\ue201``). The buffer stitches the halves so they resolve
    to one link instead of leaking."""
    full = f"See {_marker('turn0view0')} now."
    # Cut right after the second delim + "tu" inside the source id.
    cut = full.index("tu", full.index(CITE_START)) + len("tu")
    d1, d2 = full[:cut], full[cut:]
    # Sanity: delta-1 contains a partial marker.
    assert CITE_START in d1 and CITE_STOP not in d1
    assert CITE_STOP in d2
    citations = [{"source_id": "turn0view0", "url": "https://x"}]
    out = _simulate_delta_stream([d1, d2], citations)
    assert out == "See [[1]](https://x) now."
    assert _no_private_use(out)


def test_marker_split_at_start_byte():
    """Split right after the opening ``\\ue200`` byte; the buffer must hold the
    lone open byte until the rest arrives."""
    full = f"Text {_marker('sid')} done"
    cut = full.index(CITE_START) + 1  # right AFTER the open byte
    d1, d2 = full[:cut], full[cut:]
    citations = [{"source_id": "sid", "url": "https://y"}]
    out = _simulate_delta_stream([d1, d2], citations)
    assert out == "Text [[1]](https://y) done"
    assert _no_private_use(out)


def test_marker_split_across_three_deltas():
    """Worst case: marker chopped into three pieces across three deltas."""
    full = f"A {_marker('threesplit')} B"
    # Cut at two points inside the marker.
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
    """Delta-2 closes the marker AND carries trailing prose; both emit cleanly."""
    full = f"X {_marker('sid')} after"
    cut = full.index("cite") + len("ci")
    d1, d2 = full[:cut], full[cut:]
    citations = [{"source_id": "sid", "url": "https://a"}]
    out = _simulate_delta_stream([d1, d2], citations)
    assert out == "X [[1]](https://a) after"
    assert _no_private_use(out)


def test_split_marker_unknown_source_is_dropped_cleanly():
    """Split marker for an unknown source drops silently on flush."""
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
    """Stream ends mid-marker (e.g. response.incomplete); the flushed tail
    strips private-use bytes, no `E202` text leaks."""
    deltas = ["Some text ", f"{CITE_START}citetu", "rn0view0"]  # no STOP ever
    out = _simulate_delta_stream(deltas, [], flush = True)
    assert _no_private_use(out)
    assert "E200" not in out and "E202" not in out
    # Surrounding prose stays; don't assert exact marker remainder.
    assert "Some text " in out


def test_flush_resolves_marker_when_late_annotation_arrives():
    """Marker in a delta; the matching annotation arrives later (on
    response.output_text.annotation.added after the final delta). The rewriter
    reads ``all_url_citations`` LIVE at flush, so the buffered marker resolves."""
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
    """Running the rewriter twice does not double-link or corrupt brackets."""
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
    """A delta that is JUST a marker (no prose) renders correctly; it leaked
    before the empty-string short-circuit in the split helper."""
    text = _marker("solo")
    citations = [{"source_id": "solo", "url": "https://solo.example"}]
    out = _replace_openai_citation_markers(text, citations)
    assert out == "[[1]](https://solo.example)"


def test_back_to_back_markers_with_no_separator():
    """Adjacent markers resolve to concatenated links, no joining whitespace."""
    text = f"{_marker('x')}{_marker('y')}"
    citations = [
        {"source_id": "x", "url": "https://x.example"},
        {"source_id": "y", "url": "https://y.example"},
    ]
    out = _replace_openai_citation_markers(text, citations)
    assert out == "[[1]](https://x.example)[[2]](https://y.example)"


def test_split_helper_buffers_only_after_last_open_byte():
    """Complete marker followed by an unterminated one: head includes the
    complete marker, buffer holds only the trailing partial."""
    complete = _marker("done")
    partial = f"{CITE_START}cite{CITE_DELIM}half"  # no STOP
    text = f"pre {complete} mid {partial}"
    head, tail = _split_pending_citation_tail(text)
    assert head == f"pre {complete} mid "
    assert tail == partial
    # Head, once rewritten, drops every private-use byte.
    rewritten = _replace_openai_citation_markers(head, [{"source_id": "done", "url": "https://d"}])
    assert rewritten == "pre [[1]](https://d) mid "


def test_split_helper_empty_input():
    head, tail = _split_pending_citation_tail("")
    assert head == "" and tail == ""


def test_split_helper_no_open_byte():
    head, tail = _split_pending_citation_tail("nothing to see here")
    assert head == "nothing to see here" and tail == ""


def test_split_helper_complete_marker_only():
    """A delta ending with a closed marker leaves the buffer empty."""
    text = f"alpha {_marker('a')}"
    head, tail = _split_pending_citation_tail(text)
    assert head == text and tail == ""


# ---------------------------------------------------------------------------
# 8. Sources-panel: marker drop must not affect citation aggregation.
# Indices come from the url_citations list, not the marker stream.
# ---------------------------------------------------------------------------


def test_unknown_marker_does_not_perturb_citation_indexing():
    """Unknown source_id markers drop without consuming an index slot."""
    text = f"A {_marker('unknown')} B {_marker('real_a')} C {_marker('real_b')}"
    citations = [
        {"source_id": "real_a", "url": "https://example.com/a"},
        {"source_id": "real_b", "url": "https://example.com/b"},
    ]
    out = _replace_openai_citation_markers(text, citations)
    # real_a is index 1; unknown takes no slot.
    assert "[[1]](https://example.com/a)" in out
    assert "[[2]](https://example.com/b)" in out
    assert _no_private_use(out)


# ---------------------------------------------------------------------------
# Regression: unterminated marker tail must NOT leak the residual
# ``cite``-prefixed source id as plain text. PR #5713 audit P1.
# ---------------------------------------------------------------------------


def test_unterminated_marker_does_not_leak_cite_residue():
    """Stream ends mid-marker: drop the whole tail rather than strip codepoints
    and leave ``cite<sid>`` behind."""
    half = f"Hi there {CITE_START}cite{CITE_DELIM}turn0view0"
    out = _simulate_delta_stream([half], [], flush = True)
    # Prose before the marker stays; no private-use bytes or cite residue.
    assert "Hi there" in out
    assert _no_private_use(out)
    assert "citeturn0view0" not in out
    assert "cite" not in out.split("Hi there", 1)[1]


def test_unterminated_marker_only_no_prefix_drops_entirely():
    """A delta that is purely an unterminated marker flushes to ""."""
    half = f"{CITE_START}cite{CITE_DELIM}turn0view0"
    out = _simulate_delta_stream([half], [], flush = True)
    assert out == ""


def test_unterminated_marker_with_prefix_emits_only_prefix():
    """Prose then unterminated marker: prose emits, marker remnant drops."""
    half = f"prefix prose {CITE_START}cite{CITE_DELIM}abc"
    out = _simulate_delta_stream([half], [], flush = True)
    assert out == "prefix prose "


def test_closing_byte_arrives_after_pending_buffered_split():
    """Closing byte arrives in a later delta after opener + source id were
    buffered; link resolves with no residue."""
    cuts = [
        f"a {CITE_START}cite{CITE_DELIM}",
        f"sid{CITE_STOP} b",
    ]
    out = _simulate_delta_stream(
        cuts,
        [{"source_id": "sid", "url": "https://example.com/x"}],
        flush = True,
    )
    assert "[[1]](https://example.com/x)" in out
    assert "a " in out and "b" in out
    assert _no_private_use(out)
    assert "citesid" not in out
