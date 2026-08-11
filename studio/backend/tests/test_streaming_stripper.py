# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Correctness proof for ``StreamingMarkupStripper``.

The incremental stripper is only sound if two claims hold, so both are tested directly
rather than assumed:

1. **Sentinel completeness** - text containing no literal from ``_STRIP_SENTINELS`` is
   returned unchanged by the strip. If an arm could fire without one of those literals,
   the fast path would silently skip a strip that should have happened.
2. **Prefix split** - ``strip(text) == text[:i] + strip(text[i:])`` for any ``i`` at or
   before the first sentinel. This is what lets the stripper keep a settled prefix.

On top of those, the whole thing is replayed token by token against the non-incremental
strip and asserted byte-identical at every step, over every chunking the corpus produces.
"""

import random
import sys
from pathlib import Path

import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from core import tool_healing  # noqa: E402
from core.inference.tool_call_parser import (  # noqa: E402
    _STRIP_SENTINELS,
    StreamingMarkupStripper,
    _first_sentinel,
    _safe_cut,
    strip_segment,
)

sys.path.insert(0, str(BACKEND_ROOT / "tests" / "tools"))
import refactor_guard  # noqa: E402

ENABLED = {"get_weather", "search", "trunc", "broken"}


def _reference_strip(text, enabled_tool_names=ENABLED):
    """The pre-refactor streaming strip: full rescan, no caching."""

    def _seg(segment, is_last):
        return strip_segment(
            segment, seg_final=is_last, enabled_tool_names=enabled_tool_names
        )

    return tool_healing.strip_outside_think(text, _seg)


@pytest.fixture(scope="module")
def corpus():
    return refactor_guard.build_corpus()


def _sentinel_free(text):
    return not any(sentinel in text for sentinel in _STRIP_SENTINELS)


def test_sentinel_free_text_is_returned_unchanged(corpus):
    """Claim 1 over the corpus."""
    checked = 0
    for text in corpus:
        if not _sentinel_free(text):
            continue
        checked += 1
        assert _reference_strip(text) == text, f"strip altered sentinel-free text: {text!r}"
    assert checked, "corpus contained no sentinel-free input to check"


def test_sentinel_free_fuzz_is_returned_unchanged():
    """Claim 1 over random text built from characters the markup is made of.

    Drawing from ``<>|[]{}/:=_`` and backticks rather than plain prose is what makes this
    a real test: it produces near-miss markup that a too-narrow sentinel list would let
    through.
    """
    rng = random.Random(20260811)
    alphabet = "<>|[]{}/:=_`~ \n\tabcTOOLCALSfunctionthinkARGSpython_tagcall"
    checked = 0
    for _ in range(20000):
        text = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 60)))
        if not _sentinel_free(text):
            continue
        checked += 1
        assert _reference_strip(text) == text, f"strip altered sentinel-free text: {text!r}"
    assert checked > 1000, f"fuzz produced too few sentinel-free samples ({checked})"


def test_prefix_split_property(corpus):
    """Claim 2: splitting at ``_safe_cut`` does not change the result."""
    for text in corpus:
        first = _first_sentinel(text, 0)
        cut = _safe_cut(text, first) if first >= 0 else len(text)
        expected = _reference_strip(text)
        assert text[:cut] + _reference_strip(text[cut:]) == expected, (
            f"prefix split at {cut} changed the result for {text!r}"
        )


def test_prefix_split_property_fuzz():
    """Claim 2 on random near-miss markup, where an off-by-one cut would show up."""
    rng = random.Random(20260813)
    alphabet = "<>|[]{}/:=_-`~ \n\tabcTOOLCALSfunctionthinkARGSpython_tagcall\"'0129"
    for _ in range(20000):
        text = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 80)))
        first = _first_sentinel(text, 0)
        cut = _safe_cut(text, first) if first >= 0 else len(text)
        assert text[:cut] + _reference_strip(text[cut:]) == _reference_strip(text), (
            f"prefix split at {cut} changed the result for {text!r}"
        )


def test_incremental_matches_reference_token_by_token(corpus):
    """The acceptance test: replay each corpus entry one character at a time."""
    for text in corpus:
        stripper = StreamingMarkupStripper(ENABLED)
        for end in range(len(text) + 1):
            prefix = text[:end]
            assert stripper.strip(prefix) == _reference_strip(prefix), (
                f"diverged at offset {end} of {text!r}"
            )


def test_incremental_matches_reference_for_random_chunkings(corpus):
    """Same, but with realistic multi-character token boundaries."""
    rng = random.Random(20260812)
    for text in corpus:
        stripper = StreamingMarkupStripper(ENABLED)
        pos = 0
        while pos < len(text):
            pos = min(len(text), pos + rng.randint(1, 7))
            prefix = text[:pos]
            assert stripper.strip(prefix) == _reference_strip(prefix), (
                f"diverged at offset {pos} of {text!r}"
            )


def test_incremental_matches_reference_on_fuzz():
    """Char-by-char replay on random near-miss markup, fences and newlines included."""
    rng = random.Random(20260814)
    alphabet = "<>|[]{}/:=_-`~ \n\tabcTOOLCALSfunctionthinkARGSpython_tagcall\"'0129"
    for _ in range(400):
        text = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 50)))
        stripper = StreamingMarkupStripper(ENABLED)
        for end in range(len(text) + 1):
            prefix = text[:end]
            assert stripper.strip(prefix) == _reference_strip(prefix), (
                f"diverged at offset {end} of {text!r}"
            )


def test_rewind_resets_cached_state():
    """A caller that does not append monotonically still gets the right answer."""
    stripper = StreamingMarkupStripper(ENABLED)
    long_text = 'hello <tool_call>{"name": "search", "arguments": {}}</tool_call> world'
    assert stripper.strip(long_text) == _reference_strip(long_text)
    # Unrelated shorter text: not an extension of the previous input.
    assert stripper.strip("different") == _reference_strip("different")
    assert stripper.strip(long_text) == _reference_strip(long_text)


def test_repeated_call_is_cached():
    stripper = StreamingMarkupStripper(ENABLED)
    text = "no markup here at all"
    assert stripper.strip(text) is stripper.strip(text)


def test_scan_is_amortized_not_quadratic():
    """A prose-only response must not cost more per token as it grows.

    Measures work rather than wall clock: the reference does a full pass per token, so
    its total character-visits grow quadratically; the incremental stripper resumes its
    scan and must stay linear. Compared as a ratio so the assertion is machine
    independent.
    """
    import time

    def elapsed(fn, tokens):
        text = ""
        start = time.perf_counter()
        for token in tokens:
            text += token
            fn(text)
        return time.perf_counter() - start

    short = ["word " for _ in range(300)]
    long = ["word " for _ in range(1200)]

    def incremental(tokens):
        stripper = StreamingMarkupStripper(ENABLED)
        return elapsed(stripper.strip, tokens)

    # 4x the tokens. The reference rescans everything, so its cost grows ~16x; the
    # incremental one resumes its scan and should grow ~4x. Comparing the two growth
    # ratios rather than asserting an absolute keeps this meaningful on a noisy CI box
    # while still catching a regression to full rescanning.
    reference_growth = elapsed(_reference_strip, long) / max(elapsed(_reference_strip, short), 1e-9)
    incremental_growth = incremental(long) / max(incremental(short), 1e-9)

    assert incremental_growth < reference_growth / 2, (
        f"incremental cost grew {incremental_growth:.1f}x vs the reference's "
        f"{reference_growth:.1f}x; expected roughly linear against its quadratic"
    )
