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
from core.inference import tool_call_parser  # noqa: E402
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


def _reference_strip(text, enabled_tool_names = ENABLED):
    """The pre-refactor streaming strip: full rescan, no caching."""

    def _seg(segment, is_last):
        return strip_segment(segment, seg_final = is_last, enabled_tool_names = enabled_tool_names)

    return tool_healing.strip_outside_think(text, _seg)


@pytest.fixture(scope = "module")
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
        assert (
            text[:cut] + _reference_strip(text[cut:]) == expected
        ), f"prefix split at {cut} changed the result for {text!r}"


def test_prefix_split_property_fuzz():
    """Claim 2 on random near-miss markup, where an off-by-one cut would show up."""
    rng = random.Random(20260813)
    alphabet = "<>|[]{}/:=_-`~ \n\tabcTOOLCALSfunctionthinkARGSpython_tagcall\"'0129"
    for _ in range(20000):
        text = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 80)))
        first = _first_sentinel(text, 0)
        cut = _safe_cut(text, first) if first >= 0 else len(text)
        assert text[:cut] + _reference_strip(text[cut:]) == _reference_strip(
            text
        ), f"prefix split at {cut} changed the result for {text!r}"


def test_incremental_matches_reference_token_by_token(corpus):
    """The acceptance test: replay each corpus entry one character at a time."""
    for text in corpus:
        stripper = StreamingMarkupStripper(ENABLED)
        for end in range(len(text) + 1):
            prefix = text[:end]
            assert stripper.strip(prefix) == _reference_strip(
                prefix
            ), f"diverged at offset {end} of {text!r}"


def test_incremental_matches_reference_for_random_chunkings(corpus):
    """Same, but with realistic multi-character token boundaries."""
    rng = random.Random(20260812)
    for text in corpus:
        stripper = StreamingMarkupStripper(ENABLED)
        pos = 0
        while pos < len(text):
            pos = min(len(text), pos + rng.randint(1, 7))
            prefix = text[:pos]
            assert stripper.strip(prefix) == _reference_strip(
                prefix
            ), f"diverged at offset {pos} of {text!r}"


def test_incremental_matches_reference_on_fuzz():
    """Char-by-char replay on random near-miss markup, fences and newlines included."""
    rng = random.Random(20260814)
    alphabet = "<>|[]{}/:=_-`~ \n\tabcTOOLCALSfunctionthinkARGSpython_tagcall\"'0129"
    for _ in range(400):
        text = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 50)))
        stripper = StreamingMarkupStripper(ENABLED)
        for end in range(len(text) + 1):
            prefix = text[:end]
            assert stripper.strip(prefix) == _reference_strip(
                prefix
            ), f"diverged at offset {end} of {text!r}"


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

    # 4x the tokens: the reference rescans everything (~16x), the incremental one resumes
    # (~4x). Comparing growth ratios rather than absolutes keeps this meaningful on a
    # noisy CI box while still catching a regression to full rescanning.
    reference_growth = elapsed(_reference_strip, long) / max(elapsed(_reference_strip, short), 1e-9)
    incremental_growth = incremental(long) / max(incremental(short), 1e-9)

    assert incremental_growth < reference_growth / 2, (
        f"incremental cost grew {incremental_growth:.1f}x vs the reference's "
        f"{reference_growth:.1f}x; expected roughly linear against its quadratic"
    )


# Cases the corpus and the alphabet fuzz both missed. Each produced a real divergence from
# the reference strip before the guard it names was added, so each is pinned as a literal.
_MISSED_BY_THE_CORPUS = (
    # ``_GEMMA_BARE_TC_RE`` is ``call\s*:``, so a space or newline before the colon is
    # still a call. Sentinel completeness needs the literal ``call``, not ``call:``.
    "call :get_weather{city:Paris}",
    "call\n:get_weather{city:Paris}",
    "The answer.\ncall : get_weather{city:Paris}",
    # A JSON answer is data: its ``call:NAME{...}`` examples stay visible. That decision
    # keys on the whole segment, so trimming the segment must not reach it.
    '{\n  "tool_syntax": "call:get_weather{city:Paris}",\n  "note": "example"\n}',
    '[\n  "call:search{q:1}"\n]',
    # Earlier arms can leave behind a segment that is whole JSON when the untrimmed one
    # was not, which is the same hazard arrived at from the other side.
    'answer\n[TOOL_CALLS]search[ARGS]{"q":1}{\n  "k": "call:get_weather{c:P}"\n}',
    # A reasoning closer with no opener makes offset 0 of the segment meaningful, so
    # nothing may be trimmed off the front of it.
    '\n[TOOL_CALLS]search[ARGS]{"q":1}[/THINK]<function=search>{}</function>',
    '[THINK]r[/THINK]<function name="s">{}</function>[/THINK]tail',
)


@pytest.mark.parametrize("text", _MISSED_BY_THE_CORPUS)
@pytest.mark.parametrize("enabled", [ENABLED, None])
def test_incremental_matches_reference_on_known_hard_cases(text, enabled):
    stripper = StreamingMarkupStripper(enabled)
    for size in range(1, len(text) + 1):
        prefix = text[:size]
        assert stripper.strip(prefix) == _reference_strip(
            prefix, enabled
        ), f"diverged at {size} for {text!r}"


@pytest.mark.parametrize("text", _MISSED_BY_THE_CORPUS)
def test_known_hard_cases_are_still_sentinel_reachable(text):
    """Each hard case must carry a sentinel, or claim 1 is what is broken."""
    assert not _sentinel_free(text)


def test_incremental_matches_reference_on_structured_fuzz():
    """Fuzz built from whole markup fragments rather than from an alphabet.

    The alphabet fuzz above rarely assembles a complete, well-formed call, which is why
    it missed every case in ``_MISSED_BY_THE_CORPUS``. Splicing real fragments reaches
    the arms that only fire on a complete one.
    """
    fragments = _MISSED_BY_THE_CORPUS + (
        "Hello world. ",
        "I will call the tool. ",
        "<think>reasoning</think>",
        "[THINK]r[/THINK]",
        "[/THINK]",
        "</think>",
        "```py\ncode\n```\n",
        "~~~\nx\n~~~\n",
        '<tool_call>{"name": "search"}</tool_call>',
        "<function=search>{}</function>",
        '[TOOL_CALLS]search[ARGS]{"q": 1}',
        'get_weather[ARGS]{"a": 1}',
        "<|python_tag|>x",
        "<|tool_call>call:search{q:1}<tool_call|>",
        "recall: not a call",
    )
    rng = random.Random(20260811)
    for _ in range(3000):
        text = "".join(rng.choice(fragments) for _ in range(rng.randint(1, 4)))
        enabled = rng.choice([ENABLED, None, set()])
        stripper = StreamingMarkupStripper(enabled)
        for size in range(1, len(text) + 1):
            prefix = text[:size]
            assert stripper.strip(prefix) == _reference_strip(
                prefix, enabled
            ), f"diverged at {size} for {text!r}"


def test_prose_containing_the_word_call_is_still_amortized():
    """``call`` is a sentinel and an ordinary English word.

    Taking it at face value put the full strip back on the per-token path for any answer
    that says "I will call the tool", which measured slower than the code this replaces.
    ``_first_sentinel`` confirms the hit against the arm instead, and this pins that.
    """
    import time

    def elapsed(fn, tokens):
        text = ""
        start = time.perf_counter()
        for token in tokens:
            text += token
            fn(text)
        return time.perf_counter() - start

    short = ["I will call it. " for _ in range(300)]
    long = ["I will call it. " for _ in range(1200)]

    def incremental(tokens):
        return elapsed(StreamingMarkupStripper(ENABLED).strip, tokens)

    reference_growth = elapsed(_reference_strip, long) / max(elapsed(_reference_strip, short), 1e-9)
    incremental_growth = incremental(long) / max(incremental(short), 1e-9)

    assert incremental_growth < reference_growth / 2, (
        f"incremental cost grew {incremental_growth:.1f}x vs the reference's "
        f"{reference_growth:.1f}x on prose containing the word 'call'"
    )


def test_a_real_bare_call_is_still_seen_as_a_sentinel():
    """The other side of the same refinement: a real call must not be skipped.

    Including while it is still a partial, which is the state the buffer is in for every
    token but the last one of it.
    """
    text = "Sure. call:get_weather{city:Paris}"
    for size in range(text.index("call") + len("call"), len(text) + 1):
        assert _first_sentinel(text[:size], 0) == text.index(
            "call"
        ), f"lost the call anchor at {size}: {text[:size]!r}"
    assert _first_sentinel("Please call me back tomorrow.", 0) == -1
    assert _first_sentinel("I made a call: yesterday it worked.", 0) == -1


def test_the_bracket_scan_size_guard_survives_a_prefix_cut():
    """``_strip_bracket_tag_calls`` stands down over ``_MAX_BRACKET_SCAN_CHARS``.

    A cut shortens the segment, so a tail that fell under the limit would re-enable an
    arm the full scan had skipped and strip text the reference keeps.
    """
    prose = "word " * ((tool_healing._MAX_BRACKET_SCAN_CHARS // 5) + 1)
    text = prose + '\nsearch[ARGS]{"x": 1} tail'
    assert len(text) > tool_healing._MAX_BRACKET_SCAN_CHARS

    stripper = StreamingMarkupStripper(ENABLED)
    stripper.strip(prose)

    assert stripper.strip(text) == _reference_strip(text)


def test_a_prose_call_at_a_token_boundary_stays_amortized():
    """``call`` ending a token is only a possible marker, and only until the next one.

    Committing that hit sends every later token through the whole-buffer checks, so the
    cost depended on where the tokenizer happened to split rather than on the text. This
    measures the same text under two chunkings; they have to stay comparable.
    """
    import time

    def elapsed(tokens):
        stripper = StreamingMarkupStripper(ENABLED)
        text = ""
        start = time.perf_counter()
        for token in tokens:
            text += token
            stripper.strip(text)
        return time.perf_counter() - start

    split = elapsed(["I will call", " it now. "] * 800)
    joined = elapsed(["I will call it now. "] * 800)

    assert (
        split < joined * 20 + 0.5
    ), f"a token boundary after 'call' cost {split:.3f}s against {joined:.3f}s joined"


def test_a_real_call_arriving_a_character_at_a_time_is_still_caught():
    text = "Sure. call:search{q: 1} done"
    stripper = StreamingMarkupStripper(ENABLED)
    for size in range(1, len(text) + 1):
        assert stripper.strip(text[:size]) == _reference_strip(text[:size])


def test_the_caller_can_still_grow_its_buffer_in_place():
    """The stripper must not hold a reference to the buffer it is handed.

    The streaming loop grows it with ``cumulative_display += token``, and CPython only
    resizes a string in place while nothing else refers to it. One extra reference turns
    every append into a full copy, so the concatenation goes quadratic even though the
    scanning does not, and that cost lands on the caller rather than here.
    """
    import time

    def elapsed(count):
        stripper = StreamingMarkupStripper(ENABLED)
        text = ""
        start = time.perf_counter()
        for _ in range(count):
            text += "word "
            stripper.strip(text)
        return time.perf_counter() - start

    short = elapsed(8000)
    long = elapsed(32000)

    # 4x the tokens. Linear in place, ~16x if every append copies the answer so far.
    assert long < short * 8 + 0.05, (
        f"4x the tokens cost {long / max(short, 1e-9):.1f}x the time "
        f"({short:.3f}s -> {long:.3f}s); the buffer is being copied per token"
    )


def test_no_reference_to_the_buffer_is_retained():
    """The property the timing above measures, asserted directly."""
    import gc

    stripper = StreamingMarkupStripper(ENABLED)
    text = "some plain prose with no markup in it at all" * 4
    stripper.strip(text)

    assert not [
        holder
        for holder in gc.get_referrers(text)
        if holder is stripper.__class__ or holder is stripper
    ]
    assert all(getattr(stripper, slot) is not text for slot in StreamingMarkupStripper.__slots__)


def test_reset_clears_the_cached_prefix_for_a_new_buffer():
    """The streaming caller keeps one instance across tool iterations, and each iteration
    starts a fresh ``cumulative_display``. ``_is_extension`` samples rather than compares,
    so it cannot be relied on to notice: a new buffer that agreed with the old one on
    length and on every sampled window would be accepted, the scan would resume at an
    offset from the previous iteration and never look below it again. The caller says so
    explicitly rather than depending on the sampling to catch it.
    """
    stripper = StreamingMarkupStripper(ENABLED)

    first = "x" * 200 + '<tool_call>{"name": "search", "arguments": {}}</tool_call>' + "y" * 200
    stripper.strip(first)

    second = first[:64] + "z" * (len(first) - 128) + first[-64:]
    assert len(second) == len(first) and second != first

    stripper.reset()

    assert stripper.strip(second) == _reference_strip(second)


def test_early_markup_is_not_slower_than_the_code_it_replaces():
    """Once the cut is 0 and nothing is settled, both arms are the same strip.

    Paying the whole-buffer checks there cannot change the answer, and an answer with
    markup near its front would pay them on every token for the rest of the turn, which
    measured slower than the full rescan this replaces.
    """
    import time

    prefix = '`x` <tool_call>{"name": "search", "arguments": {}}</tool_call> '

    def elapsed(fn, count):
        text = prefix
        start = time.perf_counter()
        for _ in range(count):
            text += "word "
            fn(text)
        return time.perf_counter() - start

    count = 1500
    reference = min(elapsed(_reference_strip, count) for _ in range(3))
    incremental = min(elapsed(StreamingMarkupStripper(ENABLED).strip, count) for _ in range(3))

    assert (
        incremental <= reference * 1.10
    ), f"early markup cost {incremental:.3f}s against the reference's {reference:.3f}s"


def test_an_open_reasoning_block_is_scanned_incrementally():
    """A reasoning body is most of a reasoning model's answer.

    The open-block branch restarted at the opener on every token, so the body was
    rescanned in full each time: the same quadratic this class exists to remove, with a
    smaller constant. 16k tokens cost 7.4s before this and 0.05s after.
    """
    import time

    def elapsed(count):
        stripper = StreamingMarkupStripper(ENABLED)
        text = "<think>"
        start = time.perf_counter()
        for _ in range(count):
            text += "reasoning "
            stripper.strip(text)
        return time.perf_counter() - start

    short = elapsed(2000)
    long = elapsed(8000)

    # 4x the tokens. Linear resumes at ~4x; restarting at the opener is ~16x.
    assert long < short * 8 + 0.05, (
        f"4x the tokens cost {long / max(short, 1e-9):.1f}x the time "
        f"({short:.4f}s -> {long:.4f}s); the reasoning body is being rescanned"
    )


@pytest.mark.parametrize(
    "text",
    [
        "<think>reasoning goes here</think>the answer",
        "<think>reasoning with get_weather[ARGS]{} inside</think>answer",
        "[THINK]other family[/THINK]answer",
        "<think>unclosed to the end",
        "<think>a</think>b<think>c</think>d",
        "<think>mentions </think> early</think>tail",
    ],
)
def test_open_block_resume_does_not_change_the_result(text):
    """The resume must not miss a closer or a sentinel arriving mid-body."""
    stripper = StreamingMarkupStripper(ENABLED)
    for size in range(1, len(text) + 1):
        assert stripper.strip(text[:size]) == _reference_strip(
            text[:size]
        ), f"diverged at {size} for {text!r}"


def _reference_strip_non_final(text, enabled_tool_names = ENABLED):
    """What the final-answer loop asks for: no end-of-turn arms."""
    from core.inference.tool_call_parser import strip_tool_markup
    return strip_tool_markup(text, final = False, enabled_tool_names = enabled_tool_names)


@pytest.mark.parametrize(
    "text",
    [
        'Sure.<tool_call>{"name": "get_weather", "arguments": {}}</tool_call>Done.',
        "call:get_weather{city:Paris} tail",
        "get_weather[ARGS]",
        '[TOOL_CALLS]get_weather[ARGS]{"c": 1} after',
        "<think>r</think>answer",
        "```\nget_weather[ARGS]{}\n```\n",
        "prose with no markup at all",
    ],
)
def test_non_final_stripper_matches_the_non_final_strip(text):
    """The final-answer loop after the tool budget is spent calls the strip with
    ``final = False``, which leaves the end-of-turn arms off. Sharing the tool loop's
    instance would silently turn them on, so it gets its own with the flag."""
    stripper = StreamingMarkupStripper(ENABLED, seg_final = False)
    for size in range(1, len(text) + 1):
        assert stripper.strip(text[:size]) == _reference_strip_non_final(
            text[:size]
        ), f"diverged at {size} for {text!r}"


def test_the_final_answer_loop_is_not_quadratic():
    """That loop ran the whole strip over the growing buffer on every token.

    It is reached when the tool-iteration budget is exhausted, which is uncommon, but it
    was the largest single cost left in this file: 50s over a 16k-token answer.
    """
    import time

    def elapsed(fn, count):
        text = ""
        start = time.perf_counter()
        for _ in range(count):
            text += "word "
            fn(text)
        return time.perf_counter() - start

    count = 4000
    before = elapsed(_reference_strip_non_final, count)
    after = elapsed(StreamingMarkupStripper(ENABLED, seg_final = False).strip, count)

    assert (
        after < before / 10
    ), f"final-answer strip cost {after:.4f}s against the full rescan's {before:.4f}s"


def test_a_cut_never_crosses_an_open_parameter_block():
    """``_strip_function_xml_calls`` treats a ``<function>`` opener inside an unclosed
    ``<parameter>`` as a literal in an argument value, and decides that from the text
    before it. Cutting there used to lose the context and leak the nested markup."""
    names = {"a"}
    for text in (
        "Visible <parameter=x>\n<function=a></function>TEXT</function>",
        'Visible <param name="x">\n<function=a></function>TEXT</function>',
    ):
        expected = _reference_strip(text, names)
        stripper = StreamingMarkupStripper(names)
        got = None
        for i in range(1, len(text) + 1):
            got = stripper.strip(text[:i])
        assert got == expected
        assert "TEXT</function>" not in got

    # A closed parameter block still gets cut, so the guard is not blanket.
    closed = "Visible <parameter=x>v</parameter>\n<function=a></function>TEXT</function>"
    stripper = StreamingMarkupStripper(names)
    for i in range(1, len(closed) + 1):
        got = stripper.strip(closed[:i])
    assert got == _reference_strip(closed, names)


def test_a_replaced_middle_is_not_taken_for_a_continuation():
    """The extension check samples rather than compares in full, so it has to sample
    somewhere the difference can show. Head and tail alone missed a buffer whose middle
    was replaced while both ends and the length stayed put, and the stripper then skipped
    rescanning the part that changed and returned a stale answer."""
    sample = tool_call_parser._EXTENSION_SAMPLE
    names = {"a"}
    first = "A" * sample + "x" * 11 + "Z" * sample
    second = "A" * sample + "<tool_call>" + "Z" * sample
    assert len(first) == len(second)

    stripper = StreamingMarkupStripper(names)
    stripper.strip(first)
    assert stripper._is_extension(second) is False
    assert stripper.strip(second) == _reference_strip(second, names)


def test_an_append_only_stream_is_still_recognised_as_a_continuation():
    """Control for the test above: the extra sample must not push the ordinary
    append-only case onto the slow path, which is the whole point of the class."""
    names = {"a"}
    text = "some prose " * 400
    stripper = StreamingMarkupStripper(names)
    stripper.strip(text[:500])
    for i in range(600, len(text), 100):
        assert stripper._is_extension(text[:i]) is True
        stripper.strip(text[:i])
