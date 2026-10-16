# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""StreamingMarkupStripper must match the non-incremental strip, and skip only text without sentinels."""

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
    """Markup-alphabet fuzz catches near-miss text that a too-narrow sentinel list would let through."""
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
    assert stripper.strip("different") == _reference_strip("different")
    assert stripper.strip(long_text) == _reference_strip(long_text)


def test_repeated_call_is_cached():
    stripper = StreamingMarkupStripper(ENABLED)
    text = "no markup here at all"
    assert stripper.strip(text) is stripper.strip(text)


def test_scan_is_amortized_not_quadratic():
    """Prose-only text must cost constant work per token, not grow with length; compared as a ratio."""
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
    # (~4x). Ratios rather than absolutes keep this meaningful on a noisy CI box.
    reference_growth = elapsed(_reference_strip, long) / max(elapsed(_reference_strip, short), 1e-9)
    incremental_growth = incremental(long) / max(incremental(short), 1e-9)

    assert incremental_growth < reference_growth / 2, (
        f"incremental cost grew {incremental_growth:.1f}x vs the reference's "
        f"{reference_growth:.1f}x; expected roughly linear against its quadratic"
    )


# Each case diverged from the reference strip before its guard was added.
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
    """Fuzz splicing whole markup fragments, which reaches the arms that only fire on a complete call."""
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
    """The word call is also English, so a sentinel hit must be confirmed against its arm."""
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
    """A real bare call must still be seen as a sentinel while partial, not skipped."""
    text = "Sure. call:get_weather{city:Paris}"
    for size in range(text.index("call") + len("call"), len(text) + 1):
        assert _first_sentinel(text[:size], 0) == text.index(
            "call"
        ), f"lost the call anchor at {size}: {text[:size]!r}"
    assert _first_sentinel("Please call me back tomorrow.", 0) == -1
    assert _first_sentinel("I made a call: yesterday it worked.", 0) == -1


def test_the_bracket_scan_size_guard_survives_a_prefix_cut():
    """A prefix cut must not drop the segment under _MAX_BRACKET_SCAN_CHARS, or a skipped arm re-enables."""
    prose = "word " * ((tool_healing._MAX_BRACKET_SCAN_CHARS // 5) + 1)
    text = prose + '\nsearch[ARGS]{"x": 1} tail'
    assert len(text) > tool_healing._MAX_BRACKET_SCAN_CHARS

    stripper = StreamingMarkupStripper(ENABLED)
    stripper.strip(prose)

    assert stripper.strip(text) == _reference_strip(text)


def test_a_prose_call_at_a_token_boundary_stays_amortized():
    """A call at a token end is only a possible marker, so it must not commit the whole-buffer path."""
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
    """The stripper must not keep a reference to the caller's buffer, or += copies it on every token."""
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
    """reset() must clear the cached prefix, since _is_extension samples and may miss a new buffer."""
    stripper = StreamingMarkupStripper(ENABLED)

    first = "x" * 200 + '<tool_call>{"name": "search", "arguments": {}}</tool_call>' + "y" * 200
    stripper.strip(first)

    second = first[:64] + "z" * (len(first) - 128) + first[-64:]
    assert len(second) == len(first) and second != first

    stripper.reset()

    assert stripper.strip(second) == _reference_strip(second)


@pytest.mark.parametrize(
    "prefix",
    [
        pytest.param(
            '`x` <tool_call>{"name": "search", "arguments": {}}</tool_call> ', id = "near-front"
        ),
        pytest.param(
            '<tool_call>{"name": "search", "arguments": {}}</tool_call> ', id = "at-offset-0"
        ),
    ],
)
def test_early_markup_is_not_slower_than_the_code_it_replaces(monkeypatch, prefix):
    """Counts work instead of timing it: runner noise spans 0.89 to 1.09, too wide for a 10% margin."""
    work = {}

    def counting(name, fn):
        def wrapper(text, *args, **kwargs):
            work[name] = work.get(name, 0) + len(text)
            return fn(text, *args, **kwargs)

        return wrapper

    monkeypatch.setattr(
        tool_healing,
        "strip_outside_think",
        counting("strip_outside_think", tool_healing.strip_outside_think),
    )
    monkeypatch.setattr(tool_call_parser, "strip_segment", counting("strip_segment", strip_segment))
    monkeypatch.setattr(
        sys.modules[__name__], "strip_segment", counting("strip_segment", strip_segment)
    )
    monkeypatch.setattr(
        tool_call_parser,
        "_mask_blocked_bodies",
        counting("mask", tool_call_parser._mask_blocked_bodies),
    )
    needs_whole_buffer = StreamingMarkupStripper._needs_whole_buffer

    def counting_needs_whole_buffer(self, text):
        work["whole_buffer_checks"] = work.get("whole_buffer_checks", 0) + 1
        return needs_whole_buffer(self, text)

    monkeypatch.setattr(StreamingMarkupStripper, "_needs_whole_buffer", counting_needs_whole_buffer)

    count = 1500

    def run(strip):
        work.clear()
        text = prefix
        outputs = []
        for _ in range(count):
            text += "word "
            outputs.append(strip(text))
        return dict(work), outputs

    reference, expected = run(_reference_strip)
    incremental, got = run(StreamingMarkupStripper(ENABLED).strip)

    assert got == expected
    assert reference.get("strip_outside_think", 0) >= count * len(prefix)
    assert reference.get("strip_segment", 0) > 0
    assert not incremental.get("whole_buffer_checks"), (
        f"{incremental['whole_buffer_checks']} of {count} tokens paid the whole-buffer checks "
        "with markup at the front and nothing settled"
    )
    for name in ("strip_outside_think", "strip_segment"):
        assert (
            incremental.get(name, 0) <= reference[name]
        ), f"{name} read {incremental.get(name, 0)} chars against the reference's {reference[name]}"
    assert incremental.get("mask", 0) <= reference["strip_outside_think"], (
        f"masked {incremental.get('mask', 0)} chars, more than one pass per strip "
        f"({reference['strip_outside_think']})"
    )


def test_an_open_reasoning_block_is_scanned_incrementally():
    """An open reasoning block must be scanned incrementally, or each token rescans the whole body."""
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
    """The final-answer loop must not run the whole strip over the growing buffer on every token."""
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

    closed = "Visible <parameter=x>v</parameter>\n<function=a></function>TEXT</function>"
    stripper = StreamingMarkupStripper(names)
    for i in range(1, len(closed) + 1):
        got = stripper.strip(closed[:i])
    assert got == _reference_strip(closed, names)


def test_a_replaced_middle_is_not_taken_for_a_continuation():
    """Extension sampling must include the middle: head and tail alone missed a replaced middle."""
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


def test_a_bounded_scan_still_takes_the_eos_after_a_malformed_mistral_array():
    """The bounded Mistral scan must still consume the trailing </s> after a malformed array."""
    text = 'See [1]. [TOOL_CALLS] [{"name": "get_weather", "ar}]</s> Done.'

    assert tool_call_parser.strip_tool_markup(text, final = True) == "See [1].  Done."
    assert tool_call_parser.strip_tool_markup(text, final = False) == "See [1].  Done."


def test_openers_far_past_the_closer_do_not_reopen_the_quadratic_scan():
    """Bound the scan at the last closer; a window after the first closer just moves the quadratic cliff."""
    import time

    def elapsed(n):
        text = (
            '<tool_call>{"name": "search", "arguments": {}}</tool_call>'
            + "prose " * 60
            + "<tool_call>" * n
        )
        start = time.perf_counter()
        tool_healing.strip_tool_call_markup(text)
        return time.perf_counter() - start

    growth = elapsed(8000) / max(elapsed(2000), 1e-9)
    assert growth < 8.0, f"4x the openers cost {growth:.1f}x; expected roughly linear"


def test_an_unterminated_blocked_body_is_not_rescanned_per_snapshot():
    """A long blocked call keeps the stripper on its whole-buffer path, so anything quadratic
    here stalls the display. Counted rather than timed, so it cannot flake: the body scan must
    not run once per snapshot while the call is still unterminated."""
    from core import tool_healing
    from core.inference.tool_call_parser import StreamingMarkupStripper

    calls = {"n": 0}
    real = tool_healing._balanced_json_span

    def counting(text, start):
        calls["n"] += 1
        return real(text, start)

    text = 'terminal[ARGS]{"command":"%s"}' % ("A" * 2048)
    snapshots = 0
    tool_healing._balanced_json_span = counting
    try:
        stripper = StreamingMarkupStripper({"terminal", "python"})
        i = 0
        while i < len(text):
            i += 16
            stripper.strip(text[:i])
            snapshots += 1
    finally:
        tool_healing._balanced_json_span = real

    # Roughly one scan per snapshot is the design; two per snapshot means the kept call's
    # body end is being resolved eagerly again, which is what made this quadratic.
    assert calls["n"] < 1.5 * snapshots, f"{calls['n']} scans for {snapshots} snapshots"


def test_a_body_scan_with_no_closing_brace_short_circuits():
    """The span can only close on a ``}``; with none present the walk cannot succeed, so the
    early return is exactly equivalent and keeps the streaming rescan cheap."""
    from core.tool_healing import _balanced_json_span

    assert _balanced_json_span('{"command":"' + "A" * 4096, 0) is None
    assert _balanced_json_span('{"command":"x"}', 0) == 14
    assert _balanced_json_span('{"c":"}"}', 0) == 8
