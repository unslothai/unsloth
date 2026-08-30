# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Math in the corpus, and the two properties adding it had to preserve.

Corpus v1 held not one dollar sign across 519,859 characters. That is why `preprocessLaTeX`
measured as a real cost in isolation and as an exact NULL in the browser: the film gave it nothing
to do, and a benchmark that cannot see a cost is not evidence the cost is absent.

Adding content to a calibrated fixture is the easy way to invalidate it, so two things are pinned
here rather than argued in a comment: the fence share, which is what the Shiki span density rests
on, and the preamble, which is the film's only span-free stretch and therefore the only place the
onset of cost can be seen against.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from studiobench.fixture.corpus import (  # noqa: E402
    CORPUS_VERSION,
    SHIPPED_CHARS_BUDGET,
    Corpus,
    _prose,
    corpus_hash,
    generate_unit,
    units_for_chars,
)

FENCE = re.compile(r"```.*?```", re.S)
DISPLAY = (re.compile(r"\$\$\n(.*?)\n\$\$", re.S), re.compile(r"\\\[\n(.*?)\n\\\]", re.S))
INLINE = (re.compile(r"\$ ([^$\n]+?) \$"), re.compile(r"\\\( (.+?) \\\)"))

# The v1 corpus, measured before math was added.
#
# The tolerance is set from measurement rather than taste. With MATH_BLOCK_PROB at 0 the share is
# 0.4754, i.e. EXACTLY v1, which is the substitution design working: inline math is spent from the
# prose block's own budget and costs the fence share nothing. At the shipped 0.16 it reads 0.4779,
# a drift of 0.0025. At 0.60 it reads 0.4689, a drift of 0.0065. So 0.005 leaves the shipped value
# a factor of two of headroom while failing well before the parameter could be raised far enough to
# re-weight the film.
V1_FENCE_SHARE = 0.4754
FENCE_SHARE_TOLERANCE = 0.005


def _texts() -> list[str]:
    return [u.reasoning + "\n" + u.content for u in units_for_chars(SHIPPED_CHARS_BUDGET)]


def _count(text: str, patterns) -> int:
    return sum(len(p.findall(text)) for p in patterns)


# ── the gap this closes ─────────────────────────────────────────────


def test_the_corpus_contains_math_at_all():
    # The literal v1 defect, stated as an assertion so it cannot come back unnoticed.
    joined = "\n".join(_texts())
    assert joined.count("$") > 0


def test_both_delimiter_families_are_present():
    # `$...$` is what remark-math consumes directly. `\(...\)` and `\[...\]` reach the renderer
    # ONLY if preprocessLaTeX rewrites them first, so a corpus carrying only the first kind
    # exercises the renderer while leaving the preprocessor uncovered, which is the situation this
    # corpus version exists to end.
    joined = "\n".join(_texts())
    assert joined.count("$$") > 0
    assert joined.count("\\[") > 0
    assert joined.count("\\(") > 0


def test_math_is_spread_across_the_thread_not_pooled_into_one_turn():
    # A single maths-heavy turn measures one large KaTeX tree. The cost this is meant to expose
    # accumulates over a long thread, so it has to be present in most turns.
    texts = _texts()
    with_math = [t for t in texts if _count(t, DISPLAY) or _count(t, INLINE)]
    assert len(with_math) >= max(2, int(len(texts) * 0.8))


def test_there_is_both_display_and_inline_math():
    joined = "\n".join(_texts())
    assert _count(joined, DISPLAY) >= 10
    # Inline is the common case in a real reply and takes a different path through the pipeline,
    # interleaved with text rather than sitting in its own block.
    assert _count(joined, INLINE) >= _count(joined, DISPLAY)


# ── what adding it had to leave alone ───────────────────────────────


def test_the_fence_share_is_what_it_was_before_math_existed():
    # The span-density calibration in the module docstring is a statement about how much of the
    # corpus is fenced code. Math takes the PROSE slot and is drawn from the prose size
    # distribution precisely so this number does not move; if it has moved, the 5.6 chars/span
    # target no longer describes this film and the docstring is lying.
    texts = _texts()
    total = sum(len(t) for t in texts)
    fenced = sum(len(m.group(0)) for t in texts for m in FENCE.finditer(t))
    assert abs(fenced / total - V1_FENCE_SHARE) < FENCE_SHARE_TOLERANCE


def test_the_preamble_stays_free_of_math_and_fences():
    # The field capture held a flat 60 fps and exactly zero spans for its first 33,348 characters.
    # The preamble stands in for that. Give it math and the film loses the one stretch against
    # which the onset of cost is visible, and every rung starts with a rendering cost instead.
    for index in range(4):
        unit = generate_unit(index)
        head = unit.reasoning[: int(len(unit.reasoning) * 0.20)]
        assert "```" not in head
        assert "$" not in head
        assert "\\(" not in head
        assert "\\[" not in head


def test_prose_without_the_math_flag_has_none():
    # `_prose` is also what builds tool arguments and results, where a stray dollar sign would be
    # rendered by a different component than the one under test.
    import random

    text = _prose(random.Random(7), 4_000, "x")
    assert "$" not in text
    assert "\\(" not in text


def test_every_expression_is_balanced():
    """Structural check for the failure that would make this corpus measure the wrong thing.

    KaTeX renders an expression it cannot parse as an ERROR NODE rather than failing, so a corpus
    of malformed LaTeX would run, look busy, and measure the cost of drawing error messages. That
    would be worse than having no math at all, because it would come with numbers.

    A full parse needs KaTeX itself, which this test cannot import. All 430 expressions in the
    shipped corpus were checked against `katex.renderToString(..., {throwOnError: true})` and all
    430 parsed; that run is quoted in the pull request rather than repeated here. What IS repeated
    here is the check that catches every way the generator could break on a later edit: brace
    balance, and no empty group, which is what a missing interpolation would leave behind.
    """
    joined = "\n".join(_texts())
    bodies = [m.group(1) for p in DISPLAY for m in p.finditer(joined)]
    bodies += [m.group(1) for p in INLINE for m in p.finditer(joined)]
    assert bodies, "no expressions found, so this test is not checking anything"
    for body in bodies:
        depth = 0
        for ch in body:
            depth += (ch == "{") - (ch == "}")
            assert depth >= 0, body
        assert depth == 0, body
        assert "{}" not in body, body
        # A trailing backslash-command with nothing after it renders as an error node.
        assert not body.rstrip().endswith("\\"), body


# ── still frozen ────────────────────────────────────────────────────


def test_the_shipped_corpus_matches_the_generator_byte_for_byte():
    # The whole point of freezing. If this fails, someone edited the generator without re-running
    # `freeze`, and the shipped film and the generated film are two different benchmarks.
    corpus = Corpus.load()
    for index in range(min(6, len(corpus.manifest["units"]))):
        assert generate_unit(index, corpus.seed).sha256 == corpus.manifest["units"][index]["sha256"]


def test_the_manifest_records_the_math_parameters():
    # The corpus hash covers the generator's parameters as well as its bytes, so two corpora with
    # the same text and different declared densities cannot compare as identical.
    manifest = Corpus.load().manifest
    assert "math_block_prob" in manifest
    assert "inline_math_prob" in manifest


def test_changing_a_math_parameter_changes_the_corpus_hash():
    manifest = dict(Corpus.load().manifest)
    before = corpus_hash(manifest)
    manifest["math_block_prob"] = manifest["math_block_prob"] + 0.01
    assert corpus_hash(manifest) != before


def test_the_corpus_version_was_bumped_for_the_content_change():
    # A number taken on v1 and a number taken on v2 are measurements of two different films.
    # `floor_table` refuses to pool them, and it can only do that if this moved.
    assert CORPUS_VERSION >= 2
    assert Corpus.load().manifest["corpus_version"] == CORPUS_VERSION
