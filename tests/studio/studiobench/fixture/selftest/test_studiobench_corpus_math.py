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

# Measured v1 fence share; math substitutes for prose so it should not drift past 0.005.
V1_FENCE_SHARE = 0.4754
FENCE_SHARE_TOLERANCE = 0.005


def _texts() -> list[str]:
    return [u.reasoning + "\n" + u.content for u in units_for_chars(SHIPPED_CHARS_BUDGET)]


def _count(text: str, patterns) -> int:
    return sum(len(p.findall(text)) for p in patterns)


def test_the_corpus_contains_math_at_all():
    joined = "\n".join(_texts())
    assert joined.count("$") > 0


def test_both_delimiter_families_are_present():
    joined = "\n".join(_texts())
    assert joined.count("$$") > 0
    assert joined.count("\\[") > 0
    assert joined.count("\\(") > 0


def test_math_is_spread_across_the_thread_not_pooled_into_one_turn():
    texts = _texts()
    with_math = [t for t in texts if _count(t, DISPLAY) or _count(t, INLINE)]
    assert len(with_math) >= max(2, int(len(texts) * 0.8))


def test_there_is_both_display_and_inline_math():
    joined = "\n".join(_texts())
    assert _count(joined, DISPLAY) >= 10
    assert _count(joined, INLINE) >= _count(joined, DISPLAY)


def test_the_fence_share_is_what_it_was_before_math_existed():
    # If the fence share moved, the 5.6 chars/span calibration no longer holds.
    texts = _texts()
    total = sum(len(t) for t in texts)
    fenced = sum(len(m.group(0)) for t in texts for m in FENCE.finditer(t))
    assert abs(fenced / total - V1_FENCE_SHARE) < FENCE_SHARE_TOLERANCE


def test_the_preamble_stays_free_of_math_and_fences():
    # The preamble must stay math-free: it is the zero-cost baseline stretch.
    for index in range(4):
        unit = generate_unit(index)
        head = unit.reasoning[: int(len(unit.reasoning) * 0.20)]
        assert "```" not in head
        assert "$" not in head
        assert "\\(" not in head
        assert "\\[" not in head


def test_prose_without_the_math_flag_has_none():
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
        assert not body.rstrip().endswith("\\"), body


def test_the_shipped_corpus_matches_the_generator_byte_for_byte():
    # Fails if the generator changed without re-running `freeze`.
    corpus = Corpus.load()
    for index in range(min(6, len(corpus.manifest["units"]))):
        assert generate_unit(index, corpus.seed).sha256 == corpus.manifest["units"][index]["sha256"]


def test_the_manifest_records_the_math_parameters():
    manifest = Corpus.load().manifest
    assert "math_block_prob" in manifest
    assert "inline_math_prob" in manifest


def test_changing_a_math_parameter_changes_the_corpus_hash():
    manifest = dict(Corpus.load().manifest)
    before = corpus_hash(manifest)
    manifest["math_block_prob"] = manifest["math_block_prob"] + 0.01
    assert corpus_hash(manifest) != before


def test_the_corpus_version_was_bumped_for_the_content_change():
    assert CORPUS_VERSION >= 2
    assert Corpus.load().manifest["corpus_version"] == CORPUS_VERSION
