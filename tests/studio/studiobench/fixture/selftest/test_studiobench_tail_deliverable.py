# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""An explicit `--stream-tail-chars` must be delivered or refused, never quietly shrunk.

`test_studiobench_reply_axis.py` already asserts the right property -- the reply lands within 10%
of the ask and the thread does not move -- but only at the 100K rung and only at 24,000 and 96,000.
Re-run those same two assertions across the ladder before this guard existed and 10 of 15
(rung, tail) pairs failed them. The shipped test covered exactly the 2 that passed. A test can be
right about the property and wrong about the coverage, and the value of this file is the matrix.

The failure it pins is silent in both directions at once, which is why nothing caught it:
`--rungs 100K --stream-tail-chars 400000` streamed 15,405 of the 400,000 requested (3.9%) on a
thread of 18,122 characters against the rung's 397,755 (4.6%), and the cell still carried the
`r100K` id that `scoring/from_payload.py` and `report/build.py` use as the rung key. So a 4.6% rung
was reported, weighted and pooled as 100K. `--assert-liveness` returned `0 scene problems, 0 missed
slots`, exit 0, byte-identical to the default run: a saturated tail drains FASTER than the film it
was meant to outlast, so the one gate that could have noticed goes quieter as the input gets worse.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from studiobench.fixture.corpus import (  # noqa: E402
    RUNGS,
    STREAM_TAIL_CHARS,
    Corpus,
    plan_rung,
)


@pytest.fixture(scope = "module")
def corpus() -> Corpus:
    return Corpus.load()


#: Every pair the axis is plausibly asked for, and what the corpus can actually do with it.
#: `True` means deliverable, `False` means the request has to be refused. Derived by running the
#: reply-axis test's own two assertions across the ladder, not chosen: every `False` here is a
#: pair that silently under-delivered before the guard, and every `True` is one that did not.
MATRIX: list[tuple[str, int, bool]] = [
    ("1K", 24_000, False),
    ("1K", 96_000, False),
    ("1K", 400_000, False),
    ("10K", 24_000, False),
    ("10K", 48_000, False),
    ("10K", 96_000, False),
    ("100K", 24_000, True),
    ("100K", 48_000, True),
    ("100K", 96_000, True),
    ("100K", 200_000, False),
    ("100K", 400_000, False),
    ("500K", 96_000, True),
    ("500K", 200_000, True),
    ("500K", 400_000, False),
    ("1M", 96_000, True),
    ("1M", 200_000, True),
    ("1M", 400_000, False),
]


@pytest.mark.parametrize(("rung", "tail", "deliverable"), MATRIX)
def test_every_requested_tail_is_delivered_or_refused(
    corpus: Corpus, rung: str, tail: int, deliverable: bool
):
    """The whole property, over the whole matrix: no third outcome.

    The two assertions are the reply-axis test's own, applied to every pair rather than to two of
    them. Where the corpus cannot answer, the requirement is a refusal and not a smaller number.
    """
    if not deliverable:
        with pytest.raises(ValueError, match = "cannot deliver"):
            plan_rung(corpus, rung, stream_tail_chars = tail)
        return

    plan = plan_rung(corpus, rung, stream_tail_chars = tail)
    base = plan_rung(corpus, rung)
    assert abs(plan.streamed_chars - tail) < tail * 0.1, (rung, tail, plan.streamed_chars)
    assert abs(plan.total_chars - base.total_chars) < base.total_chars * 0.05, (
        rung,
        tail,
        plan.total_chars,
        base.total_chars,
    )


def test_the_refusal_names_the_collapse_and_not_only_the_short_reply(corpus: Corpus):
    """Both effects, because fixing only the one you noticed leaves the other.

    A reader told just "the reply was short" lowers the tail and moves on. The thread collapse is
    the half that corrupts the rung label, and it is the half a short reply does not imply.
    """
    with pytest.raises(ValueError) as excinfo:
        plan_rung(corpus, "100K", stream_tail_chars = 400_000)
    message = str(excinfo.value)
    assert "400,000" in message, message
    assert "3.9%" in message, message
    assert "collapse" in message, message
    assert "100K" in message, message


def test_the_refusal_does_not_name_a_maximum_it_has_not_computed(corpus: Corpus):
    """FAILURE DIRECTION, and it pins a bug this guard shipped with for one revision.

    The obvious wording is "ask for at most N", N being the unit the request landed on. It is
    wrong, and confidently so: the ceiling RISES as the request falls, because a smaller tail
    leaves a longer seeded prefix which draws the streamed turn from a larger unit. At 100K the
    first draft told the reader to ask for at most 15,405 while 96,000 succeeds -- a factor of six
    in the direction that makes someone abandon a measurement they could have taken.

    A refusal that confidently names the wrong number is worse than a vaguer one, because it sends
    the reader somewhere specific and wrong.
    """
    with pytest.raises(ValueError) as excinfo:
        plan_rung(corpus, "100K", stream_tail_chars = 400_000)
    message = str(excinfo.value)
    assert "at most 15,405" not in message.replace("NOT 'at most 15,405'", ""), message
    # And the claim the reader needs instead has to actually be there.
    assert "RISES as the request falls" in message, message
    # The thing the wrong advice would have forbidden must in fact work.
    assert plan_rung(corpus, "100K", stream_tail_chars = 96_000).streamed_chars > 90_000


def test_the_default_ladder_is_untouched(corpus: Corpus):
    """The guard is scoped to an EXPLICIT request. The pinned ladder must not acquire a new way
    to fail, at any rung, and its numbers must not move by a character."""
    for rung in RUNGS:
        plan = plan_rung(corpus, rung)
        assert plan.streamed_chars <= STREAM_TAIL_CHARS, rung
        assert plan.streamed_chars > 0, rung


def test_the_small_rungs_are_under_the_default_tail_without_being_refused(corpus: Corpus):
    """The exemption that makes the guard correct rather than merely strict.

    The 1K rung is 4,000 characters in total, less than one tail, so it streams 3,883 and is
    legitimately short. That shortfall is the rung being small, not the corpus failing to answer a
    question it was asked, and a guard that could not tell those apart would refuse the default
    ladder at its own smallest rung.
    """
    plan = plan_rung(corpus, "1K")
    assert plan.streamed_chars < STREAM_TAIL_CHARS
    assert plan.streamed_chars == 3_883, plan.streamed_chars


def test_the_default_exemption_is_currently_unreachable_and_says_so(corpus: Corpus):
    """HONESTY ABOUT COVERAGE. The `stream_tail_chars is not None` clause has no teeth today.

    Mutation-tested and NOT caught: dropping that clause leaves all 168 selftests passing and the
    default ladder byte-identical, because the guard's other half can never fire on the default
    path -- `STREAM_TAIL_CHARS` is 6,000 and the smallest unit in the frozen corpus is 11,374, so
    `tail_target >= source.chars` is unreachable without an explicit request. The clause is
    defensive, not load bearing, and claiming the matrix above covers it would be exactly the
    "right about the property, wrong about the coverage" mistake this file exists to correct.

    So pin the PREMISE instead of faking coverage of the branch. Raise `STREAM_TAIL_CHARS` above
    the smallest unit, or re-freeze a corpus with smaller units, and this fails -- at which point
    the exemption becomes load bearing and needs a real test rather than this one.
    """
    smallest = min(entry["chars"] for entry in corpus.manifest["units"])
    assert smallest > STREAM_TAIL_CHARS, (
        f"the smallest corpus unit ({smallest:,}) no longer exceeds STREAM_TAIL_CHARS "
        f"({STREAM_TAIL_CHARS:,}), so the default path can now reach the deliverability guard. "
        "The `stream_tail_chars is not None` exemption has become load bearing and needs a test "
        "that exercises it directly."
    )


def test_block_clipping_is_not_treated_as_under_delivery(corpus: Corpus):
    """The other half of the same distinction, on the EXPLICIT path.

    A tail clipped at a whole block boundary loses at most one block, and that clipping is load
    bearing: a character-aligned prefix can end inside a fence, and an unclosed fence is a
    different Streamdown path with a different cost. Only exceeding the unit is unbounded. The
    guard keys on `clipped_to`'s own early return rather than on a percentage, which is what lets
    it separate the two; a bound nobody can derive would be an unacknowledged bias, not a
    tolerance.
    """
    plan = plan_rung(corpus, "100K", stream_tail_chars = 24_000)
    assert plan.streamed_chars < 24_000, "this pair is expected to clip at a block boundary"
    assert plan.streamed_chars > 24_000 * 0.9, plan.streamed_chars
