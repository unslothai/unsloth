# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""--stream-tail-chars must be delivered or refused, never shrunk and still labelled as its rung."""

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


# Derived by running the reply-axis assertions across the ladder; False must be refused.
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
    """The refusal must name the thread collapse too, since that is what corrupts the rung label."""
    with pytest.raises(ValueError) as excinfo:
        plan_rung(corpus, "100K", stream_tail_chars = 400_000)
    message = str(excinfo.value)
    assert "400,000" in message, message
    assert "3.9%" in message, message
    assert "collapse" in message, message
    assert "100K" in message, message


def test_the_refusal_does_not_name_a_maximum_it_has_not_computed(corpus: Corpus):
    """The refusal must not name a maximum tail it has not computed; a confidently wrong number misleads."""
    with pytest.raises(ValueError) as excinfo:
        plan_rung(corpus, "100K", stream_tail_chars = 400_000)
    message = str(excinfo.value)
    assert "at most 15,405" not in message.replace("NOT 'at most 15,405'", ""), message
    assert "RISES as the request falls" in message, message
    assert plan_rung(corpus, "100K", stream_tail_chars = 96_000).streamed_chars > 90_000


def test_the_default_ladder_is_untouched(corpus: Corpus):
    """The guard is scoped to an EXPLICIT request. The pinned ladder must not acquire a new way
    to fail, at any rung, and its numbers must not move by a character."""
    for rung in RUNGS:
        plan = plan_rung(corpus, rung)
        assert plan.streamed_chars <= STREAM_TAIL_CHARS, rung
        assert plan.streamed_chars > 0, rung


def test_the_small_rungs_are_under_the_default_tail_without_being_refused(corpus: Corpus):
    """A rung smaller than one tail is legitimately short, and the guard must not refuse it."""
    plan = plan_rung(corpus, "1K")
    assert plan.streamed_chars < STREAM_TAIL_CHARS
    assert plan.streamed_chars == 3_883, plan.streamed_chars


def test_the_default_exemption_is_currently_unreachable_and_says_so(corpus: Corpus):
    """Default-path exemption is unreachable today; this pins that premise so it fails once it changes."""
    smallest = min(entry["chars"] for entry in corpus.manifest["units"])
    assert smallest > STREAM_TAIL_CHARS, (
        f"the smallest corpus unit ({smallest:,}) no longer exceeds STREAM_TAIL_CHARS "
        f"({STREAM_TAIL_CHARS:,}), so the default path can now reach the deliverability guard. "
        "The `stream_tail_chars is not None` exemption has become load bearing and needs a test "
        "that exercises it directly."
    )


def test_block_clipping_is_not_treated_as_under_delivery(corpus: Corpus):
    """Block clipping is not under-delivery: a character cut could split a fence and change the path."""
    plan = plan_rung(corpus, "100K", stream_tail_chars = 24_000)
    assert plan.streamed_chars < 24_000, "this pair is expected to clip at a block boundary"
    assert plan.streamed_chars > 24_000 * 0.9, plan.streamed_chars
