# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The rung axis is sized by the MEASURED ratio, not by the provisional 4.0.

`PROVISIONAL_CHARS_PER_TOKEN` is what a rung is planned with before anything has been tokenised.
The production caller used to leave it there forever: `build_cells` took a hard-coded 4.0, the
per-cell `measure_chars_per_token` ran only after the thread was seeded, and its answer was
recorded and read by nothing. So every cell labelled 1M tokens carried 4,000,000 characters of a
corpus tiktoken reads at about 3.34 -- roughly 1.2M tokens, a fifth over its own label, on the very
axis the onset headline is quoted against.

Two halves, and both are needed. The ladder is sized from a real tokeniser's answer, and a machine
that has none keeps the provisional ratio and SAYS SO rather than sizing the corpus from the
whitespace estimate, which reads 6.7 on this dense-code corpus and is past what the manifest was
frozen for.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from studiobench.fixture.corpus import (  # noqa: E402
    PROVISIONAL_CHARS_PER_TOKEN,
    RUNGS,
    Corpus,
)
from studiobench.runtime import session as session_mod  # noqa: E402
from studiobench.runtime.seeder import measure_chars_per_token  # noqa: E402
from studiobench.runtime.session import build_cells  # noqa: E402


def _corpus() -> Corpus:
    return Corpus.load()


def _measured(corpus: Corpus) -> dict:
    """The corpus's own ratio, measured the way the harness says it measures it."""
    text: list[str] = []
    size = 0
    for entry in corpus.manifest["units"]:
        unit = corpus.unit(entry["index"])
        text.append(unit.reasoning + unit.content)
        size += unit.chars
        if size >= 200_000:
            break
    return measure_chars_per_token("".join(text)[:200_000], "", None, "")


def test_the_ladder_is_sized_by_the_measured_ratio_and_not_the_provisional_one():
    pytest.importorskip("tiktoken", reason = "this case is the real-tokeniser one")
    corpus = _corpus()
    ratio = _measured(corpus)
    assert ratio["source"] == "tiktoken/cl100k", ratio
    assert ratio["chars_per_token"] != PROVISIONAL_CHARS_PER_TOKEN, ratio

    cells = build_cells(list(RUNGS), corpus, "full", "s0", 0)
    assert cells, "the ladder built no cells"
    for cell, plan in cells:
        assert plan.target_chars == int(RUNGS[cell.rung] * ratio["chars_per_token"]), (
            cell.rung,
            plan.target_chars,
        )
        # The bug, stated as the number it produced: the top rung was 4,000,000 characters.
        assert plan.target_chars != int(RUNGS[cell.rung] * PROVISIONAL_CHARS_PER_TOKEN), cell.rung
        assert cell.meta["ladder_chars_per_token"]["chars_per_token"] == ratio["chars_per_token"]
        assert cell.meta["ladder_chars_per_token"]["provisional"] is False


def test_a_caller_that_names_a_ratio_still_gets_that_ratio():
    corpus = _corpus()
    cells = build_cells(["10K"], corpus, "quick", "s0", 0, chars_per_token = 4.5)
    (cell, plan) = cells[0]
    assert plan.target_chars == int(10_000 * 4.5)
    assert cell.meta["ladder_chars_per_token"]["source"] == "caller"


def test_a_machine_with_no_tokeniser_keeps_the_provisional_ratio_and_says_so(monkeypatch):
    """The whitespace estimate reads 6.675 here, past `MANIFEST_CHARS_PER_TOKEN`.

    Sizing the ladder from it would move the error rather than remove it and would make `plan_rung`
    refuse the whole run. The estimate is still measured and still reported; it just does not size
    the axis.
    """
    monkeypatch.setattr(
        session_mod,
        "measure_chars_per_token",
        lambda *a, **k: {
            "chars_per_token": 6.675,
            "source": "whitespace-and-punctuation estimate",
        },
    )
    corpus = _corpus()
    cells = build_cells(["1M"], corpus, "full", "s0", 0)
    (cell, plan) = cells[0]
    assert plan.target_chars == int(1_000_000 * PROVISIONAL_CHARS_PER_TOKEN)
    meta = cell.meta["ladder_chars_per_token"]
    assert meta["provisional"] is True
    assert meta["measured"] == 6.675
    assert "no tokeniser answered" in meta["reason"]
