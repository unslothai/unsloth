# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Accuracy floor for the plan-without-action classifier, scored on a corpus of real model output."""

import json
from pathlib import Path

from core.inference.llama_cpp import _should_suppress_forced_no_tool_output
from core.inference.tool_call_parser import is_short_intent_without_action

DATA = Path(__file__).parent / "data" / "plan_vs_answer.jsonl"

# above the measured count in the table above, so wording changes alone do not fail the build
NUDGE_BUDGET = 9
# tighter than the nudge budget, because a discarded retry destroys output
DISCARD_BUDGET = 4


def _corpus():
    with open(DATA, encoding = "utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _report(rows, limit = 10):
    lines = []
    for row in rows[:limit]:
        text = " ".join(row["text"].split())
        lines.append(
            f"  [{row['model']}/{row['prompt_class']}] {row['prompt']!r}\n    {text[:200]!r}"
        )
    if len(rows) > limit:
        lines.append(f"  ... and {len(rows) - limit} more")
    return "\n".join(lines)


def test_corpus_is_intact():
    """Guards the budgets: they mean nothing if the corpus silently shrinks."""
    corpus = _corpus()
    assert len(corpus) == 300
    assert all(row["text"].strip() for row in corpus)
    assert all(row["retry_tool_calls"] == 0 for row in corpus)


def test_finished_answers_are_rarely_nudged():
    """A finished answer costs a whole extra generation when it is nudged."""
    nudged = [row for row in _corpus() if is_short_intent_without_action(row["text"])]
    assert len(nudged) <= NUDGE_BUDGET, (
        f"{len(nudged)}/300 finished answers classified as plans "
        f"(budget {NUDGE_BUDGET}):\n{_report(nudged)}"
    )


def test_finished_answers_are_not_discarded():
    """The retry's text is all the user gets, so discarding it is the worst case."""
    discarded = [
        row
        for row in _corpus()
        if row["retry_text"].strip()
        and _should_suppress_forced_no_tool_output(row["retry_text"], row["text"])
    ]
    assert len(discarded) <= DISCARD_BUDGET, (
        f"{len(discarded)}/300 finished retries would be discarded "
        f"(budget {DISCARD_BUDGET}):\n{_report(discarded)}"
    )
