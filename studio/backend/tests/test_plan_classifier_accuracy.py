# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""An accuracy floor for the plan-without-action classifier, on real model output.

The rest of the tool-loop suites pin behaviour on hand-written example sentences,
which is how the patterns here were tuned. That says nothing about how often the
classifier is right on what models actually emit, so this file scores it against a
corpus captured from local models (``tests/data/plan_vs_answer.jsonl``).

How the corpus was built: three GGUF models (Qwen3-0.6B, Qwen3-1.7B,
Llama-3.2-1B-Instruct) were driven through llama-server with the real Studio tool
schemas over prompts spanning tool-requiring questions, questions needing no tool,
list-formatted answers, ambiguous requests, non-English, and follow-ups issued after
a tool had already run. Turns cut off by the token cap were dropped, since a
truncation is not a stall.

Every turn here is a *finished answer*: the turn called no tool, and when the
production nudge was appended and the turn regenerated three times, not one retry
produced a tool call. A forceful re-prompt could not extract an action, so there was
no action left to take. Nudging these is wasted work, and in the GGUF loop the
retry's text can then be discarded, which costs the user a visible answer.

Measured when this landed, over the 300 turns:

    tree                          nudged      retry discarded
    origin/main (pre-PR)          36 (12.0%)  60 (20.2%)
    this PR                        5 ( 1.7%)   1 ( 0.3%)

The budgets below sit above the measured counts so that innocuous wording changes
do not fail the build, and far below the pre-PR counts so a real regression does.
A failure prints the offending turns: fix the pattern, or if the turn really is a
stall, correct its label here.
"""

import json
from pathlib import Path

from core.inference.llama_cpp import _should_suppress_forced_no_tool_output
from core.inference.tool_call_parser import is_short_intent_without_action

DATA = Path(__file__).parent / "data" / "plan_vs_answer.jsonl"

# Measured 5 of 300; pre-PR was 36.
NUDGE_BUDGET = 9
# Measured 1 of 300; pre-PR was 60. Tighter, because this one destroys output.
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
    # Every row is a finished answer by construction.
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
