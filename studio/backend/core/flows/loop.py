# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Closed-loop rounds: the probe/test split of the held-out rows and the failures seed.

Round 1's held-out rows are split once into a probe half (its failures feed the next round) and a
test half (never fed back; its score decides stopping). Without the split the next round trains on
rewrites of the rows it is scored on."""

from __future__ import annotations

import json
import random
import re
from difflib import SequenceMatcher
from typing import Any, Optional

# Same guesses as the Benchmark node, so a failed row maps the way it was scored.
PROMPT_KEYS = ("prompt", "instruction", "question", "input", "query", "user")
REFERENCE_KEYS = ("response", "output", "answer", "completion", "reference", "target", "assistant")
OUTPUT_KEYS = ("model_output", "output", "prediction", "generated")
FIELDS = ("failed_prompt", "failed_reference", "failed_output")
# A string this long from a test row is the row itself, not a shared label like "billing".
LEAK_MIN_CHARS = 16
# Seed defaults: a quarter of the rows rewrite failures, and no failure gets more than two of them,
# so a round with a handful of failures stays mostly fresh rows instead of near-copies of a few.
FAILURE_PERCENT = 25
PER_FAILURE = 2
# Failures whose normalized prompts match this closely (same reference) seed as one.
NEAR_SAME = 0.9


def split_holdout(rows: list[dict], probe_percent: int, seed: int) -> tuple[list[dict], list[dict]]:
    """Seeded, disjoint, covers every row; at least one row in each half when there are two."""
    if len(rows) < 2:
        return [], list(rows)
    order = list(range(len(rows)))
    random.Random(seed).shuffle(order)
    count = min(len(rows) - 1, max(1, round(len(rows) * probe_percent / 100)))
    probe = sorted(order[:count])
    test = sorted(order[count:])
    return [rows[i] for i in probe], [rows[i] for i in test]


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list) and value and isinstance(value[-1], dict):
        # A chat transcript: the last message is what was asked or answered.
        return _text(value[-1].get("content"))
    return json.dumps(value, ensure_ascii = False)


def failure_row(
    failure: Any,
    prompt_column: str = "",
    reference_column: str = "",
) -> Optional[dict]:
    if not isinstance(failure, dict):
        return None
    lowered = {str(k).lower(): v for k, v in failure.items()}
    output = next((k for k in OUTPUT_KEYS if k in lowered), None)
    rest = [k for k in lowered if k not in (output, "reason", "origin")]

    def pick(named: str, guesses: tuple[str, ...], skip: Optional[str]) -> Optional[str]:
        if named and named.lower() in lowered:
            return named.lower()
        return next((k for k in guesses if k in rest and k != skip), None)

    prompt = pick(prompt_column, PROMPT_KEYS, None)
    reference = pick(reference_column, REFERENCE_KEYS, prompt)
    row = {
        "origin": "failure",
        "failed_prompt": _text(lowered.get(prompt)) if prompt else "",
        "failed_reference": _text(lowered.get(reference)) if reference else "",
        "failed_output": _text(lowered.get(output)) if output else "",
    }
    return row if row["failed_prompt"] else None


def _strings(value: Any) -> list[str]:
    found: list[str] = []
    stack = [value]
    while stack:
        item = stack.pop()
        if isinstance(item, str):
            found.append(item.strip())
        elif isinstance(item, dict):
            stack.extend(item.values())
        elif isinstance(item, list):
            stack.extend(item)
    return found


def drop_test_rows(failures: list[dict], test_rows: list[dict]) -> list[dict]:
    """Guard: a failure carrying any long string from a test row never reaches the seed."""
    banned = {s for s in _strings(test_rows) if len(s) >= LEAK_MIN_CHARS}
    if not banned:
        return list(failures)
    return [f for f in failures if not any(s in banned for s in _strings(f))]


def _norm(text: str) -> str:
    return " ".join(re.sub(r"[^\w\s]", " ", text.lower()).split())


def distinct(rows: list[dict]) -> list[dict]:
    """Drop failure rows whose prompt is a near-copy of an earlier one with the same reference."""
    kept: list[tuple[str, str, dict]] = []
    for row in rows:
        prompt, reference = _norm(row["failed_prompt"]), _norm(row["failed_reference"])
        if any(
            reference == ref
            and (prompt == seen or SequenceMatcher(None, prompt, seen).ratio() >= NEAR_SAME)
            for seen, ref, _ in kept
        ):
            continue
        kept.append((prompt, reference, row))
    return [row for _, _, row in kept]


def seed_rows(
    failures: list[dict],
    rows: int,
    failure_percent: int = FAILURE_PERCENT,
    prompt_column: str = "",
    reference_column: str = "",
    per_failure: int = PER_FAILURE,
) -> list[dict]:
    """``rows`` seed rows: up to ``failure_percent`` of them rewrite a distinct failure, each failure at
    most ``per_failure`` times; the rest are fresh."""
    rows = max(1, rows)
    kept = distinct(
        [r for r in (failure_row(f, prompt_column, reference_column) for f in failures) if r]
    )
    take = min(rows, round(rows * failure_percent / 100), len(kept) * max(1, per_failure))
    found = [dict(kept[i % len(kept)]) for i in range(take)]
    found += [{"origin": "fresh", **{c: "" for c in FIELDS}} for _ in range(rows - take)]
    return found


def as_percent(score: Any) -> Optional[float]:
    """A score node may answer 0..1 or 0..100; a dict with `score` or `accuracy` also works."""
    if isinstance(score, dict):
        score = next((score[k] for k in ("score", "accuracy", "value") if k in score), None)
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        return None
    value = float(score) * 100 if 0 <= score <= 1 else float(score)
    return round(max(0.0, min(100.0, value)), 2)
