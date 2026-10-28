# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Closed-loop rounds: the probe/test split of the held-out rows and the failures seed.

Round 1's held-out rows are split once into a probe half (its failures feed the next round) and a
test half (never fed back; its score decides stopping). Without the split the next round trains on
rewrites of the rows it is scored on."""

from __future__ import annotations

import json
import random
from typing import Any, Optional

FIELDS = {
    "failed_prompt": ("prompt", "input", "question", "instruction", "user"),
    "failed_reference": ("reference", "expected", "answer", "target", "label", "assistant"),
    "failed_output": ("output", "prediction", "response", "generated", "completion"),
}
SEED_COLUMNS = ("origin", *FIELDS)
# A string this long from a test row is the row itself, not a shared label like "billing".
LEAK_MIN_CHARS = 16


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


def failure_row(failure: Any) -> Optional[dict]:
    if not isinstance(failure, dict):
        return None
    lowered = {str(k).lower(): v for k, v in failure.items()}
    row = {"origin": "failure"}
    for column, keys in FIELDS.items():
        row[column] = next((_text(lowered[k]) for k in keys if k in lowered), "")
    return row if any(row[c] for c in FIELDS) else None


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


def seed_rows(failures: list[dict], rows: int, failure_percent: int) -> list[dict]:
    """``rows`` seed rows: up to ``failure_percent`` of them rewrite a failure (cycled), the rest are fresh."""
    rows = max(1, rows)
    kept = [r for r in (failure_row(f) for f in failures) if r]
    take = min(rows, round(rows * failure_percent / 100)) if kept else 0
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
