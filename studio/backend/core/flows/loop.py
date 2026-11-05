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
# Score nodes whose score output is already a percent.
PERCENT_SCORERS = ("train.benchmark",)
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


def _text(
    value: Any,
    answer: bool = False,
    whole: bool = True,
) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    from core.benchmark.recipe_heldout import cell_messages, prompt_text, reference_text

    if cell_messages(value) is not None:
        # A chat transcript (OpenAI or ShareGPT turns), read as Benchmark asked and scored it.
        return reference_text(value) if answer else prompt_text(value, whole)
    return json.dumps(value, ensure_ascii = False)


# What Benchmark adds to a failed row: never part of the row's own question.
_SCORED = ("reason", "origin", "scored_reference", "scored_prompt")


def failure_row(
    failure: Any,
    prompt_column: Any = "",
    reference_column: str = "",
    as_scored: bool = True,
) -> Optional[dict]:
    """``prompt_column`` names the question's column, or (a dict) the several mapped columns
    Benchmark built it from."""
    if not isinstance(failure, dict):
        return None
    mapped = prompt_column if isinstance(prompt_column, dict) else None
    whole = True
    if mapped and mapped.get("kind") == "column":
        # One column, its assistant prefill kept when a reference column held the answer.
        prompt_column, whole = mapped.get("column"), bool(mapped.get("whole", True))
        mapped = None
    # As Benchmark reads them: " ticket " names the ticket column.
    prompt_column = "" if mapped else str(prompt_column or "").strip()
    reference_column = reference_column.strip()
    lowered = {str(k).lower(): v for k, v in failure.items()}
    output = next((k for k in OUTPUT_KEYS if k in lowered), None)
    rest = [
        k
        for k in lowered
        if k not in (output, "reason", "origin", "scored_reference", "scored_prompt")
    ]

    def pick(named: str, guesses: tuple[str, ...], skip: Optional[str]) -> Optional[str]:
        if named and named.lower() in lowered:
            return named.lower()
        return next((k for k in guesses if k in rest and k != skip), None)

    prompt = pick(prompt_column, PROMPT_KEYS, None)
    reference = pick(reference_column, REFERENCE_KEYS, prompt)
    if mapped and mapped.get("kind") == "rl":
        from core.benchmark.recipe_heldout import prompt_text
        from core.flows.nodes.benchmark import rl_format

        # The prompt training's RL formatter gives the row, as Benchmark asked it.
        own = {k: v for k, v in failure.items() if str(k).lower() not in (*OUTPUT_KEYS, *_SCORED)}
        formatted = rl_format([own], mapped.get("objective"), mapped.get("mapping"))
        question = prompt_text(formatted[0]["prompt"], False) if formatted else ""
    elif mapped:
        from core.benchmark.recipe_heldout import prompt_text
        from core.flows.nodes.benchmark import mapped_question

        columns = [*(mapped.get("columns") or []), *(mapped.get("inputs") or [])]
        # A row without those columns (a later round's own rows) asks nothing through them.
        present = any(failure.get(c) not in (None, "") for c in columns if isinstance(c, str))
        question = prompt_text(mapped_question(failure, mapped)) if present else ""
    else:
        question = _text(lowered.get(prompt), whole = whole) if prompt else ""
        extra = _text(lowered.get("input")) if prompt == "instruction" and not prompt_column else ""
        if extra.strip():
            # An Alpaca row was scored as instruction plus input; feed back the same question.
            question = f"{question}\n\n{extra.strip()}"
    scored = lowered.get("scored_prompt")
    if as_scored and isinstance(scored, str) and scored.strip():
        # Benchmark's own question when it gave one, found in the row or not.
        question = scored
    row = {
        "origin": "failure",
        "failed_prompt": question,
        # Benchmark's own scored reference when it gave one: the dataset cell may be a label id or
        # a structured answer the score read differently.
        "failed_reference": _text(lowered["scored_reference"], answer = True)
        if lowered.get("scored_reference") is not None
        else (_text(lowered.get(reference), answer = True) if reference else ""),
        "failed_output": _text(lowered.get(output), answer = True) if output else "",
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


def _system_of(row: dict, prompt_column: Any, system_column: Any) -> str:
    """The row's own system message as Benchmark sent it: its system column (or columns), else the
    system turns of a conversation prompt."""
    from core.benchmark.recipe_heldout import cell_messages
    from core.flows.nodes.benchmark import system_message
    from utils.datasets.cells import cell_text

    own = system_message(row, system_column) if system_column else ""
    if own:
        return own
    if isinstance(prompt_column, str) and prompt_column.strip():
        turns = cell_messages(row.get(prompt_column.strip())) or []
        return "\n\n".join(cell_text(t["content"]) for t in turns if t["role"] == "system").strip()
    return ""


def _asked(
    row: Any,
    prompt_column: Any = "",
    system_column: Any = "",
) -> str:
    """The input a row asks (an Alpaca row's instruction plus input), as Benchmark asked it: its
    question, after its own system message when it has one (rows sharing a question can ask
    different tasks through it)."""
    # Read from the row alone, as a test row (which carries no scored question) is read.
    found = failure_row(row, prompt_column, as_scored = False) if isinstance(row, dict) else None
    question = found["failed_prompt"].strip() if found else ""
    system = _system_of(row, prompt_column, system_column) if question else ""
    return f"{system}\n\n{question}" if system else question


def _identifying(
    row: Any,
    prompt_column: Any = "",
    system_column: Any = "",
) -> frozenset:
    """The long strings that could name one test row: its question, never its answer or labels (a rare
    category would otherwise hide every failure sharing it). An Alpaca row's question is its
    instruction plus input, and its input on its own; the instruction alone is shared by its task."""
    asked = _asked(row, prompt_column, system_column)
    if asked:
        strings = {asked}
        lowered = {str(k).lower(): v for k, v in row.items()}
        extra = lowered.get("input")
        if (
            isinstance(extra, str)
            and extra.strip()
            and isinstance(prompt_column, str)
            and "input" != prompt_column.strip().lower()
        ):
            strings.add(extra.strip())
    elif isinstance(row, dict):
        # No question column to go by: every value but the answer and the model's output.
        answers = set(REFERENCE_KEYS) | set(OUTPUT_KEYS)
        strings = set(_strings([v for k, v in row.items() if str(k).lower() not in answers]))
    else:
        strings = set(_strings(row))
    return frozenset(s for s in strings if len(s) >= LEAK_MIN_CHARS)


# A Fine-tune's input (not a port) a Loop fills with its frozen test questions from round 2 on.
EXCLUDE_INPUT = "_loop_test_questions"


def questions(
    rows: list[dict],
    prompt_column: Any = "",
    system_column: Any = "",
) -> list[str]:
    """The questions rows ask, as Benchmark asks them (an Alpaca row's instruction plus input)."""
    return sorted({q for q in (_asked(row, prompt_column, system_column) for row in rows) if q})


def without_questions(rows: list[dict], exclude: Any) -> list[dict]:
    """``rows`` less those asking one of ``exclude``'s questions, read through its column as the score
    step reads them (``{"column": ..., "system": ..., "questions": [...]}``)."""
    if not isinstance(exclude, dict):
        return list(rows)
    banned = {q for q in exclude.get("questions") or [] if isinstance(q, str)}
    column, system = exclude.get("column") or "", exclude.get("system") or ""
    return [r for r in rows if _asked(r, column, system) not in banned] if banned else list(rows)


def drop_test_rows(
    failures: list[dict],
    test_rows: list[dict],
    prompt_column: Any = "",
    system_column: Any = "",
) -> list[dict]:
    """Guard: a failure asking a question that identifies one test row never reaches the seed.
    A value many test rows share (a fixed Alpaca instruction) identifies none of them."""
    # A test question asked again is a leak at any length and however many test rows ask it.
    questions = {q for q in (_asked(row, prompt_column, system_column) for row in test_rows) if q}
    seen: dict[str, int] = {}
    # A row repeated in the test split is still one row: its question stays banned.
    distinct = {_identifying(row, prompt_column, system_column) for row in test_rows}
    for strings in distinct:
        for s in strings:
            seen[s] = seen.get(s, 0) + 1
    banned = {s for s, n in seen.items() if n == 1}
    if not banned and not questions:
        return list(failures)

    def leaks(failure: dict) -> bool:
        asked = _asked(failure, prompt_column, system_column)
        return asked in questions or asked in banned or any(s in banned for s in _strings(failure))

    return [f for f in failures if not leaks(f)]


# Words, signed numbers and arithmetic signs: "-1" and "1", or "5 - 4" and "5 + 4", stay apart.
_TOKEN = re.compile(r"[-+]?\d+(?:[.,]\d+)*|\w+|[-+*/=<>^%]")


def _norm(text: str) -> str:
    return " ".join(_TOKEN.findall(text.lower()))


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
    prompt_column: Any = "",
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


def as_percent(score: Any, percent: bool = False) -> Optional[float]:
    """A score node may answer 0..1 or 0..100; a dict with `score` or `accuracy` also works.
    ``percent`` says the node always answers 0..100 (Benchmark), so 0.5 is half a percent."""
    if isinstance(score, dict):
        score = next((score[k] for k in ("score", "accuracy", "value") if k in score), None)
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        return None
    value = float(score) * 100 if 0 <= score <= 1 and not percent else float(score)
    return round(max(0.0, min(100.0, value)), 2)
