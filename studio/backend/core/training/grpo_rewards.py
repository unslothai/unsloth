# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Curated reward functions for GRPO training.

Every preset matches TRL's ``GRPOTrainer`` reward signature — it takes
``prompts``, ``completions`` and the dataset columns as keyword arguments, and
returns one float per completion. The registry is closed on purpose: a first
pass ships only first-party callables, so no user-supplied code is executed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable, Optional

REASONING_START = "<think>"
REASONING_END = "</think>"
SOLUTION_START = "<answer>"
SOLUTION_END = "</answer>"

_REASONING_FORMAT_RE = re.compile(
    rf"^\s*{re.escape(REASONING_START)}.*?{re.escape(REASONING_END)}\s*"
    rf"{re.escape(SOLUTION_START)}.*?{re.escape(SOLUTION_END)}\s*$",
    re.DOTALL,
)
_SOLUTION_RE = re.compile(rf"{re.escape(SOLUTION_START)}(.*?){re.escape(SOLUTION_END)}", re.DOTALL)
_NUMBER_RE = re.compile(r"-?\d+(?:[\d,]*\d)?(?:\.\d+)?")


def _completion_text(completion: Any) -> str:
    """Flatten one TRL completion to text.

    Conversational datasets hand back a list of message dicts; prompt-only
    string datasets hand back a plain string.
    """
    if isinstance(completion, str):
        return completion
    if isinstance(completion, dict):
        return str(completion.get("content", "") or "")
    if isinstance(completion, (list, tuple)):
        return "".join(_completion_text(part) for part in completion)
    return "" if completion is None else str(completion)


def _normalize_answer(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value if value is not None else "")).strip().lower()


def _last_number(text: str) -> Optional[float]:
    matches = _NUMBER_RE.findall(text)
    if not matches:
        return None
    try:
        return float(matches[-1].replace(",", ""))
    except ValueError:
        return None


def _extracted_solution(text: str) -> str:
    match = _SOLUTION_RE.search(text)
    return match.group(1) if match else text


def _reference_answers(count: int, kwargs: dict[str, Any]) -> list[Any]:
    for column in ("answer", "solution", "label", "output"):
        values = kwargs.get(column)
        if isinstance(values, (list, tuple)) and len(values) == count:
            return list(values)
    return [None] * count


def exact_answer_match(prompts, completions, **kwargs) -> list[float]:
    """Full credit for matching the reference answer, half for the same number."""
    references = _reference_answers(len(completions), kwargs)
    scores: list[float] = []
    for completion, reference in zip(completions, references):
        if reference is None:
            scores.append(0.0)
            continue
        text = _extracted_solution(_completion_text(completion))
        if _normalize_answer(text) == _normalize_answer(reference):
            scores.append(1.0)
            continue
        guessed = _last_number(text)
        expected = _last_number(str(reference))
        if guessed is not None and expected is not None and guessed == expected:
            scores.append(0.5)
        else:
            scores.append(0.0)
    return scores


def reasoning_format_match(prompts, completions, **kwargs) -> list[float]:
    """Full credit when the completion is exactly <think>…</think><answer>…</answer>."""
    return [
        1.0 if _REASONING_FORMAT_RE.match(_completion_text(completion)) else 0.0
        for completion in completions
    ]


def think_tag_structure(prompts, completions, **kwargs) -> list[float]:
    """Partial credit per reasoning/solution tag, so early runs get a gradient."""
    scores: list[float] = []
    for completion in completions:
        text = _completion_text(completion)
        score = 0.0
        for tag in (REASONING_START, REASONING_END, SOLUTION_START, SOLUTION_END):
            if text.count(tag) == 1:
                score += 0.25
        scores.append(score)
    return scores


def response_length(prompts, completions, **kwargs) -> list[float]:
    """Reward completions near TARGET_CHARS, falling off linearly either side."""
    target = 512.0
    scores: list[float] = []
    for completion in completions:
        length = float(len(_completion_text(completion)))
        scores.append(max(0.0, 1.0 - abs(length - target) / target))
    return scores


@dataclass(frozen = True)
class RewardPreset:
    """A selectable reward function plus what the UI needs to describe it."""

    id: str
    name: str
    description: str
    expected_columns: tuple[str, ...]
    default_weight: float
    function: Callable[..., list[float]]


REWARD_PRESETS: tuple[RewardPreset, ...] = (
    RewardPreset(
        id = "exact_answer_match",
        name = "Exact / numeric answer match",
        description = (
            "Compares the model's answer against a reference column. Full credit for "
            "an exact match, half credit when only the final number agrees."
        ),
        expected_columns = ("answer",),
        default_weight = 2.0,
        function = exact_answer_match,
    ),
    RewardPreset(
        id = "reasoning_format_match",
        name = "Reasoning format match",
        description = (
            "Full credit when the whole completion matches "
            "<think>…</think><answer>…</answer> and nothing else."
        ),
        expected_columns = (),
        default_weight = 1.0,
        function = reasoning_format_match,
    ),
    RewardPreset(
        id = "think_tag_structure",
        name = "<think> tag structure",
        description = (
            "Quarter credit for each reasoning and answer tag that appears exactly "
            "once, so a model that is still learning the format gets a gradient."
        ),
        expected_columns = (),
        default_weight = 0.5,
        function = think_tag_structure,
    ),
    RewardPreset(
        id = "response_length",
        name = "Response length shaping",
        description = (
            "Peaks at roughly 512 characters and falls off linearly, discouraging "
            "one-word answers and runaway rollouts."
        ),
        expected_columns = (),
        default_weight = 0.25,
        function = response_length,
    ),
)

REWARD_PRESETS_BY_ID: dict[str, RewardPreset] = {preset.id: preset for preset in REWARD_PRESETS}

DEFAULT_REWARD_PRESET_IDS: tuple[str, ...] = (
    "exact_answer_match",
    "reasoning_format_match",
    "think_tag_structure",
)


def reward_preset_catalog() -> list[dict[str, Any]]:
    """Serializable preset metadata for the frontend picker."""
    return [
        {
            "id": preset.id,
            "name": preset.name,
            "description": preset.description,
            "expected_columns": list(preset.expected_columns),
            "default_weight": preset.default_weight,
            "default_selected": preset.id in DEFAULT_REWARD_PRESET_IDS,
        }
        for preset in REWARD_PRESETS
    ]


def _weighted(preset: RewardPreset, weight: float) -> Callable[..., list[float]]:
    if weight == 1.0:
        return preset.function

    def scaled(prompts, completions, **kwargs) -> list[float]:
        return [weight * score for score in preset.function(prompts, completions, **kwargs)]

    # TRL names each reward column after the callable, and the run's per-function
    # metric breakdown is keyed off that name.
    scaled.__name__ = preset.function.__name__
    return scaled


def build_reward_functions(
    selections: Optional[list[dict[str, Any]]],
) -> tuple[list[Callable[..., list[float]]], list[str]]:
    """Resolve ``[{"id": ..., "weight": ...}]`` into callables and their names.

    Raises ValueError for an unknown preset id or a non-finite weight: silently
    dropping one would train against a reward the user never asked for.
    """
    if not selections:
        selections = [
            {"id": preset_id, "weight": REWARD_PRESETS_BY_ID[preset_id].default_weight}
            for preset_id in DEFAULT_REWARD_PRESET_IDS
        ]

    functions: list[Callable[..., list[float]]] = []
    names: list[str] = []
    seen: set[str] = set()
    for selection in selections:
        preset_id = str(selection.get("id", "")).strip()
        preset = REWARD_PRESETS_BY_ID.get(preset_id)
        if preset is None:
            raise ValueError(
                f"Unknown GRPO reward function {preset_id!r}; "
                f"available: {', '.join(sorted(REWARD_PRESETS_BY_ID))}"
            )
        if preset_id in seen:
            raise ValueError(f"GRPO reward function {preset_id!r} was selected more than once")
        seen.add(preset_id)

        raw_weight = selection.get("weight", preset.default_weight)
        try:
            weight = float(raw_weight)
        except (TypeError, ValueError):
            raise ValueError(f"Weight for reward function {preset_id!r} must be a number")
        if weight != weight or weight in (float("inf"), float("-inf")):
            raise ValueError(f"Weight for reward function {preset_id!r} must be finite")
        if weight <= 0:
            raise ValueError(f"Weight for reward function {preset_id!r} must be > 0")

        functions.append(_weighted(preset, weight))
        names.append(preset.function.__name__)

    if not functions:
        raise ValueError("GRPO training requires at least one reward function")
    return functions, names
