# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from .metrics.base import MetricResult, Scorer


@dataclass
class EvalSummary:
    status: str           # "completed" | "cancelled"
    num_scored: int
    avg_score: float


def run_eval(
    *,
    examples: list[tuple[str, Any]],          # (input_text, reference)
    generate: Callable[..., str],             # (messages, system_prompt, **gen) -> text
    scorer: Scorer,
    system_prompt: str,
    template: str | None,
    gen_params: dict,
    should_cancel: Callable[[], bool],
    on_result: Callable[[int, MetricResult, str, str, Any], None],
) -> EvalSummary:
    """Pure eval loop. Model/dataset are injected so this is unit-testable.

    For each example: render the prompt, generate, score, and report via
    on_result(idx, result, prediction, input_text, reference). A generation or
    scoring error becomes a score-0 result with an error note (never aborts).
    Cancellation is checked before each example and takes effect immediately.
    """
    total = 0.0
    scored = 0
    status = "completed"
    for idx, (input_text, reference) in enumerate(examples):
        if should_cancel():
            status = "cancelled"
            break
        try:
            # Substitute only the {input} placeholder. Use str.replace (not
            # str.format) so literal braces in the template or data — e.g. JSON
            # examples or {html} — don't blow up with KeyError/ValueError.
            content = (
                template.replace("{input}", str(input_text))
                if template
                else str(input_text)
            )
            messages = [{"role": "user", "content": content}]
            prediction = generate(messages, system_prompt, **gen_params)
            result = scorer(prediction, reference)
        except Exception as exc:  # generation/scoring failure -> errored example
            prediction = ""
            result = MetricResult(score=0.0, error=f"{type(exc).__name__}: {exc}")
        total += result.score
        scored += 1
        on_result(idx, result, prediction, input_text, reference)
    avg = (total / scored) if scored else 0.0
    return EvalSummary(status=status, num_scored=scored, avg_score=avg)
