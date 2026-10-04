# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Sample answers for the Current Run tab: one prompt's group of completions and what each reward gave them.

TRL calls every reward function on the same batch in order, so the wrappers collect each function's
scores and the last one in the batch hands the finished group to the sink.
"""

from __future__ import annotations

import math
import time
from typing import Any, Callable, Optional

MIN_INTERVAL_SECONDS = 15.0
MAX_PROMPT_CHARS = 2_000
MAX_COMPLETION_CHARS = 6_000
MAX_ANSWER_CHARS = 500


def _text(value: Any, limit: int, *, last: bool = False) -> str:
    if isinstance(value, list):
        turns = [m for m in value if isinstance(m, dict)]
        if last:
            users = [m for m in turns if m.get("role") == "user"] or turns
            value = users[-1].get("content", "") if users else ""
        else:
            value = "".join(str(m.get("content", "")) for m in turns)
    text = "" if value is None else str(value)
    if len(text) <= limit:
        return text
    return ("…" + text[-limit:]) if last else (text[:limit] + "…")


def _score(value: Any) -> Optional[float]:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


class SampleRecorder:
    def __init__(
        self,
        names: list[str],
        weights: list[float],
        sink: Callable[[dict], None],
        group_size: int,
        min_interval: float = MIN_INTERVAL_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.names = names
        self.weights = weights
        self.sink = sink
        self.group_size = max(1, group_size)
        self.min_interval = min_interval
        self.clock = clock
        self._scores: dict[str, list] = {}
        self._last_emit: Optional[float] = None

    def wrap(self, index: int, func: Callable) -> Callable:
        name = self.names[index]
        is_last = index == len(self.names) - 1

        def recorded(prompts = None, completions = None, **kwargs):
            scores = func(prompts = prompts, completions = completions, **kwargs)
            try:
                self._scores[name] = list(scores or [])
                if is_last:
                    self._emit(prompts, completions, kwargs)
            except Exception:  # noqa: BLE001 - a display aid must never stop training
                self._scores.clear()
            return scores

        recorded.__name__ = getattr(func, "__name__", name)
        return recorded

    def _emit(self, prompts, completions, kwargs) -> None:
        scores, self._scores = self._scores, {}
        now = self.clock()
        if self._last_emit is not None and now - self._last_emit < self.min_interval:
            return
        completions = list(completions or [])
        if not completions:
            return
        self._last_emit = now
        n = min(self.group_size, len(completions))
        answers = kwargs.get("answer")
        state = kwargs.get("trainer_state")
        items = []
        for i in range(n):
            rewards = {
                name: _score(scores[name][i]) if i < len(scores.get(name, ())) else None
                for name in self.names
            }
            total = sum(
                (rewards[name] or 0.0) * weight for name, weight in zip(self.names, self.weights)
            )
            items.append(
                {
                    "completion": _text(completions[i], MAX_COMPLETION_CHARS),
                    "rewards": rewards,
                    "total": total,
                }
            )
        self.sink(
            {
                "step": getattr(state, "global_step", None),
                "prompt": _text((prompts or [None])[0], MAX_PROMPT_CHARS, last = True),
                "answer": (
                    _text(answers[0], MAX_ANSWER_CHARS)
                    if isinstance(answers, (list, tuple)) and answers
                    else None
                ),
                "items": items,
            }
        )


def record_samples(
    funcs: list[Callable],
    reward_specs: list[dict],
    sink: Optional[Callable[[dict], None]],
    group_size: int,
) -> list[Callable]:
    if sink is None or not funcs:
        return funcs
    recorder = SampleRecorder(
        [s["name"] for s in reward_specs],
        [float(s.get("weight", 1.0)) for s in reward_specs],
        sink,
        group_size,
    )
    return [recorder.wrap(i, f) for i, f in enumerate(funcs)]
