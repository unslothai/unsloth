# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import re
from typing import Any, Callable

Comparator = Callable[[Any, Any], float]  # returns a score in [0, 1]

_NUM_RE = re.compile(r"[-+]?\d*\.?\d+")


def _to_number(x: Any) -> float | None:
    if isinstance(x, bool):
        return None  # bools are not monetary/numeric values
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, str):
        m = _NUM_RE.search(x.replace(",", "").strip())
        if m:
            try:
                return float(m.group())
            except ValueError:
                return None
    return None


def money_comparator(rel_tol: float = 0.0, abs_tol: float = 0.0) -> Comparator:
    def score(gt: Any, pred: Any) -> float:
        a, b = _to_number(gt), _to_number(pred)
        if a is None or b is None:
            return 0.0
        diff = abs(a - b)
        if diff <= abs_tol:
            return 1.0
        denom = max(abs(a), abs(b))
        if denom == 0.0:
            return 1.0  # both ~zero
        if rel_tol and diff / denom <= rel_tol:
            return 1.0
        return max(0.0, 1.0 - diff / denom)

    return score


_REGISTRY: dict[str, Callable[..., Comparator]] = {
    "money": money_comparator,
    "numeric": money_comparator,  # semantic alias
}


def is_comparator(name: str) -> bool:
    return name in _REGISTRY


def get_comparator(name: str, **params: Any) -> Comparator:
    if name not in _REGISTRY:
        raise ValueError(
            f"Unknown comparator {name!r}. Known: {sorted(_REGISTRY)}"
        )
    try:
        return _REGISTRY[name](**params)
    except TypeError as e:
        raise ValueError(f"Invalid params for comparator {name!r}: {e}") from e
