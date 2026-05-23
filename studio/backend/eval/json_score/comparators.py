# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import math
import re
from typing import Any, Callable

Comparator = Callable[[Any, Any], float]  # returns a score in [0, 1]

# Unsigned magnitude with optional scientific-notation exponent; a leading sign
# is peeled off separately so currency symbols between the sign and the digits
# (e.g. "-$1,234.56") don't strand it.
_NUM_RE = re.compile(r"\d*\.?\d+(?:[eE][-+]?\d+)?")


def _to_number(x: Any) -> float | None:
    """Coerce a value to a finite float, or None if it isn't numeric.

    Accepts ints/floats and number-bearing strings, tolerating thousands
    separators, currency symbols, leading signs, and scientific notation.
    Booleans and non-finite floats (NaN/inf) are rejected.
    """
    if isinstance(x, bool):
        return None  # bools are not monetary/numeric values
    if isinstance(x, (int, float)):
        return float(x) if math.isfinite(x) else None
    if isinstance(x, str):
        s = x.replace(",", "").strip()
        # Direct parse first — handles signs and scientific notation cleanly.
        try:
            v = float(s)
            return v if math.isfinite(v) else None
        except ValueError:
            pass
        # Fallback: peel a leading sign, then extract the first number even if a
        # currency symbol or other prefix sits in front of it.
        sign = ""
        if s[:1] in "+-":
            sign, s = s[0], s[1:].lstrip()
        m = _NUM_RE.search(s)
        if m:
            try:
                v = float(sign + m.group())
                return v if math.isfinite(v) else None
            except ValueError:
                return None
    return None


def money_comparator(rel_tol: float = 0.0, abs_tol: float = 0.0) -> Comparator:
    """Score numeric closeness: ``max(0, 1 - |a-b| / max(|a|,|b|))``.

    Both zero scores 1.0; within ``abs_tol`` or ``rel_tol`` scores 1.0;
    uncoercible input scores 0.0.
    """
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
