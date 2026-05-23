# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import math
import re
from datetime import date, datetime
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


def categorical_comparator(case_insensitive: bool = False, strip: bool = False) -> Comparator:
    def score(gt: Any, pred: Any) -> float:
        a, b = gt, pred
        if isinstance(a, str) and isinstance(b, str):
            if strip:
                a, b = a.strip(), b.strip()
            if case_insensitive:
                a, b = a.lower(), b.lower()
        return 1.0 if a == b else 0.0

    return score


def string_comparator(threshold: float = 0.5) -> Comparator:
    from rapidfuzz.distance import Levenshtein

    def score(gt: Any, pred: Any) -> float:
        a = "" if gt is None else str(gt)
        b = "" if pred is None else str(pred)
        if not a and not b:
            return 1.0
        sim = Levenshtein.normalized_similarity(a, b)  # 1 - dist/max(len), in [0,1]
        return sim if sim >= threshold else 0.0

    return score


def _to_date(x: Any) -> datetime | None:
    if isinstance(x, datetime):
        return x
    if isinstance(x, date):
        return datetime(x.year, x.month, x.day)
    if isinstance(x, str):
        try:
            from dateutil import parser as _dateparser

            # fixed default so absent components are deterministic
            return _dateparser.parse(x, default=datetime(2000, 1, 1))
        except (ValueError, OverflowError, TypeError):
            return None
    return None


def date_comparator(granularity: str = "day", day_tol: int = 0) -> Comparator:
    def score(gt: Any, pred: Any) -> float:
        da, db = _to_date(gt), _to_date(pred)
        if da is None or db is None:
            return 0.0
        if granularity == "year":
            return 1.0 if da.year == db.year else 0.0
        if granularity == "month":
            return 1.0 if (da.year, da.month) == (db.year, db.month) else 0.0
        delta = abs((da.date() - db.date()).days)
        return 1.0 if delta <= day_tol else 0.0

    return score


_REGISTRY: dict[str, Callable[..., Comparator]] = {
    "money": money_comparator,
    "numeric": money_comparator,  # semantic alias
    "categorical": categorical_comparator,
    "string": string_comparator,
    "date": date_comparator,
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
