# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The Flow expression language: a JSON tree, never source text.

Same semantics as `studio/frontend/src/features/flows/expr.ts`; both sides run the shared cases in
`tests/fixtures/flows_expr_cases.json`. Randomness comes from `Prng` (mulberry32), implemented identically
in both languages so a seeded story or dice roll plays out the same in the player and in the engine."""

from __future__ import annotations

import json
import math
import re
from typing import Any, Optional

from core.flows.schema import EXPR_OPS

MAX_STEPS = 10_000
MAX_DEPTH = 64
MAX_DICE = 100
MAX_SIDES = 1000

_DICE_RE = re.compile(r"^\s*(\d*)\s*d\s*(\d+)\s*(?:([+-])\s*(\d+))?\s*$", re.IGNORECASE)
_MASK = 0xFFFFFFFF


class ExprError(ValueError):
    pass


def _imul(a: int, b: int) -> int:
    return (a * b) & _MASK


class Prng:
    """mulberry32. `state` is a uint32 that can be saved and restored."""

    def __init__(self, seed: int = 0):
        self.state = int(seed) & _MASK

    def random(self) -> float:
        self.state = (self.state + 0x6D2B79F5) & _MASK
        a = self.state
        t = _imul(a ^ (a >> 15), 1 | a)
        t = ((t + _imul(t ^ (t >> 7), 61 | t)) & _MASK) ^ t
        return ((t ^ (t >> 14)) & _MASK) / 4294967296.0

    def randint(self, lo: int, hi: int) -> int:
        if hi < lo:
            lo, hi = hi, lo
        return lo + math.floor(self.random() * (hi - lo + 1))


def seed_from(*parts: Any) -> int:
    """FNV-1a 32 over the parts joined by '|': a stable uint32 seed from ids and numbers."""
    h = 0x811C9DC5
    for byte in "|".join(_fmt(p) for p in parts).encode("utf-8"):
        h ^= byte
        h = (h * 0x01000193) & _MASK
    return h


def roll_dice(spec: str, rng: Any) -> tuple[int, list[int]]:
    m = _DICE_RE.match(str(spec or ""))
    if not m:
        raise ExprError(f"bad dice {spec!r}: use a form like 2d6+1")
    count = int(m.group(1) or 1)
    sides = int(m.group(2))
    if not (1 <= count <= MAX_DICE) or not (1 <= sides <= MAX_SIDES):
        raise ExprError(f"dice {spec!r} out of range")
    rolls = [_randint(rng, 1, sides) for _ in range(count)]
    bonus = int(m.group(4) or 0) * (-1 if m.group(3) == "-" else 1)
    return sum(rolls) + bonus, rolls


def _randint(rng: Any, lo: int, hi: int) -> int:
    if hasattr(rng, "randint") and isinstance(rng, Prng):
        return rng.randint(lo, hi)
    if hi < lo:
        lo, hi = hi, lo
    return lo + math.floor(rng.random() * (hi - lo + 1))


def _is_num(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


# Integers past the largest double would be Infinity in the browser player, and unbounded Python ints let a
# single expression spend seconds of CPU on the event loop.
_MAX_DOUBLE = 1.7976931348623157e308


def _norm(x: Any) -> Any:
    if isinstance(x, int) and not isinstance(x, bool) and abs(x) > _MAX_DOUBLE:
        raise ExprError("number out of range")
    if isinstance(x, float):
        if math.isnan(x) or math.isinf(x):
            raise ExprError("number out of range")
        if x.is_integer() and abs(x) < 2**53:
            return int(x)
    return x


def _fmt(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, bool):
        return "true" if x else "false"
    if _is_num(x):
        x = _norm(x)
        return str(x) if isinstance(x, int) else repr(x)
    if isinstance(x, str):
        return x
    return json.dumps(x, separators = (",", ":"), ensure_ascii = False)


def truthy(x: Any) -> bool:
    if x is None or x is False:
        return False
    if _is_num(x):
        return x != 0
    if isinstance(x, (str, list, dict)):
        return len(x) > 0
    return bool(x)


def _eq(a: Any, b: Any) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return isinstance(a, bool) and isinstance(b, bool) and a == b
    if _is_num(a) and _is_num(b):
        return a == b
    if a is None or b is None:
        return a is None and b is None
    if type(a) is not type(b):
        return False
    if isinstance(a, list):
        return len(a) == len(b) and all(_eq(x, y) for x, y in zip(a, b))
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(_eq(a[k], b[k]) for k in a)
    return a == b


def lookup(vars: dict, path: str) -> Any:
    cur: Any = vars
    for part in str(path).split("."):
        if isinstance(cur, dict):
            cur = cur.get(part)
        elif isinstance(cur, list) and re.fullmatch(r"-?\d+", part):
            i = int(part)
            cur = cur[i] if -len(cur) <= i < len(cur) else None
        else:
            return None
        if cur is None:
            return None
    return cur


def _num(x: Any, op: str) -> float:
    if not _is_num(x):
        raise ExprError(f"{op} needs numbers, got {_fmt(x)!r}")
    return x


# Past 10**308 a float scale is infinite (the browser player gives NaN); bounding digits first also keeps
# 10 ** digits from computing a huge integer on the event loop.
_ROUND_DIGITS_MAX = 308


def _round(x: float, digits: int = 0) -> Any:
    if abs(int(digits)) > _ROUND_DIGITS_MAX:
        raise ExprError("round digits out of range")
    f = 10 ** int(digits)
    try:
        return _norm(math.floor(x * f + 0.5) / f)
    except OverflowError:
        raise ExprError("number out of range") from None


class _Evaluator:
    def __init__(self, vars: dict, rng: Any):
        self.vars = vars
        self.rng = rng
        self.steps = 0

    def need_rng(self) -> Any:
        if self.rng is None:
            self.rng = Prng(0)
        return self.rng

    def ev(
        self,
        e: Any,
        depth: int = 0,
    ) -> Any:
        self.steps += 1
        if self.steps > MAX_STEPS:
            raise ExprError("expression too long")
        if depth > MAX_DEPTH:
            raise ExprError("expression too deep")
        if e is None or isinstance(e, (bool, str)):
            return e
        if _is_num(e):
            return _norm(e)
        if isinstance(e, list):
            raise ExprError('use {"list": [...]} for a list')
        if not isinstance(e, dict):
            raise ExprError("bad expression")
        if "var" in e:
            if not isinstance(e["var"], str):
                raise ExprError("var must be a string")
            return lookup(self.vars, e["var"])
        if "list" in e:
            if not isinstance(e["list"], list):
                raise ExprError("list must be an array")
            return [self.ev(x, depth + 1) for x in e["list"]]
        op = e.get("op")
        args = e.get("args", [])
        if op not in EXPR_OPS:
            raise ExprError(f"unknown op {op!r}")
        if not isinstance(args, list):
            raise ExprError("args must be an array")
        return self.apply(op, args, depth + 1)

    def apply(self, op: str, raw: list, depth: int) -> Any:
        ev = lambda x: self.ev(x, depth)  # noqa: E731

        def arity(n: int) -> None:
            if len(raw) != n:
                raise ExprError(f"{op} takes {n} argument{'s' if n != 1 else ''}")

        if op == "and":
            result = True
            for a in raw:
                result = truthy(ev(a))
                if not result:
                    return False
            return result
        if op == "or":
            for a in raw:
                if truthy(ev(a)):
                    return True
            return False
        if op == "if":
            arity(3)
            return ev(raw[1]) if truthy(ev(raw[0])) else ev(raw[2])
        if op == "rand":
            arity(0)
            return self.need_rng().random()
        args = [ev(a) for a in raw]
        if op == "not":
            arity(1)
            return not truthy(args[0])
        if op in ("==", "!="):
            arity(2)
            r = _eq(args[0], args[1])
            return r if op == "==" else not r
        if op in ("<", "<=", ">", ">="):
            arity(2)
            a, b = args
            if not ((_is_num(a) and _is_num(b)) or (isinstance(a, str) and isinstance(b, str))):
                raise ExprError(f"{op} compares two numbers or two strings")
            return {"<": a < b, "<=": a <= b, ">": a > b, ">=": a >= b}[op]
        if op == "+":
            if not args:
                return 0
            if all(isinstance(a, str) for a in args):
                return "".join(args)
            if all(isinstance(a, list) for a in args):
                return [x for a in args for x in a]
            total = 0
            for a in args:
                total += _num(a, op)
            return _norm(total)
        if op == "-":
            if len(args) == 1:
                return _norm(-_num(args[0], op))
            arity(2)
            return _norm(_num(args[0], op) - _num(args[1], op))
        if op == "*":
            total = 1
            for a in args:
                # Bound each step without changing its type, so float chains match the browser player.
                try:
                    total = total * _num(a, op)
                except OverflowError:
                    raise ExprError("number out of range") from None
                if isinstance(total, int) and abs(total) > _MAX_DOUBLE:
                    raise ExprError("number out of range")
                if isinstance(total, float) and (math.isnan(total) or math.isinf(total)):
                    raise ExprError("number out of range")
            return _norm(total)
        if op in ("/", "%"):
            arity(2)
            a, b = _num(args[0], op), _num(args[1], op)
            if b == 0:
                raise ExprError("division by zero")
            return _norm(a / b if op == "/" else a - b * math.floor(a / b))
        if op in ("min", "max"):
            vals = args[0] if len(args) == 1 and isinstance(args[0], list) else args
            if not vals:
                raise ExprError(f"{op} of nothing")
            nums = [_num(v, op) for v in vals]
            return _norm(min(nums) if op == "min" else max(nums))
        if op == "abs":
            arity(1)
            return _norm(abs(_num(args[0], op)))
        if op == "round":
            if len(args) not in (1, 2):
                raise ExprError("round takes 1 or 2 arguments")
            digits = int(_num(args[1], op)) if len(args) == 2 else 0
            return _round(_num(args[0], op), digits)
        if op == "len":
            arity(1)
            a = args[0]
            if a is None:
                return 0
            if isinstance(a, (str, list, dict)):
                return len(a)
            raise ExprError("len needs text, a list or an object")
        if op == "contains":
            arity(2)
            a, b = args
            if isinstance(a, list):
                return any(_eq(x, b) for x in a)
            if isinstance(a, str):
                return isinstance(b, str) and b in a
            if isinstance(a, dict):
                return isinstance(b, str) and b in a
            return False
        if op == "concat":
            return "".join(_fmt(a) for a in args)
        if op in ("lower", "upper"):
            arity(1)
            s = _fmt(args[0])
            return s.lower() if op == "lower" else s.upper()
        if op == "randint":
            arity(2)
            lo, hi = int(_num(args[0], op)), int(_num(args[1], op))
            return _randint(self.need_rng(), lo, hi)
        if op == "dice":
            arity(1)
            return roll_dice(args[0], self.need_rng())[0]
        raise ExprError(f"unknown op {op!r}")


def eval_expr(
    expr: Any,
    vars: Optional[dict] = None,
    rng: Any = None,
) -> Any:
    """Evaluate `expr` against `vars`. `rng` is a `Prng` (or anything with `.random()`); rand ops need one."""
    return _Evaluator(vars or {}, rng).ev(expr)


def apply_sets(
    sets: dict,
    vars: dict,
    rng: Any = None,
) -> dict:
    """Evaluate every `{var: expr}` against the vars as they were before, and return the updated copy."""
    out = dict(vars)
    for name, e in (sets or {}).items():
        out[name] = eval_expr(e, vars, rng)
    return out
