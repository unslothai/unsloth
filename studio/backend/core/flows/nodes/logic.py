# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Logic nodes: seeded randomness, dice, branching and joins. For Each is run by the engine itself."""

from __future__ import annotations

import itertools
import random
import re
from typing import Any

from core.flows.expr import ExprError, eval_expr
from core.flows.nodes._common import as_text
from core.flows.registry import FlowNodeError, NodeContext, handler
from core.flows.schema import MAX_FOREACH_ITEMS

_DICE_TERM = re.compile(r"([+-]?)\s*(?:(\d*)d(\d+)|(\d+))", re.I)


def roll(spec: str, rng: random.Random) -> tuple[int, list[int]]:
    text = (spec or "").replace(" ", "")
    if not text or not re.fullmatch(r"([+-]?(\d*d\d+|\d+))+", text, re.I):
        raise FlowNodeError(f"{spec!r} is not a dice roll like 2d6+1")
    total, rolls = 0, []
    for sign, count, sides, flat in _DICE_TERM.findall(text):
        factor = -1 if sign == "-" else 1
        if flat:
            total += factor * int(flat)
            continue
        n, s = int(count or 1), int(sides)
        if not (1 <= n <= 100 and 1 <= s <= 1000):
            raise FlowNodeError("Dice are at most 100 dice of up to 1000 sides")
        for _ in range(n):
            value = rng.randint(1, s)
            rolls.append(value)
            total += factor * value
    return total, rolls


def _eval(expr: Any, variables: dict, rng: random.Random) -> Any:
    try:
        return eval_expr(expr, variables, rng)
    except ExprError as exc:
        raise FlowNodeError(f"Expression error: {exc}") from None


@handler("logic.random_pick")
async def random_pick(ctx: NodeContext) -> dict:
    items = list(ctx.inputs.get("items") or [])
    if not items:
        raise FlowNodeError("The list is empty")
    rng = random.Random(ctx.seed)
    count = max(1, int(ctx.params.get("count") or 1))
    if ctx.params.get("unique", True):
        picked = rng.sample(items, min(count, len(items)))
    else:
        picked = [rng.choice(items) for _ in range(count)]
    return {"item": picked[0], "items": picked}


@handler("logic.random_number")
async def random_number(ctx: NodeContext) -> dict:
    rng = random.Random(ctx.seed)
    low, high = float(ctx.params.get("min") or 0), float(ctx.params.get("max") or 0)
    if high < low:
        low, high = high, low
    if ctx.params.get("integer", True):
        return {"value": rng.randint(int(low), int(high))}
    return {"value": rng.uniform(low, high)}


@handler("logic.dice")
async def dice(ctx: NodeContext) -> dict:
    total, rolls = roll(str(ctx.params.get("dice") or "1d20"), random.Random(ctx.seed))
    return {"total": total, "rolls": rolls}


@handler("logic.if")
async def if_(ctx: NodeContext) -> dict:
    value = ctx.inputs.get("value")
    variables = dict(ctx.flow_params or {})
    if isinstance(ctx.inputs.get("vars"), dict):
        variables.update(ctx.inputs["vars"])
    variables["value"] = value
    if _eval(ctx.params.get("condition"), variables, random.Random(ctx.seed)):
        return {"then": value}
    return {"else": value}


@handler("logic.switch")
async def switch(ctx: NodeContext) -> dict:
    key = as_text(ctx.inputs.get("key")).strip().lower()
    cases = [str(c).strip().lower() for c in ctx.params.get("cases") or []][:3]
    for i, case in enumerate(cases, 1):
        if key == case:
            return {f"case_{i}": ctx.inputs.get("value")}
    return {"default": ctx.inputs.get("value")}


@handler("logic.compare")
async def compare(ctx: NodeContext) -> dict:
    variables = {"a": ctx.inputs.get("a"), "b": ctx.inputs.get("b")}
    return {"result": _eval(ctx.params.get("expression"), variables, random.Random(ctx.seed))}


@handler("logic.merge")
async def merge(ctx: NodeContext) -> dict:
    for port in ("a", "b", "c"):
        if port in ctx.inputs and ctx.inputs[port] is not None:
            return {"value": ctx.inputs[port]}
    return {}


@handler("logic.collect")
async def collect(ctx: NodeContext) -> dict:
    return {"list": list(ctx.inputs.get("items") or [])}


@handler("logic.product")
async def product(ctx: NodeContext) -> dict:
    axes = ctx.inputs.get("axes")
    if axes is None:
        axes = ctx.params.get("axes")
    if not isinstance(axes, dict) or not axes:
        raise FlowNodeError('Combinations takes named lists, like {"x": [1, 2], "y": ["a", "b"]}')
    names, values = [], []
    for name, items in axes.items():
        names.append(str(name))
        values.append(items if isinstance(items, list) else [items])
    count = 1
    for items in values:
        count *= len(items)
    limit = min(int(ctx.params.get("max_items") or 64), MAX_FOREACH_ITEMS)
    if count > limit:
        # Refused rather than cut, so a sweep never measures part of its grid without saying so.
        raise FlowNodeError(
            f"{count} combinations is more than the {limit} allowed; shorten an axis"
        )
    combos = [dict(zip(names, combo)) for combo in itertools.product(*values)] if count else []
    return {"items": combos, "count": len(combos)}


@handler("logic.for_each")
async def for_each(ctx: NodeContext) -> dict:
    raise FlowNodeError("For Each is run by the engine")


@handler("flow.call")
async def call_flow(ctx: NodeContext) -> dict:
    flow_id = str(ctx.params.get("flow_id") or "").strip()
    if not flow_id:
        raise FlowNodeError("Pick the flow this step runs")
    values = ctx.inputs.get("values")
    if values is None:
        values = ctx.params.get("values") or {}
    if not isinstance(values, dict):
        raise FlowNodeError('Values for the called flow are an object, like {"topic": "otters"}')
    return {"outputs": await ctx.run_flow(flow_id, values)}
