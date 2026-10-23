# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Data transforms: edit fields, map, filter, sort, dedupe and merge, all through the Flow expression language
(a JSON tree, never source text), so a shared flow reshapes data without running code."""

from __future__ import annotations

import json
import random
from typing import Any

from core.flows.expr import ExprError, eval_expr, truthy
from core.flows.registry import FlowNodeError, NodeContext, handler

# A list transform walks at most this many items; each expression is itself step-capped by the evaluator.
MAX_ITEMS = 10_000


def _items(ctx: NodeContext, port: str = "items") -> list:
    value = ctx.inputs.get(port)
    if value is None:
        return []
    if not isinstance(value, list):
        raise FlowNodeError(f"{port} needs a list")
    if len(value) > MAX_ITEMS:
        raise FlowNodeError(f"{len(value)} items is more than the {MAX_ITEMS} a transform takes")
    return value


def _eval(
    ctx: NodeContext,
    expr: Any,
    item: Any,
    index: int,
    rng: Any = None,
) -> Any:
    variables = dict(ctx.flow_params or {})
    variables.update({"item": item, "index": index})
    try:
        return eval_expr(expr, variables, rng or random.Random(ctx.seed + index))
    except ExprError as exc:
        raise FlowNodeError(f"Expression error on item {index}: {exc}") from None


def _set_path(target: dict, path: str, value: Any) -> None:
    parts = [p for p in str(path).split(".") if p]
    if not parts:
        raise FlowNodeError("A field name is empty")
    cur = target
    for part in parts[:-1]:
        nxt = cur.get(part)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[part] = nxt
        cur = nxt
    cur[parts[-1]] = value


def _edit(ctx: NodeContext, fields: dict, keep_only: bool, item: Any, index: int) -> dict:
    base: dict = {} if keep_only or not isinstance(item, dict) else json.loads(json.dumps(item))
    if not isinstance(item, dict) and not keep_only and item is not None:
        base["value"] = item
    # One generator per item, so two random fields draw in turn instead of repeating the same first draw.
    rng = random.Random(ctx.seed + index)
    for name, expr in fields.items():
        _set_path(base, name, _eval(ctx, expr, item, index, rng))
    return base


@handler("json.set")
async def edit_fields(ctx: NodeContext) -> dict:
    fields = ctx.params.get("fields")
    if not isinstance(fields, dict) or not fields:
        raise FlowNodeError('Edit Fields takes {"name": expression, ...}')
    keep_only = bool(ctx.params.get("keep_only"))
    value = ctx.inputs.get("value")
    if isinstance(value, list):
        rows = _items(ctx, "value")
        return {"value": [_edit(ctx, fields, keep_only, row, i) for i, row in enumerate(rows)]}
    return {"value": _edit(ctx, fields, keep_only, value, 0)}


@handler("list.map")
async def map_list(ctx: NodeContext) -> dict:
    expr = ctx.params.get("expression")
    return {"items": [_eval(ctx, expr, item, i) for i, item in enumerate(_items(ctx))]}


@handler("list.filter")
async def filter_list(ctx: NodeContext) -> dict:
    expr = ctx.params.get("condition")
    kept, dropped = [], []
    for i, item in enumerate(_items(ctx)):
        (kept if truthy(_eval(ctx, expr, item, i)) else dropped).append(item)
    return {"items": kept, "dropped": dropped}


def _sort_key(value: Any) -> tuple:
    # Numbers, then text, then anything else by its JSON; empty values always sort last.
    if value is None:
        return (3, 0)
    if isinstance(value, bool):
        return (0, int(value))
    if isinstance(value, (int, float)):
        return (0, value)
    if isinstance(value, str):
        return (1, value.lower())
    return (2, json.dumps(value, sort_keys = True, default = str))


@handler("list.sort")
async def sort_list(ctx: NodeContext) -> dict:
    expr = ctx.params.get("key")
    items = _items(ctx)
    keyed = [(_sort_key(_eval(ctx, expr, item, i)), i, item) for i, item in enumerate(items)]
    descending = bool(ctx.params.get("descending"))
    present = [k for k in keyed if k[0][0] != 3]
    empty = [k for k in keyed if k[0][0] == 3]
    # Python's sort stays stable with reverse, so equal keys keep their order either way.
    present.sort(key = lambda k: k[0], reverse = descending)
    out = [k[2] for k in present + empty]
    limit = ctx.params.get("limit")
    if limit not in (None, ""):
        out = out[: max(0, int(limit))]
    return {"items": out}


@handler("list.dedupe")
async def dedupe_list(ctx: NodeContext) -> dict:
    expr = ctx.params.get("key")
    items = _items(ctx)
    seen: set[str] = set()
    kept = []
    for i, item in enumerate(items):
        key = json.dumps(_eval(ctx, expr, item, i), sort_keys = True, default = str)
        if key in seen:
            continue
        seen.add(key)
        kept.append(item)
    return {"items": kept, "removed": len(items) - len(kept)}


def _as_list(value: Any) -> list:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


@handler("json.merge")
async def merge_data(ctx: NodeContext) -> dict:
    a, b = ctx.inputs.get("a"), ctx.inputs.get("b")
    mode = str(ctx.params.get("mode") or "append")
    if mode == "append":
        merged = _as_list(a) + _as_list(b)
        if len(merged) > MAX_ITEMS:
            raise FlowNodeError(
                f"{len(merged)} items is more than the {MAX_ITEMS} a transform takes"
            )
        return {"value": merged}
    if mode == "combine":
        if isinstance(a, dict) and isinstance(b, dict):
            return {"value": {**a, **b}}
        if isinstance(a, list) and isinstance(b, list):
            if max(len(a), len(b)) > MAX_ITEMS:
                raise FlowNodeError(f"Each side takes at most {MAX_ITEMS} items")
            # Item by item, like n8n's "combine by position"; the longer list keeps its extra items.
            pairs = []
            for i in range(max(len(a), len(b))):
                left = a[i] if i < len(a) else None
                right = b[i] if i < len(b) else None
                if isinstance(left, dict) and isinstance(right, dict):
                    pairs.append({**left, **right})
                else:
                    pairs.append(right if right is not None else left)
            return {"value": pairs}
        raise FlowNodeError("Combine takes two objects or two lists")
    if mode == "by_key":
        key = str(ctx.params.get("key") or "").strip()
        if not key:
            raise FlowNodeError("Name the key field to join on")
        left, right = _as_list(a), _as_list(b)
        if len(left) > MAX_ITEMS or len(right) > MAX_ITEMS:
            raise FlowNodeError(f"Each side takes at most {MAX_ITEMS} items")
        keep = ctx.params.get("keep_unmatched", True)
        # Every right row per key: an order joined with its line items keeps all of them.
        index: dict[str, list[dict]] = {}
        keyless: list[dict] = []
        for row in right:
            if isinstance(row, dict) and key in row:
                index.setdefault(json.dumps(row[key], sort_keys = True, default = str), []).append(row)
            elif isinstance(row, dict):
                keyless.append(row)
        used: set[str] = set()
        out = []
        for row in left:
            if not isinstance(row, dict):
                continue
            k = json.dumps(row.get(key), sort_keys = True, default = str)
            matches = index.get(k) if key in row else None
            if matches:
                used.add(k)
                out.extend({**row, **match} for match in matches)
            elif keep:
                out.append(row)
            if len(out) > MAX_ITEMS:
                raise FlowNodeError(f"The join made more than {MAX_ITEMS} items")
        if keep:
            for k, rows in index.items():
                if k not in used:
                    out.extend(rows)
            out.extend(keyless)
        if len(out) > MAX_ITEMS:
            raise FlowNodeError(f"The join made more than {MAX_ITEMS} items")
        return {"value": out}
    raise FlowNodeError(f"Unknown merge mode {mode!r}")
