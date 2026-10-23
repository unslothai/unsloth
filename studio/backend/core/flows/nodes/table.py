# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Table and chart nodes: join, summarise, compare and plot result tables (pure CPU, cacheable).

A table value is ``{"kind": "table", "columns": [{"id", "label", "type", "unit"?, "role"?}], "rows": [{...}],
"meta"?: {...}}``. One over ``INLINE_MAX_BYTES`` keeps its first ``INLINE_ROWS`` rows inline and the whole table
in a JSON asset (``ref``), with ``row_count`` set; readers load the asset when they need every row.
"""

from __future__ import annotations

import json
import math
import statistics
from typing import Any, Optional

from core.flows.nodes._common import media_bytes
from core.flows.registry import FlowNodeError, NodeContext, handler

INLINE_MAX_BYTES = 256 * 1024
INLINE_ROWS = 200
MAX_ROWS = 100_000
MAX_SERIES = 50
STATS = ("mean", "median", "stddev", "min", "max", "sum", "count")


def _number(value: Any) -> Optional[float]:
    if isinstance(value, bool) or value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _column_type(values: list) -> str:
    seen = [v for v in values if v is not None]
    if seen and all(_number(v) is not None for v in seen):
        return "number"
    if seen and all(isinstance(v, bool) for v in seen):
        return "bool"
    return "text"


def make_table(
    rows: list[dict],
    columns: Optional[list[dict]] = None,
    meta: Optional[dict] = None,
) -> dict:
    """A table value from row dicts; columns are inferred (first-seen order) when not given."""
    if columns is None:
        ids: list[str] = []
        for row in rows:
            for key in row:
                if key not in ids:
                    ids.append(key)
        columns = [
            {"id": c, "label": c.replace("_", " "), "type": _column_type([r.get(c) for r in rows])}
            for c in ids
        ]
    table: dict[str, Any] = {"kind": "table", "columns": columns, "rows": rows}
    if meta:
        table["meta"] = meta
    return table


async def read_table(ctx: Any, value: Any) -> dict:
    """Normalise an input into a full table: a table value (loading its asset if truncated) or a list of dicts."""
    if (
        isinstance(value, list)
        and len(value) == 1
        and isinstance(value[0], dict)
        and value[0].get("kind") == "table"
    ):
        value = value[0]
    if isinstance(value, list):
        if not all(isinstance(r, dict) for r in value):
            raise FlowNodeError("Expected a table or a list of objects")
        return make_table(list(value))
    if not isinstance(value, dict) or value.get("kind") != "table":
        raise FlowNodeError("Expected a table")
    table = dict(value)
    if isinstance(table.get("ref"), str) and int(table.get("row_count") or 0) > len(
        table.get("rows") or []
    ):
        data, _ = await media_bytes(ctx, {"ref": table["ref"]})
        try:
            full = json.loads(data)
        except ValueError:
            raise FlowNodeError("This table's stored rows are unreadable") from None
        table["rows"] = full.get("rows") or []
    columns = table.get("columns") or []
    rows = table.get("rows") or []
    if rows and isinstance(rows[0], list):
        ids = [c.get("id") for c in columns]
        rows = [dict(zip(ids, r)) for r in rows]
    if not columns:
        return make_table(list(rows), meta = table.get("meta"))
    table["columns"], table["rows"] = list(columns), list(rows)
    table.pop("ref", None)
    table.pop("row_count", None)
    return table


async def finish_table(ctx: Any, table: dict) -> dict:
    """Keep a table small enough for run events: big ones move to a JSON asset with a preview inline."""
    rows = table.get("rows") or []
    if len(rows) > MAX_ROWS:
        raise FlowNodeError(f"Tables hold at most {MAX_ROWS} rows")
    data = json.dumps(table, default = str).encode()
    if len(data) <= INLINE_MAX_BYTES:
        return table
    media = await ctx.save_asset(data, "application/json", kind = "json")
    preview = rows[:INLINE_ROWS]
    # Wide cells can make even a short preview big; keep halving it until the inline value fits.
    while (
        preview
        and len(json.dumps({**table, "rows": preview}, default = str).encode()) > INLINE_MAX_BYTES
    ):
        preview = preview[: len(preview) // 2]
    return {**table, "rows": preview, "row_count": len(rows), "ref": media["ref"]}


def _column_ids(table: dict) -> list[str]:
    return [c["id"] for c in table["columns"]]


def _require(table: dict, names: list[str], what: str) -> None:
    missing = [n for n in names if n not in _column_ids(table)]
    if missing:
        raise FlowNodeError(f"{what}: no column {', '.join(repr(m) for m in missing)}")


def _stat(values: list[float], stat: str) -> Optional[float]:
    if stat == "count":
        return float(len(values))
    if not values:
        return None
    if stat == "mean":
        return statistics.fmean(values)
    if stat == "median":
        return statistics.median(values)
    if stat == "stddev":
        return statistics.stdev(values) if len(values) > 1 else 0.0
    if stat == "min":
        return min(values)
    if stat == "max":
        return max(values)
    return math.fsum(values)


def _fresh(name: str, taken: set[str]) -> str:
    """``name``, or ``name_2``... when a source column already uses it, so a generated column never replaces one."""
    out, i = name, 2
    while out in taken:
        out, i = f"{name}_{i}", i + 1
    taken.add(out)
    return out


def _strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [s.strip() for s in value.split(",") if s.strip()]
    return [str(v).strip() for v in (value or []) if str(v).strip()]


def concat(tables: list[dict]) -> dict:
    columns: list[dict] = []
    seen: set[str] = set()
    rows: list[dict] = []
    for table in tables:
        for col in table["columns"]:
            if col["id"] not in seen:
                seen.add(col["id"])
                columns.append(col)
        rows.extend(table["rows"])
    # Settings carry over only when every table has the same ones; an unknown one must not pass as a match.
    metas = [t.get("meta") or None for t in tables]
    meta = metas[0] if all(m == metas[0] for m in metas) else None
    return make_table(rows, columns, meta)


def aggregate(table: dict, group_by: list[str], metrics: list[str], stat: str) -> dict:
    _require(table, group_by + metrics, "Summarise")
    if not metrics:
        metrics = [
            c["id"]
            for c in table["columns"]
            if c.get("type") == "number" and c["id"] not in group_by
        ]
    groups: dict[tuple, list[dict]] = {}
    for row in table["rows"]:
        groups.setdefault(tuple(row.get(g) for g in group_by), []).append(row)
    by_id = {c["id"]: c for c in table["columns"]}
    count_id = _fresh("n", set(group_by) | set(metrics))
    rows = []
    for key, members in groups.items():
        out = dict(zip(group_by, key))
        for m in metrics:
            values = [v for v in (_number(r.get(m)) for r in members) if v is not None]
            out[m] = _stat(values, stat)
        out[count_id] = len(members)
        rows.append(out)
    columns = [by_id[g] for g in group_by]
    columns += [
        {**by_id[m], "type": "number", "label": f"{by_id[m].get('label', m)} ({stat})"}
        for m in metrics
    ]
    columns.append({"id": count_id, "label": "rows", "type": "number"})
    return make_table(rows, columns, table.get("meta"))


def _settings_mismatch(a: dict, b: dict) -> list[str]:
    sa = (a.get("meta") or {}).get("settings") or {}
    sb = (b.get("meta") or {}).get("settings") or {}
    return sorted(k for k in set(sa) | set(sb) if sa.get(k) != sb.get(k))


def compare(
    table: dict,
    baseline: Optional[dict],
    key: list[str],
    metric: str,
    *,
    baseline_key: Any = None,
    stderr: Optional[str] = None,
    allow_mismatch: bool = False,
) -> dict:
    """Delta and ratio of ``metric`` against a baseline table (joined on ``key``) or a baseline row in ``table``."""
    if not key:
        raise FlowNodeError("Compare needs at least one key column")
    extra = [stderr] if stderr else []
    _require(table, key + [metric] + extra, "Compare")
    if baseline is not None:
        _require(baseline, key + [metric] + extra, "Compare baseline")
        units = [
            {c["id"]: c.get("unit") for c in t["columns"]}.get(metric) for t in (table, baseline)
        ]
        if all(units) and units[0] != units[1]:
            raise FlowNodeError(
                f"{metric!r} is in {units[0]} here but {units[1]} in the baseline; convert one before comparing"
            )
        differ = _settings_mismatch(table, baseline)
        if differ and not allow_mismatch:
            raise FlowNodeError(
                "The two tables were measured with different settings "
                f"({', '.join(differ)}); turn on 'Allow different settings' to compare anyway"
            )
        # 4096 in one table and "4096" in the other are the same key when the column is numeric.
        numeric = {
            c["id"] for t in (table, baseline) for c in t["columns"] if c.get("type") == "number"
        }

        def key_of(row: dict) -> tuple:
            return tuple(
                _number(row.get(c))
                if c in numeric and _number(row.get(c)) is not None
                else row.get(c)
                for c in key
            )

        base_rows: dict[tuple, dict] = {}
        for r in baseline["rows"]:
            k = key_of(r)
            if k in base_rows:
                raise FlowNodeError(
                    f"The baseline has more than one row for {', '.join(key)} = {k!r}; "
                    "add key columns until each row is unique"
                )
            base_rows[k] = r

        def base_of(row: dict) -> Optional[dict]:
            return base_rows.get(key_of(row))

    else:
        if baseline_key in (None, ""):
            raise FlowNodeError("Connect a baseline table or name the baseline row")
        wanted = [s.strip() for s in str(baseline_key).split(",")]
        match = [r for r in table["rows"] if [str(r.get(k)) for k in key] == wanted]
        if not match:
            raise FlowNodeError(f"No row has {', '.join(key)} = {baseline_key!r}")
        if len(match) > 1:
            raise FlowNodeError(
                f"{len(match)} rows have {', '.join(key)} = {baseline_key!r}; "
                "add key columns until the baseline row is unique"
            )
        anchor = match[0]

        def base_of(row: dict) -> Optional[dict]:
            return anchor

    taken = set(key) | {metric}
    ids = {n: _fresh(n, taken) for n in ("baseline", "delta", "ratio", "significant")}
    rows = []
    significant_known = False
    for row in table["rows"]:
        base = base_of(row)
        value = _number(row.get(metric))
        ref = _number(base.get(metric)) if base else None
        delta = value - ref if value is not None and ref is not None else None
        ratio = value / ref if value is not None and ref not in (None, 0.0) else None
        out = {k: row.get(k) for k in key}
        out.update({metric: value, ids["baseline"]: ref, ids["delta"]: delta, ids["ratio"]: ratio})
        if stderr:
            se_a = _number(row.get(stderr))
            se_b = _number(base.get(stderr)) if base else None
            if delta is not None and se_a is not None and se_b is not None:
                significant_known = True
                out[ids["significant"]] = abs(delta) > 2 * math.sqrt(se_a**2 + se_b**2)
            else:
                out[ids["significant"]] = None
        rows.append(out)
    by_id = {c["id"]: c for c in table["columns"]}
    unit = by_id[metric].get("unit")
    columns = [by_id[k] for k in key]
    columns += [
        {**by_id[metric], "type": "number"},
        {
            "id": ids["baseline"],
            "label": "baseline",
            "type": "number",
            **({"unit": unit} if unit else {}),
        },
        {
            "id": ids["delta"],
            "label": "delta",
            "type": "number",
            "role": "delta",
            **({"unit": unit} if unit else {}),
        },
        {"id": ids["ratio"], "label": "ratio", "type": "number"},
    ]
    if stderr and significant_known:
        columns.append({"id": ids["significant"], "label": "significant", "type": "bool"})
    elif stderr:
        for r in rows:
            r.pop(ids["significant"], None)
    return make_table(rows, columns, table.get("meta"))


def chart(
    table: dict, x: str, y: list[str], *, kind: str, group: Optional[str], error: Optional[str]
) -> dict:
    if not y:
        y = [
            c["id"]
            for c in table["columns"]
            if c.get("type") == "number" and c["id"] not in (x, group, error)
        ][:1]
    if not y:
        raise FlowNodeError("Chart needs a number column for y")
    _require(table, [x] + y + ([group] if group else []) + ([error] if error else []), "Chart")
    if group and len(y) > 1:
        raise FlowNodeError("Pick one y column when grouping into series")
    by_id = {c["id"]: c for c in table["columns"]}
    numeric_x = by_id[x].get("type") == "number"

    def x_of(row: dict) -> Any:
        # Numbers stored as text ("1024") are the same coordinate as 1024 and plot on a numeric axis.
        v = row.get(x)
        return _number(v) if numeric_x and _number(v) is not None else v

    # Insertion order without a list scan per row (a 100k-row table would be quadratic).
    xs: list[Any] = list(dict.fromkeys(x_of(row) for row in table["rows"]))
    series = []
    names = (
        [(str(g), g) for g in dict.fromkeys(r.get(group) for r in table["rows"])]
        if group
        else [(by_id[m].get("label", m), None) for m in y]
    )
    if len(names) > MAX_SERIES:
        raise FlowNodeError(
            f"{group!r} has {len(names)} distinct values; a chart shows at most {MAX_SERIES} series"
        )
    for i, (name, gval) in enumerate(names):
        column = y[0] if group else y[i]
        points: dict[Any, dict] = {}
        for row in table["rows"]:
            if group and row.get(group) != gval:
                continue
            if x_of(row) in points:
                where = f" in series {name!r}" if group else ""
                raise FlowNodeError(
                    f"More than one row has {x} = {x_of(row)!r}{where}; "
                    "average repeats with Aggregate first, or add a series column"
                )
            points[x_of(row)] = row
        values = [_number(points[v].get(column)) if v in points else None for v in xs]
        entry: dict[str, Any] = {"name": name, "y": values}
        if error:
            entry["error"] = [_number(points[v].get(error)) if v in points else None for v in xs]
        series.append(entry)
    # One axis unit only when every plotted column shares it; mixed units are not labelled as one.
    units = {by_id[m].get("unit") for m in y}
    unit = units.pop() if len(units) == 1 else None
    return {
        "kind": "chart",
        "type": kind,
        "x": xs,
        "x_label": by_id[x].get("label", x),
        "y_label": by_id[y[0]].get("label", y[0]) if len(y) == 1 or group else None,
        **({"unit": unit} if unit else {}),
        "series": series,
    }


@handler("table.concat")
async def table_concat(ctx: NodeContext) -> dict:
    values = ctx.inputs.get("tables") or []
    if not isinstance(values, list):
        values = [values]
    tables = [await read_table(ctx, v) for v in values if v is not None]
    if not tables:
        raise FlowNodeError("Connect at least one table")
    return {"table": await finish_table(ctx, concat(tables))}


@handler("table.aggregate")
async def table_aggregate(ctx: NodeContext) -> dict:
    table = await read_table(ctx, ctx.inputs.get("table"))
    stat = str(ctx.params.get("stat") or "mean")
    if stat not in STATS:
        raise FlowNodeError(f"Unknown summary {stat!r}")
    out = aggregate(
        table, _strings(ctx.params.get("group_by")), _strings(ctx.params.get("metrics")), stat
    )
    return {"table": await finish_table(ctx, out)}


@handler("table.compare")
async def table_compare(ctx: NodeContext) -> dict:
    table = await read_table(ctx, ctx.inputs.get("table"))
    raw_base = ctx.inputs.get("baseline")
    baseline = await read_table(ctx, raw_base) if raw_base is not None else None
    metric = str(ctx.params.get("metric") or "").strip()
    if not metric:
        raise FlowNodeError("Pick the metric column to compare")
    out = compare(
        table,
        baseline,
        _strings(ctx.params.get("key")),
        metric,
        baseline_key = ctx.params.get("baseline_row"),
        stderr = str(ctx.params.get("stderr") or "").strip() or None,
        allow_mismatch = bool(ctx.params.get("allow_mismatch")),
    )
    # A metric or key named "delta" pushes the generated column to "delta_2": read it by role.
    col = next((c["id"] for c in out["columns"] if c.get("role") == "delta"), "delta")
    deltas = [r[col] for r in out["rows"] if _number(r.get(col)) is not None]
    return {
        "table": await finish_table(ctx, out),
        "delta": statistics.fmean(deltas) if deltas else None,
    }


@handler("table.chart")
async def table_chart(ctx: NodeContext) -> dict:
    table = await read_table(ctx, ctx.inputs.get("table"))
    x = str(ctx.params.get("x") or "").strip() or (
        table["columns"][0]["id"] if table["columns"] else ""
    )
    kind = str(ctx.params.get("type") or "bar")
    if kind not in ("bar", "line"):
        raise FlowNodeError(f"Unknown chart type {kind!r}")
    value = chart(
        table,
        x,
        _strings(ctx.params.get("y")),
        kind = kind,
        group = str(ctx.params.get("group") or "").strip() or None,
        error = str(ctx.params.get("error") or "").strip() or None,
    )
    return {"chart": value}
