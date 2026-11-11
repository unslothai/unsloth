# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import json
import math

import pytest

from core.flows.compiler import port_compatible
from core.flows.nodes import table as T
from core.flows.registry import FlowNodeError, load_handlers
from core.flows.specs import PORT_COLORS, SPECS_BY_TYPE
from ._flows_helpers import run_async


class Ctx:
    def __init__(
        self,
        inputs = None,
        params = None,
    ):
        self.inputs = inputs or {}
        self.params = params or {}
        self.saved: list[bytes] = []

    async def save_asset(
        self,
        data,
        mime,
        *,
        kind,
        name = None,
        extra = None,
    ):
        self.saved.append(data)
        return {"kind": "json", "ref": "sha256:" + "0" * 64, "mime": mime}


def bench_table(settings = None) -> dict:
    return T.make_table(
        [
            {"quant": "Q4_K_M", "ctx": 4096, "tps": 100.0, "se": 1.0},
            {"quant": "Q4_K_M", "ctx": 8192, "tps": 90.0, "se": 1.0},
            {"quant": "Q8_0", "ctx": 4096, "tps": 70.0, "se": 1.0},
            {"quant": "Q8_0", "ctx": 8192, "tps": 60.0, "se": 1.0},
        ],
        meta = {"settings": settings or {"temperature": 0}},
    )


@pytest.mark.parametrize("kind", ["table", "chart", "heatmap", "tensor"])
def test_new_value_kinds_have_colours_and_feed_json_and_any_ports(kind):
    assert kind in PORT_COLORS
    assert port_compatible({"type": kind}, {"type": "json"})
    assert port_compatible({"type": kind}, {"type": "any"})
    assert port_compatible({"type": kind}, {"type": kind})
    assert not port_compatible({"type": kind}, {"type": "image"})
    assert not port_compatible({"type": "text"}, {"type": kind})


def test_table_nodes_are_registered_cpu_and_cacheable():
    handlers = load_handlers()
    for t in ("table.concat", "table.aggregate", "table.compare", "table.chart"):
        assert t in handlers
        assert SPECS_BY_TYPE[t]["lane"] == "cpu" and SPECS_BY_TYPE[t]["cacheable"]


def test_make_table_infers_columns_and_types():
    t = T.make_table([{"a": 1, "b": "x"}, {"a": 2.5, "c": True}])
    assert [c["id"] for c in t["columns"]] == ["a", "b", "c"]
    assert [c["type"] for c in t["columns"]] == ["number", "text", "bool"]


def test_concat_merges_columns_and_keeps_shared_meta_only():
    a = T.make_table([{"x": 1}], meta = {"settings": {"t": 0}})
    b = T.make_table([{"x": 2, "y": 3}], meta = {"settings": {"t": 0}})
    c = T.make_table([{"x": 4}], meta = {"settings": {"t": 1}})
    joined = T.concat([a, b])
    assert [col["id"] for col in joined["columns"]] == ["x", "y"]
    assert [r["x"] for r in joined["rows"]] == [1, 2]
    assert joined["meta"] == {"settings": {"t": 0}}
    assert "meta" not in T.concat([a, c])


@pytest.mark.parametrize(
    "stat,expected",
    [("mean", 95.0), ("median", 95.0), ("stddev", math.sqrt(50)), ("min", 90.0), ("count", 2.0)],
)
def test_aggregate_groups_and_summarises(stat, expected):
    out = T.aggregate(bench_table(), ["quant"], ["tps"], stat)
    rows = {r["quant"]: r for r in out["rows"]}
    assert rows["Q4_K_M"]["tps"] == pytest.approx(expected)
    assert rows["Q4_K_M"]["n"] == 2


def test_aggregate_single_value_stddev_is_zero_and_skips_non_numbers():
    t = T.make_table([{"g": "a", "v": 3}, {"g": "a", "v": None}, {"g": "b", "v": "nan"}])
    out = {r["g"]: r for r in T.aggregate(t, ["g"], ["v"], "stddev")["rows"]}
    assert out["a"]["v"] == 0.0
    assert out["b"]["v"] is None


def test_compare_against_baseline_row():
    out = T.compare(bench_table(), None, ["quant", "ctx"], "tps", baseline_key = "Q4_K_M, 4096")
    rows = {(r["quant"], r["ctx"]): r for r in out["rows"]}
    assert rows[("Q8_0", 4096)]["delta"] == pytest.approx(-30.0)
    assert rows[("Q8_0", 4096)]["ratio"] == pytest.approx(0.7)
    assert rows[("Q4_K_M", 4096)]["delta"] == 0.0


def test_compare_two_tables_joins_on_key_and_flags_significance():
    head = bench_table()
    base = T.make_table(
        [
            dict(r, tps = r["tps"] - 1.0) if r["quant"] == "Q8_0" else dict(r, tps = r["tps"] - 10.0)
            for r in head["rows"]
        ],
        meta = head["meta"],
    )
    out = T.compare(head, base, ["quant", "ctx"], "tps", stderr = "se")
    rows = {(r["quant"], r["ctx"]): r for r in out["rows"]}
    assert rows[("Q4_K_M", 4096)]["delta"] == pytest.approx(10.0)
    assert rows[("Q4_K_M", 4096)]["significant"] is True
    assert rows[("Q8_0", 4096)]["significant"] is False
    assert any(c["id"] == "delta" and c.get("role") == "delta" for c in out["columns"])


def test_compare_refuses_tables_measured_with_different_settings():
    with pytest.raises(FlowNodeError, match = "different settings"):
        T.compare(
            bench_table({"temperature": 0}),
            bench_table({"temperature": 1}),
            ["quant", "ctx"],
            "tps",
        )
    out = T.compare(
        bench_table({"temperature": 0}),
        bench_table({"temperature": 1}),
        ["quant", "ctx"],
        "tps",
        allow_mismatch = True,
    )
    assert all(r["delta"] == 0.0 for r in out["rows"])


def test_compare_missing_column_is_a_clear_error():
    with pytest.raises(FlowNodeError, match = "no column 'latency'"):
        T.compare(bench_table(), None, ["quant"], "latency", baseline_key = "Q8_0")


def test_chart_groups_series_with_error_bars():
    c = T.chart(bench_table(), "ctx", ["tps"], kind = "line", group = "quant", error = "se")
    assert c["kind"] == "chart" and c["type"] == "line"
    assert c["x"] == [4096, 8192]
    assert [s["name"] for s in c["series"]] == ["Q4_K_M", "Q8_0"]
    assert c["series"][1]["y"] == [70.0, 60.0]
    assert c["series"][0]["error"] == [1.0, 1.0]


@run_async
async def test_big_tables_move_to_an_asset_and_load_back(monkeypatch):
    rows = [{"i": i, "text": "x" * 200} for i in range(3000)]
    ctx = Ctx()
    out = await T.finish_table(ctx, T.make_table(rows))
    assert (
        out["row_count"] == 3000
        and len(out["rows"]) == T.INLINE_ROWS
        and out["ref"].startswith("sha256:")
    )
    stored = ctx.saved[0]
    assert len(json.loads(stored)["rows"]) == 3000

    async def fake_bytes(_ctx, media):
        assert media["ref"] == out["ref"]
        return stored, "application/json"

    monkeypatch.setattr(T, "media_bytes", fake_bytes)
    full = await T.read_table(ctx, out)
    assert len(full["rows"]) == 3000 and "ref" not in full


@run_async
async def test_small_tables_stay_inline_and_lists_of_dicts_are_tables():
    ctx = Ctx(inputs = {"tables": [[{"a": 1}], T.make_table([{"a": 2}])]})
    out = await load_handlers()["table.concat"](ctx)
    assert [r["a"] for r in out["table"]["rows"]] == [1, 2]
    assert not ctx.saved


@run_async
async def test_compare_handler_reports_mean_delta():
    ctx = Ctx(
        inputs = {"table": bench_table()},
        params = {"key": ["quant", "ctx"], "metric": "tps", "baseline_row": "Q4_K_M,4096"},
    )
    out = await load_handlers()["table.compare"](ctx)
    assert out["delta"] == pytest.approx((0 - 10 - 30 - 40) / 4)


@run_async
async def test_chained_compares_report_the_new_delta():
    first = T.compare(
        T.make_table([{"m": "a", "v": 10.0}, {"m": "b", "v": 20.0}]),
        T.make_table([{"m": "a", "v": 0.0}, {"m": "b", "v": 0.0}]),
        ["m"],
        "v",
    )
    base = T.compare(
        T.make_table([{"m": "a", "v": 5.0}, {"m": "b", "v": 15.0}]),
        T.make_table([{"m": "a", "v": 0.0}, {"m": "b", "v": 0.0}]),
        ["m"],
        "v",
    )
    ctx = Ctx(inputs = {"table": first, "baseline": base}, params = {"key": ["m"], "metric": "delta"})
    out = await load_handlers()["table.compare"](ctx)
    assert out["delta"] == pytest.approx(5.0)


def test_concat_drops_settings_when_any_table_has_none():
    a = T.make_table([{"x": 1}], meta = {"settings": {"t": 0}})
    b = T.make_table([{"x": 2}])
    assert "meta" not in T.concat([a, b])


def test_aggregate_count_column_never_replaces_a_column_named_n():
    t = T.make_table([{"q": "a", "n": 5.0}, {"q": "a", "n": 7.0}])
    out = T.aggregate(t, ["q"], ["n"], "mean")
    ids = [c["id"] for c in out["columns"]]
    assert ids == ["q", "n", "n_2"] and out["rows"][0]["n"] == 6.0 and out["rows"][0]["n_2"] == 2


def test_compare_generated_columns_never_replace_selected_ones():
    t = T.make_table([{"baseline": "a", "delta": 2.0}, {"baseline": "b", "delta": 4.0}])
    out = T.compare(t, None, ["baseline"], "delta", baseline_key = "a")
    ids = [c["id"] for c in out["columns"]]
    assert len(ids) == len(set(ids))
    row = out["rows"][1]
    assert row["baseline"] == "b" and row["delta"] == 4.0 and row["delta_2"] == 2.0


def test_compare_refuses_a_missing_stderr_column_and_duplicate_baseline_keys():
    with pytest.raises(FlowNodeError, match = "sd"):
        T.compare(bench_table(), bench_table(), ["quant", "ctx"], "tps", stderr = "sd")
    with pytest.raises(FlowNodeError, match = "more than one row"):
        T.compare(bench_table(), bench_table(), ["quant"], "tps")


def test_chart_refuses_too_many_series_and_mixed_units():
    many = T.make_table([{"x": 1, "g": i, "y": 1.0} for i in range(T.MAX_SERIES + 1)])
    with pytest.raises(FlowNodeError, match = "series"):
        T.chart(many, "x", ["y"], kind = "line", group = "g", error = None)
    mixed = T.make_table(
        [{"x": 1, "a": 1.0, "b": 2.0}],
        [
            {"id": "x"},
            {"id": "a", "type": "number", "unit": "tok/s"},
            {"id": "b", "type": "number", "unit": "ms"},
        ],
    )
    assert "unit" not in T.chart(mixed, "x", ["a", "b"], kind = "bar", group = None, error = None)


@run_async
async def test_a_big_tables_preview_fits_the_inline_budget():
    wide = T.make_table([{"t": "x" * 4000} for _ in range(300)])
    ctx = Ctx()
    out = await T.finish_table(ctx, wide)
    assert out["row_count"] == 300 and out["ref"]
    assert len(json.dumps(out).encode()) <= T.INLINE_MAX_BYTES


def _t(rows, units = None):
    t = T.make_table(rows)
    for c in t["columns"]:
        if units and c["id"] in units:
            c["unit"] = units[c["id"]]
    return t


def test_compare_refuses_a_baseline_in_other_units():
    a = _t([{"model": "x", "lat": 120}], {"lat": "ms"})
    b = _t([{"model": "x", "lat": 0.1}], {"lat": "s"})
    with pytest.raises(FlowNodeError, match = "ms here but s"):
        T.compare(a, b, ["model"], "lat")


def test_compare_refuses_an_ambiguous_baseline_row():
    t = _t(
        [
            {"quant": "q4", "ctx": 1, "tps": 10},
            {"quant": "q4", "ctx": 2, "tps": 8},
            {"quant": "q8", "ctx": 1, "tps": 9},
        ]
    )
    with pytest.raises(FlowNodeError, match = "2 rows have"):
        T.compare(t, None, ["quant"], "tps", baseline_key = "q4")


def test_chart_refuses_repeated_x_instead_of_dropping_trials():
    t = _t([{"ctx": 1, "tps": 10}, {"ctx": 1, "tps": 12}, {"ctx": 2, "tps": 8}])
    with pytest.raises(FlowNodeError, match = "Aggregate"):
        T.chart(t, "ctx", ["tps"], kind = "line", group = None, error = None)


def test_chart_x_keeps_first_seen_order_on_a_large_table():
    rows = [{"i": i, "v": float(i)} for i in range(50_000)]
    c = T.chart(_t(rows), "i", ["v"], kind = "line", group = None, error = None)
    assert c["x"][:3] == [0, 1, 2] and len(c["x"]) == 50_000


def test_chart_numeric_x_stored_as_text_plots_as_numbers():
    t = T.make_table([{"ctx": "1024", "tps": 10}, {"ctx": "32768", "tps": 8}])
    c = T.chart(t, "ctx", ["tps"], kind = "line", group = None, error = None)
    assert c["x"] == [1024, 32768]


def test_column_name_lists_ignore_surrounding_spaces():
    assert T._strings(["score ", " tps"]) == ["score", "tps"]


def test_compare_matches_numeric_keys_stored_as_text():
    head = T.make_table([{"ctx": 4096, "tps": 12.0}])
    base = T.make_table([{"ctx": "4096", "tps": 10.0}])
    out = T.compare(head, base, ["ctx"], "tps")
    assert out["rows"][0]["delta"] == 2.0


def test_chart_treats_text_and_number_x_as_one_coordinate():
    t = T.make_table([{"ctx": 4096, "tps": 10}, {"ctx": "4096", "tps": 12}])
    with pytest.raises(FlowNodeError, match = "Aggregate"):
        T.chart(t, "ctx", ["tps"], kind = "line", group = None, error = None)


@run_async
async def test_compare_handler_mean_reads_the_generated_delta_column():
    head = T.make_table([{"k": "a", "delta": 5.0}, {"k": "b", "delta": 9.0}])
    base = T.make_table([{"k": "a", "delta": 4.0}, {"k": "b", "delta": 6.0}])
    ctx = Ctx(inputs = {"table": head, "baseline": base}, params = {"key": ["k"], "metric": "delta"})
    out = await load_handlers()["table.compare"](ctx)
    assert out["delta"] == pytest.approx(2.0)


@run_async
async def test_rows_to_dataset_saves_every_row_of_a_big_table(monkeypatch):
    from core.flows.nodes import data as D

    rows = [{"i": i, "text": "x" * 200} for i in range(3000)]
    ctx = Ctx()
    big = await T.finish_table(ctx, T.make_table(rows))
    stored = ctx.saved[0]

    async def fake_bytes(_ctx, media):
        return stored, "application/json"

    sent = []

    async def fake_upload(_ctx, rows, name, **_):
        sent.append(rows)
        return {"kind": "dataset"}

    monkeypatch.setattr(T, "media_bytes", fake_bytes)
    monkeypatch.setattr(D, "upload_rows", fake_upload)
    await load_handlers()["data.rows_to_dataset"](Ctx(inputs = {"rows": big}, params = {"name": "t"}))
    assert len(sent[0]) == 3000 and sent[0][-1] == {"i": 2999, "text": "x" * 200}
