# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The long-chat gate's WebKit leg: verdicts, known ceilings and rechecks (CI wiring: test_webkit_leg_ci.py)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _area_matrix as am  # noqa: E402
import _webkit_leg as wk  # noqa: E402

REPO = Path(__file__).resolve().parents[2]


def _cell(
    open_ms,
    close_ms = 50.0,
    frame = 40.0,
):
    return {
        "status": "OK",
        "reps": 3,
        "open_ms": open_ms,
        "open_frame": frame,
        "close_ms": close_ms,
        "close_frame": frame,
    }


def _result(
    short,
    long_,
    engine = "webkit_mac",
    mode = "default",
    surface = "browser_pane",
):
    return {
        (engine, mode, wk.SHORT): {"cells": {surface: short}},
        (engine, mode, wk.LONG): {"cells": {surface: long_}},
    }


@pytest.mark.parametrize(
    "short, long_, fails",
    [
        (150, 500, True),  # Round 11's macOS WebKit pane: 144-165 -> 460-562 ms
        (150, 180, False),
        (60, 190, False),  # over 2.5x but under 200 ms
        (300, 700, False),  # under 2.5x
    ],
)
def test_open_time_ratio(short, long_, fails):
    rows = [
        v for v in wk.judge(_result(_cell(short), _cell(long_)), known = {}) if v["metric"] == "open"
    ]
    assert [v["fail"] for v in rows] == [fails]
    if fails:
        known = {"webkit_mac:default:browser_pane:open": 4.0}
        assert not [
            v for v in wk.judge(_result(_cell(short), _cell(long_)), known = known) if v["fail"]
        ]


def test_long_frame_fails_and_can_be_pinned():
    res = _result(_cell(100, frame = 30), _cell(120, frame = 300))
    assert [v["metric"] for v in wk.judge(res, known = {}) if v["fail"]] == [
        "open_frame",
        "close_frame",
    ]
    known = {
        "webkit_mac:default:browser_pane:open_frame": 350,
        "webkit_mac:default:browser_pane:close_frame": 350,
    }
    assert not [v for v in wk.judge(res, known = known) if v["fail"]]


def test_a_surface_that_stops_running_in_the_long_chat_fails():
    res = _result(_cell(100), {"status": "NOT_RUN: never reached sel"})
    assert [v["metric"] for v in wk.judge(res, known = {}) if v["fail"]] == ["run"]


def test_a_long_page_that_does_not_load_fails():
    res = {("webkit_mac", "default", wk.LONG): {"error": "did not load"}}
    assert [v["metric"] for v in wk.judge(res, known = {}) if v["fail"]] == ["load"]


def test_a_short_page_that_does_not_load_fails():
    res = {
        ("webkit_mac", "default", wk.SHORT): {"error": "did not load"},
        ("webkit_mac", "default", wk.LONG): {"cells": {"browser_pane": _cell(10_000)}},
    }
    assert [v["metric"] for v in wk.judge(res, known = {}) if v["fail"]] == ["load"]


def test_a_surface_without_a_short_baseline_fails():
    res = _result({"status": "NOT_RUN: never reached sel"}, _cell(10_000))
    assert [v["metric"] for v in wk.judge(res, known = {}) if v["fail"]] == ["run"]


def test_a_surface_that_runs_at_neither_length_fails():
    res = _result({"status": "NOT_RUN: a"}, {"status": "NOT_RUN: b"})
    assert [v["metric"] for v in wk.judge(res, known = {}) if v["fail"]] == ["run"]


def test_a_crashed_engine_worker_fails_every_planned_page(monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("webkit did not launch")

    class _Server:
        def shutdown(self):
            pass

    monkeypatch.setattr(wk, "_engine_worker", boom)
    monkeypatch.setattr(wk.am, "serve_app", lambda dist: (_Server(), "http://x"))
    res = wk.measure(Path("."), modes = ["default"], engines = ["webkit_mac"], log = lambda *a, **k: None)
    assert set(res) == {("webkit_mac", "default", wk.SHORT), ("webkit_mac", "default", wk.LONG)}
    assert all("webkit did not launch" in page["error"] for page in res.values())
    assert [v["metric"] for v in wk.judge(res, known = {}) if v["fail"]] == ["load", "load"]


def test_short_floor_keeps_instant_surfaces_from_inflating_the_ratio():
    v = [v for v in wk.judge(_result(_cell(5), _cell(210)), known = {}) if v["metric"] == "open"][0]
    assert v["ratio"] == pytest.approx(210 / wk.SHORT_FLOOR_MS)


def test_a_recheck_only_lowers_readings():
    first = _result(_cell(150), _cell(600, frame = 300))
    again = _result(_cell(140), _cell(400, frame = 320))
    merged = wk.lower(first, again)
    cell = merged[("webkit_mac", "default", wk.LONG)]["cells"]["browser_pane"]
    assert cell["open_ms"] == 400 and cell["open_frame"] == 300


def test_every_surface_is_a_matrix_area_with_a_timing_predicate():
    for name in wk.SURFACES:
        spec = am.AREAS[name]
        if spec["open"][0] != "scroll":
            wk.pred_js(spec["ready"], 0, "x")
        if spec.get("close"):
            wk.pred_js(spec["gone"], 1, "x")


def test_known_ceilings_name_their_fix():
    text = (Path(wk.__file__)).read_text(encoding = "utf-8")
    block = text[text.index("KNOWN_DEFAULT = {") : text.index("KNOWN = {")]
    for key in wk.KNOWN_DEFAULT:
        engine, mode, surface, metric = key.split(":")
        assert engine in wk.ENGINES and mode in wk.MODES and surface in wk.SURFACES
        assert metric in ("open", "close", "open_frame", "close_frame")
    if wk.KNOWN_DEFAULT:
        assert "Fix work:" in block


def test_a_frame_that_is_already_long_in_the_short_chat_is_a_mount_cost_not_a_long_chat_one():
    # WebKit on a CI runner draws a 300-600 ms frame mounting Settings at 1 turn.
    res = _result(_cell(600, frame = 400), _cell(620, frame = 450))
    assert not [v for v in wk.judge(res, known = {}) if v["fail"]]
