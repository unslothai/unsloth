# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The long-chat gate's Firefox leg: its limits, surfaces and fail-closed runs (CI wiring: test_firefox_leg_ci.py)."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _area_matrix as am  # noqa: E402
import _firefox_leg as ff  # noqa: E402
import _webkit_leg as wk  # noqa: E402

LEG = ff.FIREFOX


def _cell(
    open_ms,
    close_ms = 60.0,
    frame = 40.0,
    close_frame = None,
):
    return {
        "status": "OK",
        "reps": 3,
        "open_ms": open_ms,
        "open_frame": frame,
        "close_ms": close_ms,
        "close_frame": frame if close_frame is None else close_frame,
    }


def _result(
    short,
    long_,
    mode = "default",
    surface = "composer_menu",
):
    return {
        ("firefox", mode, LEG.short): {"cells": {surface: short}},
        ("firefox", mode, LEG.long): {"cells": {surface: long_}},
    }


def _fails(res, known = None):
    return [
        (v["surface"], v["metric"])
        for v in ff.judge(res, known = {} if known is None else known)
        if v["fail"]
    ]


def test_the_leg_runs_firefox_on_its_own_engine_with_its_own_switch():
    assert LEG.browser == "firefox" and list(LEG.engines) == ["firefox"]
    assert LEG.env == "LONG_THREAD_FIREFOX" and LEG.name == "Firefox"
    assert LEG.modes == ("default", am.GLASS)


def test_the_modal_cover_regression_fails():
    # Main before the Firefox-only cover rule, composer menu at 150 turns: 349-425 ms open with a
    # 160-340 ms frame, against about 110 ms and a 40 ms frame at 1 turn.
    res = _result(_cell(110, close_ms = 120, frame = 40), _cell(380, close_ms = 390, frame = 280))
    assert set(_fails(res)) >= {("composer_menu", "open"), ("composer_menu", "open_frame")}


def test_the_fixed_reading_passes():
    # With the cover rule: 133-250 ms open at 150 turns, frames 33-130 ms.
    res = _result(_cell(110, close_ms = 120, frame = 40), _cell(185, close_ms = 200, frame = 110))
    assert not _fails(res)


@pytest.mark.parametrize(
    "short, long_, fails",
    [
        (110, 380, True),
        (110, 260, False),  # under 2.5x
        (60, 190, False),  # over 2.5x but under 200 ms
    ],
)
def test_open_time_rule(short, long_, fails):
    rows = [
        v for v in ff.judge(_result(_cell(short), _cell(long_)), known = {}) if v["metric"] == "open"
    ]
    assert [v["fail"] for v in rows] == [fails]


def test_a_known_ceiling_holds_a_cost_and_fails_anything_worse():
    res = _result(_cell(110), _cell(380))
    key = "firefox:default:composer_menu:open"
    assert not [f for f in _fails(res, {key: 4.0}) if f[1] == "open"]
    assert ("composer_menu", "open") in _fails(_result(_cell(110), _cell(500)), {key: 4.0})


def test_load_and_run_failures_fail_closed():
    assert _fails({("firefox", "default", LEG.long): {"error": "did not load"}}) == [("*", "load")]
    assert _fails(_result(_cell(100), {"status": "NOT_RUN: never reached overlay"})) == [
        ("composer_menu", "run")
    ]


def test_a_crashed_firefox_worker_fails_every_planned_page(monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("firefox did not launch")

    class _Server:
        def shutdown(self):
            pass

    monkeypatch.setattr(wk, "_engine_worker", boom)
    monkeypatch.setattr(wk.am, "serve_app", lambda dist: (_Server(), "http://x"))
    res = ff.measure(Path("."), modes = ["default"], log = lambda *a, **k: None)
    assert set(res) == {("firefox", "default", LEG.short), ("firefox", "default", LEG.long)}
    assert all("firefox did not launch" in page["error"] for page in res.values())
    assert _fails(res) == [("*", "load"), ("*", "load")]


def test_every_surface_is_a_matrix_area_the_leg_can_time():
    for name in LEG.surfaces:
        spec = am.AREAS[name]
        for pred in (spec["ready"], spec.get("gone")):
            if pred is None or pred[0] == "settle":
                continue  # settle: timed by the MutationObserver window, no predicate
            wk.pred_js(pred, 1, "x")
    # The modal surfaces the cover regression hit are all measured.
    assert {
        "settings",
        "composer_menu",
        "tools_menu",
        "chat_options",
        "command_palette",
        "search_chats",
    } <= set(LEG.surfaces)
    assert "browser_pane" in LEG.surfaces and "type" in LEG.surfaces


def test_settle_surfaces_ignore_state_flags_that_are_not_content():
    for flag in ("data-aui-following", "data-glass-moving", "data-glass-settling"):
        assert flag in wk.SETTLE_IGNORED


def test_the_webkit_leg_keeps_its_own_limits():
    assert wk.WEBKIT.browser == "webkit" and wk.WEBKIT.engines is wk.ENGINES
    assert wk.WEBKIT.ratio == wk.RATIO and wk.WEBKIT.known is wk.KNOWN
    assert "type" not in wk.WEBKIT.surfaces


def test_settled_counts_the_frame_that_renders_the_last_mutation():
    # A slow layout / paint after the final typing mutation ends its frame long after the mutation; that
    # frame must still be the "longest frame" (last mutation at 160 ms, a 316 ms frame ending at 460 ms).
    import shutil
    import subprocess

    node = shutil.which("node")
    if node is None:
        pytest.skip("node not available")
    script = (
        "globalThis.performance = {now: () => 10000};"
        "globalThis.requestAnimationFrame = (cb) => setTimeout(() => cb(10000), 0);"
        "globalThis.window = {__wk: {t0: 0, lastMut: 160, stop: false, mo: {disconnect() {}},"
        " frames: [[128, 16], [144, 16], [460, 316], [476, 16]]}};"
        f"const settled = ({wk.SETTLED});"
        "settled([300, 60000]).then((r) => console.log(JSON.stringify(r)));"
    )
    out = subprocess.run([node, "-e", script], capture_output = True, text = True, check = True).stdout
    assert json.loads(out) == [160, 316]
