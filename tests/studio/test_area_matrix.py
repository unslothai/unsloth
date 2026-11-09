# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The long-chat gate's area matrix covers every appearance mode and every keyboard-opened surface."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _area_matrix as am  # noqa: E402

MODEL = am.MODEL_TS.read_text(encoding = "utf-8")
MATERIALS = am.parse_registry(MODEL, "MATERIALS")
LENS_MODES = am.parse_registry(MODEL, "LENS_MODES")


def test_registries_are_what_the_app_sanitizes_with():
    # A value outside these lists is dropped by the store, so they are every mode a user can pick.
    assert "MATERIALS.includes(" in MODEL and "LENS_MODES.includes(" in MODEL
    assert (
        am.DEFAULT["material"] in MATERIALS
        and am.DEFAULT["lens"] in LENS_MODES
        and am.GLASS in MATERIALS
    )


def test_every_material_and_lens_mode_is_measured():
    modes = am.modes(MATERIALS, LENS_MODES)
    names = {m["name"] for m in modes}
    for material in MATERIALS:
        if material != am.DEFAULT["material"]:
            assert material in names, f"material {material} is not in the area matrix"
    for lens in LENS_MODES:
        if lens != am.DEFAULT["lens"]:
            assert any(
                m["set"]["lens"] == lens for m in modes
            ), f"lens mode {lens} is not in the area matrix"
    assert {
        "name": "aurora",
        "base": "default",
        "set": {**am.DEFAULT, "material": "aurora"},
    } in am.modes(MATERIALS + ["aurora"], LENS_MODES)


def test_every_mode_is_measured_at_the_short_length():
    jobs = am.plan(am.app_modes(), "default", full = False)
    assert {m["name"] for t, m, _ in jobs if t == am.SHORT} == {m["name"] for m in am.app_modes()}
    assert [m["name"] for t, m, _ in jobs if t == am.LONG] == ["default"]
    for t, m, areas in jobs:
        # every material runs every area; a setting layered on one runs the areas it changes
        assert areas == (am.with_deps(am.VARIANT_AREAS) if am.is_variant(m) else list(am.AREAS))
    # an area that reuses an open dialog brings the area that opened it
    assert "settings_tabs" in am.with_deps(["settings_scroll"])
    assert set(am.VARIANT_AREAS) <= set(am.AREAS)
    full = am.plan(am.app_modes(), "all", full = True)
    assert len(full) == 2 * len(am.app_modes()) and all(a == list(am.AREAS) for _, _, a in full)


def test_every_keyboard_shortcut_is_an_area_or_says_why_not():
    ids = am.shortcut_ids(am.SHORTCUTS_TS.read_text(encoding = "utf-8"))
    listed = set(am.SHORTCUT_AREAS) | {i for group in am.SHORTCUT_NOT_AREAS.values() for i in group}
    missing = [i for i in ids if i not in listed]
    assert not missing, (
        f"new keyboard shortcut(s) {missing}: add each to SHORTCUT_AREAS in tests/studio/_area_matrix.py with the area "
        "that measures what it opens (add an area if none does), or to SHORTCUT_NOT_AREAS with the reason"
    )
    stale = listed - set(ids)
    assert not stale, f"shortcuts listed in _area_matrix.py that no longer exist: {sorted(stale)}"
    assert set(am.SHORTCUT_AREAS.values()) <= set(am.AREAS)


def test_coverage_matcher():
    triggers = {
        "dropdown-menu-trigger:Chat options": 1,
        "reasoning-trigger:Worked for 3s": 40,
        "context-menu-trigger:Gate 40 turns": 1,
        "popover-trigger:Request failed (404)": 1,
        "dropdown-menu-trigger:Brand new menu": 1,
    }
    assert am.uncovered(triggers) == ["dropdown-menu-trigger:Brand new menu"]


def _page(cells):
    return {"cells": {a: {"status": "OK", **c} for a, c in cells.items()}}


def _result(pages):
    return {"pages": pages, "modes": {m["name"]: m for m in am.app_modes()}, "areas": ["a"]}


@pytest.mark.parametrize(
    "short, long_, fails",
    [(1000, 1000, False), (1000, 5000, True), (100, 1000, False), (4000, 7000, False)],
)
def test_length_check(short, long_, fails, monkeypatch):
    monkeypatch.setattr(am, "COUNT_FLOOR", 3000)
    pages = {
        (am.SHORT, "default"): _page({"a": {"elements": short, "layout_objects": 0, "cpu_ms": 10}}),
        (am.LONG, "default"): _page({"a": {"elements": long_, "layout_objects": 0, "cpu_ms": 10}}),
    }
    rows = [
        v
        for v in am.judge(_result(pages), known = "")
        if v["kind"] == "length" and v["metric"] == "elements"
    ]
    assert [v["fail"] for v in rows] == [fails]
    if fails:
        assert not [v for v in am.judge(_result(pages), known = "length:a=6") if v["fail"]]


@pytest.mark.parametrize(
    "base, mode, fails", [(100, 300, False), (100, 500, True), (10, 60, False)]
)
def test_appearance_check(base, mode, fails, monkeypatch):
    monkeypatch.setattr(am, "MAX_RATIO", 4.0)
    monkeypatch.setattr(am, "CPU_FLOOR_MS", 80.0)
    pages = {
        (am.SHORT, "default"): _page({"a": {"elements": 10, "cpu_ms": base}}),
        (am.SHORT, "frosted"): _page({"a": {"elements": 10, "cpu_ms": mode}}),
    }
    rows = [
        v
        for v in am.judge(_result(pages), known = "")
        if v["kind"] == "appearance" and v["metric"] == "cpu_ms"
    ]
    assert [v["fail"] for v in rows] == [fails]


# Highest readings on the GitHub runners the gate runs on (identical code spans about 1.35x
# between runs; 25 runs, 2026-11-07).
# Scrolls pause the Glass blur (glass-motion.ts), so their worst readings dropped from 8.3x (thread)
# and 14.0x (Settings) to at most 4.9x (local and runner runs of that change).
RUNNER_READINGS = [
    ("blurred", "thread_scroll", 4.9),
    (am.GLASS, "reasoning_toggle", 7.8),
    ("blurred", "settings_scroll", 4.9),
    ("frosted", "settings_scroll", 4.9),
    # Before keys and presses paused the blur (main runner, Frontend CI 42610880257).
    ("liquid", "type", 4.9),
]


@pytest.mark.parametrize("mode, area, ratio", RUNNER_READINGS)
def test_shipped_ceilings_clear_runner_noise_but_not_a_real_regression(
    mode, area, ratio, monkeypatch
):
    monkeypatch.setattr(am, "MAX_RATIO", 4.0)

    def verdict(r):
        pages = {
            (am.SHORT, "default"): _page({area: {"elements": 10, "cpu_ms": 100}}),
            (am.SHORT, mode): _page({area: {"elements": 10, "cpu_ms": 100 * r}}),
        }
        result = {**_result(pages), "areas": [area]}
        [row] = [
            v
            for v in am.judge(result, known = am.KNOWN_DEFAULT)
            if v["kind"] == "appearance" and v["metric"] == "cpu_ms" and v["mode"] == mode
        ]
        return row["fail"]

    assert not verdict(
        ratio
    ), f"{mode}:{area} at {ratio}x, a reading main already gets, fails the shipped ceiling"
    assert verdict(
        ratio * 1.5
    ), f"{mode}:{area} at 1.5x main's worst reading still passes; the ceiling catches nothing"


def test_an_area_that_stops_running_fails():
    pages = {
        (am.SHORT, "default"): _page({"a": {"elements": 10, "cpu_ms": 10}}),
        (am.SHORT, "frosted"): {"cells": {"a": {"status": "NOT_RUN: control not found"}}},
    }
    [bad] = [v for v in am.judge(_result(pages), known = "") if v["fail"]]
    assert bad["kind"] == "broken" and bad["mode"] == "frosted"
    pages[(am.SHORT, "default")] = {"cells": {"a": {"status": "NOT_RUN: control not found"}}}
    assert any(
        v["kind"] == "broken" and v["mode"] == "default" for v in am.judge(_result(pages), known = "")
    )


def test_trace_windows_split_by_marks():
    ev = [
        {"ph": "M", "name": "thread_name", "pid": 1, "tid": 1, "args": {"name": "CrRendererMain"}},
        {"ph": "M", "name": "process_name", "pid": 1, "args": {"name": "Renderer"}},
        {
            "ph": "I",
            "name": "TimeStamp",
            "ts": 100,
            "pid": 1,
            "tid": 1,
            "args": {"data": {"message": "gate:begin:a"}},
        },
        {
            "ph": "X",
            "name": "UpdateLayoutTree",
            "cat": "devtools.timeline",
            "ts": 110,
            "dur": 2000,
            "pid": 1,
            "tid": 1,
            "args": {"elementCount": 7},
        },
        {
            "ph": "X",
            "name": "RunTask",
            "cat": "toplevel",
            "ts": 105,
            "dur": 5000,
            "tdur": 4000,
            "pid": 1,
            "tid": 1,
        },
        {
            "ph": "I",
            "name": "TimeStamp",
            "ts": 9000,
            "pid": 1,
            "tid": 1,
            "args": {"data": {"message": "gate:end:a"}},
        },
        {
            "ph": "X",
            "name": "UpdateLayoutTree",
            "cat": "devtools.timeline",
            "ts": 9500,
            "dur": 1000,
            "pid": 1,
            "tid": 1,
            "args": {"elementCount": 99},
        },
    ]
    row = am.split_trace(ev)["a"]
    assert row["elements"] == 7 and row["cpu_ms"] == 4.0 and row["longest_task_ms"] == 5.0


def test_owner_token_shows_every_settings_tab():
    import base64
    import json

    # Same decode as sessionAccount() in features/auth/account-session.ts.
    payload = am.auth_token().split(".")[1]
    claims = json.loads(base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)))
    assert (
        isinstance(claims["sub"], str)
        and claims["role"] == "owner"
        and claims["exp"] > 4_000_000_000
    )
    assert len(am.settings_tabs()) == 18


class _FakePage:
    def __init__(self, result):
        self.result = result

    def evaluate(
        self,
        script,
        arg = None,
    ):
        return self.result


@pytest.mark.parametrize(
    "got, ok, words",
    [
        ({"count": 18, "missing": [], "failed": []}, True, ""),
        ({"count": 13, "missing": [], "failed": []}, False, "only 13 of 18"),
        ({"count": 18, "missing": ["agents"], "failed": []}, False, "never committed"),
        ({"count": 18, "missing": [], "failed": ["data"]}, False, "error boundary"),
    ],
)
def test_settings_tabs_area_fails_when_a_panel_is_not_measured(got, ok, words):
    page = object.__new__(am.Page)
    page.page, page.last_error = _FakePage(got), ""
    assert page.act(("tabs", None)) is ok
    assert words in page.last_error


class _Closed(Exception):
    pass


class _SweepFails:
    applied, errors = {}, []

    def __init__(self, *args):
        pass

    def sweep(self, areas):
        raise _Closed("Target page, context or browser has been closed")

    def close(self):
        raise _Closed("already closed")


def _jobs():
    modes = {m["name"]: m for m in am.app_modes()}
    return [
        (am.SHORT, modes["default"], ["type"]),
        (am.SHORT, modes["frosted"], ["type"]),
        (am.LONG, modes["default"], ["type"]),
    ]


def test_a_page_that_breaks_mid_sweep_fails_the_gate():
    out: dict = {}
    am._run_jobs(
        None, _jobs()[:2], "http://x", out, lambda *a, **k: None, 0.0, page_factory = _SweepFails
    )
    assert set(out) == {(am.SHORT, "default"), (am.SHORT, "frosted")}
    assert all("error" in v and "_Closed" in v["error"] for v in out.values())
    # a page no worker reported is an error too
    am.fill_missing(out, _jobs())
    assert "no result" in out[(am.LONG, "default")]["error"]
    result = {"pages": out, "modes": {m["name"]: m for m in am.app_modes()}, "areas": ["type"]}
    broken = [v for v in am.judge(result, known = "") if v["fail"]]
    assert {(v["turns"], v["mode"]) for v in broken} == set(out)


def test_missing_default_page_fails_coverage(monkeypatch):
    out = {(t, m["name"]): {"error": "measuring the page failed: _Closed"} for t, m, _ in _jobs()}
    fake = {
        "pages": out,
        "modes": {m["name"]: m for m in am.app_modes()},
        "areas": ["type"],
        "seconds": 0.0,
    }
    monkeypatch.setattr(am, "measure", lambda *a, **k: fake)
    lines, failures, details, _ = am.run(Path("unused"), [])
    assert any("no popup-trigger list" in f for f in failures)
    assert sum("measuring the page failed" in f for f in failures) == len(out)


def test_every_area_opened_by_a_shortcut_or_toggle_proves_it_happened():
    # A key press always "succeeds"; only a visible change shows the app acted on it.
    for name, spec in am.AREAS.items():
        if spec["open"][0] == "key":
            assert (
                spec["ready"][0] != "settle"
            ), f"{name}: shortcut area must wait for what it opens"
            if spec.get("close"):
                assert (
                    spec["gone"][0] != "settle"
                ), f"{name}: shortcut area must wait for what it closes"
    assert (
        am.AREAS["sidebar_toggle"]["ready"][0] == "changed"
        and am.AREAS["sidebar_toggle"]["gone"][0] == "restored"
    )
    assert am.AREAS["api_monitor"]["ready"] == ("sel", '[data-testid="api-monitor-panel"]')
    assert am.AREAS["rename_chat"]["ready"] == ("overlay",)


def test_message_click_is_a_trusted_click_that_must_move_focus():
    spec = am.AREAS["message_click"]
    assert spec["open"][0] == "pclick" and spec["ready"][0] == "focusin"


def test_area_status_needs_open_and_close():
    spec = {"ready": ("overlay",), "gone": ("overlay_gone",)}
    assert am.area_status(spec, True, True) == "OK"
    assert am.area_status(spec, False, True).startswith("NOT_RUN: never reached overlay")
    assert "overlay_gone after closing" in am.area_status(spec, True, False)


def test_an_area_without_a_complete_trace_window_is_not_ok():
    ev = [
        {
            "ph": "I",
            "name": "TimeStamp",
            "ts": 100,
            "pid": 1,
            "tid": 1,
            "args": {"data": {"message": "gate:begin:a"}},
        },
        {
            "ph": "I",
            "name": "TimeStamp",
            "ts": 900,
            "pid": 1,
            "tid": 1,
            "args": {"data": {"message": "gate:end:a"}},
        },
        # b lost its end mark (a full trace buffer drops the tail)
        {
            "ph": "I",
            "name": "TimeStamp",
            "ts": 1000,
            "pid": 1,
            "tid": 1,
            "args": {"data": {"message": "gate:begin:b"}},
        },
    ]
    counters = am.split_trace(ev)
    assert "a" in counters and "b" not in counters
    cells = am.finish_cells(
        ["a", "b", "c"], {"a": "OK", "b": "OK", "c": "NOT_RUN: control not found"}, {}, counters
    )
    assert cells["a"]["status"] == "OK"
    assert cells["b"]["status"].startswith("NOT_RUN: its trace window is incomplete")
    assert cells["c"]["status"] == "NOT_RUN: control not found"
    # and the default page's incomplete area fails the gate instead of reading as zero cost
    pages = {(am.SHORT, "default"): {"cells": cells}}
    result = {
        "pages": pages,
        "modes": {m["name"]: m for m in am.app_modes()},
        "areas": ["a", "b", "c"],
    }
    assert {v["area"] for v in am.judge(result, known = "") if v["fail"]} == {"b", "c"}


class _LosesData:
    applied, errors, page = {}, [], None
    losses = 0

    def __init__(self, *args):
        pass

    def sweep(self, areas):
        if _LosesData.losses > 0:
            _LosesData.losses -= 1
            raise am.TraceDataLoss("Chromium reported trace data loss")
        return {a: {"status": "OK", "cpu_ms": 1.0} for a in areas}

    def close(self):
        pass


def test_trace_data_loss_is_retried_once_then_fails():
    modes = {m["name"]: m for m in am.app_modes()}
    job = [(am.LONG, modes["frosted"], ["type"])]
    _LosesData.losses = 1
    out: dict = {}
    am._run_jobs(None, job, "http://x", out, lambda *a, **k: None, 0.0, page_factory = _LosesData)
    assert out[(am.LONG, "frosted")]["cells"]["type"]["status"] == "OK"
    _LosesData.losses = 2
    out = {}
    am._run_jobs(None, job, "http://x", out, lambda *a, **k: None, 0.0, page_factory = _LosesData)
    assert "TraceDataLoss" in out[(am.LONG, "frosted")]["error"]


def test_trace_buffer_holds_a_long_page():
    # a 200-turn sweep collects ~200k events (~60-100 MB); the buffer must be well above that
    assert am.TRACE_BUFFER_KB >= 200 * 1024


def test_settings_ready_marker_is_set_from_inside_the_panel_suspense():
    # A marker on the pane outside <Suspense> names the selected tab even while the Loading fallback
    # shows; it must come from a layout effect in the resolved panel subtree instead.
    import re

    src = (am.SRC / "features" / "settings" / "settings-dialog.tsx").read_text(encoding = "utf-8")
    suspense = re.search(r"<Suspense\b.*?</Suspense>", src, re.S).group(0)
    assert re.search(
        r"<PanelCommitted\b[^>]*>\s*<Tab\b", suspense
    ), "the panel must render inside PanelCommitted"
    committed = re.search(r"function PanelCommitted\b.*?\n}\n", src, re.S).group(0)
    # rendered by the resolved subtree itself, so the Suspense fallback removes it
    assert re.search(r"<span hidden[^>]*data-settings-panel-ready=\{tab\}", committed)
    assert (
        "data-settings-panel=" not in src
    ), "a pane-level marker would read ready before the panel resolves"
    assert "data-settings-panel-ready=" in am.TABS and "data-settings-panel-failed" in am.TABS


def _stream_result(pages):
    return {"pages": pages, "modes": {m["name"]: m for m in am.app_modes()}, "areas": ["stream"]}


@pytest.mark.parametrize(
    "short, long_, fails",
    # main reads 1.4-2.1x for a 2 s stream; 3x is the limit; a small absolute rise never fails
    [(2500, 5200, False), (2500, 9000, True), (40, 180, False), (2500, 7400, False)],
)
def test_stream_is_judged_on_cpu_growth_with_the_chat(short, long_, fails):
    cell = lambda cpu: {"elements": 1200, "layout_objects": 3000, "cpu_ms": cpu}
    pages = {
        (am.SHORT, "default"): _page({"stream": cell(short)}),
        (am.LONG, "default"): _page({"stream": cell(long_)}),
    }
    rows = [
        v
        for v in am.judge(_stream_result(pages), known = "")
        if v["kind"] == "length" and v["metric"] == "cpu_ms"
    ]
    assert [v["fail"] for v in rows] == [fails]
    counts = [v for v in am.judge(_stream_result(pages), known = "") if v["metric"] == "elements"]
    assert counts and not any(v["fail"] for v in counts), "flat counts never fail"


def test_stream_area_is_last_sends_a_fixed_reply_and_waits_for_its_end():
    assert list(am.AREAS)[-1] == "stream", "it adds a turn, so it runs after every other area"
    spec = am.AREAS["stream"]
    assert spec["setup"] == ("compose", am.STREAM_PROMPT)
    assert spec["open"] == ("key", "Enter") and spec["ready"] == ("streamed", am.STREAM_END)
    frames = am.stream_frames()
    text = "".join(f["choices"][0]["delta"].get("content", "") for f in frames if "choices" in f)
    assert text.rstrip().endswith(am.STREAM_END), "the end text is the last thing drawn"
    assert "```python" in text and "- item 5" in text, "prose, a code fence and a list"
    assert any(f["choices"][0]["delta"].get("reasoning_content") for f in frames if "choices" in f)
    kinds = [f.get("type") for f in frames]
    assert kinds.count("tool_start") == 1 and kinds.count("tool_end") == 1
    assert frames[-1]["choices"][0]["finish_reason"] == "stop"
    assert am.stream_frames() == frames, "deterministic"


def test_the_stub_has_a_resident_model_and_saves_the_streamed_turn():
    api = am.Api(2)
    assert api.bodies["/api/inference/status"]["active_model"]

    class Route:
        def __init__(self, method, path):
            self.request = type("R", (), {"method": method, "url": "http://x" + path})()
            self.got = None

        def fulfill(
            self,
            status,
            content_type = None,
            body = "",
        ):
            self.got = status

    put = Route("PUT", f"/api/chat/threads/{api.rec['id']}/messages/abc")
    api.handle(put)
    assert put.got == 200
    other = Route("PUT", "/api/chat/threads/other/messages")
    api.handle(other)
    assert other.got == 404, "only the seeded thread's saves are answered"


def test_stream_wait_is_cheap_while_the_reply_streams():
    # polled inside the measured window: no DOM walk until the shim has sent its last frame
    body = am.STREAMED.split("if (!window.__gateStreamDone) return false;")
    assert len(body) == 2 and "querySelector" not in body[0]
    assert "__gateStreamDone = true" in am.STREAM_SHIM
