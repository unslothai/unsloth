# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The heavy-thread harness must read every guard it records, and must stay portable.

`tests/studio/playwright_heavy_thread.py` measures in a browser and then decides pass/fail in
`main()`. A metric that is recorded but never compared is how a harness goes false-green, which is
the rule already pinned for the #8483 harnesses in test_autoscroll_harness_contract.py.

This file adds the constraint that is specific to this harness: it is meant to run on WebKit and
Firefox as well as Chromium, because Unsloth Desktop is a Tauri webview and not Chromium. Every
CDP counter and the Long Tasks API are Chromium-only, and the failure mode is silent -- a
`longtask` PerformanceObserver on JavaScriptCore never fires, which reads as "no jank" rather than
as "no measurement". So no growth axis and no pass/fail decision may rest on one.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STUDIO_TESTS = ROOT / "tests" / "studio"
FRONTEND = ROOT / "studio" / "frontend"

HARNESS = "playwright_heavy_thread.py"
# Recorded by the harness, produced only by Chromium. None of these may decide anything.
CHROMIUM_ONLY = (
    "layout_count",
    "layout_ms",
    "recalc_style_count",
    "recalc_style_ms",
    "task_ms",
    "long_tasks",
    "long_task_ms",
    "worst_long_task_ms",
)
# The portable four the module docstring promises as the primary numbers.
PORTABLE_PRIMARY = ("longest_stall_ms", "worst_frame_ms", "frames_over_33", "wall_ms")
ACTIONS = ("keystroke", "scroll", "jump", "menu", "delete", "reopen")


def source(name: str) -> str:
    return (STUDIO_TESTS / name).read_text(encoding = "utf-8")


def section(text: str, start: str, end: str) -> str:
    head = text.index(start)
    return text[head : text.index(end, head)]


def growth_axes() -> str:
    return section(source(HARNESS), "GROWTH_AXES = tuple(", "DISCRIMINATION_RATIO")


def verdict() -> str:
    """Everything from `def harness_failures` on: the only place a metric turns into an exit
    code."""
    text = source(HARNESS)
    return text[text.index("def harness_failures") :]


def test_every_measured_action_has_a_growth_axis() -> None:
    # An action that is driven but never checked for growth is an action whose column could be
    # constant at every size without anything failing. The axes are generated from ACTIONS, so
    # what has to hold is that ACTIONS is the generator and that it still lists all six.
    text = source(HARNESS)
    declared = section(text, "ACTIONS = (", ")")
    for action in ACTIONS:
        assert f'"{action}"' in declared, action
    axes = growth_axes()
    for metric in PORTABLE_PRIMARY:
        assert f"for a in ACTIONS" in axes and f'"{metric}"' in axes, metric


def test_the_portable_primaries_are_the_growth_axes() -> None:
    axes = growth_axes()
    for metric in PORTABLE_PRIMARY:
        assert f'"{metric}"' in axes, metric


def test_no_growth_axis_is_chromium_only() -> None:
    # The whole point of the portable metrics: a curve built on CDP counters is a curve that does
    # not exist on the engine Unsloth Desktop actually ships on macOS and Linux.
    axes = growth_axes()
    for metric in CHROMIUM_ONLY:
        assert f'"{metric}"' not in axes, metric


def test_the_verdict_never_rests_on_a_chromium_only_metric() -> None:
    decision = verdict()
    for metric in CHROMIUM_ONLY:
        assert f'["{metric}"]' not in decision, metric


def test_chromium_only_rows_say_so_in_their_own_label() -> None:
    # Off Chromium these print `-`, and a `-` that means "not supported here" must not be read as
    # a zero. The label is the only thing carrying that.
    text = source(HARNESS)
    table = section(text, "TABLE_ROWS = (", "def print_table")
    for metric in CHROMIUM_ONLY:
        for line in table.splitlines():
            if f'"{metric}"' in line:
                assert "chromium only" in line, line


def test_the_longtask_api_is_recorded_as_supported_or_not() -> None:
    # Without this flag an engine with no Long Tasks API reports zero long tasks in exactly the
    # same shape as an engine that had none.
    text = source(HARNESS)
    assert "__longTaskSupported" in text
    assert '("longtask api supported", lambda r: r["long_task_supported"])' in text


def test_the_stall_detector_is_a_timer_and_not_a_message_channel() -> None:
    # Measured, not preference: the MessageChannel ping-pong halves Firefox's frame rate before
    # any application code runs, so it changes the thing it is there to measure.
    text = source(HARNESS)
    assert "new MessageChannel(" not in text, "the recorder must not spin a port"
    assert "setTimeout(stall, 1)" in text


def test_the_verdict_asserts_the_fixture_and_not_just_its_size() -> None:
    # 300K characters of prose would produce a rising curve too, and would be measuring something
    # nobody reported.
    decision = verdict()
    assert 'plan["expectedPerCycle"]' in decision
    assert 'counts.get("highlightedTokens", 0)' in decision


def test_the_verdict_asserts_the_keystroke_reached_the_runtime() -> None:
    # The DOM value is what the harness itself wrote. A keystroke that reached nothing still
    # reports the ~33ms paint floor, which reads as a plausible timing.
    decision = verdict()
    assert 'keystroke["runtimeText"] != keystroke["domText"]' in decision


def test_the_paint_floor_is_measured_and_subtracted() -> None:
    # Two rAFs resolve no sooner than two vsync intervals, so an action that never happened still
    # reports ~33ms. Left in a ratio, that floor compresses every axis towards 1 and lets a real
    # regression sit under the discrimination threshold.
    text = source(HARNESS)
    assert "PAINT_FLOOR_JS" in text
    assert 'value -= row["paint_floor_ms"]' in section(text, "def growth(", "def report_growth")


def test_the_verdict_asserts_the_reopen_really_unmounted() -> None:
    # Without this, "re-open" is timing a thread that never left, which is free.
    decision = verdict()
    assert 'reopened["closedMs"] is None' in decision


def test_the_verdict_asserts_discrimination() -> None:
    # A harness where the largest thread costs what the smallest does is not reporting a flat
    # curve, it is reporting that it never drove the page.
    decision = verdict()
    assert 'row["discriminated"]' in decision
    assert "DISCRIMINATION_RATIO" in decision


def test_the_smoke_page_exposes_every_count_the_fixture_gate_needs() -> None:
    page = (FRONTEND / "smoke-heavy-thread-main.tsx").read_text(encoding = "utf-8")
    expected = section(page, "const EXPECTED_PER_CYCLE", "};")
    counts = section(page, "counts(): Record<string, number>", "viewportMetrics()")
    for line in expected.splitlines():
        key = line.strip().split(":")[0]
        if key.isidentifier():
            assert f"{key}:" in counts, key


def test_the_smoke_page_is_served_and_owns_its_dev_server() -> None:
    text = source(HARNESS)
    assert (FRONTEND / "smoke-heavy-thread.html").exists()
    assert (FRONTEND / "smoke-heavy-thread-main.tsx").exists()
    assert "start_vite(PORT)" in text
    assert "stop_process(vite)" in text
