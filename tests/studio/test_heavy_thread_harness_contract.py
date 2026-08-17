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

import importlib.util
import os
import sys
import types
from pathlib import Path

import pytest

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
    assert 'proof["runtimeText"] != proof["domText"]' in decision


def test_every_repetition_is_settled_before_it_is_measured() -> None:
    # Order, which no unit test can reach: the tool panes the expansion mounts carry code fences
    # of their own, and Shiki starts on them only once they exist. Settling before the expansion
    # leaves that highlighting inside the keystroke and scroll windows of every repetition after
    # the first, which is the contamination the settle exists to keep out.
    text = source(HARNESS)
    body = section(text, "def one_repetition(", "# Portable headline per action")
    expanded = body.index("expandTools()")
    settled = body.index("wait_for_highlighting_settled(")
    first_measurement = body.index('run_action(page, cdp, "keystroke"')
    assert expanded < settled < first_measurement, body


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
    assert 'proof.get("closedMs") is None' in decision


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


# ── the verdict against the summary it actually gets ──────────────────
#
# Everything above reads the source, which is all a check on ORDER or on portability can do. The
# rest runs summarise() and harness_failures() for real, because the defect they are pinned
# against is not visible in either function on its own: the summary keeps the LAST repetition's
# proofs while the headline is a median over all of them, so a repetition that measured nothing
# reaches the table with a passing verdict in front of it.


@pytest.fixture(scope = "module")
def harness(tmp_path_factory):
    """The harness module, imported for real.

    It pulls in `playwright.sync_api` and creates its artifact directory at import time, so it is
    loaded here with the browser binding stubbed when absent (this file also runs where playwright
    is not installed) and with `PW_ART_DIR` pointed at a temp dir, or the import drops a `logs/`
    tree into whatever directory pytest was run from.
    """
    stubbed: list[str] = []
    try:
        import playwright.sync_api  # noqa: F401
    except ImportError:
        package = types.ModuleType("playwright")
        binding = types.ModuleType("playwright.sync_api")
        binding.sync_playwright = None
        package.sync_api = binding
        for name, stub in (("playwright", package), ("playwright.sync_api", binding)):
            if name not in sys.modules:
                sys.modules[name] = stub
                stubbed.append(name)
    previous = os.environ.get("PW_ART_DIR")
    os.environ["PW_ART_DIR"] = str(tmp_path_factory.mktemp("heavy_thread_artifacts"))
    try:
        spec = importlib.util.spec_from_file_location(
            "_heavy_thread_under_test", STUDIO_TESTS / HARNESS
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        # Both are undone the moment the import is done. A stub left in `sys.modules` is not a
        # local shortcut: the frontend CI job runs this file in the same pytest process as
        # test_playwright_server_lifecycle.py, which imports the real binding and would get this
        # one instead. Measured, not guessed -- the first version of this fixture broke that file.
        for name in stubbed:
            sys.modules.pop(name, None)
        if previous is None:
            os.environ.pop("PW_ART_DIR", None)
        else:
            os.environ["PW_ART_DIR"] = previous


def one_good_repetition() -> dict[str, dict]:
    """What every action returns when it did what it was asked to. Only the fields the verdict
    reads; `run_action` adds the recorder's own metrics on top."""
    return {
        "keystroke": {
            "ran": True,
            "median_sample_ms": 34.0,
            "domText": "aaaaa",
            "runtimeText": "aaaaa",
        },
        "scroll": {"ran": True, "gestureMs": 700.0, "scrolledPx": 8000},
        "jump": {"ran": True, "paintedMs": 40.0, "landedAt": 0, "travelledPx": 9000},
        "menu": {
            "ran": True,
            "open_close_ms": 70.0,
            "openMs": 40.0,
            "closeMs": 30.0,
            "bodyPointerEvents": "auto",
            "bodyPointerEventsAfterClose": "auto",
            "itemsWhileOpen": 6,
            "triggersWhileHovered": 4,
        },
        "delete": {"ran": True, "ms": 50.0, "before": 220, "after": 219},
        "reopen": {"ran": True, "ms": 300.0, "closedMs": 20.0},
    }


def results_for(harness, reps: list[dict[str, dict]]) -> dict:
    """One engine at one size, healthy everywhere except what a caller put in `reps`. One size
    keeps the discrimination check, which needs two, out of the way."""
    size = 300000
    plan = {"chars": size, "messages": 220, "cycles": 10, "expectedPerCycle": {"codeBlocks": 2}}
    return {
        "engines": ["chromium"],
        "sizes": [size],
        "by_engine": {
            "chromium": {
                "by_size": {
                    str(size): {
                        "plan": plan,
                        "counts": {
                            "messages": 220,
                            "codeBlocks": 20,
                            "highlightedTokens": 35086,
                            "actionBars": 110,
                        },
                        "viewport": {"scrollHeight": 90000, "clientHeight": 800, "scrollTop": 0},
                        "stray_api_requests": 0,
                        "console_warnings": 0,
                        "first_console_warning": "-",
                        "per_repetition_census": [{"messages": 220} for _ in reps],
                        "actions": harness.summarise(reps),
                    }
                }
            }
        },
    }


def test_a_healthy_run_is_not_reported_as_broken(harness) -> None:
    # The control. Without it the two tests below would pass on a verdict that fails everything.
    reps = [one_good_repetition() for _ in range(3)]
    assert harness.harness_failures(results_for(harness, reps), {}) == []


def test_an_earlier_repetition_that_typed_into_nothing_is_caught(harness) -> None:
    # The keystroke reached the DOM but not the React runtime in repetition 1, and landed in 2 and
    # 3. Its ~33ms paint floor is in the median either way; before this the summary carried
    # repetition 3's proof and the verdict passed.
    reps = [one_good_repetition() for _ in range(3)]
    reps[0]["keystroke"]["runtimeText"] = ""
    failures = harness.harness_failures(results_for(harness, reps), {})
    assert [f for f in failures if "repetition 1" in f and "composer state" in f], failures


def test_an_earlier_repetition_whose_thread_never_unmounted_is_caught(harness) -> None:
    # `closedMs` is a proof wearing a number's clothes. Null means the thread never unmounted, and
    # REOPEN_JS's second loop then finds `messageCount() >= before` already true and returns a
    # near-zero `ms` for a thread that never left. median() drops the null and `dropped_repetitions`
    # reads only `ms`, so before this the invalid timing sat in the median with nothing to show it.
    reps = [one_good_repetition() for _ in range(3)]
    reps[0]["reopen"]["closedMs"] = None
    reps[0]["reopen"]["ms"] = 0.4
    failures = harness.harness_failures(results_for(harness, reps), {})
    assert [f for f in failures if "repetition 1" in f and "never left" in f], failures
    # The hole it went through: the summary still reports a plausible median for both.
    summary = harness.summarise(reps)["reopen"]
    assert summary["closedMs"] is not None and summary["dropped_repetitions"] == 0


def test_an_earlier_repetition_on_the_modal_layer_is_caught(harness) -> None:
    # A modal menu and a non-modal one cost wildly different amounts, so one repetition measured
    # on the other layer is not a sample of the same thing as the two beside it.
    reps = [one_good_repetition() for _ in range(3)]
    reps[0]["menu"]["bodyPointerEvents"] = "none"
    reps[0]["menu"]["bodyPointerEventsAfterClose"] = "none"
    failures = harness.harness_failures(results_for(harness, reps), {})
    assert [f for f in failures if "repetition 1" in f and "modal layer" in f], failures
    assert [f for f in failures if "same mechanism" in f], failures
