# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The heavy-thread harness must not report a number it did not measure.

`test_heavy_thread_harness_contract.py` pins the SHAPE of the harness: every recorded metric is
printed, no verdict rests on a Chromium-only counter. This file pins its ARITHMETIC and its
ORDERING, which is the other way a measurement harness goes false-green: it drives the page, it
prints a plausible table, and the numbers in it are of something else.

Five defects this file exists to keep out, each one previously live:

  * the frame recorder deciding which action a scheduled callback belongs to from a shared
    `running` flag, so an action that starts before the previous action's rAF has fired inherits
    that callback, charges the between-action gap to itself as a frame, and runs two recorder
    loops from then on;
  * `median()` dropping a null repetition instead of failing on it, which turns "the menu never
    opened once in three tries" into a clean three-repetition median;
  * a settle loop that timed out being indistinguishable, in the table and in the verdict, from a
    metric the engine does not support;
  * `menu open+close ms` being the sum of two independently double-rAF-floored timings while only
    one floor was subtracted from it;
  * the per-repetition fixture rebuild waiting for the syntax highlighter BEFORE mounting the tool
    result panes it has to highlight, and never restoring the message the delete action destroys.

The browser-side pieces run under node against the harness's own JS, sliced out of the Python
source verbatim, on a virtual clock: the frame pump is explicit, so "the next action began before
the previous action's callback fired" is a deterministic interleaving rather than a race someone
has to be lucky to catch.
"""

from __future__ import annotations

import copy
import json
import os
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _node_harness import require_node, run_harness  # noqa: E402

WORKDIR = Path(__file__).resolve().parents[2]
TEMP_ROOT = WORKDIR / "logs" / "heavy-thread-node"
HARNESS_SOURCE = (Path(__file__).resolve().parent / "playwright_heavy_thread.py").read_text(
    encoding = "utf-8"
)


def _sync_api_stub() -> types.ModuleType:
    """A `playwright.sync_api` that answers for every name a harness may import off it.

    The stub lands in `sys.modules` at collection time and stays there for the rest of the
    session, so it is not just this file's import that reads it: any later test that imports a
    harness gets these names instead of the real package's. A stub that spelled out only the
    names THIS file's harness needs therefore broke the others -- `playwright_strip_ansi_smoke`
    also imports `Page` and `expect`, and got `cannot import name 'Page' from
    'playwright.sync_api' (unknown location)` in the CPU job, from a stub two files away.

    Every name resolves to a callable that raises, so the stub can satisfy an import without a
    harness quietly measuring a browser that is not there. Dunders are left to fail: pytest and
    inspect probe those, and answering them makes the stub look like a package.
    """
    module = types.ModuleType("playwright.sync_api")

    def __getattr__(name: str):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)

        def unavailable(*args, **kwargs):
            raise RuntimeError(f"playwright is not installed: {name} needs a real browser")

        unavailable.__name__ = name
        return unavailable

    module.__getattr__ = __getattr__
    return module


def _load_harness():
    """Import the harness module without needing a browser.

    The module imports `playwright.sync_api` at the top for `sync_playwright`, which nothing in
    this file calls. Stubbing it keeps these tests runnable in the CPU test job, where the
    Playwright package is not installed, rather than skipping the arithmetic along with the
    browser.

    The probe imports the name off the submodule rather than the top-level package: a partial
    install leaves `playwright` importable while `playwright.sync_api` resolves to an empty
    namespace, which is the same failure with a longer traceback.
    """
    os.environ.setdefault("PW_ART_DIR", str(TEMP_ROOT / "artifacts"))
    if "playwright.sync_api" not in sys.modules:
        try:
            from playwright.sync_api import sync_playwright  # noqa: F401
        except ImportError:
            package = types.ModuleType("playwright")
            module = _sync_api_stub()
            package.sync_api = module
            sys.modules["playwright"] = package
            sys.modules["playwright.sync_api"] = module
    import playwright_heavy_thread

    return playwright_heavy_thread


HARNESS = _load_harness()



# performance.now() is the fake clock, requestAnimationFrame is a queue this file pumps, and setTimeout is a queue too
FAKE_ENV = """
let now = 0;
let rafs = [];
let timers = [];
globalThis.performance = { now: () => now };
class FakePerformanceObserver {
  constructor(cb) { this.cb = cb; }
  observe() {}
  disconnect() {}
}
FakePerformanceObserver.supportedEntryTypes = [];
globalThis.PerformanceObserver = FakePerformanceObserver;
globalThis.setTimeout = (fn, ms) => { timers.push({ fn, at: now + (ms || 0) }); return timers.length; };
const win = { requestAnimationFrame: (cb) => { rafs.push(cb); return rafs.length; } };
globalThis.window = win;

const drain = async () => {
  for (let i = 0; i < 40; i += 1) await new Promise((r) => setImmediate(r));
};
// One frame. Callbacks scheduled DURING the frame land in the next one, which is what the browser
// does and is the whole point: a callback scheduled by the previous action is still pending when
// the next action starts.
const tick = async (deltaMs) => {
  now += deltaMs;
  const dueTimers = timers.filter((t) => t.at <= now);
  timers = timers.filter((t) => t.at > now);
  for (const t of dueTimers) t.fn();
  const dueFrames = rafs;
  rafs = [];
  for (const cb of dueFrames) cb(now);
  await drain();
};
const idle = (ms) => { now += ms; };
const say = (value) => console.log(JSON.stringify(value));
"""


def node_script(body: str, *, sources: dict[str, str]) -> str:
    """A runnable .mts: the fake environment, the sliced harness JS, then `body`."""
    declarations = "\n".join(
        f"const {name} = {json.dumps(text)};" for name, text in sources.items()
    )
    return f"{FAKE_ENV}\n{declarations}\n{body}\n"


def run_node(body: str, sources: dict[str, str]) -> dict:
    require_node([])
    return run_harness(TEMP_ROOT, "export const unused = 0;\n", node_script(body, sources = sources))


RECORDER_SOURCES = {"RECORDER_INIT": HARNESS.RECORDER_INIT}



# ── the recorder must not leak a loop into the next action ────────────
LEAK_BODY = """
eval(RECORDER_INIT);
const hv = window.__hv;
hv.begin();
await tick(16);
await tick(16);
const first = hv.end();
// The gap between two actions: a CDP round trip and a page.evaluate, during which the page is
// busy finishing the previous action and paints nothing. The rAF the last frame callback
// scheduled is still pending when the next action begins.
idle(40);
hv.begin();
await tick(16);
await tick(16);
const second = hv.end();
say({ first_frames: first.frames, second_frames: second.frames,
      second_worst: second.worst_frame_ms });
"""


def test_the_recorder_starts_the_next_action_with_one_loop() -> None:
    # Two frames pumped, two frames recorded.
    got = run_node(LEAK_BODY, RECORDER_SOURCES)
    assert got["second_frames"] == 2, got


def test_the_recorder_does_not_charge_the_between_action_gap_as_a_frame() -> None:
    # The stale callback's own `lastFrame` is from the PREVIOUS action, so the interval it pushes spans the gap between
    # the two actions.
    # It lands in worst_frame_ms and, once the gap is over 33ms, in frames_over_33 as well: both growth axes.
    got = run_node(LEAK_BODY, RECORDER_SOURCES)
    assert got["second_worst"] == 16, got



# ── re-open must not stop the clock while Shiki is still running ──────
REOPEN_SOURCES = {"RECORDER_INIT": HARNESS.RECORDER_INIT, "REOPEN_JS": HARNESS.REOPEN_JS}

# A thread that remounts instantly and then highlights for 40 frames, which is the shape re-open actually has:
REOPEN_BODY = """
eval(RECORDER_INIT);
let mounted = true;
let tokens = 3000;
let highlightFrames = 0;
window.__heavyThread = {
  messageCount: () => (mounted ? 20 : 0),
  closeThread: () => { mounted = false; tokens = 0; },
  openThread: () => { mounted = true; highlightFrames = 40; },
  highlightedTokenCount: () => tokens,
};
const reopen = eval("(" + REOPEN_JS + ")");
let done = null;
reopen([100000, 100000, 1000, 100]).then((r) => { done = r; });
let lastChangeAt = 0;
// A FIXED number of frames, not "until it resolves": the last-token-change time has to be the
// same reference whether the settle loop stopped early or not, or an early stop moves the
// reference it is being compared against and the comparison proves nothing.
for (let i = 0; i < 200; i += 1) {
  if (highlightFrames > 0) { tokens += 75; highlightFrames -= 1; lastChangeAt = now + 16; }
  await tick(16);
}
say({
  settleMs: done === null ? null : done.settleMs,
  lastChangeAt,
  wallMs: done === null ? null : done.metrics.wall_ms,
  frames: done === null ? null : done.metrics.frames,
  medianFrameMs: done === null ? null : done.metrics.median_frame_ms,
});
"""


def test_reopen_keeps_the_clock_running_until_the_highlighter_stops() -> None:
    # quiet() settles on three sub-33ms frames, and the lull between two Shiki batches is longer than three frames
    # the same lull wait_for_highlighting_settled() needs five stable reads to see through.
    got = run_node(REOPEN_BODY, REOPEN_SOURCES)
    assert got["settleMs"] >= got["lastChangeAt"] - 20, got


def test_reopen_does_not_charge_the_grace_window_to_the_action() -> None:
    got = run_node(REOPEN_BODY, REOPEN_SOURCES)
    assert got["settleMs"] <= got["lastChangeAt"] + 120, got


def test_reopen_counts_no_frames_from_the_grace_window() -> None:
    got = run_node(REOPEN_BODY, REOPEN_SOURCES)
    # 640ms of highlighting at 16ms a frame, and nothing past it.
    assert got["frames"] <= 45, got


def test_reopen_closes_its_recorder_window_at_the_last_activity() -> None:
    # The grace is a fixed cost every size pays equally, and a constant on both ends of a ratio drags it towards 1.
    # The reported settle is the time of the LAST activity, so the 1000ms of watching that confirmed it is not in the
    # number.
    got = run_node(REOPEN_BODY, REOPEN_SOURCES)
    assert got["wallMs"] <= got["lastChangeAt"] + 120, got



# ── no-regression guards: what the timed windows may scan ───────────── GUARDS, not evidence for a finding.
# Both timed windows are measured in the browser and the harness-owned DOM scans inside them are a rounding error today:
# on Chromium at 300000 characters the re-open window makes 2 selector passes costing 0.4ms of a 2292ms re-open (0.02%),
# and the menu window makes 2 observer queries and 2 censuses costing 2.7ms of a 3208ms open+close (0.08%).
# That holds because the count of scans is FIXED while their unit cost grows with the thread (messageCount 0.01ms at
# 25000 chars, 0.105ms at 300000).
# The menu content is portaled to the END of document.body, so a querySelector for it walks the whole message list
SCAN_BUDGET_BODY = """
eval(RECORDER_INIT);
let mounted = true;
let tokens = 3000;
let highlightFrames = 0;
// The rebuild straddles this many frames, so the poll loop really does go round more than once.
let mountFrames = 4;
let scans = 0;
let censuses = 0;
let probes = 0;
window.__heavyThread = {
  messageCount: () => { scans += 1; return mounted && mountFrames <= 0 ? 20 : 0; },
  counts: () => { censuses += 1; return { messages: 20, domNodes: 4000 }; },
  closeThread: () => { mounted = false; tokens = 0; },
  openThread: () => { mounted = true; mountFrames = 4; highlightFrames = 40; },
  highlightedTokenCount: () => { probes += 1; return tokens; },
};
mountFrames = 0;
const reopen = eval("(" + REOPEN_JS + ")");
let done = null;
let doneAt = null;
const startedAt = now;
reopen([100000, 100000, 1000, 100]).then((r) => { done = r; doneAt = now; });
for (let i = 0; i < 200; i += 1) {
  if (mountFrames > 0) mountFrames -= 1;
  if (highlightFrames > 0) { tokens += 75; highlightFrames -= 1; }
  await tick(16);
}
say({
  scans, censuses, probes,
  mountFrames: 4,
  probeBudget: Math.ceil((doneAt - startedAt) / 100) + 2,
  ms: done === null ? null : done.ms,
});
"""


def test_the_reopen_window_never_runs_the_document_census() -> None:
    got = run_node(SCAN_BUDGET_BODY, REOPEN_SOURCES)
    assert got["censuses"] == 0, got


def test_the_reopen_window_scans_no_more_than_once_per_frame_of_the_rebuild() -> None:
    got = run_node(SCAN_BUDGET_BODY, REOPEN_SOURCES)
    assert got["scans"] <= 6, got


def test_the_highlight_probe_stays_on_its_interval_inside_the_settle_loop() -> None:
    # GUARD.
    # counts() is a dozen document-wide queries including getElementsByTagName("*"), and it is 25 times the cost of
    # messageCount() at 25000 chars and 24 times at 300000 (measured 0.15ms/0.01ms and 2.51ms/0.105ms on Chromium).
    got = run_node(SCAN_BUDGET_BODY, REOPEN_SOURCES)
    assert got["probes"] <= got["probeBudget"], got


MENU_SCAN_BODY = """
eval(RECORDER_INIT);
let menuOpen = false;
let observer = null;
let notifications = 0;
let querySelectorCalls = 0;
let querySelectorAllCalls = 0;
class FakeMutationObserver {
  constructor(cb) { observer = cb; }
  observe() {}
  disconnect() { observer = null; }
}
globalThis.MutationObserver = FakeMutationObserver;
globalThis.PointerEvent = class { constructor(type) { this.type = type; } };
globalThis.KeyboardEvent = class { constructor(type) { this.type = type; } };
globalThis.getComputedStyle = () => ({ pointerEvents: menuOpen ? "none" : "auto" });
const notify = () => {
  queueMicrotask(() => { if (observer) { notifications += 1; observer([]); } });
};
const trigger = {
  dispatchEvent: (e) => { if (e.type === "pointerdown") { menuOpen = true; notify(); } },
};
globalThis.document = {
  body: {},
  querySelector: (sel) => {
    querySelectorCalls += 1;
    return sel === ".aui-action-bar-more-content" && menuOpen ? {} : null;
  },
  querySelectorAll: () => { querySelectorAllCalls += 1; return []; },
  dispatchEvent: (e) => { if (e.type === "keydown") { menuOpen = false; notify(); } },
};
window.__heavyThread = { actionButton: () => trigger, openMenuItemCount: () => 5 };
const menu = eval("(" + MENU_JS + ")");
let done = null;
menu(100000).then((r) => { done = r; });
for (let i = 0; i < 40 && done === null; i += 1) await tick(16);
say({ notifications, querySelectorCalls, querySelectorAllCalls, openMs: done.openMs });
"""


def test_the_menu_reads_its_open_flag_from_the_mutation_and_not_from_a_scan() -> None:
    got = run_node(MENU_SCAN_BODY, MENU_SOURCES)
    assert got["querySelectorCalls"] <= got["notifications"] + 1, got


def test_the_menu_window_takes_one_census_and_not_one_per_frame() -> None:
    # GUARD.
    # Measured on Chromium at 300000 chars that query is 0.25ms a call against 0.025ms at 25000.
    got = run_node(MENU_SCAN_BODY, MENU_SOURCES)
    assert got["querySelectorAllCalls"] <= 1, got



MENU_SOURCES = {"RECORDER_INIT": HARNESS.RECORDER_INIT, "MENU_JS": HARNESS.MENU_JS}

# A menu that opens and closes with NO work at all:
MENU_BODY = """
eval(RECORDER_INIT);
let menuOpen = false;
let observer = null;
class FakeMutationObserver {
  constructor(cb) { observer = cb; }
  observe() {}
  disconnect() { observer = null; }
}
globalThis.MutationObserver = FakeMutationObserver;
globalThis.PointerEvent = class { constructor(type) { this.type = type; } };
globalThis.KeyboardEvent = class { constructor(type) { this.type = type; } };
globalThis.getComputedStyle = () => ({ pointerEvents: menuOpen ? "none" : "auto" });
// A MutationObserver callback is delivered as a MICROTASK ("queue a mutation observer microtask"
// in the DOM standard), never synchronously inside the mutation. That is what puts the floor
// under this metric: the settle loop's first comparison runs before the callback has updated the
// flag, so it always waits out one __nextPaint() even when the menu opened instantly.
const notify = () => { queueMicrotask(() => { if (observer) observer(); }); };
const trigger = { dispatchEvent: (e) => { if (e.type === "pointerdown") { menuOpen = true; notify(); } } };
globalThis.document = {
  body: {},
  querySelector: (sel) => (sel === ".aui-action-bar-more-content" && menuOpen ? {} : null),
  querySelectorAll: () => [],
  dispatchEvent: (e) => { if (e.type === "keydown") { menuOpen = false; notify(); } },
};
window.__heavyThread = {
  actionButton: () => trigger,
  openMenuItemCount: () => 5,
};
const menu = eval("(" + MENU_JS + ")");
let done = null;
menu(100000).then((r) => { done = r; });
for (let i = 0; i < 40 && done === null; i += 1) await tick(16);
say({ openMs: done.openMs, closeMs: done.closeMs, total: done.open_close_ms });
"""


def test_opening_the_menu_costs_a_whole_double_raf_even_when_it_is_free() -> None:
    # settle() reads `open` before the MutationObserver callback that would have updated it has been delivered, so its
    got = run_node(MENU_BODY, MENU_SOURCES)
    assert got["openMs"] >= 32, got


def test_closing_the_menu_costs_a_second_double_raf() -> None:
    # is wrong with that -- it is what makes the number a wall-clock one -- but it means the metric
    # And this is the one the single-floor subtraction missed:
    got = run_node(MENU_BODY, MENU_SOURCES)
    assert got["closeMs"] >= 32, got


def test_the_menu_growth_axis_subtracts_both_floors() -> None:
    name, pick, floors = next(a for a in HARNESS.GROWTH_AXES if a[0] == "menu open+close ms")
    assert floors == 2, f"{name} is clocked across two double rAFs, so it carries two floors"


def test_the_menu_growth_value_has_both_floors_removed() -> None:
    _, pick, floors = next(a for a in HARNESS.GROWTH_AXES if a[0] == "menu open+close ms")
    cells = {
        "25000": {"paint_floor_ms": 30.0, "actions": {"menu": {"open_close_ms": 100.0}}},
        "300000": {"paint_floor_ms": 30.0, "actions": {"menu": {"open_close_ms": 1000.0}}},
    }
    small, large = HARNESS.growth(cells, pick, floors, [25000, 300000])
    assert (small, large) == (40.0, 940.0), (small, large)




def menu_repetition(open_ms: float | None) -> dict:
    close_ms = 40.0
    total = None if open_ms is None else open_ms + close_ms
    return {
        "menu": {
            "name": "menu",
            "ran": True,
            "openMs": open_ms,
            "closeMs": close_ms,
            "open_close_ms": total,
            "bodyPointerEvents": "none",
            "bodyPointerEventsAfterClose": "auto",
            "itemsWhileOpen": 5,
            "triggersWhileHovered": 3,
        }
    }


def test_one_repetition_that_never_opened_the_menu_poisons_the_median() -> None:
    summary = HARNESS.summarise(
        [menu_repetition(80.0), menu_repetition(None), menu_repetition(120.0)]
    )
    assert summary["menu"]["openMs"] is None, summary["menu"]


def test_a_timing_that_was_null_in_every_repetition_is_still_reported() -> None:
    # KeyError instead of naming the action that never happened.
    # Present-and-None, not absent:
    summary = HARNESS.summarise([menu_repetition(None), menu_repetition(None)])
    assert "openMs" in summary["menu"] and summary["menu"]["openMs"] is None, summary["menu"]


def test_the_median_of_three_good_repetitions_is_unchanged() -> None:
    # The guard above must not cost the normal case its median.
    # Filtering the null out and taking the median of the rest silently changes the sample population, and the
    summary = HARNESS.summarise(
        [menu_repetition(80.0), menu_repetition(100.0), menu_repetition(120.0)]
    )
    assert summary["menu"]["openMs"] == 100.0, summary["menu"]




# ── the verdict must reject an action that never settled ──────────────
def clean_cell() -> dict:
    """One (engine, size) cell that harness_failures() has nothing to say about."""
    per_cycle = {"images": 3, "codeBlocks": 7, "codeChars": 12000}
    counts = {
        "messages": 20,
        "domNodes": 4000,
        "codeBlocks": 7,
        "codeChars": 12660,
        "highlightedTokens": 3000,
        "fenceBlocks": 5,
        "deferredFences": 2,
        "unhighlightedMountedFences": 0,
        "images": 3,
        "actionBars": 10,
        "tooltipTriggers": 30,
        "collapsibleOutputs": 2,
    }
    action = {"ran": True, "wall_ms": 100.0, "longest_stall_ms": 10.0, "worst_frame_ms": 20.0}
    return {
        "chars_requested": 25000,
        "engine": "chromium",
        "plan": {"chars": 26000, "messages": 20, "cycles": 1, "expectedPerCycle": per_cycle},
        "counts": counts,
        "viewport": {"scrollHeight": 20000, "clientHeight": 900, "scrollTop": 0},
        "paint_floor_ms": 30.0,
        "stray_api_requests": 0,
        "console_warnings": 0,
        "first_console_warning": "-",
        "actions": {
            "keystroke": {
                **action,
                "median_sample_ms": 40.0,
                "domText": "aaa",
                "runtimeText": "aaa",
            },
            "scroll": {**action, "gestureMs": 300.0, "settleMs": 400.0, "scrolledPx": 8000},
            "jump": {
                **action,
                "paintedMs": 90.0,
                "settleMs": 200.0,
                "travelledPx": 19100,
                "landedAt": 0,
            },
            "menu": {
                **action,
                "openMs": 100.0,
                "closeMs": 40.0,
                "open_close_ms": 140.0,
                "itemsWhileOpen": 5,
                "triggersWhileHovered": 3,
                "bodyPointerEvents": "none",
                "bodyPointerEventsAfterClose": "auto",
            },
            "delete": {**action, "ms": 100.0, "before": 20, "after": 19},
            "reopen": {
                **action,
                "ms": 400.0,
                # One double-rAF wait, the floor GROWTH_AXES subtracts from the `reopen ms` axis.
                "paintWaits": 1,
                "closePaintWaits": 1,
                "closedMs": 20.0,
                "settleMs": 900.0,
                "before": 20,
                "after": 20,
            },
        },
    }


def results_with(cell: dict) -> dict:
    return {
        "engines": ["chromium"],
        "sizes": [25000],
        "repetitions": 3,
        "by_engine": {"chromium": {"by_size": {"25000": cell}}},
    }


def discriminating_report() -> dict:
    return {"chromium": {"scroll wall ms": {"discriminated": True, "ratio": 4.0}}}


def test_a_clean_cell_produces_no_harness_failure() -> None:
    # The guards below are only worth anything if they are silent on a good run.
    assert HARNESS.harness_failures(results_with(clean_cell()), discriminating_report()) == []


def test_deferred_fences_are_not_a_harness_failure() -> None:
    # THE POINT OF MEASURING CHARACTERS.
    # A real, complete thread now holds far fewer highlighted tokens than before the default moved: 1,322 against 3,216
    # for one cycle, measured, with the fixture untouched.
    cell = copy.deepcopy(clean_cell())
    cell["counts"]["deferredFences"] = 4
    cell["counts"]["highlightedTokens"] = 300
    assert HARNESS.harness_failures(results_with(cell), discriminating_report()) == []


def test_a_fixture_that_lost_its_code_is_still_a_harness_failure() -> None:
    cell = copy.deepcopy(clean_cell())
    cell["counts"]["codeChars"] = 6000
    cell["counts"]["highlightedTokens"] = 99999
    failures = HARNESS.harness_failures(results_with(cell), discriminating_report())
    assert any("6000 codeChars, short of the 12000" in f for f in failures), failures


def test_a_fence_that_is_neither_deferred_nor_highlighted_is_a_harness_failure() -> None:
    cell = copy.deepcopy(clean_cell())
    cell["counts"]["unhighlightedMountedFences"] = 1
    failures = HARNESS.harness_failures(results_with(cell), discriminating_report())
    assert any("mounted but unhighlighted" in f for f in failures), failures


def test_a_scroll_that_never_settled_is_a_harness_failure() -> None:
    cell = copy.deepcopy(clean_cell())
    cell["actions"]["scroll"]["settleMs"] = None
    failures = HARNESS.harness_failures(results_with(cell), discriminating_report())
    assert any(
        "scroll action but it never reached a settled state" in f for f in failures
    ), failures


def test_a_reopen_that_never_settled_is_a_harness_failure() -> None:
    cell = copy.deepcopy(clean_cell())
    cell["actions"]["reopen"]["settleMs"] = None
    failures = HARNESS.harness_failures(results_with(cell), discriminating_report())
    assert any(
        "reopen action but it never reached a settled state" in f for f in failures
    ), failures


def test_a_jump_that_never_settled_is_a_harness_failure() -> None:
    # The replaced check caught a fixture that is not the heavy thread it claims to be, and this still does.
    # Tokens are left HIGH so only the character floor can fail: a thread whose code blocks quietly emptied still
    # renders, still scrolls and still curves, of something else.
    cell = copy.deepcopy(clean_cell())
    cell["actions"]["jump"]["settleMs"] = None
    failures = HARNESS.harness_failures(results_with(cell), discriminating_report())
    assert any("jump action but it never reached a settled state" in f for f in failures), failures




class StubLocator:
    def __init__(self, log: list, selector: str) -> None:
        self.log = log
        self.selector = selector

    @property
    def last(self):
        return self

    def hover(self, timeout = None) -> None:
        self.log.append(("hover", self.selector))


class StubPage:
    """Records the order of the calls one_repetition() makes, and nothing else.

    The question this answers is an ORDERING one -- does the highlighter gate run after the panes
    it has to highlight are mounted, and is the destroyed message put back before the next action
    measures the thread -- so a call log is the whole instrument.
    """

    ACTION_METRICS = {
        "metrics": {"wall_ms": 1.0, "frames": 1, "worst_frame_ms": 1.0},
        "ms": 1.0,
    }

    def __init__(self) -> None:
        self.log: list = []

    def evaluate(
        self,
        script,
        arg = None,
    ):
        for name in ("KEYSTROKE_JS", "SCROLL_JS", "JUMP_JS", "MENU_JS", "DELETE_JS", "REOPEN_JS"):
            if script is getattr(HARNESS, name):
                self.log.append(("action", name))
                return dict(self.ACTION_METRICS)
        if "__longTaskSupported" in script:
            return {"supported": False, "tasks": []}
        if "__longTasks.length" in script:
            return None
        if "expandTools" in script:
            self.log.append(("evaluate", "expandTools"))
            return 2
        if "restore()" in script:
            self.log.append(("evaluate", "restore"))
            return 20
        if "__hvTokens = undefined" in script:
            return None
        self.log.append(("evaluate", "other"))
        return None

    def wait_for_function(
        self,
        script,
        arg = None,
        timeout = None,
        polling = None,
    ):
        if "highlightedTokenCount" in script:
            self.log.append(("wait", "highlighting"))
        elif "codeExecutionPanes" in script:
            self.log.append(("wait", "toolPanes"))
        elif "messageCount" in script:
            self.log.append(("wait", "messageCount"))
        else:
            self.log.append(("wait", "other"))

    def locator(self, selector: str) -> StubLocator:
        return StubLocator(self.log, selector)


def repetition_log() -> list:
    page = StubPage()
    HARNESS.one_repetition(page, None)
    return page.log


def test_the_highlighter_gate_runs_after_the_tool_panes_are_mounted() -> None:
    log = repetition_log()
    expand = log.index(("evaluate", "expandTools"))
    highlight = log.index(("wait", "highlighting"))
    assert expand < highlight, log


def test_the_highlighter_gate_runs_before_the_first_measured_action() -> None:
    log = repetition_log()
    highlight = log.index(("wait", "highlighting"))
    assert highlight < log.index(("action", "KEYSTROKE_JS")), log


def test_the_deleted_message_is_restored_before_anything_else_is_measured() -> None:
    log = repetition_log()
    assert ("evaluate", "restore") in log, log
    assert (
        log.index(("action", "DELETE_JS"))
        < log.index(("evaluate", "restore"))
        < log.index(("action", "REOPEN_JS"))
    ), log


def test_the_restored_thread_is_re_expanded_and_re_highlighted_before_re_open() -> None:
    # The tool result panes ARE code -- two of the seven fences a content cycle produces -- and Radix does not mount
    # The delete removes a message from the repository, permanently.
    # Restoring re-imports the thread, which unmounts every tool card and throws away every highlighted fence.
    log = repetition_log()
    restore = log.index(("evaluate", "restore"))
    reopen = log.index(("action", "REOPEN_JS"))
    assert any(entry == ("wait", "highlighting") for entry in log[restore:reopen]), log
    assert any(entry == ("evaluate", "expandTools") for entry in log[restore:reopen]), log


def test_the_tool_expand_gate_is_not_satisfied_by_a_thread_of_closed_cards() -> None:
    # `collapsibleOutputs` is the CollapsibleContent element itself, and Radix keeps that element in the tree for its
    # collapse animation, so it is there while the card is shut.
    # The pane's <pre> is a CHILD of that element: 0 collapsed, 22 expanded.
    gates = [line for line in HARNESS_SOURCE.splitlines() if "counts()." in line]
    assert gates, "the fixture no longer gates on a count at all"
    for line in gates:
        assert "codeExecutionPanes" in line, line


def test_the_smoke_page_can_restore_the_thread_it_seeded() -> None:
    page = (WORKDIR / "studio" / "frontend" / "smoke-heavy-thread-main.tsx").read_text(
        encoding = "utf-8"
    )
    assert "restore(): number" in page
    assert "seeded.current = built.messages" in page




# high subtracts time the action never spent. REOPEN_JS now counts the waits it actually pays, so
# the declared paint floor ────────────────────────────────────────── `growth()` subtracts `floored` double-rAF vsync
def floor_row(
    observed,
    engine = "chromium",
    size = 25000,
):
    cell = {"actions": {"reopen": {"ms": 120.0, "paintWaits": observed}}}
    return {
        "engines": [engine],
        "sizes": [size],
        "by_engine": {engine: {"by_size": {str(size): cell}}},
    }


def test_reopen_subtracts_only_the_observation_floor() -> None:
    """A CONSTANT, and one. Reopen is driven by a React state update, so the count check straight
    after openThread() always still sees the unmounted tree and the loop always waits out one
    paint before a finished reopen can be observed at all. That wait is the instrument.

    Every further wait is not. The poll shares the rAF queue with the application's own commits,
    so on a progressive-mount build those waits are the application mounting rows and the time in
    them is real convergence latency. Reading the measured `paintWaits` here subtracted all of
    them, which is a subtraction that grows with thread size and differs between the two arms of
    a comparison -- it removed ~800ms from one arm's 300K cell and ~33ms from the other's."""
    floored = HARNESS.declared_floor("reopen ms")
    assert not callable(floored), "reopen ms reads its floor from the row again"
    assert floored == 1 == HARNESS.REOPEN_OBSERVATION_FLOOR, floored


def test_a_matching_floor_declaration_is_accepted() -> None:
    """The control. Without it the checks below could be met by rejecting every run."""
    assert HARNESS.floor_declaration_problems(floor_row(1)) == []


def test_a_multi_commit_reopen_is_not_a_mis_declared_floor() -> None:
    """The progressive mount window mounts a long thread over several frames, so `ms` -- which
    runs until messageCount() reaches `before` -- spans one paint wait per widening commit. Ten is
    an ordinary reading at 100K, not a harness fault, and reporting it as one stops the run after
    every measurement has already been taken."""
    assert HARNESS.floor_declaration_problems(floor_row(10)) == []


def test_subtracting_a_floor_the_reopen_never_paid_is_a_failure() -> None:
    """The other side of the lower bound. Equality is no longer required, so the check has to
    still catch the direction that invents time: a cell whose reopen resolved without waiting a
    paint at all, while the axis removes one."""
    problems = HARNESS.floor_declaration_problems(floor_row(0))
    assert len(problems) == 1 and "never waited out" in problems[0], problems


def test_the_progressive_mount_frames_stay_in_the_reopen_number() -> None:
    """End to end, and the regression this file exists to hold.

    Two arms of one comparison at the same size: a single-commit build that pays one wait and a
    progressive-mount build that pays twenty-four for the same 220 messages. Subtracting the
    measured count made the slower arm read as the faster one. Only the shared observation floor
    comes out, so the arms stay ordered the way the clock ordered them."""
    pick = next(p for name, p, _f in HARNESS.GROWTH_AXES if name == "reopen ms")
    floored = HARNESS.declared_floor("reopen ms")

    def cell(ms: float, waits: int) -> dict:
        return {"paint_floor_ms": 33.3, "actions": {"reopen": {"ms": ms, "paintWaits": waits}}}

    def value(ms: float, waits: int) -> float:
        row = cell(ms, waits)
        return round(pick(row) - HARNESS.resolve_floor(floored, row) * row["paint_floor_ms"], 1)

    single_commit = value(2184.1, 1)
    progressive = value(2329.7, 24)
    assert single_commit == 2150.8, single_commit
    assert progressive == 2296.4, progressive
    assert progressive > single_commit, (progressive, single_commit)


def test_an_unreported_wait_count_is_a_failure_not_a_skip() -> None:
    """A cell that stopped reporting `paintWaits` leaves the subtraction unverified. Skipping it
    is how this check would quietly stop checking anything."""
    row = floor_row(1)
    del row["by_engine"]["chromium"]["by_size"]["25000"]["actions"]["reopen"]["paintWaits"]
    problems = HARNESS.floor_declaration_problems(row)
    assert len(problems) == 1 and "unverified" in problems[0], problems


def test_a_crashed_cell_is_not_reported_twice() -> None:
    """`harness_failures` already reports the crash itself, with the reason."""
    row = floor_row(0)
    row["by_engine"]["chromium"]["by_size"]["25000"]["crashed"] = "boom"
    assert HARNESS.floor_declaration_problems(row) == []


def test_the_floor_check_decides_the_exit_code() -> None:
    """Wiring, not logic: a checker whose result never reaches `harness_failures` passes its own
    unit tests forever while the harness goes on publishing ratios built on the wrong constant."""
    body = HARNESS_SOURCE.split("def harness_failures", 1)[1]
    assert (
        "floor_declaration_problems(results)" in body
    ), "harness_failures ignores the paint-floor check, so a mis-declared floor never fails"


def test_reopen_counts_the_waits_it_pays() -> None:
    """Source-level half: the counter has to be incremented in the loop that clocks `ms`."""
    reopen = HARNESS.REOPEN_JS.split("api.openThread();", 1)[1]
    assert "paintWaits += 1;" in reopen, "the reopen timing loop no longer counts its paint waits"
    assert "paintWaits," in HARNESS.REOPEN_JS, "the count is never returned to the harness"


def test_an_action_that_did_not_run_is_not_reported_twice() -> None:
    """`harness_failures` already reports the action itself, with the reason. A second failure
    here for the same cause buries the real one."""
    row = floor_row(1)
    reopen = row["by_engine"]["chromium"]["by_size"]["25000"]["actions"]["reopen"]
    reopen["ran"] = False
    del reopen["paintWaits"]
    assert HARNESS.floor_declaration_problems(row) == []


def test_an_action_that_ran_without_a_count_is_still_reported() -> None:
    """The other side of the skip above, so it cannot be widened into a blanket exemption."""
    row = floor_row(1)
    reopen = row["by_engine"]["chromium"]["by_size"]["25000"]["actions"]["reopen"]
    reopen["ran"] = True
    del reopen["paintWaits"]
    problems = HARNESS.floor_declaration_problems(row)
    assert len(problems) == 1 and "unverified" in problems[0], problems




# the wall axes carry the floor they actually paid ──────────────────
def wall_cells(
    paint_waits: int,
    floor_ms: float = 33.0,
    wall_ms: float = 200.0,
) -> dict:
    return {
        str(size): {
            "paint_floor_ms": floor_ms,
            "actions": {"menu": {"wall_ms": wall_ms, "paint_waits": paint_waits}},
        }
        for size in (25000, 300000)
    }


def wall_axis():
    for name, pick, floored in HARNESS.GROWTH_AXES:
        if name == "menu wall ms":
            return pick, floored
    raise AssertionError("menu wall ms is no longer an axis")


def test_the_wall_axis_floor_is_read_from_the_row_not_declared() -> None:
    """`MENU_JS` opens the recorder before opening the menu and closes it after closing it, so it
    crosses the same two waits `menu open+close ms` declares. The generated wall axes declared 0
    for every action, leaving that floor in both ends of the ratio."""
    _pick, floored = wall_axis()
    assert callable(floored), "the wall axis floor is a hardcoded number again"
    row = {"actions": {"menu": {"paint_waits": 2}}}
    assert floored(row) == 2


def test_the_measured_floor_is_actually_subtracted() -> None:
    pick, floored = wall_axis()
    small, large = HARNESS.growth(wall_cells(2), pick, floored, [25000, 300000])
    assert small == 200.0 - 2 * 33.0, small
    assert large == 200.0 - 2 * 33.0, large


def test_an_axis_that_paid_no_waits_is_left_alone() -> None:
    """The control: the change must not subtract a floor from a window that never waited."""
    pick, floored = wall_axis()
    small, _large = HARNESS.growth(wall_cells(0), pick, floored, [25000, 300000])
    assert small == 200.0, small


def test_a_missing_wait_count_subtracts_nothing_rather_than_crashing() -> None:
    """An older result file has no `paint_waits`. Subtracting an unknown count would be worse
    than subtracting none, and raising would drop the axis entirely."""
    _pick, floored = wall_axis()
    assert floored({"actions": {"menu": {}}}) == 0
    assert floored({}) == 0




def error_cell(seed_errors = 0, action_errors = 0) -> dict:
    cell = copy.deepcopy(clean_cell())
    cell["seed_console_errors"] = seed_errors
    cell["first_seed_error"] = "boom" if seed_errors else "-"
    cell["console_errors"] = action_errors
    cell["first_console_error"] = "TypeError: x is not a function" if action_errors else "-"
    return cell


def test_a_clean_cell_still_passes_with_the_error_counters_present() -> None:
    """The control, so the two checks below cannot be met by rejecting every run."""
    assert HARNESS.harness_failures(results_with(error_cell()), discriminating_report()) == []


def test_one_console_error_during_the_actions_fails_the_run() -> None:
    """Previously this went into the same list as Gecko's two scroll-anchoring notices and passed
    under the `> 4` allowance, so a run could exit 0 with an application exception in it."""
    failures = HARNESS.harness_failures(
        results_with(error_cell(action_errors = 1)), discriminating_report()
    )
    assert any("console error" in f and "measured actions" in f for f in failures), failures


def test_one_console_error_during_seeding_fails_the_run() -> None:
    failures = HARNESS.harness_failures(
        results_with(error_cell(seed_errors = 1)), discriminating_report()
    )
    assert any("console error" in f and "seeding" in f for f in failures), failures


def test_the_warning_allowance_still_tolerates_engine_chatter() -> None:
    """Firefox emits two scroll-anchoring notices and the harness must still be able to report a
    Gecko number. Errors are rejected; warnings under the allowance are not."""
    cell = error_cell()
    cell["console_warnings"] = HARNESS.CONSOLE_WARNING_ALLOWANCE
    assert HARNESS.harness_failures(results_with(cell), discriminating_report()) == []


def test_warnings_and_errors_are_captured_into_separate_lists() -> None:
    """Source-level: one list for both is how the severity was lost in the first place."""
    assert "console_errors: list[str] = []" in HARNESS_SOURCE
    assert 'page.on("pageerror", lambda e: console_errors.append' in HARNESS_SOURCE
    assert 'if m.type == "error"' in HARNESS_SOURCE, "errors are no longer routed by severity"
    assert 'if m.type == "warning"' in HARNESS_SOURCE, "warnings are no longer routed by severity"
    # Keyed on the predicate alone, not on the whole expression around it:
    assert 'm.type in ("warning", "error")' not in HARNESS_SOURCE, (
        "one predicate captures both severities again, so an application exception is filed as "
        "chatter and absorbed by the warning allowance"
    )


def test_the_recorder_zeroes_its_wait_counter_at_the_start_of_each_window() -> None:
    """Without the reset the count is cumulative across every action in the page, so each window
    subtracts a larger floor than it paid and later actions go negative."""
    begin = HARNESS_SOURCE.split("    begin() {", 1)[1].split("},", 1)[0]
    assert "__paintWaits = 0" in begin, (
        "begin() no longer zeroes the paint-wait counter, so every window inherits the waits of "
        "the ones before it"
    )




# ── a counter rising from zero has to say something ─────────────────── No ratio can be formed against zero, so
# DISCRIMINATION_RATIO never applies to these axes and `large > small` was the whole test.
# The CI workflow runs one repetition on Chromium, so there is no median to smooth a stray dropped frame, and
# `harness_failures` accepts any ONE discriminating axis: 0 at 25K and 1 at 100K could carry the entire verdict while
# every latency axis was flat.
def counter_cells(
    small,
    large,
    sizes = (25000, 300000),
) -> dict:
    return {
        "engines": ["chromium"],
        "sizes": list(sizes),
        "repetitions": 1,
        "by_engine": {
            "chromium": {
                "by_size": {
                    str(sizes[0]): {
                        "paint_floor_ms": 33.0,
                        "actions": {"scroll": {"frames_over_33": small}},
                    },
                    str(sizes[1]): {
                        "paint_floor_ms": 33.0,
                        "actions": {"scroll": {"frames_over_33": large}},
                    },
                }
            }
        },
    }


def counter_axis(results):
    return HARNESS.report_growth(results)["chromium"]["scroll frames over 33ms"]


def test_one_stray_frame_does_not_discriminate() -> None:
    """The reported case: 0 at the smallest size and 1 at the largest."""
    row = counter_axis(counter_cells(0, 1))
    assert row["discriminated"] is False, row
    assert "under the" in row["reason"], row


def test_a_counter_that_really_rose_still_discriminates() -> None:
    """The control. Without it the check above could be met by rejecting every counter, which
    would leave the harness unable to report a live run at all."""
    row = counter_axis(counter_cells(0, HARNESS.ZERO_BASED_MIN_RISE))
    assert row["discriminated"] is True, row
    assert "rose from zero to" in row["reason"], row


def test_a_counter_just_under_the_threshold_does_not_discriminate() -> None:
    """Pins the boundary, so widening the threshold is a visible edit rather than a quiet one."""
    row = counter_axis(counter_cells(0, HARNESS.ZERO_BASED_MIN_RISE - 1))
    assert row["discriminated"] is False, row


def test_zero_at_both_ends_is_still_reported_as_flat() -> None:
    row = counter_axis(counter_cells(0, 0))
    assert row["discriminated"] is False and row["reason"] == "zero at both ends", row


def test_the_threshold_is_absolute_because_no_ratio_exists() -> None:
    """`DISCRIMINATION_RATIO` is a ratio and there is nothing to divide by here, so a separate
    absolute constant is required rather than a reuse of that one."""
    assert HARNESS.ZERO_BASED_MIN_RISE == 5
    assert "large >= ZERO_BASED_MIN_RISE" in HARNESS_SOURCE, (
        "the zero-based branch is back to a bare `large > small`, so one incidental dropped "
        "frame can carry the liveness verdict"
    )


# The zero branch keyed on `floored`, which only identifies a timing that had a paint floor subtracted.


# ── zero-based is not the same as counted ───────────────────────────── The zero branch keyed on `floored`, which only
# identifies a timing that had a paint floor subtracted.
# An UNFLOORED timing reads zero at the smallest size whenever the action resolves before the recorder produces a
# sample, and was then treated as a dropped-frame counter, so a noisy 5ms at the largest size read as a rise of 5 and
# discriminated.
def timing_cells(
    small,
    large,
    sizes = (25000, 300000),
) -> dict:
    return {
        "engines": ["chromium"],
        "sizes": list(sizes),
        "repetitions": 1,
        "by_engine": {
            "chromium": {
                "by_size": {
                    str(sizes[0]): {
                        "paint_floor_ms": 33.0,
                        "actions": {"scroll": {"longest_stall_ms": small}},
                    },
                    str(sizes[1]): {
                        "paint_floor_ms": 33.0,
                        "actions": {"scroll": {"longest_stall_ms": large}},
                    },
                }
            }
        },
    }


def test_an_unfloored_timing_at_zero_does_not_discriminate() -> None:
    """`scroll longest stall ms` carries no paint floor, so the old `floored` test did not catch
    it, and 0 -> 5 milliseconds read as a rise of 5 events."""
    row = HARNESS.report_growth(timing_cells(0, 5))["chromium"]["scroll longest stall ms"]
    assert row["discriminated"] is False, row
    assert "timing, not a count" in row["reason"], row


def test_a_counter_is_still_judged_as_a_counter() -> None:
    """The control: the fix must not turn every zero-based axis into an automatic no."""
    row = HARNESS.report_growth(counter_cells(0, HARNESS.ZERO_BASED_MIN_RISE))["chromium"][
        "scroll frames over 33ms"
    ]
    assert row["discriminated"] is True, row


def test_the_counter_set_is_stated_and_non_empty() -> None:
    """Inferring which axes are counts is what produced the defect. If a rename ever empties this
    set, every counter silently becomes a timing and the liveness verdict loses its only
    zero-based axis, so the set is pinned against the axis list itself."""
    assert HARNESS.COUNTER_AXES, "no axis is classified as a counter any more"
    names = {name for name, _pick, _floored in HARNESS.GROWTH_AXES}
    assert (
        HARNESS.COUNTER_AXES <= names
    ), f"COUNTER_AXES names axes that do not exist: {HARNESS.COUNTER_AXES - names}"
    # in "ms", which is wrong: the counter axis is called "frames over 33ms" and legitimately
    # Stated exactly rather than pattern-matched.
    assert HARNESS.COUNTER_AXES == frozenset(
        f"{action} frames over 33ms" for action in HARNESS.ACTIONS
    ), HARNESS.COUNTER_AXES
    timings = names - HARNESS.COUNTER_AXES
    assert (
        "scroll longest stall ms" in timings and "menu wall ms" in timings
    ), f"a timing axis is classified as a count: {HARNESS.COUNTER_AXES}"


# Making the wall floor a callable put the lambda itself into the growth report.


# ── the report has to survive being written out ─────────────────────── Making the wall floor a callable put the lambda
# itself into the growth report.
# main() attaches that report to `results` and json.dumps it, so every complete run raised "Object of type function is
# not JSON serializable" AFTER taking all its measurements.
def test_the_growth_report_is_json_serializable() -> None:
    report = HARNESS.report_growth(counter_cells(0, 9))
    json.dumps(report)


def test_the_report_is_serializable_for_a_callable_floor_axis() -> None:
    """The wall axes are the ones whose floor is a callable, so they are the case that broke."""
    report = HARNESS.report_growth(
        {
            "engines": ["chromium"],
            "sizes": [25000, 300000],
            "repetitions": 1,
            "by_engine": {
                "chromium": {
                    "by_size": {
                        size: {
                            "paint_floor_ms": 33.0,
                            "actions": {"menu": {"wall_ms": 200.0, "paint_waits": 2}},
                        }
                        for size in ("25000", "300000")
                    }
                }
            },
        }
    )
    json.dumps(report)
    assert report["chromium"]["menu wall ms"]["floored"] == [2, 2], report["chromium"][
        "menu wall ms"
    ]


def test_the_recorded_floor_is_the_count_that_was_subtracted() -> None:
    """A boolean would serialise too and say nothing. The report carries the actual count at each
    end of the ratio, so a reader can check the subtraction rather than trust it."""
    report = HARNESS.report_growth(wall_cells_for_report(2))
    assert report["chromium"]["menu wall ms"]["floored"] == [2, 2]
    report = HARNESS.report_growth(wall_cells_for_report(0))
    assert report["chromium"]["menu wall ms"]["floored"] == [0, 0]


def wall_cells_for_report(paint_waits: int) -> dict:
    return {
        "engines": ["chromium"],
        "sizes": [25000, 300000],
        "repetitions": 1,
        "by_engine": {
            "chromium": {
                "by_size": {
                    size: {
                        "paint_floor_ms": 33.0,
                        "actions": {"menu": {"wall_ms": 200.0, "paint_waits": paint_waits}},
                    }
                    for size in ("25000", "300000")
                }
            }
        },
    }




# ── halves, and counters that never left the noise ────────────────────
def test_a_fractional_median_wait_count_is_not_truncated() -> None:
    """`summarise` medians the wait count across repetitions, so an even-repetition run whose
    repetitions paid 1 and 2 reports 1.5. Truncating that to 1 left half a vsync floor in the
    wall axis, and the documented two-repetition configurations are the ones that produce it."""
    assert (
        HARNESS.resolve_floor(
            HARNESS._floor_from("menu", "paint_waits"), {"actions": {"menu": {"paint_waits": 1.5}}}
        )
        == 1.5
    )


def test_the_fractional_floor_is_actually_subtracted() -> None:
    report = HARNESS.report_growth(wall_cells_for_report(1.5))
    row = report["chromium"]["menu wall ms"]
    assert row["floored"] == [1.5, 1.5], row
    assert row["small"] == round(200.0 - 1.5 * 33.0, 2), row


def test_a_counter_with_a_nonzero_baseline_still_needs_the_noise_floor() -> None:
    """1 -> 2 dropped frames is a ratio of 2.0 and cleared DISCRIMINATION_RATIO, and one
    discriminating axis is enough for the whole run to pass."""
    row = HARNESS.report_growth(counter_cells(1, 2))["chromium"]["scroll frames over 33ms"]
    assert row["discriminated"] is False, row
    assert "distinguishable from noise" in row["reason"], row


def test_a_counter_that_rose_well_past_the_noise_floor_still_discriminates() -> None:
    """The control, so the rule above cannot be met by rejecting every nonzero counter."""
    row = HARNESS.report_growth(counter_cells(2, 40))["chromium"]["scroll frames over 33ms"]
    assert row["discriminated"] is True, row


def test_a_timing_with_a_small_nonzero_baseline_is_untouched() -> None:
    """The noise floor is a count of events. Applying it to milliseconds would silently reject
    real latency curves that happen to sit at low absolute values."""
    row = HARNESS.report_growth(timing_cells(1.0, 4.0))["chromium"]["scroll longest stall ms"]
    assert row["discriminated"] is True, row


# `quiet()` and `quietUntilIdle()` return `...


# ── whole-window axes carry the whole window's floors ───────────────── `quiet()` and `quietUntilIdle()` return `...
# - this.startedAt`, not the time they themselves took, and `gestureMs` is computed from `startedAt` too.
def axis_floor(name):
    for axis, _pick, floored in HARNESS.GROWTH_AXES:
        if axis == name:
            return floored
    raise AssertionError(f"{name} is not an axis")


@pytest.mark.parametrize("name", ("scroll gesture ms", "scroll settle ms", "jump settle ms"))
def test_whole_window_axes_use_the_measured_floor(name) -> None:
    floored = axis_floor(name)
    assert callable(floored), f"{name} declares a fixed floor but spans the whole window"
    action = name.split(" ")[0]
    assert floored({"actions": {action: {"paint_waits": 20}}}) == 20


def test_reopen_is_a_partial_window_axis_with_a_constant_floor() -> None:
    """Reopen is the third kind, and it is neither of the two below.

    `ms` is measured from `reopenStarted`, so it does not span the recorder window and cannot take
    the window's `paint_waits`. It cannot take its own `paintWaits` either, because that count is
    a property of how many frames the APPLICATION took to mount, which is the thing the axis is
    measuring. What it carries is one observation wait, always, on every build.
    """
    floored = axis_floor("reopen ms")
    assert not callable(floored), "reopen ms reads a per-row wait count again"
    assert floored == HARNESS.REOPEN_OBSERVATION_FLOOR == 1, floored


def test_the_reopen_wall_axis_does_not_take_the_window_count_either() -> None:
    """`reopen wall ms` spans the close loop and the reopen loop, so the window's `paint_waits`
    carries the progressive mount's commit frames for the same reason `reopen ms` does. Its
    honest floor is the two terminal observation waits, one per loop. Every other action keeps
    the measured window count, where those waits really are harness idle between driven steps."""
    floored = axis_floor("reopen wall ms")
    assert not callable(floored), "reopen wall ms reads the window count again"
    assert floored == 2, floored
    assert callable(axis_floor("scroll wall ms")), "the other wall axes lost their measured floor"


@pytest.mark.parametrize("name", ("jump painted ms", "menu open+close ms"))
def test_partial_window_axes_keep_their_declared_floor(name) -> None:
    """The other half of the rule, and it is not symmetry for its own sake.

    `paintedMs` starts at a mark taken after `begin()` and spans one wait while the jump's window
    holds two. `MENU_JS` awaits no paint at all, so the window count is zero while its two floors
    are real, coming from `settle()` reading the pre-MutationObserver state on entry. Applying the
    window count to either would subtract a floor the number never contained, or remove one that
    it did.
    """
    assert not callable(
        axis_floor(name)
    ), f"{name} was given the whole-window floor, which it does not carry"


def test_the_scroll_ratio_is_not_compressed_by_the_gesture_floors() -> None:
    """End to end: twenty floors left in both ends is what flattens the curve."""
    cells = {
        "engines": ["chromium"],
        "sizes": [25000, 300000],
        "repetitions": 1,
        "by_engine": {
            "chromium": {
                "by_size": {
                    "25000": {
                        "paint_floor_ms": 33.0,
                        "actions": {"scroll": {"gestureMs": 700.0, "paint_waits": 20}},
                    },
                    "300000": {
                        "paint_floor_ms": 33.0,
                        "actions": {"scroll": {"gestureMs": 1300.0, "paint_waits": 20}},
                    },
                }
            }
        },
    }
    row = HARNESS.report_growth(cells)["chromium"]["scroll gesture ms"]
    # 700 - 660 = 40, 1300 - 660 = 640: a 16x curve that reads as 1.86x with the floors left in.
    assert row["small"] == 40.0 and row["large"] == 640.0, row
    assert row["ratio"] > 10 and row["discriminated"] is True, row


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
