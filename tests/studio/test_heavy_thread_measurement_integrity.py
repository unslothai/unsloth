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


def _load_harness():
    """Import the harness module without needing a browser.

    The module imports `playwright.sync_api` at the top for `sync_playwright`, which nothing in
    this file calls. Stubbing it keeps these tests runnable in the CPU test job, where the
    Playwright package is not installed, rather than skipping the arithmetic along with the
    browser.
    """
    os.environ.setdefault("PW_ART_DIR", str(TEMP_ROOT / "artifacts"))
    if "playwright.sync_api" not in sys.modules:
        try:
            import playwright.sync_api  # noqa: F401
        except ImportError:
            package = types.ModuleType("playwright")
            module = types.ModuleType("playwright.sync_api")
            module.sync_playwright = None
            package.sync_api = module
            sys.modules["playwright"] = package
            sys.modules["playwright.sync_api"] = module
    import playwright_heavy_thread

    return playwright_heavy_thread


HARNESS = _load_harness()


# ── the node side: the harness's own JS on a virtual clock ────────────

# performance.now() is the fake clock, requestAnimationFrame is a queue this file pumps, and
# setTimeout is a queue too -- the recorder's stall loop reschedules itself forever, so a real
# timer would keep the process alive past the end of the test.
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
    # Two frames pumped, two frames recorded. A stale callback that survived into this action
    # would be recording alongside the real one, so every frame is counted twice -- and it
    # compounds, because each begin() can leave another behind.
    got = run_node(LEAK_BODY, RECORDER_SOURCES)
    assert got["second_frames"] == 2, got


def test_the_recorder_does_not_charge_the_between_action_gap_as_a_frame() -> None:
    # The stale callback's own `lastFrame` is from the PREVIOUS action, so the interval it pushes
    # spans the gap between the two actions. It lands in worst_frame_ms and, once the gap is over
    # 33ms, in frames_over_33 as well: both growth axes.
    got = run_node(LEAK_BODY, RECORDER_SOURCES)
    assert got["second_worst"] == 16, got


# ── re-open must not stop the clock while Shiki is still running ──────

REOPEN_SOURCES = {"RECORDER_INIT": HARNESS.RECORDER_INIT, "REOPEN_JS": HARNESS.REOPEN_JS}

# A thread that remounts instantly and then highlights for 40 frames, which is the shape re-open
# actually has: the message roots are back long before the fences inside them are coloured.
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
    # quiet() settles on three sub-33ms frames, and the lull between two Shiki batches is longer
    # than three frames -- the same lull wait_for_highlighting_settled() needs five stable reads to
    # see through. Settling on frames alone stops the re-open clock partway through the rebuild.
    got = run_node(REOPEN_BODY, REOPEN_SOURCES)
    assert got["settleMs"] >= got["lastChangeAt"] - 20, got


def test_reopen_does_not_charge_the_grace_window_to_the_action() -> None:
    # The grace is a fixed cost every size pays equally, and a constant on both ends of a ratio
    # drags it towards 1. The reported settle is the time of the LAST activity, so the 1000ms of
    # watching that confirmed it is not in the number.
    got = run_node(REOPEN_BODY, REOPEN_SOURCES)
    assert got["settleMs"] <= got["lastChangeAt"] + 120, got


def test_reopen_counts_no_frames_from_the_grace_window() -> None:
    # The grace window watches an idle page, so every frame in it is a fast one. Left in the
    # arrays they inflate `frames`, drag `median_frame_ms` down towards the idle rate, and do it
    # by the same fixed amount at every size -- a constant offset on a growth axis.
    got = run_node(REOPEN_BODY, REOPEN_SOURCES)
    # 640ms of highlighting at 16ms a frame, and nothing past it.
    assert got["frames"] <= 45, got


def test_reopen_closes_its_recorder_window_at_the_last_activity() -> None:
    # Same reason, for wall ms, which is a growth axis in its own right.
    got = run_node(REOPEN_BODY, REOPEN_SOURCES)
    assert got["wallMs"] <= got["lastChangeAt"] + 120, got


# ── no-regression guards: what the timed windows may scan ─────────────
#
# GUARDS, not evidence for a finding. They pass on this tree and are here to keep it that way.
#
# Both timed windows are measured in the browser and the harness-owned DOM scans inside them are
# a rounding error today: on Chromium at 300000 characters the re-open window makes 2 selector
# passes costing 0.4ms of a 2292ms re-open (0.02%), and the menu window makes 2 observer queries
# and 2 censuses costing 2.7ms of a 3208ms open+close (0.08%). That holds because the count of
# scans is FIXED while their unit cost grows with the thread (messageCount 0.01ms at 25000 chars,
# 0.105ms at 300000). Swap one of them for the full census, or move any of them inside the poll
# loop, and the harness starts adding size-dependent work to the number it publishes. These pin
# the count.

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
    # GUARD. counts() is a dozen document-wide queries including getElementsByTagName("*"), and it
    # is 25 times the cost of messageCount() at 25000 chars and 24 times at 300000 (measured
    # 0.15ms/0.01ms and 2.51ms/0.105ms on Chromium). One of those per frame inside the re-open
    # window is the harness measuring itself.
    got = run_node(SCAN_BUDGET_BODY, REOPEN_SOURCES)
    assert got["censuses"] == 0, got


def test_the_reopen_window_scans_no_more_than_once_per_frame_of_the_rebuild() -> None:
    # GUARD. Six [data-role] passes for this rebuild on the virtual clock, and the six are
    # enumerable: `before` sizes the thread, one closes the unmount loop, three run the re-open
    # loop while the stub rebuild is still blocked, and `after` proves the thread came back. Only
    # the middle four are inside the recorder window and only one of them is inside the reported
    # `ms`. A second pass per iteration doubles the size-dependent work in the number, and at
    # 300000 characters each pass is 0.105ms against 0.01ms at 25000.
    got = run_node(SCAN_BUDGET_BODY, REOPEN_SOURCES)
    assert got["scans"] <= 6, got


def test_the_highlight_probe_stays_on_its_interval_inside_the_settle_loop() -> None:
    # GUARD. quietUntilIdle()'s probe is `pre code span` over the whole document: 0.03ms at 25000
    # chars and 0.465ms at 300000 on Chromium, 2.05ms on Firefox. Per frame that is an O(nodes)
    # query inside the window being timed, growing like the signal; on the interval it is a few
    # milliseconds across the whole settle.
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
    # GUARD. The menu content is portaled to the END of document.body, so a querySelector for it
    # walks the whole message list -- and for the entire open latency it walks it and finds
    # nothing. Measured on Chromium at 300000 chars that query is 0.25ms a call against 0.025ms at
    # 25000. One per mutation is 2 in the window; one per frame would be one per 16ms of the very
    # latency being measured, on a cost that grows with the thread.
    got = run_node(MENU_SCAN_BODY, MENU_SOURCES)
    assert got["querySelectorCalls"] <= got["notifications"] + 1, got


def test_the_menu_window_takes_one_census_and_not_one_per_frame() -> None:
    # GUARD. The tooltip-trigger census is the only querySelectorAll the menu window runs
    # (measured 0.13ms at 300000 chars against 0.005ms at 25000). It is taken once, under the
    # pointer, because an action bar that never mounts and one that is autohidden at rest are
    # otherwise indistinguishable.
    got = run_node(MENU_SCAN_BODY, MENU_SOURCES)
    assert got["querySelectorAllCalls"] <= 1, got


# ── the menu total carries two paint floors, not one ──────────────────

MENU_SOURCES = {"RECORDER_INIT": HARNESS.RECORDER_INIT, "MENU_JS": HARNESS.MENU_JS}

# A menu that opens and closes with NO work at all: the flag flips inside the dispatch. Whatever
# this reports is therefore pure floor.
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
    # settle() reads `open` before the MutationObserver callback that would have updated it has
    # been delivered, so its first true comparison is on the far side of an __nextPaint(). Nothing
    # is wrong with that -- it is what makes the number a wall-clock one -- but it means the metric
    # carries a floor, and the floor has to be subtracted before a ratio.
    got = run_node(MENU_BODY, MENU_SOURCES)
    assert got["openMs"] >= 32, got


def test_closing_the_menu_costs_a_second_double_raf() -> None:
    # And this is the one the single-floor subtraction missed: closing waits out its own
    # __nextPaint() for exactly the same reason, independently of opening.
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


# ── a null repetition is a failure, not a sample to drop ──────────────


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
    # Filtering the null out and taking the median of the rest silently changes the sample
    # population, and the verdict's `openMs is None` check then reads the median of the
    # repetitions that worked.
    summary = HARNESS.summarise(
        [menu_repetition(80.0), menu_repetition(None), menu_repetition(120.0)]
    )
    assert summary["menu"]["openMs"] is None, summary["menu"]


def test_a_timing_that_was_null_in_every_repetition_is_still_reported() -> None:
    # Present-and-None, not absent: a key that is missing entirely makes the verdict raise
    # KeyError instead of naming the action that never happened.
    summary = HARNESS.summarise([menu_repetition(None), menu_repetition(None)])
    assert "openMs" in summary["menu"] and summary["menu"]["openMs"] is None, summary["menu"]


def test_the_median_of_three_good_repetitions_is_unchanged() -> None:
    # The guard above must not cost the normal case its median.
    summary = HARNESS.summarise(
        [menu_repetition(80.0), menu_repetition(100.0), menu_repetition(120.0)]
    )
    assert summary["menu"]["openMs"] == 100.0, summary["menu"]


# ── the verdict must reject an action that never settled ──────────────


def clean_cell() -> dict:
    """One (engine, size) cell that harness_failures() has nothing to say about."""
    per_cycle = {"images": 3, "codeBlocks": 7}
    counts = {
        "messages": 20,
        "domNodes": 4000,
        "codeBlocks": 7,
        "highlightedTokens": 3000,
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
    cell = copy.deepcopy(clean_cell())
    cell["actions"]["jump"]["settleMs"] = None
    failures = HARNESS.harness_failures(results_with(cell), discriminating_report())
    assert any("jump action but it never reached a settled state" in f for f in failures), failures


# ── the per-repetition fixture ────────────────────────────────────────


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
    # The tool result panes ARE code -- two of the seven fences a content cycle produces -- and
    # Radix does not mount them until they are expanded. Gating on the highlighter first and
    # expanding second leaves a fresh batch of unhighlighted fences building inside the keystroke
    # measurement, which is the very next thing timed.
    log = repetition_log()
    expand = log.index(("evaluate", "expandTools"))
    highlight = log.index(("wait", "highlighting"))
    assert expand < highlight, log


def test_the_highlighter_gate_runs_before_the_first_measured_action() -> None:
    log = repetition_log()
    highlight = log.index(("wait", "highlighting"))
    assert highlight < log.index(("action", "KEYSTROKE_JS")), log


def test_the_deleted_message_is_restored_before_anything_else_is_measured() -> None:
    # The delete removes a message from the repository, permanently. At the smallest size the
    # whole thread is one ten-kind cycle, so three repetitions take the json fence, then both
    # inline images, then the svg -- and the fixture census that would have caught it was taken
    # before any repetition ran.
    log = repetition_log()
    assert ("evaluate", "restore") in log, log
    assert (
        log.index(("action", "DELETE_JS"))
        < log.index(("evaluate", "restore"))
        < log.index(("action", "REOPEN_JS"))
    ), log


def test_the_restored_thread_is_re_expanded_and_re_highlighted_before_re_open() -> None:
    # Restoring re-imports the thread, which unmounts every tool card and throws away every
    # highlighted fence. Re-opening straight afterwards would time that rebuild as well as its own.
    log = repetition_log()
    restore = log.index(("evaluate", "restore"))
    reopen = log.index(("action", "REOPEN_JS"))
    assert any(entry == ("wait", "highlighting") for entry in log[restore:reopen]), log
    assert any(entry == ("evaluate", "expandTools") for entry in log[restore:reopen]), log


def test_the_tool_expand_gate_is_not_satisfied_by_a_thread_of_closed_cards() -> None:
    # `collapsibleOutputs` is the CollapsibleContent element itself, and Radix keeps that element
    # in the tree for its collapse animation, so it is there while the card is shut. Measured on
    # this tree at 300000 characters, straight after seeding and before any expandTools() call:
    # collapsibleOutputs 22 of the 22 expected, codeExecutionPanes 0. A `collapsibleOutputs >= n`
    # gate is therefore satisfied by the closed thread, which is a wait that cannot fail -- and it
    # released the highlighter gate below it before the two fences per cycle it exists to sequence
    # had mounted. The pane's <pre> is a CHILD of that element: 0 collapsed, 22 expanded.
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


# ── the declared paint floor ──────────────────────────────────────────
#
# `growth()` subtracts `floored` double-rAF vsync floors from both ends of every ratio, and
# `floored` was a hand-declared integer per axis. Declaring it too low leaves a ~33ms constant in
# both ends, which drags the ratio towards 1 and can report a real curve as flat; declaring it too
# high subtracts time the action never spent. REOPEN_JS now counts the waits it actually pays, so
# the constant is checked against the run rather than trusted.


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


def test_reopen_declares_the_one_paint_floor_it_pays() -> None:
    """Reopen is driven by a React state update, so the count check straight after openThread()
    always still sees the unmounted tree and the loop always waits at least one paint."""
    assert HARNESS.declared_floor("reopen ms") == 1


def test_a_matching_floor_declaration_is_accepted() -> None:
    """The control. Without it the checks below could be met by rejecting every run."""
    assert HARNESS.floor_declaration_problems(floor_row(1)) == []


def test_a_floor_declared_below_the_waits_paid_is_rejected() -> None:
    problems = HARNESS.floor_declaration_problems(floor_row(2))
    assert len(problems) == 1 and "subtracts 1" in problems[0], problems


def test_a_floor_declared_above_the_waits_paid_is_rejected() -> None:
    problems = HARNESS.floor_declaration_problems(floor_row(0))
    assert len(problems) == 1 and "paid 0 paint wait" in problems[0], problems


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


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
