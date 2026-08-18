# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""What the model picker costs as a function of how many models are on the machine.

The axis is the size of the ON DEVICE list. The Hugging Face half of the picker is deliberately
not swept: it is paged behind an IntersectionObserver sentinel (pickers.tsx:4445-4490), so its row
count is bounded by how far the user scrolled rather than by how much they own. The downloaded
half has no such bound -- it renders every cached repo -- and that is what a user with a full
model cache grows.

studio/frontend/smoke-model-picker-scale.html mounts the REAL ModelSelector and answers
/api/hub/cached-models from an in-memory fixture, which is where the downloaded rows really come
from. Seeding only the `models` prop opened a panel with ZERO rows; that is in the harness's
comments because it is the sort of thing that reads as "the picker is fast" rather than as "the
picker is empty".

WHAT IS MEASURED

    open first row     setOpen(true) -> the FIRST row is in the panel.
    open converged     the same window, stopped when all N rows are there.
                       Both, deliberately: a change can move one and not the other, and a harness
                       with only the convergence predicate cannot see a first-paint win at all.
    open settle        how long after the open the panel stops moving. Three consecutive still
                       polls over (rows, panel nodes, scrollHeight) AND no frame over 33ms,
                       because a re-render that lands the same markup is invisible to a DOM gate.
    keystroke          one character into the picker's search box, measured to the next paint.
                       The query is debounced (useDebouncedValue, pickers.tsx:2430), so this is
                       the cost of the controlled input, and `search settle` below is the cost of
                       the filter that follows.
    search settle      from the keystroke until the filtered list has stopped moving.

CONTROLS

    control sleep ms   a 200ms setTimeout. Timer-bound: MUST STAY FLAT across the throttle ladder.
    control busy ms    a fixed integer loop. CPU-bound: MUST RISE with the rate.

Ratios are the result and milliseconds are context; this host runs at load average ~60.

THIS HARNESS MEASURES, IT DOES NOT GATE. It exits non-zero only when it is measuring nothing.

Run:
    python tests/studio/playwright_model_picker_scale.py
    SMOKE_BASE_URL=http://127.0.0.1:5475 SMOKE_PICKER_MODELS=50,200 \
        SMOKE_PICKER_RATES=1 python tests/studio/playwright_model_picker_scale.py
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _playwright_robust import (  # noqa: E402
    chromium_launch_args,
    start_vite,
    stop_process,
    wait_for_smoke_page,
)

PORT = int(os.environ.get("SMOKE_PORT", "5475"))
_EXTERNAL = os.environ.get("SMOKE_BASE_URL", "").strip().rstrip("/")
BASE = _EXTERNAL or f"http://127.0.0.1:{PORT}"
OWNS_SERVER = not _EXTERNAL
LABEL = os.environ.get("SMOKE_LABEL", "tree")
OUT = Path(os.environ.get("PW_ART_DIR", "logs/playwright-model-picker-scale"))
OUT.mkdir(parents = True, exist_ok = True)

# Sorted: the growth check reads the first and last as smallest and largest, so an unsorted
# override would invert every ratio and report a good run as measuring nothing.
SIZES = sorted(int(n) for n in os.environ.get("SMOKE_PICKER_MODELS", "50,200,1000").split(","))
RATES = [float(r) for r in os.environ.get("SMOKE_PICKER_RATES", "1,2,4").split(",") if r.strip()]
ENGINES = [
    e.strip() for e in os.environ.get("SMOKE_PICKER_ENGINES", "chromium").split(",") if e.strip()
]
REPEATS = int(os.environ.get("SMOKE_PICKER_REPEATS", "3"))
ACTION_TIMEOUT_MS = int(os.environ.get("SMOKE_ACTION_TIMEOUT_MS", "180000"))
SEED_TIMEOUT_MS = int(os.environ.get("SMOKE_SEED_TIMEOUT_MS", "300000"))
SETTLE_POLL_MS = int(os.environ.get("SMOKE_SETTLE_POLL_MS", "50"))
STILL_POLLS = int(os.environ.get("SMOKE_STILL_POLLS", "3"))
KEYSTROKES = int(os.environ.get("SMOKE_PICKER_KEYSTROKES", "5"))
CONTROL_SLEEP_MS = 200
CONTROL_BUSY_ITERS = 8_000_000
CONTROL_FLAT_TOLERANCE = 1.25
CONTROL_SCALE_FLOOR = 0.6
DISCRIMINATION_RATIO = float(os.environ.get("SMOKE_DISCRIMINATION_RATIO", "1.5"))
CONSOLE_WARNING_ALLOWANCE = int(os.environ.get("SMOKE_CONSOLE_WARNING_ALLOWANCE", "40"))

ACTIONS = ("open", "search")

RECORDER_INIT = """
(() => {
  const nativeRaf = window.requestAnimationFrame.bind(window);
  window.__nextPaint = () =>
    new Promise((resolve) => nativeRaf(() => nativeRaf(() => resolve())));
  const recorder = {
    running: false,
    generation: 0,
    frames: [],
    frameAt: [],
    startedAt: 0,
    begin() {
      this.running = true;
      this.generation += 1;
      const generation = this.generation;
      this.frames = [];
      this.frameAt = [];
      this.startedAt = performance.now();
      let lastFrame = performance.now();
      const frame = () => {
        // Which begin() this callback belongs to. `running` alone cannot answer that: a callback
        // scheduled by the previous action wakes to running === true and records the whole
        // between-action gap as a frame of the new window.
        if (generation !== this.generation) return;
        const now = performance.now();
        this.frames.push(now - lastFrame);
        this.frameAt.push(now);
        lastFrame = now;
        if (this.running) nativeRaf(frame);
      };
      nativeRaf(frame);
    },
    end(untilMs) {
      this.running = false;
      this.generation += 1;
      const cutoff = untilMs === undefined ? Infinity : untilMs;
      const frames = this.frames.filter((_, i) => this.frameAt[i] <= cutoff);
      return {
        frames: frames.length,
        worst_frame_ms: Math.round(Math.max(0, ...frames) * 10) / 10,
        frames_over_33: frames.filter((ms) => ms > 33).length,
      };
    },
  };
  window.__mps = recorder;

  /**
   * Three consecutive still polls, not two: a two-poll gate is a plateau detector and releases
   * inside the lull between two React commits. Stillness is the panel signature AND the absence
   * of any frame over 33ms since the previous poll, because a re-render that lands identical
   * markup is invisible to a DOM-only gate. Returns the time of the LAST CHANGE, so the grace is
   * not a constant added to both ends of every ratio.
   */
  window.__mpsSettle = async (timeoutMs, pollMs, stillPolls) => {
    const api = window.__pickerScale;
    const started = performance.now();
    const sample = () => {
      const c = api.counts();
      return `${c.rows}|${c.panelNodes}|${c.panelScrollHeight}`;
    };
    let last = sample();
    let lastChangeAt = performance.now();
    let still = 0;
    let framesSeen = window.__mps.frames.length;
    let lastPollAt = performance.now();
    while (performance.now() - started < timeoutMs) {
      await new Promise((resolve) => setTimeout(resolve, pollMs));
      const now = performance.now();
      const next = sample();
      const fresh = window.__mps.frames.slice(framesSeen);
      framesSeen = window.__mps.frames.length;
      if (next !== last || Math.max(0, ...fresh) > 33) {
        still = 0;
        last = next;
        lastChangeAt = now;
      } else {
        still += 1;
      }
      lastPollAt = now;
      if (still >= stillPolls) {
        return { settleMs: lastChangeAt - window.__mps.startedAt, at: lastChangeAt, timedOut: false };
      }
    }
    return { settleMs: null, at: lastPollAt, timedOut: true };
  };
})();
"""

OPEN_JS = """
async ([want, timeoutMs, pollMs, stillPolls]) => {
  const api = window.__pickerScale;
  api.setOpen(false);
  const closedStarted = performance.now();
  while (performance.now() - closedStarted < timeoutMs) {
    if (api.panel() === null) break;
    await window.__nextPaint();
  }
  if (api.panel() !== null) return null;
  window.__mps.begin();
  const started = performance.now();
  api.setOpen(true);
  let firstRowMs = null;
  let convergedMs = null;
  while (performance.now() - started < timeoutMs) {
    const rows = api.counts().rows;
    if (firstRowMs === null && rows > 0) firstRowMs = performance.now() - started;
    if (rows >= want) { convergedMs = performance.now() - started; break; }
    await window.__nextPaint();
  }
  const settled = await window.__mpsSettle(timeoutMs, pollMs, stillPolls);
  const metrics = window.__mps.end(settled.at);
  const counts = api.counts();
  return {
    firstRowMs: firstRowMs === null ? null : Math.round(firstRowMs * 10) / 10,
    convergedMs: convergedMs === null ? null : Math.round(convergedMs * 10) / 10,
    settleMs: settled.settleMs === null ? null : Math.round(settled.settleMs * 10) / 10,
    settleTimedOut: settled.timedOut,
    rows: counts.rows,
    panelNodes: counts.panelNodes,
    rowMenuTriggers: counts.rowMenuTriggers,
    tooltipTriggers: counts.tooltipTriggers,
    metrics,
  };
}
"""

SEARCH_JS = """
async ([count, timeoutMs, pollMs, stillPolls]) => {
  const api = window.__pickerScale;
  const input = api.searchInput();
  if (!input) return null;
  input.focus();
  // The native value setter plus an input event: what the browser leaves behind after a real
  // keypress, and what React's controlled input reacts to. input.value = x alone does not.
  const setValue = Object.getOwnPropertyDescriptor(
    HTMLInputElement.prototype, "value",
  ).set;
  const samples = [];
  const before = api.counts().rows;
  window.__mps.begin();
  const typed = "llama";
  for (let i = 0; i < count; i += 1) {
    await window.__nextPaint();
    const started = performance.now();
    setValue.call(input, typed.slice(0, i + 1));
    input.dispatchEvent(new Event("input", { bubbles: true }));
    await window.__nextPaint();
    samples.push(performance.now() - started);
  }
  // The filter is DEBOUNCED (useDebouncedValue, pickers.tsx:2430), so for ~300ms after the last
  // keystroke the list is still the unfiltered one and nothing in the DOM moves. A settle gate
  // alone therefore releases INSIDE the debounce -- measured: it released after ~150ms with the
  // row count still at its unfiltered 200, and the run's own check caught it as "the filter did
  // not run". So wait for the row count to actually change first, and report that wait as its own
  // number: it is what a user waits between typing and seeing the list narrow.
  let filteredMs = null;
  const filterStarted = performance.now();
  while (performance.now() - filterStarted < timeoutMs) {
    if (api.counts().rows !== before) {
      filteredMs = performance.now() - window.__mps.startedAt;
      break;
    }
    await new Promise((resolve) => setTimeout(resolve, 10));
  }
  const settled = await window.__mpsSettle(timeoutMs, pollMs, stillPolls);
  const metrics = window.__mps.end(settled.at);
  const after = api.counts().rows;
  // Read the proofs BEFORE clearing. `domQuery` was in the returned object literal, which is
  // evaluated after the two lines below, so it reported the CLEARED value and the run failed its
  // own "the keystrokes never reached React" check on a keystroke that had landed perfectly well.
  const domQuery = api.query();
  // Clear it again so the next repetition starts from the unfiltered list.
  setValue.call(input, "");
  input.dispatchEvent(new Event("input", { bubbles: true }));
  const sorted = samples.slice().sort((a, b) => a - b);
  return {
    samples: samples.map((s) => Math.round(s * 10) / 10),
    // The first sample is systematically the cold one, which is why the headline is a median.
    median_sample_ms: Math.round(sorted[Math.floor(sorted.length / 2)] * 10) / 10,
    filteredMs: filteredMs === null ? null : Math.round(filteredMs * 10) / 10,
    worst_sample_ms: Math.round(sorted[sorted.length - 1] * 10) / 10,
    settleMs: settled.settleMs === null ? null : Math.round(settled.settleMs * 10) / 10,
    settleTimedOut: settled.timedOut,
    rowsBefore: before,
    rowsAfter: after,
    domQuery,
    metrics,
  };
}
"""

PAINT_FLOOR_JS = """
async (samples) => {
  const values = [];
  for (let i = 0; i < samples; i += 1) {
    await window.__nextPaint();
    const started = performance.now();
    await window.__nextPaint();
    values.push(performance.now() - started);
  }
  values.sort((a, b) => a - b);
  return values[Math.floor(values.length / 2)];
}
"""

CONTROLS_JS = """
async ([sleepMs, busyIters]) => {
  const sleeps = [];
  const busies = [];
  for (let i = 0; i < 3; i += 1) {
    const t0 = performance.now();
    await new Promise((resolve) => setTimeout(resolve, sleepMs));
    sleeps.push(performance.now() - t0);
    const t1 = performance.now();
    let acc = 0;
    for (let k = 0; k < busyIters; k += 1) acc = (acc + k) % 2147483647;
    busies.push(performance.now() - t1);
    if (acc === -1) console.log("unreachable");
  }
  const mid = (a) => a.slice().sort((x, y) => x - y)[1];
  return {
    control_sleep_ms: Math.round(mid(sleeps) * 10) / 10,
    control_busy_ms: Math.round(mid(busies) * 10) / 10,
  };
}
"""


def info(message: str) -> None:
    print(f"[model-picker-scale] {message}", flush = True)


def median(values: list) -> float | None:
    """None if ANY repetition failed: a null is a repetition in which the thing being timed never
    happened, and dropping those changes the population and hides it from the failure check."""
    if not values or any(v is None for v in values):
        return None
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return round(ordered[middle], 1)
    return round((ordered[middle - 1] + ordered[middle]) / 2, 1)


def measure_cell(context, engine: str, size: int, rate: float) -> dict:
    page = context.new_page()
    result: dict = {"models_requested": size, "engine": engine, "cpu_throttle_rate": 1.0}
    api_prefix = f"{BASE}/api/"
    stray: list = []
    warnings: list = []
    # Severity kept SEPARATE from the warnings: an application exception is not engine chatter,
    # and folding the two into one list is how a real error sits under a warning threshold.
    errors: list = []
    page.on("request", lambda r: stray.append(r.url) if r.url.startswith(api_prefix) else None)
    page.on(
        "console",
        lambda m: (
            errors.append(m.text[:200])
            if m.type == "error"
            else warnings.append(m.text[:200])
            if m.type == "warning"
            else None
        ),
    )
    page.on("pageerror", lambda e: errors.append(f"pageerror: {e}"[:200]))
    cdp = None
    try:
        page.goto(f"{BASE}/smoke-model-picker-scale.html", wait_until = "domcontentloaded")
        page.wait_for_function("() => Boolean(window.__pickerScale)", timeout = 120_000)
        if engine == "chromium":
            cdp = context.new_cdp_session(page)

        # Seeding is unthrottled and untimed: this measures interaction cost at a list size, not
        # the cost of building the fixture.
        result["plan"] = page.evaluate("(n) => window.__pickerScale.seed(n)", size)
        page.evaluate("window.__pickerScale.setOpen(true)")
        page.wait_for_function(
            "() => window.__pickerScale.panel() !== null", timeout = SEED_TIMEOUT_MS
        )
        # The downloaded list lives behind the On Device tab; the picker opens on Recommended,
        # which is the Hugging Face listing and is empty here on purpose.
        page.wait_for_function(
            "() => window.__pickerScale.onDeviceTab() !== null", timeout = SEED_TIMEOUT_MS
        )
        page.evaluate("() => window.__pickerScale.onDeviceTab().click()")
        page.wait_for_function(
            "(n) => window.__pickerScale.counts().rows >= n", arg = size, timeout = SEED_TIMEOUT_MS
        )
        result["counts"] = page.evaluate("window.__pickerScale.counts()")
        result["seed_api_requests"] = len(stray)
        result["seed_console_errors"] = len(errors)
        result["first_seed_error"] = errors[0] if errors else "-"
        result["seed_console_warnings"] = len(warnings)
        result["first_seed_warning"] = warnings[0] if warnings else "-"
        stray.clear()
        errors.clear()
        warnings.clear()

        if cdp is not None and rate != 1.0:
            cdp.send("Emulation.setCPUThrottlingRate", {"rate": rate})
            result["cpu_throttle_rate"] = rate
        result["paint_floor_ms"] = round(page.evaluate(PAINT_FLOOR_JS, 9), 2)
        result.update(page.evaluate(CONTROLS_JS, [CONTROL_SLEEP_MS, CONTROL_BUSY_ITERS]))

        reps = []
        for index in range(REPEATS):
            info(f"  {engine} {size} models @{rate}x: repetition {index + 1}/{REPEATS}")
            rep: dict = {}
            rep["open"] = page.evaluate(
                OPEN_JS, [size, ACTION_TIMEOUT_MS, SETTLE_POLL_MS, STILL_POLLS]
            )
            # OPEN_JS reopens on Recommended, so the tab has to be reselected before the search.
            page.evaluate(
                """() => {
                    const tab = window.__pickerScale.onDeviceTab();
                    if (tab) tab.click();
                }"""
            )
            page.wait_for_function(
                "(n) => window.__pickerScale.counts().rows >= n",
                arg = size,
                timeout = ACTION_TIMEOUT_MS,
            )
            rep["search"] = page.evaluate(
                SEARCH_JS, [KEYSTROKES, ACTION_TIMEOUT_MS, SETTLE_POLL_MS, STILL_POLLS]
            )
            page.wait_for_function(
                "(n) => window.__pickerScale.counts().rows >= n",
                arg = size,
                timeout = ACTION_TIMEOUT_MS,
            )
            reps.append(rep)
        if cdp is not None and rate != 1.0:
            cdp.send("Emulation.setCPUThrottlingRate", {"rate": 1})
        result["raw_repetitions"] = reps
        result["repetitions"] = REPEATS
        result["actions"] = summarise(reps)
        result["stray_api_requests"] = len(stray)
        result["stray_api_urls"] = sorted({u[len(BASE) :] for u in stray})[:8]
        result["console_errors"] = len(errors)
        result["first_console_error"] = errors[0] if errors else "-"
        result["console_warnings"] = len(warnings)
        result["first_console_warning"] = warnings[0] if warnings else "-"
    finally:
        page.close()
    return result


HEADLINE = {"open": "convergedMs", "search": "median_sample_ms"}


def summarise(reps: list) -> dict:
    out: dict = {}
    for action in ACTIONS:
        rows = [r.get(action) for r in reps]
        if any(r is None for r in rows):
            out[action] = {"ran": False}
            continue
        merged: dict = {"ran": True, "repetitions": len(rows)}
        keys = set()
        for row in rows:
            for key, value in row.items():
                # `value is None` is deliberately a key too: a timing that was null in every
                # repetition would otherwise be absent, and the failure check would KeyError
                # instead of reporting the action that never happened.
                if value is None or (
                    isinstance(value, (int, float)) and not isinstance(value, bool)
                ):
                    keys.add(key)
        for key in sorted(keys):
            merged[key] = median([r.get(key) for r in rows])
        for key, _pick in (("metrics", None),):
            for metric in ("worst_frame_ms", "frames_over_33", "frames"):
                merged[metric] = median([r[key][metric] for r in rows])
        merged["settle_timed_out"] = any(bool(r.get("settleTimedOut")) for r in rows)
        merged["domQuery"] = rows[-1].get("domQuery")
        merged["per_repetition"] = [r.get(HEADLINE[action]) for r in rows]
        out[action] = merged
    return out


def run() -> dict:
    results: dict = {
        "label": LABEL,
        "base": BASE,
        "sizes": SIZES,
        "engines": ENGINES,
        "rates": RATES,
        "repetitions": REPEATS,
        "by_engine": {},
    }
    with sync_playwright() as p:
        for engine in ENGINES:
            launcher = getattr(p, engine)
            kwargs = {"headless": os.environ.get("SMOKE_HEADLESS", "1") == "1"}
            if engine == "chromium":
                kwargs["args"] = chromium_launch_args()
            browser = launcher.launch(**kwargs)
            context = browser.new_context(viewport = {"width": 1440, "height": 900})
            context.add_init_script(RECORDER_INIT)
            context.route(
                re.compile(rf"^{re.escape(BASE)}/api/"),
                lambda route: route.fulfill(
                    status = 200, content_type = "application/json", body = "{}"
                ),
            )
            by_rate: dict = {}
            engine_rates = RATES if engine == "chromium" else [1.0]
            for rate in engine_rates:
                by_size: dict = {}
                for size in SIZES:
                    info(f"measuring {engine} at {size} models, cpu rate {rate}x")
                    try:
                        by_size[str(size)] = measure_cell(context, engine, size, rate)
                    except Exception as exc:  # noqa: BLE001 - the message is the whole point
                        info(f"CRASHED {engine} {size} @{rate}x: {type(exc).__name__}: {exc}")
                        by_size[str(size)] = {
                            "models_requested": size,
                            "engine": engine,
                            "crashed": f"{type(exc).__name__}: {exc}"[:400],
                        }
                by_rate[str(rate)] = {"by_size": by_size}
            results["by_engine"][engine] = {"version": browser.version, "by_rate": by_rate}
            context.close()
            browser.close()
    return results


def _action(action: str, key: str):
    return lambda r: r["actions"][action][key]


TABLE_ROWS = (
    ("models requested", lambda r: r["models_requested"]),
    ("models seeded", lambda r: r["plan"]["models"]),
    ("rows rendered", lambda r: r["counts"]["rows"]),
    ("row menu triggers", lambda r: r["counts"]["rowMenuTriggers"]),
    ("tooltip triggers", lambda r: r["counts"]["tooltipTriggers"]),
    ("panel dom nodes", lambda r: r["counts"]["panelNodes"]),
    ("document dom nodes", lambda r: r["counts"]["domNodes"]),
    ("cpu throttle rate", lambda r: r["cpu_throttle_rate"]),
    ("CONTROL sleep ms (flat)", lambda r: r["control_sleep_ms"]),
    ("CONTROL busy ms (scales)", lambda r: r["control_busy_ms"]),
    ("paint floor ms", lambda r: r["paint_floor_ms"]),
    ("seed api requests", lambda r: r["seed_api_requests"]),
    ("action api requests", lambda r: r["stray_api_requests"]),
    ("action api urls", lambda r: "|".join(r["stray_api_urls"]) or "-"),
    ("seed console errors", lambda r: r["seed_console_errors"]),
    ("first seed error", lambda r: r["first_seed_error"]),
    ("action console errors", lambda r: r["console_errors"]),
    ("first action error", lambda r: r["first_console_error"]),
    ("action console warnings", lambda r: r["console_warnings"]),
    ("first action warning", lambda r: r["first_console_warning"]),
    ("open ran", _action("open", "ran")),
    ("open first row ms", _action("open", "firstRowMs")),
    ("open converged ms", _action("open", "convergedMs")),
    ("open settle ms", _action("open", "settleMs")),
    ("open worst frame ms", _action("open", "worst_frame_ms")),
    ("open frames over 33ms", _action("open", "frames_over_33")),
    ("open rows after", _action("open", "rows")),
    (
        "open converged per repetition",
        lambda r: "/".join(str(v) for v in r["actions"]["open"]["per_repetition"]),
    ),
    ("search ran", _action("search", "ran")),
    ("search median keystroke ms", _action("search", "median_sample_ms")),
    ("search worst keystroke ms", _action("search", "worst_sample_ms")),
    ("search filtered ms", _action("search", "filteredMs")),
    ("search settle ms", _action("search", "settleMs")),
    ("search worst frame ms", _action("search", "worst_frame_ms")),
    ("search frames over 33ms", _action("search", "frames_over_33")),
    ("search rows before", _action("search", "rowsBefore")),
    ("search rows after", _action("search", "rowsAfter")),
    ("search dom query", _action("search", "domQuery")),
    (
        "search keystroke per repetition",
        lambda r: "/".join(str(v) for v in r["actions"]["search"]["per_repetition"]),
    ),
)

# The third field is how many double-rAF waits the metric is clocked across; each carries its own
# ~33ms vsync floor, and left in it compresses every ratio towards 1.
GROWTH_AXES = (
    ("open first row ms", _action("open", "firstRowMs"), 1),
    ("open converged ms", _action("open", "convergedMs"), 1),
    ("open settle ms", _action("open", "settleMs"), 0),
    ("search median keystroke ms", _action("search", "median_sample_ms"), 1),
    ("search filtered ms", _action("search", "filteredMs"), 0),
    ("search settle ms", _action("search", "settleMs"), 0),
    ("open worst frame ms", _action("open", "worst_frame_ms"), 0),
    ("search worst frame ms", _action("search", "worst_frame_ms"), 0),
)


def print_table(results: dict) -> None:
    columns = [
        (engine, rate, str(size))
        for engine in results["engines"]
        for rate in results["by_engine"][engine]["by_rate"]
        for size in results["sizes"]
    ]
    rows = []
    for name, pick in TABLE_ROWS:
        cells = []
        for engine, rate, size in columns:
            try:
                value = pick(results["by_engine"][engine]["by_rate"][rate]["by_size"][size])
                cells.append("-" if value is None else str(value))
            except (KeyError, TypeError):
                cells.append("-")
        rows.append((name, cells))
    label_width = max(len(n) for n, _ in rows) + 2
    headers = [f"{e[:4]}{float(r):g}x/{s}" for e, r, s in columns]
    width = max([len(c) for _, cs in rows for c in cs] + [len(h) for h in headers]) + 2
    header = "".ljust(label_width) + "".join(h.rjust(width) for h in headers)
    info(header)
    info("-" * len(header))
    for name, cells in rows:
        info(name.ljust(label_width) + "".join(c.rjust(width) for c in cells))


def report_growth(results: dict) -> dict:
    report: dict = {}
    for engine in results["engines"]:
        for rate, bucket in results["by_engine"][engine]["by_rate"].items():
            cells = bucket["by_size"]
            per_axis: dict = {}
            for name, pick, floored in GROWTH_AXES:
                try:
                    small_row = cells[str(results["sizes"][0])]
                    large_row = cells[str(results["sizes"][-1])]
                    small = pick(small_row)
                    large = pick(large_row)
                except (KeyError, TypeError):
                    small = large = None
                if small is None or large is None:
                    per_axis[name] = {
                        "small": None,
                        "large": None,
                        "ratio": None,
                        "discriminated": False,
                        "reason": "not recorded",
                    }
                    continue
                if floored:
                    small -= floored * small_row["paint_floor_ms"]
                    large -= floored * large_row["paint_floor_ms"]
                small = round(small, 2)
                large = round(large, 2)
                if small <= 0:
                    per_axis[name] = {
                        "small": small,
                        "large": large,
                        "ratio": None,
                        "discriminated": False,
                        "reason": (
                            "at or under the paint floor at the smallest size, so this axis has "
                            "no room to move"
                        ),
                    }
                    continue
                ratio = round(large / small, 2)
                per_axis[name] = {
                    "small": small,
                    "large": large,
                    "ratio": ratio,
                    "discriminated": ratio > DISCRIMINATION_RATIO,
                    "reason": "-",
                }
            report[f"{engine}@{rate}x"] = per_axis
    return report


def print_growth(results: dict, report: dict) -> None:
    for key, per_axis in report.items():
        info("")
        info(
            f"growth on {key} ({results['sizes'][0]} -> {results['sizes'][-1]} models, median of "
            f"{results['repetitions']} repetitions)"
        )
        for name, row in per_axis.items():
            mark = "DISCRIMINATES" if row["discriminated"] else "flat"
            if row["ratio"] is None:
                info(
                    f"  {name:<32} {row['small']!s:>10} -> {row['large']!s:>10}       -  "
                    f"{mark} ({row['reason']})"
                )
                continue
            info(
                f"  {name:<32} {row['small']:>10} -> {row['large']:>10}  {row['ratio']:>7.2f}x  "
                f"{mark}"
            )


def print_ladder(results: dict) -> None:
    for engine in results["engines"]:
        buckets = results["by_engine"][engine]["by_rate"]
        base = buckets.get("1.0") or buckets.get("1")
        if base is None or len(buckets) < 2:
            continue
        info("")
        info(f"cpu throttling ladder on {engine} (ratio against each size's own 1x)")
        for size in results["sizes"]:
            info(f"  {size} models")
            base_row = base["by_size"].get(str(size), {})
            for rate, bucket in buckets.items():
                row = bucket["by_size"].get(str(size), {})
                if "crashed" in row or "crashed" in base_row:
                    info(f"    {rate:>5}x  crashed")
                    continue

                def ratio(pick):
                    try:
                        a = pick(base_row)
                        b = pick(row)
                    except (KeyError, TypeError):
                        return None
                    if not a or b is None:
                        return None
                    return round(b / a, 2)

                info(
                    f"    {rate:>5}x  "
                    f"CONTROL sleep {ratio(lambda r: r['control_sleep_ms'])}x  "
                    f"CONTROL busy {ratio(lambda r: r['control_busy_ms'])}x  "
                    f"open converged {ratio(_action('open', 'convergedMs'))}x  "
                    f"search keystroke {ratio(_action('search', 'median_sample_ms'))}x  "
                    f"search settle {ratio(_action('search', 'settleMs'))}x"
                )


def harness_failures(results: dict, report: dict) -> list:
    failures: list = []
    for engine in results["engines"]:
        buckets = results["by_engine"][engine]["by_rate"]
        base = buckets.get("1.0") or buckets.get("1")
        for rate, bucket in buckets.items():
            for size in results["sizes"]:
                row = bucket["by_size"][str(size)]
                where = f"{engine} at {size} models @{rate}x"
                if "crashed" in row:
                    failures.append(f"{where} crashed: {row['crashed']}")
                    continue
                if row["stray_api_requests"]:
                    failures.append(
                        f"{where} let {row['stray_api_requests']} /api/ request(s) reach the "
                        f"network during the measured actions ({row['stray_api_urls']}); the "
                        "timings include a round trip"
                    )
                for phase, count, first in (
                    ("seeding", row["seed_console_errors"], row["first_seed_error"]),
                    ("the measured actions", row["console_errors"], row["first_console_error"]),
                ):
                    if count:
                        failures.append(
                            f"{where} logged {count} console/page error(s) during {phase}, the "
                            f"first being {first!r}; an application exception is not engine "
                            "chatter and the timings around it are not measurements"
                        )
                if row["console_warnings"] > CONSOLE_WARNING_ALLOWANCE:
                    failures.append(
                        f"{where} logged {row['console_warnings']} console warnings during the "
                        f"measured actions, the first being {row['first_console_warning']!r}"
                    )
                counts = row["counts"]
                if counts["rows"] < size:
                    failures.append(
                        f"{where} rendered {counts['rows']} of {size} rows; the seed did not land"
                    )
                for action in ACTIONS:
                    if not row["actions"][action].get("ran"):
                        failures.append(f"{where} could not run the {action} action at all")
                        continue
                    if row["actions"][action].get("settle_timed_out"):
                        failures.append(
                            f"{where} ran the {action} action but it never held still, so its "
                            "settle time is the timeout rather than a measurement"
                        )
                opened = row["actions"]["open"]
                if opened.get("ran"):
                    if opened["firstRowMs"] is None:
                        failures.append(f"{where} opened the picker and no row ever appeared")
                    if opened["convergedMs"] is None:
                        failures.append(f"{where} opened the picker and it never reached {size}")
                searched = row["actions"]["search"]
                if searched.get("ran"):
                    # The typed characters have to have reached the input, or the whole column is
                    # the paint floor with a plausible-looking number on it.
                    if not searched.get("domQuery"):
                        failures.append(
                            f"{where} typed into the picker search and the input stayed empty; "
                            "the keystrokes never reached React"
                        )
                    # And they have to have CHANGED something, or the filter was never exercised.
                    if searched["rowsAfter"] is not None and searched["rowsBefore"] is not None:
                        if searched["rowsAfter"] >= searched["rowsBefore"]:
                            failures.append(
                                f"{where} typed a query and the row count did not fall "
                                f"({searched['rowsBefore']} -> {searched['rowsAfter']}); the "
                                "filter did not run, so its column is timing nothing"
                            )
            if base is not None and float(rate) > 1.0:
                for size in results["sizes"]:
                    row = bucket["by_size"][str(size)]
                    base_row = base["by_size"][str(size)]
                    if "crashed" in row or "crashed" in base_row:
                        continue
                    drift = row["control_sleep_ms"] / (base_row["control_sleep_ms"] or 1)
                    if drift > CONTROL_FLAT_TOLERANCE or drift < 1 / CONTROL_FLAT_TOLERANCE:
                        failures.append(
                            f"{engine} at {size} models: the timer-bound CONTROL moved "
                            f"{drift:.2f}x between 1x and {rate}x; it has to stay flat, so no "
                            "ratio in this run is readable"
                        )
                    realised = row["control_busy_ms"] / (base_row["control_busy_ms"] or 1)
                    if realised < float(rate) * CONTROL_SCALE_FLOOR:
                        failures.append(
                            f"{engine} at {size} models: the CPU-bound control only slowed "
                            f"{realised:.2f}x at a requested {rate}x; the throttling did not land"
                        )
    if len(results["sizes"]) >= 2:
        for key, per_axis in report.items():
            if not any(row["discriminated"] for row in per_axis.values()):
                failures.append(
                    f"on {key} no measured axis rose by more than {DISCRIMINATION_RATIO}x from "
                    f"{results['sizes'][0]} to {results['sizes'][-1]} models; the numbers above "
                    "cannot size any change"
                )
    return failures


def main() -> int:
    vite = None
    if OWNS_SERVER:
        info(f"starting vite dev server on port {PORT}")
        vite = start_vite(PORT)
    try:
        wait_for_smoke_page(
            f"{BASE}/smoke-model-picker-scale.html",
            "smoke-model-picker-scale-main.tsx",
            proc = vite,
            info = info,
        )
        results = run()
    finally:
        if vite is not None:
            stop_process(vite)
            info("vite stopped")

    report = report_growth(results)
    results["growth"] = report
    out = OUT / f"{LABEL}.json"
    out.write_text(json.dumps(results, indent = 2), encoding = "utf-8")
    print_table(results)
    print_growth(results, report)
    print_ladder(results)
    info(f"wrote {out}")
    failures = harness_failures(results, report)
    for problem in failures:
        info(f"HARNESS-BROKEN {problem}")
    if failures:
        return 1
    info("measurement only: no budgets are asserted here, so this exits 0 on any timing.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
