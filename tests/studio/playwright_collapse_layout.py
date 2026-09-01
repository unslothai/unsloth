# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""What a collapsible toggle costs, and why that cost is a property of the WHOLE document.

Radix's `CollapsibleContentImpl` runs a layout effect on every `open` change that writes
`transitionDuration: 0s` and `animationName: none`, reads `getBoundingClientRect()`, then writes
the styles back. The read is a synchronous layout, and it is UNCONDITIONAL: it does not ask
whether any stylesheet consumes the `--radix-collapsible-content-height` it publishes. Blink's
layout is O(total layout objects) rather than O(dirty objects), so that one read is charged for
walking the entire document, not the pane that changed.

This drives studio/frontend/smoke-collapse-layout.html, whose design point is that THE PANE'S OWN
CONTENT IS IDENTICAL IN EVERY RUN. Only the filler around it changes size. So a toggle that gets
more expensive as `fillers` grows cannot be paying for the pane, and the extra cost has nowhere to
come from except the rest of the document.

The four arms and what each is for:

    radix-height     Radix content plus the `animate-collapsible-*` height keyframes. Today's
                     mechanism, and the baseline the other three are read against.
    radix-grid       Radix content plus a `grid-template-rows: 0fr -> 1fr` transition, with nothing
                     consuming `--radix-collapsible-content-height`. This arm exists TO BE
                     DISPROVED. If it forces the same full-document layout as radix-height, then
                     swapping the keyframes is not the fix, because the measurement Radix does is
                     not conditional on anything reading its result.
    unmeasured-grid  The local `UnmeasuredCollapsible` plus the same grid transition, and no
                     measurement at all. This is the arm that should not scale.
    reasoning        The real reasoning primitives, which follow GRID_COLLAPSE_REASONING_ENABLED.
                     Run the page once per flag value for the real before/after rather than a
                     model of it; the flag is a build-time constant, so this driver records the
                     value it found in the checkout it served rather than choosing one.

The documented expectation, stated here as what the arms are FOR and deliberately NOT asserted
below: radix-height and radix-grid both force the full-document layout, unmeasured-grid does not.

Two independent numbers are collected per cell, because each covers the other's blind spot:

    tracing          `Layout` trace events carry `args.beginData.dirtyObjects`, `totalObjects` and
                     `partialLayout` alongside the event's own duration, which is the only place
                     the "milliseconds of layout for a handful of dirty objects against a
                     six-figure tree" shape is visible at all.
    Performance      `Performance.getMetrics` before and after the toggles gives `LayoutCount` and
                     `LayoutDuration`, which are cheap, stable, and cannot be thrown off by a
                     trace category being renamed upstream. If the two disagree, the trace parse
                     is the side to suspect.

READ THE FORCED COLUMN, NOT THE TOTAL. Both mechanisms animate a property that layout depends on,
`height` on one side and `grid-template-rows` on the other, so both lay out on every animation
frame and the raw layout count mostly counts frames. What separates the arms is the layout Blink
did SYNCHRONOUSLY because script asked for a geometry it had invalidated, and the trace names those
exactly: only a script-forced layout carries `args.beginData.stackTrace`, and on the Radix arms
that stack reads `commitHookLayoutEffects` into the collapsible's `getBoundingClientRect`. The
report keeps both, and the forced count is the one the arms differ on.

One property of the page worth knowing before reading its radix-grid row: that arm barely animates.
Radix's presence machinery keys off animation events, and the arm has only a transition, so the
pane unmounts immediately on close and mounts already at `1fr` on open. Its total layout count is
therefore small for a reason that has nothing to do with the measurement, which is a second reason
to read the forced column there.

The last thing each cell does is grow the content of an OPEN pane and check that the rendered
height followed it, which is the case a height captured at toggle time gets wrong and `1fr` gets
right by construction. Note what that check can and cannot see here: the height keyframes carry
`fill-mode-forwards` only on close, so once the open animation ends the pane is back to its
natural height and growth after that point is followed on every arm. The clipping this looks for
is what a captured height that OUTLIVED its animation would produce, so a clean row is the
expected reading and a dirty one is a real regression, not the other way round.

There is no performance budget here on purpose. This is a measurement probe, and layout timings
vary by more across machines than the effect being measured varies across a healthy tree, so a
hard threshold would be flaky in CI and would be tuned away rather than fixed. It exits non-zero
only for a harness failure: a page that never became ready, an arm that rendered nothing, a filler
parameter that was ignored, a trace that captured no layouts at all.

Run:
    python tests/studio/playwright_collapse_layout.py
    python tests/studio/playwright_collapse_layout.py --arm radix-grid --fillers 5000
    python tests/studio/playwright_collapse_layout.py --json > collapse-layout.json

It starts and stops its own vite dev server. Point it at one you already have with
SMOKE_BASE_URL, or move the port it picks with SMOKE_PORT.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from playwright.sync_api import sync_playwright

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _playwright_robust import (  # noqa: E402
    FRONTEND,
    chromium_launch_args,
    start_vite,
    stop_process,
    wait_for_smoke_page,
)

PORT = int(os.environ.get("SMOKE_PORT", "5217"))
# Unset: start and stop our own server.
_EXTERNAL = os.environ.get("SMOKE_BASE_URL", "").strip()
BASE = _EXTERNAL or f"http://127.0.0.1:{PORT}"
OWNS_SERVER = not _EXTERNAL
# Under logs/ like every sibling harness;
OUT = Path(os.environ.get("PW_ART_DIR", "logs/playwright-collapse-layout"))
LABEL = "collapse-layout"
SMOKE_PAGE = "smoke-collapse-layout.html"
SMOKE_ENTRY = "smoke-collapse-layout-main.tsx"

ARMS = ("radix-height", "radix-grid", "unmeasured-grid", "reasoning")

# The flag the `reasoning` arm follows is a build-time constant, so the page cannot report it.
FLAG_SOURCE = FRONTEND / "src" / "components" / "assistant-ui" / "thread-feature-flags.ts"

# Two document sizes, because the claim is about SCALING and a single size cannot show it. The
DEFAULT_FILLERS = (200, 5000)
DEFAULT_PANE_PARAGRAPHS = int(os.environ.get("SMOKE_COLLAPSE_PANE_PARAGRAPHS", "40"))
DEFAULT_CYCLES = int(os.environ.get("SMOKE_COLLAPSE_CYCLES", "6"))
# Unthrottled by default: unlike a streaming render, a forced full-document layout is expensive on any machine, so
DEFAULT_THROTTLE = int(os.environ.get("SMOKE_COLLAPSE_THROTTLE", "1"))

# The pane animates for 200ms. Everything below waits past that, so a measurement never lands
# while the transition is still running and the arms are never compared at different points of it.
SETTLE_MS = 400
# Deliberately INSIDE the 200ms, for the mid-flight growth check below.
MIDFLIGHT_GROW_DELAY_MS = 60
GROW_PARAGRAPHS = 20
# Enough of a forced-layout stack to reach the frame that names the effect it ran from.
STACK_FRAMES_KEPT = 4
# A five-thousand-row page is a real render in a dev server on a cold module graph.
READY_TIMEOUT_MS = 180_000

# `Layout` is emitted under `devtools.timeline`;
# The parse below matches on the event NAME and records whatever category it arrived under, so an upstream
# recategorisation shows up in the report as a changed category rather than as a silent zero.
TRACE_CATEGORIES = (
    "devtools.timeline",
    "disabled-by-default-devtools.timeline",
    "disabled-by-default-devtools.timeline.stack",
    "blink.user_timing",
)

# Clicking through Playwright would put its own hit-testing, and the layout reads that come with it, inside the trace
TOGGLE_CYCLES_JS = """
async ([cycles, settleMs]) => {
  const trigger = () => document.querySelector('[data-probe="trigger"]');
  const wait = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
  let clicks = 0;
  for (let i = 0; i < cycles; i += 1) {
    trigger().click();
    clicks += 1;
    await wait(settleMs);
    trigger().click();
    clicks += 1;
    await wait(settleMs);
  }
  return { clicks, state: trigger().getAttribute("data-state") };
}
"""

CLICK_TRIGGER_JS = """
() => {
  const trigger = document.querySelector('[data-probe="trigger"]');
  if (!trigger) {
    throw new Error('no [data-probe="trigger"] on the page');
  }
  trigger.click();
}
"""

# The arm mounted, and the hooks App installs in its effect are there to drive it.
ARM_DRIVABLE_JS = """
() => Boolean(document.querySelector('[data-probe="trigger"]'))
    && typeof window.__probeGrow === "function"
    && typeof window.__probeReset === "function"
"""

# `overflowPx` is the whole streaming question in one number:
PANE_SNAPSHOT_JS = """
() => {
  const content = document.querySelector('[data-probe="content"]');
  if (!content) {
    return null;
  }
  const rect = content.getBoundingClientRect();
  const paragraphs = content.querySelectorAll("p");
  const last = paragraphs[paragraphs.length - 1];
  const round1 = (value) => Math.round(value * 10) / 10;
  return {
    height: round1(rect.height),
    paragraphs: paragraphs.length,
    overflowPx: last ? round1(last.getBoundingClientRect().bottom - rect.bottom) : null,
    state: content.getAttribute("data-state"),
    contentClass: String(content.className || ""),
  };
}
"""

_LOG = sys.stdout


def info(message: str) -> None:
    print(f"[{LABEL}] {message}", file = _LOG, flush = True)


def reasoning_flag_in_source() -> bool | None:
    """GRID_COLLAPSE_REASONING_ENABLED as written in the checkout this harness serves.

    None when the file cannot be read, which is the honest answer under SMOKE_BASE_URL: the
    server is then someone else's tree and this one says nothing about it.
    """
    # SMOKE_BASE_URL serves a bundle built somewhere else, so the local working tree is not evidence about it.
    if _EXTERNAL:
        return None
    try:
        text = FLAG_SOURCE.read_text(encoding = "utf-8")
    except OSError:
        return None
    found = re.search(r"GRID_COLLAPSE_REASONING_ENABLED\s*=\s*(true|false)", text)
    return None if found is None else found.group(1) == "true"


def metrics(cdp) -> dict[str, float]:
    got = cdp.send("Performance.getMetrics")
    return {m["name"]: m["value"] for m in got["metrics"]}


def delta(before: dict[str, float], after: dict[str, float], name: str) -> float:
    return round(after.get(name, 0.0) - before.get(name, 0.0), 4)


def start_tracing(cdp) -> list[dict]:
    """Begin a trace and return the list the `dataCollected` batches will land in.

    ReportEvents mode delivers nothing until `Tracing.end`, so the list stays empty until
    `stop_tracing` has pumped the events through.
    """
    collected: list[dict] = []
    cdp.on("Tracing.dataCollected", lambda payload: collected.extend(payload.get("value") or []))
    cdp.send(
        "Tracing.start",
        {
            "transferMode": "ReportEvents",
            "traceConfig": {
                "recordMode": "recordAsMuchAsPossible",
                "includedCategories": list(TRACE_CATEGORIES),
            },
        },
    )
    return collected


def stop_tracing(
    cdp,
    page,
    collected: list[dict],
    *,
    timeout_s: float = 60.0,
) -> list[dict]:
    """End the trace and block until Chromium says it has handed over every batch.

    Reading `collected` straight after `Tracing.end` returns whatever happened to have arrived,
    which on a large trace is a truncated one, and a truncated trace is indistinguishable from a
    cheap toggle. The wait is a `wait_for_timeout` rather than a `sleep` because the sync API only
    dispatches CDP events while it is inside a call into the driver.
    """
    done = {"seen": False}
    cdp.on("Tracing.tracingComplete", lambda _payload: done.__setitem__("seen", True))
    cdp.send("Tracing.end")
    deadline = time.monotonic() + timeout_s
    while not done["seen"] and time.monotonic() < deadline:
        page.wait_for_timeout(50)
    if not done["seen"]:
        raise RuntimeError(f"the trace never completed within {timeout_s}s")
    return collected


def _frame_label(frame: dict) -> str:
    """`functionName file.js:line`, with the dev server's URL and cache-busting query dropped.

    The full URL is three quarters host and `?v=` hash, which pushes the part that identifies the
    code off the end of any line the table prints.
    """
    url = str(frame.get("url") or "?").split("?")[0].rsplit("/", 1)[-1]
    return f"{frame.get('functionName') or '(anonymous)'} {url}:{frame.get('lineNumber', '?')}"


def summarise_layouts(events: list[dict]) -> dict:
    """Fold the `Layout` events into the numbers this probe is about.

    `totalObjects` is the size of the tree under the relayout root and `dirtyObjects` the count
    that actually needed it, so their ratio is the statement being tested. `partialLayout` is
    Blink's own verdict on scope: false means it laid out the whole document.

    A `stackTrace` in `beginData` is present only when script forced the layout, which is why it
    is split out here: the unqualified count is mostly animation frames and does not separate the
    arms. It needs `disabled-by-default-devtools.timeline.stack` to be recorded, so a run whose
    forced column is zero everywhere is a run whose categories did not arrive, not a tree that
    stopped measuring.
    """
    durations: list[float] = []
    forced_durations: list[float] = []
    dirty: list[int] = []
    total: list[int] = []
    forced_total: list[int] = []
    whole_document = 0
    partial = 0
    unknown_scope = 0
    without_duration = 0
    categories: set[str] = set()
    sources: set[str] = set()
    for event in events:
        if event.get("name") != "Layout":
            continue
        categories.add(str(event.get("cat", "")))
        begin = (event.get("args") or {}).get("beginData") or {}
        stack = begin.get("stackTrace") or []
        # Complete ("X") events carry `dur`.
        if event.get("dur") is None:
            without_duration += 1
        else:
            durations.append(event["dur"] / 1000.0)
            if stack:
                forced_durations.append(event["dur"] / 1000.0)
        if isinstance(begin.get("dirtyObjects"), int):
            dirty.append(begin["dirtyObjects"])
        if isinstance(begin.get("totalObjects"), int):
            total.append(begin["totalObjects"])
            if stack:
                forced_total.append(begin["totalObjects"])
        # Several frames, not one. The top of a forced-layout stack is an anonymous callee inside the bundled
        if stack:
            sources.add(" <- ".join(_frame_label(frame) for frame in stack[:STACK_FRAMES_KEPT]))
        scope = begin.get("partialLayout")
        if scope is True:
            partial += 1
        elif scope is False:
            whole_document += 1
        else:
            unknown_scope += 1
    return {
        "layouts": len(durations) + without_duration,
        "layout_ms": round(sum(durations), 2),
        "max_layout_ms": round(max(durations, default = 0.0), 2),
        "forced_layouts": len(forced_durations),
        "forced_layout_ms": round(sum(forced_durations), 2),
        "max_forced_layout_ms": round(max(forced_durations, default = 0.0), 2),
        "max_forced_total_objects": max(forced_total, default = 0),
        "forced_layout_sources": sorted(sources)[:5],
        "max_dirty_objects": max(dirty, default = 0),
        "max_total_objects": max(total, default = 0),
        "whole_document_layouts": whole_document,
        "partial_layouts": partial,
        "unknown_scope_layouts": unknown_scope,
        "events_without_duration": without_duration,
        "categories": sorted(c for c in categories if c),
    }


def grow_probe(page) -> dict:
    """Grow the content of an OPEN pane and report whether the rendered height followed it.

    Two growths, because they fail differently. The settled one is the easy case. The mid-flight
    one starts while the open animation is still running, which is where a height captured at
    toggle time is already stale by the time it is applied, and is the shape of reasoning text
    streaming into a pane the reader has just opened.
    """
    page.evaluate(CLICK_TRIGGER_JS)
    page.wait_for_timeout(SETTLE_MS)
    settled_before = page.evaluate(PANE_SNAPSHOT_JS)
    page.evaluate("(n) => window.__probeGrow(n)", GROW_PARAGRAPHS)
    page.wait_for_timeout(SETTLE_MS)
    settled_after = page.evaluate(PANE_SNAPSHOT_JS)

    # Back to a closed pane at the original size, so the mid-flight run opens from the state the settled one did rather
    page.evaluate("() => window.__probeReset()")
    page.wait_for_timeout(SETTLE_MS)
    page.evaluate(CLICK_TRIGGER_JS)
    page.wait_for_timeout(SETTLE_MS)

    # The toggle loop ran whole cycles, so the pane is closed and this click opens it.
    page.evaluate(CLICK_TRIGGER_JS)
    page.wait_for_timeout(MIDFLIGHT_GROW_DELAY_MS)
    page.evaluate("(n) => window.__probeGrow(n)", GROW_PARAGRAPHS)
    # Twice the settle, because this growth lands during the transition and the row has to finish resolving against
    page.wait_for_timeout(SETTLE_MS * 2)
    midflight_after = page.evaluate(PANE_SNAPSHOT_JS)
    return {
        "settled_before": settled_before,
        "settled_after": settled_after,
        "midflight_after": midflight_after,
        "grow_paragraphs": GROW_PARAGRAPHS,
        "midflight_delay_ms": MIDFLIGHT_GROW_DELAY_MS,
    }


def run_cell(context, arm: str, fillers: int, options: argparse.Namespace) -> dict:
    """One arm at one document size, on a page of its own.

    A fresh page per cell rather than a re-navigation: the layout tree, the style engine's caches
    and the metrics counters all start clean, and `Performance.getMetrics` is cumulative per page,
    so a shared page would make every cell after the first a delta against a warmed engine.
    """
    url = (
        f"{BASE}/{SMOKE_PAGE}?arm={arm}&fillers={fillers}"
        f"&paneParagraphs={options.pane_paragraphs}"
    )
    page = context.new_page()
    errors: list[str] = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    problems: list[str] = []
    try:
        page.goto(url, wait_until = "domcontentloaded", timeout = READY_TIMEOUT_MS)
        # The page publishes `__probeReady` after two frames and puts the element count in it, so "document size" in
        page.wait_for_function("() => Boolean(window.__probeReady)", timeout = READY_TIMEOUT_MS)
        ready = page.evaluate("() => window.__probeReady")

        try:
            page.wait_for_function(ARM_DRIVABLE_JS, timeout = 60_000)
        except PlaywrightTimeoutError as exc:
            raise RuntimeError(
                f"arm {arm!r} never rendered a [data-probe=trigger] with the grow hooks "
                f"installed; page errors so far: {errors}"
            ) from exc

        # Re-read the size AFTER the arm is drivable, and do not trust the value captured above.
        ready["elements"] = page.evaluate("() => document.getElementsByTagName('*').length")

        # The page echoes the query it parsed.
        if ready.get("arm") != arm:
            problems.append(f"asked for arm {arm!r}, page reports {ready.get('arm')!r}")
        if ready.get("fillers") != fillers:
            problems.append(f"asked for {fillers} fillers, page reports {ready.get('fillers')!r}")
        if ready.get("paneParagraphs") != options.pane_paragraphs:
            problems.append(
                f"asked for {options.pane_paragraphs} pane paragraphs, page reports "
                f"{ready.get('paneParagraphs')!r}"
            )

        cdp = context.new_cdp_session(page)
        cdp.send("Performance.enable")
        # After load, so the page's own build is never throttled in, and recorded in the report so
        # a difference between cells can never be an artefact of uneven throttling.
        if options.throttle > 1:
            cdp.send("Emulation.setCPUThrottlingRate", {"rate": options.throttle})

        # One discarded cycle. The first open mounts the pane's content and resolves its styles for the first time
        page.evaluate(TOGGLE_CYCLES_JS, [1, SETTLE_MS])

        collected = start_tracing(cdp)
        before = metrics(cdp)
        toggled = page.evaluate(TOGGLE_CYCLES_JS, [options.cycles, SETTLE_MS])
        after = metrics(cdp)
        events = stop_tracing(cdp, page, collected)

        grown = grow_probe(page)
    finally:
        page.close()

    trace = summarise_layouts(events)
    return {
        "arm": arm,
        "fillers": fillers,
        "elements": ready.get("elements"),
        "cycles": options.cycles,
        "clicks": toggled["clicks"],
        "state_after_toggles": toggled["state"],
        "trace": trace,
        "cdp": {
            "layout_count": delta(before, after, "LayoutCount"),
            "layout_ms": round(delta(before, after, "LayoutDuration") * 1000, 2),
            "recalc_style_count": delta(before, after, "RecalcStyleCount"),
            "recalc_style_ms": round(delta(before, after, "RecalcStyleDuration") * 1000, 2),
            "task_ms": round(delta(before, after, "TaskDuration") * 1000, 2),
        },
        "grow": grown,
        "page_errors": errors,
        "problems": problems,
    }


def render_table(cells: list[dict]) -> str:
    header = (
        f"{'arm':<16}{'fillers':>8}{'elements':>10}{'dirty':>7}{'total objs':>12}"
        f"{'forced':>8}{'forced ms':>11}{'max forced ms':>15}"
        f"{'layouts':>9}{'all ms':>9}{'cdp n':>8}{'cdp ms':>9}"
    )
    lines = [header, "-" * len(header)]
    for cell in cells:
        trace = cell["trace"]
        lines.append(
            f"{cell['arm']:<16}{cell['fillers']:>8,}{(cell['elements'] or 0):>10,}"
            f"{trace['max_dirty_objects']:>7,}{trace['max_total_objects']:>12,}"
            f"{trace['forced_layouts']:>8,}{trace['forced_layout_ms']:>11.2f}"
            f"{trace['max_forced_layout_ms']:>15.2f}"
            f"{trace['layouts']:>9,}{trace['layout_ms']:>9.2f}"
            f"{cell['cdp']['layout_count']:>8,.0f}{cell['cdp']['layout_ms']:>9.2f}"
        )
    return "\n".join(lines)


def render_scaling(cells: list[dict]) -> str:
    """Small versus large, per arm. This is the comparison the whole page exists to make.

    Read against the forced layouts, since that is the column the arms differ on; the totals are
    dominated by whatever the arm animates.
    """
    by_arm: dict[str, list[dict]] = {}
    for cell in cells:
        by_arm.setdefault(cell["arm"], []).append(cell)
    lines: list[str] = []
    for arm, group in by_arm.items():
        if len(group) < 2:
            continue
        ordered = sorted(group, key = lambda c: c["fillers"])
        small, large = ordered[0], ordered[-1]

        def ratio(a: float, b: float) -> str:
            return f"{(b / a):.1f}x" if a > 0 else "n/a"

        small_forced = small["trace"]["max_forced_layout_ms"]
        large_forced = large["trace"]["max_forced_layout_ms"]
        lines.append(
            f"{arm:<16}{small['fillers']:,} -> {large['fillers']:,} fillers: "
            f"forced layouts {small['trace']['forced_layouts']} -> "
            f"{large['trace']['forced_layouts']}, "
            f"worst forced {small_forced:.2f}ms -> {large_forced:.2f}ms "
            f"({ratio(small_forced, large_forced)}), "
            f"total objects {small['trace']['max_total_objects']:,} -> "
            f"{large['trace']['max_total_objects']:,}, "
            f"whole-document layouts {small['trace']['whole_document_layouts']} -> "
            f"{large['trace']['whole_document_layouts']}"
        )
    return "\n".join(lines)


def render_forced_sources(cells: list[dict]) -> str:
    """Where the forced layouts came from, so a surprising row can be attributed rather than
    argued about. On the Radix arms this is the collapsible's layout effect."""
    seen: dict[str, set[str]] = {}
    for cell in cells:
        for source in cell["trace"]["forced_layout_sources"]:
            seen.setdefault(cell["arm"], set()).add(source)
    return "\n".join(f"{arm:<16}{source}" for arm in sorted(seen) for source in sorted(seen[arm]))


def render_growth(cells: list[dict]) -> str:
    header = (
        f"{'arm':<16}{'fillers':>8}{'settled px over':>17}{'midflight px over':>19}"
        f"{'paragraphs':>12}"
    )
    lines = [header, "-" * len(header)]
    for cell in cells:
        grown = cell["grow"]
        settled = grown.get("settled_after") or {}
        midflight = grown.get("midflight_after") or {}
        lines.append(
            f"{cell['arm']:<16}{cell['fillers']:>8,}"
            f"{_px(settled.get('overflowPx')):>17}{_px(midflight.get('overflowPx')):>19}"
            f"{str(settled.get('paragraphs')):>12}"
        )
    return "\n".join(lines)


def _px(value) -> str:
    return "n/a" if value is None else f"{value:.1f}"


def collect_failures(report: dict) -> list[str]:
    """Harness failures only.

    Nothing here reads a duration. A probe that failed when a toggle got slower would be red on
    the tree it is meant to describe, and the first fix for that is always to raise the number.
    What it does refuse is a run that measured nothing, because those are the ones that look like
    a clean result.
    """
    failures: list[str] = []
    cells = report["cells"]
    if not cells:
        failures.append("no cells ran")
    for cell in cells:
        where = f"{cell['arm']}@{cell['fillers']}"
        failures.extend(f"{where}: {problem}" for problem in cell["problems"])
        if cell["page_errors"]:
            failures.append(f"{where}: page errors: {cell['page_errors']}")
        if cell["clicks"] != cell["cycles"] * 2:
            failures.append(
                f"{where}: {cell['clicks']} clicks for {cell['cycles']} cycles; the toggle loop "
                "did not run"
            )
        if cell["trace"]["layouts"] <= 0:
            failures.append(
                f"{where}: the trace captured no Layout events, so every layout number in this "
                f"row is a zero that means 'not measured'. Categories seen: "
                f"{cell['trace']['categories']}"
            )
        if cell["cdp"]["layout_count"] <= 0:
            failures.append(
                f"{where}: Performance.getMetrics saw no layouts across {cell['cycles']} toggle "
                "cycles, so the pane never actually opened"
            )
        grown = cell["grow"]
        settled_before = grown.get("settled_before") or {}
        settled_after = grown.get("settled_after") or {}
        if not settled_before or not settled_after:
            failures.append(f"{where}: the pane was not open for the growth probe")
        elif settled_after.get("paragraphs", 0) <= settled_before.get("paragraphs", 0):
            failures.append(
                f"{where}: __probeGrow added no paragraphs "
                f"({settled_before.get('paragraphs')} -> {settled_after.get('paragraphs')}), so "
                "the streaming check measured nothing"
            )

    # The forced column is the one the arms are read on, and it is the one that goes quietly to zero:
    ran_radix_height = [cell for cell in cells if cell["arm"] == "radix-height"]
    if ran_radix_height and not any(cell["trace"]["forced_layouts"] for cell in cells):
        failures.append(
            "no arm recorded a script-forced layout, including radix-height, whose measurement is "
            "unconditional upstream. The Layout events arrived without `beginData.stackTrace`, so "
            "the forced columns are zeros that mean 'not measured'. Categories seen: "
            f"{sorted({c for cell in cells for c in cell['trace']['categories']})}"
        )

    # A sweep whose cells all rendered the same document proves nothing about scaling, and the flat table it produces
    by_arm: dict[str, list[dict]] = {}
    for cell in cells:
        by_arm.setdefault(cell["arm"], []).append(cell)
    for arm, group in by_arm.items():
        if len(group) < 2:
            continue
        ordered = sorted(group, key = lambda c: c["fillers"])
        counts = [cell["elements"] or 0 for cell in ordered]
        if counts != sorted(counts) or counts[0] == counts[-1]:
            failures.append(
                f"{arm}: element counts {counts} do not grow with the filler sizes "
                f"{[cell['fillers'] for cell in ordered]}; the sweep compared identical documents"
            )
    return failures


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description = "Measure what a collapsible toggle costs against the whole document.",
    )
    parser.add_argument(
        "--arm",
        action = "append",
        choices = ARMS,
        help = "restrict the sweep to this arm; repeatable. Default: all four.",
    )
    parser.add_argument(
        "--fillers",
        default = None,
        help = (
            "comma separated filler-row counts to sweep. Default: "
            f"{','.join(str(n) for n in DEFAULT_FILLERS)}. A single value turns the sweep into "
            "one run and gives up the scaling comparison, which is the point of the sweep."
        ),
    )
    parser.add_argument(
        "--cycles",
        type = int,
        default = DEFAULT_CYCLES,
        help = f"open/close cycles per cell (default {DEFAULT_CYCLES}).",
    )
    parser.add_argument(
        "--pane-paragraphs",
        type = int,
        default = DEFAULT_PANE_PARAGRAPHS,
        help = (
            "paragraphs inside the pane. Identical across cells on purpose: it is what makes a "
            f"cost that grows with the filler attributable to the filler (default "
            f"{DEFAULT_PANE_PARAGRAPHS})."
        ),
    )
    parser.add_argument(
        "--throttle",
        type = int,
        default = DEFAULT_THROTTLE,
        help = f"CDP CPU throttling rate, 1 for off (default {DEFAULT_THROTTLE}).",
    )
    parser.add_argument(
        "--json",
        action = "store_true",
        help = "write the report to stdout as JSON; progress and tables move to stderr.",
    )
    parser.add_argument(
        "--headful",
        action = "store_true",
        help = "run the browser with a window, for watching a toggle by eye.",
    )
    args = parser.parse_args(argv)
    if args.fillers is None:
        args.filler_sizes = list(DEFAULT_FILLERS)
    else:
        try:
            args.filler_sizes = [int(part) for part in args.fillers.split(",") if part.strip()]
        except ValueError:
            parser.error(f"--fillers takes comma separated integers, got {args.fillers!r}")
        if not args.filler_sizes:
            parser.error("--fillers was empty")
    args.arms = list(args.arm) if args.arm else list(ARMS)
    if args.cycles < 1:
        parser.error("--cycles must be at least 1")
    return args


def run(options: argparse.Namespace) -> dict:
    report: dict = {
        "label": LABEL,
        "base": BASE,
        "arms": options.arms,
        "filler_sizes": options.filler_sizes,
        "cycles": options.cycles,
        "pane_paragraphs": options.pane_paragraphs,
        "cpu_throttle": options.throttle,
        "grid_collapse_reasoning_enabled": reasoning_flag_in_source(),
        "cells": [],
    }
    with sync_playwright() as p:
        browser = p.chromium.launch(headless = not options.headful, args = chromium_launch_args())
        # A fixed viewport, because a filler row's height depends on how many times it wraps and therefore on the width;
        context = browser.new_context(viewport = {"width": 1200, "height": 900})
        try:
            for arm in options.arms:
                for fillers in options.filler_sizes:
                    info(f"{arm} @ {fillers:,} fillers")
                    report["cells"].append(run_cell(context, arm, fillers, options))
        finally:
            context.close()
            browser.close()
    return report


def main(argv: list[str] | None = None) -> int:
    global _LOG
    options = parse_args(sys.argv[1:] if argv is None else argv)
    if options.json:
        _LOG = sys.stderr

    vite = None
    if OWNS_SERVER:
        info(f"starting vite dev server on port {PORT}")
        vite = start_vite(PORT)
    try:
        wait_for_smoke_page(
            f"{BASE}/{SMOKE_PAGE}",
            SMOKE_ENTRY,
            proc = vite,
            info = info,
        )
        report = run(options)
    finally:
        if vite is not None:
            stop_process(vite)
            info("vite stopped")

    report["harness_failures"] = collect_failures(report)

    out = OUT / f"{LABEL}.json"
    out.parent.mkdir(parents = True, exist_ok = True)
    out.write_text(json.dumps(report, indent = 2), encoding = "utf-8")

    flag = report["grid_collapse_reasoning_enabled"]
    info(f"GRID_COLLAPSE_REASONING_ENABLED in this checkout: {flag}")
    info(f"CPU throttle: {options.throttle}x, {options.cycles} open/close cycles per cell")
    info(
        "layout cost per cell (forced = laid out synchronously because script read a geometry; "
        "the rest is animation frames):\n" + render_table(report["cells"])
    )
    scaling = render_scaling(report["cells"])
    if scaling:
        info("scaling across document sizes:\n" + scaling)
    sources = render_forced_sources(report["cells"])
    if sources:
        info("what forced those layouts:\n" + sources)
    info(
        "growth into an open pane, as pixels of the last paragraph left below the pane's box "
        "(<= 0 means the rendered height followed the content; a small negative number is the "
        "last paragraph's own bottom margin):\n" + render_growth(report["cells"])
    )
    info(f"wrote {out}")

    # Everything human goes to stderr under --json, so stdout stays a single parseable document.
    if options.json:
        print(json.dumps(report, indent = 2), flush = True)

    if report["harness_failures"]:
        for failure in report["harness_failures"]:
            info(f"FAIL: {failure}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
