# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""How much JS heap one streamed markdown code fence leaves behind after it is gone.

WHAT IS BEING TESTED

`@streamdown/code` keeps a module-level `Map` of Shiki tokenisations whose key embeds the code's
LENGTH and its first and last 100 characters. A fence that is STREAMED is handed to the
highlighter once per throttle window with a longer prefix each time, so every window mints a new
key and a new full tokenisation of that prefix, and the package has no eviction path: no size
cap, no clear, no unmount hook. The suspicion is that a finished reply therefore leaves behind
one tokenisation per throttle window instead of one per fence.

That is a claim about the V8 heap, not about the DOM, so this file measures the V8 heap after the
rendered reply has been unmounted and thrown away:

    stream one reply -> wait for every dispatch to land -> unmount it -> force GC -> read heap

`Runtime.getHeapUsage.usedSize` is the V8 isolate's live bytes. Blink DOM nodes are NOT in it
(they live in PartitionAlloc), which is exactly why it is the right instrument here: a number that
keeps climbing across unmounted replies cannot be detached nodes.

WHY A SLOPE

One delta is a number anybody can produce by mis-timing a GC. The headline is the least-squares
SLOPE of retained heap against fence index over `SMOKE_SD_FENCES` replies, with R^2 printed next
to it, after a warm-up reply that pays the one-off grammar and theme load.

THE CONTROL ARMS

Three arms run per invocation, each in its own browser so no cache carries over:

    stream   32 KB fence delivered over many ticks. The arm under test.
    whole    the same 32 KB fence delivered in ONE update. Same content, same DOM, same React
             work, one cache key instead of many. If the retention is about prefix keying rather
             than about fence content, this arm must be far flatter than `stream`.
    prose    32 KB of plain text streamed the same way. The code highlighter is never called, so
             this arm must be flat. It is the "am I measuring anything specific" control.

And a size ladder inside `stream`: 8 KB and 32 KB. The harness FAILS if the 32 KB slope is not
clearly above the 8 KB slope, because a retention metric that does not rise with fence size is
not measuring retention.

Also reported, from a bare upstream plugin driven with no React in the picture at all: the heap
cost of ONE cache entry, which is what a fix has to remove.

Run:
    python tests/studio/playwright_shiki_retention.py
    SMOKE_PORT=5391 SMOKE_SD_LABEL=head python tests/studio/playwright_shiki_retention.py

Exit codes: 0 when the harness produced a self-consistent measurement, non-zero when the harness
itself is broken (a fixture that did not stream, a dispatch count of zero on the `stream` arm, a
size ladder that did not rise). It reports; it does not gate on the retention number.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _playwright_robust import (  # noqa: E402
    chromium_launch_args,
    start_vite,
    stop_process,
    wait_for_smoke_page,
)

PORT = int(os.environ.get("SMOKE_PORT", "5391"))
_EXTERNAL = os.environ.get("SMOKE_BASE_URL", "").strip().rstrip("/")
BASE = _EXTERNAL or f"http://127.0.0.1:{PORT}"
OWNS_SERVER = not _EXTERNAL
LABEL = os.environ.get("SMOKE_SD_LABEL", "tree")
OUT = Path(os.environ.get("PW_ART_DIR", "logs/playwright-shiki-retention"))

# Replies measured per arm, after the warm-up. Four is the floor for a slope worth printing.
FENCES = int(os.environ.get("SMOKE_SD_FENCES", "6"))
# Ticks per streamed reply and the pause between them. Together these set the wall-clock duration
# of the stream, and the number of retained cache entries is duration / 250 ms, so both are
# printed with every result. 96 ticks x 40 ms is a ~5 s reply, i.e. a fast local model.
TICKS = int(os.environ.get("SMOKE_SD_TICKS", "96"))
TICK_MS = int(os.environ.get("SMOKE_SD_TICK_MS", "40"))
SIZES = [int(n) for n in os.environ.get("SMOKE_SD_SIZES", "8192,32768").split(",")]
BIG = max(SIZES)
# Settle poll. Studio's code plugin can schedule a trailing re-tokenisation up to 250 ms after the
# last render, so a poll shorter than that would call a reply finished while work is still queued.
SETTLE_POLL_MS = int(os.environ.get("SMOKE_SD_SETTLE_POLL_MS", "350"))
# Consecutive quiet polls required. Three is the minimum that can tell "finished" from "between
# two chunks of async highlighting"; two consecutive equal samples is a plateau detector, not a
# settle, because Shiki finishes fence by fence off separate microtask chains.
SETTLE_QUIET_POLLS = int(os.environ.get("SMOKE_SD_SETTLE_QUIET", "4"))
SETTLE_TIMEOUT_S = float(os.environ.get("SMOKE_SD_SETTLE_TIMEOUT_S", "120"))
MB = 1024.0 * 1024.0


def info(message: str) -> None:
    print(f"[shiki-retention] {message}", flush = True)


def collect_garbage(cdp) -> None:
    """Three passes, because one `collectGarbage` leaves objects that only a second pass over the
    old generation reclaims, and the third is what makes the reading repeatable."""
    for _ in range(3):
        cdp.send("HeapProfiler.collectGarbage")
        time.sleep(0.12)


def heap_used(cdp) -> float:
    return float(cdp.send("Runtime.getHeapUsage")["usedSize"])


def dom_nodes(cdp) -> int:
    try:
        return int(cdp.send("Memory.getDOMCounters")["nodes"])
    except Exception:
        return -1


def sample(cdp) -> float:
    collect_garbage(cdp)
    return heap_used(cdp)


def wait_settled(page) -> dict:
    """Quiet across `SETTLE_QUIET_POLLS` consecutive polls: no dispatch in flight AND no new
    dispatch started. Both, because either alone is satisfied in the gap between two fences."""
    deadline = time.monotonic() + SETTLE_TIMEOUT_S
    quiet = 0
    last_calls = -1
    while time.monotonic() < deadline:
        time.sleep(SETTLE_POLL_MS / 1000.0)
        counters = page.evaluate("() => window.__sd.counters()")
        if counters["pending"] == 0 and counters["renderCalls"] == last_calls:
            quiet += 1
            if quiet >= SETTLE_QUIET_POLLS:
                return counters
        else:
            quiet = 0
        last_calls = counters["renderCalls"]
    raise RuntimeError(f"highlighting never settled within {SETTLE_TIMEOUT_S}s")


def fit(xs: list[float], ys: list[float]) -> tuple[float, float]:
    """Least-squares slope and R^2. R^2 is 0.0 when y is constant, which is the honest answer for
    a control arm rather than the 1.0 a naive formula reports for a flat line."""
    n = len(xs)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    sxx = sum((x - mean_x) ** 2 for x in xs)
    sxy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    slope = sxy / sxx if sxx else 0.0
    intercept = mean_y - slope * mean_x
    ss_tot = sum((y - mean_y) ** 2 for y in ys)
    ss_res = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(xs, ys))
    r2 = 0.0 if ss_tot == 0 else 1.0 - ss_res / ss_tot
    return slope, r2


def new_browser(pw):
    return pw.chromium.launch(
        args = [*chromium_launch_args(), "--js-flags=--expose-gc"],
        headless = True,
    )


def measure_arm(pw, kind: str, chars: int, tick_ms: int = TICK_MS) -> dict:
    """One arm in its own browser, so the module-level caches start empty."""
    browser = new_browser(pw)
    try:
        page = browser.new_page()
        cdp = page.context.new_cdp_session(page)
        cdp.send("HeapProfiler.enable")
        cdp.send("Runtime.enable")
        errors: list[str] = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.goto(f"{BASE}/smoke-shiki-retention.html", wait_until = "domcontentloaded")
        page.wait_for_function("() => window.__sd && window.__sd.ready", timeout = 120_000)

        fixture_hash = page.evaluate(
            "([k, c, s]) => window.__sd.fixtureHash(k, c, s)", [kind, chars, 1]
        )

        # Warm-up reply: pays the python grammar load, the theme registration and the first
        # React commit, none of which is per-fence retention.
        page.evaluate(
            "(spec) => window.__sd.runOne(spec)",
            {"kind": kind, "chars": chars, "seed": 1, "ticks": TICKS, "tickMs": tick_ms},
        )
        wait_settled(page)
        page.evaluate("() => window.__sd.teardown()")
        baseline = sample(cdp)

        rows = []
        for i in range(1, FENCES + 1):
            seed = 1000 + i
            result = page.evaluate(
                "(spec) => window.__sd.runOne(spec)",
                {"kind": kind, "chars": chars, "seed": seed, "ticks": TICKS, "tickMs": tick_ms},
            )
            counters = wait_settled(page)
            page.evaluate("() => window.__sd.teardown()")
            used = sample(cdp)
            rows.append(
                {
                    "fence": i,
                    "retained_mb": (used - baseline) / MB,
                    "used_mb": used / MB,
                    "render_calls": result["renderCalls"],
                    "dom_nodes_after_unmount": dom_nodes(cdp),
                    "text_length": result["textLength"],
                    "total_render_calls": counters["renderCalls"],
                }
            )
            info(
                f"{LABEL} {kind}@{chars}/{tick_ms}ms fence {i}: retained {rows[-1]['retained_mb']:+.2f} MB, "
                f"{result['renderCalls']} render calls, nodes {rows[-1]['dom_nodes_after_unmount']}"
            )

        slope, r2 = fit([float(r["fence"]) for r in rows], [r["retained_mb"] for r in rows])
        # Bare-plugin cost of one cache entry, React nowhere in the picture. Its own seed range,
        # so its keys cannot collide with the ones the app path already made.
        collect_garbage(cdp)
        before_raw = heap_used(cdp)
        raw_entries = 16
        landed = page.evaluate(
            "([c, s, n]) => window.__sd.rawEntries(c, s, n)", [chars, 90_001, raw_entries]
        )
        raw_used = sample(cdp)
        per_entry_mb = (raw_used - before_raw) / MB / max(1, landed)

        return {
            "kind": kind,
            "chars": chars,
            "tick_ms": tick_ms,
            "fixture_hash": fixture_hash,
            "baseline_mb": baseline / MB,
            "rows": rows,
            "slope_mb_per_fence": slope,
            "r2": r2,
            "mean_render_calls": sum(r["render_calls"] for r in rows) / len(rows),
            "raw_entries_landed": landed,
            "raw_mb_per_entry": per_entry_mb,
            "page_errors": errors,
        }
    finally:
        browser.close()


# Which cells run. `full` is the arm comparison; `ladder` varies the pause between stream ticks,
# which varies the wall-clock duration of the reply without changing a single character of the
# fixture. Retention that comes from a per-throttle-window cache entry MUST rise with the pause;
# `whole` is carried through the ladder as the CONTROL, because it delivers its fence in one
# update and so cannot care what the pause is. A control that moves with the rate means the
# ladder is measuring the harness, not the cache.
MODE = os.environ.get("SMOKE_SD_MODE", "full")
LADDER_TICK_MS = [int(n) for n in os.environ.get("SMOKE_SD_LADDER", "0,40,120,300").split(",")]
LADDER_EXPECT = os.environ.get("SMOKE_SD_LADDER_EXPECT", "rise")


def build_plan() -> list[tuple[str, int, int, str]]:
    if MODE == "ladder":
        plan = []
        for tick_ms in LADDER_TICK_MS:
            plan.append(("stream", BIG, tick_ms, f"stream@{BIG}/{tick_ms}ms"))
            plan.append(("whole", BIG, tick_ms, f"whole@{BIG}/{tick_ms}ms"))
        return plan
    plan = [("stream", size, TICK_MS, f"stream@{size}") for size in SIZES]
    plan += [("whole", BIG, TICK_MS, f"whole@{BIG}"), ("prose", BIG, TICK_MS, f"prose@{BIG}")]
    return plan


def ladder_failures(cells: dict) -> list[str]:
    """The control arm must stay flat across tick rates, and the arm under test must not."""
    control = [c for c in cells.values() if c["kind"] == "whole"]
    tested = [c for c in cells.values() if c["kind"] == "stream"]
    if len(control) < 2 or len(tested) < 2:
        return ["ladder needs at least two rates"]
    failures = []
    control_slopes = [c["slope_mb_per_fence"] for c in control]
    spread = max(control_slopes) - min(control_slopes)
    if spread > 0.35 * max(max(control_slopes), 1e-9) and spread > 0.25:
        failures.append(
            "control arm moved with the tick rate "
            f"({[round(x, 2) for x in control_slopes]}), so the ladder is measuring the harness"
        )
    tested_slopes = [c["slope_mb_per_fence"] for c in tested]
    # What the tested arm is expected to do depends on which tree it is. On a tree that still
    # keeps one cache entry per refresh window it MUST rise with the pause, or the ladder is not
    # exercising the leak. On a tree where the cache is bounded it must NOT, and a rise there is
    # the fix failing. Default is `rise`, so a tree that is silently unfixed cannot pass by
    # accident; a fixed tree has to say so.
    if LADDER_EXPECT == "flat":
        if tested_slopes[-1] > tested_slopes[0] * 1.5:
            failures.append(
                "tested arm rose with the tick rate on a tree that should be flat "
                f"({[round(x, 2) for x in tested_slopes]})"
            )
    elif tested_slopes[-1] <= tested_slopes[0] * 1.5:
        failures.append(
            f"tested arm did not rise with the tick rate ({[round(x, 2) for x in tested_slopes]})"
        )
    return failures


def harness_failures(cells: dict) -> list[str]:
    failures = []
    for key, cell in cells.items():
        if cell["page_errors"]:
            failures.append(f"{key}: page errors {cell['page_errors'][:2]}")
        if not cell["rows"]:
            failures.append(f"{key}: no rows")
        if cell["kind"] != "prose" and cell["mean_render_calls"] < 1:
            failures.append(f"{key}: highlighter was never called ({cell['mean_render_calls']})")
        # DOM nodes are read AFTER the unmount and AFTER the forced GC, so a rising count would
        # mean detached nodes are accumulating and the heap slope is not a JS-cache result.
        first_nodes = cell["rows"][0]["dom_nodes_after_unmount"]
        last_nodes = cell["rows"][-1]["dom_nodes_after_unmount"]
        if first_nodes >= 0 and last_nodes - first_nodes > 200:
            failures.append(
                f"{key}: DOM nodes grew {first_nodes} -> {last_nodes} across unmounted replies, "
                "so this is not a DOM-free measurement"
            )
    if MODE == "ladder":
        return failures + ladder_failures(cells)
    small = cells.get(f"stream@{min(SIZES)}")
    big = cells.get(f"stream@{BIG}")
    if small and big:
        # A retention metric that does not rise with fence size is measuring something else.
        if big["slope_mb_per_fence"] <= small["slope_mb_per_fence"] * 1.5:
            failures.append(
                f"size ladder did not rise: {min(SIZES)} -> {small['slope_mb_per_fence']:.2f} "
                f"MB/fence, {BIG} -> {big['slope_mb_per_fence']:.2f} MB/fence"
            )
    return failures


def main() -> int:
    OUT.mkdir(parents = True, exist_ok = True)
    vite = None
    try:
        if OWNS_SERVER:
            info(f"starting vite on port {PORT}")
            vite = start_vite(PORT)
        wait_for_smoke_page(
            f"{BASE}/smoke-shiki-retention.html",
            "/smoke-shiki-retention-main.tsx",
            proc = vite,
            info = info,
        )
        cells = {}
        with sync_playwright() as pw:
            for kind, chars, tick_ms, name in build_plan():
                cells[name] = measure_arm(pw, kind, chars, tick_ms)
    finally:
        if vite is not None:
            stop_process(vite)
            info("vite stopped")

    report = {
        "label": LABEL,
        "ticks": TICKS,
        "tick_ms": TICK_MS,
        "fences": FENCES,
        "cells": cells,
    }
    report["mode"] = MODE
    path = OUT / f"shiki-retention-{MODE}-{LABEL}.json"
    path.write_text(json.dumps(report, indent = 2), encoding = "utf-8")

    print()
    print(f"  {'arm':<20}{'slope MB/fence':>16}{'R^2':>8}{'calls/fence':>14}{'MB/entry':>10}")
    for key, cell in cells.items():
        print(
            f"  {key:<20}{cell['slope_mb_per_fence']:>16.2f}{cell['r2']:>8.3f}"
            f"{cell['mean_render_calls']:>14.1f}{cell['raw_mb_per_entry']:>10.3f}"
        )
    print()
    info(f"wrote {path}")

    failures = harness_failures(cells)
    for failure in failures:
        info(f"HARNESS FAILURE: {failure}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
