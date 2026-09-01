# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Main-thread cost of a fast streaming reply in the chat renderer.

Four merged PRs moved this path and each rebuilt a throwaway harness to prove it:

    #7892  Streamdown's transition starvation      9.86s -> 0.34s longest freeze
    #8750  incremental Markdown parsing            O(n) per update -> tail only
    #8845  publish coalescing                      4.01s -> 0.62s longest stall
    #8935  incremental fence tokenization          21x fewer characters to Shiki

None of them left anything behind that would notice the next regression, and each had to
rediscover the same methodology. This is that harness, kept.

It drives smoke-stream-pacing.html, which mounts the real MarkdownText inside a real
assistant-ui local runtime, so nothing is a mock of the code under test and assistant-ui's
own update scheduling is inside the measurement. Runs against a vite dev server; no
backend, no auth, no GPU, no model.

The reply is a fixed string. #8845's first measurement attempts failed because a real
model gave the two sides different essays and the renderer's cost is superlinear in
length, so a comparison across different text says nothing.

CPU throttling is not decoration: on a developer machine the renderer keeps up with any
rate this can feed, so an unthrottled run measures nothing on either side.

Chromium only, deliberately. Both things that make this a measurement are Chromium-only:
`Emulation.setCPUThrottlingRate` is reached over CDP, which Playwright exposes for Chromium
alone, and `longtask` PerformanceObserver entries exist in no other engine (Gecko bug
1348405 is open; WebKit has never shipped them). Neither fails loudly on firefox or webkit,
since `observe({type: "longtask"})` is specified to abort silently on an unsupported type
rather than throw, so the budgets would read a perfect zero instead of an error. The verdict
below therefore refuses a run that saw no long tasks, and the harness records whether the
engine supported them.

Run:
    python tests/studio/playwright_stream_pacing.py

It starts and stops its own vite dev server. Point it at one you already have with
SMOKE_BASE_URL, or move the port it picks with SMOKE_PORT.
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

PORT = int(os.environ.get("SMOKE_PORT", "5186"))
# Unset: start and stop our own server.
_EXTERNAL = os.environ.get("SMOKE_BASE_URL", "").strip()
BASE = _EXTERNAL or f"http://127.0.0.1:{PORT}"
OWNS_SERVER = not _EXTERNAL
# Under logs/ like every sibling harness.
# A default of "." would drop an untracked stream-pacing.json in the repo root every run;
OUT = Path(os.environ.get("PW_ART_DIR", "logs/playwright-stream-pacing"))
LABEL = "stream-pacing"

# The reply and the rate it arrives at.
# the arrival count stays modest because throttling slows the feed's own timers too, and 1,000 arrivals at 6x stretched
# a one-second stream to 43s of wall clock for no extra signal.
TOTAL_CHARS = int(os.environ.get("SMOKE_STREAM_CHARS", "24000"))
CHUNK_CHARS = int(os.environ.get("SMOKE_STREAM_CHUNK", "96"))
GAP_MS = int(os.environ.get("SMOKE_STREAM_GAP_MS", "2"))
THROTTLE = int(os.environ.get("SMOKE_STREAM_THROTTLE", "6"))

# Budgets, not targets, chosen against real regressions rather than by feel.
# main #8750 reverted #7892 reverted long tasks 5.0-8.0s 13.1s/74.4s 6.7-7.9s long task count 49-71 144/598 14-23
# longest stall 1.05-1.23s 0.97-1.40s 5.03-6.35s Reverting #8750 (incremental Markdown parsing) blows up the long-task
# total and leaves the longest stall alone.
# Machine spread: the "main" column above is 5,029/5,901ms on one machine and 6,687-8,003ms on another, so clean
# readings vary ~60% ACROSS boxes even though a single box repeats to within ~15%.
# The 10,000ms budget is NOT raised for that headroom, because the same two machines read the #8750 revert as 13,059ms
# and 74,353ms: a budget loose enough for the slower box would stop catching that regression on the faster one.
MAX_LONGEST_STALL_MS = int(os.environ.get("SMOKE_STREAM_STALL_BUDGET_MS", "2500"))
MAX_LONG_TASK_MS = int(os.environ.get("SMOKE_STREAM_LONG_TASK_BUDGET_MS", "10000"))


def info(msg: str) -> None:
    print(f"[{LABEL}] {msg}", flush = True)


def run() -> dict:
    headless = os.environ.get("SMOKE_HEADFUL") != "1"
    with sync_playwright() as p:
        browser = p.chromium.launch(headless = headless, args = chromium_launch_args())
        context = browser.new_context(viewport = {"width": 1200, "height": 900})
        page = context.new_page()
        errors: list[str] = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        try:
            page.goto(f"{BASE}/smoke-stream-pacing.html", wait_until = "load", timeout = 60_000)
            page.wait_for_function("() => window.__stream && window.__stream.ready", timeout = 60_000)

            cdp = context.new_cdp_session(page)
            # below so a difference can never be an artefact of uneven throttling.
            # After load so the harness bundle is not itself throttled in, and recorded below so a difference can never
            if THROTTLE > 1:
                cdp.send("Emulation.setCPUThrottlingRate", {"rate": THROTTLE})

            page.evaluate(
                "(o) => window.__stream.run(o)",
                {"totalChars": TOTAL_CHARS, "chunkChars": CHUNK_CHARS, "gapMs": GAP_MS},
            )

            # Poll the harness's own verdict rather than deciding out here:
            deadline = time.monotonic() + 300
            results: dict = {}
            while time.monotonic() < deadline:
                results = page.evaluate("() => window.__stream.results()")
                if results.get("done"):
                    break
                time.sleep(0.25)
            if not results.get("done"):
                raise RuntimeError(
                    f"the reply never finished painting within 300s: {json.dumps(results)}"
                )
        finally:
            context.close()
            browser.close()

    results["page_errors"] = errors
    results["cpu_throttle"] = THROTTLE
    results["total_chars"] = TOTAL_CHARS
    results["chunk_chars"] = CHUNK_CHARS
    results["gap_ms"] = GAP_MS
    return results


def main() -> int:
    vite = None
    if OWNS_SERVER:
        info(f"starting vite dev server on port {PORT}")
        vite = start_vite(PORT)
    try:
        wait_for_smoke_page(
            f"{BASE}/smoke-stream-pacing.html",
            "smoke-stream-pacing-main.tsx",
            proc = vite,
            info = info,
        )
        results = run()
    finally:
        if vite is not None:
            stop_process(vite)
            info("vite stopped")

    out = OUT / f"{LABEL}.json"
    out.parent.mkdir(parents = True, exist_ok = True)
    out.write_text(json.dumps(results, indent = 2), encoding = "utf-8")
    info(json.dumps(results, indent = 2))
    info(f"wrote {out}")

    failures: list[str] = []
    # delimiters) never reaches textContent, so rendered length is a few per cent under the
    # A page that painted nothing scores a perfect zero on every budget below, so assert the workload first.
    floor = int(TOTAL_CHARS * 0.9)
    if results["paintedChars"] < floor:
        failures.append(
            f"only {results['paintedChars']} characters painted of {TOTAL_CHARS} sent "
            f"(floor {floor}); the budgets below measured no workload"
        )
    # paintedChars is a high-water mark and survives a completion render that truncates the bubble.
    if results["settledChars"] < floor:
        failures.append(
            f"the reply settled at {results['settledChars']} characters of {TOTAL_CHARS} "
            f"sent (floor {floor}); it peaked at {results['paintedChars']} and then lost "
            "content, so the final render is incomplete"
        )
    if results["arrivals"] < TOTAL_CHARS // CHUNK_CHARS:
        failures.append(
            f"only {results['arrivals']} arrivals for {TOTAL_CHARS} characters; "
            "the stream did not run at the rate this claims to measure"
        )
    if results["longestStallMs"] > MAX_LONGEST_STALL_MS:
        failures.append(
            f"longest stall {results['longestStallMs']:.0f}ms exceeds "
            f"{MAX_LONGEST_STALL_MS}ms (the bubble stopped growing while text arrived)"
        )
    # The long-task total is the sensitive metric and the one that goes false-green most quietly: an engine without the
    # longtask entry type, or an observer that stopped delivering, reports 0ms and sails under the budget without
    # raising.
    if not results.get("longTaskSupported"):
        failures.append(
            "this engine reports no longtask entries, so the long-task budget measured "
            "nothing; run under Chromium"
        )
    elif results["longTasks"] <= 0:
        failures.append(
            "no long tasks were observed at all; the observer measured nothing, so the "
            f"{MAX_LONG_TASK_MS}ms budget below would pass on any tree"
        )
    if results["cpu_throttle"] <= 1:
        failures.append(
            f"CPU throttling was {results['cpu_throttle']}x; unthrottled, the renderer keeps "
            "up with any rate this can feed and the budgets measure nothing"
        )
    if results["longTaskMs"] > MAX_LONG_TASK_MS:
        failures.append(
            f"long tasks totalled {results['longTaskMs']:.0f}ms, over the "
            f"{MAX_LONG_TASK_MS}ms budget (the main thread is saturated by the render)"
        )
    if results["page_errors"]:
        failures.append(f"page errors: {results['page_errors']}")

    if failures:
        for f in failures:
            info(f"FAIL: {f}")
        return 1
    info(
        f"OK: longest stall {results['longestStallMs']:.0f}ms, "
        f"long tasks {results['longTaskMs']:.0f}ms, "
        f"{results['framesOver33ms']} frames over 33ms, fully painted at "
        f"{results['timeToFullyPaintedMs']:.0f}ms, {results['settledChars']} chars, "
        f"{THROTTLE}x throttle"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
