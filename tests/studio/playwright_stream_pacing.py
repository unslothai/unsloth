# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Main-thread cost of a fast streaming reply in the chat renderer.

Four merged PRs moved this path and each rebuilt a throwaway harness to prove it:

    #7892  Streamdown's transition starvation      9.86s -> 0.34s longest freeze
    #8750  incremental Markdown parsing            O(n) per update -> tail only
    #8845  publish coalescing                      4.01s -> 0.62s longest stall
    #8935  incremental fence tokenization          21x fewer characters to Shiki

Nothing was left behind that would notice the next regression, and each of those
measurements had to rediscover the same methodology. This is that harness, kept.

It drives smoke-stream-pacing.html, which mounts the real MarkdownText inside a real
assistant-ui local runtime, so nothing here is a mock of the code under test, and
assistant-ui's own update scheduling is inside the measurement. Runs against a vite dev
server; no backend, no auth, no GPU, no model.

The reply is a fixed string. #8845's first measurement attempts failed because a real
model gave the two sides different essays and the renderer's cost is superlinear in
length, so a comparison across different text says nothing.

CPU throttling is not decoration. On a developer machine the renderer keeps up with
any rate this can feed, so an unthrottled run measures nothing on either side.

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
BASE = os.environ.get("SMOKE_BASE_URL", f"http://127.0.0.1:{PORT}")
OWNS_SERVER = "SMOKE_BASE_URL" not in os.environ
OUT = Path(os.environ.get("SMOKE_OUT_DIR", "."))
LABEL = "stream-pacing"

# The reply, and the rate it arrives at. Length is what the renderer's cost is
# superlinear in, so it is the knob that matters; the arrival count is kept modest
# because the throttling slows the feed's own timers too, and 1,000 arrivals at 6x
# stretched a one-second stream to 43s of wall clock for no extra signal.
TOTAL_CHARS = int(os.environ.get("SMOKE_STREAM_CHARS", "24000"))
CHUNK_CHARS = int(os.environ.get("SMOKE_STREAM_CHUNK", "96"))
GAP_MS = int(os.environ.get("SMOKE_STREAM_GAP_MS", "2"))
THROTTLE = int(os.environ.get("SMOKE_STREAM_THROTTLE", "6"))

# Budgets, not targets, and chosen against a regression rather than by feel. Reverting
# #8750 (incremental Markdown parsing) in this harness moves them very differently:
#
#                          main    #8750 reverted
#   long tasks           5,901ms          13,059ms
#   long task count           49               144
#   fully painted       20,810ms          25,676ms
#   longest stall        1,050ms             967ms
#
# So the long-task total is the sensitive one and gets a budget that sits between the
# two, with 1.7x headroom over main. The longest stall did NOT move, which is worth
# knowing: it is kept as a coarse backstop for the freeze class of regression #7892 and
# #8845 fixed, not as the number that catches a parse getting slower.
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
            # Applied after load so the harness bundle is not itself throttled in, and
            # recorded below, so a difference can never be an artefact of one run
            # having been throttled harder than another.
            if THROTTLE > 1:
                cdp.send("Emulation.setCPUThrottlingRate", {"rate": THROTTLE})

            page.evaluate(
                "(o) => window.__stream.run(o)",
                {"totalChars": TOTAL_CHARS, "chunkChars": CHUNK_CHARS, "gapMs": GAP_MS},
            )

            # Poll for the harness's own verdict rather than deciding from out here:
            # every round trip is itself slowed by the throttling, so an outside
            # judgement of "has it finished" arrives late enough to hide the effect.
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
    out.write_text(json.dumps(results, indent = 2), encoding = "utf-8")
    info(json.dumps(results, indent = 2))
    info(f"wrote {out}")

    failures: list[str] = []
    # A harness that painted nothing measures a perfect zero on every budget below.
    # This is the assertion that stops a broken page from reading as a pass.
    #
    # Not an equality: Markdown syntax (fences, list markers, math delimiters) never
    # reaches textContent, so the rendered length is a few per cent under the bytes
    # sent. 90% is comfortably above that and far below "the render died early".
    floor = int(TOTAL_CHARS * 0.9)
    if results["paintedChars"] < floor:
        failures.append(
            f"only {results['paintedChars']} characters painted of {TOTAL_CHARS} sent "
            f"(floor {floor}); the budgets below measured no workload"
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
