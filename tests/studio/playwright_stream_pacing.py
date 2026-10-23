# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Streaming-reply main-thread cost, Chromium only: CDP CPU throttling and longtask entries."""

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
# Exported-but-empty counts as unset.
_EXTERNAL = os.environ.get("SMOKE_BASE_URL", "").strip()
BASE = _EXTERNAL or f"http://127.0.0.1:{PORT}"
OWNS_SERVER = not _EXTERNAL
OUT = Path(os.environ.get("PW_ART_DIR", "logs/playwright-stream-pacing"))
LABEL = "stream-pacing"

# Renderer cost is superlinear in length; throttling also slows the feed's timers, so keep arrivals modest.
TOTAL_CHARS = int(os.environ.get("SMOKE_STREAM_CHARS", "24000"))
CHUNK_CHARS = int(os.environ.get("SMOKE_STREAM_CHUNK", "96"))
GAP_MS = int(os.environ.get("SMOKE_STREAM_GAP_MS", "2"))
THROTTLE = int(os.environ.get("SMOKE_STREAM_THROTTLE", "6"))

# Two budgets because reverting #8750 (incremental Markdown) blows up the long-task total while
# reverting #7892 (Streamdown `animated`) blows up the longest stall. Retune from runner numbers.
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
            # Throttle after load, recorded, so differences are not throttling artefacts.
            if THROTTLE > 1:
                cdp.send("Emulation.setCPUThrottlingRate", {"rate": THROTTLE})

            page.evaluate(
                "(o) => window.__stream.run(o)",
                {"totalChars": TOTAL_CHARS, "chunkChars": CHUNK_CHARS, "gapMs": GAP_MS},
            )

            # Poll the in-page verdict: throttled round trips arrive too late and hide the effect.
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
    # Markdown syntax never reaches textContent, so rendered length is a few per cent under bytes sent.
    floor = int(TOTAL_CHARS * 0.9)
    if results["paintedChars"] < floor:
        failures.append(
            f"only {results['paintedChars']} characters painted of {TOTAL_CHARS} sent "
            f"(floor {floor}); the budgets below measured no workload"
        )
    # paintedChars is a high-water mark that survives a truncating completion render.
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
    # Without longtask support the total reads 0ms and passes silently.
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
