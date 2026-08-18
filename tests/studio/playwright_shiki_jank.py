# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Does the retained Shiki token heap cost FRAMES, or only bytes?

tests/studio/playwright_shiki_retention.py establishes that a streamed markdown code fence leaves
JS heap behind on origin/main, roughly 0.28 MB per 250 ms refresh window for a 32 KB fence, and
that a bounded cache takes that to the single-update floor. That is a claim about bytes. It says
NOTHING about whether a user sees anything, and a memory win that does not move jank is worth
saying out loud rather than implying.

So this file asks the user-facing question directly. It fills the cache the way a session does, by
streaming replies through the real markdown pipeline, and then measures frame timing while ONE
more reply streams:

    accumulate N replies -> forced GC -> read retained heap -> record frames during reply N+1

WHAT IS MEASURED

    fps                 requestAnimationFrame callbacks per second over the recorded window
    p95 frame ms        95th percentile gap between consecutive rAF callbacks
    worst frame ms      largest such gap
    frames over 33 ms   how much of the window missed two vsyncs at 60 Hz
    longest stall ms    largest gap between ticks of a 1 ms setTimeout loop

The last one is the one to trust when the others look too good. rAF stops being SCHEDULED when the
compositor decides nothing is visible, which reads as "no dropped frames" rather than as "no
measurement". A 1 ms timer keeps ticking regardless, so the gap between its ticks is the block.

THE RECORDER IS CALIBRATED BEFORE IT IS BELIEVED. Every run blocks the main thread for a known
120 ms inside a recording window and refuses to report anything unless the recorder saw at least
100 ms of it. A recorder that cannot see a stall it was told about would report every arm as
smooth, which is the most comfortable wrong answer available here.

THE CONTROL. The `prose` cell accumulates and measures replies with NO code fence, so the
highlighter is never called and neither arm retains anything. Its frame numbers must not separate
between the arms. If they do, the difference is the host or the harness, not the cache, and the
`code` cell's numbers mean nothing.

Ratios are the result and absolute milliseconds are context, because this host is shared and
loaded. Run the arms INTERLEAVED, base then head then base, and compare paired.

Run:
    SMOKE_SD_LABEL=base python tests/studio/playwright_shiki_jank.py
    SMOKE_JANK_FENCES=60 SMOKE_PORT=5395 python tests/studio/playwright_shiki_jank.py

Exit codes: 0 when the harness produced a self-consistent measurement, non-zero when the harness
itself is broken (calibration block not seen, no frames recorded, highlighter never called on the
code cell). It reports; it does not gate on the jank numbers.
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

PORT = int(os.environ.get("SMOKE_PORT", "5395"))
_EXTERNAL = os.environ.get("SMOKE_BASE_URL", "").strip().rstrip("/")
BASE = _EXTERNAL or f"http://127.0.0.1:{PORT}"
OWNS_SERVER = not _EXTERNAL
LABEL = os.environ.get("SMOKE_SD_LABEL", "tree")
OUT = Path(os.environ.get("PW_ART_DIR", "logs/playwright-shiki-jank"))

CHARS = int(os.environ.get("SMOKE_JANK_CHARS", "32768"))
# Replies streamed before the measured one. This is what separates the arms: on a tree that keeps
# one tokenisation per refresh window these leave hundreds of megabytes behind, and on a tree with
# a bounded cache they leave tens.
ACCUMULATE = int(os.environ.get("SMOKE_JANK_FENCES", "60"))
# The accumulation replies run with no pause between updates, so filling the cache costs seconds
# rather than minutes. The MEASURED reply is paced like a real one.
ACCUMULATE_TICKS = int(os.environ.get("SMOKE_JANK_ACC_TICKS", "96"))
ACCUMULATE_TICK_MS = int(os.environ.get("SMOKE_JANK_ACC_TICK_MS", "0"))
MEASURE_TICKS = int(os.environ.get("SMOKE_JANK_TICKS", "96"))
MEASURE_TICK_MS = int(os.environ.get("SMOKE_JANK_TICK_MS", "40"))
# Repetitions of the measured reply per cell. The table reports the MEDIAN, because the first
# measured reply after a long accumulation is systematically the worst.
REPEATS = int(os.environ.get("SMOKE_JANK_REPEATS", "3"))

CALIBRATION_BLOCK_MS = 120
CALIBRATION_FLOOR_MS = 100

SETTLE_POLL_MS = 350
SETTLE_QUIET_POLLS = 4
SETTLE_TIMEOUT_S = 180.0
MB = 1024.0 * 1024.0


def info(message: str) -> None:
    print(f"[shiki-jank] {message}", flush = True)


def wait_settled(page) -> None:
    """Quiet across four consecutive polls: nothing in flight AND no new highlight call started.
    Either condition alone is satisfied in the gap between two fences, and the poll interval is
    above the plugin's 250 ms trailing timer so a queued re-tokenisation cannot hide inside it."""
    deadline = time.monotonic() + SETTLE_TIMEOUT_S
    quiet = 0
    last_calls = -1
    while time.monotonic() < deadline:
        time.sleep(SETTLE_POLL_MS / 1000.0)
        counters = page.evaluate("() => window.__sd.counters()")
        if counters["pending"] == 0 and counters["renderCalls"] == last_calls:
            quiet += 1
            if quiet >= SETTLE_QUIET_POLLS:
                return
        else:
            quiet = 0
        last_calls = counters["renderCalls"]
    raise RuntimeError(f"highlighting never settled within {SETTLE_TIMEOUT_S}s")


def collect_garbage(cdp) -> None:
    for _ in range(3):
        cdp.send("HeapProfiler.collectGarbage")
        time.sleep(0.12)


def median(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def calibrate(page) -> float:
    """Block the main thread for a known 120 ms inside a recording window and see it."""
    page.evaluate("() => window.__sd.framesStart()")
    page.evaluate("(ms) => window.__sd.blockFor(ms)", CALIBRATION_BLOCK_MS)
    time.sleep(0.3)
    report = page.evaluate("() => window.__sd.framesStop()")
    return float(report["longestStallMs"])


def measure_cell(pw, kind: str) -> dict:
    browser = pw.chromium.launch(
        args = [*chromium_launch_args(), "--js-flags=--expose-gc"], headless = True
    )
    try:
        page = browser.new_page()
        cdp = page.context.new_cdp_session(page)
        cdp.send("HeapProfiler.enable")
        cdp.send("Runtime.enable")
        errors: list[str] = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.goto(f"{BASE}/smoke-shiki-retention.html", wait_until = "domcontentloaded")
        page.wait_for_function("() => window.__sd && window.__sd.ready", timeout = 120_000)

        seen_block_ms = calibrate(page)
        fixture_hash = page.evaluate(
            "([k, c, s]) => window.__sd.fixtureHash(k, c, s)", [kind, CHARS, 1]
        )

        # Warm-up reply pays the one-off grammar and theme load.
        page.evaluate(
            "(spec) => window.__sd.runOne(spec)",
            {
                "kind": kind,
                "chars": CHARS,
                "seed": 1,
                "ticks": ACCUMULATE_TICKS,
                "tickMs": ACCUMULATE_TICK_MS,
            },
        )
        wait_settled(page)
        page.evaluate("() => window.__sd.teardown()")
        collect_garbage(cdp)
        baseline = float(cdp.send("Runtime.getHeapUsage")["usedSize"])

        for i in range(ACCUMULATE):
            page.evaluate(
                "(spec) => window.__sd.runOne(spec)",
                {
                    "kind": kind,
                    "chars": CHARS,
                    "seed": 2000 + i,
                    "ticks": ACCUMULATE_TICKS,
                    "tickMs": ACCUMULATE_TICK_MS,
                },
            )
            wait_settled(page)
            page.evaluate("() => window.__sd.teardown()")
        collect_garbage(cdp)
        loaded = float(cdp.send("Runtime.getHeapUsage")["usedSize"])
        retained_mb = (loaded - baseline) / MB
        info(f"{LABEL} {kind}: {ACCUMULATE} replies accumulated, retained {retained_mb:+.1f} MB")

        reports = []
        for rep in range(REPEATS):
            page.evaluate("() => window.__sd.framesStart()")
            result = page.evaluate(
                "(spec) => window.__sd.runOne(spec)",
                {
                    "kind": kind,
                    "chars": CHARS,
                    "seed": 5000 + rep,
                    "ticks": MEASURE_TICKS,
                    "tickMs": MEASURE_TICK_MS,
                },
            )
            report = page.evaluate("() => window.__sd.framesStop()")
            report["render_calls"] = result["renderCalls"]
            reports.append(report)
            wait_settled(page)
            page.evaluate("() => window.__sd.teardown()")
            info(
                f"{LABEL} {kind} rep {rep + 1}: fps {report['fps']:.1f}, "
                f"p95 {report['p95FrameMs']:.1f} ms, worst {report['worstFrameMs']:.1f} ms, "
                f"stall {report['longestStallMs']:.1f} ms"
            )

        return {
            "kind": kind,
            "chars": CHARS,
            "accumulated": ACCUMULATE,
            "fixture_hash": fixture_hash,
            "calibration_seen_ms": seen_block_ms,
            "retained_mb": retained_mb,
            "reports": reports,
            "fps": median([r["fps"] for r in reports]),
            "p95_frame_ms": median([r["p95FrameMs"] for r in reports]),
            "worst_frame_ms": median([r["worstFrameMs"] for r in reports]),
            "frames_over_33ms": median([r["framesOver33ms"] for r in reports]),
            "longest_stall_ms": median([r["longestStallMs"] for r in reports]),
            "mean_render_calls": median([float(r["render_calls"]) for r in reports]),
            "page_errors": errors,
        }
    finally:
        browser.close()


def harness_failures(cells: dict) -> list[str]:
    failures = []
    for name, cell in cells.items():
        if cell["page_errors"]:
            failures.append(f"{name}: page errors {cell['page_errors'][:2]}")
        if cell["calibration_seen_ms"] < CALIBRATION_FLOOR_MS:
            failures.append(
                f"{name}: recorder saw only {cell['calibration_seen_ms']:.1f} ms of a "
                f"{CALIBRATION_BLOCK_MS} ms block, so it cannot see a stall"
            )
        if any(r["frames"] < 10 for r in cell["reports"]):
            failures.append(f"{name}: a repetition recorded fewer than 10 frames")
        if cell["kind"] == "code" and cell["mean_render_calls"] < 1:
            failures.append(f"{name}: the highlighter was never called")
    code = cells.get("code")
    if code and code["retained_mb"] < 1.0:
        # Not a jank failure, but it means the arms were never separated, so the comparison is
        # between two identical states and any difference in the table is noise.
        failures.append(
            f"code: accumulation retained only {code['retained_mb']:.2f} MB, "
            "so this run did not build the heap it is supposed to measure against"
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
            # `stream` is the arm under test; `prose` is the control, same shape of work with the
            # highlighter never involved.
            cells["code"] = measure_cell(pw, "stream")
            cells["prose"] = measure_cell(pw, "prose")
    finally:
        if vite is not None:
            stop_process(vite)
            info("vite stopped")

    report = {
        "label": LABEL,
        "chars": CHARS,
        "accumulated": ACCUMULATE,
        "measure_ticks": MEASURE_TICKS,
        "measure_tick_ms": MEASURE_TICK_MS,
        "repeats": REPEATS,
        "cells": cells,
    }
    path = OUT / f"shiki-jank-{LABEL}.json"
    path.write_text(json.dumps(report, indent = 2), encoding = "utf-8")

    print()
    print(
        f"  {'cell':<8}{'retained MB':>13}{'fps':>8}{'p95 ms':>9}"
        f"{'worst ms':>10}{'>33ms':>8}{'stall ms':>10}"
    )
    for name, cell in cells.items():
        print(
            f"  {name:<8}{cell['retained_mb']:>13.1f}{cell['fps']:>8.1f}"
            f"{cell['p95_frame_ms']:>9.1f}{cell['worst_frame_ms']:>10.1f}"
            f"{cell['frames_over_33ms']:>8.0f}{cell['longest_stall_ms']:>10.1f}"
        )
    print()
    info(f"wrote {path}")

    failures = harness_failures(cells)
    for failure in failures:
        info(f"HARNESS FAILURE: {failure}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
