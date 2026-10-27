# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Probe: does a compact tail leave an empty band on the first commit? Prints only; not a gate."""

from __future__ import annotations
import json, os, sys, time
from pathlib import Path
from playwright.sync_api import sync_playwright

TREE = Path(os.environ.get("PM_TREE", Path(__file__).resolve().parents[2])).resolve()
sys.path.insert(0, str(TREE / "tests" / "studio"))
from _playwright_robust import (  # noqa: E402
    chromium_launch_args,
    start_vite,
    stop_process,
    wait_for_smoke_page,
)

PORT = int(os.environ.get("PM_PORT", "5731"))
PAGE = f"http://127.0.0.1:{PORT}/smoke-heavy-thread.html"
CHARS = int(os.environ.get("PM_CHARS", "150000"))
TAIL = int(os.environ.get("PM_TAIL", "24"))
HEIGHTS = [int(h) for h in os.environ.get("PM_HEIGHTS", "900,1080,1440").split(",")]
ENGINES = os.environ.get("PM_ENGINE", "chromium").split(",")
REPEATS = int(os.environ.get("PM_REPEATS", "3"))
LABEL = os.environ.get("PM_LABEL", "tree")
OUT = Path(os.environ.get("PW_ART_DIR", str(TREE / "logs" / "pm_gap")))
OUT.mkdir(parents = True, exist_ok = True)

NEXT_PAINT = """
window.__nextPaint = () => new Promise((resolve) =>
  requestAnimationFrame(() => requestAnimationFrame(() => resolve())));
"""

RUN_JS = """
async ([total, settleFrames]) => {
  const api = window.__heavyThread;
  api.closeThread();
  while (api.messageCount() !== 0) await window.__nextPaint();
  const samples = [];
  const t0 = performance.now();
  api.openThread();
  let stable = 0;
  let last = -1;
  const deadline = performance.now() + 30000;
  while (performance.now() < deadline) {
    await new Promise((r) => requestAnimationFrame(r));
    const rows = api.messageCount();
    if (rows > 0) {
      const m = api.gapMetrics();
      m.t = Math.round(performance.now() - t0);
      samples.push(m);
    }
    if (rows === last && rows >= total) { stable += 1; } else { stable = 0; }
    last = rows;
    if (stable >= settleFrames) break;
  }
  return { total, samples };
}
"""


def info(m: str) -> None:
    print(f"[pm-gap] {m}", flush = True)


def run_engine(pw, engine: str) -> dict:
    browser_type = getattr(pw, engine)
    launch = {"args": chromium_launch_args()} if engine == "chromium" else {}
    browser = browser_type.launch(headless = True, **launch)
    out: dict = {}
    for height in HEIGHTS:
        rounds = []
        context = browser.new_context(viewport = {"width": 1280, "height": height})
        context.add_init_script(NEXT_PAINT)
        page = context.new_page()
        page.goto(PAGE, wait_until = "load", timeout = 180000)
        page.wait_for_function("() => !!window.__heavyThread", timeout = 120000)
        missing = page.evaluate(
            """() => ["seedCompactTail", "gapMetrics"].filter(
                (k) => typeof window.__heavyThread[k] !== "function")"""
        )
        if missing:
            raise SystemExit(
                "the heavy-thread harness on this tree is missing "
                + ", ".join(missing)
                + "; see the module docstring."
            )
        plan = page.evaluate(
            "([c, t]) => window.__heavyThread.seedCompactTail(c, t)", [CHARS, TAIL]
        )
        page.wait_for_function(
            "(n) => window.__heavyThread.messageCount() >= n",
            arg = plan["messages"],
            timeout = 180000,
        )
        page.wait_for_timeout(3000)
        for r in range(REPEATS):
            result = page.evaluate(RUN_JS, [plan["messages"], 6])
            samples = result["samples"]
            first = samples[0]
            # Measured against the settled gap, not zero: the bottom spacer and sticky footer leave a band.
            baseline = samples[-1]["gapBottom"]

            def netgap(s):
                return max(0, s["gapBottom"] - baseline)

            gap0 = netgap(first)
            # Measured to the frame that closes the gap, so a single-frame gap is not 0ms.
            lingering = [s for s in samples if netgap(s) > 8]
            if lingering:
                closed = next(
                    (s["t"] for s in samples if s["t"] > lingering[-1]["t"] and netgap(s) <= 8),
                    samples[-1]["t"],
                )
                gap_ms = closed - lingering[0]["t"]
            else:
                gap_ms = 0
            rounds.append(
                {
                    "firstPaintMs": first["t"],
                    "firstRows": first["mountedRows"],
                    "clientHeight": first["clientHeight"],
                    "mountedHeight": first["mountedHeight"],
                    "spacerHeight": first["spacerHeight"],
                    "gapTop0": first["gapTop"],
                    "gapBottom0": first["gapBottom"],
                    "netGap0": gap0,
                    "maxNetGap": max(netgap(s) for s in samples),
                    "gapPersistMs": gap_ms,
                    "gapFrames": len(lingering),
                    "framesToConverge": len(samples),
                    "convergedMs": samples[-1]["t"],
                    "finalRows": samples[-1]["mountedRows"],
                    "finalNetGap": netgap(samples[-1]),
                }
            )
            if r == 0:
                (OUT / f"{LABEL}-{engine}-{height}-timeline.json").write_text(
                    json.dumps(samples, indent = 1), encoding = "utf-8"
                )
        page.evaluate(
            """async () => {
                const api = window.__heavyThread;
                api.closeThread();
                while (api.messageCount() !== 0) await window.__nextPaint();
                api.openThread();
                while (api.messageCount() === 0) await new Promise(r => requestAnimationFrame(r));
            }"""
        )
        page.screenshot(path = str(OUT / f"{LABEL}-{engine}-{height}-firstcommit.png"))
        out[height] = rounds
        # Written after every height so a browser dying on the tallest viewport keeps earlier results.
        (OUT / f"{LABEL}-{engine}-rounds.json").write_text(
            json.dumps(out, indent = 1), encoding = "utf-8"
        )
        context.close()
    browser.close()
    return out


def main() -> int:
    vite = start_vite(PORT)
    try:
        wait_for_smoke_page(PAGE, "smoke-heavy-thread-main.tsx", proc = vite, info = info)
        results = {}
        with sync_playwright() as pw:
            for engine in ENGINES:
                info(f"engine {engine}")
                results[engine] = run_engine(pw, engine)
        path = OUT / f"{LABEL}-results.json"
        path.write_text(json.dumps(results, indent = 1), encoding = "utf-8")
        info(f"wrote {path}")
        for engine, byh in results.items():
            for height, rounds in byh.items():
                med = lambda k: sorted(r[k] for r in rounds)[len(rounds) // 2]
                info(
                    f"{LABEL} {engine} h={height}: rows={med('firstRows')} "
                    f"mounted={med('mountedHeight')}px client={med('clientHeight')}px "
                    f"spacer={med('spacerHeight')}px netGap0={med('netGap0')}px "
                    f"maxNetGap={med('maxNetGap')}px persist={med('gapPersistMs')}ms "
                    f"frames={med('gapFrames')} converged={med('convergedMs')}ms "
                    f"finalRows={med('finalRows')}"
                )
    finally:
        stop_process(vite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
