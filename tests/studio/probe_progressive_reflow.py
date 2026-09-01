# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Does content above a detached reader move when it relayouts for a reason that is NOT a widening?

While the mount window is open the viewport has native scroll anchoring disabled, so nothing in
the browser absorbs a `<pre>` swapping in Shiki output, a KaTeX resize or an image landing. This
injects one such reflow of a known size and measures whether the reader moves.

Measured with it: 600px injected above a detached reader mid-build-in moved them the full 600px
before the between-widenings compensation existed, and 0px after, against 0px on the merge base.
Its own first two versions measured nothing, once by sampling before the probe's own smooth-scroll
had finished (4332px of "drift" on both arms) and once by anchoring on the topmost row rather than
a visible one, so read the guards as load-bearing.

Not a gate. Prints and exits 0 on any measurement.

    SMOKE_PORT=5719 python3 tests/studio/probe_progressive_reflow.py

Needs #9016's heavy-thread harness, the same dependency probe_progressive_anchor.py has.
"""

from __future__ import annotations
import json, os, sys
from pathlib import Path
from playwright.sync_api import sync_playwright

TREE = Path(os.environ.get("PM_TREE", Path(__file__).resolve().parents[2])).resolve()
sys.path.insert(0, str(TREE / "tests" / "studio"))
try:
    from playwright_heavy_thread import RECORDER_INIT  # noqa: E402
except ModuleNotFoundError as exc:  # pragma: no cover - the message IS the behaviour
    raise SystemExit(
        "probe_progressive_reflow needs #9016's heavy-thread harness, which is not on this "
        "branch yet: tests/studio/playwright_heavy_thread.py plus "
        "studio/frontend/smoke-heavy-thread.{html,tsx}."
    ) from exc
from _playwright_robust import (  # noqa: E402
    chromium_launch_args,
    start_vite,
    stop_process,
    wait_for_smoke_page,
)

PORT = int(os.environ.get("SMOKE_PORT", "5719"))
PAGE = f"http://127.0.0.1:{PORT}/smoke-heavy-thread.html"
CHARS = int(os.environ.get("SMOKE_HEAVY_CHARS", "300000"))
REPEATS = int(os.environ.get("PM_REPEATS", "3"))
LABEL = os.environ.get("PM_LABEL", "reflow")
GROW_PX = int(os.environ.get("PM_GROW_PX", "600"))
OUT = Path(os.environ.get("PW_ART_DIR", "logs/pm_probe"))
OUT.mkdir(parents = True, exist_ok = True)


def info(m: str) -> None:
    print(f"[pm-reflow] {m}", flush = True)


# Re-open, scroll the reader up, grow one row ABOVE them by a known amount while the window is still open, and watch
RUN_JS = """
async ([growPx, settleMs]) => {
  const api = window.__heavyThread;
  const viewport = api.viewport();
  const before = api.messageCount();
  if (!before || !viewport) return null;
  api.closeThread();
  while (api.messageCount() !== 0) await window.__nextPaint();
  api.openThread();
  while (api.messageCount() === 0) await window.__nextPaint();
  const mountedAtFirstPaint = api.messageCount();
  // Let one widening land so the window is demonstrably open and mid-build.
  const deadline = performance.now() + 6000;
  while (api.messageCount() <= mountedAtFirstPaint && performance.now() < deadline) {
    await window.__nextPaint();
  }
  return { before, mountedAtFirstPaint, mounted: api.messageCount() };
}
"""

GROW_JS = """
async ([growPx, settleMs, before]) => {
  const api = window.__heavyThread;
  const viewport = api.viewport();
  const rows = () => Array.from(viewport.querySelectorAll("[data-role]"));
  const onScreen = rows().find((el) => {
    const b = el.getBoundingClientRect();
    return b.bottom > 0 && b.top < viewport.clientHeight;
  });
  if (!onScreen) return null;
  const above = rows().filter((el) => el.getBoundingClientRect().bottom <= 0);
  const target = above[above.length - 1];
  if (!target) return { skipped: "no row above the fold" };
  const screenTop = () => onScreen.getBoundingClientRect().top;
  const first = screenTop();
  const startedMounted = api.messageCount();
  // The reflow: one row above the reader gets taller, exactly as a <pre> does when Shiki's
  // highlighted output replaces the plain one.
  target.style.paddingTop = `${growPx}px`;
  await window.__nextPaint();
  await window.__nextPaint();
  const afterTwoFrames = screenTop() - first;
  const t0 = performance.now();
  while (performance.now() - t0 < settleMs) await window.__nextPaint();
  return {
    before,
    startedMounted,
    endedMounted: api.messageCount(),
    windowWasOpen: startedMounted < before,
    growPx,
    driftAfterTwoFrames: Math.round(afterTwoFrames),
    driftAtEnd: Math.round(screenTop() - first),
    distanceFromBottom: Math.round(
      viewport.scrollHeight - viewport.scrollTop - viewport.clientHeight),
  };
}
"""


def main() -> int:
    vite = start_vite(PORT)
    out = {"label": LABEL, "tree": str(TREE), "growPx": GROW_PX, "reps": []}
    try:
        wait_for_smoke_page(PAGE, "smoke-heavy-thread-main.tsx", proc = vite, info = info)
        with sync_playwright() as p:
            b = p.chromium.launch(headless = True, args = chromium_launch_args())
            ctx = b.new_context(viewport = {"width": 1280, "height": 900})
            ctx.add_init_script(RECORDER_INIT)
            pg = ctx.new_page()
            pg.goto(PAGE, wait_until = "domcontentloaded")
            pg.wait_for_function("() => Boolean(window.__heavyThread)", timeout = 120_000)
            plan = pg.evaluate("(n) => window.__heavyThread.seed(n)", CHARS)
            pg.wait_for_function(
                "(n) => window.__heavyThread.messageCount() >= n",
                arg = plan["messages"],
                timeout = 300_000,
            )
            pg.evaluate("() => window.__heavyThread.expandTools()")
            for i in range(REPEATS):
                info(f"repetition {i+1}/{REPEATS}")
                opened = pg.evaluate(RUN_JS, [GROW_PX, 2000])
                pg.mouse.move(640, 450)
                pg.mouse.wheel(0, -4000)
                # The viewport is scroll-smooth, so the wheel animates.
                # Let it finish or the probe measures its own gesture: the first version reported 4332px on BOTH arms.
                pg.wait_for_function(
                    """() => { const el = window.__heavyThread.viewport();
                        const settled = window.__rfTop === el.scrollTop;
                        window.__rfTop = el.scrollTop; return settled; }""",
                    timeout = 15_000,
                )
                r = pg.evaluate(GROW_JS, [GROW_PX, 2500, opened["before"]])
                r["mountedAtFirstPaint"] = opened["mountedAtFirstPaint"]
                out["reps"].append(r)
            b.close()
    finally:
        stop_process(vite)
        info("vite stopped")
    (OUT / f"{LABEL}.json").write_text(json.dumps(out, indent = 2), encoding = "utf-8")
    print()
    print(
        f"{'rep':>4} {'windowOpen':>11} {'mounted':>16} {'grewBy':>7} {'drift 2f':>9} {'drift end':>10} {'distBottom':>11}"
    )
    for i, r in enumerate(out["reps"], 1):
        if not r or "skipped" in r:
            print(f"{i:>4} {r}")
            continue
        print(
            f"{i:>4} {str(r['windowWasOpen']):>11} {r['startedMounted']:>6} of {r['before']:<6} "
            f"{r['growPx']:>7} {r['driftAfterTwoFrames']:>9} {r['driftAtEnd']:>10} {r['distanceFromBottom']:>11}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
