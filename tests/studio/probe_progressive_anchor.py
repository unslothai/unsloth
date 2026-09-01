# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Does the page move under a reader who scrolls up while a long thread is still widening?

The #9016 re-open action leaves the viewport pinned to the bottom, so it exercises only the branch
where the autoscroll hook already owns the correction. The branch with no other actor is the
DETACHED one: the user scrolls up the instant the thread opens and the widening keeps inserting
messages ABOVE them. If the correction there is wrong, or missing, what they are reading slides
down the page every widening frame: the regression Open WebUI shipped and reverted
(open-webui#23990).

The probe re-opens the thread, scrolls up hard on the frame the first row appears, then samples the
document-space top of a chosen visible message every frame until the thread has converged. A
correct build holds it still; a broken one shows it walking.

Not a gate. Prints and exits 0 on any measurement; exits non-zero only when the fixture did not
land, which would mean the numbers describe nothing.

It needs #9016's heavy-thread harness, which is not merged, so on this branch it exits with a
message naming the three files it is missing rather than running. That is the same dependency
every measurement in this PR has.

    SMOKE_PORT=5480 python tests/studio/probe_progressive_anchor.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    from playwright_heavy_thread import RECORDER_INIT  # noqa: E402
except ModuleNotFoundError as exc:  # pragma: no cover - the message IS the behaviour
    # #9016 is not merged, so its fixture files are absent and this import is where that becomes visible.
    raise SystemExit(
        "probe_progressive_anchor needs #9016's heavy-thread harness, which is not on this "
        "branch yet: tests/studio/playwright_heavy_thread.py plus "
        "studio/frontend/smoke-heavy-thread.{html,tsx}. Check out that branch, or copy those "
        "three files in, and run this again."
    ) from exc
from _playwright_robust import (  # noqa: E402
    chromium_launch_args,
    start_vite,
    stop_process,
    wait_for_smoke_page,
)

PORT = int(os.environ.get("SMOKE_PORT", "5480"))
BASE = f"http://127.0.0.1:{PORT}"
PAGE = f"{BASE}/smoke-heavy-thread.html"
CHARS = int(os.environ.get("SMOKE_HEAVY_CHARS", "300000"))
REPEATS = int(os.environ.get("SMOKE_ANCHOR_REPEATS", "3"))
LABEL = os.environ.get("SMOKE_LABEL", "tree")
OUT = Path(os.environ.get("PW_ART_DIR", "logs/progressive-anchor"))
OUT.mkdir(parents = True, exist_ok = True)

# How far up to scroll on the frame the thread comes back.
SCROLL_UP_PX = int(os.environ.get("SMOKE_ANCHOR_SCROLL_PX", "4000"))
SETTLE_MS = int(os.environ.get("SMOKE_ANCHOR_SETTLE_MS", "8000"))


def info(message: str) -> None:
    print(f"[anchor] {message}", flush = True)


# Stage 1, in the page: unmount and re-open, returning as soon as the first row paints so the caller can scroll up
REOPEN_JS = """
async () => {
  const api = window.__heavyThread;
  const viewport = api.viewport();
  const before = api.messageCount();
  if (!before || !viewport) return null;
  api.closeThread();
  while (api.messageCount() !== 0) await window.__nextPaint();
  api.openThread();
  while (api.messageCount() === 0) await window.__nextPaint();
  const mountedAtFirstPaint = api.messageCount();

  // Wait for the FIRST widening to land before handing control back to scroll up.
  //
  // Re-opening a thread pins it to the bottom, and that pin arrives a beat after the first row
  // paints: the viewport element is new, so the hook's fresh-attach path scrolls to the bottom and
  // opens a follow window, and thread.initialize pins again on top of that. A wheel issued before
  // those land is simply undone by them, and the probe then measures the pinned branch while
  // believing it measured the detached one. The first two versions of this probe did exactly that
  // and its own end-of-run guard is what caught it.
  const deadline = performance.now() + 4000;
  while (api.messageCount() <= mountedAtFirstPaint && performance.now() < deadline) {
    await window.__nextPaint();
  }
  return {
    before,
    mountedAtFirstPaint,
    mountedBeforeScroll: api.messageCount(),
    widenedBeforeScroll: api.messageCount() > mountedAtFirstPaint,
  };
}
"""

# Stage 3, in the page: pick an on-screen message and sample its position every frame until the thread converges.
SAMPLE_JS = """
async ([before, settleMs]) => {
  const api = window.__heavyThread;
  const viewport = api.viewport();
  if (!viewport) return null;
  const rows = () => Array.from(viewport.querySelectorAll("[data-role]"));
  const anchor = rows().find((el) => {
    const box = el.getBoundingClientRect();
    return box.bottom > 0 && box.top < viewport.clientHeight;
  }) || rows()[0];
  if (!anchor) return null;

  // THE headline is viewport space, because "did the page move under the reader" is a question
  // about where the message is on their screen. Document space is recorded alongside it as
  // context and MUST move: rows inserted above an element move it down the document by exactly
  // their height, whether or not the viewport was corrected. Reading document space as the
  // headline is the mistake the first version of this probe made, and it reported a 110,625px
  // drift on a build that had not moved anything on screen at all.
  const screenTop = () => anchor.getBoundingClientRect().top;
  const docTop = () => anchor.getBoundingClientRect().top + viewport.scrollTop;
  const samples = [];
  const docSamples = [];
  const mounted = [];
  const start = performance.now();
  const first = screenTop();
  const firstDoc = docTop();
  let previous = first;
  let worstStep = 0;
  const mountedAtSampleStart = api.messageCount();
  while (performance.now() - start < settleMs) {
    await window.__nextPaint();
    const now = screenTop();
    const step = Math.abs(now - previous);
    if (step > worstStep) worstStep = step;
    samples.push(Math.round(now - first));
    docSamples.push(Math.round(docTop() - firstDoc));
    mounted.push(api.messageCount());
    previous = now;
    if (api.messageCount() >= before && performance.now() - start > 600) break;
  }

  return {
    before,
    mountedAtSampleStart,
    mountedAtEnd: api.messageCount(),
    totalDriftPx: Math.round(previous - first),
    worstFramePx: Math.round(worstStep),
    documentShiftPx: Math.round(docTop() - firstDoc),
    // Did the sampling window actually contain a widening? If the thread was already whole before
    // the first sample there was nothing to hold still and the repetition proves nothing.
    caughtWidening: mountedAtSampleStart < before,
    scrollTopAtEnd: Math.round(viewport.scrollTop),
    distanceFromBottom: Math.round(
      viewport.scrollHeight - viewport.scrollTop - viewport.clientHeight),
    frames: samples.length,
    trace: samples.slice(0, 40),
    docTrace: docSamples.slice(0, 40),
    mountedTrace: mounted.slice(0, 40),
  };
}
"""


def main() -> int:
    info(f"starting vite dev server on port {PORT}")
    vite = start_vite(PORT)
    results: list[dict] = []
    try:
        wait_for_smoke_page(PAGE, "smoke-heavy-thread-main.tsx", proc = vite, info = info)
        with sync_playwright() as p:
            browser = p.chromium.launch(headless = True, args = chromium_launch_args())
            context = browser.new_context(viewport = {"width": 1280, "height": 900})
            # Reuse #9016's instrumentation: __nextPaint is the double-rAF this samples on, and it must be the SAME
            # clock the re-open column uses or the numbers are not comparable.
            context.add_init_script(RECORDER_INIT)
            page = context.new_page()
            page.goto(PAGE, wait_until = "domcontentloaded")
            page.wait_for_function("() => Boolean(window.__heavyThread)", timeout = 60_000)
            plan = page.evaluate("(n) => window.__heavyThread.seed(n)", CHARS)
            page.wait_for_function(
                "(n) => window.__heavyThread.messageCount() >= n",
                arg = plan["messages"],
                timeout = 300_000,
            )
            page.evaluate("() => window.__heavyThread.expandTools()")
            for repetition in range(1, REPEATS + 1):
                info(f"repetition {repetition}/{REPEATS}")
                opened = page.evaluate(REOPEN_JS)
                if opened is None:
                    info("fixture did not land")
                    return 1
                # A REAL wheel from the input pipeline, over a message rather than the scroll container.
                # A synthetic dispatchEvent on the viewport does not detach: the hook asks
                # innerScrollWillConsumeUpward(e.target) whether something nested will consume the gesture, and an event
                if not opened["widenedBeforeScroll"]:
                    info("no widening landed before the scroll; the fixture is not long enough")
                    return 1
                page.mouse.move(640, 450)
                page.mouse.wheel(0, -SCROLL_UP_PX)
                # The viewport is scroll-smooth, so the wheel animates.
                page.wait_for_function(
                    """() => {
                        const el = window.__heavyThread.viewport();
                        const settled = window.__anchorTop === el.scrollTop;
                        window.__anchorTop = el.scrollTop;
                        return settled;
                    }""",
                    timeout = 10_000,
                )
                out = page.evaluate(SAMPLE_JS, [opened["before"], SETTLE_MS])
                if out is None:
                    info("fixture did not land")
                    return 1
                out["mountedAtFirstPaint"] = opened["mountedAtFirstPaint"]
                out["mountedBeforeScroll"] = opened["mountedBeforeScroll"]
                results.append(out)
            browser.close()
    finally:
        stop_process(vite)
        info("vite stopped")

    payload = {"label": LABEL, "chars": CHARS, "repetitions": results}
    (OUT / f"{LABEL}.json").write_text(json.dumps(payload, indent = 2), encoding = "utf-8")

    print()
    print(
        f"{'rep':>4} {'caught':>7} {'mounted@1st':>13} {'ON SCREEN drift':>16} "
        f"{'worst frame':>12} {'document shift':>15} {'dist from bottom':>17}"
    )
    for i, r in enumerate(results, 1):
        print(
            f"{i:>4} {str(r['caughtWidening']):>7} "
            f"{r['mountedAtFirstPaint']:>5} of {r['before']:<5} "
            f"{r['totalDriftPx']:>16} {r['worstFramePx']:>12} "
            f"{r['documentShiftPx']:>15} {r['distanceFromBottom']:>17}"
        )
    print()
    info(f"wrote {OUT / f'{LABEL}.json'}")

    # The only failure asserted:
    if not any(r["caughtWidening"] for r in results):
        info(
            "PROBE-BROKEN the thread was already whole at the first paint in every repetition, so "
            "nothing was held still and these numbers describe nothing"
        )
        return 1
    if any(r["distanceFromBottom"] < 200 for r in results):
        info(
            "PROBE-BROKEN a repetition ended within 200px of the bottom, so the reader was not "
            "detached and this measured the pinned branch instead"
        )
        return 1
    info("measurement only: no budgets are asserted here, so this exits 0 on any drift.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
