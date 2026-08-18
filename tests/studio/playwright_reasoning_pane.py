# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""What a long THINKING block costs while it streams, sampled the way the field trace was.

WHY THIS FILE EXISTS

tests/studio/playwright_heavy_thread.py measures a heavy thread, and its fixture holds text,
code fences, tool calls, artifacts and images. It holds no reasoning content, and it cannot
usefully hold any: it seeds a FINISHED thread, a finished reasoning group resolves to closed
(features/chat/utils/reasoning-visibility.ts), and Radix unmounts a closed CollapsibleContent.
So a seeded reasoning part contributes zero DOM and zero cost. Every bottleneck number taken on
that harness was taken on a thread with no thinking block in it.

The field report this file answers is a single long generation on Unsloth Desktop (Arch,
Wayland, WebKitGTK 2.52.5), sampled every 5 seconds:

    t=5s     fps 59.4   reasoningChars  2,256   reasoningCodeSpans      0   elements    578
    t=130s   fps 39.7   reasoningChars 45,943   reasoningCodeSpans  3,520   elements  4,755
    t=216s   fps 20.5   reasoningChars 73,178   reasoningCodeSpans 11,875   elements 13,688
    t=241s   fps 18.2   reasoningChars 80,434   reasoningCodeSpans 14,433   elements 16,513
    t=271s   fps 28.3   reasoningChars 90,262   reasoningCodeSpans 16,186   elements 18,536
    t=276s   COMPLETED, reasoningCodeSpans 0, elements 621, fps recovers

Three things that trace establishes, which this harness therefore does not re-derive:

  * It is not the forced software renderer. WEBKIT_DMABUF_RENDERER_FORCE_SHM=1 would be
    uniformly slow from t=0. This starts at a clean 59.4 fps and degrades with content.
  * `reasoningCodeSpans` counts `pre code span` inside `.aui-reasoning-text`, so those 16,186
    highlight spans are inside the THINKING pane, not the answer.
  * The collapse at completion is the proof of location: 18,536 elements become 621 and fps
    recovers, in the same second, without the thread being unloaded.

TWO PRECONDITIONS. GET EITHER WRONG AND THIS FILE MEASURES ALMOST NOTHING.

1. THE ARRIVAL RATE MUST BE THE USER'S. The capture delivered ~325 characters a second, which at
   24 characters a chunk is one chunk every 73ms, and that is the default here. The first version
   of this file used a 2ms gap, about four times a real token rate. At 2ms the renderer is the
   bottleneck from the very first chunk, so the run opens at 45 fps instead of 60 and there is no
   healthy early baseline left to degrade FROM: it measured a 1.19x spread where the capture shows
   3x. The whole shape being reproduced is "idle between chunks at the start, no longer idle at the
   end", and a fixture that is saturated at t=0 cannot show it.

2. ON AN IDLE MACHINE, FRAME RATE IS NOT SENSITIVE ENOUGH. Measured on the development box: 65%
   main-thread IDLE at 15,000 highlight spans. The work per chunk really does grow with the
   content, but there is so much headroom that it never crosses a frame budget, so fps sits pinned
   at 60 for the whole run and reading fps alone calls it "no effect". The metric that moves is
   `busy%`: main-thread blocked time per window, measured against a CALIBRATED timer clamp. To see
   the frame-rate collapse itself rather than the cost that causes it, throttle:
   SMOKE_RP_CPU_THROTTLE=6 on Chromium reproduced 47.1 fps -> 9.2 fps on this fixture.

WHY THIS CANNOT BE A SEEDED THREAD (the trap that makes this file necessary)

playwright_heavy_thread.py seeds a FINISHED thread through `thread.import`. Adding reasoning
content to that fixture looks like the obvious way to cover this path and it would have shown
NOTHING, convincingly: a finished reasoning group resolves to closed
(features/chat/utils/reasoning-visibility.ts -> `resolveReasoningOpen` returns false when
`isStreaming` is false and the user has not opened it by hand), Radix unmounts a closed
`CollapsibleContent`, and the reasoning part therefore contributes ZERO DOM and zero cost. The
cost exists only while the block streams. That is why this file streams.

WHAT IS MEASURED, AND WHY NOT longTasks

The trace's `longTasks` column read 0 in every sample. That is not an absence of jank: WebKit
never shipped the Long Tasks API, so a `longtask` PerformanceObserver is accepted and then never
fires. Support is read from `PerformanceObserver.supportedEntryTypes`, never from whether
observe() throws, and the portable signal is EVENT LOOP LAG: the largest gap between ticks of a
1ms setTimeout loop. The main thread cannot answer the timer while it is busy, so the gap is the
block. Resolution is the timer clamp, about 4ms, far below anything a user feels.

fps is counted from requestAnimationFrame ticks in the sample window, which is what the field
sampler did.

SAMPLING RUNS IN THE PAGE, NOT IN THE DRIVER

`document.querySelectorAll(".aui-reasoning-text pre code span")` over 16,000 spans is not free,
and a driver that pays a CDP round trip for it every window would put its own cost inside the
number. The sampler is a setInterval in the page, it records its own cost as `sample_ms`, and
that column is printed so a reader can check that the measurement is not the measured.

INSTRUMENTATION (optional, SMOKE_RP_INSTRUMENT=1)

ReasoningText installs a MutationObserver over its whole growing subtree and answers every
mutation with `el.scrollTop = el.scrollHeight` (studio/frontend/src/components/assistant-ui/
reasoning.tsx). Reading scrollHeight forces synchronous layout. To measure that directly rather
than infer it, the optional instrumentation wraps the `scrollHeight` getter on Element.prototype
and counts reads and the wall time inside them, split by whether the element is the reasoning
pane. It also counts MutationObserver callback invocations.

That wrapper perturbs: it adds two performance.now() calls per read. So it is OFF by default,
and the honest way to use it is to run the same size with and without and check the uninstru-
mented run degrades on its own. The driver prints which mode it ran in.

WHAT ELSE IS MEASURED, AND WHY EACH ONE IS NOT ALREADY COVERED

  * FRAMES OVER 33ms, per window, as a count and as a share of that window's frames. p95 frame
    time says how bad a bad frame was; it does not say how many there were, and the two come
    apart in both directions. One 400ms stall among 59 clean frames barely moves p95. Sixty
    frames all landing at 34ms move p95 to 34ms with no single frame a user would call a stall.

  * PROCESS RSS. `heapMB` is `performance.memory.usedJSHeapSize`: Chromium only, JS heap only,
    and deliberately quantised. The 22,500-character calibration run of this file reported
    exactly 98MB for all ten samples while the element count went from 30 to 2,747. The DOM,
    the layout tree and the highlighter's retained strings are all outside it. So the browser
    process AND its renderer children are sampled from the OS with psutil, and what is reported
    as the result is `rss_growth_mb`, peak minus the first reading of the same run. Where RSS
    cannot be read it is null with a printed reason, NEVER 0.

  * TIME TO SETTLE. `collapse_ms` is a DOM predicate: it goes true when the spans reach zero,
    which can happen while the main thread is still finishing the teardown. Settle is the wall
    time from `streamDone` until two consecutive 250ms windows each show under 10% blocked time.
    Null, and said out loud, if it never happens inside the collapse budget.

THE CPU LADDER AND ITS CONTROL (SMOKE_RP_CPU_LADDER, CHROMIUM ONLY)

SMOKE_RP_CPU_LADDER=1,2,4 runs the same cell once per CDP throttling rate. One throttled run
answers "is this box too fast to notice". It does not answer "is the number I am reading wired
up correctly", and a metric that is wired up wrongly is perfectly capable of producing one
plausible figure.

So the ladder carries a CONTROL that must NOT move: the page's own `Date.now()` elapsed over the
run divided by the driver's `time.monotonic()` elapsed over the same interval. Throttling slows
how fast V8 executes; it does not touch the wall clock, so this ratio is 1.0 at every rate.
Arrivals per second is NOT a valid control and is not used as one -- it is paced by
`await sleep(gapMs)` on the main thread and it already falls from 118/s to 52/s inside a single
unthrottled run. Sampler windows per wall second is main-thread paced for the same reason and is
printed as a diagnostic only. When the control moves by more than SMOKE_RP_CONTROL_TOLERANCE_PCT
(10 by default) the run says loudly that the MEASUREMENT is wrong, not that the page is slow.

Run:
    python tests/studio/playwright_reasoning_pane.py
    SMOKE_RP_CPU_LADDER=1,2,4 SMOKE_RP_ENGINES=chromium \
      python tests/studio/playwright_reasoning_pane.py
    SMOKE_RP_CHARS=30000,90000 SMOKE_RP_ENGINES=chromium python tests/studio/playwright_reasoning_pane.py
    SMOKE_RP_INSTRUMENT=1 SMOKE_RP_ENGINES=chromium python tests/studio/playwright_reasoning_pane.py

It starts and stops its own vite dev server. Point it at one you already have with
SMOKE_BASE_URL, or move the port with SMOKE_PORT.

RUN ONE ENGINE PER INVOCATION IN CI, UNDER AN EXTERNAL TIMEOUT, for the reason
playwright_heavy_thread.py's docstring gives at length: Playwright's sync API blocks inside a
greenlet, so no in-process timeout can bound a wedged engine and only the process bound works.

THIS HARNESS MEASURES, IT DOES NOT GATE on timings. It exits non-zero only when it is not
measuring what it claims: the fixture did not reach the trace's span count, the reasoning pane
never opened, the stream did not finish, or the completion collapse did not happen.

The dev server is a limitation to read the numbers against: React runs in development mode,
nothing is minified, vite serves unbundled modules. Absolute fps is therefore lower than a
packaged Studio's. The SHAPE across the run, and the ratio between the start and the end of the
same run, are what this file is for.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

# Optional, and the harness stays useful without it. Everything psutil is used for here is
# reported as null with a printed explanation when it is missing, never as zero: a zero RSS
# reads as "the browser used no memory", which is the one thing it cannot mean.
try:
    import psutil
except ImportError:  # pragma: no cover - depends on the environment, not on the code
    psutil = None

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _playwright_robust import (  # noqa: E402
    chromium_launch_args,
    start_vite,
    stop_process,
    wait_for_smoke_page,
)

PORT = int(os.environ.get("SMOKE_PORT", "5219"))
_EXTERNAL = os.environ.get("SMOKE_BASE_URL", "").strip().rstrip("/")
BASE = _EXTERNAL or f"http://127.0.0.1:{PORT}"
OWNS_SERVER = not _EXTERNAL
LABEL = os.environ.get("SMOKE_LABEL", "tree")
OUT = Path(os.environ.get("PW_ART_DIR", "logs/playwright-reasoning-pane"))
OUT.mkdir(parents = True, exist_ok = True)

# Characters of REASONING content. The largest is the trace's own 90,000.
SIZES = sorted(int(n) for n in os.environ.get("SMOKE_RP_CHARS", "22500,45000,90000").split(","))
ENGINES = [
    e.strip() for e in os.environ.get("SMOKE_RP_ENGINES", "chromium,webkit").split(",") if e.strip()
]
# Characters per fence. This is the span DENSITY knob: the trace's 90,262 characters carried
# 16,186 spans, i.e. 5.6 characters per span, which only happens if the content is nearly all
# fenced code. The driver reports the density it achieved so this can be tuned against the trace
# instead of guessed.
FENCE_CHARS = int(os.environ.get("SMOKE_RP_FENCE_CHARS", "1800"))
# Prose between the fences. Prose is characters without spans, so this is the other half of the
# density calibration. 1,250 against a 1,800-character fence was measured to land on the trace's
# ~5.6 characters per span; an all-fence fixture measured 3.47, i.e. half again as much work per
# character as the user's own content.
PROSE_CHARS = int(os.environ.get("SMOKE_RP_PROSE_CHARS", "1250"))
# Characters of fence-free thinking before the first code fence. The field capture held a flat
# 60.0 fps and exactly 0 highlight spans for 33,348 characters, and started losing frames at the
# first CLOSED fence, so a fixture without this stretch cannot show where the onset is. Scaled to
# the same quarter of the body by the driver when the size is not the capture's own.
PREAMBLE_CHARS = int(os.environ.get("SMOKE_RP_PREAMBLE_CHARS", "0"))
CHUNK_CHARS = int(os.environ.get("SMOKE_RP_CHUNK_CHARS", "24"))
# THE ARRIVAL RATE IS PART OF THE FIXTURE, and getting it wrong is how the first version of this
# file failed to reproduce anything.
#
# The trace delivered 90,262 characters in 276 seconds: 327 characters a second, which at 24
# characters a chunk is one chunk every 73ms. The first run here used a 2ms gap, i.e. about four
# times the user's token rate, and it measured a 1.19x spread on Chromium instead of the trace's
# 3.3x. The reason is not subtle: at 2ms the renderer is the bottleneck from the very first
# chunk, so the run opens at 45 fps rather than at 59 and there is no healthy early baseline left
# to degrade FROM. The trace's whole shape is "idle between chunks at the start, no longer idle
# at the end", and a fixture that is saturated at t=0 cannot show it.
#
# So the default is the user's own rate. It makes a 90,000-character run take about as long as
# the user's did, which is the point.
GAP_MS = int(os.environ.get("SMOKE_RP_GAP_MS", "73"))
SAMPLE_MS = int(os.environ.get("SMOKE_RP_SAMPLE_MS", "1000"))
RUN_TIMEOUT_MS = int(os.environ.get("SMOKE_RP_TIMEOUT_MS", "900000"))
# How long the completion collapse is allowed to take before the harness calls it a failure to
# collapse rather than a slow one.
COLLAPSE_TIMEOUT_MS = int(os.environ.get("SMOKE_RP_COLLAPSE_TIMEOUT_MS", "60000"))
INSTRUMENT = os.environ.get("SMOKE_RP_INSTRUMENT", "0") == "1"
# Whether to require that the DOM ends up holding the capture's highlight spans.
#
# On by default, because on an unmodified tree a run that does not build them is measuring the
# wrong content. It has to be switchable, though, and the reason is not hypothetical: a candidate
# fix that bounds what the pane keeps mounted ends the run with 511 spans instead of 16,000, and
# this check failed it as "the fixture is not the capture's content" when the fixture was exactly
# right and the FIX was what removed the spans. What the fixture owes is that it SENT the content;
# what the DOM does with it is the measurement.
EXPECT_SPANS = os.environ.get("SMOKE_RP_EXPECT_SPANS", "1") == "1"
# Upper bound on what the pane is allowed to keep mounted, in characters. Unset means no bound.
#
# The counterpart to EXPECT_SPANS, and the reason it exists: turning the span floor off to measure
# a windowed tree removes the only check that the pane's CONTENT is what it should be, and every
# other assertion here is satisfied by a pane holding the entire body. A run with the window
# regressed away would then be green. This is the assertion that is false on the tree the window
# is supposed to have fixed and true on the one it did.
MAX_PANE_CHARS = int(os.environ.get("SMOKE_RP_MAX_PANE_CHARS", "0"))
# Headed, under a real display, is not a nicety here.
#
# Headless has NO COMPOSITOR. Everything this file measures about layout and script is the same
# either way, but the cost of PAINTING a large composited layer and getting it to a display is
# not measured at all in headless, and on the platform the field trace came from that path is
# deliberately the slow one: studio/src-tauri/src/linux_webkit.rs sets
# WEBKIT_DMABUF_RENDERER_FORCE_SHM=1, which forces WebKitGTK's software transport under Wayland.
# So a headless run that shows no degradation has NOT ruled the renderer out; it has not looked.
#
#     SMOKE_RP_HEADLESS=0 WEBKIT_DMABUF_RENDERER_FORCE_SHM=1 \
#       xvfb-run -s "-screen 0 1280x900x24" python tests/studio/playwright_reasoning_pane.py
HEADLESS = os.environ.get("SMOKE_RP_HEADLESS", "1") != "0"
# CHROMIUM ONLY, and off by default, because it is CDP and switching it on makes the chromium
# column incomparable with the webkit one.
#
# It is here because the first paced runs found 65% main-thread IDLE at 15,000 highlight spans on
# this box: the work per chunk really does grow with the content, but there is so much headroom
# that it never crosses a frame budget, so fps stays pinned at 60. The field trace came from a
# machine that was also running the generation. Throttling asks the question that separates "this
# work does not grow" from "this box is too fast to notice": at 4x or 6x, does the SAME curve
# cross the budget and produce the trace's shape?
CPU_THROTTLE = float(os.environ.get("SMOKE_RP_CPU_THROTTLE", "1"))
# The ladder: the SAME cell run once per throttle rate, e.g. SMOKE_RP_CPU_LADDER=1,2,4.
#
# One throttled run answers "is this box too fast to notice", which is what SMOKE_RP_CPU_THROTTLE
# is for. It does not answer "does the number I am reading respond to load the way it should",
# and a metric that is wired up wrongly is perfectly capable of producing one plausible figure.
# A ladder does: the arm at rate 1 is the control arm for the arms above it, and the direction
# and rough size of the move is a prediction the harness can be checked against.
#
# CHROMIUM ONLY. It is CDP (Emulation.setCPUThrottlingRate); WebKit and Firefox have no
# equivalent, and the printed output says so on every ladder run rather than silently giving a
# non-Chromium engine one unthrottled arm that looks like a flat ladder.
CPU_LADDER = [
    float(r.strip()) for r in os.environ.get("SMOKE_RP_CPU_LADDER", "").split(",") if r.strip()
]
# How far the throttle CONTROL is allowed to move across the ladder before the run says the
# measurement is wrong. See CONTROL in measure_cell for what the control is and why.
CONTROL_TOLERANCE_PCT = float(os.environ.get("SMOKE_RP_CONTROL_TOLERANCE_PCT", "10"))

# The trace, for the table's comparison column. Not a budget: different machine, different
# engine, different build.
TRACE = [
    {"t": 5, "fps": 59.4, "chars": 2256, "spans": 0, "elements": 578},
    {"t": 130, "fps": 39.7, "chars": 45943, "spans": 3520, "elements": 4755},
    {"t": 216, "fps": 20.5, "chars": 73178, "spans": 11875, "elements": 13688},
    {"t": 241, "fps": 18.2, "chars": 80434, "spans": 14433, "elements": 16513},
    {"t": 271, "fps": 28.3, "chars": 90262, "spans": 16186, "elements": 18536},
]


def info(message: str) -> None:
    print(f"[reasoning-pane] {message}", flush = True)


# Installed before anything else runs. Counts frames and event loop lag continuously, and lets
# the page sampler close a window and read the totals since the last one.
#
# The rAF wrapper COUNTS, it does not pump, so fps stays the page's own frame rate.
RECORDER_INIT = """
(() => {
  const nativeRaf = window.requestAnimationFrame.bind(window);
  window.__nativeRaf = nativeRaf;

  const R = {
    frames: 0,
    frameGaps: [],
    maxLagMs: 0,
    lagTicks: 0,
    lagSumMs: 0,
    blockedMs: 0,
    // The same blocked time again, but NEVER reset. `blockedMs` is drained by __rpWindow, and
    // the settle watch below has to read blocked time on its own 250ms cadence while the 1000ms
    // window sampler is still running. Two readers draining one accumulator would each see a
    // fraction of the block, so the settle watch reads deltas of this monotonic total instead
    // and disturbs the window sampler not at all.
    blockedTotalMs: 0,
    clampMs: 4.0,
    longTaskSupported: Boolean(
      (PerformanceObserver.supportedEntryTypes || []).includes("longtask"),
    ),
    longTasks: 0,
    longTaskMs: 0,
  };
  window.__rp = R;

  // ONE self-rescheduling rAF loop is the frame counter, and requestAnimationFrame is NOT
  // wrapped. A wrapper that increments per callback counts the page's frame once for the loop
  // and once more for every rAF the app happened to schedule in that frame, which is a frame
  // rate that rises with how busy the app is -- the first version of this file reported 888 fps
  // on a 60Hz page for exactly that reason.
  let lastFrame = performance.now();
  const frame = () => {
    const now = performance.now();
    R.frames += 1;
    R.frameGaps.push(now - lastFrame);
    lastFrame = now;
    nativeRaf(frame);
  };
  nativeRaf(frame);

  // 1ms setTimeout, not a MessageChannel ping-pong: the MessageChannel version ticks ~150k/s and
  // halves Firefox's frame rate before any app code runs. This ticks ~150/s and costs nothing on
  // any engine. The clamp puts a ~4ms floor under the resolution, far below a felt stall.
  // The clamp is CALIBRATED, not assumed. `setTimeout(fn, 1)` is clamped to about 4ms by spec
  // but the actual floor differs by engine and by build, and the blocked-time figure below is a
  // subtraction against it, so a guessed 4.0 would show phantom block on one engine and hide
  // real block on another. The first 60 gaps of an idle page ARE the floor; the median of them
  // is taken as the clamp and reported, so a reader can see what was subtracted.
  const CALIBRATION_TICKS = 60;
  const calibration = [];
  let lastTick = performance.now();
  const tick = () => {
    const now = performance.now();
    const gap = now - lastTick;
    lastTick = now;
    if (calibration.length < CALIBRATION_TICKS) {
      calibration.push(gap);
      if (calibration.length === CALIBRATION_TICKS) {
        const sorted = calibration.slice().sort((a, b) => a - b);
        R.clampMs = sorted[Math.floor(sorted.length / 2)];
      }
    } else {
      R.lagTicks += 1;
      R.lagSumMs += gap;
      // Everything above the clamp is time the main thread could not answer a timer it had
      // already scheduled, i.e. time it was busy. Summed over a window this is a far more
      // sensitive signal than frame rate: a page with 65% idle still paints every frame on time,
      // so fps stays pinned at 60 while the work per chunk quietly triples. That is exactly the
      // regime this harness first landed in, and reading fps alone would have called it "no
      // effect".
      R.blockedMs += Math.max(0, gap - R.clampMs);
      R.blockedTotalMs += Math.max(0, gap - R.clampMs);
      if (gap > R.maxLagMs) R.maxLagMs = gap;
    }
    setTimeout(tick, 1);
  };
  setTimeout(tick, 1);

  // Recorded as a cross-check and never as the headline. Chromium alone lists `longtask` in
  // supportedEntryTypes; on WebKit and Gecko observe() is accepted and never fires, which reads
  // as "no jank" rather than as "no measurement". That is exactly the trap the field trace's
  // all-zero longTasks column fell into.
  if (R.longTaskSupported) {
    try {
      new PerformanceObserver((list) => {
        for (const e of list.getEntries()) {
          R.longTasks += 1;
          R.longTaskMs += e.duration;
        }
      }).observe({ type: "longtask", buffered: false });
    } catch (e) {
      R.longTaskSupported = false;
    }
  }

  window.__rpWindow = () => {
    // Frame TIME as well as frame rate. Headless Chromium has no vsync, so it runs the rAF loop
    // as fast as it can and an unloaded page reports far more than 60 fps; the absolute number
    // is then not a user-visible frame rate at all. The ratio within one run still is, and p95
    // frame time is the number that stays meaningful on an engine with no display.
    const gaps = R.frameGaps.slice().sort((a, b) => a - b);
    const at = (q) => (gaps.length === 0 ? null : gaps[Math.min(gaps.length - 1,
      Math.floor(gaps.length * q))]);
    // Frames that took longer than 33ms, i.e. longer than two frames of a 60Hz display.
    //
    // p95 frame time answers "how bad is a bad frame"; this answers "how many of them were
    // there", and the two come apart. A window with one 400ms stall and 59 clean frames has a
    // p95 barely above 16.7ms, and a window where every frame lands at 34ms has a p95 of 34ms
    // and no single frame a user would call a stall. Both are jank and only the count separates
    // them. The threshold is the one tests/studio/playwright_stream_pacing.py already uses
    // (smoke-stream-pacing-main.tsx, `framesOver33ms`), so the two harnesses stay comparable.
    let over33 = 0;
    for (const g of gaps) if (g > 33) over33 += 1;
    const out = {
      frames: R.frames,
      framesOver33: over33,
      // As a share of the frames actually observed in the window, because the denominator is not
      // fixed: headless Chromium has no vsync and runs the rAF loop as fast as it can, so the
      // frame COUNT per window varies by engine and by load and a raw count alone is not
      // comparable across throttle rates.
      framesOver33Pct:
        gaps.length === 0 ? null : Math.round((over33 / gaps.length) * 1000) / 10,
      p50FrameMs: at(0.5),
      p95FrameMs: at(0.95),
      maxFrameMs: gaps.length === 0 ? null : gaps[gaps.length - 1],
      maxLagMs: R.maxLagMs,
      lagTicks: R.lagTicks,
      lagSumMs: R.lagSumMs,
      blockedMs: R.blockedMs,
      clampMs: R.clampMs,
      longTasks: R.longTasks,
      longTaskMs: R.longTaskMs,
    };
    R.frames = 0;
    R.frameGaps = [];
    R.maxLagMs = 0;
    R.lagTicks = 0;
    R.lagSumMs = 0;
    R.blockedMs = 0;
    R.longTasks = 0;
    R.longTaskMs = 0;
    return out;
  };

  // TIME TO SETTLE: how long after the stream stops before the page is quiet again.
  //
  // The trace's completion row is not instantaneous. Unmounting ~16,000 nodes is itself a
  // main-thread job that scales with the content, and the trace's own worst event loop lag,
  // 576ms, is AT completion. `collapse_ms` above already measures when the spans reach zero,
  // but that is a DOM predicate: it can go true while the main thread is still busy finishing
  // the teardown, the highlighter is still resolving, and React is still committing. So the
  // quiet is measured separately, from the same blocked-time signal everything else here uses.
  //
  // Definition, fixed by the driver's contract: two CONSECUTIVE 250ms windows each with blocked
  // time under 10% of the window. Two, not one, because a single quiet window happens in the
  // middle of teardown between two bursts. 250ms, because the 1000ms sampler is far too coarse
  // for a quantity that is often under a second.
  //
  // The clock starts when `streamState().done` flips, polled at 25ms. Polling, not a callback,
  // because the fixture has no completion hook; the resulting bias is at most one poll interval
  // and it is the same bias at every throttle rate.
  const SETTLE_WINDOW_MS = 250;
  const SETTLE_BUSY_PCT = 10;
  const SETTLE_CONSECUTIVE = 2;
  window.__rpArmSettleWatch = () => {
    const S = {
      armedAt: performance.now(),
      doneAt: null,
      settledAtMs: null,
      windows: [],
      windowMs: SETTLE_WINDOW_MS,
      busyPct: SETTLE_BUSY_PCT,
      consecutive: SETTLE_CONSECUTIVE,
    };
    window.__rpSettle = S;
    const startWatching = () => {
      S.doneAt = performance.now();
      let lastAt = S.doneAt;
      let lastBlocked = R.blockedTotalMs;
      let quiet = 0;
      const watch = setInterval(() => {
        const now = performance.now();
        const elapsed = now - lastAt;
        const blocked = R.blockedTotalMs - lastBlocked;
        lastAt = now;
        lastBlocked = R.blockedTotalMs;
        const pct = elapsed <= 0 ? 0 : (blocked / elapsed) * 100;
        S.windows.push({
          at_ms: Math.round(now - S.doneAt),
          busy_pct: Math.round(pct * 10) / 10,
        });
        quiet = pct < SETTLE_BUSY_PCT ? quiet + 1 : 0;
        if (quiet >= SETTLE_CONSECUTIVE && S.settledAtMs === null) {
          S.settledAtMs = Math.round(now - S.doneAt);
          clearInterval(watch);
        }
      }, SETTLE_WINDOW_MS);
      S.stop = () => clearInterval(watch);
    };
    const poll = setInterval(() => {
      try {
        if (window.__reasoningPane && window.__reasoningPane.streamState().done) {
          clearInterval(poll);
          startWatching();
        }
      } catch (e) {
        /* the fixture is not up yet */
      }
    }, 25);
    S.disarm = () => clearInterval(poll);
  };
})();
"""

# Optional and off by default, because it perturbs what it measures.
#
# ReasoningText answers every mutation of its subtree with `el.scrollTop = el.scrollHeight`.
# Reading scrollHeight forces synchronous layout of everything above it. This counts those reads
# and times them, split by whether the element IS the reasoning pane, so "the autoscroll observer
# forces a full layout per chunk" becomes a measured number rather than a reading of the source.
INSTRUMENT_INIT = """
(() => {
  const I = {
    scrollHeightReads: 0,
    scrollHeightMs: 0,
    reasoningReads: 0,
    reasoningMs: 0,
    scrollTopWrites: 0,
    reasoningScrollTopWrites: 0,
    mutationCallbacks: 0,
    mutationRecords: 0,
  };
  window.__rpInstr = I;

  const isPane = (el) => {
    try {
      return Boolean(el && el.classList && el.classList.contains("aui-reasoning-text"));
    } catch (e) {
      return false;
    }
  };

  const sh = Object.getOwnPropertyDescriptor(Element.prototype, "scrollHeight");
  if (sh && sh.get) {
    Object.defineProperty(Element.prototype, "scrollHeight", {
      configurable: true,
      enumerable: sh.enumerable,
      get() {
        const t0 = performance.now();
        const v = sh.get.call(this);
        const dt = performance.now() - t0;
        I.scrollHeightReads += 1;
        I.scrollHeightMs += dt;
        if (isPane(this)) {
          I.reasoningReads += 1;
          I.reasoningMs += dt;
        }
        return v;
      },
    });
  }

  const st = Object.getOwnPropertyDescriptor(Element.prototype, "scrollTop");
  if (st && st.set) {
    Object.defineProperty(Element.prototype, "scrollTop", {
      configurable: true,
      enumerable: st.enumerable,
      get() {
        return st.get.call(this);
      },
      set(value) {
        I.scrollTopWrites += 1;
        if (isPane(this)) I.reasoningScrollTopWrites += 1;
        st.set.call(this, value);
      },
    });
  }

  const NativeMO = window.MutationObserver;
  if (NativeMO) {
    window.MutationObserver = function (callback) {
      return new NativeMO((records, observer) => {
        I.mutationCallbacks += 1;
        I.mutationRecords += records.length;
        return callback(records, observer);
      });
    };
    window.MutationObserver.prototype = NativeMO.prototype;
  }

  window.__rpInstrWindow = () => {
    const out = { ...I };
    for (const k of Object.keys(I)) I[k] = 0;
    return out;
  };
})();
"""

# The in-page sampler. One setInterval, the trace's columns, and its own cost recorded so a
# reader can rule the measurement out as the cause.
SAMPLER = """
(({ sampleMs }) => {
  const rows = [];
  window.__rpRows = rows;
  const t0 = performance.now();
  let lastAt = t0;
  let seenArrivals = 0;
  const id = setInterval(() => {
    const at = performance.now();
    const w = window.__rpWindow();
    const s = window.__reasoningPane.sample();
    const instr = window.__rpInstrWindow ? window.__rpInstrWindow() : null;
    const elapsed = at - lastAt;
    lastAt = at;
    const lastArrivals = seenArrivals;
    seenArrivals = s.arrivals;
    rows.push({
      t_ms: Math.round(at - t0),
      // Absolute OS wall clock at the moment the window closed. Not a metric: it is what the
      // driver's throttle CONTROL is computed from, see CONTROL in measure_cell.
      wall_ms: Date.now(),
      fps: Math.round((w.frames / (elapsed / 1000)) * 10) / 10,
      p50_frame_ms: w.p50FrameMs === null ? null : Math.round(w.p50FrameMs * 10) / 10,
      p95_frame_ms: w.p95FrameMs === null ? null : Math.round(w.p95FrameMs * 10) / 10,
      max_frame_ms: w.maxFrameMs === null ? null : Math.round(w.maxFrameMs * 10) / 10,
      frames_over_33: w.framesOver33,
      frames_over_33_pct: w.framesOver33Pct,
      max_lag_ms: Math.round(w.maxLagMs * 10) / 10,
      // Percent of the window the main thread was busy, from the blocked time above. This is the
      // column that moves when fps does not.
      busy_pct: Math.round((w.blockedMs / elapsed) * 1000) / 10,
      clamp_ms: Math.round(w.clampMs * 100) / 100,
      mean_lag_ms:
        w.lagTicks === 0 ? null : Math.round((w.lagSumMs / w.lagTicks) * 10) / 10,
      long_tasks: w.longTasks,
      long_task_ms: Math.round(w.longTaskMs),
      reasoning_chars: s.reasoningChars,
      reasoning_spans: s.reasoningCodeSpans,
      reasoning_elements: s.reasoningElements,
      outside_reasoning: s.elementsOutsideReasoning,
      all_spans: s.allCodeSpans,
      elements: s.totalElements,
      panes: s.reasoningPanes,
      open: s.reasoningOpen,
      sent_chars: s.sentChars,
      arrivals: s.arrivals,
      // Arrivals actually delivered in this window against the arrivals the pacing asked for.
      // When the renderer stops keeping up, the generator's own `await sleep(gapMs)` starts
      // returning late and this falls below the target: the stream itself slows down, which is
      // the same thing the user sees as the reply crawling.
      arrivals_in_window: s.arrivals - lastArrivals,
      stream_done: s.streamDone,
      sample_ms: s.sampleCostMs,
      heap_mb:
        performance.memory
          ? Math.round(performance.memory.usedJSHeapSize / 1048576)
          : null,
      instr,
    });
  }, sampleMs);
  window.__rpStopSampler = () => clearInterval(id);
})
"""


# ── process RSS ─────────────────────────────────────────────────────
#
# WHY heapMB IS NOT ENOUGH, AND WHY IT IS NOT RSS.
#
# The `heapMB` column is `performance.memory.usedJSHeapSize`. Three things are wrong with
# reading it as the memory cost of a long thinking pane:
#
#   * It is Chromium only. On WebKit -- the engine the field trace came from -- the property
#     does not exist and the column is structurally null.
#   * It is the JS heap. The DOM, the layout tree, the render surfaces and the highlighter's
#     retained strings live in C++ allocations that never appear in it. 18,536 elements are
#     almost entirely outside this number.
#   * It is deliberately quantised. Chromium buckets and delays it to defeat cross-origin
#     side channels. The 22,500-character baseline run of this file reported exactly 98MB for
#     all ten samples of a run whose element count went from 30 to 2,747.
#
# So this samples the browser's real resident set from the OS, over the browser process AND its
# renderer children, because in Chromium the page's DOM lives in the renderer and the browser
# process would show almost none of the growth.
#
# The sum double counts shared pages (renderers share the zygote's mappings), so the ABSOLUTE
# figure is an upper bound on the tree's footprint. The quantity actually reported as a result
# is `rss_growth_mb`, peak minus the first sample of the same run, and shared pages are common
# to both terms of that subtraction.


def _new_descendants(root_pid: int, before: set[int]) -> list["psutil.Process"]:
    """Processes under `root_pid` that were not there in `before`, topmost first.

    Playwright does not expose the browser's pid: the browser is launched by the node driver,
    which is itself a child of this process, so the browser is a grandchild with an engine
    specific name and an executable path that differs per engine and per platform. Diffing the
    descendant set across the launch identifies it without knowing any of that.
    """
    if psutil is None:
        return []
    try:
        after = {p.pid: p for p in psutil.Process(root_pid).children(recursive = True)}
    except (psutil.Error, OSError):
        return []
    new = {pid: proc for pid, proc in after.items() if pid not in before}
    roots = []
    for pid, proc in new.items():
        try:
            if proc.ppid() not in new:
                roots.append(proc)
        except (psutil.Error, OSError):
            continue
    return roots


def _tree_rss_mb(roots: list["psutil.Process"]) -> float | None:
    """Summed RSS of `roots` and everything under them, or None if nothing could be read.

    None, never 0.0. A renderer that has exited and a permission failure both produce no
    reading, and reporting either as zero would put a fabricated floor into `rss_growth_mb`.
    """
    if psutil is None or not roots:
        return None
    total = 0
    read_any = False
    for root in roots:
        try:
            procs = [root, *root.children(recursive = True)]
        except (psutil.Error, OSError):
            continue
        for proc in procs:
            try:
                total += proc.memory_info().rss
                read_any = True
            except (psutil.Error, OSError):
                continue
    return round(total / 1048576, 1) if read_any else None


class RssSampler:
    """Samples the browser tree's RSS on a thread, on the page sampler's own cadence.

    A thread, because the driver is blocked inside Playwright's sync API for the whole run and
    cannot sample anything itself. It touches no Playwright object, only psutil, so it is safe
    alongside the sync API's greenlet.
    """

    def __init__(self, roots: list["psutil.Process"], period_ms: int) -> None:
        self.roots = roots
        self.period_s = max(0.05, period_ms / 1000)
        self.samples: list[tuple[float, float | None]] = []
        self.reason: str | None = None
        if psutil is None:
            self.reason = "psutil is not installed"
        elif not roots:
            self.reason = "the browser process could not be identified"
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._t0 = 0.0

    def start(self) -> None:
        self._t0 = time.monotonic()
        if self.reason is not None:
            return
        self._thread = threading.Thread(target = self._loop, daemon = True)
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop.is_set():
            at = time.monotonic()
            self.samples.append(((at - self._t0) * 1000, _tree_rss_mb(self.roots)))
            self._stop.wait(self.period_s)

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout = 5)
        if self.reason is None and not [s for _, s in self.samples if s is not None]:
            self.reason = "no RSS reading succeeded for the browser tree"

    def at(self, t_ms: float) -> float | None:
        """The sample nearest `t_ms` on the page sampler's clock."""
        usable = [(t, v) for t, v in self.samples if v is not None]
        if not usable:
            return None
        return min(usable, key = lambda s: abs(s[0] - t_ms))[1]

    def growth_mb(self) -> float | None:
        """Peak minus the FIRST reading of this run, not minus zero.

        A browser that is already up carries a base footprint that has nothing to do with the
        pane, and the run before this one in the same browser leaves its own. What this file
        claims is what the run ADDED.
        """
        usable = [v for _, v in self.samples if v is not None]
        if len(usable) < 2:
            return None
        return round(max(usable) - usable[0], 1)


def start_run(page, size: int) -> None:
    page.evaluate(
        SAMPLER,
        {"sampleMs": SAMPLE_MS},
    )
    page.evaluate(
        """(cfg) => window.__reasoningPane.run(cfg)""",
        {
            "totalChars": size,
            "fenceChars": FENCE_CHARS,
            "proseChars": PROSE_CHARS,
            "preambleChars": PREAMBLE_CHARS or int(size * 0.25),
            "chunkChars": CHUNK_CHARS,
            "gapMs": GAP_MS,
        },
    )


def wait_for_stream(page, timeout_ms: int) -> bool:
    """True if the stream finished. Polls the cheap bookkeeping, never the DOM counts."""
    try:
        page.wait_for_function(
            """() => window.__reasoningPane.streamState().done === true""",
            timeout = timeout_ms,
        )
        return True
    except Exception:
        return False


def _clock_pair(page) -> tuple[float, float]:
    """The page's OS wall clock and the driver's monotonic clock, as close to simultaneous as a
    CDP round trip allows.

    BRACKETED, and the driver's half taken as the MIDPOINT of a read either side of the round
    trip. The page can only answer `Date.now()` when its main thread is free, so a single driver
    reading taken after the evaluate charges the whole of a blocked main thread to the DRIVER's
    elapsed time. Measured: with the naive ordering the throttle control read 1.0004 at 1x and 2x
    and 1.0251 at 4x, i.e. 2.5% of pure round-trip skew inside a quantity whose entire job is to
    be flat. Bracketing leaves only the asymmetry of the round trip, which is sub-millisecond.
    """
    before = time.monotonic()
    page_ms = page.evaluate("""() => Date.now()""")
    after = time.monotonic()
    return page_ms, (before + after) / 2


def measure_cell(
    context,
    engine: str,
    size: int,
    *,
    throttle: float = 1.0,
    rss_roots: list["psutil.Process"] | None = None,
) -> dict:
    page = context.new_page()
    errors: list[str] = []
    page.on("pageerror", lambda e: errors.append(str(e)))

    # startswith, not `"/api/" in url`: vite serves the app's own source modules from paths that
    # contain the substring, and aborting those would blank the page.
    page.route(
        "**/*",
        lambda route: (
            route.fulfill(status = 200, content_type = "application/json", body = "{}")
            if route.request.url.startswith(f"{BASE}/api/")
            else route.continue_()
        ),
    )

    page.set_viewport_size({"width": 1280, "height": 900})
    if throttle > 1 and engine == "chromium":
        throttle_cdp = page.context.new_cdp_session(page)
        throttle_cdp.send("Emulation.setCPUThrottlingRate", {"rate": throttle})
    page.goto(f"{BASE}/smoke-reasoning-pane.html", wait_until = "domcontentloaded")
    page.wait_for_function("""() => Boolean(window.__reasoningPane)""", timeout = 120_000)

    # Armed BEFORE the run, so the clock it starts is the moment the fixture reports done and
    # not the moment the driver's wait_for_function notices.
    page.evaluate("""() => window.__rpArmSettleWatch()""")

    rss = RssSampler(rss_roots or [], SAMPLE_MS)

    # CONTROL for the throttle ladder.
    #
    # It has to be a quantity that CANNOT move when the CPU throttle moves, so that when it does
    # move the conclusion is "the measurement is wrong", not "the page got slower". Two
    # candidates were rejected before this one:
    #
    #   * arrivals per second. It is paced by `await sleep(gapMs)` on the page's main thread, so
    #     a throttled main thread delivers fewer of them per second. The baseline run above
    #     already shows it falling from 118/s to 52/s WITHIN one unthrottled run as the content
    #     grows. It measures the thing under test; it is not a control.
    #   * completed sampler windows per wall second. Also main-thread paced: setInterval cannot
    #     fire while the thread is blocked, and the callback itself walks the DOM. It is
    #     reported below as a diagnostic, and it is NOT the control.
    #
    # What is used instead is the page's own OS wall clock against the driver's: `Date.now()`
    # deltas inside the page divided by `time.monotonic()` deltas in this process over the same
    # interval. Emulation.setCPUThrottlingRate slows how fast V8 executes; it does not touch the
    # clock, so this ratio is 1.0 at every rate. If it is not, the page's timebase itself moved
    # (virtual time, a fake timer shim, a suspended or backgrounded renderer, a CDP session left
    # over from another cell), and in that case every per-second quantity in this file -- fps,
    # busy%, arr/s, frames over 33ms -- is being divided by a fictional second.
    #
    # The one place the main thread does enter is reading Date.now() itself, which can only be
    # answered when the thread is free. That error is bounded by one block, tens of milliseconds
    # against a run of tens of seconds, and the 10% tolerance covers it with room to spare.
    control_page_start, control_driver_start = _clock_pair(page)
    rss.start()
    start_run(page, size)
    finished = wait_for_stream(page, RUN_TIMEOUT_MS)
    control_page_end, control_driver_end = _clock_pair(page)
    control_page_ms = control_page_end - control_page_start
    control_driver_ms = (control_driver_end - control_driver_start) * 1000
    control_ratio = (
        round(control_page_ms / control_driver_ms, 4) if control_driver_ms > 0 else None
    )

    # The completion row, and how long it took to arrive.
    #
    # Not a fixed sleep. The collapse animation is 200ms and `retainStreamingHeight` releases on
    # the same timer, so 2s looks like plenty -- and at 90,000 characters on WebKit it was not:
    # the run reported 15,099 reasoning spans still mounted 2s after the group had already gone
    # to `open=False`, and the harness failed itself for a collapse that had merely not finished
    # yet. Unmounting ~16,000 nodes is itself a main-thread job that scales with the content, and
    # how long it takes is one of the numbers worth having: the trace's own worst event loop lag,
    # 576ms, is at completion.
    collapse_started = time.monotonic()
    try:
        page.wait_for_function(
            """() => document.querySelectorAll(".aui-reasoning-text pre code span").length === 0""",
            timeout = COLLAPSE_TIMEOUT_MS,
        )
        collapse_ms = round((time.monotonic() - collapse_started) * 1000)
    except Exception:
        collapse_ms = None
    after = page.evaluate("""() => window.__reasoningPane.sample()""")

    # The quiet is not the collapse. The spans can already be at zero while React is still
    # committing and the teardown is still on the thread, so this waits for the settle watch's
    # own verdict, on the same budget the collapse gets.
    settle_left = max(0, COLLAPSE_TIMEOUT_MS - round((time.monotonic() - collapse_started) * 1000))
    try:
        page.wait_for_function(
            """() => window.__rpSettle && window.__rpSettle.settledAtMs !== null""",
            timeout = max(1000, settle_left),
        )
    except Exception:
        pass
    settle = page.evaluate("""() => {
      const s = window.__rpSettle;
      if (!s) return null;
      return {
        settled_ms: s.settledAtMs,
        started: s.doneAt !== null,
        window_ms: s.windowMs,
        busy_pct: s.busyPct,
        consecutive: s.consecutive,
        windows: s.windows.slice(0, 40),
      };
    }""")

    page.evaluate("""() => window.__rpStopSampler && window.__rpStopSampler()""")
    rows = page.evaluate("""() => window.__rpRows""")
    state = page.evaluate("""() => window.__reasoningPane.streamState()""")
    long_task_supported = page.evaluate("""() => window.__rp.longTaskSupported""")
    rss.stop()

    for row in rows:
        row["rss_mb"] = rss.at(row["t_ms"])

    # Completed sampler windows per wall second. A DIAGNOSTIC, not the control: see CONTROL
    # above. It is printed next to the control because the two together separate "the page's
    # second is not a second" from "the page's second is fine and the page is merely late".
    windows_per_wall_s = (
        round(len(rows) / (control_driver_ms / 1000), 3) if control_driver_ms > 0 else None
    )

    page.close()
    return {
        "engine": engine,
        "size": size,
        "throttle": throttle,
        "finished": finished,
        "rows": rows,
        "after": after,
        "collapse_ms": collapse_ms,
        "settle": settle,
        "time_to_settle_ms": (settle or {}).get("settled_ms"),
        "rss_samples": [
            {"t_ms": round(t), "rss_mb": v} for t, v in rss.samples
        ],
        "rss_growth_mb": rss.growth_mb(),
        "rss_unavailable": rss.reason,
        "control_page_ms": round(control_page_ms),
        "control_driver_ms": round(control_driver_ms),
        "control_ratio": control_ratio,
        "windows_per_wall_s": windows_per_wall_s,
        "state": state,
        "long_task_supported": long_task_supported,
        "errors": errors,
    }


def ladder_for(engine: str) -> list[float]:
    """The throttle rates this engine will be run at.

    The ladder is CDP, so it exists on Chromium and nowhere else. A non-Chromium engine under
    SMOKE_RP_CPU_LADDER gets one arm at SMOKE_RP_CPU_THROTTLE and a printed line saying so,
    rather than three identical arms that would read as "throttling changes nothing here".
    """
    if not CPU_LADDER:
        return [CPU_THROTTLE]
    if engine != "chromium":
        info(
            f"{engine}: SMOKE_RP_CPU_LADDER is CHROMIUM ONLY (CDP "
            f"Emulation.setCPUThrottlingRate). Running one arm at {CPU_THROTTLE:g}x instead."
        )
        return [CPU_THROTTLE]
    return CPU_LADDER


def run() -> dict:
    results: dict[str, dict] = {}
    with sync_playwright() as p:
        for engine in ENGINES:
            launcher = getattr(p, engine, None)
            if launcher is None:
                info(f"{engine}: no such engine, skipped")
                continue
            kwargs: dict = {"headless": HEADLESS}
            if engine == "chromium":
                kwargs["args"] = chromium_launch_args()
            before = (
                {p.pid for p in psutil.Process(os.getpid()).children(recursive = True)}
                if psutil is not None
                else set()
            )
            try:
                browser = launcher.launch(**kwargs)
            except Exception as exc:
                info(f"{engine}: launch failed ({exc}), skipped")
                continue
            rss_roots = _new_descendants(os.getpid(), before)
            if psutil is None:
                info("psutil is not installed: RSS will be reported as not measured, not as 0")
            elif not rss_roots:
                info(
                    f"{engine}: the browser process could not be identified, so RSS is NOT "
                    "measured on this engine and is reported as null rather than as 0"
                )
            else:
                info(
                    f"{engine}: RSS over pid(s) "
                    f"{', '.join(str(p.pid) for p in rss_roots)} and their children"
                )
            context = browser.new_context()
            context.add_init_script(RECORDER_INIT)
            if INSTRUMENT:
                context.add_init_script(INSTRUMENT_INIT)
            rates = ladder_for(engine)
            cells: dict[str, dict] = {}
            for size in SIZES:
                for rate in rates:
                    key = str(size) if len(rates) == 1 and not CPU_LADDER else f"{size}@{rate:g}x"
                    info(f"{engine} @ {size:,} reasoning chars, cpu throttle {rate:g}x")
                    try:
                        cells[key] = measure_cell(
                            context,
                            engine,
                            size,
                            throttle = rate,
                            rss_roots = rss_roots,
                        )
                    except Exception as exc:
                        info(f"{engine} @ {key}: {type(exc).__name__}: {exc}")
                        cells[key] = {
                            "engine": engine,
                            "size": size,
                            "throttle": rate,
                            "error": str(exc),
                        }
            results[engine] = cells
            context.close()
            browser.close()
    return results


def print_run(cell: dict) -> None:
    rows = cell.get("rows") or []
    if not rows:
        info(f"  {cell['engine']} @ {cell['size']}: no samples")
        return
    instrumented = any(r.get("instr") for r in rows)
    head = (
        f"{'t(s)':>6} {'busy%':>6} {'fps':>7} {'p50fr':>6} {'p95fr':>7} {'>33fr':>6} "
        f"{'>33%':>6} {'maxLag':>8} "
        f"{'arr/s':>6} {'sent':>8} {'chars':>8} {'spans':>8} {'paneEl':>8} "
        f"{'allEl':>9} {'outside':>8} {'heapMB':>7} {'rssMB':>8} {'smpl(ms)':>9}"
    )
    if instrumented:
        head += f" {'shReads':>9} {'shMs':>8} {'paneRd':>8} {'paneMs':>8} {'moCB':>7}"
    print(head, flush = True)
    for r in rows:
        # Dash, not -1 and not 0: an unmeasured RSS must not be readable as a number.
        rss_cell = "-" if r.get("rss_mb") is None else f"{r['rss_mb']:,.1f}"
        line = (
            f"{r['t_ms'] / 1000:6.1f} {r.get('busy_pct', -1):6.1f} {r['fps']:7.1f} "
            f"{(r['p50_frame_ms'] if r['p50_frame_ms'] is not None else -1):6.1f} "
            f"{(r['p95_frame_ms'] if r['p95_frame_ms'] is not None else -1):7.1f} "
            f"{r.get('frames_over_33', -1):6,d} "
            f"{(r['frames_over_33_pct'] if r.get('frames_over_33_pct') is not None else -1):6.1f} "
            f"{r['max_lag_ms']:8.1f} "
            f"{r.get('arrivals_in_window', -1) * 1000 / max(1, SAMPLE_MS):6.1f} "
            f"{r['sent_chars']:8,d} "
            f"{r['reasoning_chars']:8,d} {r['reasoning_spans']:8,d} "
            f"{r.get('reasoning_elements', -1):8,d} "
            f"{r['elements']:9,d} {r.get('outside_reasoning', -1):8,d} "
            f"{(r['heap_mb'] if r.get('heap_mb') is not None else -1):7,d} "
            f"{rss_cell:>8} "
            f"{r['sample_ms']:9.2f}"
        )
        i = r.get("instr")
        if instrumented and i:
            line += (
                f" {i['scrollHeightReads']:9,d} {i['scrollHeightMs']:8.1f} "
                f"{i['reasoningReads']:8,d} {i['reasoningMs']:8.1f} "
                f"{i['mutationCallbacks']:7,d}"
            )
        print(line, flush = True)
    a = cell["after"]
    c = cell.get("collapse_ms")
    print(
        f"  collapse took {c}ms"
        if c is not None
        else "  collapse did NOT complete within the timeout",
        flush = True,
    )
    settled = cell.get("time_to_settle_ms")
    settle = cell.get("settle") or {}
    if settled is not None:
        print(
            f"  time to settle {settled}ms after the stream finished "
            f"({settle.get('consecutive')} consecutive {settle.get('window_ms')}ms windows under "
            f"{settle.get('busy_pct')}% blocked)",
            flush = True,
        )
    else:
        print(
            f"  time to settle NOT MEASURED: the page never showed "
            f"{settle.get('consecutive', 2)} consecutive {settle.get('window_ms', 250)}ms "
            f"windows under {settle.get('busy_pct', 10)}% blocked within the "
            f"{COLLAPSE_TIMEOUT_MS}ms collapse budget, so it is reported as null and not as a "
            "number",
            flush = True,
        )
    if cell.get("rss_unavailable"):
        print(
            f"  RSS NOT MEASURED on this engine ({cell['rss_unavailable']}); the rssMB column is "
            "blank and rss_growth_mb is null, never 0",
            flush = True,
        )
    else:
        peak_rss = max(
            (s["rss_mb"] for s in cell.get("rss_samples", []) if s["rss_mb"] is not None),
            default = None,
        )
        first_rss = next(
            (s["rss_mb"] for s in cell.get("rss_samples", []) if s["rss_mb"] is not None),
            None,
        )
        print(
            f"  browser tree RSS {first_rss:,.1f}MB -> peak {peak_rss:,.1f}MB "
            f"(+{cell.get('rss_growth_mb')}MB over the run), summed over the browser process and "
            "its renderers",
            flush = True,
        )
    if cell.get("control_ratio") is not None:
        print(
            f"  CONTROL page wall clock / driver wall clock = {cell['control_ratio']:.4f} "
            f"({cell['control_page_ms']:,}ms page vs {cell['control_driver_ms']:,}ms driver); "
            f"diagnostic: {cell.get('windows_per_wall_s')} sampler windows per wall second",
            flush = True,
        )
    print(
        f"  after completion: chars {a['reasoningChars']:,} spans "
        f"{a['reasoningCodeSpans']:,} elements {a['totalElements']:,} "
        f"open={a['reasoningOpen']}",
        flush = True,
    )


def summarise(cell: dict) -> dict:
    """First tenth of the run against the last tenth, on the same run.

    Not first sample against last sample: a single sample is one second of a loaded machine. Not
    across sizes either, because the trace is a curve WITHIN one generation, and reproducing it
    means showing the same run getting slower as its own content grows.
    """
    # The FIRST sample is dropped, always. It spans the click that starts the run, the runtime's
    # first publish and the first Shiki highlighter load, none of which are the growth this file
    # is about; keeping it reported a 229ms "stall" at 485 characters of content and made the
    # early window look worse than the late one.
    rows = [r for r in (cell.get("rows") or []) if not r.get("stream_done")][1:]
    if len(rows) < 6:
        return {}
    n = max(2, len(rows) // 10)
    early = rows[:n]
    late = rows[-n:]

    # None on an empty list, not a crash. A window in which the page produced NO frames at all
    # has a null p95, and headless Chromium really does produce runs of them: the rAF loop does
    # not tick before the first composite, so a slow vite start leaves the first few samples
    # frameless. Averaging over "the frames there were" then averages over nothing. Before this
    # guard that was a ZeroDivisionError three samples into a ladder arm, i.e. the whole run lost
    # to an arithmetic edge rather than to anything measured.
    def mean(xs: list[float]) -> float | None:
        return sum(xs) / len(xs) if xs else None

    def rnd(value: float | None, places: int) -> float | None:
        return None if value is None else round(value, places)

    early_fps = mean([r["fps"] for r in early])
    late_fps = mean([r["fps"] for r in late])
    early_busy = mean([r.get("busy_pct", 0) for r in early])
    late_busy = mean([r.get("busy_pct", 0) for r in late])
    early_p95 = mean([r["p95_frame_ms"] for r in early if r["p95_frame_ms"] is not None])
    late_p95 = mean([r["p95_frame_ms"] for r in late if r["p95_frame_ms"] is not None])
    early_over33 = mean([r.get("frames_over_33", 0) for r in early]) or 0
    late_over33 = mean([r.get("frames_over_33", 0) for r in late]) or 0
    early_over33_pct = mean(
        [r["frames_over_33_pct"] for r in early if r.get("frames_over_33_pct") is not None]
    )
    late_over33_pct = mean(
        [r["frames_over_33_pct"] for r in late if r.get("frames_over_33_pct") is not None]
    )
    # Reported, not silently averaged away. A frameless sample is not a slow one, and an early
    # window built out of them understates the baseline this file's whole ratio is taken against.
    frameless_early = sum(1 for r in early if r["p95_frame_ms"] is None)
    frameless_late = sum(1 for r in late if r["p95_frame_ms"] is None)
    rss_values = [r["rss_mb"] for r in rows if r.get("rss_mb") is not None]
    peak = max(r["reasoning_spans"] for r in rows)
    peak_chars = max(r["reasoning_chars"] for r in rows)
    peak_elements = max(r["elements"] for r in rows)
    return {
        "label": LABEL,
        "throttle": cell.get("throttle", CPU_THROTTLE),
        "early_fps": rnd(early_fps, 1),
        "early_frames_over_33": round(early_over33, 1),
        "late_frames_over_33": round(late_over33, 1),
        "frames_over_33_ratio": (
            round(late_over33 / early_over33, 2) if early_over33 else None
        ),
        "early_frames_over_33_pct": rnd(early_over33_pct, 1),
        "late_frames_over_33_pct": rnd(late_over33_pct, 1),
        "frameless_early_samples": frameless_early,
        "frameless_late_samples": frameless_late,
        "first_rss_mb": rss_values[0] if rss_values else None,
        "peak_rss_mb": max(rss_values) if rss_values else None,
        "rss_growth_mb": cell.get("rss_growth_mb"),
        "rss_unavailable": cell.get("rss_unavailable"),
        "time_to_settle_ms": cell.get("time_to_settle_ms"),
        "control_ratio": cell.get("control_ratio"),
        "windows_per_wall_s": cell.get("windows_per_wall_s"),
        "late_fps": rnd(late_fps, 1),
        "fps_ratio": round(early_fps / late_fps, 2) if early_fps and late_fps else None,
        "early_busy_pct": rnd(early_busy, 1),
        "late_busy_pct": rnd(late_busy, 1),
        "busy_ratio": round(late_busy / early_busy, 2) if early_busy and late_busy else None,
        "early_p95_frame_ms": rnd(early_p95, 1),
        "late_p95_frame_ms": rnd(late_p95, 1),
        "p95_ratio": round(late_p95 / early_p95, 2) if early_p95 and late_p95 else None,
        "early_max_lag_ms": round(max(r["max_lag_ms"] for r in early), 1),
        "late_max_lag_ms": round(max(r["max_lag_ms"] for r in late), 1),
        "peak_chars": peak_chars,
        "peak_spans": peak,
        "peak_elements": peak_elements,
        "chars_per_span": round(peak_chars / peak, 2) if peak else None,
        "after_spans": cell["after"]["reasoningCodeSpans"],
        "after_elements": cell["after"]["totalElements"],
        "collapsed": cell["after"]["reasoningCodeSpans"] == 0,
        "collapse_ms": cell.get("collapse_ms"),
        "peak_heap_mb": max((r["heap_mb"] for r in rows if r["heap_mb"] is not None), default = None),
        "final_heap_mb": cell["rows"][-1]["heap_mb"] if cell.get("rows") else None,
    }


def _mb(value: float | None) -> str:
    """`not measured`, never `0MB`. A missing reading and an empty one must not read alike."""
    return "not measured" if value is None else f"{value:,.1f}MB"


def _ms(value: float | None) -> str:
    return "not measured" if value is None else f"{value:,.0f}ms"


def print_ladder(results: dict) -> None:
    """The ladder, one row per throttle rate, with the control beside the measurements.

    The point of printing them together is that they are read together. A ladder where busy%
    climbs and the control climbs with it is not a ladder that found anything: it is a ladder
    whose seconds are not seconds.
    """
    if not CPU_LADDER:
        return
    print("\n=== CPU THROTTLING LADDER (CHROMIUM ONLY: CDP Emulation.setCPUThrottlingRate) ===",
          flush = True)
    for engine, cells in results.items():
        arms = [c for c in cells.values() if "error" not in c]
        if not arms:
            continue
        if engine != "chromium":
            print(
                f"  [{LABEL}] {engine}: no CPU throttling exists on this engine, so it has no "
                "ladder. The rows below are Chromium's alone.",
                flush = True,
            )
            continue
        by_size: dict[int, list[dict]] = {}
        for cell in arms:
            by_size.setdefault(cell["size"], []).append(cell)
        for size, group in by_size.items():
            group = sorted(group, key = lambda c: c.get("throttle", 1))
            print(f"  [{LABEL}] chromium @ {size:,} reasoning chars", flush = True)
            print(
                f"    {'rate':>6} {'busy%':>7} {'fps':>7} {'>33fr/win':>10} {'rssMB':>9} "
                f"{'growthMB':>9} {'settle(ms)':>11} {'CONTROL':>9} {'win/s':>7}",
                flush = True,
            )
            for cell in group:
                s = summarise(cell) or {}
                # A dash where there is no reading. -1 in a memory column has been read as a
                # number before now.
                peak_rss = s.get("peak_rss_mb")
                growth = cell.get("rss_growth_mb")
                settle = cell.get("time_to_settle_ms")
                control = cell.get("control_ratio")
                busy = s.get("late_busy_pct")
                fps = s.get("late_fps")
                over33 = s.get("late_frames_over_33")
                print(
                    f"    {cell.get('throttle', 1):5g}x "
                    f"{('-' if busy is None else f'{busy:.1f}'):>7} "
                    f"{('-' if fps is None else f'{fps:.1f}'):>7} "
                    f"{('-' if over33 is None else f'{over33:.1f}'):>10} "
                    f"{('-' if peak_rss is None else f'{peak_rss:,.1f}'):>9} "
                    f"{('-' if growth is None else f'{growth:,.1f}'):>9} "
                    f"{('-' if settle is None else f'{settle:,.0f}'):>11} "
                    f"{('-' if control is None else f'{control:.4f}'):>9} "
                    f"{(cell.get('windows_per_wall_s') or -1):7.3f}",
                    flush = True,
                )
            controls = [
                c["control_ratio"] for c in group if c.get("control_ratio") is not None
            ]
            if len(controls) < 2:
                continue
            spread_pct = (max(controls) - min(controls)) / min(controls) * 100
            if spread_pct > CONTROL_TOLERANCE_PCT:
                # Loud on purpose. This is not a slow page, it is a broken measurement: the
                # control is a wall-clock ratio and the CPU throttle cannot touch it. If it moved,
                # the page's own timebase moved, and every per-second column above was divided by
                # a second that did not last a second.
                print(
                    f"    !!!! [{LABEL}] CONTROL MOVED {spread_pct:.1f}% ACROSS THE LADDER "
                    f"({min(controls):.4f} .. {max(controls):.4f}, tolerance "
                    f"{CONTROL_TOLERANCE_PCT:g}%). THE MEASUREMENT IS WRONG, NOT THE PAGE SLOW: "
                    "the page wall clock and the driver wall clock disagree, so fps, busy%, "
                    "arr/s and frames over 33ms on this ladder are all divided by a fictional "
                    "second and none of the arms are comparable.",
                    flush = True,
                )
            else:
                print(
                    f"    control flat within {spread_pct:.1f}% across the ladder (tolerance "
                    f"{CONTROL_TOLERANCE_PCT:g}%), so the per-second columns above are on the "
                    "same second at every rate",
                    flush = True,
                )


def harness_failures(results: dict) -> list[str]:
    """Only the ways this file could be measuring nothing. Timings never fail a run."""
    bad: list[str] = []
    if not results:
        bad.append("no engine produced a cell")
    for engine, cells in results.items():
        for key, cell in cells.items():
            where = f"{engine} @ {key}"
            if "error" in cell:
                bad.append(f"{where}: {cell['error']}")
                continue
            if not cell.get("finished"):
                bad.append(f"{where}: the stream never finished")
            rows = cell.get("rows") or []
            if len(rows) < 5:
                bad.append(f"{where}: {len(rows)} samples, too few to be a curve")
                continue
            if not any(r["open"] for r in rows):
                bad.append(f"{where}: the reasoning group never opened, so nothing was measured")
            peak = max(r["reasoning_spans"] for r in rows)
            size = cell["size"]
            # What the fixture owes, always: it sent the content it was asked for. This is the
            # check that catches a fixture that quietly built 90,000 characters of prose, and it
            # holds whatever the renderer under test then does with them.
            sent = cell.get("state", {}).get("sentChars") or 0
            if sent < size * 0.99:
                bad.append(f"{where}: the fixture only sent {sent:,} of {size:,} characters")
            if max(r["reasoning_chars"] for r in rows) <= 0:
                bad.append(f"{where}: the reasoning pane never held any text")
            # And, on an unmodified tree, that the content really did become highlight spans. At
            # the capture's own size the floor is most of its 16,000; below that, scaled. Turn it
            # off with SMOKE_RP_EXPECT_SPANS=0 when measuring a change whose whole purpose is to
            # stop building them.
            floor = int(11_000 * size / 90_000)
            if EXPECT_SPANS and peak < floor:
                bad.append(
                    f"{where}: peak {peak:,} reasoning spans is below the floor {floor:,}; "
                    "either the fixture is not the capture's content, or the tree under test "
                    "stopped building spans and this run wanted SMOKE_RP_EXPECT_SPANS=0"
                )
            peak_chars = max(r["reasoning_chars"] for r in rows)
            if MAX_PANE_CHARS and peak_chars > MAX_PANE_CHARS:
                bad.append(
                    f"{where}: the pane peaked at {peak_chars:,} mounted characters against a "
                    f"bound of {MAX_PANE_CHARS:,} over {size:,} sent; the window did not bound "
                    "what stayed mounted, so this run measured an unwindowed tree"
                )
            after = cell["after"]
            if after["reasoningCodeSpans"] != 0:
                bad.append(
                    f"{where}: {after['reasoningCodeSpans']:,} reasoning spans survived "
                    f"completion after {COLLAPSE_TIMEOUT_MS}ms; the collapse that localises the "
                    "cost did not happen"
                )
            real_errors = [e for e in cell.get("errors", []) if "ResizeObserver" not in e]
            if real_errors:
                bad.append(f"{where}: page error {real_errors[0]}")
    return bad


def main() -> int:
    vite = None
    try:
        if OWNS_SERVER:
            info(f"starting vite dev server on port {PORT}")
            vite = start_vite(PORT)
            wait_for_smoke_page(
                f"{BASE}/smoke-reasoning-pane.html",
                "smoke-reasoning-pane-main.tsx",
                proc = vite,
            )
        # The label leads the header, and it is on every summary line below as well. Several
        # worktrees run this same file against different trees on the same box, and a table
        # pasted into a comparison with no arm on it is a table nobody can place afterwards.
        info(
            f"label={LABEL} engines={','.join(ENGINES)} sizes={SIZES} fence={FENCE_CHARS} "
            f"prose={PROSE_CHARS} "
            f"chunk={CHUNK_CHARS} gap={GAP_MS}ms sample={SAMPLE_MS}ms "
            f"instrument={'on' if INSTRUMENT else 'off'} "
            f"headless={'on' if HEADLESS else 'OFF (real compositor)'} "
            f"cpu_throttle={CPU_THROTTLE}x "
            f"cpu_ladder={','.join(f'{r:g}' for r in CPU_LADDER) if CPU_LADDER else 'off'}"
            f"{' (CHROMIUM ONLY)' if CPU_LADDER else ''} "
            f"rss={'psutil' if psutil is not None else 'NOT MEASURED (psutil missing)'}"
        )
        results = run()
    finally:
        if vite is not None:
            stop_process(vite)
            info("vite stopped")

    report: dict[str, dict] = {}
    for engine, cells in results.items():
        for key, cell in cells.items():
            if "error" in cell:
                continue
            print(
                f"\n=== [{LABEL}] {engine} @ {cell['size']:,} reasoning chars, "
                f"cpu throttle {cell.get('throttle', CPU_THROTTLE):g}x ===",
                flush = True,
            )
            if not cell.get("long_task_supported"):
                print(
                    "  longtask is NOT supported on this engine: the long-task columns are "
                    "structurally zero and mean nothing. Read max lag.",
                    flush = True,
                )
            print_run(cell)
            s = summarise(cell)
            report[f"{engine}@{key}"] = s
            if s:
                # Every summary line carries the label and the throttle rate. Same reason as the
                # header: these lines get quoted on their own.
                print(
                    f"  [{LABEL}] {engine} {cell['size']:,}ch {s['throttle']:g}x: "
                    f"main thread busy {s['early_busy_pct']}% -> {s['late_busy_pct']}% "
                    f"({s['busy_ratio']}x), "
                    f"early fps {s['early_fps']} -> late fps {s['late_fps']} "
                    f"({s['fps_ratio']}x slower), p95 frame {s['early_p95_frame_ms']}ms -> "
                    f"{s['late_p95_frame_ms']}ms ({s['p95_ratio']}x), "
                    f"max lag {s['early_max_lag_ms']}ms -> "
                    f"{s['late_max_lag_ms']}ms, peak {s['peak_spans']:,} spans over "
                    f"{s['peak_chars']:,} chars ({s['chars_per_span']} chars/span), "
                    f"elements {s['peak_elements']:,} -> {s['after_elements']:,}",
                    flush = True,
                )
                print(
                    f"  [{LABEL}] {engine} {cell['size']:,}ch {s['throttle']:g}x: "
                    f"frames over 33ms {s['early_frames_over_33']}/window "
                    f"({s['early_frames_over_33_pct']}%) -> {s['late_frames_over_33']}/window "
                    f"({s['late_frames_over_33_pct']}%), "
                    f"rss {_mb(s['first_rss_mb'])} -> peak {_mb(s['peak_rss_mb'])} "
                    f"(growth {_mb(s['rss_growth_mb'])}), "
                    f"time to settle {_ms(s['time_to_settle_ms'])}, "
                    f"control {s['control_ratio']}, "
                    f"frameless samples {s['frameless_early_samples']} early / "
                    f"{s['frameless_late_samples']} late",
                    flush = True,
                )

    print_ladder(results)

    payload = {
        "label": LABEL,
        "sizes": SIZES,
        "engines": ENGINES,
        "fence_chars": FENCE_CHARS,
        "prose_chars": PROSE_CHARS,
        "chunk_chars": CHUNK_CHARS,
        "gap_ms": GAP_MS,
        "sample_ms": SAMPLE_MS,
        "instrumented": INSTRUMENT,
        "expect_spans": EXPECT_SPANS,
        "max_pane_chars": MAX_PANE_CHARS,
        "headless": HEADLESS,
        "cpu_throttle": CPU_THROTTLE,
        "cpu_ladder": CPU_LADDER,
        "control_tolerance_pct": CONTROL_TOLERANCE_PCT,
        "rss_backend": "psutil" if psutil is not None else None,
        "trace": TRACE,
        "results": results,
        "report": report,
    }
    out = OUT / f"reasoning-pane-{LABEL}.json"
    out.write_text(json.dumps(payload, indent = 2), encoding = "utf-8")
    info(f"wrote {out}")

    bad = harness_failures(results)
    if bad:
        print("\nHARNESS FAILURES", flush = True)
        for b in bad:
            print(f"  - {b}", flush = True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
