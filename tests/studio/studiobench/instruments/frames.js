// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
// The frame recorder. Installed as an init script before any app code runs.
// SALVAGED from playwright_reasoning_pane.py's RECORDER_INIT, keeping the two things that version
// got right and fixing the one it got wrong.
// KEPT: ONE self-rescheduling rAF loop as the frame counter, and requestAnimationFrame is NOT
// wrapped. A wrapper that increments per callback counts the page's frame once for the loop and
// once more for every rAF the app scheduled, so the reported frame rate RISES with how busy the
// app is: the first version reported 888 fps on a 60Hz page, which is the metric inverted.
// KEPT: blocked time from a 1ms setTimeout, not a MessageChannel ping-pong, which ticks about
// 150,000 times a second and halves Firefox's frame rate before any app code runs. This ticks
// about 150 times a second. Blocked time is the column that MOVES WHEN FPS DOES NOT: a page with
// 65% idle still paints every frame on time, so fps stays pinned at 60 while the work per chunk
// triples.
// FIXED: the clamp is calibrated during an ENFORCED IDLE WINDOW the driver opens, not from the
// first 60 ticks of whatever the page was doing. `setTimeout(fn, 1)` has a ~4ms spec floor that
// differs by engine and build, and blocked time is a SUBTRACTION against it, so a wrong clamp
// invents block on one engine and hides it on another. Calibrating from the first 60 ticks means
// calibrating while the app is booting, or on a rung where the 'idle' floor is really the app's
// steady-state load, which reports a page pinned at 100% busy as 0.2% busy.
// And when the calibrated clamp comes out ABOVE 10ms the answer is not a number: the machine
// could not answer an idle timer promptly, so nothing was idle and there is no floor to subtract.
// busy_pct is then null with a reason.

(() => {
  if (window.__sb && window.__sb.frames) return;
  const nativeRaf = window.requestAnimationFrame.bind(window);
  window.__sbNativeRaf = nativeRaf;
  window.__sb = window.__sb || {};

  const MAX_CLAMP_MS = 10.0;
  const CALIBRATION_TICKS = 60;
  // A window long enough to exceed this is minutes of 60 Hz, which no slot in the scene is. The cap
  // stops a pathological window ballooning the payload; it is not a routine path.
  const GAPS_CAP = 50000;

  const R = {
    frames: 0,
    frameGaps: [],
    maxLagMs: 0,
    lagTicks: 0,
    lagSumMs: 0,
    blockedMs: 0,
    // The same blocked time again, NEVER reset. `blockedMs` is drained by the window reader, and a
    // settle watch has to read blocked time on its own cadence while that reader runs; two readers
    // draining one accumulator would each see a fraction of the block.
    blockedTotalMs: 0,
    clampMs: null,
    clampReason: "not calibrated",
    clampSamples: 0,
    calibrating: false,
    calibration: [],
    longTaskSupported: Boolean(
      (PerformanceObserver.supportedEntryTypes || []).includes("longtask"),
    ),
    longTasks: 0,
    longTaskMs: 0,
    // Every rAF the APP schedules, counted separately from the loop's own. Not a frame rate: it is
    // how the tri-clock check tells 'the page is idle' from 'the loop is starved'.
    appRafs: 0,
  };

  let lastFrame = performance.now();
  const frame = () => {
    const now = performance.now();
    R.frames += 1;
    R.frameGaps.push(now - lastFrame);
    lastFrame = now;
    nativeRaf(frame);
  };
  nativeRaf(frame);

  // Count the app's own rAF traffic without pumping it. A pass-through wrapper is safe here BECAUSE
  // it is not the frame counter.
  window.requestAnimationFrame = function (cb) {
    R.appRafs += 1;
    return nativeRaf(cb);
  };

  let lastTick = performance.now();
  const tick = () => {
    const now = performance.now();
    const gap = now - lastTick;
    lastTick = now;
    if (R.calibrating) {
      R.calibration.push(gap);
    } else if (R.clampMs !== null) {
      R.lagTicks += 1;
      R.lagSumMs += gap;
      const over = Math.max(0, gap - R.clampMs);
      R.blockedMs += over;
      R.blockedTotalMs += over;
      if (gap > R.maxLagMs) R.maxLagMs = gap;
    }
    setTimeout(tick, 1);
  };
  setTimeout(tick, 1);

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

  const quantile = (sorted, q) =>
    sorted.length === 0 ? null : sorted[Math.min(sorted.length - 1, Math.floor(sorted.length * q))];

  window.__sb.frames = {
    // Opened by the driver during an ENFORCED IDLE WINDOW: nothing streaming, no action running, the
    // page at rest. Returns what it measured so the driver can record it rather than trust it.
    beginCalibration() {
      R.calibrating = true;
      R.calibration = [];
      return { ticks: CALIBRATION_TICKS };
    },
    endCalibration() {
      R.calibrating = false;
      const samples = R.calibration.slice().sort((a, b) => a - b);
      R.clampSamples = samples.length;
      if (samples.length < 10) {
        R.clampMs = null;
        R.clampReason =
          "the idle window produced " + samples.length + " timer ticks, too few to find a floor";
        return { clampMs: null, reason: R.clampReason, samples: samples.length };
      }
      const median = samples[Math.floor(samples.length / 2)];
      if (median > MAX_CLAMP_MS) {
        // NOT a clamp. A 1ms timer taking longer than 10ms on an idle page means the page was not idle,
        // so there is no floor to subtract and every blocked-time figure would be a subtraction against
        // the app's own steady load.
        R.clampMs = null;
        R.clampReason =
          "the calibrated timer clamp came out at " +
          median.toFixed(2) +
          "ms, above the " +
          MAX_CLAMP_MS +
          "ms ceiling: the page was not idle during calibration, so there is no floor to subtract";
        return { clampMs: null, reason: R.clampReason, median, samples: samples.length };
      }
      R.clampMs = median;
      R.clampReason = "calibrated";
      return {
        clampMs: median,
        reason: "calibrated",
        samples: samples.length,
        p05: quantile(samples, 0.05),
        p95: quantile(samples, 0.95),
      };
    },

    reset() {
      R.frames = 0;
      R.frameGaps = [];
      R.maxLagMs = 0;
      R.lagTicks = 0;
      R.lagSumMs = 0;
      R.blockedMs = 0;
      R.longTasks = 0;
      R.longTaskMs = 0;
      R.appRafs = 0;
      return performance.now();
    },

    // Drain the window. `elapsedMs` is the DRIVER's measure, passed in rather than computed here, so
    // fps is per real elapsed time even when the page could not run its own clock reads promptly.
    read(elapsedMs) {
      const gaps = R.frameGaps.slice().sort((a, b) => a - b);
      let over33 = 0;
      for (const g of gaps) if (g > 33) over33 += 1;
      const elapsed = elapsedMs && elapsedMs > 0 ? elapsedMs : null;
      const out = {
        frames: R.frames,
        frames_attempted: true,
        app_rafs: R.appRafs,
        fps: elapsed === null ? null : Math.round((R.frames / (elapsed / 1000)) * 10) / 10,
        frames_over_33: over33,
        // As a SHARE of the frames observed, because the denominator is not fixed: headless Chromium has
        // no vsync and runs the loop as fast as it can, so a raw count is not comparable across engines
        // or loads.
        frames_over_33_pct:
          gaps.length === 0 ? null : Math.round((over33 / gaps.length) * 1000) / 10,
        p50_frame_ms: quantile(gaps, 0.5),
        p95_frame_ms: quantile(gaps, 0.95),
        // The RAW deltas, not only the summary: time_in_jank_pct and jank_index are defined over the
        // whole distribution and neither can be recovered from percentiles, so without this the scoring
        // layer would skip two of its six metrics or invent them from p95. `gaps` is sorted ASCENDING,
        // so a head slice would drop exactly the janky frames; over the cap this emits null and says
        // why, and the scoring layer reads 'failed' rather than a number built from the fastest frames.
        frame_gaps_ms:
          gaps.length > GAPS_CAP ? null : gaps.map((g) => Math.round(g * 10) / 10),
        frame_gaps_truncated: gaps.length > GAPS_CAP,
        frame_gaps_total: gaps.length,
        max_frame_ms: gaps.length === 0 ? null : gaps[gaps.length - 1],
        max_lag_ms: Math.round(R.maxLagMs * 10) / 10,
        lag_ticks: R.lagTicks,
        mean_lag_ms: R.lagTicks === 0 ? null : Math.round((R.lagSumMs / R.lagTicks) * 10) / 10,
        clamp_ms: R.clampMs === null ? null : Math.round(R.clampMs * 100) / 100,
        clamp_reason: R.clampReason,
        long_tasks: R.longTaskSupported ? R.longTasks : null,
        long_task_ms: R.longTaskSupported ? Math.round(R.longTaskMs) : null,
        // The point of the flag: without it an engine with no Long Tasks API reports zero jank in the
        // same shape as an engine that had none.
        long_task_supported: R.longTaskSupported,
      };
      if (R.clampMs === null) {
        out.busy_pct = null;
        out.busy_pct_reason = R.clampReason;
        out.blocked_ms = null;
      } else if (elapsed === null) {
        out.busy_pct = null;
        out.busy_pct_reason = "the driver reported no elapsed time for this window";
        out.blocked_ms = Math.round(R.blockedMs * 10) / 10;
      } else {
        out.blocked_ms = Math.round(R.blockedMs * 10) / 10;
        out.busy_pct = Math.round((R.blockedMs / elapsed) * 1000) / 10;
        out.busy_pct_reason = null;
      }
      this.reset();
      return out;
    },

    // For the settle watch and anything else that needs blocked time without draining the window.
    blockedTotalMs() {
      return R.blockedTotalMs;
    },
    clamp() {
      return { clampMs: R.clampMs, reason: R.clampReason, samples: R.clampSamples };
    },
  };

  // Two rAFs: the second is the frame that has PAINTED the first's work. Every action timing
  // clocked across a paint uses this, so the paint floor is one shared constant.
  window.__sbNextPaint = () =>
    new Promise((resolve) => nativeRaf(() => nativeRaf(() => resolve(performance.now()))));
})();
