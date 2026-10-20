// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
// Frame recorder, installed as an init script before any app code runs.
// rAF is NOT wrapped (wrapping inflates fps with app rAF traffic). Blocked time is measured with a
// 1ms setTimeout (MessageChannel ticks ~150k/s and perturbs fps), minus a clamp calibrated in a
// driver-enforced idle window.

(() => {
  if (window.__sb && window.__sb.frames) return;
  const nativeRaf = window.requestAnimationFrame.bind(window);
  window.__sbNativeRaf = nativeRaf;
  window.__sb = window.__sb || {};

  const MAX_CLAMP_MS = 10.0;
  const CALIBRATION_TICKS = 60;
  // Safety cap on payload size; real windows never get near it.
  const GAPS_CAP = 50000;

  const R = {
    frames: 0,
    frameGaps: [],
    maxLagMs: 0,
    lagTicks: 0,
    lagSumMs: 0,
    blockedMs: 0,
    // Never reset: lets a settle watch read blocked time without draining the window reader.
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
    // App rAF count, used to tell an idle page from a starved loop; not a frame rate.
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

  // Pass-through wrapper is safe because it is not the frame counter.
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
        // A 1ms timer over 10ms on an idle page means nothing was idle, so there is no floor.
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

    // elapsedMs comes from the driver because the page may not read its own clock promptly.
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
        // A share, not a count: headless Chromium has no vsync so raw counts are not comparable.
        frames_over_33_pct:
          gaps.length === 0 ? null : Math.round((over33 / gaps.length) * 1000) / 10,
        p50_frame_ms: quantile(gaps, 0.5),
        p95_frame_ms: quantile(gaps, 0.95),
        // Raw deltas for the jank metrics. gaps is sorted ascending, so over the cap emit null
        // rather than a head slice that drops exactly the janky frames.
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
        // Distinguishes no Long Tasks API from zero jank.
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

    blockedTotalMs() {
      return R.blockedTotalMs;
    },
    clamp() {
      return { clampMs: R.clampMs, reason: R.clampReason, samples: R.clampSamples };
    },
  };

  // Two rAFs: the second is the frame that painted the first's work.
  window.__sbNextPaint = () =>
    new Promise((resolve) => nativeRaf(() => nativeRaf(() => resolve(performance.now()))));
})();
