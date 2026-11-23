// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
// Keystroke-to-paint from real page.keyboard events; synthetic input hides queueing delay.
// Anchored on keydown.timeStamp (hardware arrival), not the input handler, to include the wait.

(() => {
  if (window.__sb && window.__sb.input) return;
  window.__sb = window.__sb || {};

  const S = {
    armed: false,
    target: null,
    baseline: "",
    samples: [],
    pending: null,
    dropped: 0,
    keyAt: null,
    unanchored: 0,
    // Denominator: every input seen, sampled or coalesced.
    seen: 0,
  };

  const nextPaint = () =>
    window.__sbNextPaint
      ? window.__sbNextPaint()
      : new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(() => r(performance.now()))));

  // Last unconsumed keydown, so a key producing no character cannot anchor the next.
  const onKeyDown = (ev) => {
    if (!S.armed || (S.target && ev.target !== S.target)) return;
    S.keyAt = ev.timeStamp;
  };

  // Implausible anchors (synthetic, 0, IME commit) fall back to this clock and are counted.
  const anchor = (now) => {
    const at = S.keyAt;
    S.keyAt = null;
    if (typeof at !== "number" || !isFinite(at) || at <= 0 || at > now || now - at > 10000) {
      S.unanchored += 1;
      return null;
    }
    return at;
  };

  const onInput = (ev) => {
    if (!S.armed || ev.target !== S.target) return;
    S.seen += 1;
    // One in flight: otherwise one paint is attributed to several keystrokes.
    if (S.pending !== null) {
      S.dropped += 1;
      S.keyAt = null;
      return;
    }
    const at = performance.now();
    const keyAt = anchor(at);
    const started = keyAt === null ? at : keyAt;
    const lengthAt = S.target.value.length;
    S.pending = at;
    nextPaint().then((paintedAt) => {
      S.samples.push({
        at_ms: Math.round(at * 10) / 10,
        latency_ms: Math.round((paintedAt - started) * 10) / 10,
        input_delay_ms: keyAt === null ? null : Math.round((at - keyAt) * 10) / 10,
        paint_ms: Math.round((paintedAt - at) * 10) / 10,
        anchored_on: keyAt === null ? "input" : "keydown",
        value_length: lengthAt,
      });
      S.pending = null;
    });
  };

  window.__sb.input = {
    // Resolved here so the driver can re-arm after the composer re-renders.
    arm(selector) {
      const el = document.querySelector(selector);
      if (!el) return { armed: false, reason: "no element matched " + selector };
      if (S.target && S.target !== el) S.target.removeEventListener("input", onInput, true);
      S.target = el;
      S.baseline = el.value === undefined ? "" : el.value;
      S.samples = [];
      S.dropped = 0;
      S.pending = null;
      S.keyAt = null;
      S.unanchored = 0;
      S.seen = 0;
      S.armed = true;
      el.addEventListener("input", onInput, true);
      // Window capture so stopPropagation cannot hide the key; removed first to stay idempotent.
      window.removeEventListener("keydown", onKeyDown, true);
      window.addEventListener("keydown", onKeyDown, true);
      return { armed: true, baseline_length: S.baseline.length };
    },

    // Driver polls this instead of a fixed wait, which would drop the slowest keystroke.
    settled() {
      return { pending: S.pending !== null, samples: S.samples.length, seen: S.seen };
    },

    collect(expected) {
      const samples = S.samples.slice();
      const latencies = samples.map((s) => s.latency_ms).sort((a, b) => a - b);
      const at = (q) =>
        latencies.length === 0 ? null : latencies[Math.min(latencies.length - 1, Math.floor(latencies.length * q))];
      const observedText = S.target ? S.target.value : null;
      const delays = samples
        .map((s) => s.input_delay_ms)
        .filter((v) => v !== null && v !== undefined)
        .sort((a, b) => a - b);
      const unanchored = S.unanchored;
      const seen = S.seen;
      // Reported so the driver can fail the reading rather than drop the in-flight sample.
      const pendingNow = S.pending !== null;
      S.samples = [];
      S.unanchored = 0;
      S.seen = 0;
      return {
        samples: samples.length,
        samples_attempted: true,
        expected: expected === undefined ? null : expected,
        inputs_seen: seen,
        pending_at_collect: pendingNow,
        // Unanchored samples understate the wait, so they are reported rather than blended in.
        unanchored: unanchored,
        input_delay_p95_ms:
          delays.length === 0 ? null : delays[Math.min(delays.length - 1, Math.floor(delays.length * 0.95))],
        coalesced: S.dropped,
        p50_ms: at(0.5),
        p95_ms: at(0.95),
        max_ms: latencies.length === 0 ? null : latencies[latencies.length - 1],
        // First sample is a cold outlier, reported separately since on a jammed page it is real.
        first_ms: samples.length === 0 ? null : samples[0].latency_ms,
        text_length: observedText === null ? null : observedText.length,
        grew_by: observedText === null ? null : observedText.length - S.baseline.length,
      };
    },

    disarm() {
      if (S.target) S.target.removeEventListener("input", onInput, true);
      window.removeEventListener("keydown", onKeyDown, true);
      S.armed = false;
      S.target = null;
      S.keyAt = null;
      return true;
    },
  };
})();
