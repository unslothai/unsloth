// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
//
// Keystroke-to-paint, measured from the page side of a REAL key event.
//
// WHY THIS IS NOT THE SALVAGED KEYSTROKE_JS. The old harness typed by calling the native value
// setter on the textarea and dispatching a synthetic `input` event. That reaches React's
// controlled input, so it measured something -- but it enters the pipeline AFTER hit testing,
// after the browser's own event routing, and it carries no `latencyInfo`, which is the trace
// field that says when the OS delivered the keypress. A synthetic event therefore cannot show
// input queueing delay at all: it is dispatched from a task that is by definition already running,
// so the queue is empty by construction and the number reads clean exactly when a user would be
// waiting longest.
//
// So the driver types with `page.keyboard`, which goes in through CDP as a real input event, and
// this file's job is only to observe from the page: mark when the character landed in the value,
// and mark the first paint after it. The subtraction is done here because a driver-side clock
// would include the CDP round trip, which is not something a user experiences.

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
  };

  const nextPaint = () =>
    window.__sbNextPaint
      ? window.__sbNextPaint()
      : new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(() => r(performance.now()))));

  const onInput = (ev) => {
    if (!S.armed || ev.target !== S.target) return;
    // One in flight at a time. A burst of characters typed faster than the page can paint would
    // otherwise attribute one paint to several keystrokes and report each of them as fast.
    if (S.pending !== null) {
      S.dropped += 1;
      return;
    }
    const at = performance.now();
    const lengthAt = S.target.value.length;
    S.pending = at;
    nextPaint().then((paintedAt) => {
      S.samples.push({
        at_ms: Math.round(at * 10) / 10,
        latency_ms: Math.round((paintedAt - at) * 10) / 10,
        value_length: lengthAt,
      });
      S.pending = null;
    });
  };

  window.__sb.input = {
    // `selector` is resolved here rather than passed as a handle, so the driver can re-arm across
    // a page that re-rendered its composer without holding a stale node.
    arm(selector) {
      const el = document.querySelector(selector);
      if (!el) return { armed: false, reason: "no element matched " + selector };
      if (S.target && S.target !== el) S.target.removeEventListener("input", onInput, true);
      S.target = el;
      S.baseline = el.value === undefined ? "" : el.value;
      S.samples = [];
      S.dropped = 0;
      S.pending = null;
      S.armed = true;
      el.addEventListener("input", onInput, true);
      return { armed: true, baseline_length: S.baseline.length };
    },

    // Drain. `expected` is how many characters the driver actually sent, so the report can say
    // "27 of 30 keystrokes produced a measurement" instead of quoting a median over an unknown
    // denominator.
    collect(expected) {
      const samples = S.samples.slice();
      const latencies = samples.map((s) => s.latency_ms).sort((a, b) => a - b);
      const at = (q) =>
        latencies.length === 0 ? null : latencies[Math.min(latencies.length - 1, Math.floor(latencies.length * q))];
      const observedText = S.target ? S.target.value : null;
      S.samples = [];
      return {
        samples: samples.length,
        samples_attempted: true,
        expected: expected === undefined ? null : expected,
        // Not the same question as `samples`. A dropped sample is a keystroke that arrived while
        // a previous one had not painted yet, which is itself the symptom.
        coalesced: S.dropped,
        p50_ms: at(0.5),
        p95_ms: at(0.95),
        max_ms: latencies.length === 0 ? null : latencies[latencies.length - 1],
        // The FIRST sample is systematically a cold outlier and is reported separately rather
        // than dropped, because on a jammed page it is also the largest real number in the set.
        first_ms: samples.length === 0 ? null : samples[0].latency_ms,
        // Proof the characters reached the controlled component and not only the DOM node.
        text_length: observedText === null ? null : observedText.length,
        grew_by: observedText === null ? null : observedText.length - S.baseline.length,
      };
    },

    disarm() {
      if (S.target) S.target.removeEventListener("input", onInput, true);
      S.armed = false;
      S.target = null;
      return true;
    },
  };
})();
