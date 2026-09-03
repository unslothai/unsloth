// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
// Keystroke-to-paint, measured from the page side of a REAL key event.
// WHY THIS IS NOT THE SALVAGED KEYSTROKE_JS. The old harness typed by calling the native value
// setter and dispatching a synthetic `input` event. That reaches React's controlled input, but
// enters the pipeline AFTER hit testing and event routing and carries no `latencyInfo`, so it
// cannot show input queueing delay at all: it is dispatched from a task already running, so the
// queue is empty by construction and the number reads clean exactly when a user would be waiting
// longest.
// So the driver types with `page.keyboard`, which goes in through CDP as a real input event, and
// this file only observes from the page: mark when the key arrived, and mark the first paint
// after the character landed in the value. The subtraction is done here because a driver-side
// clock would include the CDP round trip.
// THE CLOCK STARTS AT THE KEYDOWN, NOT AT THE INPUT HANDLER. `input` is dispatched as the default
// action of `keydown`, so a handler that blocks the main thread on the way in has ALREADY
// finished when `input` fires and a start taken there subtracts the wait out of the number.
// Against the harness's own 400 ms injected keydown stall, an input-anchored clock moved keystroke
// p95 by -14.8 ms, and the integrity gate in `instruments/selfcheck.py`, which requires 350 ms of
// movement, can never pass on it.
// A trusted event's `timeStamp` is a `DOMHighResTimeStamp` on the same origin as
// `performance.now()`, set when the occurrence happened rather than when it was dispatched, so
// `keydown.timeStamp` is the hardware arrival time and carries the queueing delay. Verified on all
// three engines: with the stall armed, `performance.now()` inside the `input` handler is ~400 ms
// past `keydown.timeStamp` and ~0 ms past the input event's own timeStamp.

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
    // Every `input` event this instrument saw, whether it became a sample or was coalesced behind an
    // unfinished paint. Without it there is no denominator: `samples` alone cannot distinguish 'the
    // page painted every keystroke' from 'most never reached here'.
    seen: 0,
  };

  const nextPaint = () =>
    window.__sbNextPaint
      ? window.__sbNextPaint()
      : new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(() => r(performance.now()))));

  // The key that produced the character being measured: the LAST unconsumed keydown, because a key
  // that produced no character must not anchor the next one that did.
  const onKeyDown = (ev) => {
    if (!S.armed || (S.target && ev.target !== S.target)) return;
    S.keyAt = ev.timeStamp;
  };

  // Consume the anchor, and refuse an implausible one rather than quoting it. A page-constructed
  // synthetic event carries its CONSTRUCTOR's time, an engine that does not put key events on the
  // performance timeline could report 0, and a composition commit produces an `input` with no
  // keydown. Each falls back to this handler's own clock, the old behaviour, and is counted so the
  // report can say how many samples were unanchored.
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
    // One in flight at a time: a burst typed faster than the page can paint would otherwise attribute
    // one paint to several keystrokes and report each as fast.
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
        // Keystroke to paint: from the key arriving to the frame that shows it.
        latency_ms: Math.round((paintedAt - started) * 10) / 10,
        // The two halves, kept separately so a regression can be attributed: how long the key waited to be
        // handled, and how long the page then took to paint it.
        input_delay_ms: keyAt === null ? null : Math.round((at - keyAt) * 10) / 10,
        paint_ms: Math.round((paintedAt - at) * 10) / 10,
        anchored_on: keyAt === null ? "input" : "keydown",
        value_length: lengthAt,
      });
      S.pending = null;
    });
  };

  window.__sb.input = {
    // `selector` is resolved here rather than passed as a handle, so the driver can re-arm across a
    // page that re-rendered its composer without holding a stale node.
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
      // On the WINDOW, in capture, so the anchor is taken however the app routes the key and even if a
      // handler stops propagation. Idempotent: re-arming across a re-rendered composer must not leave
      // two behind, each overwriting the other's anchor.
      window.removeEventListener("keydown", onKeyDown, true);
      window.addEventListener("keydown", onKeyDown, true);
      return { armed: true, baseline_length: S.baseline.length };
    },

    // IS ANYTHING STILL IN FLIGHT? The driver polls this instead of waiting a fixed interval, which
    // would lose whichever keystroke had not painted when it expired: the SLOWEST one, so the metric
    // would drop precisely the sample it exists to catch and a build that made typing worse would read
    // faster. A bigger constant has the same defect on a slower machine.
    settled() {
      return { pending: S.pending !== null, samples: S.samples.length, seen: S.seen };
    },

    // Drain. `expected` is how many characters the driver actually sent, so the report can say '27 of
    // 30 keystrokes produced a measurement' instead of quoting a median over an unknown denominator.
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
      // A sample still in flight AT THIS MOMENT is one the collect is about to lose. Reported so the
      // driver can fail the reading rather than publish a percentile over what survived.
      const pendingNow = S.pending !== null;
      S.samples = [];
      S.unanchored = 0;
      S.seen = 0;
      return {
        samples: samples.length,
        samples_attempted: true,
        expected: expected === undefined ? null : expected,
        // The denominator. `inputs_seen` is every keystroke that reached this instrument; `samples +
        // coalesced` must account for all of them, and `pending_at_collect` says whether one was thrown
        // away by the drain.
        inputs_seen: seen,
        pending_at_collect: pendingNow,
        // How many samples could not be anchored on their own key event and fell back to the input
        // handler's clock. A number quoted from those understates the wait by the queueing delay, so it
        // is reported rather than blended in.
        unanchored: unanchored,
        input_delay_p95_ms:
          delays.length === 0 ? null : delays[Math.min(delays.length - 1, Math.floor(delays.length * 0.95))],
        // Not the same question as `samples`: a dropped sample is a keystroke that arrived while a
        // previous one had not painted, which is itself the symptom.
        coalesced: S.dropped,
        p50_ms: at(0.5),
        p95_ms: at(0.95),
        max_ms: latencies.length === 0 ? null : latencies[latencies.length - 1],
        // The FIRST sample is systematically a cold outlier and is reported separately rather than
        // dropped, because on a jammed page it is also the largest real number in the set.
        first_ms: samples.length === 0 ? null : samples[0].latency_ms,
        // Proof the characters reached the controlled component and not only the DOM node.
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
