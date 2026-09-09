// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/*
 * layoutcost.js -- counts and times the DOM operations suspected of forcing synchronous layout
 * during streaming.
 *
 * Under investigation: use-intent-aware-autoscroll.tsx, whose MutationObserver on the thread
 * viewport reads `scrollHeight`, writes `--aui-scroll-stabilizer` and calls `scrollTo` on every
 * delivery -- per streamed character, at a cost proportional to the whole thread. So the five
 * instrumented operations are exactly the ones that shape is made of: scrollHeight reads (the
 * forced-layout trigger), scrollTop writes, scrollTo calls, MutationObserver callbacks AND
 * records per callback (one callback with 400 records and 400 callbacks cost wildly differently
 * and look identical in a callback-only profile), and custom-property writes with
 * `--aui-scroll-stabilizer` counted separately. Timings are here because a count cannot tell
 * 4,000 cheap reads from 4,000 that each walk a 300-message thread.
 * Configured childList + subtree + characterData + an attributeFilter.
 *
 * SELF COST. Wrapping a getter to time it makes it slower, so the distortion is measured rather
 * than assumed: `selfCostEstimate()` times N wrapped reads against N through the ORIGINAL
 * descriptor on a detached clean element, giving the per-call wrapper overhead; and the Python
 * driver runs the same cell with and without injection and compares frame statistics, which
 * catches cache effects and lost inlining that no microbenchmark sees. If the two runs disagree
 * about the app, the counts stay usable and the timings do not.
 * Reported as `overheadMsPerCall`.
 *
 * `clockGranularityMs` exists because a clean read can be faster than a clamped
 * `performance.now()` can resolve: maxMs 0 means "below the clock", not "free".
 * A genuine 0.003 ms read is indistinguishable from zero.
 *
 * OFF BY DEFAULT because it perturbs the measurement; the driver injects it only for the deep
 * tier, where the question has narrowed to "which operation". `window.__sbLayoutCostDisabled`
 * is a secondary escape hatch for bisecting the instrument itself.
 *
 * It does not measure layout time, only how often the app asks for something that can force it;
 * attribution comes from the trace. `window.scrollY`, `getBoundingClientRect`, `offsetHeight`
 * and `getComputedStyle` also force layout and are deliberately NOT wrapped: they are not in the
 * path under investigation and every wrapper makes the run less like the app.
 *
 * ZERO DISCIPLINE. A 0 because the operation did not happen and a 0 because the patch could not
 * be installed must never print the same, so `snapshot()` carries `attempted` per family and
 * `unavailable` lists names whose descriptor was missing or non-configurable. Failing to install
 * is not an error and never throws.
 * A WebKit build that refuses the patch reads as not attempted.
 */

(function () {
  "use strict";

  if (typeof window === "undefined" || !window) {
    return;
  }

  var W = window;

  // Idempotence: add_init_script runs per document; double wrapping would double every count.
  if (W.__sbLayoutCostInstalled) {
    return;
  }
  W.__sbLayoutCostInstalled = true;

  var VIEWPORT_CLASS = "aui-stream-viewport";
  var STABILIZER_PROP = "--aui-scroll-stabilizer";
  var MAX_KEYS = 24;
  var OTHER_KEY = "__other__";

  // Capture before the app can replace performance.now.
  var perf = W.performance;
  var rawNow =
    perf && typeof perf.now === "function" ? perf.now.bind(perf) : null;
  var now =
    rawNow ||
    function () {
      return Date.now();
    };
  var hasHiResClock = !!rawNow;

  var doc = W.document || null;

  function nowStamp() {
    try {
      return new Date().toISOString();
    } catch (e) {
      return null;
    }
  }


  function freshTiming() {
    return { totalMs: 0, maxMs: 0, samples: 0 };
  }

  function freshMoBucket() {
    return { callbacks: 0, records: 0, totalMs: 0, maxMs: 0 };
  }

  function freshCounters() {
    return {
      scrollHeightReads: 0,
      scrollTopWrites: 0,
      scrollToCalls: 0,
      moCallbacks: 0,
      moRecords: 0,
      customPropSets: 0,
      stabilizerPropSets: 0
    };
  }

  function freshTimings() {
    return {
      scrollHeightRead: freshTiming(),
      scrollTo: freshTiming(),
      moCallback: freshTiming()
    };
  }

  function freshMo() {
    return {
      observersConstructed: 0,
      observersMatchedViewport: 0,
      observersWithAriaExpandedFilter: 0,
      observeCalls: 0,
      viewportObserver: freshMoBucket(),
      other: freshMoBucket()
    };
  }

  var state = {
    active: false,
    counters: freshCounters(),
    timings: freshTimings(),
    mo: freshMo(),
    breakdown: Object.create(null),
    breakdownKeyCount: 0,
    breakdownCapped: false,
    attempted: {
      scrollHeight: false,
      scrollTop: false,
      scrollTo: false,
      mutationObserver: false,
      customProps: false
    },
    unavailable: [],
    selfCost: null,
    clockGranularityMs: null,
    // Sink so the discarded read cannot be optimised out.
    sink: 0
  };

  var originals = {
    scrollHeightDesc: null,
    scrollTopDesc: null,
    scrollToDesc: null,
    setPropertyDesc: null,
    MutationObserver: null
  };

  function markUnavailable(name, reason) {
    try {
      state.unavailable.push({ name: name, reason: reason });
    } catch (e) {
      /* nothing sensible to do here, and throwing would take the app with it */
    }
  }

  function record(timing, dt) {
    timing.totalMs += dt;
    timing.samples += 1;
    if (dt > timing.maxMs) {
      timing.maxMs = dt;
    }
  }

  // Keyed by a coarse stable descriptor, not element identity, so detached DOM is not pinned alive.
  // Coarse because the question is which KIND of element is read, and a 300-message thread has
  // about six kinds.
  // Cap keys so a page reading thousands of distinct elements cannot grow an unbounded object.

  function keyFor(el) {
    try {
      if (!el || typeof el !== "object") {
        return "unknown";
      }
      var tag = "";
      try {
        tag = String(el.tagName || el.nodeName || "?").toLowerCase();
      } catch (e) {
        tag = "?";
      }
      var cls = "";
      var isViewport = false;
      try {
        var list = el.classList;
        if (list) {
          if (list.length > 0) {
            cls = String(list[0] || "");
          }
          // contains(), not matches(): no selector engine inside the getter.
          isViewport = !!list.contains && list.contains(VIEWPORT_CLASS);
        }
      } catch (e) {
        /* SVG and exotic hosts: fall through with what we have */
      }
      return tag + (cls ? "." + cls : "") + (isViewport ? "[vp]" : "");
    } catch (e) {
      return "unknown";
    }
  }

  function bucketFor(el) {
    var key = keyFor(el);
    var b = state.breakdown[key];
    if (b) {
      return b;
    }
    var isOther = false;
    if (state.breakdownKeyCount >= MAX_KEYS) {
      state.breakdownCapped = true;
      key = OTHER_KEY;
      isOther = true;
      b = state.breakdown[key];
      if (b) {
        return b;
      }
    }
    b = {
      scrollHeightReads: 0,
      scrollHeightMs: 0,
      scrollHeightMaxMs: 0,
      scrollTopWrites: 0,
      scrollToCalls: 0,
      scrollToMs: 0,
      scrollToMaxMs: 0
    };
    state.breakdown[key] = b;
    if (!isOther) {
      // The overflow bucket is not charged to the budget, so breakdownKeyCount counts REAL keys.
      // At most MAX_KEYS + 1 entries.
      state.breakdownKeyCount += 1;
    }
    return b;
  }


  var ElementProto =
    typeof W.Element === "function" && W.Element.prototype
      ? W.Element.prototype
      : null;

  function grabDescriptor(proto, name) {
    if (!proto) {
      markUnavailable(name, "Element.prototype is not available");
      return null;
    }
    var d = null;
    try {
      d = Object.getOwnPropertyDescriptor(proto, name);
    } catch (e) {
      markUnavailable(name, "getOwnPropertyDescriptor threw: " + e);
      return null;
    }
    if (!d) {
      markUnavailable(name, "no own descriptor on prototype");
      return null;
    }
    if (!d.configurable) {
      markUnavailable(name, "descriptor is not configurable");
      return null;
    }
    return d;
  }

  function installScrollHeight() {
    var d = grabDescriptor(ElementProto, "scrollHeight");
    if (!d || typeof d.get !== "function") {
      if (d) {
        markUnavailable("scrollHeight", "descriptor has no getter");
      }
      return;
    }
    originals.scrollHeightDesc = d;
    var origGet = d.get;
    try {
      Object.defineProperty(ElementProto, "scrollHeight", {
        configurable: true,
        enumerable: d.enumerable,
        get: function () {
          if (!state.active) {
            return origGet.call(this);
          }
          var t0 = now();
          // Outside try/catch: a throwing native getter must reach the app unchanged.
          var value = origGet.call(this);
          try {
            var dt = now() - t0;
            state.counters.scrollHeightReads += 1;
            record(state.timings.scrollHeightRead, dt);
            var b = bucketFor(this);
            b.scrollHeightReads += 1;
            b.scrollHeightMs += dt;
            if (dt > b.scrollHeightMaxMs) {
              b.scrollHeightMaxMs = dt;
            }
          } catch (e) {
            /* bookkeeping is never allowed to break a property read */
          }
          return value;
        },
        set: d.set
      });
      state.attempted.scrollHeight = true;
    } catch (e) {
      originals.scrollHeightDesc = null;
      markUnavailable("scrollHeight", "defineProperty threw: " + e);
    }
  }

  function installScrollTop() {
    var d = grabDescriptor(ElementProto, "scrollTop");
    if (!d || typeof d.set !== "function") {
      if (d) {
        markUnavailable("scrollTop", "descriptor has no setter");
      }
      return;
    }
    originals.scrollTopDesc = d;
    var origSet = d.set;
    try {
      Object.defineProperty(ElementProto, "scrollTop", {
        configurable: true,
        enumerable: d.enumerable,
        // scrollTop's getter is left alone: not the operation under investigation.
        get: d.get,
        set: function (v) {
          if (!state.active) {
            origSet.call(this, v);
            return;
          }
          // Writes are counted, not timed; the cost lands in the next layout.
          try {
            state.counters.scrollTopWrites += 1;
            bucketFor(this).scrollTopWrites += 1;
          } catch (e) {
            /* fall through to the real setter regardless */
          }
          origSet.call(this, v);
        }
      });
      state.attempted.scrollTop = true;
    } catch (e) {
      originals.scrollTopDesc = null;
      markUnavailable("scrollTop", "defineProperty threw: " + e);
    }
  }

  function installScrollTo() {
    var d = grabDescriptor(ElementProto, "scrollTo");
    if (!d || typeof d.value !== "function") {
      if (d) {
        markUnavailable("scrollTo", "descriptor is not a method");
      }
      return;
    }
    if (!d.writable && !d.configurable) {
      markUnavailable("scrollTo", "descriptor is not writable");
      return;
    }
    originals.scrollToDesc = d;
    var orig = d.value;
    try {
      Object.defineProperty(ElementProto, "scrollTo", {
        configurable: true,
        enumerable: d.enumerable,
        writable: true,
        value: function () {
          if (!state.active) {
            return orig.apply(this, arguments);
          }
          var t0 = now();
          var out = orig.apply(this, arguments);
          try {
            var dt = now() - t0;
            state.counters.scrollToCalls += 1;
            record(state.timings.scrollTo, dt);
            var b = bucketFor(this);
            b.scrollToCalls += 1;
            b.scrollToMs += dt;
            if (dt > b.scrollToMaxMs) {
              b.scrollToMaxMs = dt;
            }
          } catch (e) {
            /* never turn a scroll into an exception */
          }
          return out;
        }
      });
      state.attempted.scrollTo = true;
    } catch (e) {
      originals.scrollToDesc = null;
      markUnavailable("scrollTo", "defineProperty threw: " + e);
    }
  }


  function installSetProperty() {
    var proto =
      typeof W.CSSStyleDeclaration === "function" &&
      W.CSSStyleDeclaration.prototype
        ? W.CSSStyleDeclaration.prototype
        : null;
    if (!proto) {
      markUnavailable("setProperty", "CSSStyleDeclaration is not available");
      return;
    }
    var d = null;
    try {
      d = Object.getOwnPropertyDescriptor(proto, "setProperty");
    } catch (e) {
      markUnavailable("setProperty", "getOwnPropertyDescriptor threw: " + e);
      return;
    }
    if (!d || typeof d.value !== "function") {
      markUnavailable("setProperty", "no own method on prototype");
      return;
    }
    if (!d.configurable && !d.writable) {
      markUnavailable("setProperty", "descriptor is not configurable");
      return;
    }
    originals.setPropertyDesc = d;
    var orig = d.value;
    try {
      Object.defineProperty(proto, "setProperty", {
        configurable: true,
        enumerable: d.enumerable,
        writable: true,
        value: function (name) {
          if (state.active) {
            try {
              // Only custom properties: ordinary style writes would bury the signal.
              if (typeof name === "string" && name.charCodeAt(0) === 45 && name.charCodeAt(1) === 45) {
                state.counters.customPropSets += 1;
                if (name === STABILIZER_PROP) {
                  state.counters.stabilizerPropSets += 1;
                }
              }
            } catch (e) {
              /* fall through */
            }
          }
          return orig.apply(this, arguments);
        }
      });
      state.attempted.customProps = true;
    } catch (e) {
      originals.setPropertyDesc = null;
      markUnavailable("setProperty", "defineProperty threw: " + e);
    }
  }

  // A subclass, not a Proxy: instanceof and the native methods keep working.
  // Two independent discriminators (viewport class, aria-expanded filter) separate the autoscroll
  // observer from React's, reported separately so disagreement is visible.
  // The split key is `viewportObserver`; no other observer requests that attributeFilter.

  function installMutationObserver() {
    var Native = W.MutationObserver;
    if (typeof Native !== "function") {
      markUnavailable("MutationObserver", "constructor is not available");
      return;
    }
    originals.MutationObserver = Native;

    var moStates =
      typeof W.WeakMap === "function" ? new W.WeakMap() : null;

    function bucketOf(obsState) {
      return obsState && (obsState.matchedViewport || obsState.hasAriaExpanded)
        ? state.mo.viewportObserver
        : state.mo.other;
    }

    var Wrapped;
    try {
      Wrapped = class extends Native {
        constructor(callback) {
          if (typeof callback !== "function") {
            // Let the native constructor produce its own TypeError.
            super(callback);
            return;
          }
          var obsState = { matchedViewport: false, hasAriaExpanded: false };
          super(function (records, observer) {
            if (!state.active) {
              return callback.call(this, records, observer);
            }
            var n = 0;
            try {
              n = records && typeof records.length === "number" ? records.length : 0;
            } catch (e) {
              n = 0;
            }
            var t0 = now();
            try {
              return callback.call(this, records, observer);
            } finally {
              // finally: a throwing callback still consumed the time.
              try {
                var dt = now() - t0;
                state.counters.moCallbacks += 1;
                state.counters.moRecords += n;
                record(state.timings.moCallback, dt);
                var b = bucketOf(obsState);
                b.callbacks += 1;
                b.records += n;
                b.totalMs += dt;
                if (dt > b.maxMs) {
                  b.maxMs = dt;
                }
              } catch (e) {
                /* bookkeeping only */
              }
            }
          });
          try {
            state.mo.observersConstructed += 1;
            if (moStates) {
              moStates.set(this, obsState);
            }
          } catch (e) {
            /* an uncounted observer is better than a failed construction */
          }
        }

        observe(target, init) {
          try {
            state.mo.observeCalls += 1;
            var obsState = moStates ? moStates.get(this) : null;
            if (obsState) {
              if (
                !obsState.matchedViewport &&
                target &&
                target.nodeType === 1 &&
                target.classList &&
                target.classList.contains &&
                target.classList.contains(VIEWPORT_CLASS)
              ) {
                obsState.matchedViewport = true;
                state.mo.observersMatchedViewport += 1;
              }
              if (!obsState.hasAriaExpanded && init && init.attributeFilter) {
                var f = init.attributeFilter;
                for (var i = 0; i < f.length; i++) {
                  if (f[i] === "aria-expanded") {
                    obsState.hasAriaExpanded = true;
                    state.mo.observersWithAriaExpandedFilter += 1;
                    break;
                  }
                }
              }
            }
          } catch (e) {
            /* classification is optional, observing is not */
          }
          // apply(arguments), not (target, init): observe() with no options has its own spec behaviour.
          return super.observe.apply(this, arguments);
        }
      };
    } catch (e) {
      originals.MutationObserver = null;
      markUnavailable("MutationObserver", "subclassing threw: " + e);
      return;
    }

    try {
      Object.defineProperty(Wrapped, "name", {
        value: "MutationObserver",
        configurable: true
      });
    } catch (e) {
      /* cosmetic only: some feature detection sniffs constructor names */
    }

    try {
      W.MutationObserver = Wrapped;
      state.attempted.mutationObserver = true;
    } catch (e) {
      originals.MutationObserver = null;
      markUnavailable("MutationObserver", "assignment to window threw: " + e);
    }
  }


  function measureClockGranularityMs() {
    var best = null;
    for (var trial = 0; trial < 5; trial++) {
      var a = now();
      var b = a;
      var guard = 0;
      while (b === a && guard < 2000000) {
        b = now();
        guard++;
      }
      if (b !== a) {
        var d = b - a;
        if (best === null || d < best) {
          best = d;
        }
      }
    }
    return best;
  }

  function copyRawState() {
    return {
      counters: copyPlain(state.counters),
      timings: copyPlain(state.timings),
      mo: copyPlain(state.mo),
      breakdown: copyPlain(state.breakdown),
      breakdownKeyCount: state.breakdownKeyCount,
      breakdownCapped: state.breakdownCapped
    };
  }

  function restoreRawState(saved) {
    state.counters = saved.counters;
    state.timings = saved.timings;
    state.mo = saved.mo;
    state.breakdown = saved.breakdown;
    state.breakdownKeyCount = saved.breakdownKeyCount;
    state.breakdownCapped = saved.breakdownCapped;
    // Rebind the aliases or a caller reading __sbLayoutCost.counters watches a detached copy.
    api.counters = state.counters;
    api.timings = state.timings;
    api.mo = state.mo;
  }

  function selfCostEstimate(sampleCount) {
    var n =
      typeof sampleCount === "number" && sampleCount > 0
        ? Math.floor(sampleCount)
        : 200;

    if (!state.attempted.scrollHeight || !originals.scrollHeightDesc) {
      return {
        attempted: false,
        reason: "scrollHeight getter was not patched, so there is no overhead to measure",
        samples: 0,
        hasHiResClock: hasHiResClock
      };
    }
    if (!doc || typeof doc.createElement !== "function") {
      return {
        attempted: false,
        reason: "no document to build a probe element in",
        samples: 0,
        hasHiResClock: hasHiResClock
      };
    }

    var origGet = originals.scrollHeightDesc.get;
    var result;
    // Save/restore counters so the probe does not contaminate the measurement it corrects.
    var saved = copyRawState();
    try {
      var probe = doc.createElement("div");
      probe.className = "sb-selfcost-probe";
      // Detached and never inserted: this measures the WRAPPER, not the app's layout.
      var i;
      var sink = 0;

      // Warm both paths so neither loop is charged for the JIT.
      for (i = 0; i < n; i++) {
        sink += probe.scrollHeight;
      }
      for (i = 0; i < n; i++) {
        sink += origGet.call(probe);
      }

      var w0 = now();
      for (i = 0; i < n; i++) {
        sink += probe.scrollHeight;
      }
      var wrappedMs = now() - w0;

      var r0 = now();
      for (i = 0; i < n; i++) {
        sink += origGet.call(probe);
      }
      var rawMs = now() - r0;

      state.sink = sink;

      var granularity = state.clockGranularityMs;
      if (granularity === null || granularity === undefined) {
        granularity = measureClockGranularityMs();
      }

      var perWrapped = wrappedMs / n;
      var perRaw = rawMs / n;
      result = {
        attempted: true,
        samples: n,
        wrappedMsPerCall: perWrapped,
        rawMsPerCall: perRaw,
        overheadMsPerCall: perWrapped - perRaw,
        wrappedTotalMs: wrappedMs,
        rawTotalMs: rawMs,
        clockGranularityMs: granularity,
        hasHiResClock: hasHiResClock,
        note:
          "Detached clean element: this is the cost of the wrapper only, not of the layout a " +
          "dirty attached element would force. Multiply overheadMsPerCall by " +
          "counters.scrollHeightReads to bound how much of the reported scrollHeight total " +
          "belongs to the instrument. Cross-check against the driver's paired run with the " +
          "instrument absent."
      };
      state.clockGranularityMs = granularity;
    } catch (e) {
      result = {
        attempted: false,
        reason: "probe threw: " + e,
        samples: 0,
        hasHiResClock: hasHiResClock
      };
    } finally {
      try {
        restoreRawState(saved);
      } catch (e2) {
        /* leaving inflated counters would be worse than this catch being empty, but there is
           nothing further to try */
      }
    }

    state.selfCost = result;
    return copyPlain(result);
  }


  function copyPlain(v) {
    // Hand written: JSON turns non-finite into null and drops undefined.
    if (v === null || typeof v !== "object") {
      return v;
    }
    var out;
    var k;
    if (Object.prototype.toString.call(v) === "[object Array]") {
      out = [];
      for (k = 0; k < v.length; k++) {
        out[k] = copyPlain(v[k]);
      }
      return out;
    }
    out = {};
    for (k in v) {
      if (Object.prototype.hasOwnProperty.call(v, k)) {
        out[k] = copyPlain(v[k]);
      }
    }
    return out;
  }

  function snapshot() {
    var out;
    try {
      out = {
        enabled: !!state.active,
        installedAt: api.installedAt,
        resetAt: api.resetAt,
        snapshotAt: nowStamp(),
        // Per family, so zero counts differ from a patch that never landed.
        attempted: copyPlain(state.attempted),
        unavailable: copyPlain(state.unavailable),
        counters: copyPlain(state.counters),
        timings: copyPlain(state.timings),
        mo: copyPlain(state.mo),
        breakdown: copyPlain(state.breakdown),
        breakdownCapped: !!state.breakdownCapped,
        breakdownKeyCount: state.breakdownKeyCount,
        breakdownMaxKeys: MAX_KEYS,
        otherKey: OTHER_KEY,
        clockGranularityMs: state.clockGranularityMs,
        hasHiResClock: hasHiResClock,
        selfCost: copyPlain(state.selfCost)
      };
    } catch (e) {
      out = { enabled: false, error: String(e) };
    }
    return out;
  }

  function reset() {
    try {
      state.counters = freshCounters();
      state.timings = freshTimings();
      state.mo = freshMo();
      state.breakdown = Object.create(null);
      state.breakdownKeyCount = 0;
      state.breakdownCapped = false;
      api.resetAt = nowStamp();
      api.counters = state.counters;
      api.timings = state.timings;
      api.mo = state.mo;
    } catch (e) {
      /* reset failing loudly mid run would lose the run */
    }
    return true;
  }

  function uninstall() {
    // Restores the captured descriptors; counters are left alone.
    state.active = false;
    try {
      if (originals.scrollHeightDesc && ElementProto) {
        Object.defineProperty(ElementProto, "scrollHeight", originals.scrollHeightDesc);
      }
    } catch (e) {
      /* leave it wrapped rather than throw */
    }
    try {
      if (originals.scrollTopDesc && ElementProto) {
        Object.defineProperty(ElementProto, "scrollTop", originals.scrollTopDesc);
      }
    } catch (e) {
      /* as above */
    }
    try {
      if (originals.scrollToDesc && ElementProto) {
        Object.defineProperty(ElementProto, "scrollTo", originals.scrollToDesc);
      }
    } catch (e) {
      /* as above */
    }
    try {
      if (originals.setPropertyDesc && W.CSSStyleDeclaration) {
        Object.defineProperty(
          W.CSSStyleDeclaration.prototype,
          "setProperty",
          originals.setPropertyDesc
        );
      }
    } catch (e) {
      /* as above */
    }
    try {
      if (originals.MutationObserver) {
        W.MutationObserver = originals.MutationObserver;
      }
    } catch (e) {
      /* as above */
    }
    api.enabled = false;
    return true;
  }

  var api = {
    version: 1,
    enabled: false,
    installedAt: nowStamp(),
    resetAt: null,
    counters: state.counters,
    timings: state.timings,
    mo: state.mo,
    unavailable: state.unavailable,
    attempted: state.attempted,
    viewportClass: VIEWPORT_CLASS,
    stabilizerProperty: STABILIZER_PROP,
    snapshot: snapshot,
    reset: reset,
    uninstall: uninstall,
    selfCostEstimate: selfCostEstimate
  };

  W.__sbLayoutCost = api;

  if (W.__sbLayoutCostDisabled === true) {
    // Injected but told to stand down: everything reads as not attempted, not a row of zeros.
    return;
  }

  installScrollHeight();
  installScrollTop();
  installScrollTo();
  installSetProperty();
  installMutationObserver();

  state.active =
    state.attempted.scrollHeight ||
    state.attempted.scrollTop ||
    state.attempted.scrollTo ||
    state.attempted.customProps ||
    state.attempted.mutationObserver;
  api.enabled = state.active;
})();
