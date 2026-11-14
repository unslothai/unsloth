// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/*
 * Counts forced-layout requests while streaming, not layout time; off by default as it perturbs;
 * zero counts must be distinguishable from never-installed patches; maxMs 0 means below
 * clockGranularityMs, not free.
 */

(function () {
  "use strict";

  if (typeof window === "undefined" || !window) {
    return;
  }

  const W = window;

  // Idempotence: add_init_script runs per document; double wrapping would double every count.
  if (W.__sbLayoutCostInstalled) {
    return;
  }
  W.__sbLayoutCostInstalled = true;

  const VIEWPORT_CLASS = "aui-stream-viewport";
  const STABILIZER_PROP = "--aui-scroll-stabilizer";
  const MAX_KEYS = 24;
  const OTHER_KEY = "__other__";

  // Capture before the app can replace performance.now.
  const perf = W.performance;
  const rawNow =
    perf && typeof perf.now === "function" ? perf.now.bind(perf) : null;
  const now =
    rawNow ||
    function () {
      return Date.now();
    };
  const hasHiResClock = !!rawNow;

  const doc = W.document || null;

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
      stabilizerPropSets: 0,
    };
  }

  function freshTimings() {
    return {
      scrollHeightRead: freshTiming(),
      scrollTo: freshTiming(),
      moCallback: freshTiming(),
    };
  }

  function freshMo() {
    return {
      observersConstructed: 0,
      observersMatchedViewport: 0,
      observersWithAriaExpandedFilter: 0,
      observeCalls: 0,
      viewportObserver: freshMoBucket(),
      other: freshMoBucket(),
    };
  }

  const state = {
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
      customProps: false,
    },
    unavailable: [],
    selfCost: null,
    clockGranularityMs: null,
    // Sink so the discarded read cannot be optimised out.
    sink: 0,
  };

  const originals = {
    scrollHeightDesc: null,
    scrollTopDesc: null,
    scrollToDesc: null,
    setPropertyDesc: null,
    MutationObserver: null,
  };

  function markUnavailable(name, reason) {
    try {
      state.unavailable.push({ name: name, reason: reason });
    } catch (e) {
      /* throwing here would take the app with it */
    }
  }

  function record(timing, dt) {
    timing.totalMs += dt;
    timing.samples += 1;
    if (dt > timing.maxMs) {
      timing.maxMs = dt;
    }
  }

  // Keyed by a coarse descriptor, not element identity, so detached DOM is not pinned; keys capped.

  function keyFor(el) {
    try {
      if (!el || typeof el !== "object") {
        return "unknown";
      }
      let tag = "";
      try {
        tag = String(el.tagName || el.nodeName || "?").toLowerCase();
      } catch (e) {
        tag = "?";
      }
      let cls = "";
      let isViewport = false;
      try {
        const list = el.classList;
        if (list) {
          if (list.length > 0) {
            cls = String(list[0] || "");
          }
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
    let key = keyFor(el);
    let b = state.breakdown[key];
    if (b) {
      return b;
    }
    let isOther = false;
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
      scrollToMaxMs: 0,
    };
    state.breakdown[key] = b;
    if (!isOther) {
      // The overflow bucket is not charged to the budget, so breakdownKeyCount counts real keys.
      state.breakdownKeyCount += 1;
    }
    return b;
  }

  const ElementProto =
    typeof W.Element === "function" && W.Element.prototype
      ? W.Element.prototype
      : null;

  function grabDescriptor(proto, name) {
    if (!proto) {
      markUnavailable(name, "Element.prototype is not available");
      return null;
    }
    let d = null;
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
    const d = grabDescriptor(ElementProto, "scrollHeight");
    if (!d || typeof d.get !== "function") {
      if (d) {
        markUnavailable("scrollHeight", "descriptor has no getter");
      }
      return;
    }
    originals.scrollHeightDesc = d;
    const origGet = d.get;
    try {
      Object.defineProperty(ElementProto, "scrollHeight", {
        configurable: true,
        enumerable: d.enumerable,
        get: function () {
          if (!state.active) {
            return origGet.call(this);
          }
          const t0 = now();
          // Outside try/catch: a throwing native getter must reach the app unchanged.
          const value = origGet.call(this);
          try {
            const dt = now() - t0;
            state.counters.scrollHeightReads += 1;
            record(state.timings.scrollHeightRead, dt);
            const b = bucketFor(this);
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
        set: d.set,
      });
      state.attempted.scrollHeight = true;
    } catch (e) {
      originals.scrollHeightDesc = null;
      markUnavailable("scrollHeight", "defineProperty threw: " + e);
    }
  }

  function installScrollTop() {
    const d = grabDescriptor(ElementProto, "scrollTop");
    if (!d || typeof d.set !== "function") {
      if (d) {
        markUnavailable("scrollTop", "descriptor has no setter");
      }
      return;
    }
    originals.scrollTopDesc = d;
    const origSet = d.set;
    try {
      Object.defineProperty(ElementProto, "scrollTop", {
        configurable: true,
        enumerable: d.enumerable,
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
        },
      });
      state.attempted.scrollTop = true;
    } catch (e) {
      originals.scrollTopDesc = null;
      markUnavailable("scrollTop", "defineProperty threw: " + e);
    }
  }

  function installScrollTo() {
    const d = grabDescriptor(ElementProto, "scrollTo");
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
    const orig = d.value;
    try {
      Object.defineProperty(ElementProto, "scrollTo", {
        configurable: true,
        enumerable: d.enumerable,
        writable: true,
        value: function () {
          if (!state.active) {
            return orig.apply(this, arguments);
          }
          const t0 = now();
          const out = orig.apply(this, arguments);
          try {
            const dt = now() - t0;
            state.counters.scrollToCalls += 1;
            record(state.timings.scrollTo, dt);
            const b = bucketFor(this);
            b.scrollToCalls += 1;
            b.scrollToMs += dt;
            if (dt > b.scrollToMaxMs) {
              b.scrollToMaxMs = dt;
            }
          } catch (e) {
            /* never turn a scroll into an exception */
          }
          return out;
        },
      });
      state.attempted.scrollTo = true;
    } catch (e) {
      originals.scrollToDesc = null;
      markUnavailable("scrollTo", "defineProperty threw: " + e);
    }
  }

  function installSetProperty() {
    const proto =
      typeof W.CSSStyleDeclaration === "function" &&
      W.CSSStyleDeclaration.prototype
        ? W.CSSStyleDeclaration.prototype
        : null;
    if (!proto) {
      markUnavailable("setProperty", "CSSStyleDeclaration is not available");
      return;
    }
    let d = null;
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
    const orig = d.value;
    try {
      Object.defineProperty(proto, "setProperty", {
        configurable: true,
        enumerable: d.enumerable,
        writable: true,
        value: function (name) {
          if (state.active) {
            try {
              if (
                typeof name === "string" &&
                name.charCodeAt(0) === 45 &&
                name.charCodeAt(1) === 45
              ) {
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
        },
      });
      state.attempted.customProps = true;
    } catch (e) {
      originals.setPropertyDesc = null;
      markUnavailable("setProperty", "defineProperty threw: " + e);
    }
  }

  // A subclass, not a Proxy, so instanceof and native methods keep working. The autoscroll observer
  // is split out (viewportObserver) by viewport class and its aria-expanded attributeFilter.

  function installMutationObserver() {
    const Native = W.MutationObserver;
    if (typeof Native !== "function") {
      markUnavailable("MutationObserver", "constructor is not available");
      return;
    }
    originals.MutationObserver = Native;

    const moStates = typeof W.WeakMap === "function" ? new W.WeakMap() : null;

    function bucketOf(obsState) {
      return obsState && (obsState.matchedViewport || obsState.hasAriaExpanded)
        ? state.mo.viewportObserver
        : state.mo.other;
    }

    let Wrapped;
    try {
      Wrapped = class extends Native {
        constructor(callback) {
          if (typeof callback !== "function") {
            super(callback);
            return;
          }
          const obsState = { matchedViewport: false, hasAriaExpanded: false };
          super(function (records, observer) {
            if (!state.active) {
              return callback.call(this, records, observer);
            }
            let n = 0;
            try {
              n =
                records && typeof records.length === "number"
                  ? records.length
                  : 0;
            } catch (e) {
              n = 0;
            }
            const t0 = now();
            try {
              return callback.call(this, records, observer);
            } finally {
              try {
                const dt = now() - t0;
                state.counters.moCallbacks += 1;
                state.counters.moRecords += n;
                record(state.timings.moCallback, dt);
                const b = bucketOf(obsState);
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
            const obsState = moStates ? moStates.get(this) : null;
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
                const f = init.attributeFilter;
                for (let i = 0; i < f.length; i++) {
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
          // apply(arguments): observe() with no options has its own spec behaviour.
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
        configurable: true,
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
    let best = null;
    for (let trial = 0; trial < 5; trial++) {
      const a = now();
      let b = a;
      let guard = 0;
      while (b === a && guard < 2000000) {
        b = now();
        guard++;
      }
      if (b !== a) {
        const d = b - a;
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
      breakdownCapped: state.breakdownCapped,
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
    const n =
      typeof sampleCount === "number" && sampleCount > 0
        ? Math.floor(sampleCount)
        : 200;

    if (!state.attempted.scrollHeight || !originals.scrollHeightDesc) {
      return {
        attempted: false,
        reason:
          "scrollHeight getter was not patched, so there is no overhead to measure",
        samples: 0,
        hasHiResClock: hasHiResClock,
      };
    }
    if (!doc || typeof doc.createElement !== "function") {
      return {
        attempted: false,
        reason: "no document to build a probe element in",
        samples: 0,
        hasHiResClock: hasHiResClock,
      };
    }

    const origGet = originals.scrollHeightDesc.get;
    let result;
    const saved = copyRawState();
    try {
      const probe = doc.createElement("div");
      probe.className = "sb-selfcost-probe";
      let i;
      let sink = 0;

      for (i = 0; i < n; i++) {
        sink += probe.scrollHeight;
      }
      for (i = 0; i < n; i++) {
        sink += origGet.call(probe);
      }

      const w0 = now();
      for (i = 0; i < n; i++) {
        sink += probe.scrollHeight;
      }
      const wrappedMs = now() - w0;

      const r0 = now();
      for (i = 0; i < n; i++) {
        sink += origGet.call(probe);
      }
      const rawMs = now() - r0;

      state.sink = sink;

      let granularity = state.clockGranularityMs;
      if (granularity === null || granularity === undefined) {
        granularity = measureClockGranularityMs();
      }

      const perWrapped = wrappedMs / n;
      const perRaw = rawMs / n;
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
          "instrument absent.",
      };
      state.clockGranularityMs = granularity;
    } catch (e) {
      result = {
        attempted: false,
        reason: "probe threw: " + e,
        samples: 0,
        hasHiResClock: hasHiResClock,
      };
    } finally {
      try {
        restoreRawState(saved);
      } catch (e2) {
        /* Leaving inflated counters would be worse than an empty catch; there is nothing further to try. */
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
    let out;
    let k;
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
    let out;
    try {
      out = {
        enabled: !!state.active,
        installedAt: api.installedAt,
        resetAt: api.resetAt,
        snapshotAt: nowStamp(),
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
        selfCost: copyPlain(state.selfCost),
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
    state.active = false;
    try {
      if (originals.scrollHeightDesc && ElementProto) {
        Object.defineProperty(
          ElementProto,
          "scrollHeight",
          originals.scrollHeightDesc,
        );
      }
    } catch (e) {
      /* leave it wrapped rather than throw */
    }
    try {
      if (originals.scrollTopDesc && ElementProto) {
        Object.defineProperty(
          ElementProto,
          "scrollTop",
          originals.scrollTopDesc,
        );
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
          originals.setPropertyDesc,
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

  // eslint-disable-next-line no-var -- hoisted functions above reference this object after initialization.
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
    selfCostEstimate: selfCostEstimate,
  };

  W.__sbLayoutCost = api;

  if (W.__sbLayoutCostDisabled === true) {
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
