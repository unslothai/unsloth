// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/*
 * layoutcost.js -- counts the DOM operations suspected of forcing synchronous layout
 * during streaming, and times them.
 *
 * WHAT IT COUNTS AND WHY THESE FIVE
 *
 * The mechanism under investigation is
 * studio/frontend/src/components/assistant-ui/use-intent-aware-autoscroll.tsx. It installs a
 * MutationObserver on the thread viewport (`.aui-thread-viewport.aui-stream-viewport`) with
 * `{childList, subtree, characterData, attributes, attributeFilter:["class","hidden",
 * "aria-hidden","aria-expanded","data-state"]}`. Every delivery of that observer runs
 * `el.scrollHeight` (a read that forces style plus layout when the DOM is dirty, and the DOM is
 * always dirty here because the mutation is what woke the observer), then
 * `el.style.setProperty("--aui-scroll-stabilizer", "<n>px")`, then
 * `el.scrollTo({top, behavior:"instant"})`. During token streaming the characterData mutations
 * arrive per streamed character, and each one costs something proportional to the size of the
 * whole thread, not to the size of the change. That is the shape of the hypothesis, so the
 * instrument records exactly the five operations that shape is made of:
 *
 *     scrollHeight READS   getter on Element.prototype. The forced-layout trigger.
 *     scrollTop WRITES     setter on Element.prototype. Cheap to issue, expensive later.
 *     scrollTo CALLS       method on Element.prototype. The pin.
 *     MutationObserver     callback invocations, and the number of MutationRecords handed to
 *                          each one. Records per callback is the number that separates "the
 *                          observer fires once with 400 records" from "it fires 400 times",
 *                          which are wildly different costs and look identical in a profile
 *                          that only counts callbacks.
 *     setProperty("--..")  custom property writes, with `--aui-scroll-stabilizer` counted on
 *                          its own so the autoscroll path is separable from every other CSS
 *                          variable the app writes.
 *
 * Counts alone would be enough to test the "per character" claim. The timings are here because
 * a count cannot distinguish 4,000 cheap reads from 4,000 reads that each walk a 300-message
 * thread, and the whole question is which of those is happening.
 *
 * THE SELF-COST PROBLEM, STATED PLAINLY
 *
 * Wrapping the `scrollHeight` getter to time it makes `scrollHeight` slower. This instrument
 * changes the thing it measures. There is no version of this technique that does not, so the
 * honest move is to measure the distortion instead of hoping it is small:
 *
 *   1. `selfCostEstimate()` times N wrapped reads and N reads through the ORIGINAL descriptor,
 *      which this file keeps a reference to for exactly this purpose, both on a detached and
 *      clean element so no real layout work is included. The difference is the per-call
 *      overhead of the wrapper as installed, including its own bookkeeping. The deep tier can
 *      subtract `overheadMsPerCall * scrollHeightReads` from the total, or at minimum print it
 *      next to the total so a reader can see whether the instrument is a rounding error or a
 *      third of the number.
 *   2. The Python driver runs the SAME benchmark cell twice, once with this instrument injected
 *      and once without, and compares the frame statistics of the two. That is the real check.
 *      Step 1 measures what the wrapper costs per call in isolation; step 2 measures what the
 *      whole instrument costs the app in situ, including cache effects and lost inlining that
 *      no microbenchmark can see. If the two runs disagree about the app, the counts from this
 *      file are still usable and its timings are not, and that has to be discovered rather than
 *      assumed.
 *
 * `clockGranularityMs`, also reported by `selfCostEstimate()`, exists because a single
 * `scrollHeight` read on a clean DOM can be faster than `performance.now()` can resolve. In a
 * browser without cross-origin isolation the timer is clamped, so a genuine 0.003 ms read is
 * indistinguishable from zero. A `maxMs` of 0 therefore means "below the clock", not "free",
 * and the granularity number is what lets the report say so.
 *
 * WHY IT IS OFF BY DEFAULT
 *
 * See above: it perturbs the measurement. The default benchmark run must be as close to the
 * shipping app as the harness can manage, so the driver injects this file only for the deep
 * tier, where the question has already narrowed to "which operation" and a known, quantified
 * distortion is worth paying for. Injection is the opt-in. Setting
 * `window.__sbLayoutCostDisabled = true` before this script runs is a secondary escape hatch
 * for bisecting the instrument itself.
 *
 * WHAT IT DOES NOT MEASURE
 *
 * It does not measure layout time. It measures how often the app asks for something that can
 * force layout, and how long the asking took from JS. Actual layout attribution comes from the
 * trace, and the point of this file is to tell the trace where to look.
 *
 * Reads through `window.scrollY`, `document.documentElement.scrollHeight` in the getter's own
 * frame, `getBoundingClientRect`, `offsetHeight` and `getComputedStyle` are NOT wrapped. They
 * force layout too. They are not in the code path under investigation, and every wrapper added
 * here makes the run less like the app.
 *
 * ZERO DISCIPLINE
 *
 * A count of 0 from this file is a real observation: the operation did not happen. A count of 0
 * because the patch could not be installed is not, and the two must never print the same. So
 * `snapshot()` carries `attempted` per instrumented family, `unavailable` lists the names whose
 * original descriptor was missing or non-configurable, and a WebKit build that refuses the
 * patch reads as "not attempted" rather than as a quiet zero. The Python side turns those into
 * Measure objects. Failing to install is not an error and never throws: a partial instrument
 * that is honest about which half ran is more useful than a page that fails to boot.
 */

(function () {
  "use strict";

  if (typeof window === "undefined" || !window) {
    return;
  }

  var W = window;

  // Idempotence. add_init_script runs per document, and a page that creates an iframe or that
  // the driver reloads must not stack wrappers: a doubly wrapped getter would double every
  // count and square nothing useful.
  if (W.__sbLayoutCostInstalled) {
    return;
  }
  W.__sbLayoutCostInstalled = true;

  var VIEWPORT_CLASS = "aui-stream-viewport";
  var STABILIZER_PROP = "--aui-scroll-stabilizer";
  var MAX_KEYS = 24;
  var OTHER_KEY = "__other__";

  // Capture these before the app can touch them. An app that replaces performance.now (some
  // test doubles do) would otherwise silently redefine every timing in this file.
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

  // ---------------------------------------------------------------------------------------
  // State
  // ---------------------------------------------------------------------------------------

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
    // Sink for the self-cost loops. A read whose result is thrown away can in principle be
    // optimised out; assigning it somewhere observable keeps both loops honest.
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

  // ---------------------------------------------------------------------------------------
  // Per-target breakdown
  //
  // Keyed by a short, stable descriptor rather than by element identity. Holding elements in a
  // Map would pin detached DOM alive for the whole run, which on a streaming thread means
  // holding every message ever rendered. The key is deliberately coarse: the question is "which
  // KIND of element is being read", and a 300-message thread has maybe six kinds.
  //
  // The cap matters. A page that reads scrollHeight on thousands of distinct elements would
  // otherwise grow an unbounded object inside the hot path, so key 25 and beyond all land in
  // one bucket and `breakdownCapped` says the truncation happened.
  // ---------------------------------------------------------------------------------------

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
          // contains(), not matches(): a class test cannot invoke the selector engine and
          // cannot be tricked into anything expensive, and this runs inside the getter.
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
      // The overflow bucket is not charged to the budget, so `breakdownKeyCount` is the number
      // of REAL keys and the object holds at most MAX_KEYS + 1 entries. Counting the bucket
      // would make a capped run report 25 keys and a reader would reasonably conclude the cap
      // does not work.
      state.breakdownKeyCount += 1;
    }
    return b;
  }

  // ---------------------------------------------------------------------------------------
  // Element.prototype patches
  // ---------------------------------------------------------------------------------------

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
          // Outside try/catch on purpose. If the native getter throws, the app must see the
          // same exception it would have seen without this file.
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
        // The getter is left exactly as it was. Reading scrollTop also forces layout, but it is
        // not the operation under investigation and wrapping it would add cost to a path the
        // hypothesis does not name.
        get: d.get,
        set: function (v) {
          if (!state.active) {
            origSet.call(this, v);
            return;
          }
          // Writes are counted, not timed. Issuing a scroll write is cheap; the cost lands in
          // the next layout, where a timer wrapped around the setter would not see it. A number
          // that looks like a duration but is not one is worse than no number.
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

  // ---------------------------------------------------------------------------------------
  // CSSStyleDeclaration.prototype.setProperty
  // ---------------------------------------------------------------------------------------

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
              // Only custom properties. Ordinary style writes are far more numerous and are not
              // part of the mechanism being tested; counting them would bury the signal.
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

  // ---------------------------------------------------------------------------------------
  // MutationObserver
  //
  // A subclass rather than a Proxy or a plain function: `instanceof MutationObserver` keeps
  // working, `observe` / `disconnect` / `takeRecords` stay native (they are inherited, and
  // `observe` is overridden only to read its arguments before delegating), and the callback is
  // the single thing that changes.
  //
  // The split into viewportObserver and other exists because the app runs several observers and
  // an aggregate count cannot tell the autoscroll one from React's own. Two independent
  // discriminators are recorded per observer: the observe() target carrying the
  // `.aui-stream-viewport` class, and an attributeFilter containing "aria-expanded", which no
  // other observer in this app requests. An observer counts as the viewport observer if EITHER
  // fires, because the classes are renamed more often than the filter is, and both booleans are
  // reported separately so a reader can see when they disagree instead of trusting the merge.
  // ---------------------------------------------------------------------------------------

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
            // Let the native constructor produce its own TypeError, unchanged.
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
              // finally, not a trailing statement: a callback that throws still consumed the
              // time, and dropping the sample would make a broken build look fast.
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
          // apply(arguments), not (target, init): observe() with a missing options argument has
          // its own spec behaviour and must keep it.
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

  // ---------------------------------------------------------------------------------------
  // Self cost
  // ---------------------------------------------------------------------------------------

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
    // The live objects were replaced, so the convenience aliases on the public API have to be
    // rebound or a caller reading __sbLayoutCost.counters directly would watch a detached copy
    // that never increments again.
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
    // The probe runs through the live wrapper, so it increments the real counters. Saving and
    // restoring them keeps the estimate from contaminating the measurement it exists to
    // correct, while still timing the wrapper exactly as the app sees it, bookkeeping included.
    var saved = copyRawState();
    try {
      var probe = doc.createElement("div");
      probe.className = "sb-selfcost-probe";
      // Detached and never inserted. An attached element would make both loops pay for real
      // layout, which is the app's cost and not the instrument's, and would swamp the
      // difference this function is trying to resolve. The number produced here is the cost of
      // the WRAPPER, which is the only part this file is responsible for.
      var i;
      var sink = 0;

      // Warm both paths first. The first few hundred calls run in the interpreter tier, and
      // whichever loop goes first would otherwise be charged for the JIT.
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

  // ---------------------------------------------------------------------------------------
  // Public surface
  // ---------------------------------------------------------------------------------------

  function copyPlain(v) {
    // Hand written rather than JSON round tripping: JSON turns a non finite number into null
    // and silently drops undefined, and a timing that reads null because of the serialiser
    // would be indistinguishable from a timing that was never taken.
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
        // Per family, so a count of zero can be told apart from a patch that never landed.
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
    // Restores the exact descriptors that were captured, so a page can be handed back to a
    // measurement that must not carry the instrument's overhead. Counters are left alone: the
    // driver usually snapshots after uninstalling.
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
    // Injected but told to stand down. Everything reads as not attempted, which is the correct
    // answer and not a row of zeros.
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
