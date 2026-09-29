// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
// The glass: instrumented accessors that turn "the autoscroll observer forces a synchronous layout
// on every streamed character" from a reading of the source into a counted, timed number.
// The direct instrument for M3. `use-intent-aware-autoscroll.tsx` installs a MutationObserver with
// `subtree: true, characterData: true` over the whole viewport; its callback synchronously reads
// `scrollHeight` (:435), writes the inherited custom property `--aui-scroll-stabilizer` on the
// scroll container (:466), which invalidates style for every descendant, and calls `scrollTo`
// (:487). All three are invisible to a React Profiler, to markdown timing and to a node census.
// PERTURBING BY CONSTRUCTION, and so off at level 0: wrapping a hot getter on Element.prototype
// costs something on every call site in the app, and the wrapper's own `performance.now()` pair is
// a large fraction of a cheap read. Level 1 and above; the headline numbers come from level 0.
// Split by WHOSE read it is. "Layout was forced 4,000 times" is not attribution; "3,980 of them
// were on the scroll viewport, from inside a MutationObserver callback" is.

(() => {
  if (window.__sb && window.__sb.glass) return;
  window.__sb = window.__sb || {};

  const I = {
    scroll_height_reads: 0,
    scroll_height_ms: 0,
    viewport_reads: 0,
    viewport_ms: 0,
    // Reads taken while a MutationObserver callback is on the stack: the number that separates "the app
    // reads scrollHeight" from "the app reads scrollHeight per mutation".
    reads_in_observer: 0,
    reads_in_observer_ms: 0,
    scroll_top_writes: 0,
    viewport_scroll_top_writes: 0,
    scroll_to_calls: 0,
    // The stabilizer property write at :466. Setting an inherited custom property on the scroll
    // container invalidates style for every descendant, so its COUNT is the multiplier on a
    // whole-thread style recalc.
    stabilizer_writes: 0,
    custom_property_writes: 0,
    mutation_callbacks: 0,
    mutation_records: 0,
    character_data_records: 0,
    observers_created: 0,
    subtree_observers: 0,
  };
  let inObserver = 0;

  const isViewport = (el) => {
    try {
      return Boolean(el && el.classList && el.classList.contains("aui-thread-viewport"));
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
        I.scroll_height_reads += 1;
        I.scroll_height_ms += dt;
        if (isViewport(this)) {
          I.viewport_reads += 1;
          I.viewport_ms += dt;
        }
        if (inObserver > 0) {
          I.reads_in_observer += 1;
          I.reads_in_observer_ms += dt;
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
        I.scroll_top_writes += 1;
        if (isViewport(this)) I.viewport_scroll_top_writes += 1;
        st.set.call(this, value);
      },
    });
  }

  const nativeScrollTo = Element.prototype.scrollTo;
  if (nativeScrollTo) {
    Element.prototype.scrollTo = function (...args) {
      I.scroll_to_calls += 1;
      return nativeScrollTo.apply(this, args);
    };
  }

  // The custom-property write at use-intent-aware-autoscroll.tsx:466.
  const nativeSetProperty = CSSStyleDeclaration.prototype.setProperty;
  if (nativeSetProperty) {
    CSSStyleDeclaration.prototype.setProperty = function (name, value, priority) {
      if (typeof name === "string" && name.charCodeAt(0) === 45 /* '-' */) {
        I.custom_property_writes += 1;
        if (name.indexOf("aui-scroll-stabilizer") >= 0) I.stabilizer_writes += 1;
      }
      return nativeSetProperty.call(this, name, value, priority);
    };
  }

  const NativeMO = window.MutationObserver;
  if (NativeMO) {
    const Wrapped = function (callback) {
      I.observers_created += 1;
      const observer = new NativeMO((records, obs) => {
        I.mutation_callbacks += 1;
        I.mutation_records += records.length;
        for (const r of records) if (r.type === "characterData") I.character_data_records += 1;
        inObserver += 1;
        try {
          return callback(records, obs);
        } finally {
          inObserver -= 1;
        }
      });
      const nativeObserve = observer.observe.bind(observer);
      observer.observe = (target, options) => {
        if (options && options.subtree) I.subtree_observers += 1;
        return nativeObserve(target, options);
      };
      return observer;
    };
    Wrapped.prototype = NativeMO.prototype;
    window.MutationObserver = Wrapped;
  }

  window.__sb.glass = {
    read() {
      const out = {};
      for (const k of Object.keys(I)) {
        out[k] = k.endsWith("_ms") ? Math.round(I[k] * 100) / 100 : I[k];
        I[k] = 0;
      }
      out.glass_attempted = true;
      return out;
    },
  };
})();
