// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/* Potency probe for `content-visibility: auto` on message roots. Installed via
 * `SBENCH_EXTRA_INIT_SCRIPT`, reporting via `SBENCH_PAGE_CONSOLE` (CSP blocks beacons); needs
 * `--password`/`--password-b`. A probe run forces layout and its payload is never scored. See the
 * studiobench README for the invocation.
 *
 * Answers only whether the browser really skipped rendering off-screen subtrees: `cvAuto`
 * (cascade only), `skipEvents` (engine-fired, so non-zero is proof) and `offUnrendered` (does
 * not work; see `descendantBoxes`). It also reports the contain-intrinsic-size sizing trap. */
(function () {
  "use strict";

  const PREFIX = "CVPOT ";
  const MESSAGE_SELECTOR = "[data-message-id]";
  const VIEWPORT_SELECTOR = ".aui-thread-viewport";
  const SAMPLE_MS = 2000;
  const MAX_ROOTS_SCANNED = 60;
  const MAX_DESCENDANTS_PER_ROOT = 40;
  /* A skipped root lands on either its declared fallback (fine) or its padding alone (a remembered
   * size of zero, the trap). They collide on this app, so each is matched against its own target,
   * fallback first. */
  const ROLE_PX = {
    assistant: { fallback: 300, padding: 18 },
    user: { fallback: 60, padding: 40 },
  };
  /* getBoundingClientRect() is the border box, so both targets carry the root's padding. */
  function targetHeight(px, which) {
    return (which === "fallback" ? px.fallback : 0) + px.padding;
  }
  /* Sub-pixel layout defeats exact equality; too small for either test to reach the other target. */
  const PX_EPS = 2;

  if (typeof window === "undefined" || !window.document) {
    return;
  }
  if (window.__cvPotInstalled) {
    return;
  }
  window.__cvPotInstalled = true;

  const doc = window.document;
  let watched = [];
  const watchedSet = typeof WeakSet === "function" ? new WeakSet() : null;
  /* Weak so a discarded root is not kept alive. */
  const recOf = typeof WeakMap === "function" ? new WeakMap() : null;
  const ev = {
    stateChange: 0,
    skip: 0,
    unskip: 0,
    watchers: 0,
    listenerErrors: 0,
  };
  let seq = 0;

  function computed(el, prop) {
    try {
      const s = window.getComputedStyle(el);
      return s ? String(s.getPropertyValue(prop) || "").trim() : "";
    } catch (e) {
      return "";
    }
  }

  function all(selector, scope) {
    try {
      const root = scope || doc;
      const list = root.querySelectorAll(selector);
      const out = [];
      for (let i = 0; i < list.length; i++) {
        out.push(list[i]);
      }
      return out;
    } catch (e) {
      return [];
    }
  }

  function rect(el) {
    try {
      return el.getBoundingClientRect();
    } catch (e) {
      return null;
    }
  }

  function intersects(a, b) {
    if (!a || !b) {
      return false;
    }
    return (
      a.bottom > b.top &&
      a.top < b.bottom &&
      a.right > b.left &&
      a.left < b.right
    );
  }

  /* Attached before anything is read, once per element: the one signal CSS cannot fake. */
  function watch(el) {
    try {
      if (watchedSet) {
        if (watchedSet.has(el)) {
          return;
        }
        watchedSet.add(el);
      } else if (el.__cvPotWatched) {
        return;
      } else {
        el.__cvPotWatched = true;
      }
      const rec = { el: el, skipped: null, events: 0 };
      el.addEventListener("contentvisibilityautostatechange", function (e) {
        ev.stateChange += 1;
        rec.events += 1;
        if (e && e.skipped) {
          ev.skip += 1;
          rec.skipped = true;
        } else {
          ev.unskip += 1;
          rec.skipped = false;
        }
      });
      watched.push(rec);
      if (recOf) {
        recOf.set(el, rec);
      } else {
        el.__cvPotRec = rec;
      }
      ev.watchers += 1;
    } catch (e) {
      ev.listenerErrors += 1;
    }
  }

  /* A KNOWN FALSE NEGATIVE, KEPT ON PURPOSE: `getClientRects()` inside a locked subtree makes
   * Chromium render it to answer. Use `ev_skip`. */
  function descendantBoxes(el, cap) {
    let kids;
    try {
      kids = el.querySelectorAll("*");
    } catch (e) {
      return -1;
    }
    const limit = Math.min(kids.length, cap);
    let n = 0;
    for (let i = 0; i < limit; i++) {
      try {
        if (kids[i].getClientRects().length > 0) {
          n += 1;
        }
      } catch (e2) {
        /* an element that cannot be measured is counted as having no box */
      }
    }
    return n;
  }

  /* `null` means no transition seen, which is not "rendered". */
  function skippedState(el) {
    let rec = null;
    try {
      rec = recOf ? recOf.get(el) : el.__cvPotRec;
    } catch (e) {
      rec = null;
    }
    return rec ? rec.skipped : null;
  }

  function roleOf(el) {
    try {
      return String(el.getAttribute("data-role") || "");
    } catch (e) {
      return "";
    }
  }

  function percentile(sorted, q) {
    if (sorted.length === 0) {
      return 0;
    }
    const i = Math.min(
      sorted.length - 1,
      Math.max(0, Math.round((sorted.length - 1) * q)),
    );
    return sorted[i];
  }

  function sample() {
    seq += 1;
    const roots = all(MESSAGE_SELECTOR);
    let i;
    for (i = 0; i < roots.length; i++) {
      watch(roots[i]);
    }

    let vp = null;
    try {
      vp = doc.querySelector(VIEWPORT_SELECTOR);
    } catch (e) {
      vp = null;
    }
    const vpRect = vp ? rect(vp) : null;

    const out = {
      seq: seq,
      origin: String(window.location.origin || ""),
      href_thread: String(
        window.location.hash || window.location.pathname || "",
      ),
      messages: roots.length,
      cvAuto: 0,
      cvVisible: 0,
      armedOffscreen: 0,
      offUnrendered: 0,
      offRendered: 0,
      armedOnscreen: 0,
      onRendered: 0,
      onUnrendered: 0,
      skippedNow: 0,
      fallbackBite: 0,
      paddingOnly: 0,
      droppedDetached: 0,
      scanned: 0,
      codeBlocks: 0,
      codeBlocksAuto: 0,
    };

    const heights = [];
    let scanned = 0;
    for (i = 0; i < roots.length; i++) {
      const el = roots[i];
      const cv = computed(el, "content-visibility");
      if (cv === "auto") {
        out.cvAuto += 1;
      } else if (cv === "visible") {
        out.cvVisible += 1;
      }
      if (out.cis === undefined) {
        out.cis = computed(el, "contain-intrinsic-size");
        out.cis_role = roleOf(el);
      }
      const r = rect(el);
      if (r) {
        heights.push(Math.round(r.height));
        /* Exclusive buckets; ties go to fallback, so a root as declared is not the trap. */
        /* Only while skipped: size containment applies only then. */
        const px = ROLE_PX[roleOf(el)];
        if (cv === "auto" && px && skippedState(el) === true) {
          if (Math.abs(r.height - targetHeight(px, "fallback")) <= PX_EPS) {
            out.fallbackBite += 1;
          } else if (
            Math.abs(r.height - targetHeight(px, "padding")) <= PX_EPS
          ) {
            out.paddingOnly += 1;
          }
        }
      }
      /* Forces layout, so capped: the probe must not become the load. */
      if (cv === "auto" && r && vpRect && scanned < MAX_ROOTS_SCANNED) {
        scanned += 1;
        const hasOwnBox = r.width > 0 || r.height > 0;
        const boxes = descendantBoxes(el, MAX_DESCENDANTS_PER_ROOT);
        const off = !intersects(r, vpRect);
        if (off) {
          out.armedOffscreen += 1;
          if (hasOwnBox && boxes === 0) {
            out.offUnrendered += 1;
          } else if (boxes > 0) {
            out.offRendered += 1;
          }
        } else {
          out.armedOnscreen += 1;
          if (boxes > 0) {
            out.onRendered += 1;
          } else {
            out.onUnrendered += 1;
          }
        }
      }
    }
    out.scanned = scanned;

    const blocks = all('[data-streamdown="code-block"]');
    out.codeBlocks = blocks.length;
    for (i = 0; i < blocks.length; i++) {
      if (computed(blocks[i], "content-visibility") === "auto") {
        out.codeBlocksAuto += 1;
      }
    }

    /* Connected roots only: `thread_reopen` detaches old roots, whose last `skipped` would stick. */
    const live = [];
    for (i = 0; i < watched.length; i++) {
      const wel = watched[i].el;
      let connected = false;
      try {
        connected = wel.isConnected !== false && doc.contains(wel);
      } catch (e) {
        connected = false;
      }
      if (!connected) {
        out.droppedDetached += 1;
        continue;
      }
      live.push(watched[i]);
      if (watched[i].skipped === true) {
        out.skippedNow += 1;
      }
    }
    watched = live;

    heights.sort(function (a, b) {
      return a - b;
    });
    out.h_min = heights.length ? heights[0] : 0;
    out.h_p50 = percentile(heights, 0.5);
    out.h_max = heights.length ? heights[heights.length - 1] : 0;
    out.h_list = heights.slice(0, 24);
    out.h_sum = 0;
    for (i = 0; i < heights.length; i++) {
      out.h_sum += heights[i];
    }

    if (vp) {
      try {
        out.vp_scrollHeight = vp.scrollHeight;
        out.vp_clientHeight = vp.clientHeight;
        out.vp_scrollTop = Math.round(vp.scrollTop);
      } catch (e) {
        /* leave the keys absent rather than reporting a zero that looks like a collapse */
      }
    }

    out.ev_stateChange = ev.stateChange;
    out.ev_skip = ev.skip;
    out.ev_unskip = ev.unskip;
    out.ev_watchers = ev.watchers;
    out.ev_listenerErrors = ev.listenerErrors;

    try {
      window.console.log(PREFIX + JSON.stringify(out));
    } catch (e) {
      /* nothing to do: the console is the only way out */
    }
  }

  window.__cvPotSample = sample;

  /* The listener must exist before the element's first transition: an off-screen root fires its
   * only event once. So roots are adopted at insertion by a MutationObserver on the Document (not
   * `documentElement`, which may not exist yet); the interval catches later-acquired properties. */
  function adoptAll() {
    const roots = all(MESSAGE_SELECTOR);
    for (let i = 0; i < roots.length; i++) {
      watch(roots[i]);
    }
  }

  /* Added nodes only: a document-wide re-scan per mutation would make the probe the load. */
  function adoptAdded(records) {
    for (let i = 0; i < records.length; i++) {
      const added = records[i].addedNodes;
      for (let j = 0; j < (added ? added.length : 0); j++) {
        const node = added[j];
        if (!node || node.nodeType !== 1) {
          continue;
        }
        try {
          if (
            typeof node.matches === "function" &&
            node.matches(MESSAGE_SELECTOR)
          ) {
            watch(node);
          }
        } catch (e) {
          ev.listenerErrors += 1;
        }
        const inner = all(MESSAGE_SELECTOR, node);
        for (let k = 0; k < inner.length; k++) {
          watch(inner[k]);
        }
      }
    }
  }

  try {
    if (typeof window.MutationObserver === "function") {
      new window.MutationObserver(adoptAdded).observe(doc, {
        childList: true,
        subtree: true,
      });
    } else {
      ev.listenerErrors += 1;
    }
  } catch (e) {
    ev.listenerErrors += 1;
  }

  adoptAll();
  try {
    window.setInterval(sample, SAMPLE_MS);
  } catch (e) {
    /* a probe that cannot schedule itself reports nothing, which is the honest failure */
  }
})();
