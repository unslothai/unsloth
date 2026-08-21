// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
//
// UI PARITY: a structural signature of the rendered thread, taken at the close of every action
// window on BOTH arms of an A/B.
//
// WHY THIS AND NOT SCREENSHOTS. The screenshot parity pair (scripts PR 217) is the right artefact
// for a human reader of a pull request: it shows the surface, side by side, and a person checks it
// against `expect`. What it cannot do is breadth. Each pair costs two isolated installs, shows one
// surface, and needs somebody to open the composite. Eighteen PRs across eighteen action surfaces
// is 324 composites and nobody will look at them.
//
// This runs inside a comparison the tool is ALREADY doing. studiobench drives both arms through
// the same eighteen scripted actions against a byte-identical seeded thread inside one session, so
// the two DOMs are already meant to agree. Digesting them costs one `evaluate` per slot and turns
// "the UI is unchanged" from a claim into a per-action assertion. It is the cheap wide net; the
// screenshot pair stays the deep, human-readable one for the surfaces that matter most.
//
// ── WHAT THIS DIGEST CAN AND CANNOT SEE ──────────────────────────────────────────────────────
//
// Read this before quoting a passing parity check at anybody. The digest is a serialisation of the
// DOM, so its sensitivity is exactly the DOM and nothing else.
//
// IT SEES: tag structure and nesting, element order, every attribute name, every attribute value
// that is not on the volatile list below (so `class`, `data-slot`, `data-state`, `data-role`,
// `aria-hidden`, `hidden`, `disabled`, `role`, `title`, `alt`, `placeholder`), and all text content
// after time normalisation. Added elements, removed elements, reordered siblings, retitled buttons
// and flipped states all move the digest.
//
// IT DOES NOT SEE, and a green parity check asserts NOTHING about any of these:
//   - CSS coming from a stylesheet. A class name is compared as a STRING. If `.aui-md-p` changes
//     its margin in a `.css` file, every class list is identical and the digest is identical while
//     the page reflows. Only the bounded computed-style probe below sees any of this, and it sees
//     three properties on at most a few dozen elements.
//   - Computed layout: geometry, sizes, positions, scroll extents, whether anything overflows or
//     is clipped, whether two elements overlap.
//   - Colour, typography, borders, shadows, opacity, transforms, animation.
//   - Raster content: images, canvas, video, SVG paint. An `<img>` is compared by its attributes,
//     and its `src` is normalised (see below), so a DIFFERENT PICTURE at the same normalised URL
//     is invisible.
//   - Anything outside the thread root and the overlay selectors: the sidebar, the header, toasts.
//   - Anything the DOM does not carry at all: focus ring appearance, caret position, text
//     selection, scroll position (deliberately normalised), `<input>` live values typed by a user
//     (the DOM attribute is not the property), shadow DOM, and any state kept only in JS.
//
// THAT LIST HAS NOW BEEN MEASURED RATHER THAN JUST ASSERTED, and the result is worse than the
// wording above suggests. A concurrent campaign measuring the sidebar drag ran this digest against
// a real, visible change and got 0 of 34 differing pairs -- and its null control also returned 0 of
// 34, so the instrument was not discriminating in either direction on that change. Three
// purpose-built captures (sidebar-inclusive structure, sidebar inline style, custom-property reach)
// each found the same change 34 of 34 with the null at zero. The two blind spots that mattered were
// the two named above: the sidebar is outside the root, and computed layout is never read.
//
// So a PARITY OK verdict from this file means NO THREAD-STRUCTURE CHANGE WAS DETECTED. It does not
// mean the UI is unchanged, and the two are far enough apart that the second should never be
// written down on the strength of the first.
//
// Extending this into a real visual diff is the screenshot pair's job, not this one's. The point
// of the digest is breadth at near-zero cost per action, and its limits are stated here so nobody
// reads "18 actions, 0 differences" as "the UI is pixel-identical". It is not that claim.
//
// ── AND A NOTE ON ANY SCAN THAT CAN RETURN ZERO ──────────────────────────────────────────────
//
// `styleProbe` below walks a hand-written selector list. If Studio renames a class the list goes
// quiet, matches nothing, and its digest becomes the hash of an empty string -- identical on both
// arms, reported as a MATCH. A scan of nothing must never be reported as agreement, so
// `compare_styles` refuses a zero-element probe instead of matching it, and `elements` travels
// with every reading so the count can be checked rather than assumed.
//
// The same team hit the general form of this and it is worth recording: their CSSOM scan returned
// a clean zero because CSS nesting gives every `CSSStyleRule` a truthy but empty `cssRules`, so
// code that treats a truthy `cssRules` as "this is a grouping rule, recurse" silently skips every
// declaration in the document. They caught it only because they had gated on a positive control.
// Nothing in this file walks the CSSOM today, so that specific bug is not present here -- but any
// scan added later that can legitimately return zero needs a positive control, and a zero without
// one should not be believed.
//
// ── WHAT IS NORMALISED AWAY, and why each one has to be ──────────────────────────────────────
//
// A digest that trips on things that legally differ between two runs of the SAME build is worse
// than no digest: it trains you to ignore it. Every entry here was OBSERVED to differ in a
// base-vs-base null control; nothing is normalised on suspicion.
//
//   generated ids       Radix mints `radix-:r1a:` per mount, and aria-controls/labelledby point at
//                       them. They differ between two runs of one build.
//   rendered durations  the action bar prints things like `295ms`. PR 217 hit exactly this on
//                       unslothai/unsloth#9054: a 295 vs 310 difference that is wall clock, not
//                       content. Any number immediately followed by a time unit is collapsed.
//   relative times      "2 minutes ago" moves on its own.
//   scroll state        `--aui-scroll-stabilizer` and transform offsets encode where the thread
//                       happens to be scrolled, which the film varies deliberately.
//   record identifiers  thread ids, message ids and attachment ids are minted by the BACKEND when
//                       the fixture seeds the thread. The two arms of an A/B are two separate
//                       Studio installs with two separate databases, so these can never agree and
//                       carry no information about the frontend.
//   absolute URLs       `src`/`href` carry the arm's own origin, and the two arms are on two
//                       ports by construction. `http://127.0.0.1:5830/x` and `...:5831/x` are the
//                       same asset. Origins are stripped; the PATH is kept and still compared.
//   blob and data URLs  an uploaded image is addressed by a per-mount `blob:` UUID.
//
// WHAT IS DELIBERATELY KEPT. Tag structure, `data-slot`, `data-state`, `data-role`, `aria-hidden`,
// `hidden`, `disabled`, the class list, and all text content. `data-state` in particular: whether
// a reasoning pane reads open or closed IS the UI, and a perf change that quietly flips it is the
// exact defect this is meant to catch.

(() => {
  if (window.__sb && window.__sb.parity) return;
  window.__sb = window.__sb || {};

  // Attributes whose VALUE is volatile between two runs of the same build. The attribute's
  // presence is still recorded; only the value is dropped.
  const VOLATILE_ATTRS = new Set([
    "id", "for", "aria-controls", "aria-labelledby", "aria-describedby", "aria-activedescendant",
    "style", "data-radix-scroll-area-viewport", "data-testid-instance",
    // Backend-minted record ids. Two arms are two installs with two databases.
    "data-thread-id", "data-message-id", "data-attachment-id", "data-part-id", "data-run-id",
    "data-tool-call-id", "data-checkpoint-id",
  ]);

  // Attributes that carry no rendered meaning at all.
  const IGNORED_ATTRS = new Set(["data-react-checksum", "data-reactroot"]);

  // Attributes whose value is a URL. The origin is stripped and the path kept, so a genuinely
  // different asset still moves the digest while the arm's own port does not.
  const URL_ATTRS = new Set(["src", "href", "srcset", "action", "poster", "data-src", "formaction"]);

  // A UUID, or a long hex run. Both are how this app spells "an id the server just made up".
  const ID_RE = /\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b|\b[0-9a-f]{16,}\b/gi;

  const normUrl = (s) =>
    (s || "")
      // blob:/data: URLs address a per-mount object. Nothing in them survives a reload.
      .replace(/^blob:[^\s]*$/i, "#BLOB")
      .replace(/^data:([^;,]*)[^\s]*$/i, "#DATA:$1")
      // Strip scheme://host:port, keep the path. `/assets/index-a1b2.js` still compares.
      .replace(/\b[a-z]+:\/\/[^/\s]+/gi, "")
      .replace(ID_RE, "#ID");

  const normText = (s) =>
    (s || "")
      // `295ms`, `1.2 s`, `3 min` -> a placeholder. Wall clock, not content.
      .replace(/\b\d+(\.\d+)?\s?(ms|s|sec|secs|second|seconds|min|mins|minute|minutes|hour|hours|day|days)\b/gi, "#T")
      // "2 minutes ago" style relatives that survived the above.
      .replace(/\b(just now|a few seconds ago|yesterday)\b/gi, "#T")
      // Absolute timestamps.
      .replace(/\b\d{1,2}:\d{2}(:\d{2})?\s?(am|pm)?\b/gi, "#T")
      // Backend-minted ids that reach the DOM as TEXT rather than as an attribute.
      .replace(ID_RE, "#ID")
      .replace(/\s+/g, " ")
      .trim();

  // FNV-1a, 32 bit, expressed as unsigned hex. Not cryptographic: this is a change detector, and
  // the pair being compared is two runs of one page rather than an adversary.
  const hash = (str) => {
    let h = 0x811c9dc5;
    for (let i = 0; i < str.length; i++) {
      h ^= str.charCodeAt(i);
      h = (h + ((h << 1) + (h << 4) + (h << 7) + (h << 8) + (h << 24))) >>> 0;
    }
    return h.toString(16).padStart(8, "0");
  };

  const attrValue = (el, name) => {
    const raw = el.getAttribute(name);
    if (URL_ATTRS.has(name)) return normUrl(raw);
    return normText(raw);
  };

  const signature = (root) => {
    if (!root) return "";
    const parts = [];
    const walk = (el, depth) => {
      // A depth cap is a TRUNCATION, and a truncated signature that reads like a complete one is
      // exactly the silent false negative this instrument exists to rule out. The marker is left
      // in the signature so anything deeper is visible as "not walked" rather than as "absent".
      if (depth > 40) { parts.push("<!depth-cap>"); return; }
      parts.push("<" + el.tagName.toLowerCase());
      const names = [];
      for (const attr of el.attributes) {
        if (IGNORED_ATTRS.has(attr.name)) continue;
        names.push(attr.name);
      }
      names.sort();
      for (const name of names) {
        if (VOLATILE_ATTRS.has(name)) {
          parts.push(" " + name + "=*");
        } else {
          parts.push(" " + name + "=" + attrValue(el, name));
        }
      }
      parts.push(">");
      for (const child of el.childNodes) {
        if (child.nodeType === 3) {
          const t = normText(child.nodeValue);
          if (t) parts.push(t);
        } else if (child.nodeType === 1) {
          walk(child, depth + 1);
        }
      }
      parts.push("</" + el.tagName.toLowerCase() + ">");
    };
    walk(root, 0);
    return parts.join("");
  };

  // The bounded computed-style probe. THREE properties on at most `STYLE_CAP` elements, and its
  // digest is reported SEPARATELY from the structural one rather than folded into it.
  //
  // Separate on purpose. `getComputedStyle` reads post-stylesheet, post-animation state, so it is
  // the only part of this file that can see a `.css` change at all -- and for the same reason it
  // is the part most likely to be caught mid-transition and differ between two runs of one build.
  // Folding it into `digest` would put the whole instrument's credibility on its least stable
  // reading. Kept apart, a structural mismatch stays a hard signal and a style mismatch is an
  // advisory the caller can weigh.
  //
  // `display` and `visibility` because they are how CSS makes something disappear without touching
  // the DOM. `pointer-events` because an overlay that stops or starts swallowing clicks is a real,
  // shipped class of defect here and leaves no structural trace at all.
  const STYLE_CAP = 64;
  const STYLE_PROPS = ["display", "visibility", "pointer-events"];
  const STYLE_SELECTORS = [
    '[data-role]', '[data-slot="reasoning-root"]', '[data-slot="tool-group-root"]',
    ".aui-thread-viewport", ".aui-composer-root", 'button[aria-label="Send message"]',
    'button[aria-label="Stop generating"]', '[data-slot="dialog-content"]', '[role="menu"]',
    "[data-radix-popper-content-wrapper]",
  ];

  const styleProbe = () => {
    const seen = new Set();
    const parts = [];
    let n = 0;
    let capped = false;
    for (const sel of STYLE_SELECTORS) {
      for (const el of document.querySelectorAll(sel)) {
        if (seen.has(el)) continue;
        seen.add(el);
        if (n >= STYLE_CAP) { capped = true; continue; }
        n += 1;
        const cs = window.getComputedStyle(el);
        parts.push(sel + "#" + n);
        for (const prop of STYLE_PROPS) parts.push(":" + prop + "=" + cs.getPropertyValue(prop));
        parts.push(";");
      }
    }
    const sig = parts.join("");
    // `capped` travels with the reading. A style digest taken over the first 64 of 300 elements
    // is a partial reading, and a partial reading that does not say so reads as a complete one.
    return { digest: hash(sig), chars: sig.length, elements: n, capped, props: STYLE_PROPS, sig };
  };

  const D = () => (window.__sb.dom || {});

  // ── VISIBLE-REGION PARITY ────────────────────────────────────────────────────────────────────
  //
  // THE POLICY THIS SERVES. All changes must preserve UI and UX idempotency, with two exemptions:
  // a difference may be accepted deliberately when performance improves dramatically, and a
  // difference that exists only OFF SCREEN is fine by definition, because rendering only what is
  // visible is an accepted technique rather than a parity violation.
  //
  // The whole-document digest above cannot express the second exemption. It compares everything
  // that is in the DOM, so ANY deferred off-screen work fails it by construction -- virtualization,
  // deferred syntax highlighting, content-visibility, lazy images. Refusing such a pair as
  // NOT_APPLICABLE was the right instinct and it withholds a verdict. This supplies the verdict.
  //
  // THE CLAIM IT SUPPORTS, exactly: every message that was visible in the viewport at any point
  // during the action is present on both arms and identical between them, and every difference
  // lies outside the viewport.
  //
  // ── TWO BOUNDARY DECISIONS, MADE EXPLICITLY ──────────────────────────────────────────────────
  //
  // These are where a visible-region check goes wrong quietly, so they are decided here in the
  // open rather than left implicit in whatever the code happens to do.
  //
  // 1. PARTIAL INTERSECTION COUNTS AS VISIBLE, AND THE ELEMENT IS DIGESTED IN FULL.
  //    A message one pixel into the viewport is visible to the user. The alternative -- digesting
  //    only the part inside the viewport -- is not definable on a DOM subtree without reading
  //    geometry per node, and reading geometry is the one thing this must not do (see below). So
  //    the whole element is compared. The error this admits is a FALSE ALARM: a difference in the
  //    off-screen tail of a partly-visible message is reported as a visible difference. The error
  //    it refuses to admit is a false pass. Given a parity gate, that is the right way round.
  //
  // 2. ANYTHING VISIBLE AT ANY POINT DURING THE ACTION IS COMPARED, not just at the end.
  //    An action that scrolls makes messages visible and then hides them again; a single sample at
  //    the close of the window would compare the DOM the scroll happened to land on and silently
  //    ignore everything the user actually saw on the way. So the observer is installed BEFORE the
  //    window opens and the compared set is the UNION of everything that ever intersected. The
  //    per-message digest is still the one taken at the close, which is a real limitation and is
  //    named as such in `now_visible` versus `ever_visible`: a message that was visible mid-action
  //    and has since changed is compared in its final state.
  //
  // ── WHY INTERSECTIONOBSERVER AND NOT GEOMETRY ────────────────────────────────────────────────
  //
  // `getBoundingClientRect()` / `getClientRects()` on content inside a `content-visibility` locked
  // subtree makes Chromium render that subtree in order to answer, so a geometry-based visibility
  // probe unlocks exactly what it came to observe and then reports that nothing was skipped. That
  // is measured, not theoretical: one session reported 0 off-screen unrendered roots while the
  // event counter recorded 22 simultaneously in the skipped state. See the content-visibility trap
  // section of CONTRIBUTING-perf.md.
  //
  // IntersectionObserver is the correct instrument and not merely a safe one: it is the same
  // mechanism Blink's own relevance machinery uses to decide whether a `content-visibility: auto`
  // subtree is skipped, so observing with it neither forces rendering nor perturbs the decision.
  // NOTHING in this section may call a geometry method on a candidate element.
  const VIS = {
    obs: null, mut: null, ever: new Set(), seen: new WeakSet(), watching: false, seen_count: 0,
  };

  const ordinalOf = (el, index) => {
    // The message's position in the THREAD, not in the mounted list. This is what makes a windowed
    // arm comparable with a fully mounted one at all: mounted index 0 is message 10 on one arm and
    // message 1 on the other, so a per-index comparison compares different messages and reports
    // every row as changed. `aria-posinset` is the windowed arm's own claim about position; a
    // fully mounted thread does not publish it and does not need to, because there the DOM order
    // IS the thread order.
    const owner = el.closest ? el.closest("[aria-posinset]") : null;
    if (owner) {
      const n = Number(owner.getAttribute("aria-posinset"));
      if (Number.isFinite(n)) return n;
    }
    return index + 1;
  };

  const observeOne = (el, fallbackIndex) => {
    if (!VIS.obs || VIS.seen.has(el)) return;
    VIS.seen.add(el);
    // The ordinal is stamped on the node, because by the time an entry is delivered the node may
    // have been unmounted and `closest()` would return nothing.
    el.__sbOrdinal = ordinalOf(el, fallbackIndex);
    VIS.obs.observe(el);
  };

  //: The FULL scan. O(the document), so it runs exactly once, when the observer is installed --
  //: which happens before the measured window opens.
  const observeAll = () => {
    if (!VIS.obs) return;
    const nodes = (D().messages && D().messages()) || [];
    for (let i = 0; i < nodes.length; i++) observeOne(nodes[i], i);
  };

  // THE TOP-UP, AND WHY IT IS NOT A RESCAN.
  //
  // A windowed list mounts rows as it scrolls, so rows appearing after the observer was installed
  // have to be picked up or a windowed arm is only ever asked about what it happened to have
  // mounted at the start. The obvious way to do that is to re-run the full scan from a
  // MutationObserver -- and the MutationObserver runs DURING the measured action window, so a
  // full `querySelectorAll` per mutation batch would charge an O(document) walk to the action,
  // once per batch, on a 64,000-element DOM, growing with exactly the quantity under
  // investigation. That is workspace task #102 verbatim: the census and the parity digest running
  // inside the window, which reported delete_message at 14.3 fps when it costs 49.0.
  //
  // So this walks only what was ADDED. `addedNodes` is small by construction -- a virtualizer
  // mounts one or two rows per scroll step -- and the cost is proportional to the mutation rather
  // than to the document.
  const observeAdded = (records) => {
    for (const rec of records) {
      const added = rec.addedNodes;
      for (let i = 0; i < added.length; i++) {
        const node = added[i];
        if (!node || node.nodeType !== 1) continue;
        if (node.hasAttribute && node.hasAttribute("data-role")) {
          observeOne(node, VIS.seen_count++);
        }
        // A row wrapper arrives with the message inside it, so the added node is the ancestor.
        // Bounded by the added subtree, never by the document.
        if (node.querySelectorAll) {
          for (const inner of node.querySelectorAll("[data-role]")) {
            observeOne(inner, VIS.seen_count++);
          }
        }
      }
    }
  };

  window.__sb.parityVisible = {
    watch() {
      try {
        const vp = D().viewport && D().viewport();
        if (!vp) return { visible_attempted: false, reason: "no thread viewport" };
        if (VIS.watching) return { visible_attempted: true, already: true };
        VIS.ever = new Set();
        VIS.seen = new WeakSet();
        VIS.seen_count = 0;
        VIS.obs = new IntersectionObserver((entries) => {
          for (const entry of entries) {
            if (!entry.isIntersecting) continue;
            const ord = entry.target.__sbOrdinal;
            if (typeof ord === "number") VIS.ever.add(ord);
          }
        }, { root: vp, threshold: 0 });
        observeAll();
        VIS.seen_count = (D().messages && D().messages().length) || 0;
        // childList only, and no attribute or character-data records: a row ENTERING the DOM is
        // the only thing that needs observing, and a row leaving it has already contributed to
        // `ever`. Text changing inside a mounted row -- which is what a stream is -- must not
        // reach this callback at all.
        VIS.mut = new MutationObserver(observeAdded);
        VIS.mut.observe(vp, { childList: true, subtree: true });
        VIS.watching = true;
        return { visible_attempted: true, already: false };
      } catch (e) {
        return { visible_attempted: false, reason: String(e) };
      }
    },

    async capture() {
      try {
        if (!VIS.watching) {
          return { visible_attempted: false, reason: "the visibility observer was never installed" };
        }
        // WAIT FOR A FRAME FIRST, and this is not defensive padding.
        //
        // IntersectionObserver computes intersections as a step of the rendering lifecycle, so
        // with no frame between `observe()` and the read there is nothing to deliver and
        // `takeRecords()` returns an empty list. An action that scrolls produces frames and hides
        // this completely; a QUIET action -- open a menu, change a setting, type a character --
        // may not, and the capture then reports that the viewport showed nothing at all. The
        // analysis would correctly refuse such a pair as NOT_COMPARABLE, so it is not a false
        // pass, but it would silently strip visible-region coverage from exactly the actions that
        // are cheapest to get right. Observed directly: at rest, with no scroll, the observer
        // reported an empty set while message 1 filled the viewport.
        await new Promise((resolve) => {
          requestAnimationFrame(() => requestAnimationFrame(resolve));
        });
        // Then deliver anything still queued. Tearing down before this drops the last entries of
        // the action -- the ones most likely to describe what it just put on screen.
        const records = VIS.obs.takeRecords();
        for (const entry of records) {
          if (entry.isIntersecting && typeof entry.target.__sbOrdinal === "number") {
            VIS.ever.add(entry.target.__sbOrdinal);
          }
        }
        const nodes = (D().messages && D().messages()) || [];
        const byOrdinal = {};
        let mounted_ever_visible = 0;
        for (let i = 0; i < nodes.length; i++) {
          const el = nodes[i];
          const ord = typeof el.__sbOrdinal === "number" ? el.__sbOrdinal : ordinalOf(el, i);
          if (!VIS.ever.has(ord)) continue;
          mounted_ever_visible += 1;
          const sig = signature(el);
          byOrdinal[String(ord)] = {
            role: el.getAttribute("data-role") || "?",
            digest: hash(sig),
            chars: sig.length,
          };
        }
        const ever = [...VIS.ever].sort((a, b) => a - b);
        return {
          visible_attempted: true,
          // Every ordinal the viewport ever showed during the window, INCLUDING any that have
          // since been unmounted. The gap between this and `messages` is the honest measure of
          // what a windowed arm could not be asked about at capture time.
          ever_visible: ever,
          ever_visible_count: ever.length,
          mounted_ever_visible,
          unmounted_at_capture: ever.length - mounted_ever_visible,
          messages: byOrdinal,
        };
      } catch (e) {
        return { visible_attempted: false, reason: String(e) };
      }
    },

    stop() {
      try {
        if (VIS.obs) VIS.obs.disconnect();
        if (VIS.mut) VIS.mut.disconnect();
      } catch (e) { /* nothing to do */ }
      VIS.obs = null;
      VIS.mut = null;
      VIS.watching = false;
    },
  };

  window.__sb.parity = {
    // Exposed so the offline unit tests can drive the exact regexes that ship, rather than a
    // second copy of them that is free to drift.
    normText,
    normUrl,
    signature,
    hash,

    // One digest for the whole thread, plus a digest PER MESSAGE so a mismatch says which message
    // moved rather than only that something did. A whole-page digest that differs is a fact you
    // cannot act on; the per-message row is what makes it a bug report.
    //
    // `opts.raw` additionally returns the signature TEXT. It is large, so it is off by default and
    // used only by the null-control hunt, which has to name the volatile rather than count it.
    capture(opts) {
      const want_raw = Boolean(opts && opts.raw);
      try {
        const dom = D();
        const rootFn = dom.threadRoot;
        const found = rootFn ? rootFn() : null;
        // WHICH root was digested travels with the digest. `threadRoot()` falls back to
        // `document.body`, which digests the sidebar, the header and every relative timestamp in
        // them. Two arms that silently digested different roots would still produce two
        // comparable-looking hashes, and the mismatch would be read as a UI change.
        const isThread = Boolean(found && found !== document.body &&
                                 found.classList && found.classList.contains("aui-thread-root"));
        const root = found || document.body;
        const whole = signature(root);
        const messages = [];
        const nodes = (dom.messages && dom.messages()) || [];
        for (let i = 0; i < nodes.length; i++) {
          const sig = signature(nodes[i]);
          const row = {
            i,
            role: nodes[i].getAttribute("data-role") || "?",
            digest: hash(sig),
            chars: sig.length,
          };
          if (want_raw) row.raw = sig;
          messages.push(row);
        }
        // Surfaces that live OUTSIDE the thread root and are therefore invisible to the digest
        // above: an open dialog, an open menu, the model picker. A perf change that alters a
        // popover would otherwise pass a thread-only parity check.
        const overlays = [];
        for (const sel of ['[data-slot="dialog-content"]', '[role="menu"]', '[role="dialog"]',
                           ".unsloth-model-selector-menu", "[data-radix-popper-content-wrapper]"]) {
          for (const el of document.querySelectorAll(sel)) {
            const sig = signature(el);
            const row = { sel, digest: hash(sig), chars: sig.length };
            if (want_raw) row.raw = sig;
            overlays.push(row);
          }
        }
        const styles = styleProbe();
        if (!want_raw) delete styles.sig;
        const out = {
          parity_attempted: true,
          root_kind: isThread ? "thread" : "body",
          digest: hash(whole),
          chars: whole.length,
          messages,
          overlays,
          styles,
          // HOW MUCH OF THE THREAD THIS DIGEST COVERS.
          //
          // Every per-message row above is keyed by `i`, its position in the MOUNTED list. On the
          // shipped build that is also its position in the conversation, so the key is meaningful
          // and two arms can be compared row by row. On an arm that mounts a window it is not:
          // msg3 on one side and msg3 on the other are different messages, and comparing their
          // digests produces a wall of mismatches that says nothing about either build.
          //
          // Carrying the two numbers means the comparison layer can REFUSE rather than report
          // eighteen false differences. Identical on the shipped build, so nothing changes for a
          // normal pair.
          mounted_messages: messages.length,
          thread_total: (dom.threadTotal && dom.threadTotal()) || messages.length,
        };
        if (want_raw) out.raw = whole;
        return out;
      } catch (err) {
        // A failure is reported as a failure. A parity check that returns an empty digest on
        // error would read as "everything matched", which is the single worst thing it could do.
        return { parity_attempted: false, reason: String(err && err.message ? err.message : err) };
      }
    },
  };
})();
