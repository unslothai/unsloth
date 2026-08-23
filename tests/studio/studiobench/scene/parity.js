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

  // VIRTUALIZATION BOOKKEEPING. Dropped -- name and value -- from the VISIBLE-REGION digest only,
  // and passed in by that caller rather than applied here, because the two comparisons want
  // different things from these two attributes. The whole argument is at the call site, in
  // `parityVisible.capture()`.
  const VIRTUALIZATION_ATTRS = new Set(["aria-posinset", "aria-setsize"]);

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

  // `dropAttrs`, when given, is a Set of attribute names this digest does not compare AT ALL --
  // neither presence nor value, unlike VOLATILE_ATTRS which keeps the presence. Optional and off
  // by default: every existing caller (the whole-thread digest, the per-message rows, the overlays
  // and the node-driven unit tests) passes nothing and gets exactly what it got before.
  const signature = (root, dropAttrs) => {
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
        if (dropAttrs && dropAttrs.has(attr.name)) continue;
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
      // ADJACENT TEXT NODES ARE JOINED BEFORE THEY ARE NORMALISED, and this is the difference
      // between a normaliser that works and one that looks like it works.
      //
      // React renders `Thought for {n} seconds` as THREE text nodes -- "Thought for ", "3",
      // " seconds" -- because the interpolation is its own child. Normalising each node on its
      // own, `normText("3")` sees a bare digit with no unit after it, the rendered-duration rule
      // cannot match, and a wall-clock number reaches the digest. Two runs of a BYTE-IDENTICAL
      // build then disagree by one character, which is exactly what a null control here measured:
      // `msg22(assistant):17334->17334c`, same length, `3` against `2`, hunted down to this line
      // by `sweep/parity_null_control.py --hunt`. It was the single largest source of false
      // alarms in that control, and every one of them read as "this pull request changed the UI".
      //
      // The run is joined RAW and normalised once, so what gets normalised is the text as the
      // page renders it rather than the accident of where React split it. Element boundaries
      // still break a run, so this cannot weld two separate labels into one string.
      let run = "";
      const flush = () => {
        if (!run) return;
        const t = normText(run);
        if (t) parts.push(t);
        run = "";
      };
      for (const child of el.childNodes) {
        if (child.nodeType === 3) {
          run += child.nodeValue == null ? "" : child.nodeValue;
        } else if (child.nodeType === 1) {
          flush();
          walk(child, depth + 1);
        }
      }
      flush();
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
    obs: null, mut: null, ever: new Set(), watching: false,
    // Nodes already counted in `unplaced`, so a row offered and refused several times before it
    // mounts does not inflate the diagnostic. NOT a "do not look at this again" set: placement is
    // re-read on every offer, because a recycled row changes position. See `observeOne`.
    unplacedSeen: new WeakSet(),
    // Rows that were observed and could NOT be placed in the thread at all: no `aria-posinset`
    // and not among the messages the DOM currently holds. Such a row is stamped with no ordinal,
    // so it is silently absent from `ever_visible`, and a silence that cannot be counted is the
    // failure this whole file is written against. Reported with the capture.
    unplaced: 0,
    // The batch-scoped position index. See `positionIndex`.
    index: null,
  };

  const ordinalOf = (el, position) => {
    // The message's position in the THREAD, not in the mounted list. This is what makes a windowed
    // arm comparable with a fully mounted one at all: mounted index 0 is message 10 on one arm and
    // message 1 on the other, so a per-index comparison compares different messages and reports
    // every row as changed. `aria-posinset` is the windowed arm's own claim about position; a
    // fully mounted thread does not publish it and does not need to, because there the DOM order
    // IS the thread order, which is what `position` carries.
    const owner = el.closest ? el.closest("[aria-posinset]") : null;
    if (owner) {
      const n = Number(owner.getAttribute("aria-posinset"));
      if (Number.isFinite(n)) return n;
    }
    return position;
  };

  // THE POSITION INDEX, AND WHY THE FALLBACK ORDINAL IS NOT A COUNTER.
  //
  // A row publishing no `aria-posinset` is stamped with its position among the thread's messages
  // in the DOM AS IT STANDS. That has to be resolved when the row is OBSERVED rather than when an
  // entry is delivered, which is the reason the ordinal is stamped onto the node in the first
  // place: by delivery time the row may have been unmounted and `closest()` would answer nothing.
  //
  // A LIFETIME COUNT OF OBSERVED NODES IS NOT THAT POSITION, and the gap between them was a
  // measured failure rather than a tidiness point. `thread_reopen` makes a fully mounted arm
  // remove and recreate every message row inside one document. Those rebuilt rows legitimately
  // publish no `aria-posinset` -- only a windowed arm publishes one -- so a counter that had
  // already seen the thread's N rows stamped the rebuilt ones N+1..2N, while the windowed arm on
  // the other side of the A/B stamped its real 1..N. `compare_visible` then saw two disjoint
  // visible sets and reported "the two arms put DIFFERENT MESSAGES on screen" for a rebuild that
  // was identical, failing thread_reopen on every base-versus-windowed pair.
  //
  // THE COST, WHICH IS WHY A COUNTER WAS TEMPTING. `observeAdded` runs inside the MEASURED action
  // window, so an O(document) lookup per mutation is workspace task #102 exactly: the instrument
  // charging its own walk to the action, on a DOM whose size is the quantity under investigation.
  // Three things keep that walk off the per-mutation path:
  //
  //   1. `aria-posinset` is read FIRST, so a windowed arm -- the only kind that mounts rows by the
  //      hundred as it scrolls -- never builds an index at all.
  //   2. The index is built at most ONCE PER MUTATION BATCH, and only by a batch that actually
  //      mounted a message element. A stream is text churn inside rows that are already mounted:
  //      it adds no elements, so it builds nothing, which is what
  //      `test_the_top_up_is_proportional_to_the_mutation_not_to_the_document` pins.
  //   3. It is read from the live DOM inside the callback, i.e. after every mutation in the batch
  //      has been applied, so ONE read describes the batch's settled state rather than some
  //      intermediate one. A rebuild that removes N rows and adds N rows in one commit is read
  //      once, and reads N rows rather than 2N.
  //
  // What is left paying for it is an arm publishing no ordinals that mounts rows mid-action:
  // send_turn's one or two new messages, and thread_reopen's single rebuild. That is a handful of
  // document reads per action window, in exchange for the ordinals being right.
  const positionIndex = () => {
    if (VIS.index) return VIS.index;
    const nodes = (D().messages && D().messages()) || [];
    const map = new Map();
    for (let i = 0; i < nodes.length; i++) map.set(nodes[i], i + 1);
    VIS.index = map;
    return map;
  };

  //: `position` is the row's 1-based position among the thread's messages when the caller already
  //: knows it -- the full scan below holds the whole ordered list and needs no index. Left
  //: undefined, the position is resolved from the index, and only when no ordinal was published.
  const observeOne = (el, position) => {
    if (!VIS.obs) return;
    let ord = ordinalOf(el, typeof position === "number" ? position : null);
    if (ord === null) ord = positionIndex().get(el) || 0;
    // The ordinal is stamped on the node, because by the time an entry is delivered the node may
    // have been unmounted and `closest()` would return nothing.
    if (ord > 0) {
      // RE-READ RATHER THAN WRITE ONCE. A placed node used to be marked `seen` and skipped for the
      // rest of the window, which assumed a row's position in the thread cannot change while the
      // node lives. A virtualizer that RECYCLES rows breaks that assumption on purpose: it hands
      // the same node to another item and renumbers it. The stamp then stayed at the old ordinal
      // for the rest of the run, so the row's content was digested under a position it no longer
      // held, and the message it now showed was never reported visible at all.
      //
      // Re-reading is bounded work. The full scan runs once; every other call arrives from
      // `observeAdded`, which walks only what a mutation added, and `positionIndex` is built at
      // most once per batch.
      const had = el.__sbOrdinal;
      el.__sbOrdinal = ord;
      if (typeof had === "number" && had !== ord) {
        // A RENUMBERED ROW THAT NEVER STOPPED INTERSECTING REPORTS NOTHING. IntersectionObserver
        // delivers on a CHANGE of intersection, and this node's intersection did not change --
        // only its identity did -- so `ever` would never learn the new ordinal. Re-registering the
        // target makes the observer deliver an initial entry for it, which is the documented
        // behaviour of `observe()` and the only way to ask "is this thing on screen right now"
        // without calling a geometry method, which this section may not do.
        VIS.obs.unobserve(el);
      }
    } else {
      // NO ORDINAL RATHER THAN A MADE-UP ONE. A row that publishes nothing and is not in the
      // thread's message list has no position this instrument can honestly claim, and a guessed
      // one lands in `ever_visible` as a message the other arm never showed.
      //
      // NOT MARKED `seen`, WHICH IS THE WHOLE POINT. Marking it here is what made an unplaced row
      // able to hide a VISIBLE one. A row is unplaceable when it is observed while detached --
      // added and removed inside one task, so no paint could have shown it -- and a virtualizer
      // that RECYCLES DOM nodes hands that same node back on a later batch, mounted and about to
      // be shown. With the node already in `seen` the early return fired, it was never stamped,
      // and every intersection it went on to report was dropped for want of an ordinal: visible
      // on this arm, absent from `ever_visible`, and `compare_visible` free to call MATCH on the
      // strength of the rows that did place. Left out of `seen` it is simply placed on the batch
      // that mounts it, which is the batch that can place it.
      //
      // Counted once per node all the same. `unplacedSeen` is a separate WeakSet so a node that
      // is offered and refused several times before it mounts does not inflate the diagnostic
      // into a number nobody can interpret.
      if (!VIS.unplacedSeen.has(el)) {
        VIS.unplacedSeen.add(el);
        VIS.unplaced += 1;
      }
    }
    // Idempotent by specification: `observe()` on a target already being observed returns without
    // adding a second registration, so re-offering an unplaced row costs nothing.
    VIS.obs.observe(el);
  };

  //: The FULL scan. O(the document), so it runs exactly once, when the observer is installed --
  //: which happens before the measured window opens.
  const observeAll = () => {
    if (!VIS.obs) return;
    const nodes = (D().messages && D().messages()) || [];
    for (let i = 0; i < nodes.length; i++) observeOne(nodes[i], i + 1);
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
  // than to the document. The one exception is stated in full at `positionIndex` above: an arm
  // that publishes no ordinals needs the thread's current message list to place a row it just
  // mounted, and that list is read ONCE for the whole batch and never once per row.
  const observeAdded = (records) => {
    // ONE INDEX PER BATCH AT MOST, and none at all for a batch that mounts nothing needing one.
    // Dropped again on the way out so the next batch cannot be answered from a stale list: rows
    // mount and unmount between batches, and a position read from the previous DOM is precisely
    // the wrong answer -- the same class of mistake as the lifetime counter this replaced.
    VIS.index = null;
    for (const rec of records) {
      // A RENUMBERED ROW ARRIVES AS ITS OWN TARGET, not in anybody's `addedNodes`. The ordinal
      // lives on `[aria-posinset]`, which `ordinalOf` reaches with `closest()`, so the row that
      // has to be re-placed is the target's message descendant, or the target itself when the
      // attribute is on the message.
      if (rec.type === "attributes") {
        const t = rec.target;
        if (!t || t.nodeType !== 1) continue;
        if (t.hasAttribute && t.hasAttribute("data-role")) observeOne(t);
        if (t.querySelectorAll) {
          for (const inner of t.querySelectorAll("[data-role]")) observeOne(inner);
        }
        continue;
      }
      const added = rec.addedNodes;
      for (let i = 0; i < added.length; i++) {
        const node = added[i];
        if (!node || node.nodeType !== 1) continue;
        if (node.hasAttribute && node.hasAttribute("data-role")) {
          observeOne(node);
        }
        // A row wrapper arrives with the message inside it, so the added node is the ancestor.
        // Bounded by the added subtree, never by the document.
        if (node.querySelectorAll) {
          for (const inner of node.querySelectorAll("[data-role]")) {
            observeOne(inner);
          }
        }
      }
    }
    VIS.index = null;
  };

  window.__sb.parityVisible = {
    watch() {
      try {
        const vp = D().viewport && D().viewport();
        if (!vp) return { visible_attempted: false, reason: "no thread viewport" };
        if (VIS.watching) return { visible_attempted: true, already: true };
        VIS.ever = new Set();
        VIS.unplacedSeen = new WeakSet();
        VIS.unplaced = 0;
        VIS.index = null;
        VIS.obs = new IntersectionObserver((entries) => {
          for (const entry of entries) {
            if (!entry.isIntersecting) continue;
            const ord = entry.target.__sbOrdinal;
            if (typeof ord === "number") VIS.ever.add(ord);
          }
        }, { root: vp, threshold: 0 });
        observeAll();
        // childList, plus ONE attribute by name, and no character-data records. A row ENTERING the
        // DOM is most of what needs observing, and a row leaving it has already contributed to
        // `ever`. Text changing inside a mounted row -- which is what a stream is -- must not
        // reach this callback at all, and with `attributeFilter` naming a single attribute it
        // cannot: a stream mutates text, not `aria-posinset`.
        //
        // `aria-posinset` is here because it is the one attribute whose change means the row is a
        // DIFFERENT MESSAGE. A virtualizer that recycles a still-connected row renumbers it in
        // place, which is not a childList mutation, so without this the renumbering is invisible
        // and the node keeps a stamp that now names the wrong message. Filtered to that name the
        // callback fires when a windowed arm re-uses a row and at no other time.
        VIS.mut = new MutationObserver(observeAdded);
        VIS.mut.observe(vp, {
          childList: true,
          subtree: true,
          attributes: true,
          attributeFilter: ["aria-posinset"],
        });
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
          const ord = typeof el.__sbOrdinal === "number" ? el.__sbOrdinal : ordinalOf(el, i + 1);
          if (!VIS.ever.has(ord)) continue;
          mounted_ever_visible += 1;
          // WITHOUT THE VIRTUALIZATION BOOKKEEPING, and the asymmetry is the reason.
          //
          // runtime/readiness.py deliberately accepts `aria-posinset` / `aria-setsize` either on
          // the `[data-role]` message or on an ancestor row wrapper -- it walks with `closest()`,
          // because the ordinal belongs on whichever element is the member of the set. An arm that
          // takes the first option therefore carries two attributes on EVERY message that the
          // fully mounted arm publishes on none, so the per-message digests differed on every
          // message while the rendered content was identical, and visible-region parity reported a
          // wall of differences for a DOM shape the gate explicitly permits. That makes auto-mode
          // parity unusable for the very arm it exists to score.
          //
          // NOT normalised in the whole-document structural digest, which is a separate decision
          // and a deliberate one. That digest is only ever applied to pairs where NEITHER arm is
          // windowing (see `decide_modes`), and between two fully mounted arms an ordinal that
          // appears, disappears or changes IS a change worth seeing. `signature` is shared with
          // the thread digest, the per-message rows, the overlays and the node-driven unit tests,
          // so the exclusion is passed in HERE by the one caller that wants it rather than baked
          // into the function every caller uses.
          //
          // What this gives up: a windowed arm publishing a WRONG ordinal on a message is no
          // longer visible in this digest. It is not unchecked -- runtime/readiness.py gates on
          // the ordinals being real positions (at least 1, distinct, no larger than the declared
          // set size, reaching the seeded total) and `probe_thread_completeness` walks them for
          // holes -- and those are the checks that can say what a right ordinal would be, which a
          // digest comparison against an arm that publishes none never could.
          const sig = signature(el, VIRTUALIZATION_ATTRS);
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
          // Rows observed that this instrument could not place in the thread: no published
          // ordinal, and absent from the message list when they were observed. They carry no
          // ordinal and so appear nowhere above, which is a hole in the compared set rather than
          // agreement, and it is counted here so a reader can see it is zero.
          unplaced_rows: VIS.unplaced,
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
