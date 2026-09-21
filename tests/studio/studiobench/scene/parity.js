// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
// UI PARITY: a structural signature of the rendered thread, taken at the close of every action
// window on BOTH arms of an A/B.
// Cheap breadth, unlike the screenshot parity pair: one `evaluate` per slot inside a comparison
// the tool already runs, versus 324 composites nobody will open.
// IT SEES: tag structure and nesting, element order, attribute names, non-volatile attribute
// values (class, data-slot, data-state, data-role, aria-hidden, hidden, disabled, role, title,
// alt, placeholder) and all text after time normalisation.
// IT DOES NOT SEE: stylesheet CSS (a class name compares as a string), computed layout and
// geometry, colour, typography and animation, raster content behind a normalised src, anything
// outside the thread root, and anything the DOM does not carry (focus, caret, selection, scroll
// position, live input values, shadow DOM, JS state).
// Measured, not asserted: against a real visible sidebar-drag change this digest found 0 of 34
// differing pairs with its null control also at 0, while three purpose-built captures found it
// 34 of 34. The blind spots that mattered were the sidebar being outside the root and computed
// layout never being read.
// So a PARITY OK verdict means NO THREAD-STRUCTURE CHANGE WAS DETECTED. It does not mean the UI is unchanged.
// Any scan that can return zero needs a positive control: `styleProbe` walks a hand-written
// selector list, so a renamed class makes it hash the empty string identically on both arms.
// `compare_styles` refuses a zero-element probe, and `elements` travels with every reading.
// Everything normalised below was OBSERVED to differ in a base-vs-base null control, nothing on
// suspicion: Radix generated ids, rendered durations (unslothai/unsloth#9054), relative times,
// scroll state, backend-minted record ids, absolute URLs (two ports; the path is kept), and
// blob/data URLs.
// A 295 vs 310 difference that is wall clock, not a build change.
// DELIBERATELY KEPT: tag structure, data-slot, data-state, data-role, aria-hidden, hidden,
// disabled, the class list and all text. A perf change that quietly flips a reasoning pane's
// data-state is the exact defect this catches.

(() => {
  if (window.__sb && window.__sb.parity) return;
  window.__sb = window.__sb || {};

  // Attributes whose VALUE is volatile between two runs of the same build; the presence is still recorded.
  const VOLATILE_ATTRS = new Set([
    "id", "for", "aria-controls", "aria-labelledby", "aria-describedby", "aria-activedescendant",
    "style", "data-radix-scroll-area-viewport", "data-testid-instance",
    // Backend-minted record ids. Two arms are two installs with two databases.
    "data-thread-id", "data-message-id", "data-attachment-id", "data-part-id", "data-run-id",
    "data-tool-call-id", "data-checkpoint-id",
  ]);

  // Attributes that carry no rendered meaning at all.
  const IGNORED_ATTRS = new Set(["data-react-checksum", "data-reactroot"]);

  // VIRTUALIZATION BOOKKEEPING, dropped name and value from the VISIBLE-REGION digest only and
  // passed in by that caller; the whole argument is at `parityVisible.capture()`.
  const VIRTUALIZATION_ATTRS = new Set(["aria-posinset", "aria-setsize"]);

  // URL-valued attributes: the origin is stripped and the path kept, so a genuinely different asset
  // still moves the digest while the arm's own port does not.
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

  // FNV-1a, 32 bit, unsigned hex. Not cryptographic: a change detector over two runs of one page.
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

  // `dropAttrs` is a Set of attributes this digest does not compare AT ALL, unlike VOLATILE_ATTRS
  // which keeps the presence. `elide` is a Set of ELEMENTS whose subtree is not serialised: a marker
  // carrying the tag and `data-role` goes in their place, so presence, position and role still
  // compare and a vanished message still moves the digest. Both off by default.
  const signature = (root, dropAttrs, elide) => {
    if (!root) return "";
    const parts = [];
    const walk = (el, depth) => {
      // A depth cap is a TRUNCATION, so the marker is left in the signature and anything deeper reads
      // as "not walked" rather than as absent.
      if (depth > 40) { parts.push("<!depth-cap>"); return; }
      if (elide && elide.has(el)) {
        // Named, not silent: a reader of the raw signature sees which element was held back and why.
        parts.push(
          "<!in-flight " + el.tagName.toLowerCase() +
          " role=" + ((el.getAttribute && el.getAttribute("data-role")) || "?") + ">"
        );
        return;
      }
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
      // ADJACENT TEXT NODES ARE JOINED BEFORE THEY ARE NORMALISED. React renders `Thought for {n}
      // seconds` as three text nodes, so per-node normalisation sees a bare digit, the rendered-duration
      // rule cannot match, and a wall-clock number reaches the digest -- the largest single source of
      // false alarms in the null control. Joined raw and normalised once.
      // Found by `sweep/parity_null_control.py --hunt`.
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

  // The bounded computed-style probe: three properties on at most `STYLE_CAP` elements, digested
  // SEPARATELY from the structural signature. `getComputedStyle` is the only part of this file that
  // can see a `.css` change, and the most likely to be caught mid-transition, so folding it in
  // would stake the instrument's credibility on its least stable reading. `display` and
  // `visibility` are how CSS hides something without touching the DOM; `pointer-events` because an
  // overlay that swallows clicks leaves no structural trace.
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
    // `capped` travels with the reading: a style digest over the first 64 of 300 elements is partial,
    // and a partial reading that does not say so reads as a complete one.
    return { digest: hash(sig), chars: sig.length, elements: n, capped, props: STYLE_PROPS, sig };
  };

  const D = () => (window.__sb.dom || {});

  // THE POLICY: changes must preserve UI and UX idempotency, with three exemptions -- a dramatic
  // performance win, a difference that exists only OFF SCREEN, and a select-all that need not
  // select all provided the copy stays complete. The third is scored in analysis/behaviour.py.
  // The whole-document digest cannot express the off-screen exemption: it compares everything in
  // the DOM, so any deferred off-screen work fails it by construction. Refusing such a pair as
  // NOT_APPLICABLE withholds a verdict; this supplies one.
  // THE CLAIM: every message visible in the viewport at any point during the action is present on
  // both arms and identical between them, and every difference lies outside the viewport.
  // 1. PARTIAL INTERSECTION COUNTS AS VISIBLE and the element is digested IN FULL: digesting only
  // the visible part would need per-node geometry, which this must not read. The admitted error is
  // a FALSE ALARM, never a false pass.
  // 2. ANYTHING VISIBLE AT ANY POINT IS COMPARED, not just at the end: the observer is installed
  // BEFORE the window opens and the compared set is the UNION of everything that intersected. The
  // per-message digest is still the one taken at the close, named `now_visible` vs `ever_visible`.
  // GEOMETRY IS FORBIDDEN HERE: `getBoundingClientRect()` on content inside a `content-visibility`
  // locked subtree makes Chromium render it, so a geometry probe unlocks exactly what it came to
  // observe -- one session reported 0 off-screen unrendered roots while the counter recorded 22
  // skipped. IntersectionObserver is what Blink's own relevance machinery uses.
  // `getClientRects()` too.
  const VIS = {
    obs: null, mut: null, ever: new Set(), watching: false,
    // Nodes already counted in `unplaced`, so a row refused several times before it mounts does not
    // inflate the diagnostic. NOT a "do not look at this again" set: placement is re-read on every
    // offer, because a recycled row changes position.
    unplacedSeen: new WeakSet(),
    // Rows observed that could NOT be placed in the thread at all: no `aria-posinset` and not among
    // the messages the DOM holds. Such a row is stamped with no ordinal and so is silently absent
    // from `ever_visible`, which is why it is reported with the capture.
    unplaced: 0,
    // The batch-scoped position index. See `positionIndex`.
    index: null,
  };

  const ordinalOf = (el, position) => {
    // The message's position in the THREAD, not in the mounted list: mounted index 0 is message 10 on
    // one arm and message 1 on the other, so a per-index comparison reports every row as changed.
    // `aria-posinset` is the windowed arm's own claim; a fully mounted thread's DOM order IS it.
    const owner = el.closest ? el.closest("[aria-posinset]") : null;
    if (owner) {
      const n = Number(owner.getAttribute("aria-posinset"));
      if (Number.isFinite(n)) return n;
    }
    return position;
  };

  // THE POSITION INDEX, AND WHY THE FALLBACK ORDINAL IS NOT A COUNTER. A row publishing no
  // `aria-posinset` is stamped with its position among the thread's messages in the DOM as it
  // stands, resolved when the row is OBSERVED, because by delivery time it may be unmounted.
  // A LIFETIME COUNT OF OBSERVED NODES IS NOT THAT POSITION: `thread_reopen` recreates every message
  // row inside one document, and rebuilt rows legitimately publish no ordinal, so a counter stamped
  // them N+1..2N against the windowed arm's real 1..N and `compare_visible` reported two disjoint
  // visible sets for an identical rebuild.
  // THE COST, which is why a counter was tempting: `observeAdded` runs inside the MEASURED window,
  // so an O(document) lookup per mutation is the trap. Three things keep that walk off the
  // per-mutation path: `aria-posinset` is read FIRST, so a windowed arm never builds an index; the
  // index is built at most once per mutation batch and only by a batch that mounted a message
  // element; and it is read from the live DOM inside the callback, so a rebuild reads N not 2N.
  // Workspace task #102. The per-mutation bound is held by
  // test_the_top_up_is_proportional_to_the_mutation_not_to_the_document.
  const positionIndex = () => {
    if (VIS.index) return VIS.index;
    const nodes = (D().messages && D().messages()) || [];
    const map = new Map();
    for (let i = 0; i < nodes.length; i++) map.set(nodes[i], i + 1);
    VIS.index = map;
    return map;
  };

  // `position` is the row's 1-based position among the thread's messages when the caller already
  // knows it; left undefined it is resolved from the index, and only when no ordinal was published.
  const observeOne = (el, position) => {
    if (!VIS.obs) return;
    let ord = ordinalOf(el, typeof position === "number" ? position : null);
    if (ord === null) ord = positionIndex().get(el) || 0;
    // The ordinal is stamped on the node: by the time an entry is delivered the node may be unmounted
    // and `closest()` would return nothing.
    if (ord > 0) {
      // RE-READ RATHER THAN WRITE ONCE: a virtualizer that RECYCLES rows renumbers them, so a `seen`
      // stamp stayed at the old ordinal, the row's content was digested under a position it no longer
      // held, and the message it now showed was never reported visible. Re-reading is bounded: the full
      // scan runs once and every other call walks only what a mutation added.
      const had = el.__sbOrdinal;
      el.__sbOrdinal = ord;
      if (typeof had === "number" && had !== ord) {
        // A RENUMBERED ROW THAT NEVER STOPPED INTERSECTING REPORTS NOTHING: IntersectionObserver delivers
        // on a CHANGE of intersection and only this node's identity changed. Re-registering the target
        // makes the observer deliver an initial entry, the only way to ask "is this on screen now"
        // without calling a geometry method.
        VIS.obs.unobserve(el);
      }
    } else {
      // NO ORDINAL RATHER THAN A MADE-UP ONE: a guessed position lands in `ever_visible` as a message
      // the other arm never showed. NOT marked `seen`, which is the whole point -- a row observed while
      // detached is handed back by a recycling virtualizer, and with the node already in `seen` every
      // later intersection was dropped and `compare_visible` was free to call MATCH.
      if (!VIS.unplacedSeen.has(el)) {
        VIS.unplacedSeen.add(el);
        VIS.unplaced += 1;
      }
    }
    // Idempotent by specification: `observe()` on a target already being observed adds no second
    // registration, so re-offering an unplaced row costs nothing.
    VIS.obs.observe(el);
  };

  // The FULL scan. O(the document), so it runs exactly once, when the observer is installed, which
  // is before the measured window opens.
  const observeAll = () => {
    if (!VIS.obs) return;
    const nodes = (D().messages && D().messages()) || [];
    for (let i = 0; i < nodes.length; i++) observeOne(nodes[i], i + 1);
  };

  // THE TOP-UP, AND WHY IT IS NOT A RESCAN. A windowed list mounts rows as it scrolls, but
  // re-running the full scan from a MutationObserver would charge an O(document) walk to the
  // measured action once per batch on a 64,000-element DOM, which reported delete_message at 14.3
  // fps where it costs 49.0. So this walks only what was ADDED, and the one document read an
  // ordinal-less arm needs is taken once for the whole batch.
  const observeAdded = (records) => {
    // ONE INDEX PER BATCH AT MOST, dropped on the way out so the next batch cannot be answered from a
    // stale list: a position read from the previous DOM is precisely the wrong answer, the same
    // class of mistake as the lifetime counter this replaced.
    VIS.index = null;
    for (const rec of records) {
      // A RENUMBERED ROW ARRIVES AS ITS OWN TARGET, not in anybody's `addedNodes`, so the row to
      // re-place is the target's message descendant, or the target itself when the attribute is on it.
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
        // A row wrapper arrives with the message inside it, so the added node is the ancestor. Bounded by
        // the added subtree, never by the document.
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
        // childList plus ONE attribute by name and no character-data records: text changing inside a
        // mounted row -- which is what a stream is -- must not reach this callback. `aria-posinset` is
        // here because its change means the row is a DIFFERENT MESSAGE, which is not a childList
        // mutation when a virtualizer renumbers a still-connected row in place.
        // `ordinalOf` reaches `[aria-posinset]` with `closest()`.
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
        // WAIT FOR A FRAME FIRST: IntersectionObserver computes intersections as a step of the rendering
        // lifecycle, so with no frame between `observe()` and the read `takeRecords()` returns nothing.
        // Observed directly, the observer reported an empty set while message 1 filled the viewport.
        await new Promise((resolve) => {
          requestAnimationFrame(() => requestAnimationFrame(resolve));
        });
        // Then deliver anything still queued: tearing down first drops the last entries of the action.
        const records = VIS.obs.takeRecords();
        for (const entry of records) {
          if (entry.isIntersecting && typeof entry.target.__sbOrdinal === "number") {
            VIS.ever.add(entry.target.__sbOrdinal);
          }
        }
        const dom = D();
        const nodes = (dom.messages && dom.messages()) || [];
        // THE SAME NODE SET THE POSITIVE CONTROL BELOW COUNTS: read from one call, so the control cannot
        // fire while a row still claims to be in flight, or the reverse.
        const live = new Set((dom.streamingMessages && dom.streamingMessages()) || []);
        const byOrdinal = {};
        let mounted_ever_visible = 0;
        // TWO MOUNTED ROWS CANNOT BE ONE MESSAGE: the assignment below is keyed by ordinal, so a second
        // row reusing one silently replaced the first row's digest and `VIS.ever` collapsed them too.
        // Counted directly, not derived: `unmounted_at_capture` reads a clean 0 over a live collision
        // because the extra row and the vacancy cancel.
        let ordinal_collisions = 0;
        const collided = [];
        for (let i = 0; i < nodes.length; i++) {
          const el = nodes[i];
          const ord = typeof el.__sbOrdinal === "number" ? el.__sbOrdinal : ordinalOf(el, i + 1);
          if (!VIS.ever.has(ord)) continue;
          mounted_ever_visible += 1;
          // WITHOUT THE VIRTUALIZATION BOOKKEEPING: runtime/readiness.py accepts `aria-posinset` /
          // `aria-setsize` on the message or on an ancestor row, so an arm taking the first option carries
          // two attributes on every message the fully mounted arm publishes on none, and visible-region
          // parity reported a wall of differences for a DOM shape the gate permits. NOT normalised in the
          // whole-document digest, which only runs where neither arm windows, so the exclusion is passed in
          // by this caller. A windowed arm publishing a WRONG ordinal is covered by the readiness gate.
          const sig = signature(el, VIRTUALIZATION_ATTRS);
          if (Object.prototype.hasOwnProperty.call(byOrdinal, String(ord))) {
            ordinal_collisions += 1;
            if (collided.indexOf(ord) === -1) collided.push(ord);
          }
          byOrdinal[String(ord)] = {
            role: el.getAttribute("data-role") || "?",
            digest: hash(sig),
            chars: sig.length,
            // Still being written at capture time, so its digest names a point in a stream rather than a
            // rendering: residue, never a difference.
            in_flight: live.has(el),
          };
        }
        const ever = [...VIS.ever].sort((a, b) => a - b);
        // THE SAME POSITIVE CONTROL THE STRUCTURAL CAPTURE CARRIES, because this payload is scored by
        // `compare_visible` alone. The per-row `in_flight` comes from the SAME `streamingMessages()`
        // call, and the two arms are two different builds, so a hook renamed on the treatment only is
        // asymmetric rather than cancelling. READ GLOBALLY, not over the visible rows: a reply streaming
        // below the fold is ordinary and must not refuse anything.
        // `--ab REF` installs the treatment under its own `UNSLOTH_STUDIO_HOME`.
        const generating = dom.generating
          ? Boolean(dom.generating())
          : Boolean(dom.isRunning && dom.isRunning());
        // WHY THE PROBE IS QUIET is three questions, not one: parts published with none saying running
        // means the hook's VOCABULARY changed, blind unless this arm is windowing; the last assistant
        // message publishing nothing is ordinary unless no assistant message anywhere publishes, and
        // then the hook itself is gone.
        const lastPublishes = dom.lastAssistantPublishesStatus
          ? Boolean(dom.lastAssistantPublishesStatus())
          : false;
        const anyPublishes = dom.statusHookPresent ? Boolean(dom.statusHookPresent()) : true;
        const windowedArm = dom.isWindowed ? Boolean(dom.isWindowed()) : false;
        const probeBlind = lastPublishes ? !windowedArm : !anyPublishes;
        return {
          visible_attempted: true,
          streaming: generating,
          // Carried so a reader can tell the two zero-in-flight readings apart in the record: a stream this
          // capture could not see, versus a hook that is gone.
          status_hook_present: anyPublishes,
          // `hookGone` and not merely "nothing is running": a windowed arm scrolled away from the tail can
          // UNMOUNT the message being written, and `streamingMessages()` scans only mounted DOM, so
          // `live.size` is zero on a build whose hooks are intact.
          // See `dom.statusHookPresent`.
          // And `dom.lastAssistantPublishesStatus`.
          in_flight_unplaced: Boolean(generating && live.size === 0 && probeBlind),
          // Every ordinal the viewport ever showed, including any since unmounted; the gap between this and
          // `messages` measures what a windowed arm could not be asked about at capture time.
          ever_visible: ever,
          ever_visible_count: ever.length,
          mounted_ever_visible,
          unmounted_at_capture: ever.length - mounted_ever_visible,
          // Rows this instrument could not place in the thread: no published ordinal and absent from the
          // message list when observed. They appear nowhere above, which is a hole in the compared set
          // rather than agreement, so it is counted here.
          unplaced_rows: VIS.unplaced,
          // Mounted, ever-visible rows whose ordinal a row already written had: the digest map and
          // `ever_visible` both lose one of the two, so a nonzero count means neither can be compared.
          ordinal_collisions,
          collided_ordinals: collided.sort((a, b) => a - b),
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
    // Exposed so the offline unit tests drive the exact regexes that ship rather than a second copy free to drift.
    normText,
    normUrl,
    signature,
    hash,

    // One digest for the whole thread plus one PER MESSAGE, so a mismatch says which message moved
    // rather than only that something did. `opts.raw` also returns the signature TEXT: large, so it
    // is off by default and used only by the null-control hunt.
    capture(opts) {
      const want_raw = Boolean(opts && opts.raw);
      try {
        const dom = D();
        const rootFn = dom.threadRoot;
        const found = rootFn ? rootFn() : null;
        // WHICH root was digested travels with the digest: `threadRoot()` falls back to `document.body`,
        // and two arms that silently digested different roots would still produce comparable-looking
        // hashes.
        const isThread = Boolean(found && found !== document.body &&
                                 found.classList && found.classList.contains("aui-thread-root"));
        const root = found || document.body;
        const nodes = (dom.messages && dom.messages()) || [];
        // THE STREAMED MESSAGE GETS ITS OWN DIGEST. A message still being written has no defined moment:
        // the two arms share one pacer so the bytes are identical, but each has its own send click and
        // paint clock while the digest is taken at a wall-clock offset in the film, so the arms are
        // compared at two different points in the same reply.
        // The renderer amplifies it: mid-stream `parseIncompleteMarkdown` remends the tail, KaTeX renders
        // the repaired formula, Shiki re-tokenises the repaired fence, and the trailing code block
        // carries `data-incomplete`. None of that is monotonic in how much text has arrived, so a
        // same-length mismatch is not evidence of a missed volatile.
        // MEASURED on the shipped corpus (unit 9, 4,238 characters): stepping by the pacer's 24-character
        // chunk, 175 of 175 adjacent pairs produce a different digest, so one chunk of skew fails a stable
        // action outright. At one-character resolution the serialised length even moves DOWNWARDS at 52
        // of 4,237 steps.
        // SO: the in-flight message is named and a SECOND whole-thread digest is taken with its subtree
        // elided, a marker keeping its presence, position and role. The comparison layer scores the
        // settled document on the settled digest and refuses a verdict on the in-flight message.
        // THE SCAFFOLD DIGEST ELIDES EVERY MESSAGE, not only the streaming ones: whether a message is in
        // flight is a property of ONE arm at the moment ITS digest was taken, and the ordinary case is
        // that the arms disagree, so per-arm elision would make the two walks differ because of the
        // elision itself. Nothing is given up -- the scaffold plus the per-message rows IS the digest.
        const inFlightNodes = (dom.streamingMessages && dom.streamingMessages()) || [];
        const inFlight = new Set(inFlightNodes);
        const running = Boolean(dom.isRunning && dom.isRunning());
        // The narrower reading, needed by the positive control below: `isRunning()` is true whenever the
        // composer refuses a fresh send, including a prompt queued on an IDLE thread.
        // `dom.generating` is the reading; `analysis/parity.compare` refuses to fold it into a pass.
        const generating = dom.generating ? Boolean(dom.generating()) : running;
        // Whether the app still publishes the status contract at all: a hook that is GONE is a blind
        // instrument, while one present with nothing running is an ordinary settled thread.
        // WHY THE PROBE IS QUIET is three questions, not one: parts published with none saying running
        // means the hook's VOCABULARY changed; nothing published at all is ordinary unless no assistant
        // message anywhere publishes, and then the hook is gone.
        const lastPublishes = dom.lastAssistantPublishesStatus
          ? Boolean(dom.lastAssistantPublishesStatus())
          : false;
        const anyPublishes = dom.statusHookPresent ? Boolean(dom.statusHookPresent()) : true;
        const windowedArm = dom.isWindowed ? Boolean(dom.isWindowed()) : false;
        const probeBlind = lastPublishes ? !windowedArm : !anyPublishes;
        const whole = signature(root);
        const scaffold = signature(root, undefined, new Set(nodes));
        const messages = [];
        const in_flight = [];
        for (let i = 0; i < nodes.length; i++) {
          const sig = signature(nodes[i]);
          const row = {
            i,
            role: nodes[i].getAttribute("data-role") || "?",
            digest: hash(sig),
            chars: sig.length,
          };
          if (inFlight.has(nodes[i])) {
            row.in_flight = true;
            in_flight.push(i);
          }
          if (want_raw) row.raw = sig;
          messages.push(row);
        }
        // Surfaces OUTSIDE the thread root and therefore invisible to the digest above -- an open dialog,
        // an open menu, the model picker -- so a perf change that alters a popover cannot pass a
        // thread-only parity check.
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
          // The thread with every message replaced by a marker carrying its tag, role and position.
          // `digest` = this plus the per-message rows below, which is what lets one message be withheld
          // from the comparison without the rest of the reading going with it.
          digest_scaffold: hash(scaffold),
          chars_scaffold: scaffold.length,
          // Mounted indices of the messages still being written; an empty list on a page where nothing is
          // running is the ordinary case.
          in_flight,
          streaming: running,
          // THE POSITIVE CONTROL: `streamingMessages()` walks selectors written against Unsloth's markup,
          // so renaming `data-status` makes every reading silently become "nothing was streaming". The app
          // also reports a running reply through the Stop button, so a disagreement is carried out rather
          // than resolved here, and `analysis/parity.comparability` refuses the pair. It compares
          // `generating()` and NOT `isRunning()`, which is also true while
          // a queued prompt waits on an idle thread. And it requires the hook to be GONE, not merely quiet:
          // `streamingMessages()` scans mounted DOM.
          in_flight_unplaced: Boolean(
            generating && inFlightNodes.length === 0 && probeBlind
          ),
          status_hook_present: anyPublishes,
          // The queued-idle interval itself, recorded rather than resolved away, so a reader can tell why a
          // capture with `streaming: true` placed no message in flight.
          // INCLUDING THE DISPATCHED WAIT: `isRunning()` matches "Stop generating" and "Queue message"
          // only, so the interval rendering "Stop queued message" recorded both flags false, identical to a
          // settled Send arm, and a pair straddling it looked like a rendering regression.
          queued_idle: Boolean(
            (running || (dom.stopQueuedButton ? Boolean(dom.stopQueuedButton()) : false)) &&
              !generating
          ),
          // The composer's run-state slot as a token, so the comparison layer can tell a scaffold that
          // differs because the arms were at different points in one turn from one that differs because
          // something rendered differently.
          // See `dom.runStateControl` and `analysis/parity.generation_disagrees`.
          composer_control: dom.runStateControl ? dom.runStateControl() : null,
          messages,
          overlays,
          styles,
          // HOW MUCH OF THE THREAD THIS DIGEST COVERS. Every per-message row is keyed by its position in
          // the MOUNTED list, which equals its conversation position only on the shipped build; on a
          // windowed arm msg3 and msg3 are different messages and a row-by-row comparison says nothing.
          // Carrying the two numbers lets the comparison layer REFUSE instead.
          mounted_messages: messages.length,
          thread_total: (dom.threadTotal && dom.threadTotal()) || messages.length,
        };
        if (want_raw) out.raw = whole;
        return out;
      } catch (err) {
        // A failure is reported as a failure: a parity check that returned an empty digest on error would
        // read as "everything matched".
        return { parity_attempted: false, reason: String(err && err.message ? err.message : err) };
      }
    },
  };
})();
