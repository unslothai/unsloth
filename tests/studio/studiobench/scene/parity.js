// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
// UI PARITY: a structural signature of the rendered thread taken at the close of every action
// window on both arms. It sees DOM structure, non-volatile attributes and normalised text; it
// does not see CSS, layout, anything outside the thread root, or focus/selection/JS state.
// So a PARITY OK verdict means NO THREAD-STRUCTURE CHANGE WAS DETECTED. It does not mean the UI is unchanged.
// styleProbe walks a fixed selector list, so a renamed class hashes empty on both arms:
// compare_styles refuses a zero-element probe and `elements` travels with every reading.
// Everything normalised below was observed to differ in a base-vs-base null control.

(() => {
  if (window.__sb && window.__sb.parity) return;
  window.__sb = window.__sb || {};

  const VOLATILE_ATTRS = new Set([
    "id", "for", "aria-controls", "aria-labelledby", "aria-describedby", "aria-activedescendant",
    "style", "data-radix-scroll-area-viewport", "data-testid-instance",
    // Two arms are two installs with two databases.
    "data-thread-id", "data-message-id", "data-attachment-id", "data-part-id", "data-run-id",
    "data-tool-call-id", "data-checkpoint-id",
  ]);

  const IGNORED_ATTRS = new Set(["data-react-checksum", "data-reactroot"]);

  // Dropped only from the visible-region digest; see parityVisible.capture().
  const VIRTUALIZATION_ATTRS = new Set(["aria-posinset", "aria-setsize"]);

  // Origin stripped, path kept, so a different asset still moves the digest but the port does not.
  const URL_ATTRS = new Set(["src", "href", "srcset", "action", "poster", "data-src", "formaction"]);

  const ID_RE = /\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b|\b[0-9a-f]{16,}\b/gi;

  const normUrl = (s) =>
    (s || "")
      .replace(/^blob:[^\s]*$/i, "#BLOB")
      .replace(/^data:([^;,]*)[^\s]*$/i, "#DATA:$1")
      .replace(/\b[a-z]+:\/\/[^/\s]+/gi, "")
      .replace(ID_RE, "#ID");

  const normText = (s) =>
    (s || "")
      .replace(/\b\d+(\.\d+)?\s?(ms|s|sec|secs|second|seconds|min|mins|minute|minutes|hour|hours|day|days)\b/gi, "#T")
      .replace(/\b(just now|a few seconds ago|yesterday)\b/gi, "#T")
      .replace(/\b\d{1,2}:\d{2}(:\d{2})?\s?(am|pm)?\b/gi, "#T")
      .replace(ID_RE, "#ID")
      .replace(/\s+/g, " ")
      .trim();

  // FNV-1a, 32 bit. Not cryptographic, only a change detector.
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

  // A fence's highlight state is scroll history: code-fence-defer renders unreached fences as a
  // plain shell and upgrades them once on approach. Each message therefore carries per-fence
  // latch state and a line-normalised text digest so analysis/parity.fence_latch_residue can
  // compare a fence latched on one arm only by language and text.
  const FENCE_ATTR = "data-streamdown";
  const FENCE_ROOT = "code-block";
  const FENCE_BODY = "code-block-body";
  const FENCE_DEFERRED_ATTR = "data-unsloth-fence-deferred";
  const FENCE_MARKER = "<!fence>";

  // One marker per line of a highlighted fence; the shell's <code> holds only text.
  const FENCE_LINE = "\u0001";
  const fenceText = (raw) => {
    const s = raw || "";
    const lines = s.includes(FENCE_LINE)
      ? s
          .split(FENCE_LINE)
          .filter((line, k) => k > 0 || line !== "")
          .map((line) => line.replace(/\r?\n/g, ""))
      : s.split(/\r?\n/);
    // Not normText: inside code, durations, timestamps and ids are content.
    return lines.join("\n").replace(/\n+$/, "");
  };

  // dropAttrs: attributes not compared at all. elide: elements replaced by a tag/role marker.
  // fences: optional output array; the returned parts are byte-identical either way.
  const walkParts = (root, dropAttrs, elide, fences) => {
    if (!root) return [];
    const parts = [];
    // Nested fences belong to their outer fence's reading.
    let fence = null;
    let inBody = false;
    const walk = (el, depth) => {
      // The depth-cap marker stays so deeper content reads as "not walked", not absent.
      if (depth > 40) { parts.push("<!depth-cap>"); return; }
      if (elide && elide.has(el)) {
        parts.push(
          "<!in-flight " + el.tagName.toLowerCase() +
          " role=" + ((el.getAttribute && el.getAttribute("data-role")) || "?") + ">"
        );
        return;
      }
      let opened = false;
      let bodyHere = false;
      if (fences && el.getAttribute) {
        const role = el.getAttribute(FENCE_ATTR);
        if (fence === null && role === FENCE_ROOT) {
          fence = {
            start: parts.length,
            latched: el.getAttribute(FENCE_DEFERRED_ATTR) !== "true",
            lang: el.getAttribute("data-language"),
            all: "",
            body: null,
          };
          opened = true;
        } else if (fence !== null && !inBody && role === FENCE_BODY) {
          if (fence.body === null) fence.body = "";
          inBody = true;
          bodyHere = true;
        }
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
      // Adjacent text nodes are joined before normalising: React splits `Thought for {n} seconds`
      // into three nodes and per-node normalisation would leak the wall-clock digit.
      let run = "";
      const flush = () => {
        if (!run) return;
        const t = normText(run);
        if (t) parts.push(t);
        run = "";
      };
      for (const child of el.childNodes) {
        if (child.nodeType === 3) {
          const v = child.nodeValue == null ? "" : child.nodeValue;
          run += v;
          if (fence !== null) {
            fence.all += v;
            if (inBody) fence.body += v;
          }
        } else if (child.nodeType === 1) {
          flush();
          if (fence !== null && inBody && el.tagName === "CODE") fence.body += FENCE_LINE;
          walk(child, depth + 1);
        }
      }
      flush();
      parts.push("</" + el.tagName.toLowerCase() + ">");
      if (bodyHere) inBody = false;
      if (opened) {
        // A highlighted fence also mounts a copy/download bar the shell does not, so use the body text.
        fences.push({
          start: fence.start,
          end: parts.length,
          latched: fence.latched,
          lang: fence.lang,
          text: fenceText(fence.body === null ? fence.all : fence.body),
        });
        fence = null;
      }
    };
    walk(root, 0);
    return parts;
  };

  const signature = (root, dropAttrs, elide) => walkParts(root, dropAttrs, elide).join("");

  // fences is omitted when a message has none, keeping fence-free payloads byte-identical.
  const messageReading = (el) => {
    const marks = [];
    const parts = walkParts(el, undefined, undefined, marks);
    const sig = parts.join("");
    const out = { sig };
    if (marks.length) {
      const rest = [];
      let at = 0;
      const fences = [];
      for (const m of marks) {
        for (let k = at; k < m.start; k++) rest.push(parts[k]);
        rest.push(FENCE_MARKER);
        at = m.end;
        fences.push({
          latched: m.latched,
          lang: m.lang,
          digest: hash(parts.slice(m.start, m.end).join("")),
          text: hash(m.text),
        });
      }
      for (let k = at; k < parts.length; k++) rest.push(parts[k]);
      out.fences = fences;
      out.digest_unfenced = hash(rest.join(""));
    }
    return out;
  };

  // Computed-style probe digested separately from the structure: getComputedStyle is the only
  // reading that sees CSS changes and the most likely to catch a transition mid-way.
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
    return { digest: hash(sig), chars: sig.length, elements: n, capped, props: STYLE_PROPS, sig };
  };

  const D = () => (window.__sb.dom || {});

  // Visible-region parity: every message visible at any point during the action must match on
  // both arms; differences off screen are allowed. Partial intersection counts as visible (false
  // alarms, never false passes). No geometry calls here: getBoundingClientRect/getClientRects
  // force Chromium to render content-visibility subtrees; IntersectionObserver does not.
  const VIS = {
    obs: null, mut: null, ever: new Set(), watching: false,
    // Only dedupes the diagnostic count; placement is re-read on every offer since rows recycle.
    unplacedSeen: new WeakSet(),
    // Rows with no aria-posinset and not among mounted messages; absent from ever_visible, so counted.
    unplaced: 0,
    index: null,
  };

  const ordinalOf = (el, position) => {
    // Position in the thread, not the mounted list; aria-posinset is the windowed arm's own claim.
    const owner = el.closest ? el.closest("[aria-posinset]") : null;
    if (owner) {
      const n = Number(owner.getAttribute("aria-posinset"));
      if (Number.isFinite(n)) return n;
    }
    return position;
  };

  // The fallback ordinal is the row's DOM position at observe time, not a lifetime counter:
  // thread_reopen rebuilds rows in one document. The index is built at most once per mutation
  // batch and only when needed, keeping O(document) work off the per-mutation path.
  const positionIndex = () => {
    if (VIS.index) return VIS.index;
    const nodes = (D().messages && D().messages()) || [];
    const map = new Map();
    for (let i = 0; i < nodes.length; i++) map.set(nodes[i], i + 1);
    VIS.index = map;
    return map;
  };

  const observeOne = (el, position) => {
    if (!VIS.obs) return;
    let ord = ordinalOf(el, typeof position === "number" ? position : null);
    if (ord === null) ord = positionIndex().get(el) || 0;
    // Stamped on the node: at delivery time it may be unmounted and closest() returns nothing.
    if (ord > 0) {
      // Re-read every time because a recycling virtualizer renumbers rows.
      const had = el.__sbOrdinal;
      el.__sbOrdinal = ord;
      if (typeof had === "number" && had !== ord) {
        // IntersectionObserver only delivers on intersection change; re-observing forces an initial entry.
        VIS.obs.unobserve(el);
      }
    } else {
      // No ordinal rather than a guessed one, and not marked seen, so a recycled row is not dropped later.
      if (!VIS.unplacedSeen.has(el)) {
        VIS.unplacedSeen.add(el);
        VIS.unplaced += 1;
      }
    }
    VIS.obs.observe(el);
  };

  // O(document), so it runs once, before the measured window opens.
  const observeAll = () => {
    if (!VIS.obs) return;
    const nodes = (D().messages && D().messages()) || [];
    for (let i = 0; i < nodes.length; i++) observeOne(nodes[i], i + 1);
  };

  // Walks only added nodes: a full rescan per batch charged an O(document) walk to the measured action.
  const observeAdded = (records) => {
    // Dropped after each batch so the next batch cannot read a stale position.
    VIS.index = null;
    for (const rec of records) {
      // A renumbered row arrives as its own attribute-record target, not in addedNodes.
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
        // No characterData: streamed text must not reach this callback. aria-posinset is watched
        // because a virtualizer renumbers rows in place without a childList mutation.
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
        // IntersectionObserver computes during rendering; with no frame in between takeRecords() is empty.
        await new Promise((resolve) => {
          requestAnimationFrame(() => requestAnimationFrame(resolve));
        });
        // Drain queued entries before teardown, or the action's last entries are lost.
        const records = VIS.obs.takeRecords();
        for (const entry of records) {
          if (entry.isIntersecting && typeof entry.target.__sbOrdinal === "number") {
            VIS.ever.add(entry.target.__sbOrdinal);
          }
        }
        const dom = D();
        const nodes = (dom.messages && dom.messages()) || [];
        // Same node set as the positive control below, read from one call so they cannot disagree.
        const live = new Set((dom.streamingMessages && dom.streamingMessages()) || []);
        const byOrdinal = {};
        let mounted_ever_visible = 0;
        // Counted directly: unmounted_at_capture can read 0 over a collision because errors cancel.
        let ordinal_collisions = 0;
        const collided = [];
        for (let i = 0; i < nodes.length; i++) {
          const el = nodes[i];
          const ord = typeof el.__sbOrdinal === "number" ? el.__sbOrdinal : ordinalOf(el, i + 1);
          if (!VIS.ever.has(ord)) continue;
          mounted_ever_visible += 1;
          // Without aria-posinset/aria-setsize, which readiness.py permits on either the message or a row;
          // otherwise windowed vs fully mounted arms differ on every message.
          const sig = signature(el, VIRTUALIZATION_ATTRS);
          if (Object.prototype.hasOwnProperty.call(byOrdinal, String(ord))) {
            ordinal_collisions += 1;
            if (collided.indexOf(ord) === -1) collided.push(ord);
          }
          byOrdinal[String(ord)] = {
            role: el.getAttribute("data-role") || "?",
            digest: hash(sig),
            chars: sig.length,
            // Still streaming at capture time: residue, never a difference.
            in_flight: live.has(el),
          };
        }
        const ever = [...VIS.ever].sort((a, b) => a - b);
        // Read globally, not over visible rows: a reply streaming below the fold must not refuse anything.
        const generating = dom.generating
          ? Boolean(dom.generating())
          : Boolean(dom.isRunning && dom.isRunning());
        const lastPublishes = dom.lastAssistantPublishesStatus
          ? Boolean(dom.lastAssistantPublishesStatus())
          : false;
        const anyPublishes = dom.statusHookPresent ? Boolean(dom.statusHookPresent()) : true;
        const windowedArm = dom.isWindowed ? Boolean(dom.isWindowed()) : false;
        const probeBlind = lastPublishes ? !windowedArm : !anyPublishes;
        return {
          visible_attempted: true,
          streaming: generating,
          status_hook_present: anyPublishes,
          // A windowed arm can unmount the streaming message, so live.size can be 0 with intact hooks.
          in_flight_unplaced: Boolean(generating && live.size === 0 && probeBlind),
          ever_visible: ever,
          ever_visible_count: ever.length,
          mounted_ever_visible,
          unmounted_at_capture: ever.length - mounted_ever_visible,
          unplaced_rows: VIS.unplaced,
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
    // Exposed so the offline unit tests drive the shipped regexes.
    normText,
    normUrl,
    signature,
    messageReading,
    hash,

    // Per-message digests say which message moved. opts.raw returns the large signature text.
    capture(opts) {
      const want_raw = Boolean(opts && opts.raw);
      try {
        const dom = D();
        const rootFn = dom.threadRoot;
        const found = rootFn ? rootFn() : null;
        // threadRoot() falls back to document.body; which root was digested must travel with the digest.
        const isThread = Boolean(found && found !== document.body &&
                                 found.classList && found.classList.contains("aui-thread-root"));
        const root = found || document.body;
        const nodes = (dom.messages && dom.messages()) || [];
        // The in-flight message has no defined moment (each arm samples a different point of the same
        // reply, and remend/KaTeX/Shiki output is not monotonic), so it is elided into a marker. The
        // scaffold elides every message, since in-flight status differs per arm.
        const inFlightNodes = (dom.streamingMessages && dom.streamingMessages()) || [];
        const inFlight = new Set(inFlightNodes);
        const running = Boolean(dom.isRunning && dom.isRunning());
        // isRunning() is also true while a prompt is queued on an idle thread; generating() is narrower.
        const generating = dom.generating ? Boolean(dom.generating()) : running;
        // A gone hook is a blind instrument; a present hook with nothing running is a settled thread.
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
          const reading = messageReading(nodes[i]);
          const sig = reading.sig;
          const row = {
            i,
            role: nodes[i].getAttribute("data-role") || "?",
            digest: hash(sig),
            chars: sig.length,
          };
          if (reading.fences) {
            row.fences = reading.fences;
            row.digest_unfenced = reading.digest_unfenced;
          }
          if (inFlight.has(nodes[i])) {
            row.in_flight = true;
            in_flight.push(i);
          }
          if (want_raw) row.raw = sig;
          messages.push(row);
        }
        // Overlays outside the thread root (dialogs, menus, model picker) the digest above cannot see.
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
          // digest = scaffold + per-message rows, so one message can be withheld without losing the rest.
          digest_scaffold: hash(scaffold),
          chars_scaffold: scaffold.length,
          in_flight,
          streaming: running,
          // Positive control: a renamed data-status hook makes streamingMessages() silently empty, so a
          // disagreement with generating() is carried out and analysis/parity.comparability refuses it.
          in_flight_unplaced: Boolean(
            generating && inFlightNodes.length === 0 && probeBlind
          ),
          status_hook_present: anyPublishes,
          // Includes the dispatched wait ("Stop queued message"), which isRunning() does not match.
          queued_idle: Boolean(
            (running || (dom.stopQueuedButton ? Boolean(dom.stopQueuedButton()) : false)) &&
              !generating
          ),
          // See dom.runStateControl and analysis/parity.generation_disagrees.
          composer_control: dom.runStateControl ? dom.runStateControl() : null,
          messages,
          overlays,
          styles,
          // Rows are keyed by mounted index, which equals conversation position only when fully mounted.
          mounted_messages: messages.length,
          thread_total: (dom.threadTotal && dom.threadTotal()) || messages.length,
        };
        if (want_raw) out.raw = whole;
        return out;
      } catch (err) {
        // An empty digest on error would read as "everything matched".
        return { parity_attempted: false, reason: String(err && err.message ? err.message : err) };
      }
    },
  };
})();
