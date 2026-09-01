// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
//
// The selector adapter for Unsloth's surfaces OUTSIDE the chat thread: the other routes, the
// settings dialog's twelve tabs, the sidebar and its menus, the hub, the media pages.
//
// Separate from dom.js on purpose. dom.js is the chat thread's adapter and the eighteen film
// actions all read it; this file is read only by the surface sweep, so a selector added for a
// route cannot break an action.
//
// THE TRAP THIS FILE EXISTS TO GUARD. ChatPage, ImagesPage, VideoPage and AudioPage are mounted
// PERSISTENTLY by the root layout (`shouldMountChat` and friends in app/routes/__root.tsx) so an
// in-flight generation survives leaving the tab. Off-route they are `class="hidden"` and `inert`,
// but they are still in the document. So on /hub, `document.querySelector(".aui-thread-root")`
// still returns the chat thread, and a digest taken with parity.js's default root would digest
// the HIDDEN CHAT THREAD on every route in the sweep -- forty surfaces reporting one identical
// digest, and every one of them reading as a pass. Every surface therefore names its own root and
// the digest is scoped to it.
//
// HOW THE SCOPING IS DONE, and why it is not a second digest implementation. Surface digests are
// only worth taking if they are comparable with the film's action digests, which means the same
// normalisation and the same hash -- so this calls `window.__sb.parity.capture()` rather than
// walking the DOM itself. parity.js reads its root from `window.__sb.dom.threadRoot()`, so the
// root is moved for the duration of one capture and put back afterwards. `probeScoping()` below
// checks that the move is actually observed, because if a future parity.js stopped reading
// threadRoot this would silently go back to digesting the thread on every surface, which is the
// failure this file was written to prevent.

(() => {
  if (window.__sb && window.__sb.surfaces) return;
  window.__sb = window.__sb || {};

  const q = (sel) => {
    // Selector strings come from a registry, and one bad selector must cost that surface a
    // reason rather than cost the whole sweep an exception.
    try {
      return document.querySelector(sel);
    } catch (err) {
      return null;
    }
  };

  const qa = (sel) => {
    try {
      return Array.from(document.querySelectorAll(sel));
    } catch (err) {
      return [];
    }
  };

  // Rendered, not merely mounted. `hidden`, `display:none` and a zero box all mean the user
  // cannot see it, and the keep-alive route containers are exactly the first case.
  const isVisible = (el) => {
    if (!el) return false;
    if (el.hasAttribute && el.hasAttribute("hidden")) return false;
    const rect = el.getBoundingClientRect();
    if (rect.width <= 0 && rect.height <= 0) return false;
    const style = window.getComputedStyle(el);
    return style.display !== "none" && style.visibility !== "hidden";
  };

  // The ACTIVE route's container, found by the property the root layout actually sets.
  //
  // The layout renders the keep-alive pages and the routed page as siblings, and marks every
  // page that is NOT the current route `inert` (plus `class="hidden"`). So the active container
  // is the one visible sibling of an inert one. Keyed on `inert` rather than on the class string
  // because `hidden` is a Tailwind utility that a restyle can change, while `inert` is what makes
  // the off-route pages unreachable and cannot be dropped without changing behaviour.
  //
  // When nothing is inert, no page has been kept alive yet and the inset itself is the container.
  const routeContainer = () => {
    const inset = q('[data-slot="sidebar-inset"]');
    if (!inset) return null;
    const inertNode = inset.querySelector("[inert]");
    if (!inertNode || !inertNode.parentElement) return inset;
    const siblings = Array.from(inertNode.parentElement.children);
    const active = siblings.filter((el) => !el.hasAttribute("inert") && isVisible(el));
    // Exactly one, or the assumption is wrong and the inset is the honest answer. Silently
    // picking the first of several would scope the digest to an arbitrary part of the page.
    return active.length === 1 ? active[0] : inset;
  };

  const SPECIAL = {
    "@route": routeContainer,
    "@shell": () => q('[data-slot="sidebar-wrapper"]') || q("main"),
    "@sidebar": () => q('[data-slot="sidebar-container"]') || q('[data-slot="sidebar"]'),
  };

  const resolve = (sel) => (SPECIAL[sel] ? SPECIAL[sel]() : q(sel));

  const S = {
    // ── roots ────────────────────────────────────────────────────
    //
    // `candidates` is ordered: the first one that is present AND visible wins. The auth-flow
    // routes render no sidebar wrapper at all, so every list ends at a fallback that exists.
    resolveRoot(candidates) {
      for (const sel of candidates || []) {
        const el = resolve(sel);
        if (el && isVisible(el)) return { el, sel };
      }
      for (const sel of candidates || []) {
        const el = resolve(sel);
        // Present but not visible is still reported, with the selector, so a surface that
        // rendered into a hidden container is a readable finding rather than a fallback to body.
        if (el) return { el, sel, visible: false };
      }
      return { el: document.body, sel: "body", fallback: true };
    },

    // ── the scoped capture ───────────────────────────────────────
    capture(candidates) {
      const parity = (window.__sb || {}).parity;
      if (!parity || typeof parity.capture !== "function") {
        return { parity_attempted: false, reason: "parity.js is not loaded on this page" };
      }
      const dom = (window.__sb || {}).dom;
      if (!dom || typeof dom.threadRoot !== "function") {
        return { parity_attempted: false, reason: "dom.js is not loaded on this page" };
      }
      const found = S.resolveRoot(candidates);
      const original = dom.threadRoot;
      let out;
      try {
        dom.threadRoot = () => found.el;
        out = parity.capture();
      } catch (err) {
        out = { parity_attempted: false,
                reason: String(err && err.message ? err.message : err) };
      } finally {
        dom.threadRoot = original;
      }
      out.root_selector = found.sel;
      out.root_visible = found.visible !== false;
      out.root_is_fallback = Boolean(found.fallback);
      return out;
    },

    // Does parity.capture() actually honour a moved root? Pointed at a detached element with one
    // short text node, an honouring capture returns a signature of a few dozen characters and a
    // non-honouring one returns the whole page. The threshold is two orders of magnitude clear of
    // both, so this cannot answer "yes" for a page that ignored the move.
    probeScoping() {
      const dom = (window.__sb || {}).dom;
      const parity = (window.__sb || {}).parity;
      if (!dom || !parity) {
        return { scoped: false, scoping_attempted: false,
                 reason: "dom.js or parity.js is not loaded on this page" };
      }
      const probe = document.createElement("div");
      probe.setAttribute("data-sb-scope-probe", "1");
      probe.textContent = "scope probe";
      const original = dom.threadRoot;
      let got;
      try {
        dom.threadRoot = () => probe;
        got = parity.capture();
      } catch (err) {
        dom.threadRoot = original;
        return { scoped: false, scoping_attempted: true,
                 reason: "parity.capture() raised while the root was moved: " +
                         String(err && err.message ? err.message : err) };
      }
      dom.threadRoot = original;
      const chars = got && typeof got.chars === "number" ? got.chars : null;
      if (chars === null) {
        return { scoped: false, scoping_attempted: true, probe_chars: -1,
                 reason: "parity.capture() returned no `chars`, so scoping cannot be verified" };
      }
      if (chars > 500) {
        // The digest is the whole page, not the probe. Every surface digest would then be the
        // same page-wide reading and the sweep would report forty identical passes.
        return { scoped: false, scoping_attempted: true, probe_chars: chars,
                 reason: "parity.capture() ignored the moved root (" + chars + " chars from a " +
                         "detached probe element), so surface digests would not be scoped" };
      }
      return { scoped: true, scoping_attempted: true, probe_chars: chars };
    },

    // ── settle predicates ────────────────────────────────────────
    //
    // Evaluated by the sweep in a poll loop. Each returns a plain boolean plus the observation it
    // was made from, so a surface that never settled records WHAT it was waiting for.
    settled(spec) {
      if (!spec) return { ok: true, detail: "no settle condition" };
      if (spec.visible) {
        const el = q(spec.visible);
        return { ok: Boolean(el) && isVisible(el),
                 detail: el ? "present, visible=" + isVisible(el) : "not present" };
      }
      if (spec.hidden) {
        const el = q(spec.hidden);
        return { ok: !el || !isVisible(el), detail: el ? "still visible" : "gone" };
      }
      if (spec.count_at_least) {
        const [sel, n] = spec.count_at_least;
        const got = qa(sel).filter(isVisible).length;
        return { ok: got >= n, detail: got + " visible, wanted " + n };
      }
      if (spec.text) {
        const hay = (document.body.innerText || "");
        return { ok: hay.includes(spec.text), detail: "text " + JSON.stringify(spec.text) };
      }
      if (spec.js) {
        try {
          // eslint-disable-next-line no-new-func
          const got = Function("return (" + spec.js + ")")();
          return { ok: Boolean(got), detail: "js -> " + String(got) };
        } catch (err) {
          return { ok: false, detail: "js raised: " + String(err && err.message) };
        }
      }
      return { ok: true, detail: "unrecognised settle spec, treated as satisfied" };
    },

    // What the sweep records alongside every surface so a digest can be read against the state it
    // was taken in. A surface whose root holds three elements did not render.
    facts(candidates) {
      const found = S.resolveRoot(candidates);
      return {
        facts_attempted: true,
        pathname: location.pathname,
        search: location.search,
        root_selector: found.sel,
        root_is_fallback: Boolean(found.fallback),
        root_elements: found.el ? found.el.getElementsByTagName("*").length : -1,
        root_text_chars: found.el ? (found.el.innerText || "").length : -1,
        open_dialogs: qa('[role="dialog"], [data-slot="dialog-content"]').filter(isVisible).length,
        open_menus: qa('[role="menu"], [role="listbox"]').filter(isVisible).length,
        popovers: qa("[data-radix-popper-content-wrapper]").filter(isVisible).length,
      };
    },

    // The known state the sweep returns to between surfaces. Reported rather than asserted: the
    // sweep decides what to do about a dirty state, and it needs the observation to decide.
    isClean() {
      return {
        clean_attempted: true,
        open_dialogs: qa('[role="dialog"], [data-slot="dialog-content"]').filter(isVisible).length,
        open_menus: qa('[role="menu"], [role="listbox"]').filter(isVisible).length,
        popovers: qa("[data-radix-popper-content-wrapper]").filter(isVisible).length,
        pathname: location.pathname,
      };
    },

    visible(sel) {
      return isVisible(q(sel));
    },

    count(sel) {
      return qa(sel).filter(isVisible).length;
    },
  };

  window.__sb.surfaces = S;
})();
