// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
// Selector adapter for surfaces outside the chat thread. Digests go through `window.__sb.parity.capture()` so they stay comparable with the film's actions.

(() => {
  if (window.__sb && window.__sb.surfaces) return;
  window.__sb = window.__sb || {};

  const q = (sel) => {
    // One bad selector must cost only its surface, not the whole sweep.
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

  const isVisible = (el) => {
    if (!el) return false;
    if (el.hasAttribute && el.hasAttribute("hidden")) return false;
    const rect = el.getBoundingClientRect();
    if (rect.width <= 0 && rect.height <= 0) return false;
    const style = window.getComputedStyle(el);
    return style.display !== "none" && style.visibility !== "hidden";
  };

  // The active route container is the visible sibling of an `inert` one; keyed on `inert`, not the
  // Tailwind `hidden` class. With nothing inert the inset is the container.
  const routeContainer = () => {
    const inset = q('[data-slot="sidebar-inset"]');
    if (!inset) return null;
    const inertNode = inset.querySelector("[inert]");
    if (!inertNode || !inertNode.parentElement) return inset;
    const siblings = Array.from(inertNode.parentElement.children);
    const active = siblings.filter(
      (el) => !el.hasAttribute("inert") && isVisible(el),
    );
    // Exactly one, or fall back to the inset rather than pick an arbitrary one.
    return active.length === 1 ? active[0] : inset;
  };

  const SPECIAL = {
    "@route": routeContainer,
    "@shell": () => q('[data-slot="sidebar-wrapper"]') || q("main"),
    "@sidebar": () =>
      q('[data-slot="sidebar-container"]') || q('[data-slot="sidebar"]'),
  };

  const resolve = (sel) => (SPECIAL[sel] ? SPECIAL[sel]() : q(sel));

  const S = {
    // First present and visible wins; auth routes have no sidebar, so lists end with a fallback.
    resolveRoot(candidates) {
      for (const sel of candidates || []) {
        const el = resolve(sel);
        if (el && isVisible(el)) return { el, sel };
      }
      for (const sel of candidates || []) {
        const el = resolve(sel);
        if (el) return { el, sel, visible: false };
      }
      return { el: document.body, sel: "body", fallback: true };
    },

    capture(candidates) {
      const parity = (window.__sb || {}).parity;
      if (!parity || typeof parity.capture !== "function") {
        return {
          parity_attempted: false,
          reason: "parity.js is not loaded on this page",
        };
      }
      const dom = (window.__sb || {}).dom;
      if (!dom || typeof dom.threadRoot !== "function") {
        return {
          parity_attempted: false,
          reason: "dom.js is not loaded on this page",
        };
      }
      const found = S.resolveRoot(candidates);
      const original = dom.threadRoot;
      let out;
      try {
        dom.threadRoot = () => found.el;
        out = parity.capture();
      } catch (err) {
        out = {
          parity_attempted: false,
          reason: String(err && err.message ? err.message : err),
        };
      } finally {
        dom.threadRoot = original;
      }
      out.root_selector = found.sel;
      out.root_visible = found.visible !== false;
      out.root_is_fallback = Boolean(found.fallback);
      return out;
    },

    // An honouring capture of a detached one-text-node root returns a few dozen chars, else the page.
    probeScoping() {
      const dom = (window.__sb || {}).dom;
      const parity = (window.__sb || {}).parity;
      if (!dom || !parity) {
        return {
          scoped: false,
          scoping_attempted: false,
          reason: "dom.js or parity.js is not loaded on this page",
        };
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
        return {
          scoped: false,
          scoping_attempted: true,
          reason:
            "parity.capture() raised while the root was moved: " +
            String(err && err.message ? err.message : err),
        };
      }
      dom.threadRoot = original;
      const chars = got && typeof got.chars === "number" ? got.chars : null;
      if (chars === null) {
        return {
          scoped: false,
          scoping_attempted: true,
          probe_chars: -1,
          reason:
            "parity.capture() returned no `chars`, so scoping cannot be verified",
        };
      }
      if (chars > 500) {
        return {
          scoped: false,
          scoping_attempted: true,
          probe_chars: chars,
          reason:
            "parity.capture() ignored the moved root (" +
            chars +
            " chars from a " +
            "detached probe element), so surface digests would not be scoped",
        };
      }
      return { scoped: true, scoping_attempted: true, probe_chars: chars };
    },

    settled(spec) {
      if (!spec) return { ok: true, detail: "no settle condition" };
      if (spec.visible) {
        const el = q(spec.visible);
        return {
          ok: Boolean(el) && isVisible(el),
          detail: el ? "present, visible=" + isVisible(el) : "not present",
        };
      }
      if (spec.hidden) {
        const el = q(spec.hidden);
        return {
          ok: !el || !isVisible(el),
          detail: el ? "still visible" : "gone",
        };
      }
      if (spec.count_at_least) {
        const [sel, n] = spec.count_at_least;
        const got = qa(sel).filter(isVisible).length;
        return { ok: got >= n, detail: got + " visible, wanted " + n };
      }
      if (spec.text) {
        const hay = document.body.innerText || "";
        return {
          ok: hay.includes(spec.text),
          detail: "text " + JSON.stringify(spec.text),
        };
      }
      if (spec.js) {
        try {
          const got = Function("return (" + spec.js + ")")();
          return { ok: Boolean(got), detail: "js -> " + String(got) };
        } catch (err) {
          return {
            ok: false,
            detail: "js raised: " + String(err && err.message),
          };
        }
      }
      return {
        ok: true,
        detail: "unrecognised settle spec, treated as satisfied",
      };
    },

    facts(candidates) {
      const found = S.resolveRoot(candidates);
      return {
        facts_attempted: true,
        pathname: location.pathname,
        search: location.search,
        root_selector: found.sel,
        root_is_fallback: Boolean(found.fallback),
        root_elements: found.el
          ? found.el.getElementsByTagName("*").length
          : -1,
        root_text_chars: found.el ? (found.el.innerText || "").length : -1,
        open_dialogs: qa(
          '[role="dialog"], [data-slot="dialog-content"]',
        ).filter(isVisible).length,
        open_menus: qa('[role="menu"], [role="listbox"]').filter(isVisible)
          .length,
        popovers: qa("[data-radix-popper-content-wrapper]").filter(isVisible)
          .length,
      };
    },

    isClean() {
      return {
        clean_attempted: true,
        open_dialogs: qa(
          '[role="dialog"], [data-slot="dialog-content"]',
        ).filter(isVisible).length,
        open_menus: qa('[role="menu"], [role="listbox"]').filter(isVisible)
          .length,
        popovers: qa("[data-radix-popper-content-wrapper]").filter(isVisible)
          .length,
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
