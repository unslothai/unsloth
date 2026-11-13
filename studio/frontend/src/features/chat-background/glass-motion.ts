// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/*
 * Under software compositing every frame redraws the window and recomputes every backdrop-filter, the
 * still sidebar's included. So while something moves, or keys and presses arrive, the moving surface and the standing glass carry
 * data-glass-moving, which index.css turns into a near-opaque fill; the blur returns after SETTLE_MS.
 * The flag sits on each surface and only that surface's rules read it, so it never restyles the chat.
 */

/** Glass surfaces that can move: the composer, menus, popovers, tooltips, dialogs and floating buttons. */
export const GLASS_MOTION_SURFACES = [
  ".unsloth-composer-surface",
  ".unsloth-files-panel",
  '[data-slot$="menu-content"]',
  '[data-slot$="menu-sub-content"]',
  '[data-slot="select-content"]',
  '[data-slot="combobox-content"]',
  '[data-slot="popover-content"]',
  '[data-slot="hover-card-content"]',
  ".menu-soft-surface",
  ".menu-soft-surface-up",
  ".dropdown-surface",
  ".tooltip-compact",
  ".tooltip-rich",
  '[data-slot="dialog-content"]',
  "[data-glass-float]",
  ".aui-thread-scroll-to-bottom",
].join(", ");

/** Surfaces that pause only themselves: hovering a control shows them, which should not flicker the sidebar. */
const LOCAL_ONLY = '.tooltip-compact, .tooltip-rich, [data-slot="hover-card-content"]';
/** The page's standing glass, paused with anything that moves. */
const PAGE_SIDEBAR = '[data-slot="sidebar-inner"]';
const PAGE_FRAME = "[data-windows-glass-frame]";
const PAGE_CLASSES = ["unsloth-composer-surface", "unsloth-files-panel"];
/** The standing glass is looked up again at most this often (or when one of them left the page). */
const PAGE_REFRESH_MS = 2000;
const THREAD_SCROLLER = ".aui-thread-viewport";
const TEXT_FIELDS = 'textarea, input, [contenteditable]:not([contenteditable="false"])';

/** Keys, presses and edits (paste, IME, autocomplete) each draw frames: the glass pauses through a burst of them. */
export const INTERACTION_EVENTS = ["keydown", "pointerdown", "input"] as const;

export const GLASS_MOVING_ATTR = "data-glass-moving";
export const GLASS_SETTLING_ATTR = "data-glass-settling";
/** How long transitions stay off after the blur returns: a few frames, so its style is resolved without one. */
export const SETTLING_MS = 100;
/** How long a surface must be still before its blur comes back. */
export const SETTLE_MS = 200;
/** A surface that just mounted: its open animation (100-200 ms) plus SETTLE_MS. */
export const MOUNT_HOLD_MS = 350;
/** Upper bound while an animation runs whose end event never arrives (an element removed mid-animation). */
export const ANIMATION_HOLD_MS = 1000;

type Timer = ReturnType<typeof setTimeout>;

export interface GlassMotion {
  /** Mark `el` moving for `ms` from now (extends a hold already running). Idempotent. */
  hold(el: Element, ms: number): void;
  /** Surfaces a scroll of `scroller` moves: the surface holding it and the glass floating over it (null: typing). */
  scrollTargets(scroller: Element): Element[] | null;
  stop(): void;
}

let active: GlassMotion | null = null;

/** Pause `el` for `ms`: a surface animated from script fires no animation events. No-op while glass motion is off. */
export function holdGlassMotion(el: Element | null, ms: number): void {
  if (el && active) active.hold(el, ms);
}

function isElement(node: unknown): node is Element {
  return typeof Element !== "undefined" && node instanceof Element;
}

/** Start pausing glass blur during motion; returns the controller (stop() removes every listener and flag). */
export function startGlassMotion(doc: Document = document): GlassMotion {
  const timers = new Map<Element, Timer>();
  const settling = new Map<Element, Timer>();
  const settled = (el: Element) => {
    settling.delete(el);
    if (el.hasAttribute(GLASS_SETTLING_ATTR)) el.removeAttribute(GLASS_SETTLING_ATTR);
  };
  // The blur comes back in one frame: for that frame the surface's own transitions are off (data-glass-settling),
  // or a dialog's transition-all would fade the blur in over 100 ms and compute it again on every one of those frames.
  const release = (el: Element) => {
    timers.delete(el);
    until.delete(el);
    if (!el.hasAttribute(GLASS_MOVING_ATTR)) return;
    el.setAttribute(GLASS_SETTLING_ATTR, "");
    el.removeAttribute(GLASS_MOVING_ATTR);
    const prev = settling.get(el);
    if (prev !== undefined) clearTimeout(prev);
    settling.set(el, setTimeout(() => settled(el), SETTLING_MS));
  };
  const until = new Map<Element, number>();
  // `extend`: only ever lengthen the hold, for glass that several motions may be holding at once.
  const hold = (el: Element, ms: number, extend = false) => {
    const end = performance.now() + ms;
    if (extend && (until.get(el) ?? Number.NEGATIVE_INFINITY) >= end) return;
    const prev = timers.get(el);
    if (prev !== undefined) clearTimeout(prev);
    // Only write on a change: an attribute write restyles the element even when the value is the same.
    if (!el.hasAttribute(GLASS_MOVING_ATTR)) el.setAttribute(GLASS_MOVING_ATTR, "");
    until.set(el, end);
    timers.set(el, setTimeout(() => release(el), ms));
  };

  // The native macOS material keeps its sidebar: there the window itself is the frosted layer.
  let page: Element[] = [];
  let pageAt = Number.NEGATIVE_INFINITY;
  const pageGlass = (): Element[] => {
    const now = performance.now();
    if (now - pageAt < PAGE_REFRESH_MS && page.every((el) => el.isConnected)) return page;
    page = [];
    pageAt = now;
    if (doc.documentElement?.getAttribute("data-glass") !== "macos") {
      const sidebar = doc.querySelector(PAGE_SIDEBAR);
      if (sidebar) {
        page.push(sidebar);
        const frame = sidebar.closest(PAGE_FRAME);
        if (frame) page.push(frame);
      }
    }
    for (const name of PAGE_CLASSES) for (const el of doc.getElementsByClassName(name)) page.push(el);
    return page;
  };
  /** Pause `el` (a surface that moves) for `ms` and, unless it is a tooltip, the page's standing glass for
   *  `pageMs` with it; the standing glass is only ever held longer, since other motion may be holding it too. */
  const holdMotion = (el: Element | null, ms: number, pageMs = ms) => {
    if (el) hold(el, ms);
    if (el && el.matches(LOCAL_ONLY)) return;
    for (const still of pageGlass()) if (still !== el) hold(still, pageMs, true);
  };

  const scrollTargets = (scroller: Element): Element[] | null => {
    // Text scrolling inside a field is typing, not motion.
    if (scroller.matches(TEXT_FIELDS)) return null;
    const out: Element[] = [];
    const own = scroller.closest(GLASS_MOTION_SURFACES);
    if (own) out.push(own);
    // The glass floating over the chat (the composer dock, the scroll-to-bottom button) sits beside the
    // thread viewport. Only those siblings' own subtrees are searched, never the viewport's.
    const parent = scroller.parentElement;
    if (parent && !own && scroller.matches(THREAD_SCROLLER)) {
      for (const sibling of parent.children) {
        if (sibling === scroller) continue;
        if (sibling.matches(GLASS_MOTION_SURFACES)) out.push(sibling);
        for (const el of sibling.querySelectorAll(GLASS_MOTION_SURFACES)) out.push(el);
      }
    }
    return out;
  };

  // Scrolls fire every frame; the targets are looked up once per scroller per gesture.
  let lastScroller: Element | null = null;
  let lastTargets: Element[] | null = null;
  let lastScrollAt = 0;
  const onScroll = (event: Event) => {
    const target = event.target;
    const scroller = isElement(target)
      ? target
      : target === doc
        ? doc.scrollingElement
        : null;
    if (!scroller) return;
    const now = performance.now();
    if (scroller !== lastScroller || now - lastScrollAt > SETTLE_MS) {
      lastScroller = scroller;
      lastTargets = scrollTargets(scroller);
    }
    lastScrollAt = now;
    if (!lastTargets) return;
    for (const el of lastTargets) hold(el, SETTLE_MS);
    holdMotion(null, SETTLE_MS);
  };

  // A key or a press redraws the window like any motion: typing a character, pasting, an IME step or a click
  // that opens a reasoning block would otherwise compute every blur on screen again for each of its frames.
  // The standing glass, and the surface the event lands in (a dialog's search field, a menu), pause with it.
  const onInteraction = (event: Event) => {
    const target = event.target;
    const own = isElement(target) ? target.closest(GLASS_MOTION_SURFACES) : null;
    holdMotion(own, SETTLE_MS);
  };

  const onAnimationStart = (event: Event) => {
    const el = event.target;
    if (isElement(el) && el.matches(GLASS_MOTION_SURFACES)) holdMotion(el, ANIMATION_HOLD_MS, MOUNT_HOLD_MS);
  };
  // The animation is over: everything it paused gets its blur back SETTLE_MS from now.
  const onAnimationEnd = (event: Event) => {
    const el = event.target;
    if (isElement(el) && timers.has(el)) holdMotion(el, SETTLE_MS);
  };

  // Menus, dialogs and tooltips mount into portals on <body>. Marking them before their first frame keeps
  // even that frame (the costliest: the blur is created) free of the blur. Only body's own children are
  // watched, not the subtree, so streaming chat text never wakes this.
  const onMutations = (records: MutationRecord[]) => {
    for (const record of records) {
      for (const node of record.addedNodes) {
        if (!isElement(node)) continue;
        const surfaces = [...node.querySelectorAll(GLASS_MOTION_SURFACES)];
        if (node.matches(GLASS_MOTION_SURFACES)) surfaces.unshift(node);
        for (const el of surfaces) holdMotion(el, MOUNT_HOLD_MS);
      }
    }
  };
  const observer =
    typeof MutationObserver !== "undefined" ? new MutationObserver(onMutations) : null;
  if (observer && doc.body) observer.observe(doc.body, { childList: true });

  const opts = { capture: true, passive: true } as const;
  doc.addEventListener("scroll", onScroll, opts);
  for (const type of INTERACTION_EVENTS) doc.addEventListener(type, onInteraction, opts);
  doc.addEventListener("animationstart", onAnimationStart, opts);
  doc.addEventListener("animationend", onAnimationEnd, opts);
  doc.addEventListener("animationcancel", onAnimationEnd, opts);

  const stop = () => {
    if (active === controller) active = null;
    observer?.disconnect();
    doc.removeEventListener("scroll", onScroll, opts);
    for (const type of INTERACTION_EVENTS) doc.removeEventListener(type, onInteraction, opts);
    doc.removeEventListener("animationstart", onAnimationStart, opts);
    doc.removeEventListener("animationend", onAnimationEnd, opts);
    doc.removeEventListener("animationcancel", onAnimationEnd, opts);
    until.clear();
    for (const [el, timer] of timers) {
      clearTimeout(timer);
      timers.delete(el);
      if (el.hasAttribute(GLASS_MOVING_ATTR)) el.removeAttribute(GLASS_MOVING_ATTR);
    }
    for (const [el, timer] of settling) {
      clearTimeout(timer);
      settled(el);
    }
    lastScroller = null;
    lastTargets = null;
    page = [];
  };
  const controller: GlassMotion = { hold, scrollTargets, stop };
  active = controller;
  return controller;
}
