// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Marks message rows far from the reader so index.css can skip their layout: a narrower chat
// column (the browser pane halves it) then re-wraps only the rows near the viewport. A row is
// marked two frames after the observer finds it distant, once it has rendered, so the engine
// sizes it by its remembered height (contain-intrinsic-size: auto) rather than a guess.
export const ROW_DISTANT_ATTRIBUTE = "data-row-distant";
export const ROW_REACH_PX = 1500;
const GATE_ATTRIBUTE = "data-message-row-containment";
const ROW_SELECTOR = "[data-role][data-message-id]";
const NUMBERED_EQUATION = ".eqn-num, .mml-eqn-num";

/**
 * The reach must exceed the engine's own content-visibility: auto margin, so a row is released
 * (and the reader compensated) before the engine lays it out on its own. Chromium renders a skipped
 * element within 150% of the window's height of the window (kViewportMarginPercentage in
 * third_party/blink/renderer/core/display_lock/display_lock_document_state.cc); WebKit only once
 * it intersects the window (no root margin, Source/WebCore/dom/ContentVisibilityDocumentState.cpp).
 * Twice the window height covers Chromium's margin plus the thread scroller's offset inside it.
 */
export const rowReachPx = (windowHeight: number): number =>
  Math.max(ROW_REACH_PX, Math.ceil(2 * (Number.isFinite(windowHeight) ? windowHeight : 0)));

type Adjust = (deltaPx: number) => unknown;
type Watch = { observer: IntersectionObserver; reach: number; rows: Set<Element> };

/** Iterated on resize; an entry leaves with its last row. */
const watches = new Map<Element, Watch>();
const adjusters = new WeakMap<Element, Adjust>();

/** Rows are stacked in document order, so the first one reaching the fold is a binary search. */
function firstVisibleRow(viewport: Element): { anchor: Element; before: number } | null {
  const rows = viewport.querySelectorAll(ROW_SELECTOR);
  const fold = viewport.getBoundingClientRect().top;
  let lo = 0;
  let hi = rows.length - 1;
  let found = -1;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (rows[mid].getBoundingClientRect().bottom > fold) {
      found = mid;
      hi = mid - 1;
    } else lo = mid + 1;
  }
  if (found < 0) return null;
  return { anchor: rows[found], before: rows[found].getBoundingClientRect().top };
}

/** Toggle rows, keeping the first visible row where the reader left it. A released row lays out at
 *  the current width, which may differ from the remembered one (the column changed while it was
 *  skipped); a newly skipped row should not move, but if it does the reader must not see it either. */
function toggleRows(viewport: Element, changes: Array<[Element, boolean]>) {
  const known = changes.length > 0 ? firstVisibleRow(viewport) : null;
  for (const [row, on] of changes) row.toggleAttribute(ROW_DISTANT_ATTRIBUTE, on);
  if (!known || !known.anchor.isConnected) return;
  // Zero when the engine anchored natively, so this never corrects twice.
  const delta = known.anchor.getBoundingClientRect().top - known.before;
  if (Math.abs(delta) < 0.5) return;
  // Through the autoscroll owner, which re-pins a following reader instead of reading a scroll-up.
  const owner = adjusters.get(viewport);
  if (owner) owner(delta);
  else viewport.scrollTo({ top: viewport.scrollTop + delta, behavior: "instant" });
}

const wanted = new WeakMap<Element, boolean>();
const settling = new WeakMap<Element, Set<Element>>();

/** Skip a row only after it has stayed distant for two frames, so the engine has recorded the
 *  size it was laid out at (a chunk mounted above the reader lays out once, then settles). */
function settleLater(viewport: Element, rows: Element[]) {
  let pending = settling.get(viewport);
  const scheduled = pending !== undefined;
  if (!pending) {
    pending = new Set();
    settling.set(viewport, pending);
  }
  for (const row of rows) pending.add(row);
  if (scheduled) return;
  requestAnimationFrame(() => requestAnimationFrame(() => {
    const due = settling.get(viewport);
    settling.delete(viewport);
    if (!due) return;
    const changes: Array<[Element, boolean]> = [];
    for (const row of due) {
      if (row.isConnected && wanted.get(row) && !row.hasAttribute(ROW_DISTANT_ATTRIBUTE)) changes.push([row, true]);
    }
    toggleRows(viewport, changes);
  }));
}

function createObserver(viewport: Element, reach: number): IntersectionObserver {
  return new IntersectionObserver((entries) => {
    // Read before writing. A wide row owns horizontal scroll range, which containment would drop.
    // Numbered equations count across messages; containment would scope their counter to the row
    // (the same exemption math-block-marker.ts gives a single block).
    const distant = entries.map(({ target, boundingClientRect, rootBounds }) =>
      rootBounds !== null &&
      (boundingClientRect.bottom < rootBounds.top || boundingClientRect.top > rootBounds.bottom) &&
      target.scrollWidth <= target.clientWidth &&
      target.querySelector(NUMBERED_EQUATION) === null
    );
    const release: Array<[Element, boolean]> = [];
    const skip: Element[] = [];
    entries.forEach(({ target }, i) => {
      wanted.set(target, distant[i]);
      if (!distant[i] && target.hasAttribute(ROW_DISTANT_ATTRIBUTE)) release.push([target, false]);
      else if (distant[i] && !target.hasAttribute(ROW_DISTANT_ATTRIBUTE)) skip.push(target);
    });
    toggleRows(viewport, release);
    if (skip.length > 0) settleLater(viewport, skip);
  }, { root: viewport, rootMargin: `${reach}px 0px` });
}

const windowHeight = (): number => (typeof window === "undefined" ? 0 : window.innerHeight);

/** A taller window widens the engine's margin, so the reach follows; unchanged reach is a no-op. */
function refreshReach() {
  const reach = rowReachPx(windowHeight());
  for (const [viewport, watch] of watches) {
    if (watch.reach === reach) continue;
    watch.observer.disconnect();
    watch.reach = reach;
    watch.observer = createObserver(viewport, reach);
    for (const row of watch.rows) watch.observer.observe(row);
  }
}

/** Select All renders every skipped row at once (WebKit lays out selected content), so a reader
 *  scrolled up after the column narrowed would jump. Release them first, anchored, before the
 *  browser's default action picks its range (the same order message-window-controller.ts uses). */
export function releaseAllRows() {
  for (const [viewport, watch] of watches) {
    const changes: Array<[Element, boolean]> = [];
    for (const row of watch.rows) if (row.hasAttribute(ROW_DISTANT_ATTRIBUTE)) changes.push([row, false]);
    toggleRows(viewport, changes);
  }
}

function onKeyDown(event: KeyboardEvent) {
  if (!(event.ctrlKey || event.metaKey) || event.key.toLowerCase() !== "a") return;
  const target = event.target as Element | null;
  if (target?.closest?.("input, textarea, [contenteditable=true]")) return;
  releaseAllRows();
}

let resizeHooked = false;

export function observeDistantRow(row: HTMLElement, adjust?: Adjust) {
  if (typeof document === "undefined" || !document.documentElement.hasAttribute(GATE_ATTRIBUTE)) return;
  const viewport = row.closest(".aui-thread-viewport");
  if (!viewport || typeof IntersectionObserver === "undefined") return;
  if (adjust) adjusters.set(viewport, adjust);
  if (!resizeHooked && typeof window !== "undefined") {
    resizeHooked = true;
    window.addEventListener("resize", refreshReach, { passive: true });
    document.addEventListener("keydown", onKeyDown, true);
  }

  let watch = watches.get(viewport);
  if (!watch) {
    const reach = rowReachPx(windowHeight());
    watch = { observer: createObserver(viewport, reach), reach, rows: new Set() };
    watches.set(viewport, watch);
  }
  watch.rows.add(row);
  watch.observer.observe(row);
  const own = watch;
  return () => {
    own.observer.unobserve(row);
    own.rows.delete(row);
    if (own.rows.size === 0) {
      own.observer.disconnect();
      if (watches.get(viewport) === own) watches.delete(viewport);
    }
    wanted.delete(row);
    row.removeAttribute(ROW_DISTANT_ATTRIBUTE);
  };
}
