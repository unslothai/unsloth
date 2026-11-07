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

type Adjust = (deltaPx: number) => unknown;

const observers = new WeakMap<Element, IntersectionObserver>();
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

export function observeDistantRow(row: HTMLElement, adjust?: Adjust) {
  if (typeof document === "undefined" || !document.documentElement.hasAttribute(GATE_ATTRIBUTE)) return;
  const viewport = row.closest(".aui-thread-viewport");
  if (!viewport || typeof IntersectionObserver === "undefined") return;
  if (adjust) adjusters.set(viewport, adjust);

  let observer = observers.get(viewport);
  if (!observer) {
    observer = new IntersectionObserver((entries) => {
      // Read before writing. A wide row owns horizontal scroll range, which containment would drop.
      const distant = entries.map(({ target, boundingClientRect, rootBounds }) =>
        rootBounds !== null &&
        (boundingClientRect.bottom < rootBounds.top || boundingClientRect.top > rootBounds.bottom) &&
        target.scrollWidth <= target.clientWidth
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
    }, { root: viewport, rootMargin: `${ROW_REACH_PX}px 0px` });
    observers.set(viewport, observer);
  }
  observer.observe(row);
  return () => {
    observer.unobserve(row);
    wanted.delete(row);
    row.removeAttribute(ROW_DISTANT_ATTRIBUTE);
  };
}
