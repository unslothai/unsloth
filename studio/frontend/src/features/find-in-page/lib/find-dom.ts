// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The DOM half of find-in-page: offsets back into `Range`s, painting them, and moving the reader
// to one. The arithmetic and the flatten live in find-text-index.ts, which is pure.

import {
  type FindMatch,
  type FindTextIndex,
  FIND_SCOPE_ATTRIBUTE,
  FIND_SKIP_ATTRIBUTE,
  endPositionAt,
  startPositionAt,
} from "./find-text-index.ts";

/**
 * Registry names for the two highlights: every match paints with the first, the active one with the
 * second, on top.
 *
 * The CSS Custom Highlight API is what keeps highlighting off the document: a `Highlight` is a set
 * of `Range`s painted over text already laid out, so nothing is inserted and nothing reflows.
 * Wrapping matches in `<mark>` would mutate the thread on every keystroke, splitting text nodes
 * that streaming markdown, `thread-fast-copy` and the export path all read back.
 */
export const FIND_HIGHLIGHT = "unsloth-find";
export const FIND_HIGHLIGHT_ACTIVE = "unsloth-find-active";

/**
 * How many matches are painted at once. The window travels with the active match, so what is on
 * screen is always painted; the rest are counted and reachable, just not tinted yet.
 */
export const MAX_PAINTED_RANGES = 400;

/** Distance kept from a scroller's edge before a match counts as needing to be scrolled to. */
const REVEAL_INSET_PX = 24;

type HighlightLike = { priority: number };
type HighlightConstructor = new (...ranges: Range[]) => HighlightLike;
type HighlightRegistry = {
  set(name: string, highlight: HighlightLike): void;
  delete(name: string): void;
};

/**
 * The registry and constructor, or null on an engine without them.
 *
 * Read through `globalThis` rather than typed globals: the app's `lib` is ES2022 + DOM, which has
 * no `Highlight` in it. WebKitGTK is the engine that lands here, and `selectRangeFallback` covers it.
 */
function highlightApi(): {
  registry: HighlightRegistry;
  Highlight: HighlightConstructor;
} | null {
  const scope = globalThis as {
    CSS?: { highlights?: HighlightRegistry };
    Highlight?: HighlightConstructor;
  };
  const registry = scope.CSS?.highlights;
  const Highlight = scope.Highlight;
  if (!registry || typeof Highlight !== "function") return null;
  return { registry, Highlight };
}

/** True when this engine can paint highlights without touching the document. */
export function supportsHighlightApi(): boolean {
  return highlightApi() !== null;
}

/** The subtree the bar searches: the shell's content region, which leaves out the sidebar and the
 *  bar itself. `document.body` is the fallback for a route that renders no shell. */
export function resolveFindScope(): Element | null {
  if (typeof document === "undefined") return null;
  return (
    document.querySelector(`[${FIND_SCOPE_ATTRIBUTE}]`) ?? document.body ?? null
  );
}

/** What the walk will actually read, asked of one element: inside the scope, and under nothing the
 *  walk turns back at. The shell keeps every workspace mounted and marks the off-route ones `inert`,
 *  so being in the document is not the same as being searchable. */
export function indexReaches(
  scope: Element | null,
  element: Element | null,
): boolean {
  if (scope === null || element === null) return false;
  if (!scope.contains(element)) return false;
  return (
    element.closest(
      `[aria-hidden="true"], [inert], [${FIND_SKIP_ATTRIBUTE}]`,
    ) === null
  );
}

/**
 * A live `Range` over one match, or null when the index has drifted from the document.
 *
 * Drift is expected, not exceptional: a streaming reply rewrites text nodes under the index. A
 * stale offset past the end of a shortened node throws, so a null drops that match until the next
 * rebuild.
 */
export function rangeForMatch(
  index: FindTextIndex,
  match: FindMatch,
): Range | null {
  const start = startPositionAt(index.segments, match.start);
  const end = endPositionAt(index.segments, match.end);
  if (!start || !end) return null;
  try {
    const range = document.createRange();
    range.setStart(start.node as unknown as Node, start.offset);
    range.setEnd(end.node as unknown as Node, end.offset);
    return range;
  } catch {
    return null;
  }
}

/** The slice of `matches` to paint, centred on `active`. Pure so the window can be tested without
 *  a document: the active match is always inside it, and it is never wider than the cap. */
export function paintWindow(
  total: number,
  active: number,
  cap = MAX_PAINTED_RANGES,
): { from: number; to: number } {
  if (total <= cap) return { from: 0, to: total };
  const half = Math.floor(cap / 2);
  const from = Math.min(Math.max(0, active - half), total - cap);
  return { from, to: from + cap };
}

/** Paint `ranges`, with `activeRange` on top. Both registries are replaced wholesale, which is one
 *  write per navigation rather than one per match. */
export function paintHighlights(
  ranges: Range[],
  activeRange: Range | null,
): void {
  const api = highlightApi();
  if (!api) return;
  const { registry, Highlight } = api;
  if (ranges.length === 0) {
    registry.delete(FIND_HIGHLIGHT);
  } else {
    registry.set(FIND_HIGHLIGHT, new Highlight(...ranges));
  }
  if (!activeRange) {
    registry.delete(FIND_HIGHLIGHT_ACTIVE);
    return;
  }
  const active = new Highlight(activeRange);
  // The same text is in both sets, and registration order is not a guarantee the spec makes.
  active.priority = 1;
  registry.set(FIND_HIGHLIGHT_ACTIVE, active);
}

/** Take both highlights back down. Safe to call on an engine that never had them. */
export function clearHighlights(): void {
  const api = highlightApi();
  if (!api) return;
  api.registry.delete(FIND_HIGHLIGHT);
  api.registry.delete(FIND_HIGHLIGHT_ACTIVE);
}

/**
 * Show the active match by selecting it, for an engine with no highlight registry. This is what a
 * browser's own find does, so the tint is the platform's. A selection is one range, so on those
 * engines the bar counts every match and tints one.
 */
export function selectRangeFallback(range: Range | null): void {
  if (typeof window === "undefined") return;
  const selection = window.getSelection();
  if (!selection) return;
  if (range === null) {
    // Only ever clear a selection this put there. Opening the bar paints before a query is typed,
    // and on these engines that would otherwise throw away whatever the reader had highlighted to
    // copy, with no way back: closing cannot restore what was already gone.
    if (!ownsSelection) return;
    selection.removeAllRanges();
    ownsSelection = false;
    return;
  }
  selection.removeAllRanges();
  selection.addRange(range);
  ownsSelection = true;
}

/** Whether the selection on screen is the one above, rather than the reader's own. */
let ownsSelection = false;

/** True when this element scrolls its own overflow on the given axis. */
function scrollsAxis(element: Element, axis: "x" | "y"): boolean {
  const overflowing =
    axis === "y"
      ? element.scrollHeight > element.clientHeight + 1
      : element.scrollWidth > element.clientWidth + 1;
  if (!overflowing) return false;
  const style = getComputedStyle(element);
  const overflow = axis === "y" ? style.overflowY : style.overflowX;
  return overflow === "auto" || overflow === "scroll" || overflow === "overlay";
}

/**
 * Scroll one container so `rect` is comfortably inside it, or leave it alone when the match already
 * is. A match already on screen must not move the page, or stepping through matches inside one
 * paragraph would jerk the conversation on every press.
 */
function revealWithin(scroller: Element, rect: DOMRect): boolean {
  const view = scroller.getBoundingClientRect();
  let top = scroller.scrollTop;
  let left = scroller.scrollLeft;
  let moved = false;

  if (scrollsAxis(scroller, "y")) {
    const overTop = rect.top - (view.top + REVEAL_INSET_PX);
    const overBottom = rect.bottom - (view.bottom - REVEAL_INSET_PX);
    if (overTop < 0 || overBottom > 0) {
      top += rect.top - view.top - Math.max(0, (view.height - rect.height) / 2);
      moved = true;
    }
  }
  if (scrollsAxis(scroller, "x")) {
    const overLeft = rect.left - (view.left + REVEAL_INSET_PX);
    const overRight = rect.right - (view.right - REVEAL_INSET_PX);
    if (overLeft < 0 || overRight > 0) {
      left +=
        rect.left - view.left - Math.max(0, (view.width - rect.width) / 2);
      moved = true;
    }
  }
  if (!moved) return false;
  // `instant`, not the viewport's own `scroll-smooth`: holding Enter outruns a smooth scroll.
  scroller.scrollTo({ top, left, behavior: "instant" });
  return true;
}

/** The element a range starts in, which is where both walks below begin. */
function elementFor(range: Range): Element | null {
  const start = range.startContainer;
  return start.nodeType === 1
    ? (start as Element)
    : (start.parentElement ?? null);
}

/**
 * A rect to aim at for `range`, or null when nothing about it is laid out.
 *
 * A `content-visibility: auto` subtree the reader has not reached is skipped, so a range inside it
 * has a collapsed rect at the origin while the subtree's own box still has its placeholder
 * geometry. Aiming at the nearest ancestor that is laid out gets the reader there; the subtree
 * renders on the way and the highlight lands once it does.
 */
export function revealRect(range: Range): DOMRect | null {
  const rect = range.getBoundingClientRect();
  if (rect.width !== 0 || rect.height !== 0) return rect;
  let element = elementFor(range);
  while (element) {
    const box = element.getBoundingClientRect();
    if (box.width !== 0 || box.height !== 0) return box;
    element = element.parentElement;
  }
  return null;
}

/** Where `range` sits vertically, through the same fallback. */
export function rangeTop(range: Range): number | null {
  return revealRect(range)?.top ?? null;
}

/**
 * The top edge of the scroll container `range` sits in, in window coordinates.
 *
 * Not zero: the thread viewport starts below the navbar and the chat header, so a match clipped
 * just off the top of it still has a positive window-relative `top`. Treating that as visible sends
 * a fresh query backwards to an occurrence the reader cannot see.
 */
export function scrollViewportTop(range: Range): number {
  let element = elementFor(range);
  while (element) {
    if (scrollsAxis(element, "y")) return element.getBoundingClientRect().top;
    element = element.parentElement;
  }
  return 0;
}

/**
 * Bring `range` into view, innermost scroller first.
 *
 * Nested scrollers are real here (a wide code fence inside the thread viewport), and the rect has to
 * be re-read after each one. The window itself is never scrolled: the shell is a fixed-height
 * `100dvh` grid with `overflow-hidden`.
 */
export function scrollRangeIntoView(range: Range): void {
  let element = elementFor(range);
  // Bounded by DOM depth, and each step is one rect read plus at most one scroll write.
  while (element) {
    if (scrollsAxis(element, "y") || scrollsAxis(element, "x")) {
      // Re-read each time: scrolling the inner container decides what the outer one still owes.
      const rect = revealRect(range);
      if (!rect) return;
      revealWithin(element, rect);
    }
    element = element.parentElement;
  }
}
