// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  FIND_PORTAL_ATTRIBUTE,
  FIND_SCOPE_ATTRIBUTE,
  FIND_SKIP_ATTRIBUTE,
} from "./find-attributes.ts";
import {
  type FindMatch,
  type FindTextIndex,
  endPositionAt,
  startPositionAt,
} from "./find-text-index.ts";

/** Custom Highlight API paints without mutating the DOM, unlike `<mark>`. */
export const FIND_HIGHLIGHT = "unsloth-find";
export const FIND_HIGHLIGHT_ACTIVE = "unsloth-find-active";

/** The paint window travels with the active match; the rest are counted, not tinted. */
export const MAX_PAINTED_RANGES = 400;

const REVEAL_INSET_PX = 24;

type HighlightLike = { priority: number; clear(): void };
type HighlightConstructor = new (...ranges: Range[]) => HighlightLike;
type HighlightRegistry = {
  get(name: string): HighlightLike | undefined;
  set(name: string, highlight: HighlightLike): void;
  delete(name: string): void;
};

/** Via globalThis since lib has no `Highlight`. Null on WebKitGTK; see selectRangeFallback. */
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

export function supportsHighlightApi(): boolean {
  return highlightApi() !== null;
}

export function resolveFindScope(): Element | null {
  if (typeof document === "undefined") return null;
  return (
    document.querySelector(`[${FIND_SCOPE_ATTRIBUTE}]`) ?? document.body ?? null
  );
}

const DISMISSIBLE_PORTAL_SURFACE_SELECTOR =
  '[data-slot="popover-content"], [role="menu"], [role="listbox"]';
const SEARCHABLE_PORTAL_SURFACE_SELECTOR = `[${FIND_PORTAL_ATTRIBUTE}], ${DISMISSIBLE_PORTAL_SURFACE_SELECTOR}`;

function resolveSurfaces(scope: Element | null, selector: string): Element[] {
  if (typeof document === "undefined" || scope === null) return [];
  const found: Element[] = [];
  for (const element of document.querySelectorAll(selector)) {
    if (scope.contains(element)) continue;
    if (found.some((taken) => taken.contains(element))) continue;
    // Dismissible surfaces animate closed and keep a box until that finishes.
    if (element.getAttribute("data-state") === "closed") continue;
    found.push(element);
  }
  return found;
}

export function resolvePortalSurfaces(scope: Element | null): Element[] {
  return resolveSurfaces(scope, SEARCHABLE_PORTAL_SURFACE_SELECTOR);
}

export function resolveDismissiblePortalSurfaces(
  scope: Element | null,
): Element[] {
  return resolveSurfaces(scope, DISMISSIBLE_PORTAL_SURFACE_SELECTOR);
}

/** Index offset nearest the viewport top, by binary search; only needed at the cap. */
export function viewportOffset(index: FindTextIndex): number {
  const segments = index.segments;
  if (segments.length === 0) return 0;
  let lo = 0;
  let hi = segments.length - 1;
  let found = -1;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    const segment = segments[mid];
    const range = rangeForMatch(index, {
      start: segment.start,
      end: segment.start + 1,
    });
    if (!range) return 0;
    const top = rangeTop(range);
    if (top === null) return 0;
    if (top >= scrollViewportTop(range)) {
      found = mid;
      hi = mid - 1;
    } else {
      lo = mid + 1;
    }
  }
  return found === -1 ? index.text.length : segments[found].start;
}

/** Off-route workspaces are parked under `hidden`/`inert`, and Radix aria-hides behind
 *  modals. Class-hidden regions are caught via resolved style instead. */
const SKIPPED_REGION_SELECTOR = `[aria-hidden="true"]:not(.katex-html), [inert], [hidden], [${FIND_SKIP_ATTRIBUTE}]`;

export function indexReaches(
  scope: Element | null,
  element: Element | null,
): boolean {
  if (scope === null || element === null) return false;
  if (!scope.contains(element)) return false;
  return element.closest(SKIPPED_REGION_SELECTOR) === null;
}

export function mutatesSearchableText(record: {
  target: { nodeType: number; parentElement: Element | null };
  type: string;
  attributeName: string | null;
}): boolean {
  const target = record.target;
  const element =
    target.nodeType === 1
      ? (target as unknown as Element)
      : (target.parentElement ?? null);
  if (!element) return true;
  // Attribute changes are checked from the parent so parked or newly indexed regions reindex.
  const from = record.type === "attributes" ? element.parentElement : element;
  if (!from) return true;
  return from.closest(SKIPPED_REGION_SELECTOR) === null;
}

/** Null when the index drifted from the document (streaming rewrites text nodes). */
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

function removeRegisteredHighlight(
  registry: HighlightRegistry,
  name: string,
): boolean {
  const highlight = registry.get(name);
  if (!highlight) return false;
  // Tauri's WebKit may drop the registry entry but keep painted ranges; clear the set first.
  highlight.clear();
  registry.delete(name);
  return true;
}

/** WebKitGTK can leave stale highlight pixels; a tiny opacity change forces a repaint. */
function forceHighlightRepaint(): void {
  if (typeof document === "undefined") return;
  // The index includes body-level portals, so repaint the document root.
  const root = document.documentElement ?? resolveFindScope();
  if (
    root === null ||
    typeof HTMLElement === "undefined" ||
    !(root instanceof HTMLElement)
  )
    return;
  const previous = root.style.opacity;
  root.style.opacity = previous === "0.999999" ? "0.999998" : "0.999999";
  void root.offsetHeight;
  root.style.opacity = previous;
}

export function paintHighlights(
  ranges: Range[],
  activeRange: Range | null,
): void {
  const api = highlightApi();
  if (!api) return;
  const { registry, Highlight } = api;
  let removed = false;
  if (ranges.length === 0) {
    removed = removeRegisteredHighlight(registry, FIND_HIGHLIGHT);
  } else {
    removeRegisteredHighlight(registry, FIND_HIGHLIGHT);
    registry.set(FIND_HIGHLIGHT, new Highlight(...ranges));
  }
  if (!activeRange) {
    removed =
      removeRegisteredHighlight(registry, FIND_HIGHLIGHT_ACTIVE) || removed;
    if (removed) forceHighlightRepaint();
    return;
  }
  removeRegisteredHighlight(registry, FIND_HIGHLIGHT_ACTIVE);
  const active = new Highlight(activeRange);
  // Both sets hold the same text and registration order is not guaranteed by the spec.
  active.priority = 1;
  registry.set(FIND_HIGHLIGHT_ACTIVE, active);
}

export function clearHighlights(): void {
  const api = highlightApi();
  if (!api) return;
  const removedActive = removeRegisteredHighlight(
    api.registry,
    FIND_HIGHLIGHT_ACTIVE,
  );
  const removedAll = removeRegisteredHighlight(api.registry, FIND_HIGHLIGHT);
  if (removedActive || removedAll) forceHighlightRepaint();
}

type CaretHold = {
  field: HTMLInputElement | HTMLTextAreaElement;
  start: number | null;
  end: number | null;
};

/** On WebKit and Blink, moving the selection steals the focused field's caret, so save it. */
function holdCaret(): CaretHold | null {
  // Via globalThis: the node suite uses a hand-rolled window without these constructors.
  const scope = globalThis as {
    document?: { activeElement?: unknown };
    HTMLInputElement?: unknown;
    HTMLTextAreaElement?: unknown;
  };
  const active = scope.document?.activeElement;
  if (
    typeof scope.HTMLInputElement !== "function" ||
    typeof scope.HTMLTextAreaElement !== "function"
  ) {
    return null;
  }
  if (
    active instanceof HTMLInputElement ||
    active instanceof HTMLTextAreaElement
  ) {
    return {
      field: active,
      start: active.selectionStart,
      end: active.selectionEnd,
    };
  }
  return null;
}

function releaseCaret(held: CaretHold | null): void {
  if (!held) return;
  const { field, start, end } = held;
  field.focus({ preventScroll: true });
  if (start === null || end === null) return;
  try {
    field.setSelectionRange(start, end);
  } catch {
    // Some input types refuse a range. Focus is the half that matters.
  }
}

/** Fallback without a highlight registry: select the active match, then restore the caret. */
export function selectRangeFallback(range: Range | null): void {
  if (typeof window === "undefined") return;
  const selection = window.getSelection();
  if (!selection) return;
  if (range === null) {
    // Only clear a selection this put there that is still on screen.
    const owned = ownedSelection;
    ownedSelection = null;
    if (owned === null || !sameBoundaries(owned, currentRange(selection)))
      return;
    selection.removeAllRanges();
    return;
  }
  const held = holdCaret();
  selection.removeAllRanges();
  selection.addRange(range);
  ownedSelection = currentRange(selection) ?? range;
  releaseCaret(held);
}

let ownedSelection: Range | null = null;

function currentRange(selection: Selection): Range | null {
  return selection.rangeCount > 0 ? selection.getRangeAt(0) : null;
}

/** Compare boundaries: engines differ on whether `getRangeAt` returns the added range. */
function sameBoundaries(a: Range, b: Range | null): boolean {
  return (
    b !== null &&
    a.startContainer === b.startContainer &&
    a.startOffset === b.startOffset &&
    a.endContainer === b.endContainer &&
    a.endOffset === b.endOffset
  );
}

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

/** Leave the scroller alone when the match is already comfortably visible. */
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
  // `instant`, not smooth: holding Enter outruns a smooth scroll.
  scroller.scrollTo({ top, left, behavior: "instant" });
  return true;
}

function elementFor(range: Range): Element | null {
  const start = range.startContainer;
  return start.nodeType === 1
    ? (start as Element)
    : (start.parentElement ?? null);
}

/** Skipped `content-visibility: auto` ranges give a collapsed rect, so aim at an ancestor. */
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

export function rangeTop(range: Range): number | null {
  return revealRect(range)?.top ?? null;
}

/** Not zero: the thread viewport starts below the navbar and header. */
export function scrollViewportTop(range: Range): number {
  let element = elementFor(range);
  while (element) {
    if (scrollsAxis(element, "y")) return element.getBoundingClientRect().top;
    element = element.parentElement;
  }
  return 0;
}

/** Innermost scroller first, re-reading the rect after each; the window never scrolls. */
export function scrollRangeIntoView(range: Range): boolean {
  let element = elementFor(range);
  let moved = false;
  while (element) {
    if (scrollsAxis(element, "y") || scrollsAxis(element, "x")) {
      const rect = revealRect(range);
      if (!rect) return moved;
      if (revealWithin(element, rect)) moved = true;
    }
    element = element.parentElement;
  }
  return moved;
}

/** Re-reveal while the view keeps moving: `content-visibility: auto` placeholders grow once
 *  rendered. Ends when a pass moves nothing; `tries` bounds it. */
export function revealRangeWhenPainted(range: Range, tries = 8): void {
  cancelRevealPasses();
  revealPass(range, tries, revealGeneration);
}

let revealGeneration = 0;

/** The workspace stays mounted after close, so `isConnected` alone cannot stop a chain. */
export function cancelRevealPasses(): void {
  revealGeneration += 1;
}

function revealPass(range: Range, tries: number, generation: number): void {
  if (!scrollRangeIntoView(range) || tries <= 1) return;
  if (typeof requestAnimationFrame !== "function") return;
  requestAnimationFrame(() => {
    if (generation !== revealGeneration) return;
    if (!range.startContainer.isConnected) return;
    revealPass(range, tries - 1, generation);
  });
}
