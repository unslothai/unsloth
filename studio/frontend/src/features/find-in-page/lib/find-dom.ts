// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The DOM half of find-in-page: offsets back into `Range`s, painting them, and moving the reader
// to one. The arithmetic and the flatten live in find-text-index.ts, which is pure.

import {
  FIND_SCOPE_ATTRIBUTE,
  FIND_SKIP_ATTRIBUTE,
  type FindMatch,
  type FindTextIndex,
  endPositionAt,
  startPositionAt,
} from "./find-text-index.ts";

/**
 * Registry names for the two highlights: every match with the first, the active one on top.
 *
 * The Custom Highlight API keeps highlighting off the document: a `Highlight` is a set of `Range`s
 * painted over text already laid out, so nothing is inserted and nothing reflows. `<mark>` would
 * mutate the thread on every keystroke, splitting nodes that streaming markdown, `thread-fast-copy`
 * and the export path read back.
 */
export const FIND_HIGHLIGHT = "unsloth-find";
export const FIND_HIGHLIGHT_ACTIVE = "unsloth-find-active";

/** How many matches are painted at once. The window travels with the active match, so what is on
 *  screen is always painted; the rest are counted and reachable, just not tinted. */
export const MAX_PAINTED_RANGES = 400;

/** Distance kept from a scroller's edge before a match counts as needing to be scrolled to. */
const REVEAL_INSET_PX = 24;

type HighlightLike = { priority: number };
type HighlightConstructor = new (...ranges: Range[]) => HighlightLike;
type HighlightRegistry = {
  set(name: string, highlight: HighlightLike): void;
  delete(name: string): void;
};

/** The registry and constructor, or null on an engine without them. Read through `globalThis`
 *  because the app's `lib` is ES2022 + DOM, which has no `Highlight`. WebKitGTK lands here, and
 *  `selectRangeFallback` covers it. */
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

/**
 * Surfaces the shell renders in front of the workspace but outside it.
 *
 * A popover portals to the body, so the model picker's list is on screen but not in the scope, and
 * searching the page behind it is a lie. Narrow on purpose: the layers a reader works IN, so a
 * toast or a tooltip does not change the count under them.
 */
const PORTAL_SURFACE_SELECTOR =
  '[data-slot="popover-content"], [role="menu"], [role="listbox"]';

export function resolvePortalSurfaces(scope: Element | null): Element[] {
  if (typeof document === "undefined" || scope === null) return [];
  const found: Element[] = [];
  for (const element of document.querySelectorAll(PORTAL_SURFACE_SELECTOR)) {
    // Inside the scope already, or nested in a surface already taken.
    if (scope.contains(element)) continue;
    if (found.some((taken) => taken.contains(element))) continue;
    // On its way out: these animate closed and keep a box until that finishes, so nothing else
    // here would turn them down.
    if (element.getAttribute("data-state") === "closed") continue;
    found.push(element);
  }
  return found;
}

/**
 * The index offset nearest the top of the reader's viewport, or 0 when nothing can be measured.
 *
 * Binary search over the segments: a dozen or so rect reads whatever the document's size. Only
 * asked when a query is common enough to hit the match cap, which is what needs to know where the
 * reader is.
 */
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
  // Everything is above the fold, so the reader is at the end of it.
  return found === -1 ? index.text.length : segments[found].start;
}

/**
 * What the walk turns back at, as a selector.
 *
 * The shell keeps every workspace mounted and parks off-route ones under `hidden` and `inert` so a
 * long generation survives navigation, and Radix marks the page `aria-hidden` behind a modal. Being
 * in the document is not the same as being searchable. Attributes only: a region hidden by a CLASS
 * is skipped through resolved style, which no selector can see.
 */
const SKIPPED_REGION_SELECTOR = `[aria-hidden="true"], [inert], [hidden], [${FIND_SKIP_ATTRIBUTE}]`;

/** What the walk will actually read, asked of one element: inside the scope, and under nothing the
 *  walk turns back at. */
export function indexReaches(
  scope: Element | null,
  element: Element | null,
): boolean {
  if (scope === null || element === null) return false;
  if (!scope.contains(element)) return false;
  return element.closest(SKIPPED_REGION_SELECTOR) === null;
}

/**
 * True when a mutation touched text the index covers, rather than the bar's own chrome. `some()`
 * short-circuits, so this costs a `closest` call only on batches the bar itself produced.
 *
 * Lives here, not beside its caller, so a test can hand it a record: the hook imports React and
 * cannot be loaded under `node --test`.
 */
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
  // Whichever attribute changed, ask from the PARENT. `closest` matches where it starts, so an
  // element just parked answers with itself and its own record is dropped - the one record saying a
  // region left the index. Removal reindexed fine, addition never did, though the observer watches
  // both, and every attribute it watches decides searchability. A detached target counts as a
  // change.
  const from = record.type === "attributes" ? element.parentElement : element;
  if (!from) return true;
  // Not just the bar's own chrome: an off-route workspace goes on streaming a reply into the
  // document, and each character of it used to buy a full flatten that then correctly threw that
  // text away. Once per throttle for as long as the generation ran.
  return from.closest(SKIPPED_REGION_SELECTOR) === null;
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

/** The caret inside a focused text field, so moving the document selection can give it back. */
type CaretHold = {
  field: HTMLInputElement | HTMLTextAreaElement;
  start: number | null;
  end: number | null;
};

/**
 * The find field's caret, captured before the selection is moved out from under it.
 *
 * On WebKit and Blink, moving the selection into ordinary text while a field is focused takes the
 * caret with it: `activeElement` still reports the field, but every keystroke is swallowed, so the
 * query freezes at one character. Gecko keeps the two apart. WebKit is the case that matters: an
 * engine with no highlight registry is Firefox below 140 or the desktop build's WebKitGTK.
 */
function holdCaret(): CaretHold | null {
  // Through `globalThis`: the node suite drives this with a hand-rolled window, no document and
  // none of the constructors below.
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

/** Put the caret back where it was, so the next keystroke reaches the field. */
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

/**
 * Show the active match by selecting it, for an engine with no highlight registry, which is what a
 * browser's own find does. A selection is one range, so there the bar counts every match and tints
 * one.
 *
 * The caret is handed back afterwards, which keeps the field typable. Whether the selection
 * survives is the engine's call: Gecko keeps both, WebKit and Blink drop it. Losing the tint is a
 * worse look; losing the field is a broken feature.
 */
export function selectRangeFallback(range: Range | null): void {
  if (typeof window === "undefined") return;
  const selection = window.getSelection();
  if (!selection) return;
  if (range === null) {
    // Only clear the selection this put there, and only while it is still the one on screen:
    // opening paints before a query is typed, and dragging replaces the match, and neither has a
    // way back.
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

/** The range this put on screen, or null when the selection is the reader's own. */
let ownedSelection: Range | null = null;

function currentRange(selection: Selection): Range | null {
  return selection.rangeCount > 0 ? selection.getRangeAt(0) : null;
}

/** Boundary points, not identity: engines differ on whether `getRangeAt` hands back what was added. */
function sameBoundaries(a: Range, b: Range | null): boolean {
  return (
    b !== null &&
    a.startContainer === b.startContainer &&
    a.startOffset === b.startOffset &&
    a.endContainer === b.endContainer &&
    a.endOffset === b.endOffset
  );
}

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

/** Scroll one container so `rect` sits comfortably inside it, or leave it alone when it already
 *  does: a match on screen must not move the page, or stepping through one paragraph would jerk
 *  the conversation on every press. */
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
 * A skipped `content-visibility: auto` subtree gives a range inside it a collapsed rect at the
 * origin, while the subtree's own box keeps its placeholder geometry. Aiming at the nearest laid
 * out ancestor gets the reader there, and the subtree renders on the way.
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

/** The top edge of the scroll container `range` sits in, in window coordinates. Not zero: the
 *  thread viewport starts below the navbar and header, so a match clipped just off its top still
 *  has a positive `top`, and treating that as visible sends a query backwards out of sight. */
export function scrollViewportTop(range: Range): number {
  let element = elementFor(range);
  while (element) {
    if (scrollsAxis(element, "y")) return element.getBoundingClientRect().top;
    element = element.parentElement;
  }
  return 0;
}

/** Bring `range` into view, innermost scroller first. Nested scrollers are real (a wide code fence
 *  in the thread viewport), so the rect is re-read after each. The window is never scrolled: the
 *  shell is a fixed-height `100dvh` grid with `overflow-hidden`. */
export function scrollRangeIntoView(range: Range): boolean {
  let element = elementFor(range);
  let moved = false;
  // Bounded by DOM depth, and each step is one rect read plus at most one scroll write.
  while (element) {
    if (scrollsAxis(element, "y") || scrollsAxis(element, "x")) {
      // Re-read each time: scrolling the inner container decides what the outer one still owes.
      const rect = revealRect(range);
      if (!rect) return moved;
      if (revealWithin(element, rect)) moved = true;
    }
    element = element.parentElement;
  }
  return moved;
}

/**
 * Bring `range` into view, and again for as long as the view keeps moving.
 *
 * A scroll reaches only the scrollHeight the engine knows, and a `content-visibility: auto` subtree
 * contributes its placeholder until it renders: a Hub README block can stand at 140px for a 2904px
 * reality. Reaching toward it is what makes it render, so the next frame has more to scroll and the
 * match has moved. Without this the reader was left 3415px below an 800px viewport on all three
 * engines. Typing hides it, since every keystroke reveals again; one reveal from a paste or an
 * Enter stays wrong.
 *
 * "Still moving" rather than asking whether the subtree was skipped, because the engines that most
 * need this are the ones that do not answer `checkVisibility`. Ends when a pass moves nothing: 2
 * frames on webkit, 3 on chromium and firefox. The bound stops anything spinning.
 */
export function revealRangeWhenPainted(range: Range, tries = 8): void {
  cancelRevealPasses();
  revealPass(range, tries, revealGeneration);
}

/** The chain in flight. A new reveal or a closing bar supersedes the last one. */
let revealGeneration = 0;

/** Abandon any queued reveal. The workspace stays mounted after the bar closes, so `isConnected`
 *  alone would let the old chain keep scrolling the reader toward a match nobody asked for. */
export function cancelRevealPasses(): void {
  revealGeneration += 1;
}

function revealPass(range: Range, tries: number, generation: number): void {
  if (!scrollRangeIntoView(range) || tries <= 1) return;
  if (typeof requestAnimationFrame !== "function") return;
  requestAnimationFrame(() => {
    if (generation !== revealGeneration) return;
    // A streaming reply rewrites the nodes under the index, so the range can be gone by now.
    if (!range.startContainer.isConnected) return;
    revealPass(range, tries - 1, generation);
  });
}
