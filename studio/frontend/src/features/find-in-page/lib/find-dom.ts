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

/**
 * Surfaces the shell renders in front of the workspace but outside it.
 *
 * A popover portals its content to the body, so the model picker's list is on screen and not in the
 * scope. Searching the page behind it while it is up is a lie, and it is the one thing the reader
 * can see. Narrow on purpose: the layers a reader works IN, not every portal, so a toast arriving
 * or a tooltip on the pointer does not change the count under them.
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
    // On its way out. These animate closed and are only unmounted when that finishes, and until
    // then they still have a box, so nothing else here would turn them down.
    if (element.getAttribute("data-state") === "closed") continue;
    found.push(element);
  }
  return found;
}

/**
 * The index offset nearest the top of the reader's viewport, or 0 when nothing can be measured.
 *
 * Binary search over the segments, the same shape as the one over matches: a dozen or so rect reads
 * whatever the size of the document. Only asked when a query is common enough to hit the match cap,
 * which is what needs to know where the reader is.
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
 * The shell keeps every workspace mounted and parks the off-route ones under `hidden` and `inert`
 * (`__root.tsx`) so a long generation is not cancelled by navigating away, and Radix marks the page
 * `aria-hidden` behind a modal. Being in the document is not the same as being searchable.
 *
 * Attributes only. A region hidden by a CLASS is skipped by the index all the same, through
 * resolved style, which no selector here can see.
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
 * short-circuits on the first qualifying record, so this costs a `closest` call only on the batches
 * the bar itself produced.
 *
 * Lives here rather than beside its caller so a test can hand it a record: the hook imports React
 * and cannot be loaded under `node --test`.
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
  // Whichever attribute changed, ask from the PARENT. `closest` matches the element it starts at,
  // so an element that has just been parked answers with itself and its own record is filtered out
  // - the one record that exists to say a region left the index. Removing the attribute reindexed
  // fine, adding it never did, though the observer asks for both. Every attribute the observer
  // watches is one that decides whether a region is searchable, so the rule is the same for all of
  // them. A detached target has no parent to ask, so it counts as a change.
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
 * The find field's caret, captured before the document selection is moved out from under it.
 *
 * Moving the selection into ordinary text while a field is focused takes the caret with it on
 * WebKit and Blink: `activeElement` still reports the field, but every keystroke after that is
 * swallowed, so the query freezes at one character and the bar cannot be typed into at all. Gecko
 * keeps the two apart and does not need this. WebKit is the case that matters, because an engine
 * with no highlight registry is either Firefox below 140 or the WebKitGTK the desktop build was
 * handed, and the second one lands here.
 */
function holdCaret(): CaretHold | null {
  // Read through `globalThis`: the node suite drives this with a hand-rolled window and has no
  // document, and the constructors below are not defined there either.
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
 * Show the active match by selecting it, for an engine with no highlight registry. This is what a
 * browser's own find does, so the tint is the platform's. A selection is one range, so on those
 * engines the bar counts every match and tints one.
 *
 * The caret is handed back afterwards, which is what keeps the field typable. Whether the selection
 * survives that is the engine's call: Gecko keeps both, WebKit and Blink drop the selection to give
 * the caret back. Losing the tint is a worse look; losing the field is a broken feature.
 */
export function selectRangeFallback(range: Range | null): void {
  if (typeof window === "undefined") return;
  const selection = window.getSelection();
  if (!selection) return;
  if (range === null) {
    // Only ever clear the selection this put there, and only while it is still the one on screen.
    // Opening the bar paints before a query is typed, and dragging over other text while the bar is
    // open replaces the match: either way there is no way back, since closing cannot restore what
    // was already gone.
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
 * A scroll reaches only as far as the scrollHeight the engine knows about, and a
 * `content-visibility: auto` subtree contributes its `contain-intrinsic-size` placeholder until it
 * renders. The Hub puts that containment on README prose and on each top-level child (hub.css), so
 * a long block can stand at 140px for a 2904px reality. Reaching toward it is itself what makes it
 * relevant, so the next frame has more document to scroll through and the match has moved.
 *
 * Measured on exactly that shape, without this: the reader was left 3415px below an 800px viewport
 * on chromium, firefox and webkit, the match counted and highlighted somewhere off screen. Typing
 * hides it, since every keystroke reveals again and those repeats do by accident what this does on
 * purpose; a pasted query, an Enter, or a click on the walk button is one reveal and stays wrong.
 *
 * "Still moving" rather than asking whether the subtree was skipped: `checkVisibility` is the
 * direct question but the engines that most need this are the ones that do not answer it, and a
 * scroll that changed nothing is the same news from any of them. It ends when a pass moves nothing:
 * measured, webkit took 2 frames and chromium and firefox 3. The bound is there so nothing can
 * spin, and 8 frames of reading rects is the whole cost of reaching it.
 */
export function revealRangeWhenPainted(range: Range, tries = 8): void {
  if (!scrollRangeIntoView(range) || tries <= 1) return;
  if (typeof requestAnimationFrame !== "function") return;
  requestAnimationFrame(() => {
    // A streaming reply rewrites the nodes under the index, so the range can be gone by now.
    if (!range.startContainer.isConnected) return;
    revealRangeWhenPainted(range, tries - 1);
  });
}
