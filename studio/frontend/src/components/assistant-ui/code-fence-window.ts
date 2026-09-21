// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/*
 * WHICH LINES OF A FENCE CARRY TOKEN SPANS, decided in one pure function.
 *
 * Kept out of `code-fence-defer.tsx` for the same reason `code-fence-mode.ts` is: that is a
 * `.tsx`, the frontend's tests run under `node --experimental-strip-types`, and that runner cannot
 * load JSX. Every rule below is RUN by `tests/code-fence-window.test.ts` rather than checked by a
 * regex over a file nothing can import.
 *
 * WHAT IS BEING WINDOWED, and what is NOT. A fence's text is always in the document, every
 * character of it, whatever this function returns. The window decides only whether a line renders
 * as Shiki's token spans or as one plain text node holding the same characters. So a windowed
 * fence is still selectable, still copyable, still reachable by find-in-page and still printable,
 * and the only thing a line outside the window loses is its COLOUR -- off screen, by construction.
 * That is what separates this from virtualization, which `progressive-mount-controller.ts` rejects
 * on exactly those grounds, and from `content-visibility: auto`, which `index.css` bans on code
 * blocks because WebKit before Safari 26 cannot find-in-page skipped content.
 *
 * WHY IT IS SAFE TO SWAP. The swap replaces the CHILDREN of a line element, never the element. A
 * line is a block box whose height comes from the code block's `line-height`, and `<pre>` does not
 * wrap, so a line is exactly one line tall whether it holds forty spans or one text node. Nothing
 * above or below it moves. This matters more than it sounds: `use-intent-aware-autoscroll.tsx`
 * absorbs only 64px of sub-frame shrink and `progressive-messages.tsx` has already turned native
 * scroll anchoring off, so a windowing scheme that changed heights would be visible immediately.
 */

/** An inclusive line range. `null` means "no window": every line renders its token spans. */
export type LineWindow = { first: number; last: number };

/*
 * BELOW THE CAP, NOTHING HAPPENS AT ALL. A fence with fewer rendered lines than this is
 * highlighted end to end and is never downgraded, so it produces the DOM main produces, byte for
 * byte. The benchmark corpus averages 1,800 characters per fence and the field capture that
 * corpus was calibrated against carried 5.6 characters per span; neither comes anywhere near this,
 * which is deliberate. The mechanism exists for the 140K-character single fence in #10769, and a
 * cap is how it stays off everywhere else.
 * Counted in LINES rather than characters because the cost this bounds is the mounted element
 * count, and that tracks tokens, which track lines far better than they track characters.
 */
export const WINDOW_CAP_LINES = 2_000;

/*
 * How far past the viewport the window reaches, in viewports, each way. The same one-viewport
 * lookahead `REACH_MARGIN` gives whole-fence deferral, for the same reason: the reader should
 * cross into already-coloured code rather than watch it arrive.
 */
export const OVERSCAN_VIEWPORTS = 1;

/*
 * HOW FAR THE READER MUST GET TOWARDS AN EDGE BEFORE THE WINDOW MOVES, in viewports.
 * Without this, one line of scroll re-selects the range and re-renders both boundaries, which is
 * the thrash the earlier whole-fence bidirectional gate was killed for. With a full viewport of
 * slack inside the overscan, ordinary reading scrolls do not move the window at all, and when it
 * does move the old and new ranges overlap heavily, so most lines keep their identity and their
 * elements.
 */
export const HYSTERESIS_VIEWPORTS = 0.5;

export type WindowGeometry = {
  /** Total lines in the fence body. */
  lineCount: number;
  /** One line's height in px, MEASURED from the rendered block, never assumed. */
  lineHeight: number;
  /** Top of the code content box, in the same coordinate space as `viewportTop`. */
  contentTop: number;
  /** Top of the clipping viewport (the fence's nearest scrolling ancestor, or the window). */
  viewportTop: number;
  /** Height of that viewport. */
  viewportHeight: number;
  /** The window currently rendered, so an ordinary scroll can decide to keep it. */
  previous: LineWindow | null;
  /** Overridable for tests; defaults to `WINDOW_CAP_LINES`. */
  cap?: number;
};

const clamp = (value: number, low: number, high: number): number =>
  value < low ? low : value > high ? high : value;

/*
 * The lines the viewport actually covers, inclusive, clamped to the fence.
 * Line `i` occupies `[contentTop + i*lineHeight, contentTop + (i+1)*lineHeight)`, so this is
 * arithmetic rather than a measurement per line: reading 20,000 line boxes to find out which two
 * are on screen would cost more than the rendering this exists to avoid.
 */
const visibleRange = (geometry: WindowGeometry): LineWindow => {
  const { lineCount, lineHeight, contentTop, viewportTop, viewportHeight } = geometry;
  const top = (viewportTop - contentTop) / lineHeight;
  const bottom = (viewportTop + viewportHeight - contentTop) / lineHeight;
  return {
    first: clamp(Math.floor(top), 0, lineCount - 1),
    last: clamp(Math.ceil(bottom) - 1, 0, lineCount - 1),
  };
};

/**
 * Which lines should carry token spans.
 *
 * `null` means every line does, which is both the small-fence answer and the answer whenever the
 * geometry cannot be trusted: a zero or negative line height (fonts still loading, a detached
 * node, a display:none ancestor) must degrade to today's rendering rather than to an arbitrary
 * range. Nothing here can lose text, so the worst a wrong answer costs is colour.
 */
export const selectLineWindow = (geometry: WindowGeometry): LineWindow | null => {
  const cap = geometry.cap ?? WINDOW_CAP_LINES;
  const { lineCount, lineHeight, viewportHeight } = geometry;
  if (lineCount <= cap) return null;
  if (!(lineHeight > 0) || !(viewportHeight > 0)) return null;
  if (!Number.isFinite(geometry.contentTop) || !Number.isFinite(geometry.viewportTop)) {
    return null;
  }

  const visible = visibleRange(geometry);
  const linesPerViewport = Math.ceil(viewportHeight / lineHeight);

  /*
     * KEEP THE CURRENT WINDOW IF THE READER IS STILL WELL INSIDE IT. Checked against the VISIBLE
     * range rather than the overscanned one: overscan is a rendering margin, and asking whether it
     * still fits would move the window on every scroll, which is the whole failure this avoids.
     */
  const slack = Math.ceil(linesPerViewport * HYSTERESIS_VIEWPORTS);
  const previous = geometry.previous;
  if (
    previous !== null
    && previous.first <= Math.max(0, visible.first - slack)
    && previous.last >= Math.min(lineCount - 1, visible.last + slack)
  ) {
    return previous;
  }

  const overscan = linesPerViewport * OVERSCAN_VIEWPORTS;
  let first = clamp(visible.first - overscan, 0, lineCount - 1);
  let last = clamp(visible.last + overscan, 0, lineCount - 1);

  /*
     * THE CAP BINDS THE WINDOW TOO, not just the decision to have one. A viewport tall enough to
     * hold more than `cap` lines is not a thing a reader has, but a zoomed-out print layout or a
     * 4K window with a tiny font can approach it, and a window wider than the cap would quietly
     * restore the cost this bounds. Shrink from the END first so the lines the reader is heading
     * towards outlive the ones they have passed.
     */
  if (last - first + 1 > cap) {
    last = first + cap - 1;
    if (last < visible.last) {
      last = clamp(visible.last, 0, lineCount - 1);
      first = Math.max(0, last - cap + 1);
    }
  }

  return { first, last };
};

/** Is line `index` inside `window`? A `null` window highlights everything. */
export const lineIsWindowed = (window: LineWindow | null, index: number): boolean =>
  window === null || (index >= window.first && index <= window.last);

/*
 * WHAT A LINE RENDERS AS, on either side of the window. Structurally typed rather than importing
 * Shiki's `ThemedToken`, so this module stays loadable by a test runner that has no bundler: the
 * two functions below read `content` and nothing else.
 */
type LineToken = { content: string };

/**
 * A LINE WITH NO CONTENT IS ONE LINE TALL, not nothing.
 *
 * Same rule as `shellBody` in `code-fence-defer.tsx`, and the same reason: a `<span>` holding an
 * empty text node has no line box at all, so a blank line in the middle of a fence would close up
 * and everything below it would move. Streamdown special-cases the same two shapes -- an empty
 * token array, and a single token whose content is empty -- to a bare newline.
 */
export const isBlankLine = (line: readonly LineToken[]): boolean =>
  line.length === 0 || (line.length === 1 && line[0].content === "");

/**
 * A line's text, taken from its TOKENS rather than from a slice of the source.
 *
 * This is the invariant the whole window rests on: a line must render the same characters whether
 * it is inside the window or outside it, or the fence's text would change as the reader scrolled.
 * Reading it back out of the tokens makes that true by construction. A source slice would not:
 * Shiki drops the CR of a CRLF pair from token content, so a line that left the window would
 * silently gain a character, and `code-plugin.ts` goes to some trouble over exactly that boundary
 * (`completedEnd`, and the CRLF cases in `code-plugin-incremental.test.ts`).
 */
export const plainLineText = (line: readonly LineToken[]): string => {
  let text = "";
  for (const token of line) text += token.content;
  return text;
};
