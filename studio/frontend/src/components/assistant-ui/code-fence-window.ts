// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/*
 * Which lines of a fence carry token spans. JSX-free so the tests can RUN it.
 *
 * A line outside the window loses its COLOUR and nothing else: every character stays in the
 * document, which is what separates this from virtualization (rejected in
 * `progressive-mount-controller.ts`) and from `content-visibility: auto` (banned on code blocks in
 * `index.css`; WebKit before Safari 26 cannot find-in-page skipped content).
 *
 * The swap replaces a line's CHILDREN, never the element, so heights cannot move. That matters:
 * `use-intent-aware-autoscroll.tsx` absorbs only 64px of sub-frame shrink and
 * `progressive-messages.tsx` has scroll anchoring off.
 */

/** An inclusive line range. `null` means "no window": every line renders its token spans. */
export type LineWindow = { first: number; last: number };

// Below the cap a fence is highlighted end to end and produces main's DOM byte for byte; the
// mechanism exists for the 140K-character fence in #10769. Lines, not characters: the bounded cost
// is mounted elements, which track lines.
export const WINDOW_CAP_LINES = 2_000;

// Viewports of lookahead each way, matching `REACH_MARGIN`: the reader crosses into already
// coloured code rather than watching it arrive.
export const OVERSCAN_VIEWPORTS = 1;

// Slack before the window moves, in viewports. Without it one line of scroll re-renders both
// boundaries, the thrash that killed the earlier bidirectional gate.
export const HYSTERESIS_VIEWPORTS = 0.5;

export type WindowGeometry = {
  lineCount: number;
  /** MEASURED from the rendered block, never assumed. */
  lineHeight: number;
  contentTop: number;
  /** The fence's nearest scrolling ancestor, or the window. */
  viewportTop: number;
  viewportHeight: number;
  /** The window currently rendered, so an ordinary scroll can keep it. */
  previous: LineWindow | null;
  cap?: number;
};

const clamp = (value: number, low: number, high: number): number =>
  value < low ? low : value > high ? high : value;

// Arithmetic, not a measurement per line: reading 20,000 line boxes would cost more than the
// rendering this avoids.
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
 * Which lines carry token spans. `null` (every line) is both the small-fence answer and the answer
 * for untrustworthy geometry, so a zero line height degrades to today's rendering, never to an
 * arbitrary range. A wrong answer costs colour, never text.
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

  // Checked against the VISIBLE range, not the overscanned one: overscan is a rendering margin,
  // and testing it would move the window on every scroll.
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

  // The cap binds the window itself, or a zoomed-out print layout restores the cost it bounds.
  // Shrink from the END so lines the reader is heading towards outlive ones they passed.
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

// Structurally typed rather than importing Shiki's `ThemedToken`, so a bundler-less test runner
// can load this module.
type LineToken = { content: string };

/**
 * A blank line is one line tall, not nothing: a `<span>` holding an empty text node has no line box,
 * so the fence would close up and everything below it would move. Streamdown special-cases the same
 * two shapes to a bare newline.
 */
export const isBlankLine = (line: readonly LineToken[]): boolean =>
  line.length === 0 || (line.length === 1 && line[0].content === "");

/**
 * A line's text from its TOKENS, never a source slice. The invariant the window rests on is that a
 * line shows the same characters on both sides of it; reading it back out of the tokens makes that
 * true by construction. A slice would not: Shiki drops the CR of a CRLF pair, so a line leaving the
 * window would gain a character (`code-plugin.ts` `completedEnd`, and the CRLF cases in
 * `code-plugin-incremental.test.ts`).
 */
export const plainLineText = (line: readonly LineToken[]): string => {
  let text = "";
  for (const token of line) text += token.content;
  return text;
};
