// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/* Which lines carry token spans. Off-window lines lose only colour, never text or height. */

/** An inclusive line range. `null` means "no window": every line renders its token spans. */
export type LineWindow = { first: number; last: number };

// Below the cap a fence is highlighted end to end. Lines, not characters: cost tracks mounted elements.
export const WINDOW_CAP_LINES = 2_000;

// Viewports of lookahead each way, matching `REACH_MARGIN`.
export const OVERSCAN_VIEWPORTS = 1;

// Slack before the window moves; without it one line of scroll re-renders both boundaries.
export const HYSTERESIS_VIEWPORTS = 0.5;

export type WindowGeometry = {
  lineCount: number;
  lineHeight: number;
  contentTop: number;
  viewportTop: number;
  viewportHeight: number;
  previous: LineWindow | null;
  cap?: number;
};

const clamp = (value: number, low: number, high: number): number =>
  value < low ? low : value > high ? high : value;

// Arithmetic, not per-line measurement: reading thousands of line boxes costs more than it saves.
const visibleRange = (geometry: WindowGeometry): LineWindow => {
  const { lineCount, lineHeight, contentTop, viewportTop, viewportHeight } = geometry;
  const top = (viewportTop - contentTop) / lineHeight;
  const bottom = (viewportTop + viewportHeight - contentTop) / lineHeight;
  return {
    first: clamp(Math.floor(top), 0, lineCount - 1),
    last: clamp(Math.ceil(bottom) - 1, 0, lineCount - 1),
  };
};

/** `null` (every line) also answers untrustworthy geometry, so a wrong answer costs colour, never text. */
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

  // Checked against the visible range, not the overscanned one, or every scroll moves the window.
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

  // The cap binds the window itself; shrink from the END so lines ahead outlive ones passed.
  if (last - first + 1 > cap) {
    last = first + cap - 1;
    if (last < visible.last) {
      last = clamp(visible.last, 0, lineCount - 1);
      first = Math.max(0, last - cap + 1);
    }
  }

  return { first, last };
};

export const lineIsWindowed = (window: LineWindow | null, index: number): boolean =>
  window === null || (index >= window.first && index <= window.last);

// Structurally typed, not Shiki's ThemedToken, so a bundler-less test runner can load this.
type LineToken = { content: string };

/** A blank line must still be one line tall: an empty text node has no line box. */
export const isBlankLine = (line: readonly LineToken[]): boolean =>
  line.length === 0 || (line.length === 1 && line[0].content === "");

/** From the TOKENS, not a source slice: Shiki drops the CR of CRLF, so a slice would gain a char. */
export const plainLineText = (line: readonly LineToken[]): string => {
  let text = "";
  for (const token of line) text += token.content;
  return text;
};
