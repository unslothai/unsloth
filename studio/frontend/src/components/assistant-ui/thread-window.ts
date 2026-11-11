// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Sliding mount window for a long thread: only rows near the reader are mounted, and spacers
// sized from measured heights (or an estimate until a row has rendered once) stand in for the
// rest. Pure: the DOM reads and the React commit live in progressive-messages.tsx.

/**
 * Rows `[start, end)` are mounted. So are `pins`: rows outside that range the reader is still
 * working in (focus, an open edit, playing media), which stay mounted where they are however far
 * the window moves. Ascending.
 */
export type RowWindow = { start: number; end: number; pins?: readonly number[] };

/** Rows are mounted this far past each viewport edge. */
export const WINDOW_REACH_PX = 2000;
/** A mounted row unmounts once it is this far past a viewport edge; the gap to REACH is hysteresis. */
export const WINDOW_KEEP_PX = 4000;
/** Rows a thread opens with at most; the first step sizes the window by pixels. */
export const WINDOW_INITIAL_ROWS = 16;
/**
 * A thread is windowed once it is this tall, whatever its message count: an agent chat is often
 * a few turns of very long replies. Under it the whole thread mounts, as a short chat always has.
 */
export const WINDOW_MIN_TOTAL_PX = 40_000;
/** With fewer messages there is no row to leave out worth the bookkeeping. */
export const WINDOW_MIN_ROWS = 8;

export const worthWindowing = (count: number, totalPx: number): boolean =>
  count >= WINDOW_MIN_ROWS && totalPx >= WINDOW_MIN_TOTAL_PX;

export type RowMeta = {
  id: string;
  role: string;
  /** Lines the message is expected to render; only its proportion to height matters. */
  weight: number;
  /** False for a message that renders no row (system prompts, a prompt being edited elsewhere). */
  renders: boolean;
};

/** Height of a row before its first line: padding, the action bar, the gap to the next row. */
const BASE_PX: Record<string, number> = { user: 96, assistant: 72 };
const DEFAULT_BASE_PX = 72;
/** What a line adds until a row has been measured: prose line height at the default size. */
const PRIOR_PX_PER_LINE = 24;
/** Lines the prior counts for, so the first measured rows outweigh it. */
const PRIOR_LINES = 8;
const MIN_ROW_PX = 24;
/** A source line wraps at about this many characters in the chat column. */
const CHARS_PER_LINE = 90;
/** A tool call or reasoning block renders collapsed: a header of about two lines. */
const COLLAPSED_PART_LINES = 2;
const MEDIA_PART_LINES = 12;

type Part = { type?: string; text?: string };

/** Lines a text renders: one per source line, more where a long one wraps. Counting lines and
 *  not characters is what keeps code (many short lines) from being estimated as a paragraph. */
function textLines(text: string): number {
  let lines = 0;
  let from = 0;
  for (;;) {
    const at = text.indexOf("\n", from);
    const length = (at === -1 ? text.length : at) - from;
    lines += 1 + Math.floor(length / CHARS_PER_LINE);
    if (at === -1) return lines;
    from = at + 1;
  }
}

/** Proportional to how tall the message renders: the lines of its text, plus a fixed share per
 *  tool call or reasoning block (both render collapsed) and per image. Sources are inline badges. */
export function messageWeight(content: readonly Part[] | undefined): number {
  if (!content) return 0;
  let weight = 0;
  for (const part of content) {
    if (part.type === "text") weight += part.text ? textLines(part.text) : 0;
    else if (part.type === "image" || part.type === "file") weight += MEDIA_PART_LINES;
    else if (part.type !== "source") weight += COLLAPSED_PART_LINES;
  }
  return weight;
}

type Density = { lines: number; px: number };

/**
 * Heights by message id. A row that has rendered keeps the height it had when last measured; one
 * that has not is estimated as a base plus its lines times the px a line has taken in the rows
 * measured so far (per role), starting from a prior. A ratio of sums and not a fitted line: the
 * rows measured first are the few at the tail, often alike, and a line through them is undefined.
 */
export class RowHeights {
  private measured = new Map<string, number>();
  private densities = new Map<string, Density>();
  /** What each measured row added to its role's density, so a re-measure replaces it. */
  private counted = new Map<string, { role: string; lines: number; px: number }>();
  private width = 0;

  /** A new column width re-wraps every row, so every height is forgotten. */
  resize(width: number): void {
    if (width === this.width) return;
    if (this.width > 0) {
      this.measured.clear();
      this.densities.clear();
      this.counted.clear();
    }
    this.width = width;
  }

  measure(meta: RowMeta, height: number): void {
    if (!(height >= 0)) return;
    if (this.measured.get(meta.id) === height) return;
    this.measured.set(meta.id, height);
    let density = this.densities.get(meta.role);
    if (!density) {
      density = { lines: PRIOR_LINES, px: PRIOR_LINES * PRIOR_PX_PER_LINE };
      this.densities.set(meta.role, density);
    }
    const before = this.counted.get(meta.id);
    if (before && before.role === meta.role) {
      density.lines -= before.lines;
      density.px -= before.px;
    }
    const px = Math.max(0, height - (BASE_PX[meta.role] ?? DEFAULT_BASE_PX));
    density.lines += meta.weight;
    density.px += px;
    this.counted.set(meta.id, { role: meta.role, lines: meta.weight, px });
  }

  has(id: string): boolean {
    return this.measured.has(id);
  }

  heightOf(meta: RowMeta): number {
    if (!meta.renders) return 0;
    const known = this.measured.get(meta.id);
    if (known !== undefined) return known;
    const density = this.densities.get(meta.role);
    const perLine = density ? density.px / density.lines : PRIOR_PX_PER_LINE;
    const base = BASE_PX[meta.role] ?? DEFAULT_BASE_PX;
    return Math.max(MIN_ROW_PX, Math.round(base + perLine * meta.weight));
  }
}

/** `win` holding `pins` (ascending): those inside the thread and outside `[start, end)`. */
export function withPins(win: RowWindow, pins: readonly number[], count: number): RowWindow {
  const kept = pins.filter(
    (index) => index >= 0 && index < count && (index < win.start || index >= win.end),
  );
  return kept.length > 0
    ? { start: win.start, end: win.end, pins: kept }
    : { start: win.start, end: win.end };
}

/**
 * A stretch of the thread as it is laid out: a mounted row, or a spacer standing in for rows
 * `[from, to)`. The `top` and `bottom` spacers sit against the window; a `gap` separates pinned
 * rows from each other and from the ends of the thread.
 */
export type ThreadSlot =
  | { kind: "row"; index: number }
  | { kind: "spacer"; name: "top" | "bottom" | "gap"; from: number; to: number };

/** The thread in document order. A full mount still has its two spacers, standing in for nothing. */
export function threadSlots(win: RowWindow | null, count: number): ThreadSlot[] {
  const start = win == null ? 0 : Math.max(0, Math.min(win.start, count));
  const end = win == null ? count : Math.max(start, Math.min(win.end, count));
  const pins = (win?.pins ?? []).filter((index) => index >= 0 && index < count);
  const slots: ThreadSlot[] = [];
  let at = 0;
  let from = 0;
  for (; at < pins.length && pins[at] < start; at += 1) {
    slots.push({ kind: "spacer", name: "gap", from, to: pins[at] });
    slots.push({ kind: "row", index: pins[at] });
    from = pins[at] + 1;
  }
  slots.push({ kind: "spacer", name: "top", from, to: start });
  for (let index = start; index < end; index += 1) slots.push({ kind: "row", index });
  while (at < pins.length && pins[at] < end) at += 1;
  slots.push({ kind: "spacer", name: "bottom", from: end, to: at < pins.length ? pins[at] : count });
  for (; at < pins.length; at += 1) {
    slots.push({ kind: "row", index: pins[at] });
    slots.push({
      kind: "spacer",
      name: "gap",
      from: pins[at] + 1,
      to: at + 1 < pins.length ? pins[at + 1] : count,
    });
  }
  return slots;
}

/** The rows the spacers against the window stand for: `[topFrom, start)` and `[end, bottomTo)`.
 *  The whole thread on either side, unless a pinned row is mounted there. */
export function spacerRows(win: RowWindow, count: number): { topFrom: number; bottomTo: number } {
  const start = Math.max(0, Math.min(win.start, count));
  const end = Math.max(start, Math.min(win.end, count));
  let topFrom = 0;
  let bottomTo = count;
  for (const pin of win.pins ?? []) {
    if (pin < start) topFrom = Math.max(topFrom, pin + 1);
    else if (pin >= end && pin < bottomTo) bottomTo = pin;
  }
  return { topFrom, bottomTo };
}

/** Heights of the rows the spacers before and after the window stand for. */
export function spacerHeights(
  heights: RowHeights,
  metas: readonly RowMeta[],
  win: RowWindow | null,
): { top: number; bottom: number } {
  if (win == null) return { top: 0, bottom: 0 };
  const { topFrom, bottomTo } = spacerRows(win, metas.length);
  let top = 0;
  let bottom = 0;
  const start = Math.min(win.start, metas.length);
  for (let index = topFrom; index < start; index += 1) top += heights.heightOf(metas[index]);
  for (let index = Math.max(win.end, start); index < bottomTo; index += 1) {
    bottom += heights.heightOf(metas[index]);
  }
  return { top, bottom };
}

/** The window after the thread's length changed: one that reached the tail follows it, and one
 *  past a shorter thread is clamped (null when nothing of it is left). */
export function followCount(win: RowWindow, previous: number, count: number): RowWindow | null {
  if (win.start >= count) return null;
  const end = win.end >= previous || win.end > count ? count : win.end;
  const next = withPins({ start: win.start, end }, win.pins ?? [], count);
  return sameWindow(next, win) ? win : next;
}

/** The window a thread opens with: enough of the tail to cover the reach, by estimated height. */
export function initialRowWindow(
  count: number,
  heightOf: (index: number) => number,
  coverPx: number,
  maxRows = WINDOW_INITIAL_ROWS,
): RowWindow | null {
  let total = 0;
  for (let index = 0; index < count; index += 1) total += heightOf(index);
  if (!worthWindowing(count, total)) return null;
  let start = count;
  let covered = 0;
  let rows = 0;
  while (start > 0 && rows < maxRows && covered < coverPx) {
    start -= 1;
    const height = heightOf(start);
    if (height > 0) rows += 1;
    covered += height;
  }
  if (start <= 0) return null;
  return { start, end: count };
}

/** A mounted row's box, in px from the viewport's top edge. */
export type MountedRow = { index: number; top: number; bottom: number };

export type SlideInput = {
  win: RowWindow;
  count: number;
  /** The window's mounted rows in document order (not its pins). */
  rows: readonly MountedRow[];
  viewHeight: number;
  /** Top of the thread (its first spacer), in px from the viewport's top edge. */
  threadTop: number;
  /** Px the leading spacer carries beyond its rows' heights (see the glue's slack). */
  slack: number;
  heightOf: (index: number) => number;
  /** Rows mounted per commit at most; a commit is paid on the main thread. */
  maxStep: number;
  /** False keeps every mounted row (a selection, a held pointer). */
  trim: boolean;
  reach?: number;
  keep?: number;
};

/**
 * The window the reader's position asks for: extend towards whichever edge is within reach of the
 * viewport, drop rows past the keep distance, and re-centre when the viewport sits in a spacer
 * (a scrollbar drag or a jump outran the window). Pins are the caller's: the rows this leaves
 * out are unmounted unless the caller pins them.
 */
export function planSlide(input: SlideInput): RowWindow {
  const { win, count, rows, viewHeight, heightOf, maxStep } = input;
  const reach = input.reach ?? WINDOW_REACH_PX;
  const keep = input.keep ?? WINDOW_KEEP_PX;
  let start = Math.max(0, Math.min(win.start, count));
  let end = Math.max(start, Math.min(win.end, count));

  const first = rows[0];
  const last = rows[rows.length - 1];

  if (first && last) {
    // The viewport left the mounted rows behind by more than the keep distance: nothing mounted
    // is worth keeping, so open a fresh window at the row the offset lands on. Walked from the
    // first row by the heights the spacers are sized from, which is where the row will be put.
    const above = first.top - viewHeight;
    const below = -last.bottom;
    if (above > keep || below > keep) {
      let index = 0;
      let offset = -input.threadTop - input.slack;
      while (index < count - 1) {
        const height = heightOf(index);
        if (offset < height) break;
        offset -= height;
        index += 1;
      }
      let covered = -offset;
      let next = index;
      let mounted = 0;
      while (next < count && mounted < maxStep && covered < viewHeight + reach) {
        const height = heightOf(next);
        if (height > 0) mounted += 1;
        covered += height;
        next += 1;
      }
      return { start: index, end: Math.max(next, index + 1) };
    }
  }

  let extendedUp = false;
  let extendedDown = false;
  if (start > 0) {
    let gap = (first ? first.top : 0) + reach;
    let mounted = 0;
    while (gap > 0 && start > 0 && mounted < maxStep) {
      start -= 1;
      const height = heightOf(start);
      if (height > 0) mounted += 1;
      gap -= height;
      extendedUp = true;
    }
  }
  if (end < count) {
    let gap = viewHeight + reach - (last ? last.bottom : viewHeight);
    let mounted = 0;
    while (gap > 0 && end < count && mounted < maxStep) {
      const height = heightOf(end);
      if (height > 0) mounted += 1;
      gap -= height;
      end += 1;
      extendedDown = true;
    }
  }

  if (input.trim) {
    if (!extendedUp) {
      for (const row of rows) {
        if (row.bottom >= -keep) break;
        start = Math.max(start, row.index + 1);
      }
    }
    if (!extendedDown) {
      for (let at = rows.length - 1; at >= 0; at -= 1) {
        const row = rows[at];
        if (row.top <= viewHeight + keep) break;
        end = Math.min(end, row.index);
      }
    }
  }
  if (end <= start) end = Math.min(count, start + 1);
  return { start, end };
}

/** Estimated height one pass of a full mount adds at most: about what a commit can lay out
 *  without holding input for long, however few rows that is. */
export const FILL_BUDGET_PX = 20_000;

/**
 * Widen both ways, a row at a time on alternate sides, by up to `chunk` rows each way and
 * `budgetPx` of estimated height in all, but always by one row; null once the window covers the
 * thread. The budget is what bounds a pass over very long replies, where `chunk` of them is
 * seconds of layout; short messages still go `chunk` at a time.
 */
export function planFill(
  win: RowWindow,
  count: number,
  chunk: number,
  heightOf: (index: number) => number = () => 0,
  budgetPx = FILL_BUDGET_PX,
): RowWindow | null {
  let start = Math.max(0, Math.min(win.start, count));
  let end = Math.max(start, Math.min(win.end, count));
  let up = 0;
  let down = 0;
  let spent = 0;
  let mounted = 0;
  for (;;) {
    const canUp = start > 0 && up < chunk;
    const canDown = end < count && down < chunk;
    if (!canUp && !canDown) break;
    const goUp = canUp && (!canDown || up <= down);
    const height = heightOf(goUp ? start - 1 : end);
    if (mounted > 0 && spent + height > budgetPx) break;
    if (goUp) {
      start -= 1;
      up += 1;
    } else {
      end += 1;
      down += 1;
    }
    spent += height;
    if (height > 0) mounted += 1;
  }
  if (start <= 0 && end >= count) return null;
  return { start, end };
}

/**
 * The window to re-open over a fully mounted thread, or null when the thread is short or too
 * little would unmount to be worth a commit.
 */
export function planEngage(
  rows: readonly MountedRow[],
  count: number,
  viewHeight: number,
  keep = WINDOW_KEEP_PX,
): RowWindow | null {
  const first = rows[0];
  const last = rows[rows.length - 1];
  if (!first || !last || !worthWindowing(count, last.bottom - first.top)) return null;
  let start = 0;
  let end = count;
  let keptTop = first.top;
  let keptBottom = last.bottom;
  for (const row of rows) {
    if (row.bottom >= -keep) {
      keptTop = row.top;
      break;
    }
    start = row.index + 1;
  }
  for (let at = rows.length - 1; at >= 0; at -= 1) {
    const row = rows[at];
    if (row.top <= viewHeight + keep) {
      keptBottom = row.bottom;
      break;
    }
    end = row.index;
  }
  const freed = keptTop - first.top + (last.bottom - keptBottom);
  if (end <= start || freed < keep) return null;
  return { start, end };
}

function samePins(a: readonly number[] = [], b: readonly number[] = []): boolean {
  return a.length === b.length && a.every((pin, at) => pin === b[at]);
}

/** Rows a tail window may grow by to reach the prompt it answers. */
export const TAIL_TURN_MAX_ROWS = 4;

/** A tail window also holds the last prompt: a reply taller than the keep distance would drop it. */
export function holdTailTurn(
  win: RowWindow | null,
  count: number,
  tailTurn: number,
): RowWindow | null {
  if (win == null || win.end < count || tailTurn < 0 || win.start <= tailTurn) return win;
  if (win.start - tailTurn > TAIL_TURN_MAX_ROWS) return win;
  return withPins({ start: tailTurn, end: win.end }, win.pins ?? [], count);
}

export const sameWindow = (a: RowWindow | null, b: RowWindow | null): boolean =>
  a === b ||
  (a != null && b != null && a.start === b.start && a.end === b.end && samePins(a.pins, b.pins));

/**
 * Where the leading spacer must end up so the reader's anchor does not move: the spacer takes the
 * shift, and only what it cannot take (it would go negative, or the window reached the first row
 * and it must be zero) is left for a scroll correction.
 */
export function absorbShift(
  spacerPx: number,
  shiftPx: number,
  atFirstRow: boolean,
): { spacer: number; scroll: number } {
  if (atFirstRow) return { spacer: 0, scroll: shiftPx - spacerPx };
  const wanted = spacerPx - shiftPx;
  if (wanted >= 0) return { spacer: wanted, scroll: 0 };
  return { spacer: 0, scroll: -wanted };
}
