// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Sliding mount window maths; the DOM and React glue is progressive-messages.tsx.

import { findCodeBlockRegions } from "../../lib/latex.ts";

/** `pins` (ascending): rows outside `[start, end)` the reader is working in, kept where they are. */
export type RowWindow = { start: number; end: number; pins?: readonly number[] };

export const WINDOW_REACH_PX = 2000;
/** Unmount distance; the gap to REACH is hysteresis. */
export const WINDOW_KEEP_PX = 4000;
export const WINDOW_INITIAL_ROWS = 16;
/** By height, not count: an agent chat is often a few turns of very long replies. */
export const WINDOW_MIN_TOTAL_PX = 40_000;
export const WINDOW_MIN_ROWS = 8;

export const worthWindowing = (count: number, totalPx: number): boolean =>
  count >= WINDOW_MIN_ROWS && totalPx >= WINDOW_MIN_TOTAL_PX;

export type RowMeta = {
  id: string;
  role: string;
  /** Expected rendered lines; only its proportion to height matters. */
  weight: number;
  renders: boolean;
  equations?: number;
  /** The content measured: an edit (another tab, a reload) voids what was read off the page. */
  revision?: unknown;
};

/** Padding, action bar and gap: a row's height before its first line. */
const BASE_PX: Record<string, number> = { user: 96, assistant: 72 };
const DEFAULT_BASE_PX = 72;
const PRIOR_PX_PER_LINE = 24;
const PRIOR_LINES = 8;
const MIN_ROW_PX = 24;
const CHARS_PER_LINE = 90;
/** Tool calls and reasoning render collapsed. */
const COLLAPSED_PART_LINES = 2;
const MEDIA_PART_LINES = 12;

type Part = { type?: string; text?: string; argsText?: string; result?: unknown };

/** Thinking and tool cards the "Always expanded" settings open on every message. */
export type Expanded = { reasoning: boolean; tools: boolean };
const COLLAPSED: Expanded = { reasoning: false, tools: false };

/** Lines, not characters: code (many short lines) must not be estimated as a paragraph. */
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

export function messageWeight(
  content: readonly Part[] | undefined,
  expanded: Expanded = COLLAPSED,
): number {
  if (!content) return 0;
  let weight = 0;
  for (const part of content) {
    if (part.type === "text") weight += part.text ? textLines(part.text) : 0;
    else if (part.type === "image" || part.type === "file") weight += MEDIA_PART_LINES;
    else if (part.type === "source") continue;
    else {
      weight += COLLAPSED_PART_LINES;
      if (part.type === "reasoning" && expanded.reasoning && part.text) {
        weight += textLines(part.text);
      } else if (part.type === "tool-call" && expanded.tools) {
        if (part.argsText) weight += textLines(part.argsText);
        if (typeof part.result === "string") weight += textLines(part.result);
      }
    }
  }
  return weight;
}

const NUMBERED_ENV = /\\begin\{(equation|multline|align|gather|alignat|flalign|eqnarray)\}([\s\S]*?)\\end\{\1\}/g;
/** Display math only: KaTeX refuses these environments inline. */
const DISPLAY_MATH = /\$\$([\s\S]*?)\$\$|\\\[([\s\S]*?)\\\]/g;

/** Code renders verbatim, so an environment in it is an example, not math. The markdown
 *  pipeline's own regions: fenced (in containers too), indented and inline. */
function withoutCode(text: string): string {
  let prose = "";
  let cursor = 0;
  for (const [start, end] of findCodeBlockRegions(text)) {
    prose += `${text.slice(cursor, start)}\n`;
    cursor = end;
  }
  return prose + text.slice(cursor);
}

const TEX_COMMENT = /(?<!\\)%[^\n]*/g;
const UNNUMBERED = /\\(?:notag|nonumber|tag)\b/;
const ENV_EDGE = /\\(begin|end)\{[^}]*\}/y;

/** Split on `\\` at the environment's own level: a nested cases or matrix row takes no number. */
function topLevelRows(body: string): string[] {
  const rows: string[] = [];
  let depth = 0;
  let from = 0;
  for (let at = 0; at < body.length; at += 1) {
    const char = body[at];
    if (char === "{") depth += 1;
    else if (char === "}") depth -= 1;
    else if (char === "\\") {
      ENV_EDGE.lastIndex = at;
      const edge = ENV_EDGE.exec(body);
      if (edge) {
        depth += edge[1] === "begin" ? 1 : -1;
        at = ENV_EDGE.lastIndex - 1;
      } else if (body[at + 1] === "\\" && depth === 0) {
        rows.push(body.slice(from, at));
        from = at + 2;
        at += 1;
      } else {
        at += 1;
      }
    }
  }
  rows.push(body.slice(from));
  return rows;
}

/**
 * KaTeX numbers these on page-wide CSS counters, so a row left out of the DOM must still count
 * them: one per environment, one per top-level row of the multi-line ones, less tagged rows.
 */
export function numberedEquations(
  content: readonly Part[] | undefined,
  expanded: Expanded = COLLAPSED,
): number {
  let total = 0;
  for (const part of content ?? []) {
    // A trace renders markdown too, but is on the page only while open.
    const shown = part.type === "text" || (part.type === "reasoning" && expanded.reasoning);
    if (!shown || !part.text?.includes("\\begin{")) continue;
    const prose = withoutCode(part.text);
    for (const [, dollars, brackets] of prose.matchAll(DISPLAY_MATH)) {
      for (const [, env, source] of (dollars ?? brackets).matchAll(NUMBERED_ENV)) {
        // KaTeX drops a comment to the end of its line, `\notag` in it included.
        const body = source.replace(TEX_COMMENT, "");
        const rows = env === "equation" || env === "multline" ? [body] : topLevelRows(body);
        total += rows.filter((row) => row.trim() !== "" && !UNNUMBERED.test(row)).length;
      }
    }
  }
  return total;
}

export type Equations = { html: number; mathml: number };

type Density = { lines: number; px: number };

/**
 * Measured heights by message id; an unmeasured row is estimated from px per line seen so far per
 * role. A ratio of sums, not a fitted line: the first rows measured are alike, so a fit is undefined.
 */
export class RowHeights {
  private measured = new Map<string, { height: number; revision: unknown }>();
  private densities = new Map<string, Density>();
  private counted = new Map<string, { role: string; lines: number; px: number }>();
  private layout: string | null = null;

  /** A new width or typography re-wraps every row: forget every height. True when it did. */
  relayout(layout: string): boolean {
    if (layout === this.layout) return false;
    const was = this.layout;
    this.layout = layout;
    if (was === null) return false;
    this.measured.clear();
    this.densities.clear();
    this.counted.clear();
    // Opening or closing every trace adds or drops its equations too.
    this.equations.clear();
    return true;
  }

  private equations = new Map<string, { counted: Equations; revision: unknown }>();

  /** True when the height changed. */
  measure(meta: RowMeta, height: number): boolean {
    if (!(height >= 0)) return false;
    const prior = this.measured.get(meta.id);
    if (prior !== undefined && prior.height === height && prior.revision === meta.revision) {
      return false;
    }
    this.measured.set(meta.id, { height, revision: meta.revision });
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
    return true;
  }

  countEquations(meta: RowMeta, counted: Equations): void {
    this.equations.set(meta.id, { counted, revision: meta.revision });
  }

  /** Counter increments rows `[from, to)` would have made, for the spacer standing in for them. */
  equationsIn(metas: readonly RowMeta[], from: number, to: number): Equations {
    const sum = { html: 0, mathml: 0 };
    for (let index = from; index < Math.min(to, metas.length); index += 1) {
      const meta = metas[index];
      const entry = this.equations.get(meta.id);
      const known = entry !== undefined && entry.revision === meta.revision ? entry.counted : null;
      sum.html += known ? known.html : (meta.equations ?? 0);
      sum.mathml += known ? known.mathml : (meta.equations ?? 0);
    }
    return sum;
  }

  has(id: string): boolean {
    return this.measured.has(id);
  }

  heightOf(meta: RowMeta): number {
    if (!meta.renders) return 0;
    const known = this.measured.get(meta.id);
    if (known !== undefined && known.revision === meta.revision) return known.height;
    const density = this.densities.get(meta.role);
    const perLine = density ? density.px / density.lines : PRIOR_PX_PER_LINE;
    const base = BASE_PX[meta.role] ?? DEFAULT_BASE_PX;
    return Math.max(MIN_ROW_PX, Math.round(base + perLine * meta.weight));
  }
}

export function withPins(win: RowWindow, pins: readonly number[], count: number): RowWindow {
  const kept = pins.filter(
    (index) => index >= 0 && index < count && (index < win.start || index >= win.end),
  );
  return kept.length > 0
    ? { start: win.start, end: win.end, pins: kept }
    : { start: win.start, end: win.end };
}

/** A mounted row, or a spacer for rows `[from, to)`; a `gap` spacer separates pinned rows. */
export type ThreadSlot =
  | { kind: "row"; index: number }
  | { kind: "spacer"; name: "top" | "bottom" | "gap"; from: number; to: number };

/** Document order. A full mount still has its two (empty) spacers. */
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

/** `[topFrom, start)` and `[end, bottomTo)`: up to the nearest pin on each side. */
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

/** A window at the tail follows it; one past a shorter thread is clamped (null if nothing is left). */
export function followCount(win: RowWindow, previous: number, count: number): RowWindow | null {
  if (win.start >= count) return null;
  const end = win.end >= previous || win.end > count ? count : win.end;
  const next = withPins({ start: win.start, end }, win.pins ?? [], count);
  return sameWindow(next, win) ? win : next;
}

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

/** Px from the viewport's top edge. */
export type MountedRow = { index: number; top: number; bottom: number };

export type SlideInput = {
  win: RowWindow;
  count: number;
  /** Not its pins. */
  rows: readonly MountedRow[];
  viewHeight: number;
  threadTop: number;
  /** Px the leading spacer carries beyond its rows' heights. */
  slack: number;
  heightOf: (index: number) => number;
  maxStep: number;
  trim: boolean;
  reach?: number;
  keep?: number;
};

/** Extend toward an edge within reach, drop rows past keep, re-centre when the viewport is in a
 *  spacer. Pins are the caller's. */
export function planSlide(input: SlideInput): RowWindow {
  const { win, count, rows, viewHeight, heightOf, maxStep } = input;
  const reach = input.reach ?? WINDOW_REACH_PX;
  const keep = input.keep ?? WINDOW_KEEP_PX;
  let start = Math.max(0, Math.min(win.start, count));
  let end = Math.max(start, Math.min(win.end, count));

  const first = rows[0];
  const last = rows[rows.length - 1];

  if (first && last) {
    // A jump outran the window: open a fresh one where the offset lands, walked by the heights
    // the spacers are sized from.
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

export const FILL_BUDGET_PX = 20_000;

/** Widen both ways by up to `chunk` rows each and `budgetPx` in all (one row at least), so a pass
 *  over very long replies is not seconds of layout. Null once the window covers the thread. */
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

/** Null when the thread is short or too little would unmount to be worth a commit. */
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

/** The leading spacer takes an anchor shift; only what it cannot take is left for scrollTop. */
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
