// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Geometry shared by the Benchmarks charts. No React here, so the layout is unit tested.

export const W = 1000;
export const LABEL_W = 300;
export const RIGHT_W = 170;
export const TOP = 18;
export const AXIS_H = 52;
export const FOOT_LINE = 17;
export const FONT = "Inter Variable, ui-sans-serif, system-ui, sans-serif";
/** Height of a group's heading row in a grouped bar chart. */
export const GROUP_H = 30;

/** SVG text never wraps: break long footer lines on their " · " joins, about 160 chars at 11px. */
const FOOT_CHARS = 160;
export function wrapFooter(lines: string[]): string[] {
  const out: string[] = [];
  for (const line of lines) {
    let cur = "";
    for (const part of line.split(" · ")) {
      const next = cur ? `${cur} · ${part}` : part;
      if (cur && next.length > FOOT_CHARS) {
        out.push(cur);
        cur = part;
      } else cur = next;
    }
    if (cur) out.push(cur);
  }
  return out;
}

export function niceMax(max: number): number {
  if (max <= 0) return 1;
  const base = 10 ** Math.floor(Math.log10(max));
  for (const m of [1, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10])
    if (m * base >= max * 1.08) return m * base;
  return 10 * base;
}

export function ticks(max: number): number[] {
  return Array.from({ length: 6 }, (_, i) =>
    Number(((i * max) / 5).toFixed(6)),
  );
}

export function tickText(t: number): string {
  return Number.isInteger(t) ? String(t) : t.toFixed(1);
}

/** A bar anchored flat at the baseline with only its data end rounded. */
export function barPath(
  x0: number,
  x1: number,
  y: number,
  h: number,
  r: number,
): string {
  const rr = Math.min(r, Math.max(0, x1 - x0), h / 2);
  return `M${x0},${y} H${x1 - rr} a${rr},${rr} 0 0 1 ${rr},${rr} V${y + h - rr} a${rr},${rr} 0 0 1 -${rr},${rr} H${x0} Z`;
}

export function clip(s: string, n: number): string {
  return s.length > n ? `${s.slice(0, n - 1)}…` : s;
}

/** Rows tighten as a chart fills up, so twenty settings still fit a screen. */
export function barDensity(count: number): {
  rowH: number;
  barH: number;
  fontSize: number;
} {
  return {
    rowH: count > 14 ? 28 : count > 9 ? 34 : 46,
    barH: count > 14 ? 17 : count > 9 ? 21 : 26,
    fontSize: count > 14 ? 12.5 : 13.5,
  };
}

export interface BarLayout {
  rowH: number;
  barH: number;
  fontSize: number;
  /** Top of each group's heading, or null for an unlabelled group. */
  headTops: (number | null)[];
  /** Top of each row's band, per group. */
  rowTops: number[][];
  pendingTops: number[];
  /** Bottom of the last row, where the axis starts. */
  bottom: number;
  plotH: number;
}

/** Vertical placement of every row; an unlabelled group takes no heading space. */
export function layoutBars(
  groups: { labelled: boolean; rows: number }[],
  pending: number,
): BarLayout {
  const count = groups.reduce((n, g) => n + g.rows, 0) + pending;
  const { rowH, barH, fontSize } = barDensity(count);
  let y = TOP;
  const headTops: (number | null)[] = [];
  const rowTops: number[][] = [];
  for (const g of groups) {
    headTops.push(g.labelled ? y : null);
    if (g.labelled) y += GROUP_H;
    rowTops.push(Array.from({ length: g.rows }, (_, i) => y + i * rowH));
    y += g.rows * rowH;
  }
  const pendingTops = Array.from({ length: pending }, (_, i) => y + i * rowH);
  const bottom = y + pending * rowH;
  return {
    rowH,
    barH,
    fontSize,
    headTops,
    rowTops,
    pendingTops,
    bottom,
    plotH: bottom + AXIS_H,
  };
}

/** A whisker clamped to the axis, or null when there is nothing to draw. */
export function clampWhisker(
  low: number | null | undefined,
  high: number | null | undefined,
  max: number,
): [number, number] | null {
  if (low == null || high == null) return null;
  const lo = Math.min(max, Math.max(0, low));
  const hi = Math.min(max, Math.max(0, high));
  return hi > lo ? [lo, hi] : null;
}
