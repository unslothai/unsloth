// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export interface WaveRange {
  start_s: number;
  end_s: number;
}

export interface RangeLimits {
  durationS: number;
  maxRanges: number;
  beyondEndS?: number;
}

export const MIN_RANGE_S = 0.25;
export const NUDGE_S = 0.1;
export const NUDGE_LARGE_S = 1;

const round = (seconds: number) => Math.round(seconds * 100) / 100;

export function timelineEnd(limits: RangeLimits): number {
  const duration = Math.max(0, limits.durationS);
  return duration + Math.max(0, limits.beyondEndS ?? 0);
}

export function pxToSeconds(
  px: number,
  widthPx: number,
  timelineS: number,
): number {
  if (!(widthPx > 0) || !(timelineS > 0)) return 0;
  return Math.min(timelineS, Math.max(0, (px / widthPx) * timelineS));
}

export function secondsToPx(
  seconds: number,
  widthPx: number,
  timelineS: number,
): number {
  if (!(timelineS > 0)) return 0;
  return (seconds / timelineS) * widthPx;
}

export function secondsToFraction(seconds: number, timelineS: number): number {
  if (!(timelineS > 0)) return 0;
  return Math.min(1, Math.max(0, seconds / timelineS));
}

export function clampRange(
  range: WaveRange,
  limits: RangeLimits,
): WaveRange | null {
  const end = timelineEnd(limits);
  if (!(end >= MIN_RANGE_S)) return null;
  if (!(Number.isFinite(range.start_s) && Number.isFinite(range.end_s)))
    return null;
  let start = Math.min(range.start_s, range.end_s);
  let stop = Math.max(range.start_s, range.end_s);
  start = Math.min(Math.max(0, start), end);
  stop = Math.min(Math.max(0, stop), end);
  if (stop - start < MIN_RANGE_S) {
    stop = start + MIN_RANGE_S;
    if (stop > end) {
      stop = end;
      start = end - MIN_RANGE_S;
    }
  }
  return { start_s: round(start), end_s: round(stop) };
}

export function mergeRanges(
  ranges: readonly WaveRange[],
  limits: RangeLimits,
): WaveRange[] {
  const clamped = ranges
    .map((range) => clampRange(range, limits))
    .filter((range): range is WaveRange => range !== null)
    .sort((a, b) => a.start_s - b.start_s || a.end_s - b.end_s);
  const merged: WaveRange[] = [];
  for (const range of clamped) {
    const last = merged[merged.length - 1];
    if (last && range.start_s <= last.end_s) {
      last.end_s = Math.max(last.end_s, range.end_s);
    } else {
      merged.push({ ...range });
    }
  }
  return merged;
}

export function normalizeRanges(
  ranges: readonly WaveRange[],
  limits: RangeLimits,
): WaveRange[] {
  return mergeRanges(ranges, limits).slice(
    0,
    Math.max(0, Math.floor(limits.maxRanges)),
  );
}

export function addRange(
  ranges: readonly WaveRange[],
  range: WaveRange,
  limits: RangeLimits,
): WaveRange[] {
  if (limits.maxRanges <= 0) return [];
  if (limits.maxRanges === 1) return normalizeRanges([range], limits);
  const next = mergeRanges([...ranges, range], limits);
  return next.length > limits.maxRanges
    ? normalizeRanges(ranges, limits)
    : next;
}

export function removeRange(
  ranges: readonly WaveRange[],
  index: number,
): WaveRange[] {
  return ranges.filter((_, at) => at !== index);
}

export type RangeEdge = "start" | "end" | "both";

export function moveRange(
  ranges: readonly WaveRange[],
  index: number,
  edge: RangeEdge,
  deltaS: number,
  limits: RangeLimits,
): WaveRange[] {
  const range = ranges[index];
  if (!range) return [...ranges];
  const end = timelineEnd(limits);
  let next: WaveRange;
  if (edge === "both") {
    const width = range.end_s - range.start_s;
    const start = Math.min(
      Math.max(0, range.start_s + deltaS),
      Math.max(0, end - width),
    );
    next = { start_s: start, end_s: start + width };
  } else if (edge === "start") {
    next = {
      start_s: Math.min(range.start_s + deltaS, range.end_s - MIN_RANGE_S),
      end_s: range.end_s,
    };
  } else {
    next = {
      start_s: range.start_s,
      end_s: Math.max(range.end_s + deltaS, range.start_s + MIN_RANGE_S),
    };
  }
  const rest = ranges.map((item, at) => (at === index ? next : item));
  return normalizeRanges(rest, limits);
}

export function nudgeStep(event: { key: string; shiftKey?: boolean }):
  | number
  | null {
  const size = event.shiftKey ? NUDGE_LARGE_S : NUDGE_S;
  switch (event.key) {
    case "ArrowRight":
    case "ArrowUp":
      return size;
    case "ArrowLeft":
    case "ArrowDown":
      return -size;
    default:
      return null;
  }
}

export function dragToRange(
  anchorS: number,
  currentS: number,
  limits: RangeLimits,
): WaveRange | null {
  if (Math.abs(currentS - anchorS) < MIN_RANGE_S / 2) return null;
  return clampRange({ start_s: anchorS, end_s: currentS }, limits);
}

export function defaultRange(limits: RangeLimits): WaveRange | null {
  const duration = Math.max(0, limits.durationS);
  const width = Math.max(MIN_RANGE_S, duration / 5);
  return clampRange(
    { start_s: (duration - width) / 2, end_s: (duration + width) / 2 },
    limits,
  );
}

export function nextFreeRange(
  ranges: readonly WaveRange[],
  limits: RangeLimits,
): WaveRange | null {
  const duration = Math.max(0, limits.durationS);
  if (duration < MIN_RANGE_S || ranges.length >= limits.maxRanges) return null;
  if (ranges.length === 0) return defaultRange(limits);
  const sorted = [...ranges].sort((a, b) => a.start_s - b.start_s);
  let best: { start: number; end: number } | null = null;
  let cursor = 0;
  for (const range of [...sorted, { start_s: duration, end_s: duration }]) {
    const end = Math.min(range.start_s, duration);
    if (end - cursor > (best ? best.end - best.start : 0))
      best = { start: cursor, end };
    cursor = Math.max(cursor, range.end_s);
  }
  if (!best || best.end - best.start < MIN_RANGE_S + 0.2) return null;
  const gap = best.end - best.start - 0.2;
  const width = Math.max(MIN_RANGE_S, Math.min(gap, duration / 5));
  const start = best.start + 0.1 + (gap - width) / 2;
  return clampRange({ start_s: start, end_s: start + width }, limits);
}

export function secondsBeyondEnd(range: WaveRange, durationS: number): number {
  return Math.max(0, round(range.end_s - durationS));
}

export function formatRangeTime(seconds: number): string {
  const tenths = Math.max(0, Math.round(seconds * 10));
  const minutes = Math.floor(tenths / 600);
  const rest = (tenths % 600) / 10;
  return `${minutes}:${rest.toFixed(1).padStart(4, "0")}`;
}

export function formatRange(range: WaveRange): string {
  return `${formatRangeTime(range.start_s)}–${formatRangeTime(range.end_s)}`;
}

export function rangeLabel(index: number, range: WaveRange): string {
  const say = (seconds: number) => String(Math.round(seconds * 10) / 10);
  return `Range ${index + 1}, ${say(range.start_s)} to ${say(range.end_s)} seconds`;
}
