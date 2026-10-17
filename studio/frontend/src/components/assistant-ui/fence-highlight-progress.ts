// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export const HIGHLIGHT_BATCH_TOKENS = 512;
export const HIGHLIGHT_BATCH_LINES = 64;
export const HIGHLIGHT_TOKEN_CAP = 4096;

export type HighlightProgress = {
  counts: readonly number[];
  cursor: number;
  located: boolean;
};

type Lines = readonly { length: number }[];

export const EMPTY_HIGHLIGHT_PROGRESS: HighlightProgress = {
  counts: [], cursor: 0, located: false,
};

export function needsHighlightBatches(lines: Lines, lineCap: number): boolean {
  if (lines.length > lineCap) return true;
  let tokens = 0;
  for (const line of lines) {
    tokens += line.length;
    if (tokens > HIGHLIGHT_TOKEN_CAP) return true;
  }
  return false;
}

export function advanceHighlightProgress(
  lines: Lines,
  progress: HighlightProgress,
  visibleLine = 0,
): HighlightProgress | null {
  const requested = progress.located ? progress.cursor : visibleLine;
  let cursor = Number.isFinite(requested) && requested >= 0 && requested < lines.length
    ? Math.trunc(requested) : 0;
  let counts: number[] | null = null;
  let remaining = HIGHLIGHT_BATCH_TOKENS;
  let touched = 0;
  let scanned = 0;
  while (scanned < lines.length) {
    const length = lines[cursor].length;
    const before = counts?.[cursor] ?? progress.counts[cursor] ?? 0;
    const missing = length - before;
    if (missing > 0) {
      if (remaining === 0 || touched === HIGHLIGHT_BATCH_LINES
        || (length <= HIGHLIGHT_BATCH_TOKENS && missing > remaining)) break;
      counts ??= progress.counts.length === lines.length
        ? progress.counts.slice()
        : Array.from({ length: lines.length }, (_, index) => progress.counts[index] ?? 0);
      const added = Math.min(missing, remaining);
      counts[cursor] = before + added;
      remaining -= added;
      touched += 1;
      if (added < missing) break;
    }
    cursor = (cursor + 1) % lines.length;
    scanned += 1;
  }
  return scanned === lines.length
    ? null
    : { counts: counts ?? progress.counts, cursor, located: true };
}

export function initialHighlightProgress(lines: Lines): HighlightProgress | null {
  const progress = advanceHighlightProgress(lines, EMPTY_HIGHLIGHT_PROGRESS);
  return progress === null ? null : { ...progress, located: false };
}

export function retainHighlightProgress(
  previous: Lines | null,
  next: Lines,
  progress: HighlightProgress | null,
): HighlightProgress | null {
  let complete = true;
  const counts = next.map((line, index) => {
    const old = progress === null ? previous?.[index]?.length ?? 0 : progress.counts[index] ?? 0;
    const count = Math.min(old, line.length);
    if (count < line.length) complete = false;
    return count;
  });
  if (complete) return null;
  return {
    counts,
    cursor: Math.min(progress?.cursor ?? 0, Math.max(0, next.length - 1)),
    located: progress?.located ?? false,
  };
}
