// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { FENCE_CHUNK_LINES, type FenceLineChunks } from "./code-fence-chunks.ts";

export const HIGHLIGHT_BATCH_TOKENS = 512;
export const HIGHLIGHT_BATCH_LINES = 64;
export const HIGHLIGHT_TOKEN_CAP = 4096;

export type HighlightProgress = {
  counts: FenceLineChunks<number>;
  cursor: number;
  located: boolean;
};

type Lines = readonly { length: number }[];

export const EMPTY_HIGHLIGHT_COUNTS: readonly number[] = [];

export const EMPTY_HIGHLIGHT_PROGRESS: HighlightProgress = {
  counts: [], cursor: 0, located: false,
};

export const highlightLineCount = (progress: HighlightProgress, line: number): number =>
  progress.counts[Math.floor(line / FENCE_CHUNK_LINES)]?.[line % FENCE_CHUNK_LINES] ?? 0;

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
  let counts: (readonly number[])[] | null = null;
  const changed = new Map<number, number[]>();
  let remaining = HIGHLIGHT_BATCH_TOKENS;
  let touched = 0;
  let scanned = 0;
  while (scanned < lines.length) {
    const length = lines[cursor].length;
    const groupIndex = Math.floor(cursor / FENCE_CHUNK_LINES);
    const offset = cursor % FENCE_CHUNK_LINES;
    const before = counts?.[groupIndex]?.[offset] ?? highlightLineCount(progress, cursor);
    const missing = length - before;
    if (missing > 0) {
      if (remaining === 0 || touched === HIGHLIGHT_BATCH_LINES
        || (length <= HIGHLIGHT_BATCH_TOKENS && missing > remaining)) break;
      const groupCount = Math.ceil(lines.length / FENCE_CHUNK_LINES);
      counts ??= progress.counts.length === groupCount
        ? progress.counts.slice()
        : Array.from({ length: groupCount }, (_, index) => progress.counts[index] ?? EMPTY_HIGHLIGHT_COUNTS);
      let group = changed.get(groupIndex);
      if (!group) {
        const length = Math.min(FENCE_CHUNK_LINES, lines.length - groupIndex * FENCE_CHUNK_LINES);
        const previous = progress.counts[groupIndex];
        group = previous?.length === length
          ? previous.slice()
          : Array.from({ length }, (_, index) => previous?.[index] ?? 0);
        changed.set(groupIndex, group);
        counts[groupIndex] = group;
      }
      const added = Math.min(missing, remaining);
      group[offset] = before + added;
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
  const counts: (readonly number[])[] = [];
  for (let start = 0; start < next.length; start += FENCE_CHUNK_LINES) {
    const length = Math.min(FENCE_CHUNK_LINES, next.length - start);
    const oldGroup = progress?.counts[start / FENCE_CHUNK_LINES];
    let group: number[] | null = null;
    for (let offset = 0; offset < length; offset += 1) {
      const index = start + offset;
      const old = progress === null ? previous?.[index]?.length ?? 0 : oldGroup?.[offset] ?? 0;
      const count = Math.min(old, next[index].length);
      if (count < next[index].length) complete = false;
      if (count !== (oldGroup?.[offset] ?? 0)) {
        group ??= Array.from({ length }, (_, index) => oldGroup?.[index] ?? 0);
        group[offset] = count;
      }
    }
    counts.push(group ?? (oldGroup && oldGroup.length <= length ? oldGroup
      : oldGroup?.slice(0, length) ?? EMPTY_HIGHLIGHT_COUNTS));
  }
  if (complete) return null;
  return {
    counts,
    cursor: Math.min(progress?.cursor ?? 0, Math.max(0, next.length - 1)),
    located: progress?.located ?? false,
  };
}
