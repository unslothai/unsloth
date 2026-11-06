// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// With the fold preference on, one Thinking block holds a span's run-up to the answer: its
// thoughts, tool calls and later thinking. Only non-blank answer text ends a span.

interface PartLike {
  readonly type: string;
  readonly text?: unknown;
}

export function reasoningRoundKey(messageId: string, endIndex: number): string {
  return `${messageId}:${endIndex}`;
}

/** A text part with nothing to read in it. Text of an unexpected shape is not assumed blank. */
export function isBlankTextPart(part: PartLike | undefined): boolean {
  return (
    part?.type === "text" &&
    typeof part.text === "string" &&
    part.text.trim() === ""
  );
}

function belongsToRun(part: PartLike | undefined): boolean {
  const type = part?.type;
  return type === "reasoning" || type === "tool-call" || isBlankTextPart(part);
}

/** `[start, end)` of the run of thinking, tool calls and blank text around `index`, or null
 *  when `index` is answer text. */
export function foldRun(
  parts: readonly PartLike[],
  index: number,
): { start: number; end: number } | null {
  if (index < 0 || index >= parts.length || !belongsToRun(parts[index])) {
    return null;
  }
  let start = index;
  while (start > 0 && belongsToRun(parts[start - 1])) start -= 1;
  let end = index + 1;
  while (end < parts.length && belongsToRun(parts[end])) end += 1;
  return { start, end };
}

/** Last part of the first reasoning group in the run around `index`: the block that heads it.
 *  Null when the run has no thinking, or `index` is not in a run. */
export function leadReasoningEnd(
  parts: readonly PartLike[],
  index: number,
): number | null {
  const run = foldRun(parts, index);
  if (run === null) return null;
  let end: number | null = null;
  for (let i = run.start; i < run.end; i += 1) {
    if (parts[i]?.type === "reasoning") {
      end = i;
      continue;
    }
    if (end !== null) break;
  }
  return end;
}

export function foldEnd(parts: readonly PartLike[], leadEnd: number): number {
  return foldRun(parts, leadEnd)?.end ?? leadEnd + 1;
}

/** The block a group at `startIndex` folds under: its run's lead, when the group comes after
 *  it. Null for the lead itself, for calls with no thinking before them in the run, and for
 *  anything outside a run. */
export function governingReasoningEnd(
  parts: readonly PartLike[],
  startIndex: number,
): number | null {
  const lead = leadReasoningEnd(parts, startIndex);
  if (lead === null || startIndex <= lead) return null;
  return lead;
}

export function isFoldedReasoningGroup(
  parts: readonly PartLike[],
  startIndex: number,
): boolean {
  return governingReasoningEnd(parts, startIndex) !== null;
}

/** Tool calls the lead holds; runs kept visible (tool-fold-exemptions.ts) are excluded. */
export function countFoldedToolParts(
  parts: readonly PartLike[],
  leadEnd: number,
  runIsExempt: (start: number, end: number) => boolean = () => false,
): number {
  const end = foldEnd(parts, leadEnd);
  let count = 0;
  let i = leadEnd + 1;
  while (i < end) {
    if (parts[i]?.type !== "tool-call") {
      i += 1;
      continue;
    }
    let runEnd = i;
    while (runEnd + 1 < end && parts[runEnd + 1]?.type === "tool-call") {
      runEnd += 1;
    }
    if (!runIsExempt(i, runEnd)) count += runEnd - i + 1;
    i = runEnd + 1;
  }
  return count;
}

/** Sum of the run's reasoning durations, skipping unknown rounds; undefined if none known. */
export function foldedTurnDuration(
  parts: readonly PartLike[],
  leadEnd: number,
  resolve: (
    parts: readonly PartLike[],
    startIndex: number,
  ) => number | undefined,
): number | undefined {
  const run = foldRun(parts, leadEnd);
  if (run === null) return undefined;
  let total: number | undefined;
  let previousWasReasoning = false;
  for (let i = run.start; i < run.end; i += 1) {
    const isReasoning = parts[i]?.type === "reasoning";
    if (isReasoning && !previousWasReasoning) {
      const duration = resolve(parts, i);
      if (duration !== undefined) total = (total ?? 0) + duration;
    }
    previousWasReasoning = isReasoning;
  }
  return total;
}

export function endsFoldedSpan(
  parts: readonly PartLike[],
  endIndex: number,
): boolean {
  const run = foldRun(parts, endIndex);
  if (run === null || run.end >= parts.length) return false;
  if (isBlankTextPart(parts[endIndex])) return false;
  if (leadReasoningEnd(parts, endIndex) === null) return false;
  let last = run.end - 1;
  while (last > endIndex && isBlankTextPart(parts[last])) last -= 1;
  return last === endIndex;
}

/** Closed-header summary, worded like the tool group trigger so the count matches. */
export function foldedToolSummary(count: number): string | null {
  if (count <= 0) return null;
  return `${count} tool ${count === 1 ? "call" : "calls"}`;
}
