// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// With the fold preference on, one Thinking block holds a span's whole run-up to the answer: its
// own thoughts, every tool call, and any thinking that follows those calls. The parts are grouped
// by type, so those pieces render as siblings of the block; these helpers say which of them it
// holds and which stay where they are.
//
// A span is a maximal run of thinking, tool calls and blank text, and is led by its first
// reasoning group. Only answer text with something in it ends a span: the whitespace a provider
// leaves between a closed think block and its tool call is not an answer, and neither is the
// previous reply a continuation is seeded with, which sits before the new round rather than
// ending it. A message can therefore hold several spans, each folded under its own lead.

interface PartLike {
  readonly type: string;
  readonly text?: unknown;
}

/** One Thinking block, named by the message and the last part of its reasoning group. */
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

/** Index just past the run the lead heads: where the answer starts, or the end of the parts. */
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

/** A later reasoning group that shows inside its lead instead of heading its own block. */
export function isFoldedReasoningGroup(
  parts: readonly PartLike[],
  startIndex: number,
): boolean {
  return governingReasoningEnd(parts, startIndex) !== null;
}

/** Tool calls the lead is holding, for its collapsed header. A run the thread keeps visible
 *  (see tool-fold-exemptions.ts) is left out, so the header never claims a call that is on
 *  screen below it. */
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

/** Time the lead reports: every reasoning group in its run, added up. Rounds with no saved
 *  duration are left out rather than zeroing the total: a reply saved with only the legacy
 *  last-round duration still reports that. Undefined only when no round is known, so the
 *  caller falls back to its own clock. */
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

/** Whether the part at `endIndex` is the last thing shown under its lead and the answer comes
 *  right after: where the rule that closes the trace goes. */
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

/** What the header says it is holding while closed. Same wording as the tool group trigger, so
 *  the count reads the same once the block is open. */
export function foldedToolSummary(count: number): string | null {
  if (count <= 0) return null;
  return `${count} tool ${count === 1 ? "call" : "calls"}`;
}
