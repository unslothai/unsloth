// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// With the fold preference on, one Thinking block holds a turn's whole run-up to the answer: its
// own thoughts, every tool call, and any thinking that follows those calls. The parts are grouped
// by type, so those pieces render as siblings of the block; these helpers say which of them it
// holds and which stay where they are.

interface PartLike {
  readonly type: string;
}

/** One Thinking block, named by the message and the last part of its reasoning group. */
export function reasoningRoundKey(messageId: string, endIndex: number): string {
  return `${messageId}:${endIndex}`;
}

/** Last part of the first reasoning group, the block that heads the fold. Null when the message
 *  answers before it thinks, or never thinks. */
export function leadReasoningEnd(parts: readonly PartLike[]): number | null {
  let end: number | null = null;
  for (let i = 0; i < parts.length; i += 1) {
    const type = parts[i]?.type;
    if (type === "reasoning") {
      end = i;
      continue;
    }
    if (end !== null) return end;
    if (type !== "tool-call") return null;
  }
  return end;
}

/** Index of the first part after the lead that is neither thinking nor a tool call: where the
 *  answer starts. Everything between the lead and it folds under the lead. */
export function foldEnd(parts: readonly PartLike[], leadEnd: number): number {
  let i = leadEnd + 1;
  while (i < parts.length) {
    const type = parts[i]?.type;
    if (type !== "reasoning" && type !== "tool-call") break;
    i += 1;
  }
  return i;
}

/** The block a group at `startIndex` folds under: the lead, while the group comes before the
 *  answer. Null for the lead itself, for calls with no thinking before them, and for anything
 *  after the answer has started. */
export function governingReasoningEnd(
  parts: readonly PartLike[],
  startIndex: number,
): number | null {
  const lead = leadReasoningEnd(parts);
  if (lead === null || startIndex <= lead) return null;
  return startIndex < foldEnd(parts, lead) ? lead : null;
}

/** A later reasoning group that shows inside the lead instead of heading its own block. */
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
  runIsExempt: (start: number, end: number) => boolean = () => false,
): number {
  const lead = leadReasoningEnd(parts);
  if (lead === null) return 0;
  const end = foldEnd(parts, lead);
  let count = 0;
  let i = lead + 1;
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

/** Time the lead reports: every reasoning group it holds, added up. Rounds with no saved
 *  duration are left out rather than zeroing the total: a reply saved with only the legacy
 *  last-round duration still reports that. Undefined only when no round is known, so the
 *  caller falls back to its own clock. */
export function foldedTurnDuration(
  parts: readonly PartLike[],
  resolve: (
    parts: readonly PartLike[],
    startIndex: number,
  ) => number | undefined,
): number | undefined {
  const lead = leadReasoningEnd(parts);
  if (lead === null) return undefined;
  const end = foldEnd(parts, lead);
  let total: number | undefined;
  let previousWasReasoning = false;
  for (let i = 0; i < end; i += 1) {
    const isReasoning = parts[i]?.type === "reasoning";
    if (isReasoning && !previousWasReasoning) {
      const duration = resolve(parts, i);
      if (duration !== undefined) total = (total ?? 0) + duration;
    }
    previousWasReasoning = isReasoning;
  }
  return total;
}

/** Whether the part at `endIndex` is the last thing shown under the lead and the answer comes
 *  right after it: where the rule that closes the trace goes. */
export function endsFoldedSpan(
  parts: readonly PartLike[],
  endIndex: number,
): boolean {
  const lead = leadReasoningEnd(parts);
  if (lead === null) return false;
  return (
    foldEnd(parts, lead) === endIndex + 1 &&
    parts[endIndex + 1]?.type === "text"
  );
}

/** What the header says it is holding while closed. Same wording as the tool group trigger, so
 *  the count reads the same once the block is open. */
export function foldedToolSummary(count: number): string | null {
  if (count <= 0) return null;
  return `${count} tool ${count === 1 ? "call" : "calls"}`;
}
