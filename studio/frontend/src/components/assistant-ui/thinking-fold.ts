// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// A turn's tool calls render as siblings of the thinking block, not inside it, because the parts
// are grouped by type: reasoning runs become one group, tool runs another. Folding them away puts
// a run of tool calls under the thinking block that precedes it, which these functions identify.

interface PartLike {
  readonly type: string;
}

/** One thinking round, named by the message and the last part of its reasoning group. */
export function reasoningRoundKey(messageId: string, endIndex: number): string {
  return `${messageId}:${endIndex}`;
}

/** The round a run of tool parts belongs to: the reasoning group just before it. Null when the
 *  tools follow text or open the message, so there is no thinking block to fold them under. */
export function governingReasoningEnd(
  parts: readonly PartLike[],
  startIndex: number,
): number | null {
  for (let i = Math.min(startIndex, parts.length) - 1; i >= 0; i -= 1) {
    const type = parts[i]?.type;
    if (type === "reasoning") return i;
    if (type !== "tool-call") return null;
  }
  return null;
}

/** Tool parts in the round ending at `reasoningEnd`, for the count the collapsed header shows. */
export function countRoundToolParts(
  parts: readonly PartLike[],
  reasoningEnd: number,
): number {
  let count = 0;
  for (
    let i = reasoningEnd + 1;
    i < parts.length && parts[i]?.type === "tool-call";
    i += 1
  ) {
    count += 1;
  }
  return count;
}

/** What the header says it is holding while the tools are hidden. Same wording as the tool group
 *  trigger, so the count reads the same before and after it is unfolded. */
export function foldedToolSummary(count: number): string | null {
  if (count <= 0) return null;
  return `${count} tool ${count === 1 ? "call" : "calls"}`;
}
