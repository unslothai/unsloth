// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

type MessagePartLike = {
  type?: unknown;
  text?: unknown;
};

type ReasoningMetadata = {
  reasoningDuration?: unknown;
  reasoningDurations?: unknown;
};

function asDuration(value: unknown): number | undefined {
  return typeof value === "number" && Number.isFinite(value) && value >= 0
    ? value
    : undefined;
}

function getReasoningGroupIndex(
  parts: readonly MessagePartLike[],
  endIndex: number,
): number {
  let index = -1;
  let previousWasReasoning = false;

  const limit = Math.min(endIndex, parts.length - 1);
  for (let partIndex = 0; partIndex <= limit; partIndex += 1) {
    const isReasoning = parts[partIndex]?.type === "reasoning";
    if (isReasoning && !previousWasReasoning) {
      index += 1;
    }
    previousWasReasoning = isReasoning;
  }

  return index;
}

export function countReasoningGroups(
  parts: readonly MessagePartLike[],
): number {
  return getReasoningGroupIndex(parts, parts.length - 1) + 1;
}

/** Total reasoning text in the LAST reasoning group, the one any new reasoning would join. The
 *  adapter compares this across chunks to tell "still thinking" from "moved on to the answer":
 *  a provider that closes every reasoning block atomically would otherwise freeze the group's
 *  timer at its first close. */
export function lastReasoningGroupTextLength(
  parts: readonly MessagePartLike[],
): number {
  let total = 0;
  let inGroup = false;
  for (let index = parts.length - 1; index >= 0; index -= 1) {
    if (parts[index]?.type !== "reasoning") {
      if (inGroup) break;
      continue;
    }
    inGroup = true;
    const text = parts[index]?.text;
    total += typeof text === "string" ? text.length : 0;
  }
  return total;
}

export function resolveReasoningGroupDuration(
  parts: readonly MessagePartLike[],
  startIndex: number,
  custom: ReasoningMetadata | null | undefined,
): number | undefined {
  const index = getReasoningGroupIndex(parts, startIndex);
  if (index < 0) {
    return undefined;
  }

  if (Array.isArray(custom?.reasoningDurations)) {
    return asDuration(custom.reasoningDurations[index]);
  }

  if (index !== getReasoningGroupIndex(parts, parts.length - 1)) {
    return undefined;
  }
  return asDuration(custom?.reasoningDuration);
}

export function createReasoningDurationTracker(
  now: () => number = Date.now,
) {
  let durations: number[] = [];
  // First time each group index became visible. A group can be closed and reopened -- several
  // complete <think>...</think> blocks in a row are coalesced into one rendered group -- so the
  // duration is always measured from the first sighting, not the last.
  const startedAt: number[] = [];
  let activeIndex: number | null = null;
  let groupCount = 0;
  // Reasoning text seen so far per group, used to decide whether a closed group is still growing and should reopen.
  const reasoningLength: number[] = [];
  // The group a server summary would land on. The backend emits one summary at the end of each
  // visible reasoning pass, before the next can begin, so "the group that started most recently"
  // is the correct target. A FIFO queue mis-assigns as soon as one group has no summary.
  let serverSummaryTargetIndex: number | null = null;
  // Indices whose duration came from the server; local timing must not overwrite an authoritative value.
  const serverClaimed = new Set<number>();

  const setDuration = (index: number, duration: number) => {
    if (durations[index] === duration) {
      return;
    }
    const next = [...durations];
    next[index] = duration;
    durations = next;
  };
  const measure = (index: number, finishedAt: number) => {
    if (serverClaimed.has(index)) {
      return;
    }
    const from = startedAt[index];
    if (from === undefined) {
      return;
    }
    setDuration(index, Math.max(0, Math.round((finishedAt - from) / 1000)));
  };
  const finishGroupAt = (finishedAt: number) => {
    if (activeIndex === null) {
      return;
    }
    const index = activeIndex;
    activeIndex = null;
    measure(index, finishedAt);
  };

  return {
    get groupCount() {
      return groupCount;
    },
    get hasActiveGroup() {
      return activeIndex !== null;
    },
    startGroup(index = groupCount) {
      if (activeIndex === index) {
        return;
      }
      const at = now();
      finishGroupAt(at);
      // A single delta can reveal more than one group at once. Any index we skipped became visible and
      // closed within this same chunk, so give it a measured zero rather than leaving a hole.
      for (let skipped = groupCount; skipped < index; skipped += 1) {
        if (startedAt[skipped] === undefined) {
          startedAt[skipped] = at;
        }
        measure(skipped, at);
      }
      if (startedAt[index] === undefined) {
        startedAt[index] = at;
      }
      activeIndex = index;
      groupCount = Math.max(groupCount, index + 1);
      serverSummaryTargetIndex = index;
    },
    /** Reopen a group that already closed, but only while its reasoning text is still growing.
     *  Providers that emit each reasoning block as a complete chunk close the group every chunk;
     *  without this it would freeze at the first close. Gating on growth is what keeps the timer
     *  from running on into the answer. */
    resumeGroup(index: number, currentReasoningLength: number) {
      const seen = reasoningLength[index] ?? 0;
      if (currentReasoningLength <= seen) {
        return;
      }
      reasoningLength[index] = currentReasoningLength;
      if (activeIndex === index || startedAt[index] === undefined) {
        return;
      }
      finishGroupAt(now());
      activeIndex = index;
    },
    finishGroup() {
      finishGroupAt(now());
    },
    recordServerDuration(reasoningMs: unknown): boolean {
      if (
        typeof reasoningMs !== "number" ||
        !Number.isFinite(reasoningMs) ||
        reasoningMs < 0
      ) {
        return false;
      }
      if (serverSummaryTargetIndex !== null) {
        serverClaimed.add(serverSummaryTargetIndex);
        setDuration(
          serverSummaryTargetIndex,
          Math.max(0, Math.round(reasoningMs / 1000)),
        );
        serverSummaryTargetIndex = null;
      }
      return true;
    },
    metadata() {
      if (durations.length === 0) {
        return {};
      }
      return {
        reasoningDuration: durations.at(-1) ?? 0,
        reasoningDurations: durations,
      };
    },
  };
}
