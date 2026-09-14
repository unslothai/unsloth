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

/** `seed` is what a reader that arrives mid-run already holds: the durations the tab that started
 *  the run measured for the groups that closed before this one attached, and therefore how many
 *  groups already exist. A replay cannot re-measure those -- the frames before its cursor were never
 *  folded -- so it inherits them, and a group that opens after it attaches takes the NEXT index
 *  instead of overwriting a finished group's slot.
 *
 *  How many groups exist is NOT the same as which of them are FINISHED. Consecutive closed reasoning
 *  blocks coalesce into one rendered group, so a tab that shut mid-thought leaves a number for a group this
 *  reader watches GROW, and `lastGroupTextLength` -- how much of that last group the closing tab had already
 *  read -- is the only thing that tells "the reader folded more of the same thought" from "the answer started
 *  streaming". Without it the last seeded group has no `startedAt` to resume from, its stale pre-close number
 *  is all anyone will ever know about it, and a server summary arriving afterwards has no target either. */
export function createReasoningDurationTracker(
  now: () => number = Date.now,
  seed?: { durations?: readonly number[]; lastGroupTextLength?: number },
) {
  let durations: number[] = (seed?.durations ?? []).map(
    (duration) => asDuration(duration) ?? 0,
  );
  // First time each group index became visible. A group can be closed and reopened -- several
  // complete <think>...</think> blocks in a row are coalesced into one rendered group -- so the
  // duration is always measured from the first sighting, not the last.
  const startedAt: number[] = [];
  let activeIndex: number | null = null;
  // Seeded from what the closing tab already measured, so a replay counts the groups it inherited and
  // opens the NEXT one; the ones it never saw a frame of keep the value they arrived with.
  let groupCount = durations.length;
  // What each group cost where the tab before this one stopped measuring. This reader's own window starts at
  // the first frame IT folded, so the two windows sit end to end and the group's total is their sum -- which is
  // why measuring a resumed group adds to this instead of writing over it.
  const inherited = [...durations];
  // Reasoning text seen so far per group, used to decide whether a closed group is still growing and should reopen.
  const reasoningLength: number[] = [];
  // The seeded group that may still have been open when the other tab shut. Only the LAST one can be: anything
  // before it is separated from the running text by a part of another kind, so nothing more can ever join it.
  const seenByClosingTab = seed?.lastGroupTextLength;
  const resumableIndex =
    seenByClosingTab === undefined ? -1 : durations.length - 1;
  if (resumableIndex >= 0 && seenByClosingTab !== undefined) {
    reasoningLength[resumableIndex] = seenByClosingTab;
  }
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
    // An inherited group keeps what the closing tab measured; only the part THIS reader watched is new.
    const measured = Math.max(0, Math.round((finishedAt - from) / 1000));
    setDuration(index, (inherited[index] ?? 0) + measured);
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
      if (activeIndex === index) {
        return;
      }
      if (startedAt[index] === undefined) {
        // A group that arrived with the seed has no first sighting here. It resumes from the frame this tab
        // actually saw it grow on, and keeps counting on top of the value it arrived with.
        if (index !== resumableIndex) {
          return;
        }
        startedAt[index] = now();
      }
      finishGroupAt(now());
      activeIndex = index;
      // A summary arriving now describes THIS group, seeded number or no: it is the pass running now.
      serverSummaryTargetIndex = index;
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
