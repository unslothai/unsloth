// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

type MessagePartLike = {
  type?: unknown;
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

type ActiveReasoningGroup = {
  index: number;
  startedAt: number;
};

export function createReasoningDurationTracker(
  now: () => number = Date.now,
) {
  let durations: number[] = [];
  let activeGroup: ActiveReasoningGroup | null = null;
  let groupCount = 0;
  let serverSummaryTargetIndex: number | null = null;

  const setDuration = (index: number, duration: number) => {
    if (durations[index] === duration) {
      return;
    }
    const next = [...durations];
    next[index] = duration;
    durations = next;
  };
  const finishGroupAt = (finishedAt: number) => {
    if (activeGroup === null) {
      return;
    }
    const { index, startedAt } = activeGroup;
    activeGroup = null;
    if (durations[index] === undefined) {
      setDuration(
        index,
        Math.max(0, Math.round((finishedAt - startedAt) / 1000)),
      );
    }
  };

  return {
    get groupCount() {
      return groupCount;
    },
    get hasActiveGroup() {
      return activeGroup !== null;
    },
    startGroup(index = groupCount) {
      if (activeGroup?.index === index) {
        return;
      }
      const startedAt = now();
      finishGroupAt(startedAt);
      activeGroup = { index, startedAt };
      groupCount = Math.max(groupCount, index + 1);
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
