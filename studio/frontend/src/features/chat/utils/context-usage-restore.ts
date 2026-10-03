// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export type PersistedContextUsage = {
  promptTokens: number;
  completionTokens: number;
  totalTokens: number;
  cachedTokens: number;
  cacheWriteTokens?: number;
  modelId?: string;
};

function record(value: unknown): Record<string, unknown> | null {
  return value !== null && typeof value === "object"
    ? (value as Record<string, unknown>)
    : null;
}

function tokenCount(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value) && value >= 0;
}

function persistedContextUsage(
  metadata: unknown,
): PersistedContextUsage | null {
  const usage = record(record(metadata)?.contextUsage);
  if (!usage) {
    return null;
  }
  const requiredCounts = [
    usage.promptTokens,
    usage.completionTokens,
    usage.totalTokens,
    usage.cachedTokens,
  ];
  if (!requiredCounts.every(tokenCount)) {
    return null;
  }
  if (
    usage.cacheWriteTokens !== undefined &&
    !tokenCount(usage.cacheWriteTokens)
  ) {
    return null;
  }
  if (usage.modelId !== undefined && typeof usage.modelId !== "string") {
    return null;
  }
  const [promptTokens, completionTokens, totalTokens, cachedTokens] =
    requiredCounts as [number, number, number, number];

  return {
    promptTokens,
    completionTokens,
    totalTokens,
    cachedTokens,
    ...(usage.cacheWriteTokens !== undefined
      ? { cacheWriteTokens: usage.cacheWriteTokens }
      : {}),
    ...(usage.modelId ? { modelId: usage.modelId } : {}),
  };
}

/**
 * Return the exact usage saved on a completed assistant turn when it belongs to the model that is
 * active now. Local checkpoints intentionally start empty after an app restart, so callers retry
 * this after status hydration before falling back to a full-transcript estimate.
 */
export function restorableContextUsage(
  metadata: unknown,
  checkpoint: string,
  contextLength: number | null,
): PersistedContextUsage | null {
  if (!checkpoint) {
    return null;
  }

  const usage = persistedContextUsage(metadata);
  if (!usage) {
    return null;
  }
  const localWindowKnown = contextLength !== null && contextLength > 0;
  if (usage.modelId ? usage.modelId !== checkpoint : !localWindowKnown) {
    return null;
  }
  if (localWindowKnown && usage.totalTokens > contextLength) {
    return null;
  }

  return usage;
}
