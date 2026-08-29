// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { MessageRecord } from "../types";

/** The independently validated server total for the newest saved assistant request. */
export type SidebarLastRequestUsage = {
  totalTokens: number;
};

type ContextUsageRecord = {
  promptTokens?: unknown;
  completionTokens?: unknown;
  totalTokens?: unknown;
  cachedTokens?: unknown;
  cacheWriteTokens?: unknown;
};

function isFiniteNonNegativeNumber(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value) && value >= 0;
}

function isValidOptionalCounter(value: unknown): boolean {
  return value === undefined || isFiniteNonNegativeNumber(value);
}

function readContextUsage(
  metadata: MessageRecord["metadata"],
): SidebarLastRequestUsage | undefined {
  const candidate = metadata?.contextUsage;
  if (!candidate || typeof candidate !== "object" || Array.isArray(candidate)) {
    return undefined;
  }
  const usage = candidate as ContextUsageRecord;
  if (
    !isFiniteNonNegativeNumber(usage.promptTokens) ||
    !isFiniteNonNegativeNumber(usage.completionTokens) ||
    !isFiniteNonNegativeNumber(usage.totalTokens) ||
    !isValidOptionalCounter(usage.cachedTokens) ||
    !isValidOptionalCounter(usage.cacheWriteTokens)
  ) {
    return undefined;
  }
  // This remains the server-reported total: cache values and the component
  // counters are validation only, never ingredients in a recalculation.
  return { totalTokens: usage.totalTokens };
}

/**
 * Return usage for only the newest chronological assistant record. A newer
 * partial or legacy assistant result intentionally hides an older value.
 */
export function selectSidebarLastRequestUsage(
  messages: readonly MessageRecord[],
): SidebarLastRequestUsage | undefined {
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    const message = messages[index];
    if (message.role === "assistant") return readContextUsage(message.metadata);
  }
  return undefined;
}
