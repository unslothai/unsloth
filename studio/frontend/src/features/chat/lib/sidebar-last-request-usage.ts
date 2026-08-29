// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { MessageRecord } from "../types";

/** The independently validated server total for the newest saved assistant request. */
export type SidebarLastRequestUsage = {
  totalTokens: number;
};

export type SidebarAssistantUsageUpdate =
  | { threadId: string; hasAssistant: false }
  | {
      threadId: string;
      hasAssistant: true;
      assistantId: string;
      createdAt: number;
      metadata?: MessageRecord["metadata"] | null;
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

export function selectSidebarLastRequestUsageFromMetadata(
  metadata: MessageRecord["metadata"] | null,
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
  const message = newestAssistantMessage(messages);
  return message
    ? selectSidebarLastRequestUsageFromMetadata(message.metadata)
    : undefined;
}

function newestAssistantMessage(
  messages: readonly MessageRecord[],
): MessageRecord | undefined {
  let newest: MessageRecord | undefined;
  for (const message of messages) {
    if (
      message.role === "assistant" &&
      (!newest ||
        message.createdAt > newest.createdAt ||
        (message.createdAt === newest.createdAt && message.id > newest.id))
    ) {
      newest = message;
    }
  }
  return newest;
}

export function newestSidebarAssistantUsageUpdate(
  threadId: string,
  messages: readonly MessageRecord[],
): SidebarAssistantUsageUpdate {
  const message = newestAssistantMessage(messages);
  if (message)
    return {
      threadId,
      hasAssistant: true,
      assistantId: message.id,
      createdAt: message.createdAt,
      metadata: message.metadata,
    };
  return { threadId, hasAssistant: false };
}

export function applySidebarAssistantUsageUpdate<
  T extends {
    id: string;
    sidebarLastRequestUsage?: SidebarLastRequestUsage;
    lastAssistantId?: string | null;
    lastAssistantCreatedAt?: number | null;
  },
>(threads: readonly T[], update: SidebarAssistantUsageUpdate): T[] {
  return threads.map((thread) => {
    if (thread.id !== update.threadId) return thread;
    if (!update.hasAssistant) {
      return {
        ...thread,
        sidebarLastRequestUsage: undefined,
        lastAssistantId: undefined,
        lastAssistantCreatedAt: undefined,
      };
    }
    const currentCreatedAt = thread.lastAssistantCreatedAt;
    const currentId = thread.lastAssistantId;
    if (
      typeof currentCreatedAt === "number" &&
      (update.createdAt < currentCreatedAt ||
        (update.createdAt === currentCreatedAt &&
          typeof currentId === "string" &&
          update.assistantId < currentId))
    ) {
      return thread;
    }
    return {
      ...thread,
      sidebarLastRequestUsage: selectSidebarLastRequestUsageFromMetadata(
        update.metadata ?? null,
      ),
      lastAssistantId: update.assistantId,
      lastAssistantCreatedAt: update.createdAt,
    };
  });
}
