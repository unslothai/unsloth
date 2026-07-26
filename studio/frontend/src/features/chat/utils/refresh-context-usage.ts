// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { ThreadMessage } from "@assistant-ui/react";
import {
  buildLocalTokenCountExtras,
  buildOutboundMessagesForTokenCount,
} from "../api/chat-adapter";
import { countChatInputTokens } from "../api/chat-api";
import { isExternalModelId } from "../external-providers";
import { useChatRuntimeStore } from "../stores/chat-runtime-store";
import type { MessageRecord } from "../types";
import { listStoredChatMessages } from "./chat-history-storage";

let refreshGeneration = 0;

type RuntimeMessagesGetter = () => readonly ThreadMessage[];

let runtimeMessagesGetter: RuntimeMessagesGetter | null = null;

export function registerRuntimeMessagesGetter(
  getter: RuntimeMessagesGetter | null,
): void {
  runtimeMessagesGetter = getter;
}

export type SavedContextUsage = {
  promptTokens: number;
  completionTokens: number;
  totalTokens: number;
  cachedTokens: number;
  cacheWriteTokens?: number;
  modelId?: string;
};

export function getSavedContextUsageFromMessages(
  msgs: MessageRecord[],
  checkpoint: string | null | undefined,
  ggufContextLength: number | null | undefined,
): SavedContextUsage | undefined {
  if (!checkpoint) return undefined;
  const lastAssistant = [...msgs]
    .reverse()
    .find((m) => m.role === "assistant");
  const savedUsage = (lastAssistant?.metadata as Record<string, unknown>)
    ?.contextUsage as SavedContextUsage | undefined;
  if (!savedUsage) return undefined;

  const withinLocalLimit =
    !ggufContextLength ||
    (savedUsage.totalTokens ?? 0) <= ggufContextLength;
  const modelMatches = savedUsage.modelId
    ? savedUsage.modelId === checkpoint
    : typeof ggufContextLength === "number" && ggufContextLength > 0;
  if (!withinLocalLimit || !modelMatches) return undefined;
  return savedUsage;
}

function storedMessageToRunMessage(record: MessageRecord): ThreadMessage {
  const content =
    Array.isArray(record.content) && record.content.length > 0
      ? structuredClone(record.content)
      : [{ type: "text" as const, text: "" }];

  if (record.role === "user") {
    return {
      id: record.id,
      createdAt: new Date(record.createdAt),
      role: "user",
      content: content as Extract<ThreadMessage, { role: "user" }>["content"],
      attachments: record.attachments
        ? structuredClone(record.attachments)
        : [],
      metadata: { custom: {} },
    };
  }

  const custom = (record.metadata as Record<string, unknown>) ?? {};
  const savedTiming = custom.timing as
    | import("@assistant-ui/react").MessageTiming
    | undefined;
  return {
    id: record.id,
    createdAt: new Date(record.createdAt),
    role: "assistant",
    content: content as Extract<ThreadMessage, { role: "assistant" }>["content"],
    status: { type: "complete", reason: "unknown" },
    metadata: {
      custom,
      ...(savedTiming ? { timing: savedTiming } : {}),
      steps: [],
      unstable_annotations: [],
      unstable_data: [],
      unstable_state: null,
    },
  };
}

// Same order the history adapter sorts stored records in before importing them.
const ROLE_ORDER: Record<string, number> = { system: 0, user: 1, assistant: 2 };

/**
 * Reproduce the branch the runtime actually displays for a stored thread.
 *
 * Mirrors the history adapter: sort by (createdAt, role, id), give legacy
 * records without a parentId the previous record as parent, then take the
 * ancestor chain of the last record -- assistant-ui imports without a headId,
 * so it resets the head to the last message and shows that chain. A greedy
 * newest-child descent picks a different branch (and drops pre-parentId
 * history), which counts a conversation the user is not looking at.
 */
function orderBySelectedBranch<T extends MessageRecord>(messages: T[]): T[] {
  const sorted = messages.slice().sort((a, b) => {
    if (a.createdAt !== b.createdAt) return a.createdAt - b.createdAt;
    const aOrder = ROLE_ORDER[a.role] ?? 99;
    const bOrder = ROLE_ORDER[b.role] ?? 99;
    if (aOrder !== bOrder) return aOrder - bOrder;
    return a.id < b.id ? -1 : a.id > b.id ? 1 : 0;
  });

  const byId = new Map<string, T>();
  const parentOf = new Map<string, string | null>();
  let previousId: string | null = null;
  for (const m of sorted) {
    byId.set(m.id, m);
    parentOf.set(m.id, m.parentId ?? previousId);
    previousId = m.id;
  }

  const chain: T[] = [];
  const seen = new Set<string>();
  let cur: string | null = sorted.at(-1)?.id ?? null;
  while (cur != null && !seen.has(cur)) {
    seen.add(cur);
    const record = byId.get(cur);
    if (!record) break;
    chain.push(record);
    cur = parentOf.get(cur) ?? null;
  }
  return chain.reverse();
}

/**
 * The runtime's message list is the branch actually on screen, so a post-load
 * recount follows it instead of re-deriving one from storage: after the user
 * steps back to an earlier retry sibling, the newest stored record belongs to a
 * branch nobody is looking at. Only used after a model load -- the history
 * adapter's own recount runs while assistant-ui is still importing, when the
 * getter can still hold the outgoing thread. Message ids are unique, so
 * requiring the displayed tail to be one of this thread's records also rejects
 * a getter parked on another thread (compare pane, mid switch).
 */
function runtimeSelectedBranch(
  records: MessageRecord[],
): readonly ThreadMessage[] | null {
  const runtimeMessages = runtimeMessagesGetter?.();
  if (!runtimeMessages || runtimeMessages.length === 0) return null;
  const tailId = runtimeMessages[runtimeMessages.length - 1].id;
  return records.some((record) => record.id === tailId)
    ? runtimeMessages
    : null;
}

function zeroContextUsage(): SavedContextUsage {
  return {
    promptTokens: 0,
    completionTokens: 0,
    totalTokens: 0,
    cachedTokens: 0,
    cacheWriteTokens: 0,
  };
}

/**
 * Re-count prompt tokens for the active local GGUF chat and populate the
 * context-usage bar without waiting for the next completion.
 */
export async function refreshContextUsage(options?: {
  threadId?: string;
  /** When true, skip the modelLoading guard (post-load recount). */
  afterModelLoad?: boolean;
}): Promise<void> {
  const store = useChatRuntimeStore.getState();
  const threadId = options?.threadId ?? store.activeThreadId;
  const checkpoint = store.params.checkpoint;

  if (
    !checkpoint ||
    isExternalModelId(checkpoint) ||
    (!options?.afterModelLoad && store.modelLoading) ||
    store.ggufContextLength == null
  ) {
    return;
  }

  // Bump only once this call is going to do work, so a call that bails here
  // cannot cancel a recount that is already in flight.
  const generation = ++refreshGeneration;

  const capturedThreadId = threadId ?? null;
  const capturedCheckpoint = checkpoint;

  if (!threadId) {
    if (
      generation !== refreshGeneration ||
      useChatRuntimeStore.getState().params.checkpoint !== capturedCheckpoint
    ) {
      return;
    }
    if (
      capturedThreadId != null &&
      useChatRuntimeStore.getState().activeThreadId !== capturedThreadId
    ) {
      return;
    }
    useChatRuntimeStore.getState().setContextUsage(zeroContextUsage());
    return;
  }

  try {
    let runMessages: readonly ThreadMessage[];
    const records = await listStoredChatMessages(threadId);
    if (generation !== refreshGeneration) return;
    if (useChatRuntimeStore.getState().params.checkpoint !== capturedCheckpoint) {
      return;
    }

    if (records.length > 0) {
      runMessages =
        (options?.afterModelLoad ? runtimeSelectedBranch(records) : null) ??
        orderBySelectedBranch(records).map(storedMessageToRunMessage);

      const savedUsage = getSavedContextUsageFromMessages(
        records,
        capturedCheckpoint,
        useChatRuntimeStore.getState().ggufContextLength,
      );
      if (
        savedUsage &&
        generation === refreshGeneration &&
        useChatRuntimeStore.getState().params.checkpoint === capturedCheckpoint &&
        (capturedThreadId == null ||
          useChatRuntimeStore.getState().activeThreadId === capturedThreadId)
      ) {
        useChatRuntimeStore.getState().setContextUsage(savedUsage);
      }
    } else {
      runMessages = runtimeMessagesGetter?.() ?? [];
    }

    if (generation !== refreshGeneration) return;
    if (useChatRuntimeStore.getState().params.checkpoint !== capturedCheckpoint) {
      return;
    }

    const outbound = await buildOutboundMessagesForTokenCount(
      runMessages,
      threadId,
    );
    if (generation !== refreshGeneration) return;
    if (useChatRuntimeStore.getState().params.checkpoint !== capturedCheckpoint) {
      return;
    }

    const toolExtras = await buildLocalTokenCountExtras(threadId, outbound);
    // A completion finishing mid-count writes the exact usage for a turn this
    // count predates, so drop the recount rather than roll the bar backwards.
    const usageBeforeCount = useChatRuntimeStore.getState().contextUsage;

    let inputTokens = 0;
    if (outbound.length > 0) {
      const result = await countChatInputTokens({
        model: capturedCheckpoint,
        messages: outbound,
        ...toolExtras,
      });
      inputTokens = result.input_tokens;
    }

    if (generation !== refreshGeneration) return;
    if (useChatRuntimeStore.getState().params.checkpoint !== capturedCheckpoint) {
      return;
    }
    if (
      capturedThreadId != null &&
      useChatRuntimeStore.getState().activeThreadId !== capturedThreadId
    ) {
      return;
    }
    if (useChatRuntimeStore.getState().contextUsage !== usageBeforeCount) {
      return;
    }

    useChatRuntimeStore.getState().setContextUsage({
      promptTokens: inputTokens,
      completionTokens: 0,
      totalTokens: inputTokens,
      cachedTokens: 0,
      cacheWriteTokens: 0,
    });
  } catch {
    // Background recount should not interrupt chat; saved usage stays visible.
  }
}
