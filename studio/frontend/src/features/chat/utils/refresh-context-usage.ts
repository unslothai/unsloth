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

function orderByParentChain<T extends MessageRecord>(
  messages: T[],
): T[] {
  const byId = new Map<string, T>(messages.map((m) => [m.id, m]));
  const childrenOf = new Map<string | null, T[]>();
  for (const m of messages) {
    const pid = m.parentId ?? null;
    if (!childrenOf.has(pid)) childrenOf.set(pid, []);
    childrenOf.get(pid)!.push(m);
  }

  const result: T[] = [];
  let cur: string | null = null;
  while (childrenOf.has(cur)) {
    const children: T[] = childrenOf.get(cur)!;
    const next: T = children.reduce((a: T, b: T) =>
      a.createdAt >= b.createdAt ? a : b,
    );
    result.push(next);
    cur = next.id;
    byId.delete(next.id);
  }
  return result;
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
  const generation = ++refreshGeneration;
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
      const hasParentIds = records.some((m) => m.parentId != null);
      const ordered = hasParentIds
        ? orderByParentChain(records)
        : records
            .slice()
            .sort((a, b) => a.createdAt - b.createdAt);
      runMessages = ordered.map(storedMessageToRunMessage);

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
