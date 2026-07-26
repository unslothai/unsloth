// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { ThreadMessage } from "@assistant-ui/react";
import { buildOutboundMessagesForTokenCount } from "../api/chat-adapter";
import { countChatInputTokens } from "../api/chat-api";
import { isExternalModelId } from "../external-providers";
import { useChatRuntimeStore } from "../stores/chat-runtime-store";
import type { MessageRecord } from "../types";
import { listStoredChatMessages } from "./chat-history-storage";

let refreshGeneration = 0;

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

/**
 * Re-count prompt tokens for the active local GGUF chat and populate the
 * context-usage bar without waiting for the next completion.
 */
export async function refreshContextUsage(options?: {
  threadId?: string;
}): Promise<void> {
  const generation = ++refreshGeneration;
  const store = useChatRuntimeStore.getState();
  const threadId = options?.threadId ?? store.activeThreadId;
  const checkpoint = store.params.checkpoint;

  if (
    !threadId ||
    !checkpoint ||
    isExternalModelId(checkpoint) ||
    store.modelLoading ||
    store.ggufContextLength == null
  ) {
    return;
  }

  const capturedCheckpoint = checkpoint;

  try {
    const records = await listStoredChatMessages(threadId);
    if (generation !== refreshGeneration) return;
    if (useChatRuntimeStore.getState().params.checkpoint !== capturedCheckpoint) {
      return;
    }

    const runMessages = records
      .slice()
      .sort((a: MessageRecord, b: MessageRecord) => a.createdAt - b.createdAt)
      .map(storedMessageToRunMessage);

    const outbound = await buildOutboundMessagesForTokenCount(
      runMessages,
      threadId,
    );
    if (generation !== refreshGeneration) return;
    if (useChatRuntimeStore.getState().params.checkpoint !== capturedCheckpoint) {
      return;
    }

    let inputTokens = 0;
    if (outbound.length > 0) {
      const result = await countChatInputTokens({
        model: capturedCheckpoint,
        messages: outbound,
      });
      inputTokens = result.input_tokens;
    }

    if (generation !== refreshGeneration) return;
    if (useChatRuntimeStore.getState().params.checkpoint !== capturedCheckpoint) {
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
    // Background recount should not interrupt chat; the bar repopulates on send.
  }
}
