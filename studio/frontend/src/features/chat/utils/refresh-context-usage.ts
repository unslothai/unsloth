// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { ThreadMessage } from "@assistant-ui/react";
import {
  buildLocalTokenCountExtras,
  buildLocalTokenCountHistory,
  buildLocalTokenCountReasoning,
  findLatestUserAudioBase64,
  findLatestUserVideoBase64,
  messagesContainImage,
} from "../api/chat-adapter";
import { countChatInputTokens } from "../api/chat-api";
import { isExternalModelId } from "../external-providers";
import { useChatRuntimeStore } from "../stores/chat-runtime-store";
import type { MessageRecord } from "../types";
import { listStoredChatMessages } from "./chat-history-storage";
import { orderBySelectedBranch } from "./message-order";

// Per thread so a hidden compare pane cannot invalidate the visible thread's count.
const refreshGenerations = new Map<string | null, number>();

function nextGeneration(threadKey: string | null): number {
  const generation = (refreshGenerations.get(threadKey) ?? 0) + 1;
  refreshGenerations.set(threadKey, generation);
  return generation;
}

function superseded(threadKey: string | null, generation: number): boolean {
  return refreshGenerations.get(threadKey) !== generation;
}

const countsInFlight = new Set<string | null>();

// Replay a deferred trigger only if the in-flight count settled without publishing.
const retryAfterInFlight = new Map<string | null, RefreshOptions>();

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

function foldHash(text: string, seed: number): number {
  let hash = seed;
  for (let i = 0; i < text.length; i += 1) {
    hash = (Math.imul(hash, 31) + text.charCodeAt(i)) | 0;
  }
  return hash;
}

function foldPart(part: unknown, seed: number): number {
  let serialized: string;
  try {
    serialized = JSON.stringify(part) ?? "";
  } catch {
    serialized = String((part as { type?: unknown })?.type);
  }
  return foldHash(serialized, seed);
}

/** Hashed, not measured: a run mutates a turn in place. */
function branchSignature(messages: readonly ThreadMessage[]): string {
  let hash = 0;
  let parts = 0;
  for (const message of messages) {
    for (const part of message.content as readonly unknown[]) {
      parts += 1;
      hash = foldPart(part, hash);
    }
    const attachments = (message as { attachments?: readonly unknown[] })
      .attachments;
    for (const attachment of attachments ?? []) {
      parts += 1;
      hash = foldPart(attachment, hash);
    }
  }
  return `${messages.length}:${parts}:${messages.at(-1)?.id ?? ""}:${hash}`;
}

type ActiveBranchReader = () => readonly ThreadMessage[] | null;

let readActiveBranch: ActiveBranchReader | null = null;

export function setActiveBranchReader(reader: ActiveBranchReader | null): void {
  readActiveBranch = reader;
}

type RefreshOptions =
  | {
      threadId?: string;
      afterModelLoad?: boolean;
      invalidate?: boolean;
    }
  | undefined;

export async function refreshContextUsage(
  options?: RefreshOptions,
): Promise<void> {
  const store = useChatRuntimeStore.getState();
  const threadId = options?.threadId ?? store.activeThreadId;
  const checkpoint = store.params.checkpoint;

  if (
    !checkpoint ||
    isExternalModelId(checkpoint) ||
    (!options?.afterModelLoad && store.modelLoading) ||
    store.loadedContextLength == null
  ) {
    return;
  }

  // Output-only audio GGUFs never get chat usage to correct the count.
  const activeModel = store.models?.find(
    (model: { id: string }) => model.id === checkpoint,
  );
  if (activeModel?.isAudio && !activeModel?.hasAudioInput) return;

  if (options?.invalidate) store.setContextUsage(null);

  // The endpoint refuses during any run, external included, and run end re-fires this effect.
  if (Object.values(store.runningByThreadId ?? {}).some(Boolean)) return;

  const capturedThreadId = threadId ?? null;
  const capturedCheckpoint = checkpoint;

  if (countsInFlight.has(capturedThreadId)) {
    retryAfterInFlight.set(capturedThreadId, options);
    return;
  }

  const generation = nextGeneration(capturedThreadId);

  const stale = (): boolean =>
    superseded(capturedThreadId, generation) ||
    useChatRuntimeStore.getState().params.checkpoint !== capturedCheckpoint;

  countsInFlight.add(capturedThreadId);
  let published = false;
  try {
    // Exclude a captured null: New Chat keeps the old conversation mounted until the switch settles.
    const readOwnBranch = (): readonly ThreadMessage[] | null =>
      capturedThreadId != null &&
      useChatRuntimeStore.getState().activeThreadId === capturedThreadId
        ? (readActiveBranch?.() ?? null)
        : null;

    const liveBranch = readOwnBranch();

    let runMessages: readonly ThreadMessage[];
    let countedBranch: string | null = null;
    let countedLastId: string | null = null;
    const fromLiveBranch = Boolean(liveBranch && liveBranch.length > 0);
    if (fromLiveBranch) {
      runMessages = liveBranch as readonly ThreadMessage[];
    } else {
      const records = threadId ? await listStoredChatMessages(threadId) : [];
      if (stale()) return;
      runMessages = orderBySelectedBranch(records).map(
        storedMessageToRunMessage,
      );
    }

    // /chat/count_tokens 503s on images; bail before hashing megabytes of base64.
    if (messagesContainImage(runMessages)) return;

    // toOpenAIMessages has no audio or video branch, so counting would underprice.
    if (findLatestUserAudioBase64(runMessages)) return;

    if (findLatestUserVideoBase64(runMessages)) return;

    if (fromLiveBranch) {
      countedBranch = branchSignature(runMessages);
    } else {
      countedLastId = runMessages.at(-1)?.id ?? "";
    }

    const usageBeforeCount = useChatRuntimeStore.getState().contextUsage;

    const payloadThreadId = threadId ?? undefined;
    const countHistory = await buildLocalTokenCountHistory(
      runMessages,
      payloadThreadId,
    );
    if (stale()) return;
    const countExtras = await buildLocalTokenCountExtras(payloadThreadId);
    if (stale()) return;

    // Always ask the server: templates and `--enable-tools` add tokens the client cannot see.
    const { input_tokens: inputTokens, model: countedModel } =
      await countChatInputTokens({
        model: capturedCheckpoint,
        ...countHistory,
        ...buildLocalTokenCountReasoning(),
        ...countExtras,
      });

    if (stale()) return;
    if (typeof inputTokens !== "number" || !Number.isFinite(inputTokens)) return;
    // The endpoint counts with whatever model is resident, which may be another tab's load.
    if (countedModel != null && countedModel !== capturedCheckpoint) {
      return;
    }
    if (useChatRuntimeStore.getState().activeThreadId !== capturedThreadId) {
      return;
    }
    if (useChatRuntimeStore.getState().contextUsage !== usageBeforeCount) {
      return;
    }
    if (useChatRuntimeStore.getState().runningByThreadId[capturedThreadId ?? "__default"]) {
      return;
    }
    if (countedBranch != null) {
      const current = readActiveBranch?.();
      if (current != null && branchSignature(current) !== countedBranch) {
        return;
      }
    } else if (countedLastId != null) {
      const current = readOwnBranch();
      if (
        current != null &&
        current.length > 0 &&
        (current.at(-1)?.id ?? "") !== countedLastId
      ) {
        return;
      }
    }

    useChatRuntimeStore.getState().setContextUsage({
      promptTokens: inputTokens,
      completionTokens: 0,
      totalTokens: inputTokens,
      cachedTokens: 0,
      cacheWriteTokens: 0,
    });
    published = true;
  } catch {
    // Background recount should not interrupt chat; saved usage stays visible.
  } finally {
    countsInFlight.delete(capturedThreadId);
    const queued = retryAfterInFlight.get(capturedThreadId);
    const hadQueued = retryAfterInFlight.delete(capturedThreadId);
    if (hadQueued && !published) void refreshContextUsage(queued);
  }
}
