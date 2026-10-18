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
import { savedBranchHead } from "./branch-head";
import { contextUsageReasoningKey } from "./context-usage-reasoning";
import { listStoredChatMessages } from "./chat-history-storage";
import { orderBySelectedBranch } from "./message-order";

// Threads with a count on the wire. A model load fires two triggers, and both would otherwise
// render the template and tokenize.
const countsInFlight = new Set<string | null>();

// A trigger deferred behind an in-flight count, replayed once that count settles WITHOUT
// publishing. Dropping it loses the bar: a run stopped before it emits usage fires the
// retry this depends on. Replaying it unconditionally would restore the doubled model-load
// count, where the first trigger publishes.
const retryAfterInFlight = new Map<string | null, RefreshOptions>();

let completedCount: {
  key: string;
  usage: NonNullable<
    ReturnType<typeof useChatRuntimeStore.getState>["contextUsage"]
  >;
} | null = null;

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
      metadata: { custom: (record.metadata as Record<string, unknown>) ?? {} },
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
    content: content as Extract<
      ThreadMessage,
      { role: "assistant" }
    >["content"],
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

/** Rolling 32-bit hash. Only has to change when the input does, not resist an adversary. */
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
    // Unserializable payload: fall back to the part kind, which both sides derive the same way.
    serialized = String((part as { type?: unknown })?.type);
  }
  return foldHash(serialized, seed);
}

/** Identity of the branch a count priced. Content is hashed, not measured: a run mutates a
 *  turn in place, so neither a length nor a part tally sees it. Attachments are folded in
 *  because the counter prices them too. */
function branchSignature(messages: readonly ThreadMessage[]): string {
  let hash = 0;
  let parts = 0;
  for (const message of messages) {
    hash = foldPart(
      [
        message.id,
        message.role,
        message.metadata?.custom?.revision,
        message.metadata?.custom?.contextTruncation,
      ],
      hash,
    );
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

/** The branch the mounted runtime is showing for the thread the store calls active. */
type ActiveBranchReader = () => readonly ThreadMessage[] | null;

let readActiveBranch: ActiveBranchReader | null = null;

/** Publish the mounted runtime's visible branch so the recount prices it instead of the
 *  persisted records. Only the single-chat pane registers one. */
export function setActiveBranchReader(reader: ActiveBranchReader | null): void {
  readActiveBranch = reader;
}

export function currentContextUsageReasoningKey(
  state = useChatRuntimeStore.getState(),
): string | undefined {
  if (
    !state.params.checkpoint ||
    isExternalModelId(state.params.checkpoint) ||
    !(state.supportsReasoning || state.supportsPreserveThinking)
  ) {
    return undefined;
  }
  return contextUsageReasoningKey(buildLocalTokenCountReasoning(state));
}

type RefreshOptions =
  | {
      threadId?: string;
      /** When true, skip the modelLoading guard (post-load recount). */
      afterModelLoad?: boolean;
      invalidate?: boolean;
    }
  | undefined;

/** Re-count prompt tokens for the active local chat and fill the usage bar. */
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

  // An output-only audio GGUF never sends a chat completion: the adapter routes the turn to
  // /audio/generate, which returns no usage, so a chat-template total is never corrected.
  const activeModel = store.models?.find(
    (model: { id: string }) => model.id === checkpoint,
  );
  if (activeModel?.isAudio && !activeModel?.hasAudioInput) return;

  if (options?.invalidate) {
    completedCount = null;
    store.setContextUsage(null);
  }

  // Never count while anything is generating: the endpoint refuses, and the recount effect
  // depends on this so the last run finishing re-fires it. runningByThreadId, not the
  // narrower local one: the endpoint refuses during an external-provider run too, and
  // nothing re-fires this effect when an external run ends unless it is in the deps.
  if (Object.values(store.runningByThreadId ?? {}).some(Boolean)) return;

  const capturedThreadId = threadId ?? null;
  const capturedCheckpoint = checkpoint;
  const capturedReasoning = JSON.stringify(buildLocalTokenCountReasoning());
  const capturedSettingsEpoch = store.queuedSettingsEpoch;

  if (countsInFlight.has(capturedThreadId)) {
    retryAfterInFlight.set(capturedThreadId, options);
    return;
  }

  const stale = (): boolean =>
    useChatRuntimeStore.getState().params.checkpoint !== capturedCheckpoint ||
    useChatRuntimeStore.getState().queuedSettingsEpoch !==
      capturedSettingsEpoch ||
    JSON.stringify(buildLocalTokenCountReasoning()) !== capturedReasoning;

  countsInFlight.add(capturedThreadId);
  let published = false;
  let replayQueued = true;
  let inputsChanged = stale;
  try {
    // Prefer the mounted runtime: it is what the next request reads from. A captured null is
    // EXCLUDED, not matched: New Chat leaves the outgoing conversation mounted until
    // switchToNewThread() settles, so null === null would price it into the empty chat.
    const readOwnBranch = (): readonly ThreadMessage[] | null =>
      capturedThreadId != null &&
      useChatRuntimeStore.getState().activeThreadId === capturedThreadId
        ? (readActiveBranch?.() ?? null)
        : null;

    const liveBranch = readOwnBranch();

    let runMessages: readonly ThreadMessage[];
    // re-read before publishing so a turn sent during the count invalidates it.
    let countedBranch: string | null = null;
    // only the last id witnesses the fallback because stored and runtime records hash differently.
    let countedLastId: string | null = null;
    const fromLiveBranch = liveBranch != null;
    if (fromLiveBranch) {
      runMessages = liveBranch as readonly ThreadMessage[];
    } else {
      const records = threadId ? await listStoredChatMessages(threadId) : [];
      if (stale()) return;
      runMessages = orderBySelectedBranch(
        records,
        threadId ? savedBranchHead(threadId, records) : undefined,
      ).map(storedMessageToRunMessage);
    }

    // /chat/count_tokens rejects images; hashing or sending their base64 can block the UI thread.
    if (messagesContainImage(runMessages)) return;

    // audio is omitted from toOpenAIMessages, so counting would price only its text.
    if (findLatestUserAudioBase64(runMessages)) return;

    // video counting misses server-side frames; declining also avoids hashing up to 85 MB.
    if (findLatestUserVideoBase64(runMessages)) return;

    if (fromLiveBranch) {
      countedBranch = branchSignature(runMessages);
    } else {
      countedLastId = runMessages.at(-1)?.id ?? "";
    }
    inputsChanged = (): boolean => {
      if (stale()) return true;
      const current = readOwnBranch();
      if (current == null) return false;
      return countedBranch != null
        ? branchSignature(current) !== countedBranch
        : current.length > 0 && (current.at(-1)?.id ?? "") !== countedLastId;
    };

    // A completion finishing mid-count writes exact usage for a turn this count predates, so drop
    // the recount rather than roll the bar backwards.
    const usageBeforeCount = useChatRuntimeStore.getState().contextUsage;

    const canPublish = (): boolean => {
      const currentStore = useChatRuntimeStore.getState();
      return (
        !inputsChanged() &&
        currentStore.activeThreadId === capturedThreadId &&
        currentStore.contextUsage === usageBeforeCount &&
        !Object.values(currentStore.runningByThreadId ?? {}).some(Boolean)
      );
    };

    // undefined, not null: a chat with no persisted thread has no project to resolve from.
    const payloadThreadId = threadId ?? undefined;
    const countHistory = await buildLocalTokenCountHistory(
      runMessages,
      payloadThreadId,
    );
    if (stale()) return;
    const countExtras = await buildLocalTokenCountExtras(
      payloadThreadId,
      countHistory.messages,
    );
    if (!canPublish()) return;

    const ragScope = countExtras.rag_scope as
      | {
          autoinject?: boolean;
          whole_doc?: boolean;
          kb_id?: string;
          thread_id?: string;
          project_id?: string;
        }
      | undefined;
    const lastRole = countHistory.messages.at(-1)?.role;
    const pendingRetrieval =
      ragScope != null &&
      (lastRole === "user" || lastRole === "tool") &&
      (store.loadedIsGguf ||
        (countExtras.enable_tools === true &&
          countExtras.max_tool_calls_per_message !== 0 &&
          (countExtras.permission_mode !== "ask" ||
            countExtras.bypass_permissions === true) &&
          Boolean(
            ragScope.kb_id || ragScope.thread_id || ragScope.project_id,
          ) &&
          (ragScope.autoinject !== false ||
            Boolean(
              ragScope.thread_id &&
                !ragScope.kb_id &&
                ragScope.whole_doc !== false,
            ))));
    if (pendingRetrieval) {
      replayQueued = false;
      return;
    }

    const payload = {
      model: capturedCheckpoint,
      ...countHistory,
      ...buildLocalTokenCountReasoning(),
      ...countExtras,
    };
    const countKey = JSON.stringify([
      capturedThreadId,
      branchSignature(runMessages),
      store.activeLoadId,
      store.activeGgufVariant,
      store.loadedContextLength,
      store.loadedChatTemplateOverride,
      payload,
    ]);
    if (completedCount?.key === countKey) {
      if (usageBeforeCount == null || usageBeforeCount.estimated) {
        useChatRuntimeStore.getState().setContextUsage(completedCount.usage);
      }
      published = true;
      return;
    }

    const { input_tokens: inputTokens, model: countedModel } =
      await countChatInputTokens(payload);

    if (stale()) return;
    // The response type is a compile-time assertion only: anything else answering 200 would put
    // undefined on the bar and throw from toLocaleString.
    if (
      typeof inputTokens !== "number" ||
      !Number.isFinite(inputTokens) ||
      inputTokens < 0
    ) {
      replayQueued = false;
      return;
    }
    // The endpoint counts with whatever is resident, never the model asked for: a load from
    // another tab returns another tokenizer's total, which the checkpoint guards cannot see.
    if (countedModel != null && countedModel !== capturedCheckpoint) {
      replayQueued = false;
      return;
    }
    if (!canPublish()) return;

    useChatRuntimeStore.getState().setContextUsage({
      promptTokens: inputTokens,
      completionTokens: 0,
      totalTokens: inputTokens,
      cachedTokens: 0,
      cacheWriteTokens: 0,
      reasoningKey: contextUsageReasoningKey(JSON.parse(capturedReasoning)),
    });
    const usage = useChatRuntimeStore.getState().contextUsage;
    if (usage) completedCount = { key: countKey, usage };
    published = true;
  } catch {
    replayQueued = false;
    // Background recount should not interrupt chat; saved usage stays visible.
  } finally {
    countsInFlight.delete(capturedThreadId);
    const queued = retryAfterInFlight.get(capturedThreadId);
    const hadQueued = retryAfterInFlight.delete(capturedThreadId);
    if (
      hadQueued &&
      !published &&
      (replayQueued || inputsChanged() || queued?.invalidate)
    ) {
      void refreshContextUsage(queued);
    }
  }
}
