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
import { getMcpServerMutationEpoch } from "../api/mcp-server-mutation-tracker";
import { isExternalModelId } from "../external-providers";
import { localMediaToolsForTurn, useChatRuntimeStore } from "../stores/chat-runtime-store";
import type { MessageRecord } from "../types";
import { savedBranchHead } from "./branch-head";
import { contextUsageReasoningKey } from "./context-usage-reasoning";
import { listStoredChatMessages } from "./chat-history-storage";
import { orderBySelectedBranch } from "./message-order";

// Threads with a count on the wire. A model load fires two triggers, and both would otherwise
// render the template and tokenize.
const countsInFlight = new Map<string | null, {
  afterModelLoad: boolean;
  loadId: string | null;
}>();

// Replay a deferred trigger only when the in-flight count did not publish.
const retryAfterInFlight = new Map<string | null, RefreshOptions>();
let countEpoch = 0;

let completedCount: {
  key: string;
  payload: string;
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

type ActiveBranchReader = () => readonly ThreadMessage[] | null;

let readActiveBranch: ActiveBranchReader | null = null;

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

export function currentContextUsageSettingsKey(
  state: ReturnType<typeof useChatRuntimeStore.getState>,
  browserTools: string,
): string {
  return JSON.stringify([
    state.params.systemPrompt,
    state.params.systemVariables,
    state.params.maxTokens,
    state.loadedChatTemplateOverride,
    state.autoCompactEnabled,
    state.supportsTools,
    state.toolsEnabled,
    state.codeToolsEnabled,
    state.codeToolsDeclinedUnderFullAccess,
    state.mcpEnabledForChat,
    state.ragEnabled,
    state.ragSource,
    state.ragMode,
    state.ragTopK,
    state.ragAutoInject,
    state.ragAutoInjectMinScore,
    state.autoHealToolCalls,
    state.bypassPermissions,
    state.deepResearchEnabled,
    state.permissionMode,
    state.sandboxLevel,
    state.maxToolCallsPerMessage,
    browserTools,
    localMediaToolsForTurn(state),
  ]);
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

  const pendingCount = countsInFlight.get(threadId ?? null);
  if (
    options?.invalidate ||
    (options?.afterModelLoad &&
      (!pendingCount?.afterModelLoad || pendingCount.loadId !== store.activeLoadId))
  ) {
    completedCount = null;
    countEpoch += 1;
  }

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

  if (options?.invalidate) {
    store.setContextUsage(null);
  }
  const usageBeforeCount = useChatRuntimeStore.getState().contextUsage;

  // The endpoint refuses during any run, external included, and run end re-fires this effect.
  if (Object.values(store.runningByThreadId ?? {}).some(Boolean)) return;

  const capturedThreadId = threadId ?? null;
  const capturedCheckpoint = checkpoint;
  const capturedReasoning = JSON.stringify(buildLocalTokenCountReasoning());
  const capturedSettingsEpoch = store.queuedSettingsEpoch;
  const capturedCountEpoch = countEpoch;
  const capturedMcpEpoch = getMcpServerMutationEpoch();

  if (countsInFlight.has(capturedThreadId)) {
    retryAfterInFlight.set(capturedThreadId, options);
    return;
  }

  const promptChanged = (): boolean =>
    countEpoch !== capturedCountEpoch ||
    (store.mcpEnabledForChat && getMcpServerMutationEpoch() !== capturedMcpEpoch) ||
    useChatRuntimeStore.getState().params.checkpoint !== capturedCheckpoint ||
    JSON.stringify(buildLocalTokenCountReasoning()) !== capturedReasoning;
  const stale = (): boolean =>
    promptChanged() ||
    useChatRuntimeStore.getState().queuedSettingsEpoch !==
      capturedSettingsEpoch;

  countsInFlight.set(capturedThreadId, {
    afterModelLoad: options?.afterModelLoad === true,
    loadId: store.activeLoadId,
  });
  let published = false;
  let replayQueued = true;
  let inputsChanged = stale;
  try {
    // Exclude a captured null: New Chat keeps the old conversation mounted until the switch settles.
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

    if (
      !store.loadedIsGguf &&
      runMessages.some((message) => message.metadata?.custom?.contextTruncation)
    ) return;

    if (fromLiveBranch) {
      countedBranch = branchSignature(runMessages);
    } else {
      countedLastId = runMessages.at(-1)?.id ?? "";
    }
    const branchChanged = (): boolean => {
      const current = readOwnBranch();
      if (current == null) return false;
      return countedBranch != null
        ? branchSignature(current) !== countedBranch
        : current.length > 0 && (current.at(-1)?.id ?? "") !== countedLastId;
    };
    inputsChanged = () => stale() || branchChanged();

    const canPublish = (): boolean => {
      const currentStore = useChatRuntimeStore.getState();
      return (
        !stale() &&
        // Compared even when null: a count started with no thread must not land on one since opened.
        currentStore.activeThreadId === capturedThreadId &&
        currentStore.contextUsage === usageBeforeCount &&
        !Object.values(currentStore.runningByThreadId ?? {}).some(Boolean)
      );
    };

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

    const payload = {
      model: capturedCheckpoint,
      ...countHistory,
      ...buildLocalTokenCountReasoning(),
      ...countExtras,
    };
    const serializedPayload = JSON.stringify(payload);
    const countKey = JSON.stringify([
      capturedThreadId,
      countedBranch ?? branchSignature(runMessages),
      store.activeLoadId,
      store.activeGgufVariant,
      store.loadedContextLength,
      store.loadedChatTemplateOverride,
      countExtras.mcp_enabled ? capturedMcpEpoch : null,
    ]);
    if (completedCount?.key === countKey && completedCount.payload === serializedPayload) {
      if (branchChanged()) return;
      if (usageBeforeCount == null || usageBeforeCount.estimated) {
        useChatRuntimeStore.getState().setContextUsage(completedCount.usage);
      }
      published = true;
      return;
    }

    const { input_tokens: inputTokens, model: countedModel } =
      await countChatInputTokens(payload, serializedPayload);

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
    const usage = {
      promptTokens: inputTokens,
      completionTokens: 0,
      totalTokens: inputTokens,
      cachedTokens: 0,
      cacheWriteTokens: 0,
      reasoningKey: contextUsageReasoningKey(JSON.parse(capturedReasoning)),
    };
    if (promptChanged() || branchChanged()) return;
    completedCount = { key: countKey, payload: serializedPayload, usage };
    if (!canPublish()) return;

    useChatRuntimeStore.getState().setContextUsage(usage);
    published = true;
  } catch {
    replayQueued = false;
    // Background recount should not interrupt chat; saved usage stays visible.
  } finally {
    countsInFlight.delete(capturedThreadId);
    const queued = retryAfterInFlight.get(capturedThreadId);
    const hadQueued = retryAfterInFlight.delete(capturedThreadId);
    const current = useChatRuntimeStore.getState();
    const newerExactUsage =
      current.contextUsage !== usageBeforeCount &&
      current.contextUsage != null &&
      !current.contextUsage.estimated;
    const settingsChanged =
      current.activeThreadId === capturedThreadId &&
      current.params.checkpoint === capturedCheckpoint &&
      (current.queuedSettingsEpoch !== capturedSettingsEpoch ||
        (store.mcpEnabledForChat && getMcpServerMutationEpoch() !== capturedMcpEpoch));
    if (
      !published &&
      !newerExactUsage &&
      (settingsChanged ||
        (hadQueued && (replayQueued || inputsChanged() || queued?.invalidate)))
    ) {
      void refreshContextUsage(
        hadQueued ? queued : { ...options, invalidate: false },
      );
    }
  }
}
