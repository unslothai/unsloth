// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { ThreadMessage } from "@assistant-ui/react";
import {
  buildLocalTokenCountExtras,
  buildOutboundMessagesForTokenCount,
  findLatestUserAudioBase64,
} from "../api/chat-adapter";
import { countChatInputTokens } from "../api/chat-api";
import { isExternalModelId } from "../external-providers";
import { useChatRuntimeStore } from "../stores/chat-runtime-store";
import type { MessageRecord } from "../types";
import { listStoredChatMessages } from "./chat-history-storage";

// Cancellation is per thread, not per module: in compare mode a hidden pane's history load
// would otherwise invalidate the visible thread's count, blanking the bar. Same-thread wins.
const refreshGenerations = new Map<string | null, number>();

function nextGeneration(threadKey: string | null): number {
  const generation = (refreshGenerations.get(threadKey) ?? 0) + 1;
  refreshGenerations.set(threadKey, generation);
  return generation;
}

function superseded(threadKey: string | null, generation: number): boolean {
  return refreshGenerations.get(threadKey) !== generation;
}

type RuntimeMessagesGetter = () => readonly ThreadMessage[];

// Compare mode mounts several providers at once, so this is a stack, not a slot: unmounting
// the newest uncovers the one underneath. Callers that know which records they want search it.
const runtimeMessagesGetters: RuntimeMessagesGetter[] = [];

function currentRuntimeMessagesGetter(): RuntimeMessagesGetter | null {
  // Registration records no owner, so with compare panes mounted the newest
  // getter may be a sibling's and the caller's thread guard cannot tell.
  // Declining prices the template alone; the wrong pane reports a confident
  // wrong number (#7453).
  return runtimeMessagesGetters.length === 1
    ? (runtimeMessagesGetters[0] ?? null)
    : null;
}

/** Register the mounted runtime's message list; returns its own disposer. */
export function registerRuntimeMessagesGetter(
  getter: RuntimeMessagesGetter,
): () => void {
  runtimeMessagesGetters.push(getter);
  return () => {
    const index = runtimeMessagesGetters.lastIndexOf(getter);
    if (index >= 0) runtimeMessagesGetters.splice(index, 1);
  };
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
 * Reproduce the branch the runtime displays for a stored thread, as the history adapter does:
 * sort by (createdAt, role, id), parent legacy records to the previous one, then take the
 * ancestor chain of the last record (an import without a headId resets the head there). A
 * greedy newest-child descent picks a different branch and drops pre-parentId history.
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
 * The branch on screen, which a post-load recount follows instead of re-deriving one from
 * storage: after stepping back to a retry sibling, the newest stored record is on a branch
 * nobody is looking at. Post-load only, since the history adapter's own recount runs mid-import
 * while the getter still holds the old thread. Ids are unique, so the displayed tail identifies
 * the provider; getters parked on another thread are skipped, and no match at all falls back.
 */
function runtimeSelectedBranch(
  records: MessageRecord[],
): readonly ThreadMessage[] | null {
  const recordIds = new Set(records.map((record) => record.id));
  // Walk the whole stack, not just its top: with two panes mounted the newest is often the
  // other thread, and reading only that falls back to the storage branch, the wrong sibling.
  // Newest-first keeps single-provider order; a provider throwing mid-teardown is skipped.
  for (let index = runtimeMessagesGetters.length - 1; index >= 0; index--) {
    let runtimeMessages: readonly ThreadMessage[] | undefined;
    try {
      runtimeMessages = runtimeMessagesGetters[index]();
    } catch {
      continue;
    }
    if (!runtimeMessages || runtimeMessages.length === 0) continue;
    if (recordIds.has(runtimeMessages[runtimeMessages.length - 1].id)) {
      return runtimeMessages;
    }
  }
  return null;
}

/** The stored records behind a runtime branch, in the order it displays them. */
function recordsForRuntimeBranch(
  records: MessageRecord[],
  branch: readonly ThreadMessage[],
): MessageRecord[] {
  const byId = new Map(records.map((record) => [record.id, record]));
  return branch
    .map((message) => byId.get(message.id))
    .filter((record): record is MessageRecord => record != null);
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

/** Re-count prompt tokens for the active local GGUF chat and fill the usage bar. */
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

  const capturedThreadId = threadId ?? null;
  const capturedCheckpoint = checkpoint;

  // Bump only once this call will do work, so a bail-out cannot cancel an in-flight recount.
  const generation = nextGeneration(capturedThreadId);

  if (!threadId) {
    // Show something immediately for a chat with no persisted thread; the count below refines
    // it, since a system prompt or enabled tools are already in the request.
    useChatRuntimeStore.getState().setContextUsage(zeroContextUsage());
  }

  try {
    let runMessages: readonly ThreadMessage[];
    const records = threadId ? await listStoredChatMessages(threadId) : [];
    if (superseded(capturedThreadId, generation)) return;
    if (useChatRuntimeStore.getState().params.checkpoint !== capturedCheckpoint) {
      return;
    }

    if (records.length > 0) {
      const runtimeBranch = options?.afterModelLoad
        ? runtimeSelectedBranch(records)
        : null;
      // Scope the fallback to the branch being counted: a global reverse scan picks the newest
      // assistant, maybe on a sibling branch, and that stale value sticks if the recount misses.
      const branchRecords = runtimeBranch
        ? recordsForRuntimeBranch(records, runtimeBranch)
        : orderBySelectedBranch(records);
      runMessages =
        runtimeBranch ?? branchRecords.map(storedMessageToRunMessage);

      const savedUsage = getSavedContextUsageFromMessages(
        branchRecords,
        capturedCheckpoint,
        useChatRuntimeStore.getState().ggufContextLength,
      );
      if (
        savedUsage &&
        !superseded(capturedThreadId, generation) &&
        useChatRuntimeStore.getState().params.checkpoint === capturedCheckpoint &&
        (capturedThreadId == null ||
          useChatRuntimeStore.getState().activeThreadId === capturedThreadId)
      ) {
        useChatRuntimeStore.getState().setContextUsage(savedUsage);
      }
    } else {
      // A persisted-but-empty read (incognito thread) keeps messages only in the mounted
      // runtime. The getter is shared, so trust it only for the active thread, not a pane's.
      runMessages =
        threadId &&
        threadId === useChatRuntimeStore.getState().activeThreadId
          ? (currentRuntimeMessagesGetter()?.() ?? [])
          : [];
    }

    if (superseded(capturedThreadId, generation)) return;
    if (useChatRuntimeStore.getState().params.checkpoint !== capturedCheckpoint) {
      return;
    }

    // The real request replays the newest user audio as audio_base64 but toOpenAIMessages has
    // no audio branch, so counting would price a text-only prompt. Decline as images do.
    if (findLatestUserAudioBase64(runMessages)) return;

    // A completion finishing mid-count writes exact usage for a turn this count predates, so
    // drop the recount rather than roll the bar backwards. Sampled as soon as runMessages is
    // fixed (after our own saved-usage write): the payload build awaits storage, and a
    // completion landing in that window would otherwise be captured here and compare equal.
    const usageBeforeCount = useChatRuntimeStore.getState().contextUsage;

    // undefined, not null: a chat with no persisted thread has no project to resolve from.
    const payloadThreadId = threadId ?? undefined;
    const outbound = await buildOutboundMessagesForTokenCount(
      runMessages,
      payloadThreadId,
    );
    if (superseded(capturedThreadId, generation)) return;
    if (useChatRuntimeStore.getState().params.checkpoint !== capturedCheckpoint) {
      return;
    }

    const toolExtras = await buildLocalTokenCountExtras(payloadThreadId, outbound);

    // Always ask the server: the template itself has tokens, and `unsloth run --enable-tools`
    // injects schemas the client cannot see.
    const result = await countChatInputTokens({
      model: capturedCheckpoint,
      messages: outbound,
      ...toolExtras,
    });
    const inputTokens = result.input_tokens;

    // The endpoint counts whatever is loaded now, so another client's switch is invisible to
    // the checkpoint guards below. When the backend names the tokenizer, require it to be ours.
    if (result.model != null && result.model !== capturedCheckpoint) return;

    if (superseded(capturedThreadId, generation)) return;
    if (useChatRuntimeStore.getState().params.checkpoint !== capturedCheckpoint) {
      return;
    }
    // Compared even when null: a count started with no thread must not land on one since opened.
    if (useChatRuntimeStore.getState().activeThreadId !== capturedThreadId) {
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
