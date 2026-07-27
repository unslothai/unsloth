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

// Cancellation is per thread, not per module. Every mounted provider recounts
// its own thread when its history loads, so in compare mode a hidden pane's
// load would otherwise invalidate the visible thread's in-flight count, and
// the pane's own result is then dropped by the activeThreadId guard -- leaving
// the bar blank. Two calls for the SAME thread still supersede each other,
// which is what this counter is for.
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

// Compare mode mounts several providers at once, so this is a stack rather than
// a slot: unmounting the newest uncovers the provider underneath instead of
// leaving nothing registered. Callers that know which records they want search
// it (runtimeSelectedBranch); only the ones that cannot tell fall back to the
// newest mount.
const runtimeMessagesGetters: RuntimeMessagesGetter[] = [];

function currentRuntimeMessagesGetter(): RuntimeMessagesGetter | null {
  return runtimeMessagesGetters.at(-1) ?? null;
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
 * getter can still hold the outgoing thread. Message ids are unique, so the
 * displayed tail also identifies which mounted provider is showing these
 * records; getters parked on another thread (compare pane, mid switch) are
 * skipped, and only a stack with no match at all falls back to storage.
 */
function runtimeSelectedBranch(
  records: MessageRecord[],
): readonly ThreadMessage[] | null {
  const recordIds = new Set(records.map((record) => record.id));
  // Walk the whole stack, not just its top: with two panes mounted the newest
  // mount is often the other thread, and reading only that one fails the id
  // check and drops back to the storage branch -- the wrong retry sibling for
  // the pane that actually asked. Newest-first keeps the single-provider
  // order, and a provider that throws mid-teardown is skipped rather than
  // aborting the recount.
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

  const capturedThreadId = threadId ?? null;
  const capturedCheckpoint = checkpoint;

  // Bump only once this call is going to do work, so a call that bails here
  // cannot cancel a recount that is already in flight.
  const generation = nextGeneration(capturedThreadId);

  if (!threadId) {
    // Show something immediately for a chat with no persisted thread, then let
    // the count below refine it: a configured system prompt, project
    // instructions or enabled tools are already part of the next request, so
    // zero is only right when that payload turns out to be empty.
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
      // Scope the fallback to the branch being counted. A reverse scan over
      // every record picks the newest assistant globally, which can sit on a
      // sibling branch, and that stale value is what stays on the bar whenever
      // the recount does not land (an image thread, a failed count).
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
      // The runtime getter is for a persisted-but-empty read (an incognito
      // thread), where the messages live only in the mounted runtime. It is a
      // single slot shared by every provider, so only trust it for the active
      // thread; a compare pane reading it would get the other pane's branch.
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

    // The newest user turn's audio is replayed as audio_base64 by the real
    // request, but toOpenAIMessages has no audio branch, so counting here would
    // price a text-only prompt. Decline instead, exactly as the endpoint does
    // for images, and leave the usage already on the bar.
    if (findLatestUserAudioBase64(runMessages)) return;

    // A completion finishing mid-count writes the exact usage for a turn this
    // count predates, so drop the recount rather than roll the bar backwards.
    // Sampled the moment runMessages is fixed (and after the saved-usage
    // restore above, which is our own write): the payload build awaits storage
    // and the project/RAG lookups, and a completion landing in that window
    // would otherwise be captured here and compare equal.
    const usageBeforeCount = useChatRuntimeStore.getState().contextUsage;

    // undefined, not null: a chat with no persisted thread has no project to
    // resolve instructions or a RAG scope from.
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

    // Always ask the server, even with nothing to send: the template itself has
    // tokens, and a process-level policy (unsloth run --enable-tools) injects
    // tool schemas and their nudge that the client cannot see.
    const result = await countChatInputTokens({
      model: capturedCheckpoint,
      messages: outbound,
      ...toolExtras,
    });
    const inputTokens = result.input_tokens;

    // The endpoint counts against whatever is loaded at the moment it runs, so a
    // model switch by another API client after our last status refresh is
    // invisible to the checkpoint guards below -- they only see that our own
    // store is unchanged. When the backend names the tokenizer it used, require
    // it to be ours. Older backends omit the field; keep counting for them.
    if (result.model != null && result.model !== capturedCheckpoint) return;

    if (superseded(capturedThreadId, generation)) return;
    if (useChatRuntimeStore.getState().params.checkpoint !== capturedCheckpoint) {
      return;
    }
    // Compared even when null: a count started on a chat with no persisted
    // thread must not land on a thread the user has since opened.
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
