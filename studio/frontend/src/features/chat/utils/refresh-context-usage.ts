// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { ThreadMessage } from "@assistant-ui/react";
import {
  buildLocalTokenCountExtras,
  buildLocalTokenCountReasoning,
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

/** The branch the mounted runtime is showing for the thread the store calls active. */
type ActiveBranchReader = () => readonly ThreadMessage[] | null;

let readActiveBranch: ActiveBranchReader | null = null;

/**
 * Publish the mounted runtime's view of the visible branch, so the recount can price it
 * instead of the persisted records. Only the single-chat pane registers one; compare
 * panes never own the bar. Pass null on unmount.
 */
export function setActiveBranchReader(reader: ActiveBranchReader | null): void {
  readActiveBranch = reader;
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

  // The checkpoint can move under any await below (another load, or the user switching
  // back), and a later recount for the same thread supersedes this one; publishing after
  // either is another model's number on the bar.
  const stale = (): boolean =>
    superseded(capturedThreadId, generation) ||
    useChatRuntimeStore.getState().params.checkpoint !== capturedCheckpoint;

  try {
    // The mounted runtime is what the next request reads from, so prefer it: a
    // temporary/incognito thread persists nothing (listStoredChatMessages returns [] by
    // design) and would otherwise be priced as a bare template, and after a retry or an
    // edit the newest stored leaf is a branch the user has switched away from. Only for
    // the thread the store calls active, since that is the one the bar belongs to; the
    // history loader's own call runs before the import, so it falls through below.
    const liveBranch =
      useChatRuntimeStore.getState().activeThreadId === capturedThreadId
        ? readActiveBranch?.()
        : null;

    let runMessages: readonly ThreadMessage[];
    if (liveBranch && liveBranch.length > 0) {
      runMessages = liveBranch;
    } else {
      const records = threadId ? await listStoredChatMessages(threadId) : [];
      if (stale()) return;
      runMessages = orderBySelectedBranch(records).map(
        storedMessageToRunMessage,
      );
    }

    // The real request replays the newest user audio as audio_base64 but toOpenAIMessages has
    // no audio branch, so counting would price a text-only prompt. Decline as images do.
    if (findLatestUserAudioBase64(runMessages)) return;

    // A completion finishing mid-count writes exact usage for a turn this count predates, so
    // drop the recount rather than roll the bar backwards. Sampled as soon as runMessages is
    // fixed: the payload build awaits storage, and a completion landing in that window would
    // otherwise be captured here and compare equal.
    const usageBeforeCount = useChatRuntimeStore.getState().contextUsage;

    // undefined, not null: a chat with no persisted thread has no project to resolve from.
    const payloadThreadId = threadId ?? undefined;
    const outbound = await buildOutboundMessagesForTokenCount(
      runMessages,
      payloadThreadId,
    );
    if (stale()) return;
    const countExtras = await buildLocalTokenCountExtras(payloadThreadId);
    if (stale()) return;

    // Always ask the server: the template itself has tokens, and `unsloth run --enable-tools`
    // injects schemas the client cannot see.
    const { input_tokens: inputTokens } = await countChatInputTokens({
      model: capturedCheckpoint,
      messages: outbound,
      ...buildLocalTokenCountReasoning(),
      ...countExtras,
    });

    if (stale()) return;
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
