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

// Per thread, not per module: in compare mode a hidden pane's history load would otherwise
// invalidate the visible thread's count, blanking the bar.
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
 * The branch the runtime displays for a stored thread, rebuilt as the history adapter does: sort
 * by (createdAt, role, id), parent legacy records to the previous one, then walk the last record's
 * ancestor chain. Greedy newest-child descent picks another branch and drops pre-parentId history.
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

/**
 * Identity of the branch a count priced. Content is hashed, not measured: a run mutates an existing
 * turn without moving the count or the last id (streaming grows its text; a tool result lands on a
 * part with no `text`), so neither a length nor a part tally sees it. Attachments are folded in
 * because the counter prices them too and deleting one rewrites `attachments` alone.
 */
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

/** The branch the mounted runtime is showing for the thread the store calls active. */
type ActiveBranchReader = () => readonly ThreadMessage[] | null;

let readActiveBranch: ActiveBranchReader | null = null;

/**
 * Publish the mounted runtime's visible branch so the recount prices it instead of the persisted
 * records. Only the single-chat pane registers one (compare panes never own the bar); null on unmount.
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

  // An output-only audio GGUF never sends a chat completion: the adapter routes the turn to
  // /audio/generate, which returns no usage, so a chat-template total is a number nothing corrects.
  const activeModel = store.models?.find(
    (model: { id: string }) => model.id === checkpoint,
  );
  if (activeModel?.isAudio && !activeModel?.hasAudioInput) return;

  // Never issue a count while the local model is decoding. /apply-template and /tokenize take
  // no inference slot, so the measured cost is inside the noise, but this endpoint shares a
  // process with generation and the budget here is zero rather than small. localRunByThreadId,
  // not runningByThreadId: an external-provider run never touches llama-server. The recount
  // effect lists this in its dependencies, so the last run finishing re-fires it.
  if (Object.values(store.localRunByThreadId ?? {}).some(Boolean)) return;

  const capturedThreadId = threadId ?? null;
  const capturedCheckpoint = checkpoint;

  // Bump only once this call will do work, so a bail-out cannot cancel an in-flight recount.
  const generation = nextGeneration(capturedThreadId);

  // The checkpoint can move under any await below, and a later recount for the same thread
  // supersedes this one; publishing after either puts another model's number on the bar.
  const stale = (): boolean =>
    superseded(capturedThreadId, generation) ||
    useChatRuntimeStore.getState().params.checkpoint !== capturedCheckpoint;

  try {
    // Prefer the mounted runtime: it is what the next request reads from. An incognito thread
    // persists nothing, and after a retry or edit the newest stored leaf is a branch the user
    // switched away from. A captured null is EXCLUDED, not matched: New Chat leaves the outgoing
    // conversation mounted until switchToNewThread() settles, so null === null would price that
    // branch into the empty chat -- which should be a bare template.
    const readOwnBranch = (): readonly ThreadMessage[] | null =>
      capturedThreadId != null &&
      useChatRuntimeStore.getState().activeThreadId === capturedThreadId
        ? (readActiveBranch?.() ?? null)
        : null;

    const liveBranch = readOwnBranch();

    let runMessages: readonly ThreadMessage[];
    // Re-read before publishing, so a turn sent while this count was in flight drops it.
    let countedBranch: string | null = null;
    // The stored fallback's witness: storage records and runtime ThreadMessages are different
    // shapes, so content hashes are not comparable; only ids survive both, and the last one moves
    // as soon as a turn is sent or an edit mints a message.
    let countedLastId: string | null = null;
    if (liveBranch && liveBranch.length > 0) {
      runMessages = liveBranch;
      countedBranch = branchSignature(liveBranch);
    } else {
      const records = threadId ? await listStoredChatMessages(threadId) : [];
      if (stale()) return;
      runMessages = orderBySelectedBranch(records).map(
        storedMessageToRunMessage,
      );
      countedLastId = runMessages.at(-1)?.id ?? "";
    }

    // The real request replays the newest user audio as audio_base64 but toOpenAIMessages has
    // no audio branch, so counting would price a text-only prompt. Decline as images do.
    if (findLatestUserAudioBase64(runMessages)) return;

    // A completion finishing mid-count writes exact usage for a turn this count predates, so drop
    // the recount rather than roll the bar backwards. Sampled as soon as runMessages is fixed: the
    // payload build only awaits storage, so anything landing in that window compares equal.
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
    const { input_tokens: inputTokens, model: countedModel } =
      await countChatInputTokens({
        model: capturedCheckpoint,
        messages: outbound,
        ...buildLocalTokenCountReasoning(),
        ...countExtras,
      });

    if (stale()) return;
    // The response type is a compile-time assertion only: anything else answering 200 here would
    // put undefined on the bar and throw from toLocaleString.
    if (typeof inputTokens !== "number" || !Number.isFinite(inputTokens)) return;
    // The endpoint counts with whatever is resident, never the model asked for: a load from
    // another tab returns another tokenizer's total, which the checkpoint guards cannot see.
    if (countedModel != null && countedModel !== capturedCheckpoint) {
      return;
    }
    // Compared even when null: a count started with no thread must not land on one since opened.
    if (useChatRuntimeStore.getState().activeThreadId !== capturedThreadId) {
      return;
    }
    if (useChatRuntimeStore.getState().contextUsage !== usageBeforeCount) {
      return;
    }
    // A run writes its own usage when it lands, so declining while one is live never loses a
    // number. A first turn has no thread id yet and files under "__default".
    if (useChatRuntimeStore.getState().runningByThreadId[capturedThreadId ?? "__default"]) {
      return;
    }
    // The usage snapshot above only sees a completion that WROTE usage, so a run stopped before
    // emitting any leaves it equal and the branch is the only witness. An empty current branch is
    // a mismatch: deleting the sole exchange mid-count would leave the old total on the thread.
    if (countedBranch != null) {
      const current = readActiveBranch?.();
      if (current != null && branchSignature(current) !== countedBranch) {
        return;
      }
    } else if (countedLastId != null) {
      // The count priced storage because the runtime had not mounted this thread yet; if it has
      // since, the last id is the one witness the two shapes share. readOwnBranch, not
      // readActiveBranch: an empty New Chat still sees the conversation it is leaving.
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
  } catch {
    // Background recount should not interrupt chat; saved usage stays visible.
  }
}
