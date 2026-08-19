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

// Threads with a count on the wire. A model load fires two triggers (the post-load call and the
// modelLoading effect), and both would otherwise render the template and tokenize.
const countsInFlight = new Set<string | null>();

// A trigger deferred behind an in-flight count, replayed once that count settles WITHOUT
// publishing. Dropping it loses the bar: a run stopped before it emits usage fires the retry this
// depends on, and the in-flight count then rejects its now-stale branch. Replaying it
// unconditionally would restore the doubled model-load count, where the first trigger publishes.
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

// Same order the history adapter sorts stored records in before importing them.
const ROLE_ORDER: Record<string, number> = { system: 0, user: 1, assistant: 2 };

/**
 * The displayed branch, rebuilt as the history adapter does: sort by (createdAt, role, id), parent
 * legacy records to the previous one, then walk the last record's ancestor chain. Greedy
 * newest-child descent would pick another branch and drop pre-parentId history.
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
 * Identity of the branch a count priced. Content is hashed, not measured: a run mutates a turn in
 * place (streaming grows its text; a tool result lands on a part with no `text`), so neither a
 * length nor a part tally sees it. Attachments are folded in because the counter prices them too.
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
 * records. Only the single-chat pane registers one; compare panes never own the bar.
 */
export function setActiveBranchReader(reader: ActiveBranchReader | null): void {
  readActiveBranch = reader;
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

  if (options?.invalidate) store.setContextUsage(null);

  // Never count while anything is generating: the endpoint refuses, and the recount effect depends
  // on this, so the last run finishing re-fires it. runningByThreadId, not the narrower
  // localRunByThreadId: the endpoint refuses during an external-provider run too, so gating on less
  // than the server refuses on would spend a request to be told 503 and lose the retry -- nothing
  // re-fires this effect when an external run ends unless it is in the dependency array.
  if (Object.values(store.runningByThreadId ?? {}).some(Boolean)) return;

  const capturedThreadId = threadId ?? null;
  const capturedCheckpoint = checkpoint;

  if (countsInFlight.has(capturedThreadId)) {
    retryAfterInFlight.set(capturedThreadId, options);
    return;
  }

  // Bump only once this call will do work, so a bail-out cannot cancel an in-flight recount.
  const generation = nextGeneration(capturedThreadId);

  // The checkpoint can move under any await below, and a later recount for the same thread
  // supersedes this one; publishing after either puts another model's number on the bar.
  const stale = (): boolean =>
    superseded(capturedThreadId, generation) ||
    useChatRuntimeStore.getState().params.checkpoint !== capturedCheckpoint;

  countsInFlight.add(capturedThreadId);
  let published = false;
  try {
    // Prefer the mounted runtime: it is what the next request reads from (an incognito thread
    // persists nothing, and after a retry the newest stored leaf is a branch the user left). A
    // captured null is EXCLUDED, not matched: New Chat leaves the outgoing conversation mounted
    // until switchToNewThread() settles, so null === null would price it into the empty chat.
    const readOwnBranch = (): readonly ThreadMessage[] | null =>
      capturedThreadId != null &&
      useChatRuntimeStore.getState().activeThreadId === capturedThreadId
        ? (readActiveBranch?.() ?? null)
        : null;

    const liveBranch = readOwnBranch();

    let runMessages: readonly ThreadMessage[];
    // Re-read before publishing, so a turn sent while this count was in flight drops it.
    let countedBranch: string | null = null;
    // The stored fallback's witness: storage records and ThreadMessages hash differently, so only
    // ids survive both, and the last one moves as soon as a turn is sent or an edit mints one.
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

    // /chat/count_tokens always 503s on images: /apply-template swaps each for a marker. Declining
    // before the hash below keeps the base64 out of it and out of a request body that can reach
    // megabytes, both synchronous on the UI thread.
    if (messagesContainImage(runMessages)) return;

    // The real request replays the newest user audio as audio_base64 but toOpenAIMessages has
    // no audio branch, so counting would price a text-only prompt. Decline as images do.
    if (findLatestUserAudioBase64(runMessages)) return;

    // Same for video, and more so: the real request replays the clip as
    // video_base64 and llama-server expands it into frames, while
    // toOpenAIMessages has no video branch -- so counting would price a
    // text-only prompt and the bar would show room the window does not have.
    // /chat/count_tokens 503s on video for the same reason. Declining here also
    // keeps up to 85 MB of base64 out of branchSignature's JSON.stringify,
    // which is the synchronous main-thread cost the image bail above exists for.
    if (findLatestUserVideoBase64(runMessages)) return;

    if (fromLiveBranch) {
      countedBranch = branchSignature(runMessages);
    } else {
      countedLastId = runMessages.at(-1)?.id ?? "";
    }

    // A completion finishing mid-count writes exact usage for a turn this count predates, so drop
    // the recount rather than roll the bar backwards. Sampled once runMessages is fixed.
    const usageBeforeCount = useChatRuntimeStore.getState().contextUsage;

    // undefined, not null: a chat with no persisted thread has no project to resolve from.
    const payloadThreadId = threadId ?? undefined;
    const countHistory = await buildLocalTokenCountHistory(
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
        ...countHistory,
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
