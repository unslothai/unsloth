// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { ChatGenerationStatus } from "../api/chat-generation-api";
import { extractDeltaText } from "./parse-assistant-content";

export type StoredGenerationStatus = ChatGenerationStatus;

const TERMINAL = new Set<StoredGenerationStatus>([
  "cancelled",
  "completed",
  "failed",
]);

export function generationChunkHasSubstantiveDelta(payload: unknown): boolean {
  const delta = (
    payload as {
      choices?: Array<{
        delta?: {
          content?: unknown;
          reasoning_content?: unknown;
          reasoning_details?: unknown;
        };
      }>;
    }
  )?.choices?.[0]?.delta;
  const reasoning =
    typeof delta?.reasoning_content === "string" ? delta.reasoning_content : "";
  const reasoningDetails = Array.isArray(delta?.reasoning_details)
    ? delta.reasoning_details.some(
        (part) =>
          part !== null &&
          typeof part === "object" &&
          typeof (part as { text?: unknown }).text === "string" &&
          Boolean((part as { text: string }).text),
      )
    : false;
  return Boolean(
    reasoning || reasoningDetails || extractDeltaText(delta?.content).text,
  );
}

export function generationChunkCountsTowardTiming(payload: unknown): boolean {
  const chunk = payload as
    | {
        _reasoningDurationMs?: unknown;
        context_truncated?: unknown;
        usage?: unknown;
        choices?: unknown[];
      }
    | null
    | undefined;
  if (!chunk || typeof chunk !== "object") return false;
  if ("_reasoningDurationMs" in chunk || chunk.context_truncated) return false;
  return !(chunk.usage && Array.isArray(chunk.choices) && chunk.choices.length === 0);
}

export function recoveredReasoningSummaryMetadata(
  current: Record<string, unknown>,
  reasoningMs: unknown,
): Record<string, unknown> {
  if (
    typeof reasoningMs !== "number" ||
    !Number.isFinite(reasoningMs) ||
    reasoningMs < 0
  ) {
    return current;
  }
  const durations = Array.isArray(current.reasoningDurations)
    ? current.reasoningDurations.filter(
        (duration): duration is number =>
          typeof duration === "number" && Number.isFinite(duration),
      )
    : [];
  const duration = Math.max(0, Math.round(reasoningMs / 1000));
  return {
    ...current,
    reasoningDuration: duration,
    reasoningDurations: [...durations, duration],
  };
}

export function generationIsSettled(
  status: StoredGenerationStatus | null,
  cursor: number,
  lastEventSeq: number,
): boolean {
  return status !== null && TERMINAL.has(status) && cursor >= lastEventSeq;
}

export async function loadGenerationOverlaySnapshot<TMessage, TRun>(
  threadId: string,
  listActiveRuns: (id: string) => Promise<TRun[]>,
  listMessages: (id: string) => Promise<TMessage[]>,
): Promise<{
  messages: TMessage[];
  activeRuns: TRun[];
  /** False when the active-run read failed, so its empty list is "unknown", not "none". A caller
   *  deciding a reply is dead from an absent run has to tell the two apart. */
  activeRunsLoaded: boolean;
}> {
  // Runs first closes the create-between-snapshots gap. If a run commits after this read, the
  // later message snapshot already carries its durable metadata.
  let activeRunsLoaded = true;
  const activeRuns = await listActiveRuns(threadId).catch(() => {
    activeRunsLoaded = false;
    return [];
  });
  const messages = await listMessages(threadId);
  return { messages, activeRuns, activeRunsLoaded };
}

type RecoveryUsage = {
  prompt_tokens?: unknown;
  completion_tokens?: unknown;
  total_tokens?: unknown;
  prompt_tokens_details?: { cached_tokens?: unknown };
  cache_creation_input_tokens?: unknown;
  cache_read_input_tokens?: unknown;
};

type RecoveryTimings = {
  cache_n?: unknown;
  predicted_per_second?: unknown;
  [key: string]: unknown;
};

export function recoveredGenerationFinalMetadata(options: {
  current: Record<string, unknown>;
  run: {
    id: string;
    requestPayload: { model?: unknown };
    createdAt: number;
    startedAt: number | null;
    completedAt: number | null;
  };
  usage?: RecoveryUsage;
  timings?: RecoveryTimings;
  firstChunkAt?: number;
  totalChunks: number;
}): Record<string, unknown> {
  const { current, run, usage, timings, firstChunkAt, totalChunks } = options;
  const modelId =
    typeof run.requestPayload.model === "string"
      ? run.requestPayload.model
      : "Unknown model";
  const startedAt = run.startedAt ?? run.createdAt;
  const finishedAt = run.completedAt ?? Date.now();
  const completionTokens =
    typeof usage?.completion_tokens === "number"
      ? usage.completion_tokens
      : undefined;
  const tokensPerSecond =
    typeof timings?.predicted_per_second === "number"
      ? timings.predicted_per_second
      : completionTokens !== undefined && finishedAt > startedAt
        ? completionTokens / ((finishedAt - startedAt) / 1000)
        : undefined;
  const next = { ...current };

  if (next.serverTimings === undefined && timings !== undefined) {
    next.serverTimings = timings;
  }
  if (
    next.contextUsage === undefined &&
    typeof usage?.prompt_tokens === "number" &&
    completionTokens !== undefined &&
    typeof usage.total_tokens === "number"
  ) {
    next.contextUsage = {
      promptTokens: usage.prompt_tokens,
      completionTokens,
      totalTokens: usage.total_tokens,
      cachedTokens:
        (typeof timings?.cache_n === "number" ? timings.cache_n : undefined) ??
        (typeof usage.prompt_tokens_details?.cached_tokens === "number"
          ? usage.prompt_tokens_details.cached_tokens
          : undefined) ??
        (typeof usage.cache_read_input_tokens === "number"
          ? usage.cache_read_input_tokens
          : 0),
      cacheWriteTokens:
        typeof usage.cache_creation_input_tokens === "number"
          ? usage.cache_creation_input_tokens
          : 0,
      modelId,
    };
  }
  if (next.responseDetails === undefined) {
    next.responseDetails = {
      modelId,
      modelLabel: modelId,
      responseModelId: modelId,
      providerName: "Local model",
      providerType: "local",
      startedAt,
      finishedAt,
      durationMs: Math.max(0, finishedAt - startedAt),
      cancelId: run.id,
      toolCalls: [],
    };
  }
  if (next.timing === undefined) {
    next.timing = {
      streamStartTime: startedAt,
      firstTokenTime:
        firstChunkAt === undefined ? undefined : Math.max(0, firstChunkAt - startedAt),
      totalStreamTime: Math.max(0, finishedAt - startedAt),
      tokenCount: completionTokens,
      tokensPerSecond,
      totalChunks,
      toolCallCount: 0,
    };
  }
  return next;
}

/** The reply as one string, with reasoning back inside `<think>` tags. The recovery replays a
 *  run's chunk events, so it keeps the reply as the text those chunks carried rather than as
 *  parts. Inverse of `parseAssistantContent`, so a parts body can be compared to a delta one. */
export function generationRawContent(content: unknown): {
  raw: string;
  reasoningOpen: boolean;
} {
  if (typeof content === "string") {
    return { raw: content, reasoningOpen: false };
  }
  if (!Array.isArray(content)) return { raw: "", reasoningOpen: false };
  let raw = "";
  let reasoningOpen = false;
  for (const part of content) {
    if (!part || typeof part !== "object") continue;
    const record = part as { type?: string; text?: unknown };
    const text = typeof record.text === "string" ? record.text : "";
    if (record.type === "reasoning") {
      if (reasoningOpen) raw += text;
      else raw += `<think>${text}`;
      reasoningOpen = true;
    } else if (record.type === "text") {
      if (reasoningOpen) raw += "</think>";
      raw += text;
      reasoningOpen = false;
    }
  }
  return { raw, reasoningOpen };
}

/** Which body a recovery publish should show. A recovery replays the run's events from the
 *  last saved cursor, one publish per event, each paying a storage write. When the replayed
 *  run is also the one this tab is streaming, that walk is far behind the live stream and
 *  its body is a PREFIX of what the reader sees, so importing it rewinds the reply twice a
 *  second. Prefix, not length: a body that genuinely disagrees is the server's and wins,
 *  since storage is authoritative. Only a body carrying nothing new is refused. */
export function recoveredContentToImport<TContent>(
  viewContent: TContent,
  recoveredContent: TContent,
): TContent {
  const view = generationRawContent(viewContent).raw;
  const recovered = generationRawContent(recoveredContent).raw;
  if (recovered.length < view.length && view.startsWith(recovered)) {
    return viewContent;
  }
  return recoveredContent;
}

export function generationNeedsRecovery(
  metadata: Record<string, unknown>,
): boolean {
  const status = String(metadata.generationStatus) as StoredGenerationStatus;
  // Stamped by a follower that hit its no-progress deadline. Re-following on every recovery
  // trigger turned one stuck run into a permanently blocked composer; history.load clears the
  // marker if /chat-runs/active still names the run. Non-terminal only: a run the backend
  // finished is stored completed, and /chat-runs/active excludes completed runs, so honouring
  // the marker there would leave that reply running forever.
  if (metadata.generationLocallyInterrupted === true && !TERMINAL.has(status)) {
    return false;
  }
  return (
    typeof metadata.generationRunId === "string" &&
    (metadata.generationSettled !== true || !TERMINAL.has(status))
  );
}

export function generationRecoveryMetadata(options: {
  current: Record<string, unknown>;
  runId: string;
  status: StoredGenerationStatus;
  cursor: number;
  lastEventSeq: number;
  lengthLimited: boolean;
  firstChunkAt?: number;
  totalChunks?: number;
  usage?: unknown;
  timings?: unknown;
}): Record<string, unknown> {
  const {
    current,
    runId,
    status,
    cursor,
    lastEventSeq,
    lengthLimited,
    firstChunkAt,
    totalChunks,
    usage,
    timings,
  } = options;
  const settled = generationIsSettled(status, cursor, lastEventSeq);
  const next: Record<string, unknown> = {
    ...current,
    generationRunId: runId,
    generationSeq: cursor,
    generationStatus: status,
    generationSettled: settled,
    serverManaged: true,
  };
  if (status === "completed") {
    if (lengthLimited) {
      next.incomplete = { reason: "length" };
    } else {
      next.incomplete = undefined;
    }
  } else if (status === "failed") {
    next.incomplete = { reason: "interrupted" };
  } else {
    next.incomplete = { reason: "cancelled" };
  }
  if (firstChunkAt !== undefined) {
    next.generationFirstChunkAt = firstChunkAt;
  }
  if (totalChunks !== undefined) {
    next.generationChunkCount = totalChunks;
  }
  // Carried with the cursor for the same reason as the two above: the usage chunk arrives before
  // the terminal event, so a cursor published past it and reloaded would resume after it and
  // lose the token counts and server timings for good.
  if (usage !== undefined) {
    next.generationRecoveryUsage = usage;
  }
  if (timings !== undefined) {
    next.generationRecoveryTimings = timings;
  }
  return next;
}

export function shouldPreserveGenerationMetadata(
  existing: Record<string, unknown> | undefined,
  incoming: Record<string, unknown> | undefined,
): boolean {
  if (typeof existing?.generationRunId !== "string") {
    return false;
  }
  const sameRun = existing.generationRunId === incoming?.generationRunId;
  const existingSeq = Number(existing.generationSeq ?? -1);
  const incomingSeq = Number(incoming?.generationSeq ?? -1);
  const existingStatus = String(existing.generationStatus);
  return (
    !sameRun ||
    incoming?.serverManaged !== true ||
    existingSeq > incomingSeq ||
    (TERMINAL.has(existingStatus as StoredGenerationStatus) &&
      incoming?.generationStatus !== existing.generationStatus) ||
    (existing.generationSettled === true &&
      incoming?.generationSettled !== true)
  );
}

type RecoveryEventTarget = Pick<
  EventTarget,
  "addEventListener" | "removeEventListener"
>;
type RecoveryVisibilityTarget = RecoveryEventTarget & {
  readonly visibilityState: string;
};

export function subscribeGenerationRecoveryTriggers(
  windowTarget: RecoveryEventTarget,
  documentTarget: RecoveryVisibilityTarget,
  recover: () => void,
): () => void {
  const onVisible = () => {
    if (documentTarget.visibilityState === "visible") {
      recover();
    }
  };
  windowTarget.addEventListener("online", recover);
  windowTarget.addEventListener("pageshow", recover);
  documentTarget.addEventListener("visibilitychange", onVisible);
  return () => {
    windowTarget.removeEventListener("online", recover);
    windowTarget.removeEventListener("pageshow", recover);
    documentTarget.removeEventListener("visibilitychange", onVisible);
  };
}

/** Runs this tab is streaming itself, so a recovery never follows one. A durable run otherwise
 *  gets TWO readers in the tab that started it: the adapter streams it and
 *  scheduleGenerationRecovery replays it from storage, since its only gate is
 *  generationNeedsRecovery and history.load force-writes generationSettled false. That
 *  second reader publishes on EVERY chunk, each publish re-parsing the whole reply and
 *  awaiting a PUT of the entire message: quadratic in the answer length, on the main thread.
 *  Module state deliberately: a reload is exactly when this tab has stopped being the
 *  producer, and a reload clears this by construction. */
const liveGenerationRuns = new Set<string>();
// runId -> owning thread. The Set above answers "is this run live"; the checkpoint scheduler
// needs the inverse, "does this thread have a durable run at all".
const liveGenerationThreads = new Map<string, string>();

/** Runs claimed before the server admitted them. The early claim stops a recovery trigger
 *  starting a second follower during that await. Boundedness is separate: until the POST
 *  lands the thread's checkpoints are its only persistence, and the create retries until
 *  aborted, so the await can outlast the cap. */
const provisionalGenerationRuns = new Set<string>();

/** Claim a run as streamed by this tab. Pair with `releaseLiveGenerationRun` in a finally. */
export function claimLiveGenerationRun(
  runId: string,
  threadId?: string,
  options?: { provisional?: boolean },
): void {
  liveGenerationRuns.add(runId);
  if (threadId) liveGenerationThreads.set(runId, threadId);
  // A later non-provisional claim confirms admission, so this clears as well as sets.
  if (options?.provisional) {
    provisionalGenerationRuns.add(runId);
  } else {
    provisionalGenerationRuns.delete(runId);
  }
}

/** Release a run this tab was streaming. Must be unconditional, in a finally: a run left
 *  claimed after its stream died is one this tab will never recover. */
export function releaseLiveGenerationRun(runId: string): void {
  liveGenerationRuns.delete(runId);
  liveGenerationThreads.delete(runId);
  provisionalGenerationRuns.delete(runId);
}

/** Whether this tab is the one streaming `runId`. */
export function isLiveGenerationRun(runId: string): boolean {
  return liveGenerationRuns.has(runId);
}

/** Whether `threadId` has a durable run, streaming here or named by the server. A
 *  subscriber-owned stream has none, and its periodic checkpoint is its ONLY persistence. */
export function threadHasDurableGenerationRun(threadId: string): boolean {
  for (const [runId, owner] of liveGenerationThreads.entries()) {
    if (owner === threadId && !provisionalGenerationRuns.has(runId)) return true;
  }
  for (const owner of serverActiveGenerationRuns.values()) {
    if (owner === threadId) return true;
  }
  return false;
}

/** Runs the server has named as still going, keyed by the thread whose load asked. Persisted
 *  metadata is not evidence of a live run: a run that never terminalises leaves
 *  `generationStatus: "running"` in storage for good. Only /chat-runs/active can say a run
 *  is still going. Module state, so a reload drops the previous session's word for it. */
const serverActiveGenerationRuns = new Map<string, string>();

/** Replace what the server last said about `threadId`. Call only after a SUCCESSFUL read of
 *  the active-run list: a failed read is not a report of "nothing is running", and treating
 *  it as one would mark a live reply interrupted. */
export function syncServerActiveGenerationRuns(
  threadId: string,
  runIds: Iterable<string>,
): void {
  for (const [runId, owner] of [...serverActiveGenerationRuns]) {
    if (owner === threadId) serverActiveGenerationRuns.delete(runId);
  }
  for (const runId of runIds) serverActiveGenerationRuns.set(runId, threadId);
  serverAnsweredThreads.add(threadId);
}

/** Threads the server has answered the active-run question for. Per thread, not per process:
 *  one global flag let A's answer turn a FAILED read for B into an empty list. */
const serverAnsweredThreads = new Set<string>();

export function serverHasAnsweredActiveRuns(threadId: string): boolean {
  return serverAnsweredThreads.has(threadId);
}

/** Forget what the server last said about `threadId`, because the newest read failed. The
 *  answer is a point in time, not a permanent property: another tab can start a run after a
 *  successful read, so keeping the old answer would restore a running reply as interrupted. */
export function markServerActiveGenerationRunsUnknown(threadId: string): void {
  serverAnsweredThreads.delete(threadId);
  // The run mappings go with it. They are the same stale answer in another shape, and
  // isServerActiveGenerationRun is consulted BEFORE the local-interruption marker, so a
  // leftover entry restores the message as running while generationNeedsRecovery refuses to
  // start a follower: a blocked composer with neither recovery nor a local Stop handle.
  for (const [runId, owner] of [...serverActiveGenerationRuns]) {
    if (owner === threadId) serverActiveGenerationRuns.delete(runId);
  }
}

/** Drop one run from the server-active map now that it has reached a terminal status. Only
 *  another successful sync would otherwise remove it, so the thread would keep reading as
 *  durable and a later subscriber-owned stream on it would be capped, losing its only saves. */
export function forgetServerActiveGenerationRun(runId: string): void {
  serverActiveGenerationRuns.delete(runId);
}

/** Test-only: forget every active-run answer. */
export function resetServerActiveGenerationRuns(): void {
  serverActiveGenerationRuns.clear();
  serverAnsweredThreads.clear();
  liveGenerationThreads.clear();
  liveGenerationRuns.clear();
  provisionalGenerationRuns.clear();
}

/** Whether the server has named `runId` as still going in this session. */
export function isServerActiveGenerationRun(runId: string): boolean {
  return serverActiveGenerationRuns.has(runId);
}

/** Whether a restored assistant message may be shown as still generating. The corroboration
 *  gate: unfinished metadata says only that this reply once had a run. */
export function generationIsCorroboratedLive(
  metadata: Record<string, unknown>,
  threadId?: string,
): boolean {
  const runId = metadata.generationRunId;
  if (typeof runId !== "string") return false;
  if (isLiveGenerationRun(runId) || isServerActiveGenerationRun(runId)) return true;
  // A follower already gave up on this run locally. Without this it keeps taking the benefit of
  // the doubt below, so every online/pageshow/visibility trigger starts another follower that
  // republishes the message as running. Only the server naming the run live may revive it.
  if (
    metadata.generationLocallyInterrupted === true &&
    !TERMINAL.has(String(metadata.generationStatus) as StoredGenerationStatus)
  ) {
    return false;
  }
  // No answer for THIS thread is not a "no". Stay with the persisted status until this thread's
  // own read has landed; the recovery follower settles it from there.
  return threadId === undefined || !serverHasAnsweredActiveRuns(threadId);
}
