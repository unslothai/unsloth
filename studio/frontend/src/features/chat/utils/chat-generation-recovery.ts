// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { ChatGenerationStatus } from "../api/chat-generation-api";
import {
  createThinkTagTracker,
  extractDeltaText,
  parseAssistantContent,
} from "./parse-assistant-content";

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
        quote_cut?: unknown;
        usage?: unknown;
        choices?: unknown[];
      }
    | null
    | undefined;
  if (!chunk || typeof chunk !== "object") return false;
  if ("_reasoningDurationMs" in chunk || chunk.context_truncated || chunk.quote_cut) {
    return false;
  }
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

export function createRecoveryPublishSchedule(
  intervalMs: number,
  now: () => number = Date.now,
): {
  attach(lastEventSeq: number): void;
  shouldPublish(cursor: number, settled: boolean): boolean;
  takeSave(settled: boolean): boolean;
} {
  let attachSeq = 0;
  let lastSaveAt = Number.NEGATIVE_INFINITY;
  return {
    attach(lastEventSeq) {
      attachSeq = lastEventSeq;
    },
    shouldPublish(cursor, settled) {
      return settled || cursor >= attachSeq;
    },
    takeSave(settled) {
      const at = now();
      if (!settled && at - lastSaveAt < intervalMs) return false;
      lastSaveAt = at;
      return true;
    },
  };
}

export async function loadGenerationOverlaySnapshot<TMessage, TRun>(
  threadId: string,
  listActiveRuns: (id: string) => Promise<TRun[]>,
  listMessages: (id: string) => Promise<TMessage[]>,
): Promise<{
  messages: TMessage[];
  activeRuns: TRun[];
  /** False when the read failed, so an empty list means unknown, not none. */
  activeRunsLoaded: boolean;
}> {
  // Reading runs first closes the create-between-snapshots gap.
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
  prompt_tokens_details?: { cached_tokens?: unknown; cache_write_tokens?: unknown };
  cache_creation_input_tokens?: unknown;
  cache_read_input_tokens?: unknown;
};

export function usageCacheWriteTokens(usage: RecoveryUsage | undefined): number {
  if (typeof usage?.cache_creation_input_tokens === "number") {
    return usage.cache_creation_input_tokens;
  }
  const details = usage?.prompt_tokens_details?.cache_write_tokens;
  return typeof details === "number" ? details : 0;
}

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
  toolCalls?: string[];
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
      cacheWriteTokens: usageCacheWriteTokens(usage),
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
      toolCalls: options.toolCalls ?? [],
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
      toolCallCount: options.toolCalls?.length ?? 0,
    };
  }
  return next;
}

// Keep offsets aligned with parseAssistantContent's tags.
const THINK_OPEN = "<think>";
const THINK_CLOSE = "</think>";

export type CarriedPart = { at: number; part: unknown };

export function generationRawContent(content: unknown): {
  raw: string;
  reasoningOpen: boolean;
  carried: CarriedPart[];
} {
  if (typeof content === "string") {
    return { raw: content, reasoningOpen: false, carried: [] };
  }
  if (!Array.isArray(content)) {
    return { raw: "", reasoningOpen: false, carried: [] };
  }
  let raw = "";
  let reasoningOpen = false;
  const carried: CarriedPart[] = [];
  for (const part of content) {
    if (!part || typeof part !== "object") continue;
    const record = part as { type?: string; text?: unknown };
    const text = typeof record.text === "string" ? record.text : "";
    if (record.type === "reasoning") {
      if (reasoningOpen) raw += text;
      else raw += `${THINK_OPEN}${text}`;
      reasoningOpen = true;
    } else if (record.type === "text") {
      if (reasoningOpen) raw += THINK_CLOSE;
      raw += text;
      reasoningOpen = false;
    } else {
      carried.push({ at: raw.length, part });
    }
  }
  return { raw, reasoningOpen, carried };
}

export function restoreCarriedParts<TPart>(
  parts: readonly TPart[],
  carried: readonly CarriedPart[],
): TPart[] {
  if (carried.length === 0) return [...parts];
  const pending = [...carried].sort((a, b) => a.at - b.at);
  const out: TPart[] = [];
  let next = 0;
  let offset = 0;
  let reasoningOpen = false;
  const flushUpTo = (limit: number) => {
    while (next < pending.length && pending[next].at <= limit) {
      out.push(pending[next].part as TPart);
      next += 1;
    }
  };
  for (const part of parts) {
    const record = part as { type?: string; text?: unknown };
    const text = typeof record.text === "string" ? record.text : "";
    flushUpTo(offset);
    if (record.type === "reasoning") {
      if (!reasoningOpen) offset += THINK_OPEN.length;
      reasoningOpen = true;
    } else if (record.type === "text") {
      if (reasoningOpen) offset += THINK_CLOSE.length;
      reasoningOpen = false;
    } else {
      out.push(part);
      continue;
    }
    flushUpTo(offset);
    let cut = 0;
    while (next < pending.length && pending[next].at < offset + text.length) {
      const at = pending[next].at - offset;
      if (at > cut) out.push({ ...record, text: text.slice(cut, at) } as TPart);
      out.push(pending[next].part as TPart);
      cut = at;
      next += 1;
    }
    if (cut === 0) out.push(part);
    else if (cut < text.length) {
      out.push({ ...record, text: text.slice(cut) } as TPart);
    }
    offset += text.length;
  }
  while (next < pending.length) {
    out.push(pending[next].part as TPart);
    next += 1;
  }
  return out;
}

/** Think tags can split across chunks; treat a tag as atomic and put the card past it. */
function pastThinkTag(raw: string, at: number): number {
  for (const tag of [THINK_OPEN, THINK_CLOSE]) {
    const start = raw.lastIndexOf(tag, at);
    if (start !== -1 && at > start && at < start + tag.length) {
      return start + tag.length;
    }
  }
  return at;
}

/** Split before parsing, since parsing can coalesce separate think blocks. */
export function restoreCarriedPartsFromRaw(
  raw: string,
  carried: readonly CarriedPart[],
  { parseThink = true }: { parseThink?: boolean } = {},
): ReturnType<typeof parseAssistantContent> {
  if (carried.length === 0) return parseAssistantContent(raw, { parseThink });
  const out: ReturnType<typeof parseAssistantContent> = [];
  const tracker = createThinkTagTracker();
  let cursor = 0;
  const appendUntil = (end: number) => {
    const text = raw.slice(cursor, end);
    out.push(
      ...parseAssistantContent(
        parseThink && tracker.endsInsideThink() ? `${THINK_OPEN}${text}` : text,
        { parseThink },
      ),
    );
    tracker.append(text);
    cursor = end;
  };
  for (const entry of [...carried].sort((a, b) => a.at - b.at)) {
    appendUntil(
      Math.max(cursor, pastThinkTag(raw, Math.min(entry.at, raw.length))),
    );
    out.push(entry.part as (typeof out)[number]);
  }
  appendUntil(raw.length);
  return out;
}

// Read from the stored request: the placeholder lacks parseThinkTags until the first save.
export function requestParsesThinkTags(payload: {
  enable_thinking?: boolean | null;
  reasoning_effort?: string | null;
  thinking?: { type?: string } | null;
}): boolean {
  return !(
    payload.thinking?.type === "disabled" ||
    payload.enable_thinking === false ||
    payload.reasoning_effort === "none"
  );
}

function carriedPartKey({ at, part }: CarriedPart): string {
  const record = part as { type?: string; toolCallId?: string; id?: string };
  const id = record.toolCallId ?? record.id;
  return id === undefined
    ? JSON.stringify([at, part])
    : JSON.stringify([record.type, id]);
}

type ToolIdentity = {
  type?: string;
  toolCallId?: string;
  toolName?: string;
  backendToolCallId?: string;
  generationToolCallId?: string;
  toolApprovalId?: string;
};

function followingCarriedMatches(matches: (number | undefined)[]) {
  let next: number | undefined;
  const following = matches.map(() => undefined as number | undefined);
  for (let i = matches.length - 1; i >= 0; i--) {
    following[i] = next;
    next = matches[i] ?? next;
  }
  return following;
}

function carriedPartMatches(view: CarriedPart[], recovered: CarriedPart[]) {
  // Every occurrence: sources are not deduplicated, so a repeated url needs a slot each.
  const byId = new Map<string, number[]>();
  recovered.forEach((entry, i) => {
    const key = carriedPartKey(entry);
    const seen = byId.get(key);
    if (seen) seen.push(i);
    else byId.set(key, [i]);
  });
  const byGeneration = new Map<string, number>();
  recovered.forEach(({ part }, i) => {
    const id = (part as ToolIdentity).generationToolCallId;
    if (id) byGeneration.set(id, i);
  });
  const used = new Set<number>();
  const matches = view.map((entry) => {
    const id = (entry.part as ToolIdentity).generationToolCallId;
    const index =
      byId.get(carriedPartKey(entry))?.shift() ??
      (id ? byGeneration.get(id) : undefined);
    if (index === undefined || used.has(index)) return undefined;
    used.add(index);
    return index;
  });
  const following = followingCarriedMatches(matches);
  let previous: number | undefined;
  view.forEach((entry, i) => {
    const live = entry.part as ToolIdentity;
    if (matches[i] !== undefined) {
      previous = matches[i];
      return;
    }
    if (live.type !== "tool-call" || live.generationToolCallId) return;
    const next = following[i];
    const index = recovered.findIndex((candidate, j) => {
      const saved = candidate.part as ToolIdentity;
      if (
        used.has(j) ||
        (previous !== undefined && j <= previous) ||
        (next !== undefined && j >= next) ||
        candidate.at !== entry.at ||
        saved.type !== "tool-call" ||
        !saved.generationToolCallId ||
        saved.toolName !== live.toolName
      )
        return false;
      if (
        saved.toolApprovalId &&
        (live.toolApprovalId === saved.toolApprovalId ||
          live.toolCallId === saved.toolApprovalId ||
          live.toolCallId?.endsWith(`:${saved.toolApprovalId}`))
      )
        return true;
      if (saved.toolApprovalId && live.toolApprovalId) return false;
      const backendId = saved.backendToolCallId;
      return (
        Boolean(backendId) &&
        (live.backendToolCallId !== undefined
          ? live.backendToolCallId === backendId
          : live.toolCallId === backendId ||
            live.toolCallId?.startsWith(`${backendId}:`))
      );
    });
    if (index < 0) return;
    matches[i] = index;
    previous = index;
    used.add(index);
  });
  return matches;
}

function mergeCarriedParts(
  view: CarriedPart[],
  recovered: CarriedPart[],
  matches: (number | undefined)[],
): CarriedPart[] {
  const before = new Map<number, CarriedPart[]>();
  const after = new Map<number, CarriedPart[]>();
  let previous: number | undefined;
  const following = followingCarriedMatches(matches);
  view.forEach((entry, i) => {
    const match = matches[i];
    if (match !== undefined) {
      previous = match;
      return;
    }
    const buckets = previous === undefined ? before : after;
    const anchor = previous ?? following[i] ?? 0;
    const bucket = buckets.get(anchor) ?? [];
    bucket.push(entry);
    buckets.set(anchor, bucket);
  });
  if (recovered.length === 0) return view;
  return recovered.flatMap((entry, i) => [
    ...(before.get(i) ?? []),
    entry,
    ...(after.get(i) ?? []),
  ]);
}

export function recoveredContentToImport<TContent>(
  viewContent: TContent,
  recoveredContent: TContent,
): TContent {
  const view = generationRawContent(viewContent);
  const recovered = generationRawContent(recoveredContent);
  if (
    recovered.raw.length < view.raw.length &&
    view.raw.startsWith(recovered.raw)
  ) {
    return viewContent;
  }
  if (
    view.carried.length > 0 &&
    recovered.raw.startsWith(view.raw) &&
    Array.isArray(recoveredContent)
  ) {
    const matches = carriedPartMatches(view.carried, recovered.carried);
    if (matches.every((index) => index !== undefined)) {
      return recoveredContent;
    }
    // Only a recovered reply with text disagrees; an empty projection prefixes every reply.
    if (!view.raw && recovered.raw) {
      return recoveredContent;
    }
    const spoken = recoveredContent.filter(
      (part) =>
        (part as { type?: string })?.type === "text" ||
        (part as { type?: string })?.type === "reasoning",
    );
    return restoreCarriedParts(
      spoken,
      mergeCarriedParts(view.carried, recovered.carried, matches),
    ) as TContent;
  }
  return recoveredContent;
}

export function generationNeedsRecovery(
  metadata: Record<string, unknown>,
): boolean {
  const status = String(metadata.generationStatus) as StoredGenerationStatus;
  // Set by a follower that hit its no-progress deadline. Non-terminal only: completed runs are
  // absent from /chat-runs/active, so honouring it would leave them running forever.
  if (metadata.generationLocallyInterrupted === true && !TERMINAL.has(status)) {
    return false;
  }
  return (
    typeof metadata.generationRunId === "string" &&
    (metadata.generationSettled !== true || !TERMINAL.has(status))
  );
}

/** Usage arrives before the terminal event, so a saved cursor must carry it or lose it. */
export function generationReplayMetadata(state: {
  cursor: number;
  firstChunkAt?: number;
  totalChunks?: number;
  usage?: unknown;
  timings?: unknown;
}): Record<string, unknown> {
  const next: Record<string, unknown> = { generationSeq: state.cursor };
  if (state.firstChunkAt !== undefined) {
    next.generationFirstChunkAt = state.firstChunkAt;
  }
  if (state.totalChunks !== undefined) {
    next.generationChunkCount = state.totalChunks;
  }
  if (state.usage !== undefined) {
    next.generationRecoveryUsage = state.usage;
  }
  if (state.timings !== undefined) {
    next.generationRecoveryTimings = state.timings;
  }
  return next;
}

export function generationRecoveryMetadata(options: {
  current: Record<string, unknown>;
  runId: string;
  status: StoredGenerationStatus;
  cursor: number;
  lastEventSeq: number;
  lengthLimited: boolean;
  quoteCut?: boolean;
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
    quoteCut = false,
    firstChunkAt,
    totalChunks,
    usage,
    timings,
  } = options;
  const settled = generationIsSettled(status, cursor, lastEventSeq);
  const next: Record<string, unknown> = {
    ...current,
    ...generationReplayMetadata({
      cursor,
      firstChunkAt,
      totalChunks,
      usage,
      timings,
    }),
    generationRunId: runId,
    generationStatus: status,
    generationSettled: settled,
    serverManaged: true,
  };
  if (status === "completed") {
    if (lengthLimited) {
      next.incomplete = { reason: "length" };
    } else if (quoteCut) {
      // Must match the producer's stamp, or the server refuses the settle.
      next.incomplete = { reason: "quote_cut" };
    } else {
      next.incomplete = undefined;
    }
  } else if (status === "failed") {
    next.incomplete = { reason: "interrupted" };
  } else {
    next.incomplete = { reason: "cancelled" };
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
  const onFocus = () => {
    if (documentTarget.visibilityState !== "hidden") {
      recover();
    }
  };
  windowTarget.addEventListener("online", recover);
  windowTarget.addEventListener("pageshow", recover);
  windowTarget.addEventListener("focus", onFocus);
  documentTarget.addEventListener("visibilitychange", onVisible);
  return () => {
    windowTarget.removeEventListener("online", recover);
    windowTarget.removeEventListener("pageshow", recover);
    windowTarget.removeEventListener("focus", onFocus);
    documentTarget.removeEventListener("visibilitychange", onVisible);
  };
}

/** Runs this tab streams itself, so recovery never adds a second (quadratic) reader.
 *  Module state: a reload is exactly when this tab stops being the producer. */
const liveGenerationRuns = new Set<string>();
const liveGenerationThreads = new Map<string, string>();

/** Claimed before server admission so a recovery trigger cannot start a second follower. */
const provisionalGenerationRuns = new Set<string>();

const recoveredRunStops = new Map<string, () => void>();

export function registerRecoveredRunStop(
  threadId: string,
  stop: () => void,
): () => void {
  recoveredRunStops.set(threadId, stop);
  return () => {
    if (recoveredRunStops.get(threadId) === stop) {
      recoveredRunStops.delete(threadId);
    }
  };
}

export function stopRecoveredRun(threadId: string | null | undefined): boolean {
  const stop = threadId ? recoveredRunStops.get(threadId) : undefined;
  stop?.();
  return stop !== undefined;
}

export function claimLiveGenerationRun(
  runId: string,
  threadId?: string,
  options?: { provisional?: boolean },
): void {
  liveGenerationRuns.add(runId);
  if (threadId) liveGenerationThreads.set(runId, threadId);
  if (options?.provisional) {
    provisionalGenerationRuns.add(runId);
  } else {
    provisionalGenerationRuns.delete(runId);
  }
}

/** Must run unconditionally in a finally, or this tab never recovers the run. */
export function releaseLiveGenerationRun(runId: string): void {
  liveGenerationRuns.delete(runId);
  liveGenerationThreads.delete(runId);
  provisionalGenerationRuns.delete(runId);
}

export function isLiveGenerationRun(runId: string): boolean {
  return liveGenerationRuns.has(runId);
}

/** Subscriber-owned streams have no durable run; their checkpoint is the only persistence. */
export function threadHasDurableGenerationRun(threadId: string): boolean {
  for (const [runId, owner] of liveGenerationThreads.entries()) {
    if (owner === threadId && !provisionalGenerationRuns.has(runId)) return true;
  }
  for (const owner of serverActiveGenerationRuns.values()) {
    if (owner === threadId) return true;
  }
  return false;
}

/** Only /chat-runs/active proves a run is live; stored "running" metadata can be stale forever. */
const serverActiveGenerationRuns = new Map<string, string>();

/** Call only after a successful read; a failed read is not "nothing running". */
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

/** Per thread: a global flag let one thread's answer mask another's failed read. */
const serverAnsweredThreads = new Set<string>();

export function serverHasAnsweredActiveRuns(threadId: string): boolean {
  return serverAnsweredThreads.has(threadId);
}

export function markServerActiveGenerationRunsUnknown(threadId: string): void {
  serverAnsweredThreads.delete(threadId);
  // Drop run mappings too: a leftover entry restores "running" with no follower or Stop handle.
  for (const [runId, owner] of [...serverActiveGenerationRuns]) {
    if (owner === threadId) serverActiveGenerationRuns.delete(runId);
  }
}

/** Drop a terminal run, or the thread stays durable and later streams lose saves. */
export function forgetServerActiveGenerationRun(runId: string): void {
  serverActiveGenerationRuns.delete(runId);
}

export function resetServerActiveGenerationRuns(): void {
  serverActiveGenerationRuns.clear();
  serverAnsweredThreads.clear();
  liveGenerationThreads.clear();
  liveGenerationRuns.clear();
  provisionalGenerationRuns.clear();
}

export function isServerActiveGenerationRun(runId: string): boolean {
  return serverActiveGenerationRuns.has(runId);
}

export function generationIsCorroboratedLive(
  metadata: Record<string, unknown>,
  threadId?: string,
): boolean {
  const runId = metadata.generationRunId;
  if (typeof runId !== "string") return false;
  if (isLiveGenerationRun(runId) || isServerActiveGenerationRun(runId)) return true;
  // A follower already gave up locally; only the server naming the run live may revive it.
  if (
    metadata.generationLocallyInterrupted === true &&
    !TERMINAL.has(String(metadata.generationStatus) as StoredGenerationStatus)
  ) {
    return false;
  }
  // No answer for this thread is not a "no"; keep the persisted status until its read lands.
  return threadId === undefined || !serverHasAnsweredActiveRuns(threadId);
}
