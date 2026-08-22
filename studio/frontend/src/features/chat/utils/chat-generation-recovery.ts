// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { extractDeltaText } from "./parse-assistant-content";

export type StoredGenerationStatus =
  | "queued"
  | "running"
  | "cancelling"
  | "cancelled"
  | "completed"
  | "failed";

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
): Promise<{ messages: TMessage[]; activeRuns: TRun[] }> {
  // Runs first closes the create-between-snapshots gap. If a run commits after
  // this read, the later message snapshot already carries its durable metadata.
  const activeRuns = await listActiveRuns(threadId).catch(() => []);
  const messages = await listMessages(threadId);
  return { messages, activeRuns };
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

export function generationNeedsRecovery(
  metadata: Record<string, unknown>,
): boolean {
  const status = String(metadata.generationStatus) as StoredGenerationStatus;
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
  if (status === "completed" && settled) {
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
