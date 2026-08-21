// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// eslint-disable-next-line no-restricted-imports -- Avoid the auth barrel's React login page.
import { authFetch } from "@/features/auth/api";
import type {
  OpenAIChatChunk,
  OpenAIChatCompletionsRequest,
} from "../types/api";

export type ChatGenerationStatus =
  | "queued"
  | "running"
  | "cancelling"
  | "cancelled"
  | "completed"
  | "failed";

export interface ChatGenerationRun {
  id: string;
  threadId: string;
  userMessageId: string;
  assistantMessageId: string;
  requestHash: string;
  requestPayload: OpenAIChatCompletionsRequest;
  status: ChatGenerationStatus;
  cancelRequested: boolean;
  lastEventSeq: number;
  finishReason: string | null;
  error: string | null;
  createdAt: number;
  updatedAt: number;
  startedAt: number | null;
  completedAt: number | null;
  created?: boolean;
}

export interface CreateChatGenerationRunInput {
  runId: string;
  threadId: string;
  userMessageId: string;
  assistantMessageId: string;
  requestPayload: OpenAIChatCompletionsRequest;
}

export interface ChatGenerationEvent {
  seq: number;
  type: string;
  payload: OpenAIChatChunk | Record<string, unknown>;
  createdAt: number;
  run?: ChatGenerationRun;
}

export interface ChatGenerationRunUpdate {
  run: ChatGenerationRun;
  event?: ChatGenerationEvent;
  source: "snapshot" | "event";
}

export function normalizeChatGenerationChunkPayload(
  payload: OpenAIChatChunk | Record<string, unknown>,
): OpenAIChatChunk | Record<string, unknown> {
  if (
    payload !== null &&
    typeof payload === "object" &&
    "type" in payload &&
    payload.type === "reasoning_summary"
  ) {
    return {
      _reasoningDurationMs: (payload as { duration_ms?: unknown }).duration_ms,
    } as unknown as OpenAIChatChunk;
  }
  return payload;
}

const TERMINAL_STATUSES = new Set<ChatGenerationStatus>([
  "cancelled",
  "completed",
  "failed",
]);

export class ChatGenerationApiError extends Error {
  readonly status: number;

  constructor(message: string, status: number) {
    super(message);
    this.name = "ChatGenerationApiError";
    this.status = status;
  }
}

export function isToolEnabledChatGenerationAdmissionError(
  error: unknown,
): boolean {
  return (
    error instanceof ChatGenerationApiError &&
    error.status === 400 &&
    error.message === "Tool-enabled chat runs use the legacy streaming path"
  );
}

export function isLegacyFallbackChatGenerationAdmissionError(
  error: unknown,
): boolean {
  return (
    isToolEnabledChatGenerationAdmissionError(error) ||
    (error instanceof ChatGenerationApiError &&
      error.status === 400 &&
      error.message === "Credentials cannot be persisted")
  );
}

async function json<T>(response: Response): Promise<T> {
  const body = await response.json().catch(() => null);
  if (!response.ok) {
    const detail = (body as { detail?: unknown; message?: unknown } | null)
      ?.detail;
    const message = (body as { message?: unknown } | null)?.message;
    throw new ChatGenerationApiError(
      typeof detail === "string"
        ? detail
        : typeof message === "string"
          ? message
          : `Chat generation request failed (${response.status})`,
      response.status,
    );
  }
  return body as T;
}

function isPermanent(error: unknown): boolean {
  return (
    error instanceof ChatGenerationApiError &&
    error.status >= 400 &&
    error.status < 500 &&
    error.status !== 408 &&
    error.status !== 429
  );
}

function reconnectDelay(failures: number): number {
  return Math.min(8_000, 500 * 2 ** Math.max(0, failures - 1));
}

function waitForReconnect(ms: number, signal?: AbortSignal): Promise<void> {
  if (signal?.aborted) return Promise.resolve();
  return new Promise((resolve) => {
    const finish = () => {
      globalThis.clearTimeout(timer);
      signal?.removeEventListener("abort", finish);
      resolve();
    };
    const timer = globalThis.setTimeout(finish, ms);
    signal?.addEventListener("abort", finish, { once: true });
  });
}

export function isTerminalChatGenerationRun(run: ChatGenerationRun): boolean {
  return TERMINAL_STATUSES.has(run.status);
}

export function explicitStopSignal(signal: AbortSignal): {
  signal: AbortSignal;
  dispose: () => void;
} {
  const controller = new AbortController();
  const forward = () => {
    const detached = Boolean(
      (signal.reason as { detach?: boolean } | undefined)?.detach,
    );
    if (!detached) controller.abort(signal.reason);
  };
  if (signal.aborted) {
    forward();
  } else {
    signal.addEventListener("abort", forward, { once: true });
  }
  return {
    signal: controller.signal,
    dispose: () => signal.removeEventListener("abort", forward),
  };
}

export async function supportsChatGenerationRuns(
  threadId: string,
  signal?: AbortSignal,
): Promise<boolean> {
  const query = new URLSearchParams({ threadId });
  const response = await authFetch(
    `/api/inference/chat-runs/active?${query.toString()}`,
    { signal },
  );
  if (response.status === 404 || response.status === 405) return false;
  await json<{ runs: ChatGenerationRun[] }>(response);
  return true;
}

export async function getActiveChatGenerationRuns(
  threadId: string,
  signal?: AbortSignal,
): Promise<ChatGenerationRun[]> {
  const query = new URLSearchParams({ threadId });
  const response = await authFetch(
    `/api/inference/chat-runs/active?${query.toString()}`,
    { signal },
  );
  if (response.status === 404 || response.status === 405) return [];
  return (await json<{ runs: ChatGenerationRun[] }>(response)).runs ?? [];
}

export async function createChatGenerationRun(
  input: CreateChatGenerationRunInput,
): Promise<ChatGenerationRun> {
  let failures = 0;
  while (true) {
    try {
      return await json<ChatGenerationRun>(
        await authFetch("/api/inference/chat-runs", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(input),
        }),
      );
    } catch (error) {
      if (isPermanent(error)) throw error;
      failures += 1;
      await waitForReconnect(reconnectDelay(failures));
    }
  }
}

/** Start idempotently, but let an explicit Stop return before a slow create reply. */
export async function createChatGenerationRunUntilAbort(
  input: CreateChatGenerationRunInput,
  signal: AbortSignal,
): Promise<ChatGenerationRun | null> {
  const createPromise = createChatGenerationRun(input);
  let resolveAbort: (() => void) | undefined;
  const aborted = new Promise<null>((resolve) => {
    resolveAbort = () => resolve(null);
    if (signal.aborted) {
      resolveAbort();
    } else {
      signal.addEventListener("abort", resolveAbort, { once: true });
    }
  });
  try {
    const run = await Promise.race([createPromise, aborted]);
    if (run) {
      return run;
    }
    const detached = Boolean(
      (signal.reason as { detach?: boolean } | undefined)?.detach,
    );
    createPromise
      .then((created) =>
        detached ? undefined : cancelChatGenerationRun(created.id),
      )
      .catch(() => undefined);
    return null;
  } finally {
    if (resolveAbort) {
      signal.removeEventListener("abort", resolveAbort);
    }
  }
}

export async function getChatGenerationRun(
  id: string,
  signal?: AbortSignal,
): Promise<ChatGenerationRun> {
  return json<ChatGenerationRun>(
    await authFetch(`/api/inference/chat-runs/${encodeURIComponent(id)}`, {
      signal,
    }),
  );
}

export async function cancelChatGenerationRun(
  id: string,
): Promise<ChatGenerationRun> {
  return json<ChatGenerationRun>(
    await authFetch(
      `/api/inference/chat-runs/${encodeURIComponent(id)}/cancel`,
      { method: "POST" },
    ),
  );
}

// biome-ignore lint/complexity/noExcessiveCognitiveComplexity: SSE framing retains state across reader chunks.
async function* streamChatGenerationEvents(
  id: string,
  after: number,
  signal?: AbortSignal,
): AsyncGenerator<ChatGenerationEvent> {
  const response = await authFetch(
    `/api/inference/chat-runs/${encodeURIComponent(id)}/events?after=${Math.max(0, after)}`,
    { method: "POST", headers: { accept: "text/event-stream" }, signal },
  );
  if (!response.ok) await json(response);
  if (!response.body)
    throw new Error("Chat generation event stream returned no body");
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  try {
    while (true) {
      const { done, value } = await reader.read();
      buffer += decoder.decode(value, { stream: !done });
      buffer = buffer.replace(/\r\n/g, "\n");
      let boundary = buffer.indexOf("\n\n");
      while (boundary >= 0) {
        const block = buffer.slice(0, boundary);
        buffer = buffer.slice(boundary + 2);
        const data: string[] = [];
        for (const line of block.split("\n")) {
          if (line.startsWith("data:")) data.push(line.slice(5).trimStart());
        }
        if (data.length > 0) {
          const event = JSON.parse(data.join("\n")) as ChatGenerationEvent;
          if (event.type === "chunk") {
            event.payload = normalizeChatGenerationChunkPayload(event.payload);
          }
          yield event;
        }
        boundary = buffer.indexOf("\n\n");
      }
      if (done) return;
    }
  } finally {
    await reader.cancel().catch(() => undefined);
  }
}

/** Replay from the caller's applied cursor and reconnect until the run is terminal. */
export async function* followChatGenerationRun(
  id: string,
  options: {
    initialRun?: ChatGenerationRun;
    replayFrom?: number;
    signal?: AbortSignal;
  } = {},
): AsyncGenerator<ChatGenerationRunUpdate> {
  const { replayFrom, signal } = options;
  let run = options.initialRun;
  let failures = 0;
  while (!(run || signal?.aborted)) {
    try {
      run = await getChatGenerationRun(id, signal);
    } catch (error) {
      if (signal?.aborted) return;
      if (isPermanent(error)) throw error;
      failures += 1;
      await waitForReconnect(reconnectDelay(failures), signal);
    }
  }
  if (!run || signal?.aborted) return;
  let currentRun = run;
  let cursor = replayFrom ?? run.lastEventSeq;
  yield { run, source: "snapshot" };
  if (isTerminalChatGenerationRun(run) && replayFrom === undefined) return;

  while (!signal?.aborted) {
    try {
      for await (const event of streamChatGenerationEvents(
        id,
        cursor,
        signal,
      )) {
        if (event.seq <= cursor) continue;
        cursor = event.seq;
        if (event.run) currentRun = event.run;
        failures = 0;
        yield { run: currentRun, event, source: "event" };
        if (
          isTerminalChatGenerationRun(currentRun) &&
          cursor >= currentRun.lastEventSeq
        ) {
          return;
        }
      }
    } catch (error) {
      if (signal?.aborted) return;
      if (isPermanent(error)) throw error;
      failures += 1;
    }
    if (signal?.aborted) return;
    try {
      const fresh = await getChatGenerationRun(id, signal);
      const changed =
        fresh.status !== currentRun.status ||
        fresh.updatedAt !== currentRun.updatedAt ||
        fresh.lastEventSeq !== currentRun.lastEventSeq;
      currentRun = fresh;
      if (changed || cursor < fresh.lastEventSeq) {
        yield { run: fresh, source: "snapshot" };
      }
      if (isTerminalChatGenerationRun(fresh) && cursor >= fresh.lastEventSeq)
        return;
    } catch (error) {
      if (signal?.aborted) return;
      if (isPermanent(error)) throw error;
      failures += 1;
    }
    await waitForReconnect(reconnectDelay(failures), signal);
  }
}
