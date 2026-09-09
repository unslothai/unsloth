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

/** The follower gave up on a run that stopped making progress. Distinct from the caller's Stop:
 *  the backend may still be generating, so the reply is incomplete. */
export class ChatGenerationStalledError extends Error {
  constructor(runId: string) {
    super(`Chat generation run ${runId} made no progress`);
    this.name = "ChatGenerationStalledError";
  }
}

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
      error.message === "Credentials cannot be persisted") ||
    (error instanceof ChatGenerationApiError &&
      error.status === 404 &&
      error.message === "Thread not found") ||
    (error instanceof ChatGenerationApiError &&
      error.status === 400 &&
      error.message ===
        "userMessageId must identify a user message in the thread")
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

/** Where a Stop has to be sent, given how far durable admission has got. Admission resolves well
 *  after the abort listener is installed: the turn first has to auto-load a model, retrieve
 *  RAG, upload attachments and save history. A Stop landing in that window has no run id yet
 *  and may still end up on the legacy stream, so it needs the `cancel_id` POST, whose server
 *  side stashes the cancel for a generation that registers afterwards. That POST is safe once
 *  the run exists too, since the durable request pins `cancel_id` to the same run id. */
export function chatGenerationStopPlan(
  decision: "pending" | "durable" | "legacy",
  runId: string | null,
): { cancelRunId: string | null; postLegacyCancel: boolean } {
  if (runId) return { cancelRunId: runId, postLegacyCancel: false };
  if (decision === "durable") return { cancelRunId: null, postLegacyCancel: false };
  return { cancelRunId: null, postLegacyCancel: true };
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

/** The events stream's own comment. Pinned by a test against the route that emits it. */
const KEEPALIVE_PREFIX = ": keep-alive";

// biome-ignore lint/complexity/noExcessiveCognitiveComplexity: SSE framing retains state across reader chunks.
async function* streamChatGenerationEvents(
  id: string,
  after: number,
  signal?: AbortSignal,
  /** Called for each event, and for each keep-alive with that keep-alive's progress stamp. A
   *  keep-alive is only progress if the stamp MOVED, which this generator cannot decide: it is
   *  re-invoked on every reconnect, so a per-connection memory would treat the first keep-alive
   *  after each reconnect as progress and rearm the caller's deadline forever. */
  onActivity?: (keepAliveStamp?: string) => void,
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
          // Reported, not judged. Whether this counts as progress depends on the last stamp seen across
          // ALL connections for this run, and this generator is re-invoked on every reconnect.
          else if (line.startsWith(KEEPALIVE_PREFIX)) {
            const stamp = line.slice(KEEPALIVE_PREFIX.length).trim();
            if (stamp !== "") onActivity?.(stamp);
          }
        }
        if (data.length > 0) {
          const event = JSON.parse(data.join("\n")) as ChatGenerationEvent;
          if (event.type === "chunk") {
            event.payload = normalizeChatGenerationChunkPayload(event.payload);
          }
          onActivity?.();
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

/** How long a follower tolerates a run that makes no progress. Progress, not connectedness, ends
 *  a follow: every event and every change to the run row resets it. A deadline on SILENCE
 *  rather than duration is what lets it stay short while the backend tolerates far longer
 *  work. Preparation waits emit no events, but the keep-alive comments carry the run's
 *  progress stamp, which the lease renewals move, so a two hour download rearms this while a
 *  wedged run does not. Bytes alone are deliberately NOT progress: keep-alives keep arriving
 *  for as long as the socket holds. */
export const CHAT_GENERATION_STALL_TIMEOUT_MS = 30 * 60_000;

/** Replay from the caller's applied cursor and reconnect until the run is terminal. */
export async function* followChatGenerationRun(
  id: string,
  options: {
    initialRun?: ChatGenerationRun;
    replayFrom?: number;
    signal?: AbortSignal;
    /** Overridable so a test can reach the deadline without waiting out the default. */
    stallTimeoutMs?: number;
  } = {},
): AsyncGenerator<ChatGenerationRunUpdate> {
  const { replayFrom } = options;
  const stallTimeoutMs =
    options.stallTimeoutMs ?? CHAT_GENERATION_STALL_TIMEOUT_MS;
  // The deadline must reach the open stream as well as the sleep between reconnects: a stream
  // that stays open and sends nothing parks the reader just as permanently. One controller
  // downstream of the caller's signal covers both.
  const deadline = new AbortController();
  const callerSignal = options.signal;
  const forwardAbort = () => deadline.abort(callerSignal?.reason);
  if (callerSignal?.aborted) {
    deadline.abort(callerSignal.reason);
  } else {
    callerSignal?.addEventListener("abort", forwardAbort, { once: true });
  }
  const signal = deadline.signal;
  let stallTimer: ReturnType<typeof globalThis.setTimeout> | undefined;
  // Both abort `signal` but must not end the same way: a caller abort is a clean stop, while the
  // deadline means we walked away from a run the backend may still be working on, so it must
  // reach the consumer as a failure rather than a complete reply.
  let stalled = false;
  let settled = false;
  // Spans reconnects on purpose: see the onActivity contract in streamChatGenerationEvents.
  let lastKeepAliveStamp: string | null = null;
  const noteProgress = (keepAliveStamp?: string): void => {
    if (keepAliveStamp !== undefined) {
      if (keepAliveStamp === lastKeepAliveStamp) return;
      lastKeepAliveStamp = keepAliveStamp;
    }
    if (signal.aborted) return;
    if (stallTimer !== undefined) globalThis.clearTimeout(stallTimer);
    stallTimer = globalThis.setTimeout(() => {
      stalled = true;
      deadline.abort(new ChatGenerationStalledError(id));
    }, stallTimeoutMs);
  };

  try {
    noteProgress();
    let run = options.initialRun;
    let failures = 0;
    while (!(run || signal.aborted)) {
      try {
        run = await getChatGenerationRun(id, signal);
      } catch (error) {
        if (signal.aborted) return;
        if (isPermanent(error)) throw error;
        failures += 1;
        await waitForReconnect(reconnectDelay(failures), signal);
      }
    }
    if (!run || signal.aborted) return;
    let currentRun = run;
    let cursor = replayFrom ?? run.lastEventSeq;
    yield { run, source: "snapshot" };
    if (isTerminalChatGenerationRun(run) && replayFrom === undefined) {
      settled = true;
      return;
    }

    while (!signal.aborted) {
      try {
        for await (const event of streamChatGenerationEvents(
          id,
          cursor,
          signal,
          noteProgress,
        )) {
          if (event.seq <= cursor) continue;
          cursor = event.seq;
          if (event.run) currentRun = event.run;
          failures = 0;
          noteProgress();
          yield { run: currentRun, event, source: "event" };
          if (
            isTerminalChatGenerationRun(currentRun) &&
            cursor >= currentRun.lastEventSeq
          ) {
            settled = true;
            return;
          }
        }
      } catch (error) {
        if (signal.aborted) return;
        if (isPermanent(error)) throw error;
        failures += 1;
      }
      if (signal.aborted) return;
      try {
        const fresh = await getChatGenerationRun(id, signal);
        const changed =
          fresh.status !== currentRun.status ||
          fresh.updatedAt !== currentRun.updatedAt ||
          fresh.lastEventSeq !== currentRun.lastEventSeq;
        currentRun = fresh;
        if (changed || cursor < fresh.lastEventSeq) {
          noteProgress();
          yield { run: fresh, source: "snapshot" };
        }
        if (isTerminalChatGenerationRun(fresh) && cursor >= fresh.lastEventSeq) {
          settled = true;
          return;
        }
      } catch (error) {
        if (signal.aborted) return;
        if (isPermanent(error)) throw error;
        failures += 1;
      }
      await waitForReconnect(reconnectDelay(failures), signal);
    }
  } finally {
    if (stallTimer !== undefined) globalThis.clearTimeout(stallTimer);
    callerSignal?.removeEventListener("abort", forwardAbort);
    // Every exit path funnels here, including the two `while (!signal.aborted)` loop conditions, so
    // this is the one place that catches all of them. `settled` keeps a run that reached a
    // terminal status on the same tick as the timer from being reported as stalled.
    if (stalled && !settled) throw new ChatGenerationStalledError(id);
  }
}
