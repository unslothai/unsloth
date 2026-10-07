// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// eslint-disable-next-line no-restricted-imports -- Avoid the auth barrel's React login page.
import { authFetch } from "@/features/auth/api";
import type {
  OpenAIChatChunk,
  OpenAIChatCompletionsRequest,
} from "../types/api";

import { skillLoadCardEvent } from "./skill-load-event";
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
  if (payload !== null && typeof payload === "object" && "type" in payload) {
    const frameType = (payload as { type?: unknown }).type;
    if (frameType === "skill_load") {
      return { _toolEvent: skillLoadCardEvent(payload) } as unknown as OpenAIChatChunk;
    }
    if (frameType === "reasoning_summary") {
      return {
        _reasoningDurationMs: (payload as { duration_ms?: unknown }).duration_ms,
      } as unknown as OpenAIChatChunk;
    }
    // Persisted control frames arrive as plain chunks (the decoder drops the SSE event field); re-tag
    // them as chat-api.ts does or a resumed run loses tool activity.
    if (frameType === "tool_status") {
      return {
        _toolStatus: (payload as { content?: unknown }).content ?? "",
      } as unknown as OpenAIChatChunk;
    }
    if (
      frameType === "tool_start" ||
      frameType === "tool_end" ||
      frameType === "tool_output" ||
      frameType === "tool_args"
    ) {
      // The consumer accumulates tool events by tag, so keep the frame whole.
      return { _toolEvent: payload } as unknown as OpenAIChatChunk;
    }
    if (frameType === "diffusion_frame") {
      return { _diffusionFrame: payload } as unknown as OpenAIChatChunk;
    }
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
    // A media payload rejection is a policy signal to use the legacy stream, not an error.
    (error instanceof ChatGenerationApiError &&
      error.status === 400 &&
      error.message === "Media chat runs use the legacy streaming path") ||
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

/** A Stop before admission resolves has no run id and may still go legacy, so it needs the
 *  `cancel_id` POST, which the server stashes; also safe once the run exists. */
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

/** A reopened tab cannot tell an answered approval from a parked one, so ask the server. 404/405
 *  (older backend) means "cannot tell", keeping the arm-everything fallback. */
export async function toolApprovalIsPending(
  approvalId: string,
  sessionId: string,
  signal?: AbortSignal,
): Promise<boolean> {
  const response = await authFetch("/api/inference/tool-approval-status", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ approval_id: approvalId, session_id: sessionId }),
    signal,
  });
  if (response.status === 404 || response.status === 405) return true;
  return (await json<{ pending?: boolean }>(response)).pending !== false;
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
  /** A keep-alive is progress only if its stamp MOVED, judged by the caller across reconnects. */
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
          // Reported, not judged: progress depends on stamps across all reconnects of this run.
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

/** Deadline on no PROGRESS, not duration: keep-alives carry a progress stamp that lease renewals
 *  move. Bytes alone are not progress. */
export const CHAT_GENERATION_STALL_TIMEOUT_MS = 30 * 60_000;

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
  // One controller downstream of the caller covers both the open stream and reconnect sleeps.
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
  // Caller abort is a clean stop; the deadline must surface as a failure, not a complete reply.
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
    // Every exit funnels here; `settled` stops a same-tick terminal run being reported stalled.
    if (stalled && !settled) throw new ChatGenerationStalledError(id);
  }
}
