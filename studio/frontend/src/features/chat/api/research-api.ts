// SPDX-License-Identifier: AGPL-3.0-only

import { authFetch } from "@/features/auth";
import type {
  CreateResearchRunInput,
  ResearchEvent,
  ResearchPlan,
  ResearchRun,
} from "../types/research";

type JsonObject = Record<string, unknown>;
const TERMINAL_RESEARCH_STATUSES = new Set([
  "completed",
  "failed",
  "cancelled",
]);

class ResearchApiError extends Error {
  readonly status: number;

  constructor(message: string, status: number) {
    super(message);
    this.name = "ResearchApiError";
    this.status = status;
  }
}

function camelize(value: unknown): unknown {
  if (Array.isArray(value)) {
    return value.map(camelize);
  }
  if (!value || typeof value !== "object") {
    return value;
  }
  return Object.fromEntries(
    Object.entries(value as JsonObject).map(([key, child]) => [
      key.replace(/_([a-z])/g, (_, letter: string) => letter.toUpperCase()),
      camelize(child),
    ]),
  );
}

async function json<T>(response: Response): Promise<T> {
  const body = await response.json().catch(() => null);
  if (!response.ok) {
    const detail = (body as { detail?: unknown; message?: unknown } | null)
      ?.detail;
    const message = (body as { message?: unknown } | null)?.message;
    throw new ResearchApiError(
      typeof detail === "string"
        ? detail
        : typeof message === "string"
          ? message
          : `Research request failed (${response.status})`,
      response.status,
    );
  }
  return camelize(body) as T;
}

export async function createResearchRun(
  input: CreateResearchRunInput,
): Promise<ResearchRun> {
  return json<ResearchRun>(
    await authFetch("/api/chat/research-runs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(input),
    }),
  );
}

export async function getResearchRun(
  id: string,
  signal?: AbortSignal,
): Promise<ResearchRun> {
  return json<ResearchRun>(
    await authFetch(`/api/chat/research-runs/${id}`, { signal }),
  );
}

export async function getResearchThreadState(
  threadId: string,
): Promise<{ activeRun: ResearchRun | null; hasRun: boolean }> {
  const query = new URLSearchParams({ threadId });
  const response = await authFetch(`/api/chat/research-runs/active?${query}`);
  if (response.status === 404) {
    return { activeRun: null, hasRun: false };
  }
  const { runs, hasRun } = await json<{
    runs: ResearchRun[];
    hasRun: boolean;
  }>(response);
  return { activeRun: runs.at(-1) ?? null, hasRun };
}

async function mutate(
  id: string,
  action: string,
  body?: Record<string, unknown>,
): Promise<ResearchRun> {
  return json<ResearchRun>(
    await authFetch(`/api/chat/research-runs/${id}/${action}`, {
      method: "POST",
      ...(body
        ? {
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(body),
          }
        : {}),
    }),
  );
}

export const approveResearchRun = (
  id: string,
  planRevision: number,
  planHash: string,
) => mutate(id, "approve", { planRevision, planHash });
export const cancelResearchRun = (id: string) => mutate(id, "cancel");
export const retryResearchRun = (id: string) => mutate(id, "retry");

export async function updateResearchPlan(
  id: string,
  plan: ResearchPlan,
  expectedRevision: number,
): Promise<ResearchRun> {
  return json<ResearchRun>(
    await authFetch(`/api/chat/research-runs/${id}/plan`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ plan, expectedRevision }),
    }),
  );
}

// biome-ignore lint/complexity/noExcessiveCognitiveComplexity: Incremental SSE parsing must retain framing state across reader chunks.
export async function* streamResearchEvents(
  id: string,
  after: number,
  signal?: AbortSignal,
): AsyncGenerator<ResearchEvent> {
  const response = await authFetch(
    `/api/chat/research-runs/${id}/events?after=${Math.max(0, after)}`,
    { headers: { accept: "text/event-stream" }, signal },
  );
  if (!response.ok) {
    await json(response);
  }
  if (!response.body) {
    throw new Error("Research event stream returned no response body");
  }
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  try {
    while (true) {
      const { done, value } = await reader.read();
      buffer += decoder.decode(value, { stream: !done }).replace(/\r\n/g, "\n");
      let boundary = buffer.indexOf("\n\n");
      while (boundary >= 0) {
        const block = buffer.slice(0, boundary);
        buffer = buffer.slice(boundary + 2);
        let event = "message";
        let eventId = after;
        const data: string[] = [];
        for (const line of block.split("\n")) {
          if (line.startsWith("id:")) {
            eventId = Number(line.slice(3).trim()) || eventId;
          } else if (line.startsWith("event:")) {
            event = line.slice(6).trim();
          } else if (line.startsWith("data:")) {
            data.push(line.slice(5).trimStart());
          }
        }
        if (data.length > 0) {
          const parsed = camelize(JSON.parse(data.join("\n"))) as JsonObject;
          const candidate = parsed.run as ResearchRun | undefined;
          if (candidate?.id && candidate.status) {
            yield {
              id: eventId,
              event: event as ResearchEvent["event"],
              createdAt:
                typeof parsed.createdAt === "number"
                  ? parsed.createdAt
                  : candidate.updatedAt,
              data: parsed as unknown as ResearchEvent["data"],
              run: candidate,
            };
          }
        }
        boundary = buffer.indexOf("\n\n");
      }
      if (done) {
        return;
      }
    }
  } finally {
    await reader.cancel().catch(() => undefined);
  }
}

export interface ResearchRunUpdate {
  run: ResearchRun;
  event?: ResearchEvent;
  source: "snapshot" | "event";
}

function isPermanentResearchError(error: unknown): boolean {
  return (
    error instanceof ResearchApiError &&
    error.status >= 400 &&
    error.status < 500 &&
    error.status !== 408 &&
    error.status !== 429
  );
}

function waitForReconnect(ms: number, signal?: AbortSignal): Promise<void> {
  if (signal?.aborted) {
    return Promise.resolve();
  }
  return new Promise((resolve) => {
    const finish = () => {
      window.clearTimeout(timer);
      signal?.removeEventListener("abort", finish);
      resolve();
    };
    const timer = window.setTimeout(finish, ms);
    signal?.addEventListener("abort", finish, { once: true });
  });
}

/** Follow a durable run across clean SSE EOFs and transient network failures. */
// biome-ignore lint/complexity/noExcessiveCognitiveComplexity: The retry, cursor, abort, and terminal states belong to one reconnect state machine.
export async function* followResearchRun(
  id: string,
  options: {
    initialRun?: ResearchRun;
    signal?: AbortSignal;
    replayFrom?: number;
  } = {},
): AsyncGenerator<ResearchRunUpdate> {
  const { signal, replayFrom } = options;
  let run = options.initialRun;
  let failures = 0;
  while (!(run || signal?.aborted)) {
    try {
      run = await getResearchRun(id, signal);
    } catch (error) {
      if (signal?.aborted) {
        return;
      }
      if (isPermanentResearchError(error)) {
        throw error;
      }
      failures += 1;
      await waitForReconnect(
        Math.min(8_000, 500 * 2 ** (failures - 1)),
        signal,
      );
    }
  }
  if (!run || signal?.aborted) {
    return;
  }
  failures = 0;
  yield { run, source: "snapshot" };
  if (
    (TERMINAL_RESEARCH_STATUSES.has(run.status) && replayFrom === undefined) ||
    signal?.aborted
  ) {
    return;
  }
  let cursor = replayFrom ?? run.lastEventSeq;
  while (!signal?.aborted) {
    try {
      for await (const event of streamResearchEvents(id, cursor, signal)) {
        cursor = Math.max(cursor, event.id);
        run = event.run;
        failures = 0;
        yield { run, event, source: "event" };
        if (
          (event.event === "run.completed" ||
            event.event === "run.failed" ||
            event.event === "run.cancelled") &&
          TERMINAL_RESEARCH_STATUSES.has(event.run.status) &&
          (event.data.attempt ?? 0) === (event.run.retryCount ?? 0)
        ) {
          return;
        }
      }
    } catch (error) {
      if (signal?.aborted) {
        return;
      }
      if (isPermanentResearchError(error)) {
        throw error;
      }
      failures += 1;
    }

    if (signal?.aborted) {
      return;
    }
    try {
      const fresh = await getResearchRun(id, signal);
      const changed =
        fresh.lastEventSeq !== run.lastEventSeq ||
        fresh.updatedAt !== run.updatedAt ||
        fresh.status !== run.status ||
        fresh.report !== run.report;
      const needsCatchup = cursor < fresh.lastEventSeq;
      run = fresh;
      if (replayFrom === undefined) {
        cursor = Math.max(cursor, fresh.lastEventSeq);
      }
      if (changed || needsCatchup) {
        yield { run, source: "snapshot" };
      }
      if (
        TERMINAL_RESEARCH_STATUSES.has(run.status) &&
        cursor >= run.lastEventSeq
      ) {
        return;
      }
    } catch (error) {
      if (signal?.aborted) {
        return;
      }
      if (isPermanentResearchError(error)) {
        throw error;
      }
      failures += 1;
    }
    await waitForReconnect(
      Math.min(8_000, 500 * 2 ** Math.max(0, failures - 1)),
      signal,
    );
  }
}
