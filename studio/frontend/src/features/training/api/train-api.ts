// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";
import { readFastApiError } from "@/lib/format-fastapi-error";
import type {
  TrainingResetResponse,
  TrainingStartRequest,
  TrainingStartRequestStatusResponse,
  TrainingStartResponse,
  TrainingStopResponse,
} from "../types/api";
import type {
  TrainingMetricsResponse,
  TrainingProgressPayload,
  TrainingStatusResponse,
} from "../types/runtime";

function isAbortError(error: unknown): boolean {
  return error instanceof DOMException && error.name === "AbortError";
}

const readError = (r: Response): Promise<string> => readFastApiError(r);

class TrainingStartOutcomeUnknownError extends Error {
  constructor(error: unknown) {
    super(
      error instanceof Error
        ? error.message
        : "Training start outcome is unknown.",
    );
    this.name = "TrainingStartOutcomeUnknownError";
  }
}

export class TrainingStartError extends Error {
  readonly errorCode: string | null;

  constructor(message: string, errorCode: string | null = null) {
    super(message);
    this.name = "TrainingStartError";
    this.errorCode = errorCode;
  }
}

export function isTrainingStartOutcomeUnknownError(error: unknown): boolean {
  return error instanceof TrainingStartOutcomeUnknownError;
}

async function parseJson<T>(response: Response): Promise<T> {
  if (!response.ok) {
    throw new Error(await readError(response));
  }
  return (await response.json()) as T;
}

async function readTrainingStartError(
  response: Response,
): Promise<TrainingStartError> {
  const fallbackResponse = response.clone();
  try {
    const payload = (await response.json()) as { detail?: unknown };
    const detail = payload.detail;
    if (detail && typeof detail === "object" && !Array.isArray(detail)) {
      const structured = detail as { code?: unknown; message?: unknown };
      const message =
        typeof structured.message === "string" && structured.message
          ? structured.message
          : null;
      if (message) {
        return new TrainingStartError(
          message,
          typeof structured.code === "string" ? structured.code : null,
        );
      }
    }
  } catch {
    return new TrainingStartError(await readFastApiError(fallbackResponse));
  }
  return new TrainingStartError(await readFastApiError(fallbackResponse));
}

export async function startTraining(
  payload: TrainingStartRequest,
  startRequestId: string,
): Promise<TrainingStartResponse> {
  let response: Response;
  try {
    response = await authFetch(
      "/api/train/start",
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ...payload, start_request_id: startRequestId }),
      },
      { retryNetworkErrors: false },
    );
  } catch (error) {
    throw new TrainingStartOutcomeUnknownError(error);
  }
  if (!response.ok) {
    const error = await readTrainingStartError(response);
    await acknowledgeTrainingStartRequest(startRequestId).catch(
      () => undefined,
    );
    throw error;
  }
  try {
    const result = (await response.json()) as TrainingStartResponse;
    if (result.status === "error") {
      await acknowledgeTrainingStartRequest(startRequestId).catch(
        () => undefined,
      );
    }
    return result;
  } catch (error) {
    throw new TrainingStartOutcomeUnknownError(error);
  }
}

export async function getTrainingStartRequestStatus(
  startRequestId: string,
): Promise<TrainingStartRequestStatusResponse | null> {
  const response = await authFetch(
    `/api/train/start-requests/${encodeURIComponent(startRequestId)}`,
  );
  if (response.status === 404) {
    return null;
  }
  return parseJson<TrainingStartRequestStatusResponse>(response);
}

export async function acknowledgeTrainingStartRequest(
  startRequestId: string,
): Promise<void> {
  const response = await authFetch(
    `/api/train/start-requests/${encodeURIComponent(startRequestId)}/acknowledge`,
    { method: "POST" },
  );
  if (!response.ok) {
    throw new Error(await readError(response));
  }
}

interface TrainingJobScope {
  expectedJobId?: string;
}

function scopedTrainingBody(
  payload: Record<string, unknown>,
  scope?: TrainingJobScope,
): string {
  return JSON.stringify({
    ...payload,
    ...(scope?.expectedJobId !== undefined
      ? { expected_job_id: scope.expectedJobId }
      : {}),
  });
}

export async function stopTraining(
  save = true,
  scope?: TrainingJobScope,
): Promise<TrainingStopResponse> {
  const response = await authFetch("/api/train/stop", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: scopedTrainingBody({ save }, scope),
  });
  return parseJson<TrainingStopResponse>(response);
}

export async function resetTraining(
  scope?: TrainingJobScope,
): Promise<TrainingResetResponse> {
  const hasScope = scope?.expectedJobId !== undefined;
  const response = await authFetch("/api/train/reset", {
    method: "POST",
    ...(hasScope
      ? {
          headers: { "Content-Type": "application/json" },
          body: scopedTrainingBody({}, scope),
        }
      : {}),
  });
  return parseJson<TrainingResetResponse>(response);
}

export async function getTrainingStatus(): Promise<TrainingStatusResponse> {
  const response = await authFetch("/api/train/status");
  return parseJson<TrainingStatusResponse>(response);
}

export async function getTrainingMetrics(): Promise<TrainingMetricsResponse> {
  const response = await authFetch("/api/train/metrics");
  return parseJson<TrainingMetricsResponse>(response);
}

type ProgressEventName = "progress" | "heartbeat" | "complete" | "error";

interface ParsedSseEvent {
  event: ProgressEventName;
  payload: TrainingProgressPayload;
  id: number | null;
}

function parseSseEvent(rawEvent: string): ParsedSseEvent | null {
  const lines = rawEvent.split(/\r?\n/);
  let eventName: ProgressEventName = "progress";
  let id: number | null = null;
  const dataLines: string[] = [];

  for (const line of lines) {
    if (!line) {
      continue;
    }
    if (line.startsWith("event:")) {
      const value = line.slice(6).trim();
      if (
        value === "progress" ||
        value === "heartbeat" ||
        value === "complete" ||
        value === "error"
      ) {
        eventName = value;
      }
      continue;
    }
    if (line.startsWith("id:")) {
      const value = Number(line.slice(3).trim());
      id = Number.isFinite(value) ? value : null;
      continue;
    }
    if (line.startsWith("data:")) {
      dataLines.push(line.slice(5).trimStart());
    }
  }

  if (dataLines.length === 0) {
    return null;
  }

  const parsed = JSON.parse(dataLines.join("\n")) as TrainingProgressPayload;
  return { event: eventName, payload: parsed, id };
}

export async function streamTrainingProgress(options: {
  signal: AbortSignal;
  lastEventId?: number | null;
  onOpen?: () => void;
  onEvent: (event: ParsedSseEvent) => void;
}): Promise<void> {
  const headers = new Headers();
  if (typeof options.lastEventId === "number") {
    headers.set("Last-Event-ID", String(options.lastEventId));
  }

  const response = await authFetch("/api/train/progress", {
    method: "GET",
    headers,
    signal: options.signal,
  });

  if (!response.ok) {
    throw new Error(await readError(response));
  }

  if (!response.body) {
    throw new Error("Progress stream unavailable");
  }

  options.onOpen?.();

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) {
        break;
      }

      buffer += decoder.decode(value, { stream: true });

      let separatorIndex = buffer.search(/\r?\n\r?\n/);
      while (separatorIndex >= 0) {
        const rawEvent = buffer.slice(0, separatorIndex);
        const separatorLength = buffer[separatorIndex] === "\r" ? 4 : 2;
        buffer = buffer.slice(separatorIndex + separatorLength);

        if (rawEvent.startsWith("retry:")) {
          separatorIndex = buffer.search(/\r?\n\r?\n/);
          continue;
        }

        try {
          const event = parseSseEvent(rawEvent);
          if (event) {
            options.onEvent(event);
          }
        } catch (error) {
          if (!isAbortError(error)) {
            throw error;
          }
        }

        separatorIndex = buffer.search(/\r?\n\r?\n/);
      }
    }
  } finally {
    // Release the stream lock now instead of leaking the reader until GC.
    try {
      await reader.cancel();
    } catch {
      // already closed
    }
  }
}

export { isAbortError };
