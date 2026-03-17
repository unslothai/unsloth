// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";
import type {
  TrainingStartRequest,
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

async function readError(response: Response): Promise<string> {
  try {
    const payload = (await response.json()) as { detail?: string; message?: string };
    return payload.detail || payload.message || `Request failed (${response.status})`;
  } catch {
    return `Request failed (${response.status})`;
  }
}

async function parseJson<T>(response: Response): Promise<T> {
  if (!response.ok) {
    throw new Error(await readError(response));
  }
  return (await response.json()) as T;
}

export async function startTraining(
  payload: TrainingStartRequest,
): Promise<TrainingStartResponse> {
  const response = await authFetch("/api/train/start", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  return parseJson<TrainingStartResponse>(response);
}

export async function stopTraining(save = true): Promise<TrainingStopResponse> {
  const response = await authFetch("/api/train/stop", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ save }),
  });
  return parseJson<TrainingStopResponse>(response);
}

export async function resetTraining(): Promise<void> {
  const response = await authFetch("/api/train/reset", { method: "POST" });
  if (!response.ok) {
    throw new Error(await readError(response));
  }
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
}

export { isAbortError };
