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

export async function streamTrainingProgress(options: {
  signal: AbortSignal;
  lastEventId?: number | null;
  onOpen?: () => void;
  onEvent: (event: ParsedSseEvent) => void;
}): Promise<void> {
  // Build WebSocket URL from current page location
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  const params = new URLSearchParams();

  // Pass auth token as query param (WebSocket can't use Authorization header)
  const token = localStorage.getItem("unsloth_auth_token");
  if (token) params.set("token", token);
  if (typeof options.lastEventId === "number") {
    params.set("last_event_id", String(options.lastEventId));
  }

  const url = `${protocol}//${window.location.host}/api/train/progress/ws?${params}`;

  return new Promise<void>((resolve, reject) => {
    const ws = new WebSocket(url);

    // Wire up AbortSignal to close the socket
    const onAbort = () => ws.close();
    options.signal.addEventListener("abort", onAbort);

    ws.onopen = () => {
      options.onOpen?.();
    };

    ws.onmessage = (messageEvent) => {
      try {
        const msg = JSON.parse(messageEvent.data) as {
          event: ProgressEventName;
          id: number | null;
          data: TrainingProgressPayload;
        };
        options.onEvent({
          event: msg.event,
          id: msg.id,
          payload: msg.data,
        });
      } catch {
        // Ignore parse errors for malformed messages
      }
    };

    ws.onclose = () => {
      options.signal.removeEventListener("abort", onAbort);
      resolve();
    };

    ws.onerror = () => {
      options.signal.removeEventListener("abort", onAbort);
      reject(new Error("WebSocket connection failed"));
    };
  });
}

export { isAbortError };
