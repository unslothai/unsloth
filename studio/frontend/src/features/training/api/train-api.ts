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

export { isAbortError };
