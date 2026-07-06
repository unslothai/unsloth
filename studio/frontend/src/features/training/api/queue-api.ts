// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";
import { readFastApiError } from "@/lib/format-fastapi-error";
import type { TrainingStartRequest } from "../types/api";
import type { TrainingQueueItem, TrainingQueueState } from "../types/queue";

async function parseJson<T>(response: Response): Promise<T> {
  if (!response.ok) {
    throw new Error(await readFastApiError(response));
  }
  return (await response.json()) as T;
}

export async function getQueueState(
  signal?: AbortSignal,
): Promise<TrainingQueueState> {
  const response = await authFetch("/api/train/queue", { signal });
  return parseJson<TrainingQueueState>(response);
}

export async function enqueueTrainingJob(
  payload: TrainingStartRequest,
): Promise<TrainingQueueItem> {
  const response = await authFetch("/api/train/queue/items", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  return parseJson<TrainingQueueItem>(response);
}

export async function removeQueueItem(itemId: string): Promise<void> {
  const response = await authFetch(
    `/api/train/queue/items/${encodeURIComponent(itemId)}`,
    { method: "DELETE" },
  );
  if (!response.ok) {
    throw new Error(await readFastApiError(response));
  }
}

export async function moveQueueItem(
  itemId: string,
  direction: "up" | "down",
): Promise<TrainingQueueState> {
  const response = await authFetch(
    `/api/train/queue/items/${encodeURIComponent(itemId)}/move`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ direction }),
    },
  );
  return parseJson<TrainingQueueState>(response);
}

export async function pauseQueue(): Promise<TrainingQueueState> {
  const response = await authFetch("/api/train/queue/pause", { method: "POST" });
  return parseJson<TrainingQueueState>(response);
}

export async function resumeQueue(): Promise<TrainingQueueState> {
  const response = await authFetch("/api/train/queue/resume", { method: "POST" });
  return parseJson<TrainingQueueState>(response);
}
