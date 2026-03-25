// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";
import type {
  TrainingRunDeleteResponse,
  TrainingRunDetailResponse,
  TrainingRunListResponse,
} from "../types/history";

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

export async function listTrainingRuns(
  limit = 50,
  offset = 0,
  signal?: AbortSignal,
): Promise<TrainingRunListResponse> {
  const response = await authFetch(
    `/api/train/runs?limit=${limit}&offset=${offset}`,
    { signal },
  );
  return parseJson<TrainingRunListResponse>(response);
}

export async function getTrainingRun(
  runId: string,
  signal?: AbortSignal,
): Promise<TrainingRunDetailResponse> {
  const response = await authFetch(
    `/api/train/runs/${encodeURIComponent(runId)}`,
    { signal },
  );
  return parseJson<TrainingRunDetailResponse>(response);
}

export async function deleteTrainingRun(
  runId: string,
  signal?: AbortSignal,
): Promise<TrainingRunDeleteResponse> {
  const response = await authFetch(
    `/api/train/runs/${encodeURIComponent(runId)}`,
    { method: "DELETE", signal },
  );
  return parseJson<TrainingRunDeleteResponse>(response);
}
