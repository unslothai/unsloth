// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";
import { readFastApiError } from "@/lib/format-fastapi-error";
import type {
  TrainingRunDeleteResponse,
  TrainingRunDetailResponse,
  TrainingRunListResponse,
  TrainingRunSummary,
} from "../types/history";

const readError = (r: Response): Promise<string> => readFastApiError(r);

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
  options?: { deleteArtifacts?: boolean; signal?: AbortSignal },
): Promise<TrainingRunDeleteResponse> {
  const deleteArtifacts = options?.deleteArtifacts ?? false;
  const params = new URLSearchParams({
    delete_artifacts: deleteArtifacts ? "true" : "false",
  });
  const response = await authFetch(
    `/api/train/runs/${encodeURIComponent(runId)}?${params.toString()}`,
    { method: "DELETE", signal: options?.signal },
  );
  return parseJson<TrainingRunDeleteResponse>(response);
}

export async function renameTrainingRun(
  runId: string,
  displayName: string | null,
  signal?: AbortSignal,
): Promise<TrainingRunSummary> {
  const response = await authFetch(
    `/api/train/runs/${encodeURIComponent(runId)}`,
    {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ display_name: displayName }),
      signal,
    },
  );
  return parseJson<TrainingRunSummary>(response);
}
