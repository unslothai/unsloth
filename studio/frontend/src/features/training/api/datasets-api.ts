// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type {
  CheckFormatResponse,
  LocalDatasetsResponse,
  UploadDatasetResponse,
} from "../types/datasets";
import { authFetch } from "@/features/auth";
import { readFastApiError } from "@/lib/format-fastapi-error";

type CheckDatasetFormatArgs = {
  datasetName: string;
  hfToken: string | null;
  subset?: string | null;
  split?: string | null;
  isVlm?: boolean;
};

export async function checkDatasetFormat({
  datasetName,
  hfToken,
  subset,
  split,
  isVlm,
}: CheckDatasetFormatArgs): Promise<CheckFormatResponse> {
  const res = await authFetch("/api/datasets/check-format", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      dataset_name: datasetName,
      hf_token: hfToken || undefined,
      subset: subset || undefined,
      split: split || "train",
      is_vlm: !!isVlm,
    }),
  });

  if (!res.ok) {
    throw new Error(await readFastApiError(res));
  }

  return res.json();
}

export async function uploadTrainingDataset(
  file: File,
): Promise<UploadDatasetResponse> {
  const form = new FormData();
  form.append("file", file);

  const res = await authFetch("/api/datasets/upload", {
    method: "POST",
    body: form,
  });

  if (!res.ok) {
    throw new Error(await readFastApiError(res, "Upload failed"));
  }

  return res.json();
}

// ── AI Assist ────────────────────────────────────────────────────────

type AiAssistMappingArgs = {
  columns: string[];
  samples: Record<string, unknown>[];
  datasetName?: string | null;
  hfToken?: string | null;
  modelName?: string | null;
  modelType?: "text" | "vision" | "audio" | "embeddings" | null;
};

export type AiAssistMappingResponse = {
  success: boolean;
  suggested_mapping?: Record<string, string> | null;
  warning?: string | null;
  // Conversion advisor fields
  system_prompt?: string | null;
  label_mapping?: Record<string, Record<string, string>> | null;
  dataset_type?: string | null;
  is_conversational?: boolean | null;
  user_notification?: string | null;
};

export async function aiAssistMapping({
  columns,
  samples,
  datasetName,
  hfToken,
  modelName,
  modelType,
}: AiAssistMappingArgs): Promise<AiAssistMappingResponse> {
  const res = await authFetch("/api/datasets/ai-assist-mapping", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      columns,
      samples: samples.slice(0, 5),
      dataset_name: datasetName || undefined,
      hf_token: hfToken || undefined,
      model_name: modelName || undefined,
      model_type: modelType || undefined,
    }),
  });

  if (!res.ok) {
    throw new Error(await readFastApiError(res, "AI assist failed"));
  }

  return res.json();
}

export async function listLocalDatasets(): Promise<LocalDatasetsResponse> {
  const res = await authFetch("/api/datasets/local");
  if (!res.ok) {
    throw new Error(await readFastApiError(res));
  }
  return res.json();
}

export interface CachedDatasetRepo {
  repo_id: string;
  size_bytes: number;
  cache_path?: string;
  partial?: boolean;
}

export async function listCachedDatasets(): Promise<CachedDatasetRepo[]> {
  const res = await authFetch("/api/datasets/cached");
  if (!res.ok) {
    if (res.status === 404) {
      console.warn(
        "GET /api/datasets/cached returned 404 — backend may need a restart to expose the cached-datasets endpoint.",
      );
    } else {
      console.warn(
        "GET /api/datasets/cached failed with status",
        res.status,
      );
    }
    return [];
  }
  const data = (await res.json().catch(() => null)) as
    | { cached?: CachedDatasetRepo[] }
    | null;
  return data?.cached ?? [];
}

export async function deleteCachedDataset(repoId: string): Promise<void> {
  const res = await authFetch("/api/datasets/cached", {
    method: "DELETE",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ repo_id: repoId }),
  });
  if (!res.ok) {
    const body = await res.json().catch(() => null);
    throw new Error(body?.detail || `Failed to delete dataset (${res.status})`);
  }
}
