// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type {
  CheckFormatResponse,
  LocalDatasetsResponse,
  UploadDatasetResponse,
} from "../types/datasets";
import { authFetch } from "@/features/auth";

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
    const body = await res.json().catch(() => null);
    throw new Error(body?.detail || `Request failed (${res.status})`);
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
    const body = await res.json().catch(() => null);
    throw new Error(body?.detail || `Upload failed (${res.status})`);
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
    const body = await res.json().catch(() => null);
    throw new Error(body?.detail || `AI assist failed (${res.status})`);
  }

  return res.json();
}

export async function listLocalDatasets(): Promise<LocalDatasetsResponse> {
  const res = await authFetch("/api/datasets/local");
  if (!res.ok) {
    const body = await res.json().catch(() => null);
    throw new Error(body?.detail || `Request failed (${res.status})`);
  }
  return res.json();
}
