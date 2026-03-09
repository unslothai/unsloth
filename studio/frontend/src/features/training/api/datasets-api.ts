// SPDX-License-Identifier: AGPL-3.0-only - See /studio/LICENSE.AGPL-3.0
// Copyright © 2025 Unsloth AI

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

export async function listLocalDatasets(): Promise<LocalDatasetsResponse> {
  const res = await authFetch("/api/datasets/local");
  if (!res.ok) {
    const body = await res.json().catch(() => null);
    throw new Error(body?.detail || `Request failed (${res.status})`);
  }
  return res.json();
}
