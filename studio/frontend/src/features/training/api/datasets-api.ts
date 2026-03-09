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

type UploadDatasetArgs = {
  filename: string;
  contentBase64: string;
};

export async function uploadTrainingDataset({
  filename,
  contentBase64,
}: UploadDatasetArgs): Promise<UploadDatasetResponse> {
  const res = await authFetch("/api/datasets/upload", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      filename,
      content_base64: contentBase64,
    }),
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
