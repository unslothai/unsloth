// SPDX-License-Identifier: AGPL-3.0-only - See /studio/LICENSE.AGPL-3.0
// Copyright © 2025 Unsloth AI

import { authFetch } from "@/features/auth";

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

export interface CheckpointInfo {
  display_name: string;
  path: string;
  loss?: number | null;
}

export interface ModelCheckpoints {
  name: string;
  checkpoints: CheckpointInfo[];
  base_model?: string | null;
  peft_type?: string | null;
  lora_rank?: number | null;
}

export interface CheckpointListResponse {
  outputs_dir: string;
  models: ModelCheckpoints[];
}

export interface ExportOperationResponse {
  success: boolean;
  message: string;
  details?: Record<string, unknown> | null;
}

export async function fetchCheckpoints(): Promise<CheckpointListResponse> {
  const response = await authFetch("/api/models/checkpoints");
  return parseJson<CheckpointListResponse>(response);
}

export async function loadCheckpoint(params: {
  checkpoint_path: string;
  max_seq_length?: number;
  load_in_4bit?: boolean;
  /** Allow loading models with custom code. Only enable for checkpoints you trust. */
  trust_remote_code?: boolean;
}): Promise<ExportOperationResponse> {
  const response = await authFetch("/api/export/load-checkpoint", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(params),
  });
  return parseJson<ExportOperationResponse>(response);
}

export async function exportMerged(params: {
  save_directory: string;
  format_type?: string;
  push_to_hub?: boolean;
  repo_id?: string | null;
  hf_token?: string | null;
  private?: boolean;
}): Promise<ExportOperationResponse> {
  const response = await authFetch("/api/export/export/merged", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(params),
  });
  return parseJson<ExportOperationResponse>(response);
}

export async function exportBase(params: {
  save_directory: string;
  push_to_hub?: boolean;
  repo_id?: string | null;
  hf_token?: string | null;
  private?: boolean;
  base_model_id?: string | null;
}): Promise<ExportOperationResponse> {
  const response = await authFetch("/api/export/export/base", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(params),
  });
  return parseJson<ExportOperationResponse>(response);
}

export async function exportGGUF(params: {
  save_directory: string;
  quantization_method: string;
  push_to_hub?: boolean;
  repo_id?: string | null;
  hf_token?: string | null;
}): Promise<ExportOperationResponse> {
  const response = await authFetch("/api/export/export/gguf", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(params),
  });
  return parseJson<ExportOperationResponse>(response);
}

export async function exportLoRA(params: {
  save_directory: string;
  push_to_hub?: boolean;
  repo_id?: string | null;
  hf_token?: string | null;
  private?: boolean;
}): Promise<ExportOperationResponse> {
  const response = await authFetch("/api/export/export/lora", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(params),
  });
  return parseJson<ExportOperationResponse>(response);
}

export async function cleanupExport(): Promise<ExportOperationResponse> {
  const response = await authFetch("/api/export/cleanup", { method: "POST" });
  return parseJson<ExportOperationResponse>(response);
}
