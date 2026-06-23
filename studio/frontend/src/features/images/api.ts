// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";
import { readFastApiError } from "@/lib/format-fastapi-error";

export interface DiffusionStatus {
  loaded: boolean;
  loading: boolean;
  repo_id: string | null;
  family: string | null;
  base_repo: string | null;
  device: string | null;
  dtype: string | null;
  cpu_offload: boolean;
}

export interface DiffusionLoadProgress {
  phase: "downloading" | "finalizing" | "ready" | "error" | null;
  bytes_downloaded: number;
  bytes_total: number;
  fraction: number;
  error: string | null;
}

export interface DiffusionLoadRequest {
  model_path: string;
  gguf_filename?: string;
  base_repo?: string;
  family_override?: string;
  hf_token?: string;
  cpu_offload?: boolean;
}

export interface DiffusionGenerateRequest {
  prompt: string;
  negative_prompt?: string;
  width?: number;
  height?: number;
  steps?: number;
  guidance?: number;
  seed?: number;
}

export interface DiffusionGenerateResponse {
  image_b64: string;
  mime: string;
  seed: number;
}

async function parseJson<T>(response: Response): Promise<T> {
  if (!response.ok) {
    throw new Error(await readFastApiError(response));
  }
  return (await response.json()) as T;
}

export async function getDiffusionStatus(): Promise<DiffusionStatus> {
  return parseJson(await authFetch("/api/inference/images/status"));
}

export async function getDiffusionLoadProgress(): Promise<DiffusionLoadProgress> {
  return parseJson(await authFetch("/api/inference/images/load-progress"));
}

export async function loadDiffusionModel(body: DiffusionLoadRequest): Promise<DiffusionStatus> {
  return parseJson(
    await authFetch("/api/inference/images/load", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }),
  );
}

export async function generateDiffusionImage(
  body: DiffusionGenerateRequest,
): Promise<DiffusionGenerateResponse> {
  return parseJson(
    await authFetch("/api/inference/images/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }),
  );
}

export async function unloadDiffusionModel(): Promise<DiffusionStatus> {
  return parseJson(await authFetch("/api/inference/images/unload", { method: "POST" }));
}
