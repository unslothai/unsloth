// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";
import { readFastApiError } from "@/lib/format-fastapi-error";

export interface DiffusionStatus {
  loaded: boolean;
  repo_id: string | null;
  family: string | null;
  base_repo: string | null;
  device: string | null;
  dtype: string | null;
  cpu_offload: boolean;
}

export interface DiffusionGenerateProgress {
  active: boolean;
  step: number;
  total_steps: number;
  fraction: number;
  eta_seconds: number | null;
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
  gguf_filename: string;
  base_repo?: string;
  family_override?: string;
  hf_token?: string;
}

export interface DiffusionGenerateRequest {
  prompt: string;
  negative_prompt?: string;
  width?: number;
  height?: number;
  steps?: number;
  guidance?: number;
  seed?: number;
  batch_size?: number;
}

// A persisted image's full generation recipe (also embedded in the PNG).
export interface GalleryImage {
  id: string;
  url: string;
  prompt: string;
  negative_prompt: string | null;
  width: number;
  height: number;
  steps: number;
  guidance: number;
  seed: number;
  batch_index: number;
  batch_size: number;
  model: string | null;
  created_at: number;
}

export interface DiffusionGenerateResponse {
  images: GalleryImage[];
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

export async function getGenerateProgress(): Promise<DiffusionGenerateProgress> {
  return parseJson(await authFetch("/api/inference/images/generate-progress"));
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

export interface GalleryPage {
  images: GalleryImage[];
  has_more: boolean;
}

export async function getGallery(offset = 0, limit = 50): Promise<GalleryPage> {
  return parseJson(
    await authFetch(`/api/inference/images/gallery?offset=${offset}&limit=${limit}`),
  );
}

export async function deleteGalleryImage(id: string): Promise<void> {
  const res = await authFetch(`/api/inference/images/gallery/${id}`, { method: "DELETE" });
  if (!res.ok) throw new Error(await readFastApiError(res));
}

export async function clearGallery(): Promise<void> {
  const res = await authFetch("/api/inference/images/gallery", { method: "DELETE" });
  if (!res.ok) throw new Error(await readFastApiError(res));
}

/** Fetch a gallery PNG (auth-protected, so it can't be a plain <img src>) and
 *  wrap it in an object URL. Callers must revoke the URL when done. */
export async function fetchGalleryObjectUrl(url: string): Promise<string> {
  const res = await authFetch(url);
  if (!res.ok) throw new Error(await readFastApiError(res));
  return URL.createObjectURL(await res.blob());
}
