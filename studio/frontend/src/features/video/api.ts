// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";
// Same plan shape as the images backend: both /download-plan routes share a response model.
import type { DiffusionDownloadPlan } from "@/features/images/api";
import { readFastApiError } from "@/lib/format-fastapi-error";

// One Advanced control's resolved value + provenance, for the "Auto: X" badges. Same shape the
// diffusion status uses. `value` is the engaged value (scheme/mode string, null when off, or a
// boolean); `source` is "auto" or "explicit"; `reason` is the tooltip why.
export interface VideoResolvedControl {
  value: string | boolean | null;
  source: "auto" | "explicit";
  reason: string;
}

// Per-family generation defaults + shape constraints, from status.defaults when loaded.
export interface VideoGenerationDefaults {
  steps: number;
  guidance: number;
  num_frames: number;
  fps: number;
  // Temporal lattice: valid frame counts are k * frame_step + 1.
  frame_step: number;
  // Width/height must be divisible by this.
  resolution_multiple: number;
  // (width, height) presets the UI offers, default first.
  resolution_presets: Array<[number, number]>;
}

export interface VideoStatus {
  loaded: boolean;
  repo_id: string | null;
  family: string | null;
  base_repo: string | null;
  device: string | null;
  dtype: string | null;
  // Resolved load kind: "gguf" | "single_file" | "pipeline". Null when not loaded.
  model_kind?: string | null;
  // Resolved offload policy: none | group | model | sequential.
  offload_policy?: string | null;
  vae_tiling: boolean;
  memory_mode?: string | null;
  speed_mode?: string | null;
  // Speed optimisations actually engaged.
  speed_optims: string[];
  attention_backend?: string | null;
  transformer_cache?: string | null;
  // Dense DiT precision actually engaged ("int8" | "fp8" | ...) or null for bf16.
  transformer_quant?: string | null;
  // Whether the loaded family produces a synchronized audio track.
  has_audio: boolean;
  // Per-family generation defaults + shape constraints; null when unloaded.
  defaults?: VideoGenerationDefaults | null;
  // Per-Advanced-control provenance, keyed by control name (memory_mode, speed_mode,
  // attention_backend, transformer_cache). The "Auto: X" badges read it. Null when
  // nothing is loaded or on a backend that doesn't record it.
  resolved?: Record<string, VideoResolvedControl> | null;
}

export interface VideoGenerateProgress {
  active: boolean;
  // "queued" | "denoise" | "export" | "completed" | "failed" | null. The terminal
  // phases carry the outcome of the background job POST /video/generate started.
  phase?: string | null;
  step: number;
  total: number;
  eta_seconds?: number | null;
  // Saved gallery record when phase is "completed".
  video?: GalleryVideo | null;
  // Client-safe failure detail when phase is "failed".
  error?: string | null;
}

export interface VideoLoadProgress {
  phase: "downloading" | "finalizing" | "ready" | "error" | null;
  downloaded_bytes: number;
  // null when the total isn't known yet.
  expected_bytes?: number | null;
  error?: string | null;
}

export interface VideoLoadRequest {
  model_path: string;
  // Required for the gguf / single_file kinds, omitted for a full pipeline (a
  // diffusers repo loaded via from_pretrained).
  gguf_filename?: string;
  // How to load the model (omit to auto-detect from gguf_filename): "gguf" (single-file
  // GGUF transformer), "single_file" (single-file safetensors transformer), or "pipeline"
  // (a full diffusers repo). Non-GGUF kinds are restricted to unsloth/* or family bases.
  model_kind?: "gguf" | "single_file" | "pipeline";
  base_repo?: string;
  family_override?: string;
  hf_token?: string;
  // Advanced (load-time) tuning. All optional; omit for the backend's auto defaults.
  memory_mode?: "auto" | "fast" | "balanced" | "low_vram";
  speed_mode?: "off" | "eager" | "default" | "max";
  attention_backend?:
    | "auto"
    | "native"
    | "sdpa"
    | "cudnn"
    | "flash"
    | "flash2"
    | "flash3"
    | "flash4"
    | "sage"
    | "xformers"
    | "aiter";
  transformer_cache?: "off" | "fbcache";
  transformer_cache_threshold?: number;
  // Dense DiT precision on full-pipeline loads (omit for the hardware-ladder auto;
  // "none" pins plain bf16). GGUF / single-file checkpoints carry their own precision.
  transformer_quant?: "none" | "fp8" | "int8" | "nvfp4" | "mxfp8";
}

export interface VideoGenerateRequest {
  prompt: string;
  negative_prompt?: string;
  // Width/height/num_frames/fps default per loaded family (the backend snaps them to the
  // family's required multiples/lattice), so they are optional here.
  width?: number;
  height?: number;
  num_frames?: number;
  fps?: number;
  steps?: number;
  guidance?: number;
  seed?: number;
}

// A persisted clip's full generation recipe (the JSON sidecar of the MP4).
export interface GalleryVideo {
  id: string;
  // Relative URL to fetch the MP4 bytes (auth-protected).
  url: string;
  prompt: string;
  negative_prompt?: string | null;
  width: number;
  height: number;
  num_frames: number;
  fps: number;
  duration_s: number;
  steps: number;
  guidance: number;
  seed: number;
  has_audio: boolean;
  model?: string | null;
  // Creation time (ISO 8601 timestamp).
  created_at: string;
}

// Acknowledgement that the generation job started; the saved record arrives via
// getVideoGenerateProgress when its phase reaches "completed".
export interface VideoGenerateResponse {
  status: "started";
  // Always null (kept for response-shape compatibility).
  video?: GalleryVideo | null;
}

async function parseJson<T>(response: Response): Promise<T> {
  if (!response.ok) {
    throw new Error(await readFastApiError(response));
  }
  return (await response.json()) as T;
}

export async function getVideoStatus(): Promise<VideoStatus> {
  return parseJson(await authFetch("/api/inference/video/status"));
}

export async function getVideoLoadProgress(): Promise<VideoLoadProgress> {
  return parseJson(await authFetch("/api/inference/video/load-progress"));
}

export async function getVideoGenerateProgress(): Promise<VideoGenerateProgress> {
  return parseJson(await authFetch("/api/inference/video/generate-progress"));
}

export async function loadVideoModel(body: VideoLoadRequest): Promise<VideoStatus> {
  return parseJson(
    await authFetch("/api/inference/video/load", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }),
  );
}

/** What to stage through the download manager before loading this pick. */
export async function getVideoDownloadPlan(
  body: VideoLoadRequest,
): Promise<DiffusionDownloadPlan> {
  return parseJson(
    await authFetch("/api/inference/video/download-plan", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }),
  );
}

/** Start a generation job. Returns as soon as the backend accepts it (the clip takes
 *  minutes, and secure mode's tunnel caps responses near 100s, so the POST cannot span
 *  the generation); poll getVideoGenerateProgress for completion. */
export async function generateVideo(
  body: VideoGenerateRequest,
): Promise<VideoGenerateResponse> {
  return parseJson(
    await authFetch("/api/inference/video/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }),
  );
}

/** Request a cancel of the in-flight generation. Best-effort: the backend stops at the
 *  next step boundary and raises the cancelled sentinel, which the caller maps to a 409. */
export async function cancelVideoGeneration(): Promise<{ cancelled: boolean }> {
  return parseJson(
    await authFetch("/api/inference/video/generate/cancel", { method: "POST" }),
  );
}

export async function unloadVideoModel(): Promise<VideoStatus> {
  return parseJson(await authFetch("/api/inference/video/unload", { method: "POST" }));
}

export interface VideoGalleryPage {
  videos: GalleryVideo[];
  has_more: boolean;
}

export async function getVideoGallery(offset = 0, limit = 50): Promise<VideoGalleryPage> {
  return parseJson(
    await authFetch(`/api/inference/video/gallery?offset=${offset}&limit=${limit}`),
  );
}

export async function deleteGalleryVideo(id: string): Promise<void> {
  const res = await authFetch(`/api/inference/video/gallery/${id}`, { method: "DELETE" });
  if (!res.ok) throw new Error(await readFastApiError(res));
}

export async function clearVideoGallery(): Promise<void> {
  const res = await authFetch("/api/inference/video/gallery", { method: "DELETE" });
  if (!res.ok) throw new Error(await readFastApiError(res));
}

/** A directly playable, range-capable URL for one gallery clip.
 *
 *  NOT the blob treatment the images gallery uses: an MP4 is tens to hundreds of MB, so
 *  `res.blob()` would download the whole clip before playback could start, defeat seeking, and
 *  pin those bytes in the webview for as long as the entry is cached -- one long clip can exceed
 *  the entire cache budget by itself. The backend's file route already streams and serves ranges;
 *  it just cannot be a plain <video src> because it is bearer-gated, so mint a short-lived signed
 *  link and let the element fetch only what it plays. */
export async function fetchGalleryVideoSignedUrl(id: string): Promise<string> {
  const res = await authFetch(
    `/api/inference/video/gallery/${encodeURIComponent(id)}/signed-url`,
  );
  if (!res.ok) throw new Error(await readFastApiError(res));
  const body = (await res.json()) as { url?: string };
  if (!body.url) throw new Error("The server returned no video link.");
  return body.url;
}

/** Server-side transcode for the Download menu (WebM / GIF). The backend 501s
 *  with a readable message when the codec for that format is unavailable. */
export async function fetchGalleryVideoExport(
  id: string,
  format: "webm" | "gif",
): Promise<Blob> {
  const res = await authFetch(
    `/api/inference/video/gallery/${id}/export?format=${format}`,
  );
  if (!res.ok) throw new Error(await readFastApiError(res));
  return res.blob();
}
