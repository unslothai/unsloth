// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { withBackgroundLoadNotice } from "@/lib/model-lifecycle-events";
import { authFetch } from "@/features/auth";
// Both /download-plan routes share a response model.
import type { DiffusionDownloadPlan } from "@/features/images/api";
import { apiUrl } from "@/lib/api-base";
import { readFastApiError } from "@/lib/format-fastapi-error";

// `requested` null = left to the backend; `source` is "auto" or "explicit"; `reason` is the tooltip.
export interface VideoResolvedControl {
  value: string | boolean | null;
  // Absent on backends predating the requested/actual split.
  requested?: string | boolean | null;
  source: "auto" | "explicit";
  // "applied" also covers nothing asked. Absent on older backends.
  status?: "applied" | "fell_back" | "unsupported";
  reason: string;
  // "prequant:<repo>/<file>" when a hosted checkpoint was seeded; absent on a runtime quantise.
  artifact?: string | null;
}

export interface VideoGenerationDefaults {
  steps: number;
  guidance: number;
  num_frames: number;
  fps: number;
  // Valid frame counts are k * frame_step + frame_offset.
  frame_step: number;
  frame_offset: number;
  duration_presets: number[];
  resolution_multiple: number;
  // Default first.
  resolution_presets: Array<[number, number]>;
  canvas_short_edge?: number | null;
  canvas_max_pixels?: number | null;
  flow_shift?: number | null;
  audio_flow_shift?: number | null;
  supports_audio_flow_shift?: boolean;
}

export interface VideoStatus {
  loaded: boolean;
  repo_id: string | null;
  display_repo_id?: string | null;
  family: string | null;
  supported_families?: string[];
  modular_families?: string[];
  base_repo: string | null;
  device: string | null;
  dtype: string | null;
  model_kind?: string | null;
  engine?: "diffusers" | "sd_cpp" | null;
  // Newer backends report this separately from the compute dtype.
  gguf_variant?: string | null;
  offload_policy?: string | null;
  vae_tiling: boolean;
  memory_mode?: string | null;
  speed_mode?: string | null;
  speed_optims: string[];
  attention_backend?: string | null;
  transformer_cache?: string | null;
  transformer_cache_stats?: {
    mode?: string;
    every?: number;
    planned_skips?: number;
    stats?: { calls?: number; computed?: number; skipped?: number };
  } | null;
  // null for bf16.
  transformer_quant?: string | null;
  transformer_quant_backend?: string | null;
  transformer_quant_backend_reason?: string | null;
  // null for dense bf16.
  text_encoder_quant?: string | null;
  has_audio: boolean;
  supports_cfg: boolean;
  supports_keyframes?: boolean;
  supports_references?: boolean;
  h3_task?: string | null;
  defaults?: VideoGenerationDefaults | null;
  // Read by the "Auto: X" badges. Keys: memory_mode, speed_mode, attention_backend, transformer_cache.
  resolved?: Record<string, VideoResolvedControl> | null;
}

export interface VideoGenerateProgress {
  active: boolean;
  // Terminal phases carry the background job's outcome.
  phase?: string | null;
  step: number;
  total: number;
  eta_seconds?: number | null;
  // Small JPEG data URL of the first frame, plus a counter that moves with each one.
  preview?: string | null;
  preview_seq?: number;
  video?: GalleryVideo | null;
  error?: string | null;
}

export interface VideoLoadProgress {
  phase: "downloading" | "finalizing" | "ready" | "error" | null;
  downloaded_bytes: number;
  expected_bytes?: number | null;
  error?: string | null;
}

export interface VideoLoadRequest {
  model_path: string;
  display_repo_id?: string;
  // Required for gguf / single_file; omitted for a from_pretrained pipeline.
  gguf_filename?: string;
  // Omit to auto-detect from gguf_filename. Non-GGUF kinds are restricted to unsloth/* or family bases.
  model_kind?: "gguf" | "single_file" | "pipeline";
  base_repo?: string;
  family_override?: string;
  hf_token?: string;
  // Advanced load-time tuning; omit for the backend's auto defaults.
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
  transformer_cache?: "off" | "fbcache" | "static";
  transformer_cache_threshold?: number;
  // Omit for the hardware ladder; "none" pins bf16. GGUF / single-file carry their own.
  transformer_quant?: "none" | "fp8" | "int8" | "nvfp4" | "mxfp8";
  h3_task?: "fl2va" | "ref2va";
  // Omit for automatic. Neither engine shards, so several cards resolve to the one with most free VRAM.
  gpu_ids?: number[];
  // Omit to keep dense bf16. Refused with a 409 when the host cannot run it.
  text_encoder_quant?: "fp8" | "fp8_dynamic" | "int8" | "nvfp4";
}

export interface VideoReferenceVideo {
  // Base64/data-URL video file, 2 to 15 seconds.
  video: string;
  // Omitted takes the soundtrack embedded in the file.
  audio?: string;
  // Both endpoints are required together; duration must be 2 to 15s.
  trim_start_seconds?: number;
  trim_end_seconds?: number;
}

export interface VideoGenerateRequest {
  prompt: string;
  // Omitted = the server default (on).
  live_preview?: boolean;
  negative_prompt?: string;
  // Optional; defaults per family. When sent they must be a resolution preset and num_frames on the
  // k*frame_step+1 lattice, or the backend answers 422.
  width?: number;
  height?: number;
  num_frames?: number;
  fps?: number;
  steps?: number;
  guidance?: number;
  seed?: number;
  // Omit both dimensions to match the source aspect.
  first_frame?: string;
  last_frame?: string;
  // Grouped in the model's image, video, then audio order.
  reference_images?: string[];
  reference_videos?: VideoReferenceVideo[];
  reference_audios?: string[];
  // "max" uses Diffusers' 2048px short-edge policy; "match" uses the clip area.
  reference_image_size?: "match" | "max";
  // Video schedule shift, and the audio one (Diffusers engine only).
  flow_shift?: number;
  audio_flow_shift?: number;
}

export interface GalleryVideo {
  id: string;
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
  // Absent on older clips.
  conditioning?: string | null;
  flow_shift?: number | null;
  audio_flow_shift?: number | null;
  model?: string | null;
  // ENGAGED load-time values, so the recipe names its precision after unload. Absent on older sidecars.
  model_kind?: string | null;
  gguf_filename?: string | null;
  transformer_quant?: string | null;
  text_encoder_quant?: string | null;
  memory_mode?: string | null;
  offload_policy?: string | null;
  created_at: string;
  // Library state, not recipe; absent on older sidecars.
  pinned?: boolean;
  archived?: boolean;
  /** The server's unpinned sort key: the drag key, else the file mtime. */
  order_at?: number | null;
}

// The saved record arrives via getVideoGenerateProgress at phase "completed".
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

export async function getVideoStatus(
  signal?: AbortSignal,
): Promise<VideoStatus> {
  return parseJson(await authFetch("/api/inference/video/status", { signal }));
}

export async function getVideoLoadProgress(
  signal?: AbortSignal,
): Promise<VideoLoadProgress> {
  return parseJson(
    await authFetch("/api/inference/video/load-progress", { signal }),
  );
}

export async function getVideoGenerateProgress(): Promise<VideoGenerateProgress> {
  return parseJson(await authFetch("/api/inference/video/generate-progress"));
}

export async function loadVideoModel(body: VideoLoadRequest): Promise<VideoStatus> {
  // This POST only starts the load; load-progress settles the notice. See images.
  return withBackgroundLoadNotice(
    "video",
    body.model_path,
    async () =>
      parseJson<VideoStatus>(
        await authFetch("/api/inference/video/load", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        }),
      ),
    async (signal) => (await getVideoLoadProgress(signal)).phase,
  );
}

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

/** Returns once accepted: a clip takes minutes and the secure tunnel caps responses near 100s. */
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

/** Best-effort: stops at the next step boundary; the caller maps the sentinel to a 409. */
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

/** `archived`: false is the strip, true is the archive. */
export async function getVideoGallery(
  offset = 0,
  limit = 50,
  archived = false,
): Promise<VideoGalleryPage> {
  return parseJson(
    await authFetch(
      `/api/inference/video/gallery?offset=${offset}&limit=${limit}&archived=${archived}`,
    ),
  );
}

/** Pin/unpin or archive/restore one clip; omitted flags are left alone. Returns the new record. */
/** `afterId` null = front. The server also decides the pin. */
export async function moveGalleryVideo(id: string, afterId: string | null): Promise<GalleryVideo> {
  return parseJson(
    await authFetch(`/api/inference/video/gallery/${id}/move`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ after_id: afterId }),
    }),
  );
}

export async function addGalleryVideoToProject(
  id: string,
  projectId: string,
): Promise<{ path: string; already: boolean }> {
  return parseJson(
    await authFetch(`/api/inference/video/gallery/${id}/project`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ project_id: projectId }),
    }),
  );
}

export async function setGalleryVideoFlags(
  id: string,
  flags: { pinned?: boolean; archived?: boolean },
): Promise<GalleryVideo> {
  return parseJson(
    await authFetch(`/api/inference/video/gallery/${id}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(flags),
    }),
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

/** res.blob() would download a whole MP4 before playback and pin it; the file route streams
 * ranges but is bearer-gated, so mint a short-lived signed link. */
export async function fetchGalleryVideoSignedUrl(id: string): Promise<string> {
  const res = await authFetch(
    `/api/inference/video/gallery/${encodeURIComponent(id)}/signed-url`,
  );
  if (!res.ok) throw new Error(await readFastApiError(res));
  const body = (await res.json()) as { url?: string };
  if (!body.url) throw new Error("The server returned no video link.");
  // Absolute because consumers bypass authFetch and Tauri resolves relative paths to the webview.
  return apiUrl(body.url);
}

/** Bearer-gated, so hold the bytes in a revocable object URL. */
export async function fetchGalleryVideoThumbnail(
  id: string,
): Promise<{ url: string; bytes: number }> {
  const res = await authFetch(
    `/v1/videos/${encodeURIComponent(id)}/content?variant=thumbnail`,
  );
  if (!res.ok) throw new Error(await readFastApiError(res));
  const blob = await res.blob();
  // An empty 200 would cache a card that never renders and end every retry.
  if (blob.size === 0) throw new Error("The thumbnail response was empty.");
  return { url: URL.createObjectURL(blob), bytes: blob.size };
}

/** Server-side transcode for the Download menu (WebM / GIF). The backend 501s with a readable
 *  message when the codec is unavailable. */
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
