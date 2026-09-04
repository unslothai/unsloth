// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { withBackgroundLoadNotice } from "@/lib/model-lifecycle-events";
import { authFetch } from "@/features/auth";
// Same plan shape as the images backend: both /download-plan routes share a response model.
import type { DiffusionDownloadPlan } from "@/features/images/api";
import { apiUrl } from "@/lib/api-base";
import { readFastApiError } from "@/lib/format-fastapi-error";

// One Advanced control's resolved value plus provenance, same shape as the diffusion status.
// `value` is engaged, `requested` is what the caller asked for (null = left to the backend),
// `source` is "auto" or "explicit", `status` says whether the ask survived, `reason` is the
// tooltip.
export interface VideoResolvedControl {
  value: string | boolean | null;
  // Absent on backends predating the requested/actual split.
  requested?: string | boolean | null;
  source: "auto" | "explicit";
  // "applied" (honored, or nothing was asked) | "fell_back" | "unsupported". Absent on older backends.
  status?: "applied" | "fell_back" | "unsupported";
  reason: string;
}

// Per-family generation defaults + shape constraints, from status.defaults when loaded.
export interface VideoGenerationDefaults {
  steps: number;
  guidance: number;
  num_frames: number;
  fps: number;
  // Temporal lattice: valid frame counts are k * frame_step + frame_offset.
  frame_step: number;
  frame_offset: number;
  duration_presets: number[];
  // Width/height must be divisible by this.
  resolution_multiple: number;
  // (width, height) presets the UI offers, default first.
  resolution_presets: Array<[number, number]>;
  // Backend-owned keyframe canvas rule, or null when unsupported.
  canvas_short_edge?: number | null;
  canvas_max_pixels?: number | null;
  // Released schedule shifts, or null when unsupported.
  flow_shift?: number | null;
  audio_flow_shift?: number | null;
  // Whether the active engine can apply audio_flow_shift.
  supports_audio_flow_shift?: boolean;
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
  engine?: "diffusers" | "sd_cpp" | null;
  // Selected GGUF quant. Newer backends report this separately from the compute dtype.
  gguf_variant?: string | null;
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
  // Text-encoder quant actually engaged ("fp8" | "fp8_dynamic" | "int8" | "nvfp4") or null for dense bf16.
  text_encoder_quant?: string | null;
  // Whether the loaded family produces a synchronized audio track.
  has_audio: boolean;
  supports_cfg: boolean;
  // Conditioning supported by the loaded checkpoint.
  supports_keyframes?: boolean;
  supports_references?: boolean;
  // Resident MiniMax-H3 denoiser partition, if any.
  h3_task?: string | null;
  // Per-family generation defaults + shape constraints; null when unloaded.
  defaults?: VideoGenerationDefaults | null;
  // Per-control provenance keyed by control name, read by the "Auto: X" badges. Null when
  // nothing is loaded or the backend does not record it.
  // The names are memory_mode, speed_mode, attention_backend and transformer_cache.
  resolved?: Record<string, VideoResolvedControl> | null;
}

export interface VideoGenerateProgress {
  active: boolean;
  // "queued" | "denoise" | "decode" | "export" | "completed" | "failed" | null; the terminal
  // phases carry the background job's outcome.
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
  // Required for the gguf / single_file kinds, omitted for a full pipeline loaded via
  // from_pretrained.
  gguf_filename?: string;
  // How to load the model (omit to auto-detect from gguf_filename): "gguf", "single_file"
  // (safetensors transformer) or "pipeline". Non-GGUF kinds are restricted to unsloth/* or
  // family bases.
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
  // Dense DiT precision on full-pipeline loads (omit for the hardware ladder; "none" pins bf16).
  // GGUF / single-file checkpoints carry their own.
  transformer_quant?: "none" | "fp8" | "int8" | "nvfp4" | "mxfp8";
  // Pipeline denoiser partition. GGUF filenames already identify theirs.
  h3_task?: "fl2va" | "ref2va";
  // CUDA / ROCm physical indices this load may use; omit for automatic. Neither engine shards a
  // checkpoint, so several cards resolve to the one with the most free VRAM.
  gpu_ids?: number[];
  // Text-encoder precision (omit to keep the dense bf16 encoder). Refused with a 409 when the host cannot run it.
  text_encoder_quant?: "fp8" | "fp8_dynamic" | "int8" | "nvfp4";
}

/** One reference video, with the soundtrack MiniMax-H3 conditions on alongside it. */
export interface VideoReferenceVideo {
  // Base64/data-URL video file, 2 to 15 seconds.
  video: string;
  // Base64/data-URL soundtrack for THIS video; omitted takes the one embedded in the file.
  audio?: string;
  // Optional explicit interval. Both endpoints are required together; duration must be 2 to 15s.
  trim_start_seconds?: number;
  trim_end_seconds?: number;
}

export interface VideoGenerateRequest {
  prompt: string;
  negative_prompt?: string;
  // Width/height/num_frames/fps default per loaded family, so they are optional. When sent they
  // must match that family's rules -- a resolution preset, and num_frames on the
  // k*frame_step+1 lattice -- or the backend answers 422 with the supported shapes.
  // Width/height must be one of status.defaults.resolution_presets.
  width?: number;
  height?: number;
  num_frames?: number;
  fps?: number;
  steps?: number;
  guidance?: number;
  seed?: number;
  // MiniMax-H3 keyframes as data URLs. Omit both dimensions to match the source aspect.
  first_frame?: string;
  last_frame?: string;
  // Ref2VA references, grouped in the model's image, video, then audio order.
  reference_images?: string[];
  reference_videos?: VideoReferenceVideo[];
  reference_audios?: string[];
  // "max" uses Diffusers' 2048px short-edge policy; "match" uses the clip area.
  reference_image_size?: "match" | "max";
  // Sigma shift of the video schedule, and of the audio one (Diffusers engine only).
  flow_shift?: number;
  audio_flow_shift?: number;
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
  // MiniMax-H3 task name, absent on older clips.
  conditioning?: string | null;
  flow_shift?: number | null;
  audio_flow_shift?: number | null;
  model?: string | null;
  // The load-time BUILD, all ENGAGED values, so a clip's recipe still names the precision it ran
  // at once the model is unloaded. Absent on sidecars written before this existed.
  model_kind?: string | null;
  gguf_filename?: string | null;
  transformer_quant?: string | null;
  text_encoder_quant?: string | null;
  memory_mode?: string | null;
  offload_policy?: string | null;
  // Creation time (ISO 8601 timestamp).
  created_at: string;
  // Library state, not recipe: stored beside the clip, absent on sidecars written before this existed.
  pinned?: boolean;
  archived?: boolean;
}

// Acknowledgement that the job started; the saved record arrives via getVideoGenerateProgress at phase "completed".
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
  // Announced so the indicator shows the load while the toast does, and settled from
  // load-progress because this POST only starts it. See images.
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

/** Start a generation job. Returns as soon as the backend accepts it (a clip takes minutes and
 *  secure mode's tunnel caps responses near 100s); poll getVideoGenerateProgress. */
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

/** Request a cancel. Best-effort: the backend stops at the next step boundary and raises the
 *  cancelled sentinel, which the caller maps to a 409. */
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

/** `archived` picks WHICH shelf to page over: false is the strip, true is the archive. */
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

/** A directly playable, range-capable URL for one gallery clip. Not the images-gallery blob
 *  treatment: an MP4 is tens to hundreds of MB, so res.blob() would download the whole clip
 *  before playback, defeat seeking and pin those bytes. The backend's file route streams
 *  ranges but is bearer-gated, so mint a short-lived signed link. */
export async function fetchGalleryVideoSignedUrl(id: string): Promise<string> {
  const res = await authFetch(
    `/api/inference/video/gallery/${encodeURIComponent(id)}/signed-url`,
  );
  if (!res.ok) throw new Error(await readFastApiError(res));
  const body = (await res.json()) as { url?: string };
  if (!body.url) throw new Error("The server returned no video link.");
  // Absolute because consumers bypass authFetch, and a relative path under Tauri resolves
  // against the webview origin. No-op in the browser (empty apiBase).
  return apiUrl(body.url);
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
