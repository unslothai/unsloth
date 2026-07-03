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
  // Resolved load kind: "gguf" | "single_file" | "pipeline". Gates GGUF-only controls
  // (the dense transformer_quant fast path only engages on gguf). Null when not loaded.
  model_kind?: string | null;
  cpu_offload: boolean;
  // Image workflows the loaded family supports (drives tab gating): txt2img, img2img,
  // inpaint. Absent/empty when nothing is loaded or on the native sd.cpp engine.
  workflows?: string[];
  // Whether the loaded model + quantisation can apply LoRA adapters (drives the LoRA
  // picker's enabled state). False on unsupported families/quant.
  supports_lora?: boolean;
  // Whether the loaded model can apply a ControlNet (drives the ControlNet picker's enabled
  // state). Diffusers only, for families with a ControlNet pipeline; false otherwise.
  supports_controlnet?: boolean;
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
  // Optional now: required for the gguf / single_file kinds, omitted for a full
  // pipeline (a diffusers repo loaded via from_pretrained).
  gguf_filename?: string;
  // How to load the model (omit to auto-detect from gguf_filename): "gguf" (single-file
  // GGUF transformer), "single_file" (single-file safetensors transformer, e.g. fp8), or
  // "pipeline" (a full diffusers repo). Non-GGUF kinds are restricted to unsloth/* repos.
  model_kind?: "gguf" | "single_file" | "pipeline";
  base_repo?: string;
  family_override?: string;
  hf_token?: string;
  cpu_offload?: boolean;
  // Advanced (load-time) tuning. All optional; omit for the backend's auto defaults.
  speed_mode?: "off" | "eager" | "default" | "max";
  transformer_quant?: "auto" | "int8" | "fp8" | "nvfp4" | "mxfp8";
  attention_backend?:
    | "auto"
    | "native"
    | "cudnn"
    | "flash"
    | "flash2"
    | "flash3"
    | "flash4"
    | "sage"
    | "xformers"
    | "aiter";
  memory_mode?: "auto" | "fast" | "balanced" | "low_vram";
  transformer_cache?: "off" | "fbcache";
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
  // Image-conditioned workflows. init_image alone = img2img; init_image + mask_image =
  // inpaint. Base64 or data-URL. strength is the denoise amount (0 keeps source, 1 redraws).
  init_image?: string;
  mask_image?: string;
  strength?: number;
  // Upscale (hires fix): factor > 1 with an init_image enlarges the source and re-denoises
  // it at low strength. Requires init_image; ignored for txt2img/inpaint/edit.
  upscale?: number;
  // Additional reference images for the FLUX.2 reference workflow, combined with init_image.
  reference_images?: string[];
  // LoRA adapters to apply for this generation (by discovery id + weight, 0..2). Omitted
  // or empty applies none. Rejected (400) when the loaded model/quant can't apply LoRA.
  loras?: LoraSpecInput[];
  // ControlNet conditioning for this generation. Omitted applies none. Rejected (400) when
  // the loaded model/quant can't apply ControlNet.
  controlnet?: ControlNetSpecInput;
}

// One LoRA selection sent with a generation.
export interface LoraSpecInput {
  id: string;
  weight: number;
}

// A ControlNet selection sent with a generation.
export interface ControlNetSpecInput {
  id: string;
  // Base64/data-URL control image (a source image or an already-made control map).
  image: string;
  // "canny" preprocesses edges from a source image; any other type (passthrough, or a
  // union type like depth/pose) is an already-made map the backend maps to a control mode.
  control_type: string;
  strength: number;
  guidance_start?: number;
  guidance_end?: number;
}

// A discoverable ControlNet model (from GET /api/models/diffusion-controlnets).
export interface DiffusionControlNetInfo {
  id: string;
  display_name: string;
  source: "local" | "hub";
  families: string[];
  control_types: string[];
  is_union: boolean;
}

// A discoverable diffusion LoRA adapter (from GET /api/models/diffusion-loras).
export interface DiffusionLoraInfo {
  id: string;
  display_name: string;
  source: "local" | "hub";
  format: "safetensors" | "gguf";
  families: string[];
  size_bytes: number;
  weight_default: number;
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
  loras?: string[];
  controlnet?: string | null;
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

/** List diffusion LoRA adapters, optionally filtered to a model family. */
export async function listDiffusionLoras(family?: string): Promise<DiffusionLoraInfo[]> {
  const qs = family ? `?family=${encodeURIComponent(family)}` : "";
  const data = await parseJson<{ loras: DiffusionLoraInfo[] }>(
    await authFetch(`/api/models/diffusion-loras${qs}`),
  );
  return data.loras ?? [];
}

/** List diffusion ControlNet models, optionally filtered to a model family. */
export async function listDiffusionControlNets(
  family?: string,
): Promise<DiffusionControlNetInfo[]> {
  const qs = family ? `?family=${encodeURIComponent(family)}` : "";
  const data = await parseJson<{ controlnets: DiffusionControlNetInfo[] }>(
    await authFetch(`/api/models/diffusion-controlnets${qs}`),
  );
  return data.controlnets ?? [];
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

// ── Diffusion (SDXL) LoRA training ────────────────────────────────────────────
// Mirrors DiffusionTrainingStartRequest on the backend; only the paths are required.
export interface DiffusionTrainingStartRequest {
  base_model: string;
  data_dir: string;
  output_dir: string;
  instance_prompt?: string | null;
  resolution?: number;
  train_steps?: number;
  learning_rate?: number;
  train_batch_size?: number;
  gradient_accumulation_steps?: number;
  lora_rank?: number;
  lora_alpha?: number | null;
  lora_target_modules?: string[];
  max_grad_norm?: number;
  seed?: number;
  mixed_precision?: "bf16" | "fp16" | "no";
  gradient_checkpointing?: boolean;
  lr_scheduler?: string;
  // Forwarded to StableDiffusionXLPipeline.from_pretrained for a gated/private base repo.
  hf_token?: string | null;
}

// A snapshot of the current diffusion training job (GET /api/train/diffusion/status).
export interface DiffusionTrainingStatus {
  active: boolean;
  job_id: string | null;
  status: string;
  message: string;
  step: number;
  total_steps: number;
  loss: number | null;
  avg_loss: number | null;
  learning_rate: number | null;
  num_images: number | null;
  in_model_load: boolean;
  output_dir: string | null;
  lora_path: string | null;
  started_at: number | null;
  updated_at: number | null;
}

export async function startDiffusionTraining(
  body: DiffusionTrainingStartRequest,
): Promise<{ job_id: string; status: string }> {
  return parseJson(
    await authFetch("/api/train/diffusion/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }),
  );
}

export async function stopDiffusionTraining(): Promise<{ status: string }> {
  return parseJson(await authFetch("/api/train/diffusion/stop", { method: "POST" }));
}

export async function getDiffusionTrainingStatus(): Promise<DiffusionTrainingStatus> {
  return parseJson(await authFetch("/api/train/diffusion/status"));
}

// One image-dataset folder under the Studio datasets root (GET /api/train/diffusion/info).
export interface DiffusionDatasetSummary {
  name: string;
  path: string;
  image_count: number;
  caption_count: number;
}

// Where diffusion training reads/writes on this Studio, plus usable dataset folders.
export interface DiffusionTrainingInfo {
  datasets_root: string;
  outputs_root: string;
  datasets: DiffusionDatasetSummary[];
}

export async function getDiffusionTrainingInfo(): Promise<DiffusionTrainingInfo> {
  return parseJson(await authFetch("/api/train/diffusion/info"));
}

export interface DiffusionDatasetUploadResult extends DiffusionDatasetSummary {
  uploaded: number;
}

/** Upload images (+ optional caption .txt / metadata.jsonl) into a named dataset folder.
 * Repeat uploads into the same name accumulate; the returned name is a valid data_dir
 * for startDiffusionTraining. */
export async function uploadDiffusionDataset(
  name: string,
  files: File[],
): Promise<DiffusionDatasetUploadResult> {
  const form = new FormData();
  form.append("name", name);
  for (const f of files) form.append("files", f);
  return parseJson(
    await authFetch("/api/train/diffusion/dataset", { method: "POST", body: form }),
  );
}
