// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { withBackgroundLoadNotice } from "@/lib/model-lifecycle-events";
import { authFetch } from "@/features/auth";
import { readFastApiError } from "@/lib/format-fastapi-error";
import {
  isMemoryEstimateRefusal,
  MEMORY_REFUSAL_HEADER,
  MemoryEstimateRefusalError,
} from "./lib/memory-refusal";

export interface DiffusionResolvedControl {
  value: string | boolean | null;
  requested?: string | boolean | null;
  source: "auto" | "explicit";
  status?: "applied" | "fell_back" | "unsupported";
  reason: string;
  artifact?: string | null;
  replaced?: string | null;
}

export interface DiffusionStatus {
  loaded: boolean;
  repo_id: string | null;
  display_repo_id?: string | null;
  family: string | null;
  supported_families?: string[];
  base_repo: string | null;
  device: string | null;
  dtype: string | null;
  model_kind?: string | null;
  gguf_filename?: string | null;
  gguf_variant?: string | null;
  cpu_offload: boolean;
  // ENGAGED runtime build, not the load request. null = the GGUF ran as-is.
  transformer_quant?: string | null;
  transformer_quant_backend?: string | null;
  transformer_quant_backend_reason?: string | null;
  text_encoder_quant?: string | null;
  memory_mode?: string | null;
  offload_policy?: string | null;
  speed_mode?: string | null;
  speed_optims?: string[];
  attention_backend?: string | null;
  transformer_cache?: string | null;
  vae_tiling?: boolean;
  workflows?: string[];
  // Absent on older backends: callers keep the historical limits (4 images, RGB, 16 px, 2048).
  conditioning?: DiffusionConditioning | null;
  supports_lora?: boolean;
  supports_controlnet?: boolean;
  supports_negative_prompt?: boolean;
  // Per-Advanced-control provenance, keyed by control name. Present only when a model is loaded on a
  // backend that records it; absent on older backends.
  resolved?: Record<string, DiffusionResolvedControl> | null;
}

export interface DiffusionConditioning {
  // Total input images per call, including the source.
  max_condition_images: number;
  alpha: boolean;
  dimension_multiple: number;
  max_output_side: number;
  max_output_pixels: number;
  reference_resolutions: number[];
  unified_edit: boolean;
  localized_edit_modes: LocalizedEditMode[];
  notes?: string[];
}

export type LocalizedEditMode = "annotate" | "paint" | "mask";

export interface DiffusionGenerateProgress {
  active: boolean;
  step: number;
  total_steps: number;
  fraction: number;
  eta_seconds: number | null;
  // Absent (sd.cpp engine) means "denoise".
  phase?: "encode" | "denoise" | "decode" | null;
  preview?: string | null;
  preview_seq?: number;
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
  display_repo_id?: string;
  gguf_filename?: string;
  // Non-GGUF kinds are restricted to unsloth/* repos.
  model_kind?: "gguf" | "single_file" | "pipeline";
  base_repo?: string;
  family_override?: string;
  hf_token?: string;
  cpu_offload?: boolean;
  speed_mode?: "off" | "eager" | "default" | "max";
  transformer_quant?: "auto" | "none" | "off" | "int8" | "fp8" | "nvfp4" | "mxfp8";
  // "none"/"off" pin the bf16 encoder; omitting it lets the family choose. 409 if the host cannot run it.
  text_encoder_quant?:
    | "auto"
    | "none"
    | "off"
    | "fp8"
    | "fp8_dynamic"
    | "int8"
    | "nvfp4";
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
  // Neither engine shards, so several cards resolve to the one with the most free VRAM.
  gpu_ids?: number[];
  transformer_cache?: "off" | "fbcache" | "static";
  // torchao int8/fp8 builds can only take LoRAs before quantisation, so they must be baked at load.
  loras?: LoraSpecInput[];
}

export interface DiffusionGenerateRequest {
  prompt: string;
  live_preview?: boolean;
  negative_prompt?: string;
  width?: number;
  height?: number;
  steps?: number;
  guidance?: number;
  seed?: number;
  batch_size?: number;
  init_image?: string;
  mask_image?: string;
  strength?: number;
  upscale?: number;
  allow_oversized?: boolean;
  reference_images?: string[];
  workflow?: "edit" | "reference" | "outpaint";
  reference_resolution?: number;
  localized_edit?: { mode: LocalizedEditMode; image: string };
  loras?: LoraSpecInput[];
  controlnet?: ControlNetSpecInput;
}

export interface LoraSpecInput {
  id: string;
  weight: number;
}

export interface ControlNetSpecInput {
  id: string;
  image: string;
  control_type: string;
  strength: number;
  guidance_start?: number;
  guidance_end?: number;
}

export interface DiffusionControlNetInfo {
  id: string;
  display_name: string;
  source: "local" | "hub";
  families: string[];
  control_types: string[];
  is_union: boolean;
}

export interface DiffusionLoraInfo {
  id: string;
  display_name: string;
  source: "local" | "hub";
  format: "safetensors" | "gguf";
  families: string[];
  size_bytes: number;
  weight_default: number;
}

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
  batch_seed?: number | null;
  batch_index: number;
  batch_size: number;
  model: string | null;
  model_kind?: string | null;
  gguf_filename?: string | null;
  transformer_quant?: string | null;
  text_encoder_quant?: string | null;
  memory_mode?: string | null;
  offload_policy?: string | null;
  speed_mode?: string | null;
  attention_backend?: string | null;
  transformer_cache?: string | null;
  cpu_offload?: boolean | null;
  schema_version?: number | null;
  baked_loras?: string[];
  loras?: string[];
  controlnet?: string | null;
  // Source/mask/reference images are not persisted, only these settings.
  workflow?: string | null;
  strength?: number | null;
  upscale?: number | null;
  controlnet_guidance?: string | null;
  reference_image_count?: number | null;
  reference_resolution?: number | null;
  localized_edit?: LocalizedEditMode | null;
  created_at: number;
  pinned?: boolean;
  archived?: boolean;
  order_at?: number | null;
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

export async function getDiffusionStatus(
  signal?: AbortSignal,
): Promise<DiffusionStatus> {
  return parseJson(await authFetch("/api/inference/images/status", { signal }));
}

export interface DiffusionInferenceInfo {
  family: string;
  transformer_bf16_gb: number;
  text_encoders_bf16_gb: number;
  vae_bf16_gb: number;
  estimated_resident_gb: Record<string, number>;
}

export interface DiffusionInferenceInfoResponse {
  families: DiffusionInferenceInfo[];
}

export async function getDiffusionInferenceInfo(): Promise<DiffusionInferenceInfoResponse> {
  return parseJson(await authFetch("/api/inference/images/info"));
}

export async function getDiffusionLoadProgress(
  signal?: AbortSignal,
): Promise<DiffusionLoadProgress> {
  return parseJson(
    await authFetch("/api/inference/images/load-progress", { signal }),
  );
}

export async function getGenerateProgress(): Promise<DiffusionGenerateProgress> {
  return parseJson(await authFetch("/api/inference/images/generate-progress"));
}

export async function loadDiffusionModel(body: DiffusionLoadRequest): Promise<DiffusionStatus> {
  // This POST only starts the load, so the notice settles from load-progress, not the response.
  return withBackgroundLoadNotice(
    "image",
    body.model_path,
    async () =>
      parseJson<DiffusionStatus>(
        await authFetch("/api/inference/images/load", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        }),
      ),
    async (signal) => (await getDiffusionLoadProgress(signal)).phase,
  );
}

export interface DiffusionDownloadPlan {
  plan_failed?: boolean;
  entries: {
    repo_id: string;
    files: string[];
    bytes: number;
    gguf_filename: string | null;
    /** Only the planner knows: a gated pick is staged from an ungated mirror under another repo id. */
    checkpoint?: boolean;
  }[];
  total_bytes: number;
  required_bytes?: number;
  checkpoint_bytes?: number;
  /** Why this pick cannot load as selected, or null when nothing is known to be wrong. */
  incompatible_reason?: string | null;
}

export async function getDiffusionDownloadPlan(
  body: DiffusionLoadRequest,
): Promise<DiffusionDownloadPlan> {
  return parseJson(
    await authFetch("/api/inference/images/download-plan", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }),
  );
}

/** The POST response was lost in transit rather than refused, so the generation is still running. */
export class GenerateResponseLostError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "GenerateResponseLostError";
  }
}

// 524 is Cloudflare's ~100s cap, which a slow run routinely exceeds.
const RESPONSE_LOST_STATUSES = new Set([408, 502, 503, 504, 522, 524]);

/** A run past the proxy window throws GenerateResponseLostError; settle it, do not retry. */
export async function generateDiffusionImage(
  body: DiffusionGenerateRequest,
): Promise<DiffusionGenerateResponse> {
  let response: Response;
  try {
    response = await authFetch("/api/inference/images/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
  } catch (err) {
    throw new GenerateResponseLostError(
      err instanceof Error ? err.message : "Lost connection during image generation",
    );
  }
  if (!response.ok && RESPONSE_LOST_STATUSES.has(response.status)) {
    const detail = await readFastApiError(response);
    // A proxy answers with HTML or nothing; an app error is JSON and means the run never started.
    if (
      response.status === 503 &&
      (response.headers.get("content-type") || "").toLowerCase().includes("application/json")
    ) {
      throw new Error(detail);
    }
    throw new GenerateResponseLostError(detail);
  }
  if (
    !response.ok &&
    isMemoryEstimateRefusal(response.status, response.headers.get(MEMORY_REFUSAL_HEADER))
  ) {
    throw new MemoryEstimateRefusalError(await readFastApiError(response));
  }
  return parseJson(response);
}

/** Best-effort: stops at the next step boundary and the generate POST unwinds as a 409. */
export async function cancelDiffusionGeneration(
  signal?: AbortSignal,
): Promise<{ cancelled: boolean }> {
  return parseJson(
    // No retry: the endpoint targets whatever generation is active NOW, which may be a newer run.
    await authFetch(
      "/api/inference/images/generate/cancel",
      { method: "POST", signal },
      { retryNetworkErrors: false },
    ),
  );
}

export async function unloadDiffusionModel(): Promise<DiffusionStatus> {
  return parseJson(await authFetch("/api/inference/images/unload", { method: "POST" }));
}

export async function listDiffusionLoras(family?: string): Promise<DiffusionLoraInfo[]> {
  const qs = family ? `?family=${encodeURIComponent(family)}` : "";
  const data = await parseJson<{ loras: DiffusionLoraInfo[] }>(
    await authFetch(`/api/models/diffusion-loras${qs}`),
  );
  return data.loras ?? [];
}

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

export async function getGallery(offset = 0, limit = 50, archived = false): Promise<GalleryPage> {
  return parseJson(
    await authFetch(
      `/api/inference/images/gallery?offset=${offset}&limit=${limit}&archived=${archived}`,
    ),
  );
}

export async function moveGalleryImage(id: string, afterId: string | null): Promise<GalleryImage> {
  return parseJson(
    await authFetch(`/api/inference/images/gallery/${id}/move`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ after_id: afterId }),
    }),
  );
}

export async function addGalleryImageToProject(
  id: string,
  projectId: string,
): Promise<{ path: string; already: boolean }> {
  return parseJson(
    await authFetch(`/api/inference/images/gallery/${id}/project`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ project_id: projectId }),
    }),
  );
}

export async function setGalleryImageFlags(
  id: string,
  flags: { pinned?: boolean; archived?: boolean },
): Promise<GalleryImage> {
  return parseJson(
    await authFetch(`/api/inference/images/gallery/${id}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(flags),
    }),
  );
}

export async function deleteGalleryImage(id: string): Promise<void> {
  const res = await authFetch(`/api/inference/images/gallery/${id}`, { method: "DELETE" });
  if (!res.ok && res.status !== 404) throw new Error(await readFastApiError(res));
}

export async function clearGallery(): Promise<void> {
  const res = await authFetch("/api/inference/images/gallery", { method: "DELETE" });
  if (!res.ok) throw new Error(await readFastApiError(res));
}

export async function fetchGalleryResponse(url: string): Promise<Response> {
  const res = await authFetch(url);
  if (!res.ok) throw new Error(await readFastApiError(res));
  return res;
}

export async function fetchGalleryBlob(url: string): Promise<Blob> {
  return (await fetchGalleryResponse(url)).blob();
}

/** Auth-protected, so it cannot be a plain <img src>. Callers must revoke the URL. */
export async function fetchGalleryObjectUrl(
  url: string,
): Promise<{ url: string; bytes: number }> {
  const blob = await fetchGalleryBlob(url);
  return { url: URL.createObjectURL(blob), bytes: blob.size };
}

export function galleryThumbnailUrl(url: string, thumb = 256): string {
  return `${url}?thumb=${thumb}`;
}

// Mirrors DiffusionTrainingStartRequest on the backend.
export interface DiffusionTrainingStartRequest {
  base_model: string;
  model_family?: string | null;
  data_dir: string;
  output_dir: string;
  instance_prompt?: string | null;
  resolution?: number;
  train_steps?: number;
  // > 0 overrides train_steps with that many epochs.
  num_epochs?: number;
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
  lr_warmup_steps?: number;
  base_precision?: "nf4" | "bf16" | "int8" | "fp8" | "mxfp8" | "auto";
  compile_transformer?: "off" | "on" | "auto";
  cache_latents?: boolean;
  cache_variants?: number;
  enable_tf32?: boolean;
  // 0 writes none; a stop-and-save always writes one.
  save_steps?: number;
  save_total_limit?: number;
  // train_steps then means the TARGET TOTAL, not additional steps.
  resume_from_checkpoint?: string | null;
  resumed_from_job_id?: string | null;
  hf_token?: string | null;
}

// `lr` entries may be null so a sparse series still aligns by index.
export interface DiffusionMetricHistory {
  steps: number[];
  loss: number[];
  lr: Array<number | null>;
  grad_norm?: Array<number | null>;
}

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
  grad_norm?: number | null;
  num_images: number | null;
  in_model_load: boolean;
  output_dir: string | null;
  lora_path: string | null;
  ema_path?: string | null;
  started_at: number | null;
  updated_at: number | null;
  catalog_path?: string | null;
  family?: string | null;
  base_model?: string | null;
  samples_per_second?: number | null;
  peak_memory_gb?: number | null;
  checkpoint_path?: string | null;
  checkpoint_step?: number | null;
  resume_blocked_reason?: string | null;
  resumed_from_step?: number | null;
  metric_history?: DiffusionMetricHistory | null;
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

export async function stopDiffusionTraining(save = true): Promise<{ status: string }> {
  return parseJson(
    await authFetch("/api/train/diffusion/stop", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ save }),
    }),
  );
}

export interface DiffusionTrainingRunSummary {
  job_id: string;
  status: string;
  message?: string;
  adapter?: string | null;
  family?: string | null;
  base_model?: string | null;
  step: number;
  total_steps: number;
  avg_loss?: number | null;
  saved: boolean;
  catalog_path?: string | null;
  instance_prompt?: string | null;
  started_at?: number | null;
  ended_at?: number | null;
  output_dir?: string | null;
  can_resume?: boolean;
  checkpoint_step?: number | null;
  checkpoint_path?: string | null;
  resume_blocked_reason?: string | null;
  resumed_from_job_id?: string | null;
  resumed_from_step?: number | null;
}

export interface DiffusionTrainingRunDetail extends DiffusionTrainingRunSummary {
  loss?: number | null;
  samples_per_second?: number | null;
  peak_memory_gb?: number | null;
  num_images?: number | null;
  lora_path?: string | null;
  ema_path?: string | null;
  config?: Record<string, unknown> | null;
  metric_history?: DiffusionMetricHistory | null;
}

export async function listDiffusionTrainingRuns(
  limit = 20,
): Promise<{ runs: DiffusionTrainingRunSummary[] }> {
  return parseJson(await authFetch(`/api/train/diffusion/runs?limit=${limit}`));
}

export async function getDiffusionTrainingRun(
  jobId: string,
): Promise<DiffusionTrainingRunDetail> {
  return parseJson(await authFetch(`/api/train/diffusion/runs/${encodeURIComponent(jobId)}`));
}

export async function getDiffusionTrainingStatus(): Promise<DiffusionTrainingStatus> {
  return parseJson(await authFetch("/api/train/diffusion/status"));
}

export interface DiffusionDatasetSummary {
  name: string;
  path: string;
  image_count: number;
  clip_count?: number;
  caption_count: number;
}

export function datasetItemCount(d: {
  image_count: number;
  clip_count?: number;
}): number {
  return d.image_count + (d.clip_count ?? 0);
}

export interface DiffusionTrainableFamily {
  name: string;
  label: string;
  default_base: string;
  base_repos: string[];
  defaults?: {
    lora_rank?: number;
    learning_rate?: number;
    resolution?: number;
    train_steps?: number;
    train_batch_size?: number;
    mixed_precision?: "bf16" | "fp16" | "no";
    // Warmup only applies under a scheduler that reads it, so the backend sends both or neither.
    lr_scheduler?: string;
    lr_warmup_steps?: number;
  } | null;
  vram_note?: string | null;
  gated?: boolean | null;
  params?: string | null;
  qlora_vram_gb?: number | null;
  note?: string | null;
  precision_modes?: string[];
  recommended_precision?: string;
  supports_compile?: boolean;
  // Undefined on an older backend, which reads as true.
  supports_checkpoints?: boolean;
  max_train_batch_size?: number | null;
  // Deploy previews on this repo instead of the trained checkpoint (Krea trains on Raw, runs on Turbo).
  deploy_base?: string | null;
  deploy_bases?: Record<string, string>;
  base_specs?: Record<
    string,
    {
      params?: string | null;
      qlora_vram_gb?: number | null;
      gated?: boolean | null;
      note?: string | null;
    }
  >;
}

export interface DiffusionTrainingInfo {
  datasets_root: string;
  outputs_root: string;
  datasets: DiffusionDatasetSummary[];
  families?: DiffusionTrainableFamily[];
}

export async function getDiffusionTrainingInfo(): Promise<DiffusionTrainingInfo> {
  return parseJson(await authFetch("/api/train/diffusion/info"));
}

export interface DiffusionDatasetUploadResult extends DiffusionDatasetSummary {
  uploaded: number;
}

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

// Missing `kind` (older backends) means "image".
export interface DiffusionDatasetImageRecord {
  filename: string;
  caption: string | null;
  caption_source: "sidecar" | "metadata" | "none";
  kind?: "image" | "clip";
  width: number;
  height: number;
  size_bytes: number;
}

/** Clips have no thumbnail endpoint. */
export function imageRecordsOnly(
  records: DiffusionDatasetImageRecord[],
): DiffusionDatasetImageRecord[] {
  return records.filter((r) => (r.kind ?? "image") === "image");
}

export interface DiffusionDatasetImages {
  name: string;
  path: string;
  images: DiffusionDatasetImageRecord[];
}

export async function listDiffusionDatasetImages(
  name: string,
): Promise<DiffusionDatasetImages> {
  return parseJson(
    await authFetch(`/api/train/diffusion/dataset/${encodeURIComponent(name)}/images`),
  );
}

export function diffusionDatasetImageUrl(
  name: string,
  filename: string,
  thumb = 256,
): string {
  const q = thumb > 0 ? `?thumb=${thumb}` : "";
  return `/api/train/diffusion/dataset/${encodeURIComponent(name)}/image/${encodeURIComponent(filename)}${q}`;
}

export async function setDiffusionDatasetCaption(
  name: string,
  filename: string,
  caption: string,
): Promise<DiffusionDatasetImageRecord> {
  return parseJson(
    await authFetch(
      `/api/train/diffusion/dataset/${encodeURIComponent(name)}/caption/${encodeURIComponent(filename)}`,
      {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ caption }),
      },
    ),
  );
}

export async function deleteDiffusionDatasetImage(
  name: string,
  filename: string,
): Promise<void> {
  const res = await authFetch(
    `/api/train/diffusion/dataset/${encodeURIComponent(name)}/image/${encodeURIComponent(filename)}`,
    { method: "DELETE" },
  );
  if (!res.ok) throw new Error(await readFastApiError(res));
}

export interface DiffusionDatasetExample {
  id: string;
  label: string;
  repo: string;
  description: string;
  license: string;
  image_cap: number;
  suggested_trigger?: string | null;
}

export async function listDiffusionDatasetExamples(): Promise<DiffusionDatasetExample[]> {
  const data = await parseJson<{ examples: DiffusionDatasetExample[] }>(
    await authFetch("/api/train/diffusion/dataset-examples"),
  );
  return data.examples;
}

export interface DiffusionDatasetImportResult {
  name: string;
  path: string;
  image_count: number;
  clip_count?: number;
  caption_count: number;
  imported: number;
  license: string;
  source_repo: string;
}

export async function importDiffusionDatasetExample(
  id: string,
  name?: string,
): Promise<DiffusionDatasetImportResult> {
  return parseJson(
    await authFetch("/api/train/diffusion/dataset/import-example", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ id, name }),
    }),
  );
}
