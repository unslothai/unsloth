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
  transformer_quant?: "auto" | "none" | "off" | "int8" | "fp8" | "nvfp4" | "mxfp8";
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

// ── Diffusion LoRA training ───────────────────────────────────────────────────
// Mirrors DiffusionTrainingStartRequest on the backend; only the paths are required.
export interface DiffusionTrainingStartRequest {
  base_model: string;
  // Explicit family (sdxl / flux.1 / qwen-image / z-image). Optional: the backend
  // resolves it from base_model when omitted, but the Train tab always sends it so a
  // custom base still trains under the intended family.
  model_family?: string | null;
  data_dir: string;
  output_dir: string;
  instance_prompt?: string | null;
  resolution?: number;
  train_steps?: number;
  // 0 or omitted uses train_steps. > 0 overrides train_steps with that many epochs
  // (full passes over the dataset, in optimizer steps).
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
  // DiT-family quantised base precision (nf4 QLoRA by default). Ignored for sdxl, which
  // uses mixed_precision instead. "auto" lets the backend pick per family.
  base_precision?: "nf4" | "bf16" | "int8" | "fp8" | "mxfp8" | "auto";
  // Whether to torch.compile the transformer (any family whose /info reports
  // supports_compile; that includes the SDXL U-Net). "auto" lets the backend decide;
  // "off"/"on" force it.
  compile_transformer?: "off" | "on" | "auto";
  // Precompute + cache the VAE latents before the loop (skips re-encoding each epoch).
  cache_latents?: boolean;
  // How many augmentation variants to cache per image when caching latents (1..16).
  cache_variants?: number;
  // Allow TF32 matmuls on Ampere+ for a throughput win at negligible quality cost.
  enable_tf32?: boolean;
  // Forwarded to the pipeline's from_pretrained for a gated/private base repo (e.g. FLUX).
  hf_token?: string | null;
}

// Paired step-indexed history arrays for the live loss + LR charts. `lr` entries may be
// null so a sparse learning-rate series still aligns with `steps` by index.
export interface DiffusionMetricHistory {
  steps: number[];
  loss: number[];
  lr: Array<number | null>;
  // Total pre-clip gradient norm per step (the training health signal the charts show).
  grad_norm?: Array<number | null>;
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
  grad_norm?: number | null;
  num_images: number | null;
  in_model_load: boolean;
  output_dir: string | null;
  lora_path: string | null;
  started_at: number | null;
  updated_at: number | null;
  // Where the trained adapter was mirrored into the Studio LoRA catalog, and the family /
  // base it was trained from -- lets the Train tab deploy the adapter onto the right base.
  catalog_path?: string | null;
  family?: string | null;
  base_model?: string | null;
  // Live throughput + peak VRAM (from the trainer's progress events).
  samples_per_second?: number | null;
  peak_memory_gb?: number | null;
  // Bounded step/loss/lr history for the live charts.
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

// Request a stop of the running job. `save` (default true) writes the current adapter
// before halting ("Stop and save"); false discards it ("Stop").
export async function stopDiffusionTraining(save = true): Promise<{ status: string }> {
  return parseJson(
    await authFetch("/api/train/diffusion/stop", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ save }),
    }),
  );
}

// One persisted (terminal) diffusion training run, as listed in the previous-runs
// history. The detail adds the scrubbed start config + the full metric logs.
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
}

export interface DiffusionTrainingRunDetail extends DiffusionTrainingRunSummary {
  loss?: number | null;
  samples_per_second?: number | null;
  peak_memory_gb?: number | null;
  num_images?: number | null;
  lora_path?: string | null;
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

// One image-dataset folder under the Studio datasets root (GET /api/train/diffusion/info).
export interface DiffusionDatasetSummary {
  name: string;
  path: string;
  image_count: number;
  caption_count: number;
}

// Per-family training defaults (from GET /api/train/diffusion/info families[], added by
// the DiT-trainer backend). Absent on older backends; the Train tab falls back to a
// hardcoded family list when it is.
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
  } | null;
  vram_note?: string | null;
  gated?: boolean | null;
  // Quantised base precisions this family can train in (subset of
  // ["nf4","bf16","int8","fp8","auto"]); empty for sdxl, which uses mixed_precision.
  precision_modes?: string[];
  // The precision the backend recommends for this family (marked "(recommended)").
  recommended_precision?: string;
  // Whether the family's transformer can be torch.compile'd (gates the Speed > Compile row).
  supports_compile?: boolean;
}

// Where diffusion training reads/writes on this Studio, plus usable dataset folders.
export interface DiffusionTrainingInfo {
  datasets_root: string;
  outputs_root: string;
  datasets: DiffusionDatasetSummary[];
  // Added by the multi-family trainer backend; tolerate its absence.
  families?: DiffusionTrainableFamily[];
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

// ── Dataset labeling + example imports (GET/PUT/DELETE .../dataset/{name}/...) ──
// One image in a training dataset folder, with its resolved caption. `caption_source`
// records where the caption came from ("metadata" beats a per-image "sidecar"; "none"
// when uncaptioned) so the labeling grid can highlight images that still need one.
export interface DiffusionDatasetImageRecord {
  filename: string;
  caption: string | null;
  caption_source: "sidecar" | "metadata" | "none";
  width: number;
  height: number;
  size_bytes: number;
}

export interface DiffusionDatasetImages {
  name: string;
  path: string;
  images: DiffusionDatasetImageRecord[];
}

/** List every image in a dataset folder (including uncaptioned ones) for the grid. */
export async function listDiffusionDatasetImages(
  name: string,
): Promise<DiffusionDatasetImages> {
  return parseJson(
    await authFetch(`/api/train/diffusion/dataset/${encodeURIComponent(name)}/images`),
  );
}

/** Build the auth-protected thumbnail URL for a dataset image. Fetch it via
 * fetchGalleryObjectUrl (Bearer auth) into an object URL; it can't be a plain <img src>. */
export function diffusionDatasetImageUrl(
  name: string,
  filename: string,
  thumb = 256,
): string {
  const q = thumb > 0 ? `?thumb=${thumb}` : "";
  return `/api/train/diffusion/dataset/${encodeURIComponent(name)}/image/${encodeURIComponent(filename)}${q}`;
}

/** Write (or, when blank, clear) a per-image caption sidecar. Returns the updated record. */
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

/** Delete an image (and its caption + thumbnail) from a dataset folder. */
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

// A curated, one-click-importable example image dataset. `license` is shown verbatim so
// users see the terms before importing; `suggested_trigger` seeds the trigger prompt.
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
  caption_count: number;
  imported: number;
  license: string;
  source_repo: string;
}

/** Materialize a curated example dataset (by id) into a Studio dataset folder. */
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
