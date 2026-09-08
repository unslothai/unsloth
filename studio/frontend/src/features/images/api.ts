// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { withBackgroundLoadNotice } from "@/lib/model-lifecycle-events";
import { authFetch } from "@/features/auth";
import { readFastApiError } from "@/lib/format-fastapi-error";

// One Advanced control's resolved value and provenance, for the Advanced-panel badges. `value` is
// the engaged value (null when off), `requested` is what the caller asked for (null = left to
// the backend), `source` is "auto" or "explicit", `status` says whether the ask survived, and
// `reason` is the tooltip why.
export interface DiffusionResolvedControl {
  value: string | boolean | null;
  // Absent on backends predating the requested/actual split.
  requested?: string | boolean | null;
  source: "auto" | "explicit";
  // "applied" (honored, or nothing was asked) | "fell_back" | "unsupported". Absent on older backends.
  status?: "applied" | "fell_back" | "unsupported";
  reason: string;
}

export interface DiffusionStatus {
  loaded: boolean;
  repo_id: string | null;
  family: string | null;
  base_repo: string | null;
  device: string | null;
  dtype: string | null;
  // Resolved load kind: "gguf" | "single_file" | "pipeline". Gates GGUF-only controls. Null when not loaded.
  model_kind?: string | null;
  // Selected GGUF quant. Newer backends report this separately from the compute dtype.
  gguf_variant?: string | null;
  cpu_offload: boolean;
  // The ENGAGED runtime build. The backend has always sent these; declaring them is what lets the UI
  // report what actually ran instead of echoing the load request back. Transformer quant
  // engaged on the dense fast path ("int8" / "fp8" / ...), null = the GGUF ran as-is.
  transformer_quant?: string | null;
  // Text-encoder quant engaged ("fp8" | "fp8_dynamic" | "int8" | "nvfp4"), null = dense bf16.
  text_encoder_quant?: string | null;
  // Memory mode the load ran under: "auto" | "fast" | "balanced" | "low_vram".
  memory_mode?: string | null;
  // Offload policy actually engaged: "none" | "group" | "model" | "sequential".
  offload_policy?: string | null;
  speed_mode?: string | null;
  // Speed optimisations actually engaged.
  speed_optims?: string[];
  // Attention backend engaged via the diffusers dispatcher (e.g. "_native_cudnn"), null = default SDPA.
  attention_backend?: string | null;
  transformer_cache?: string | null;
  vae_tiling?: boolean;
  // Image workflows the loaded family supports (drives tab gating). Absent when nothing is loaded or
  // on the native engine.
  workflows?: string[];
  // Whether the loaded model + quantisation can apply LoRA adapters (drives the LoRA picker enabled state).
  supports_lora?: boolean;
  // Whether the loaded model can apply a ControlNet. Diffusers only, for families with a ControlNet pipeline.
  supports_controlnet?: boolean;
  // Per-Advanced-control provenance, keyed by control name. Present only when a model is loaded on a
  // backend that records it; absent on older backends.
  resolved?: Record<string, DiffusionResolvedControl> | null;
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
  // Optional now: required for the gguf / single_file kinds, omitted for a full pipeline loaded via from_pretrained.
  gguf_filename?: string;
  // How to load the model (omit to auto-detect from gguf_filename). Non-GGUF kinds are restricted to unsloth/* repos.
  model_kind?: "gguf" | "single_file" | "pipeline";
  base_repo?: string;
  family_override?: string;
  hf_token?: string;
  cpu_offload?: boolean;
  // Advanced (load-time) tuning. All optional; omit for the backend's auto defaults.
  speed_mode?: "off" | "eager" | "default" | "max";
  transformer_quant?: "auto" | "none" | "off" | "int8" | "fp8" | "nvfp4" | "mxfp8";
  // Text-encoder precision (omit to keep the dense bf16 encoder). Refused with a 409 when the host
  // cannot run it, rather than loading dense and reporting nothing.
  text_encoder_quant?: "fp8" | "fp8_dynamic" | "int8" | "nvfp4";
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
  // CUDA / ROCm physical indices this load may use; omit for automatic. Neither engine shards a
  // checkpoint, so several cards resolve to the one with the most free VRAM.
  gpu_ids?: number[];
  transformer_cache?: "off" | "fbcache";
  // LoRA adapters to BAKE into a torchao int8/fp8 build: they can only attach to the dense
  // transformer BEFORE quantisation and compile, so a quantized load that omits them rejects
  // every generation. Ignored by bf16 / bnb-4bit, which apply at generate time.
  loras?: LoraSpecInput[];
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
  // Image-conditioned workflows. init_image alone = img2img; plus mask_image = inpaint. strength is
  // the denoise amount (0 keeps source).
  init_image?: string;
  mask_image?: string;
  strength?: number;
  // Upscale (hires fix): factor > 1 with an init_image enlarges the source and re-denoises at low strength.
  upscale?: number;
  // Additional reference images for the FLUX.2 reference workflow, combined with init_image.
  reference_images?: string[];
  // LoRA adapters for this generation (discovery id + weight, 0..2). Rejected with a 400 when the
  // loaded model cannot apply LoRA.
  loras?: LoraSpecInput[];
  // ControlNet conditioning for this generation. Rejected (400) when the loaded model cannot apply ControlNet.
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
  // "canny" preprocesses edges from a source image; any other type is an already-made map the
  // backend maps to a control mode.
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
  batch_seed?: number | null;
  batch_index: number;
  batch_size: number;
  model: string | null;
  // The load-time build. The repo id alone does not identify a pipeline (quant choice, torchao
  // scheme, baked adapters), so these are what a recipe needs once the model is unloaded.
  model_kind?: string | null;
  gguf_filename?: string | null;
  transformer_quant?: string | null;
  // The rest of the precision picture, all ENGAGED values. Absent on records written before this existed.
  text_encoder_quant?: string | null;
  memory_mode?: string | null;
  offload_policy?: string | null;
  baked_loras?: string[];
  loras?: string[];
  controlnet?: string | null;
  // Conditioned-workflow settings. The source/mask/reference/control images are not persisted, so
  // these say what ran and let restore name the inputs to supply again.
  workflow?: string | null;
  strength?: number | null;
  upscale?: number | null;
  controlnet_guidance?: string | null;
  reference_image_count?: number | null;
  created_at: number;
  // Library state, not recipe: stored beside the PNG, absent on records written before this existed.
  pinned?: boolean;
  archived?: boolean;
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

// One family's bf16 component sizes and estimated resident footprint per quant scheme.
// Hardware-independent, so it can be fetched before anything is loaded.
export interface DiffusionInferenceInfo {
  family: string;
  transformer_bf16_gb: number;
  text_encoders_bf16_gb: number;
  vae_bf16_gb: number;
  // Estimated resident GB keyed by scheme: bf16, int8, fp8, mxfp8, nvfp4.
  estimated_resident_gb: Record<string, number>;
}

export interface DiffusionInferenceInfoResponse {
  families: DiffusionInferenceInfo[];
}

/** Static per-family footprint summary for the Advanced Dtype tradeoff. Hardware-independent, so
 *  it is safe to fetch before a load. */
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
  // Announced so the loaded models indicator shows the load for as long as the toast does, rather
  // than up to one 5s poll later. This POST only starts the load, so the notice settles from
  // load-progress, not from the response.
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
  entries: {
    repo_id: string;
    files: string[];
    bytes: number;
    gguf_filename: string | null;
    /** Whether this entry holds the selected model. Only the planner knows, because a gated pick is
     *  staged from an ungated mirror under a different repo id. Absent on an older backend. */
    checkpoint?: boolean;
  }[];
  total_bytes: number;
  /** Full declared footprint, including files already present in cache. */
  required_bytes?: number;
  /** Selected checkpoint's contribution to required_bytes. */
  checkpoint_bytes?: number;
  /** Why this pick cannot load as selected (a FLUX.2 GGUF paired with a different-size base), or
   *  null when nothing is known to be wrong. The backend reads metadata only, so it stays silent
   *  rather than guessing; when it does speak, refuse the pick here, since the alternative is
   *  the loader saying the same thing after a ~19 GB download. */
  incompatible_reason?: string | null;
}

/** What to stage through the download manager before loading this pick. */
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

/** The generate POST response was lost in transit (proxy/tunnel timeout, dropped connection) rather
 *  than refused, so the generation is still running. */
export class GenerateResponseLostError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "GenerateResponseLostError";
  }
}

// Gateway statuses meaning the origin never got to answer: 524 is Cloudflare's ~100s cap, which a
// slow run routinely exceeds.
const RESPONSE_LOST_STATUSES = new Set([408, 502, 503, 504, 522, 524]);

/** Generate synchronously: the response carries the finished images. A run past the proxy window
 *  throws GenerateResponseLostError with the generation still in flight, so the caller settles
 *  it via getGenerateProgress and the gallery instead of retrying. */
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
    // fetch itself rejected: the connection dropped, not a backend refusal.
    throw new GenerateResponseLostError(
      err instanceof Error ? err.message : "Lost connection during image generation",
    );
  }
  if (!response.ok && RESPONSE_LOST_STATUSES.has(response.status)) {
    const detail = await readFastApiError(response);
    // A proxy answers with HTML (or nothing); the app answers with JSON. Only the former means the
    // request may still be running: settling an application error would poll for a generation
    // that never started and hide the reason the backend gave.
    if (
      response.status === 503 &&
      (response.headers.get("content-type") || "").toLowerCase().includes("application/json")
    ) {
      throw new Error(detail);
    }
    throw new GenerateResponseLostError(detail);
  }
  return parseJson(response);
}

/** Request a cancel. Best-effort: the diffusers sampler stops at the next step boundary (the native
 *  engine kills its sd-cli run outright) and the in-flight generate POST unwinds as a 409 with
 *  nothing persisted. `cancelled` is false when there was nothing to stop. */
export async function cancelDiffusionGeneration(
  signal?: AbortSignal,
): Promise<{ cancelled: boolean }> {
  return parseJson(
    // No network retry, and abortable. The endpoint always targets whichever generation is active NOW,
    // so a retry or a 401 refresh-and-replay firing after the stopped run settled can land on a
    // run the user started meanwhile. The signal lets the caller drop a pending one.
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

/** `archived` picks WHICH shelf to page over: false is the strip, true is the archive. */
export async function getGallery(offset = 0, limit = 50, archived = false): Promise<GalleryPage> {
  return parseJson(
    await authFetch(
      `/api/inference/images/gallery?offset=${offset}&limit=${limit}&archived=${archived}`,
    ),
  );
}

/** Pin/unpin or archive/restore one image; omitted flags are left alone. Returns the new record. */
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
  if (!res.ok) throw new Error(await readFastApiError(res));
}

export async function clearGallery(): Promise<void> {
  const res = await authFetch("/api/inference/images/gallery", { method: "DELETE" });
  if (!res.ok) throw new Error(await readFastApiError(res));
}

/** Fetch an auth-protected gallery image as its original blob. */
export async function fetchGalleryBlob(url: string): Promise<Blob> {
  const res = await authFetch(url);
  if (!res.ok) throw new Error(await readFastApiError(res));
  return res.blob();
}

/** Fetch a gallery PNG (auth-protected, so it cannot be a plain <img src>) and wrap it in an object
 *  URL. Callers must revoke it. */
export async function fetchGalleryObjectUrl(
  url: string,
): Promise<{ url: string; bytes: number }> {
  // The blob size travels with the URL: the gallery cache is budgeted in bytes, which the caller
  // cannot work out from the URL.
  const blob = await fetchGalleryBlob(url);
  return { url: URL.createObjectURL(blob), bytes: blob.size };
}

// Diffusion LoRA training. Mirrors DiffusionTrainingStartRequest on the backend; only the paths
// are required.
export interface DiffusionTrainingStartRequest {
  base_model: string;
  // Explicit family. Optional (the backend resolves it from base_model), but the Train tab always
  // sends it so a custom base trains under the intended family.
  model_family?: string | null;
  data_dir: string;
  output_dir: string;
  instance_prompt?: string | null;
  resolution?: number;
  train_steps?: number;
  // 0 or omitted uses train_steps. > 0 overrides it with that many epochs (full passes, in optimizer steps).
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
  // DiT-family quantised base precision (nf4 QLoRA by default). Ignored for sdxl, which uses mixed_precision.
  base_precision?: "nf4" | "bf16" | "int8" | "fp8" | "mxfp8" | "auto";
  // Whether to torch.compile the transformer (any family whose /info reports supports_compile).
  // "auto" lets the backend decide.
  compile_transformer?: "off" | "on" | "auto";
  // Precompute + cache the VAE latents before the loop (skips re-encoding each epoch).
  cache_latents?: boolean;
  // How many augmentation variants to cache per image when caching latents (1..16).
  cache_variants?: number;
  // Allow TF32 matmuls on Ampere+ for a throughput win at negligible quality cost.
  enable_tf32?: boolean;
  // Write a resumable checkpoint every N optimizer steps. 0 (the default) writes none; a
  // stop-and-save always writes one, so Resume stays available either way.
  save_steps?: number;
  save_total_limit?: number;
  // Continue a previous run: its output_dir, or one explicit checkpoint-<N> directory inside it.
  // train_steps then means the TARGET TOTAL, so a checkpoint at 11 with train_steps 500 runs
  // 12..500.
  resume_from_checkpoint?: string | null;
  // The run being continued. Recorded in the history for lineage only.
  resumed_from_job_id?: string | null;
  // Forwarded to the pipeline's from_pretrained for a gated/private base repo (e.g. FLUX).
  hf_token?: string | null;
}

// Paired step-indexed history arrays for the live loss and LR charts. `lr` entries may be null so
// a sparse series still aligns by index.
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
  // The second, EMA-averaged adapter, present only when the run enabled ema_decay.
  ema_path?: string | null;
  started_at: number | null;
  updated_at: number | null;
  // Where the trained adapter was mirrored into the Unsloth LoRA catalog, and the family / base it trained from.
  catalog_path?: string | null;
  family?: string | null;
  base_model?: string | null;
  // Live throughput + peak VRAM (from the trainer's progress events).
  samples_per_second?: number | null;
  peak_memory_gb?: number | null;
  // The newest resume checkpoint this job wrote, the step it holds, why one could not be written,
  // and the step a resumed job picked up from. Absent on an older backend.
  checkpoint_path?: string | null;
  checkpoint_step?: number | null;
  resume_blocked_reason?: string | null;
  resumed_from_step?: number | null;
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

// Request a stop of the running job. `save` (default true) writes the current adapter before
// halting; false discards it.
export async function stopDiffusionTraining(save = true): Promise<{ status: string }> {
  return parseJson(
    await authFetch("/api/train/diffusion/stop", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ save }),
    }),
  );
}

// One persisted (terminal) diffusion training run. The detail adds the scrubbed start config + the full metric logs.
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
  // The run's adapter directory, which is what a Resume replays as resume_from_checkpoint.
  output_dir?: string | null;
  // Whether the run can be continued, re-derived from the checkpoints on disk on every read, with
  // the step the newest bundle holds. When false, resume_blocked_reason says why.
  can_resume?: boolean;
  checkpoint_step?: number | null;
  // The exact bundle a resume would continue; sent back as resume_from_checkpoint.
  checkpoint_path?: string | null;
  resume_blocked_reason?: string | null;
  // Lineage: the run this one continued, and the step it picked up from.
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

// One dataset folder under the Unsloth datasets root (GET /api/train/diffusion/info): images,
// clips, or both. `clip_count` is absent on older backends, hence optional.
export interface DiffusionDatasetSummary {
  name: string;
  path: string;
  image_count: number;
  clip_count?: number;
  caption_count: number;
}

/** trainable items in a dataset folder, of whichever kind. */
export function datasetItemCount(d: {
  image_count: number;
  clip_count?: number;
}): number {
  return d.image_count + (d.clip_count ?? 0);
}

// Per-family training defaults (from GET /api/train/diffusion/info). Absent on older backends; the
// Train tab then falls back to a hardcoded list.
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
    // The LR ramp, as one pair: a warmup count only ramps under a scheduler that reads it, so the
    // backend advertises both or neither. Absent on a backend older than that pairing.
    lr_scheduler?: string;
    lr_warmup_steps?: number;
  } | null;
  vram_note?: string | null;
  gated?: boolean | null;
  // vram_note's facts as fields. Absent on an older backend.
  params?: string | null;
  qlora_vram_gb?: number | null;
  note?: string | null;
  // Quantised base precisions this family can train in; empty for sdxl, which uses mixed_precision.
  precision_modes?: string[];
  // The precision the backend recommends for this family (marked "(recommended)").
  recommended_precision?: string;
  // Whether the family's transformer can be torch.compile'd (gates the Speed > Compile row).
  supports_compile?: boolean;
  // Whether the family's loop writes checkpoint bundles (gates the "Checkpoint every" field).
  // Undefined on an older backend, which has no checkpointless family, so it reads as true.
  supports_checkpoints?: boolean;
  /** 1 for a family whose forward covers one packed sequence; null/absent means unrestricted. */
  max_train_batch_size?: number | null;
  // When set, deploying a LoRA trained on this family previews it on this repo instead of the
  // checkpoint it trained on (Krea trains on Raw, runs on Turbo).
  deploy_base?: string | null;
  // Variant-specific training-base to inference-base pairs, including public mirror ids.
  deploy_bases?: Record<string, string>;
  // Per-checkpoint facts that overlay the family-level chips for multi-size families.
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

// Where diffusion training reads/writes on this Unsloth, plus usable dataset folders.
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

/** Upload images (plus optional caption .txt / metadata.jsonl) into a named dataset folder. Repeat
 *  uploads accumulate; the returned name is a valid data_dir. */
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

// One item in a training dataset folder, with its resolved caption. `caption_source` records where
// it came from, so the labeling grid can highlight uncaptioned items. `kind` is absent on
// older backends, which listed images only; treat a missing value as "image".
export interface DiffusionDatasetImageRecord {
  filename: string;
  caption: string | null;
  caption_source: "sidecar" | "metadata" | "none";
  kind?: "image" | "clip";
  width: number;
  height: number;
  size_bytes: number;
}

/** the records the labeling grid can render: clips have no thumbnail endpoint. */
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

/** List every image in a dataset folder (including uncaptioned ones) for the grid. */
export async function listDiffusionDatasetImages(
  name: string,
): Promise<DiffusionDatasetImages> {
  return parseJson(
    await authFetch(`/api/train/diffusion/dataset/${encodeURIComponent(name)}/images`),
  );
}

/** Build the auth-protected thumbnail URL for a dataset image. Fetch it via fetchGalleryObjectUrl
 *  into an object URL; it cannot be a plain <img src>. */
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

// A curated, one-click-importable example image dataset. `license` is shown verbatim so users see
// the terms before importing.
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

/** Materialize a curated example dataset (by id) into an Unsloth dataset folder. */
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
