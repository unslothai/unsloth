// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  isRawTextDatasetFormat,
  toBackendTrainingType,
} from "../lib/training-methods";
import type { SystemInfoResponse } from "@/hooks/use-system";
import type { TrainingStartRequest } from "../types/api";
import type { TrainingConfigState } from "../types/config";

function parseSliceValue(value: string | null): number | null {
  if (value == null) return null;
  const trimmed = value.trim();
  if (!trimmed) return null;
  const num = Number(trimmed);
  if (!Number.isFinite(num) || !Number.isInteger(num) || num < 0) return null;
  return num;
}

function buildS3PayloadConfig(config: TrainingConfigState) {
  const s3 = config.datasetSource === "s3" ? config.s3Config : null;
  if (!s3) {
    return null;
  }
  if (s3.useIamRole) {
    return {
      bucket: s3.bucket,
      region: s3.region,
      prefix: s3.prefix,
      useIamRole: s3.useIamRole,
    };
  }
  return s3;
}

/** Whether this configuration asks the backend for a bnb 4-bit load.
 *
 * Exported so the UI can say what the run will do: the backend refuses 4-bit for models
 * routed to the latest-transformers sidecar, and a preview reading "QLoRA · 4-bit" for a
 * 16-bit run understates its VRAM by a wide margin. */
export function trainingLoadsIn4Bit(
  config: Pick<TrainingConfigState, "trainingMethod" | "selectedModel">,
): boolean {
  const isCpt = config.trainingMethod === "cpt";
  const adapterMethod = config.trainingMethod !== "full";
  const isQloraMethod = config.trainingMethod === "qlora";
  const isFourBitModel = (config.selectedModel ?? "")
    .toLowerCase()
    .includes("4bit");
  return (adapterMethod && isQloraMethod) || (isCpt && isFourBitModel);
}

/** Whether a run can offload layers: LoRA on the main trainer (embedding and decision models
 * train elsewhere, and the Whisper / codec audio paths have no decoder stack to stream) with
 * gradient checkpointing on, which swapped layers need. */
export function offloadSupported(
  config: Pick<
    TrainingConfigState,
    "trainingMethod" | "isEmbeddingModel" | "isAudioModel" | "modelType" | "gradientCheckpointing"
  >,
): boolean {
  return (
    config.trainingMethod !== "full" &&
    !config.isEmbeddingModel &&
    !config.isAudioModel &&
    config.modelType !== "embeddings" &&
    config.modelType !== "decision" &&
    config.gradientCheckpointing !== "none"
  );
}

/** Whether this host can swap layers to system RAM, by core's install_block_swap rule: a CUDA or
 * ROCm backend with at least one card that is not a unified-memory APU (XPU, MLX and CPU have no
 * swap path). True until `/api/system` answers; the backend refuses the same cases itself. */
export function offloadHardwareSupported(
  system: Pick<SystemInfoResponse, "status" | "device_backend" | "gpu"> | null | undefined,
): boolean {
  if (!system || system.status !== "ready") return true;
  if (system.device_backend !== "cuda" && system.device_backend !== "rocm") return false;
  const devices = system.gpu?.devices ?? [];
  return devices.length === 0 || devices.some((device) => device.unified_memory !== true);
}

/** The offload fields, sent off whenever `offloadSupported` is false (the controls are hidden then).
 * `gpuIndices` are the GPUs training can see; with more than one, the budget goes out per card. */
export function offloadPayload(
  config: Pick<
    TrainingConfigState,
    | "trainingMethod"
    | "isEmbeddingModel"
    | "isAudioModel"
    | "modelType"
    | "gradientCheckpointing"
    | "offloadLayers"
    | "offloadVramGb"
    | "offloadVramGbPerDevice"
    | "prefetchDepth"
  >,
  gpuIndices: readonly number[] = [],
  system: Parameters<typeof offloadHardwareSupported>[0] = null,
): Pick<
  TrainingStartRequest,
  "offload_layers" | "offload_vram_gb" | "offload_vram_gb_per_device" | "prefetch_depth"
> {
  const layers = !(offloadSupported(config) && offloadHardwareSupported(system))
    ? 0
    : config.offloadLayers === "auto"
      ? "auto"
      : Math.min(1024, Math.max(0, Math.floor(config.offloadLayers || 0)));
  const depth =
    config.prefetchDepth === "auto"
      ? "auto"
      : Math.min(8, Math.max(1, Math.floor(config.prefetchDepth || 2)));
  const multi = gpuIndices.length > 1;
  return {
    offload_layers: layers,
    // The budget only sizes "auto"; with a fixed count or off it would cap the run for nothing.
    // Several cards hide the single input, so a value left in it from a one-card setup is not sent.
    offload_vram_gb:
      !multi && layers === "auto" && config.offloadVramGb && config.offloadVramGb > 0
        ? config.offloadVramGb
        : null,
    offload_vram_gb_per_device:
      multi && layers === "auto"
        ? perDeviceBudgetPayload(config.offloadVramGbPerDevice, gpuIndices)
        : null,
    prefetch_depth: depth,
  };
}

/** Entry i is GPU index i's budget, null where a card has none; null when no card has one. */
export function perDeviceBudgetPayload(
  perDevice: Record<string, number | null> | undefined,
  gpuIndices: readonly number[],
): (number | null)[] | null {
  const valid = gpuIndices.filter((i) => Number.isInteger(i) && i >= 0);
  if (!valid.length) return null;
  const out: (number | null)[] = Array.from({ length: Math.max(...valid) + 1 }, () => null);
  let any = false;
  for (const i of valid) {
    const gb = perDevice?.[String(i)];
    if (typeof gb === "number" && gb > 0 && gb <= 4096) {
      out[i] = gb;
      any = true;
    }
  }
  return any ? out : null;
}

/** The GPU indices training sees, from the torch inventory in `/api/system`. */
export function trainingGpuIndices(
  gpu: { available?: boolean; devices?: { index?: number | null }[] } | null | undefined,
): number[] {
  if (!gpu?.available) return [];
  return (gpu.devices ?? [])
    .map((d) => d.index)
    .filter((i): i is number => typeof i === "number");
}

export function buildTrainingStartPayload(
  config: TrainingConfigState,
  hfToken: string | null,
  system: SystemInfoResponse | null = null,
): TrainingStartRequest {
  const isDecision = config.modelType === "decision";
  // Laya trains in 16-bit (LoRA or full); Clef and an LLM decision model also take QLoRA.
  const hasLlmBackbone =
    isDecision &&
    (config.decisionLayout === "clef" || config.decisionLayout === "llm");
  const trainingMethod =
    isDecision &&
    config.trainingMethod !== "full" &&
    !(hasLlmBackbone && config.trainingMethod === "qlora")
      ? "lora"
      : config.trainingMethod;
  const isCpt = trainingMethod === "cpt";
  const adapterMethod = trainingMethod !== "full";
  const loraVariants = adapterMethod && !isDecision;
  const _selectedModelLower = (config.selectedModel ?? "").toLowerCase();
  // DeepSeek OCR ignores user-selected image size; do not send it.
  const isDeepseekOcr =
    _selectedModelLower.includes("deepseek") &&
    _selectedModelLower.includes("ocr");
  const isEmbedding =
    config.isEmbeddingModel || config.modelType === "embeddings";
  const isRawText = isRawTextDatasetFormat(config.datasetFormat);
  const hfDataset =
    config.datasetSource === "huggingface" ? config.dataset : null;
  const localDatasets =
    config.datasetSource === "upload" && config.uploadedFile
      ? [config.uploadedFile]
      : [];
  const s3Config = buildS3PayloadConfig(config);
  const customFormatMapping: Record<string, unknown> | undefined =
    !isDecision && Object.keys(config.datasetManualMapping).length > 0
      ? { ...config.datasetManualMapping }
      : undefined;

  // Inject conversion advisor metadata into the mapping (__ prefix keys)
  const hasAdvisorMeta =
    config.datasetSystemPrompt ||
    Object.keys(config.datasetLabelMapping).length > 0;
  if (customFormatMapping && hasAdvisorMeta) {
    if (config.datasetSystemPrompt) {
      customFormatMapping.__system_prompt = config.datasetSystemPrompt;
    }
    if (Object.keys(config.datasetLabelMapping).length > 0) {
      customFormatMapping.__label_mapping = config.datasetLabelMapping;
    }
  }

  return {
    model_name: config.selectedModel ?? "",
    project_name: (config.projectName || "").trim() || null,
    training_type: toBackendTrainingType(trainingMethod),
    hf_token: hfToken,
    model_known_cached: config.modelKnownCached,
    model_local_path: config.modelKnownCached ? config.modelLocalPath : null,
    model_format: config.modelFormat,
    load_in_4bit: trainingLoadsIn4Bit({ ...config, trainingMethod }),
    // Hidden for decision runs: Laya always trains at 1024 tokens, Clef and LLMs at the recipe's length.
    max_seq_length: isDecision && !hasLlmBackbone ? 1024 : config.contextLength,
    vision_image_size:
      config.isVisionModel && config.isDatasetImage === true && !isDeepseekOcr
        ? config.visionImageSize
        : null,
    trust_remote_code: config.trustRemoteCode ?? false,
    approved_remote_code_fingerprint:
      config.approvedRemoteCodeFingerprint ?? null,
    hf_dataset: hfDataset,
    dataset_known_cached:
      hfDataset && !config.datasetStreaming ? config.datasetKnownCached : false,
    dataset_local_path:
      hfDataset && !config.datasetStreaming ? config.datasetLocalPath : null,
    subset: hfDataset ? config.datasetSubset : null,
    train_split: hfDataset ? config.datasetSplit : null,
    eval_split: hfDataset ? config.datasetEvalSplit : null,
    dataset_streaming:
      hfDataset && !isDecision ? config.datasetStreaming : false,
    dataset_slice_start: parseSliceValue(config.datasetSliceStart),
    dataset_slice_end: parseSliceValue(config.datasetSliceEnd),
    local_datasets: localDatasets,
    local_eval_datasets:
      config.datasetSource === "upload" && config.uploadedEvalFile
        ? [config.uploadedEvalFile]
        : [],
    s3_config: s3Config,
    format_type: config.datasetFormat,
    custom_format_mapping: customFormatMapping,
    num_epochs: config.epochs,
    learning_rate: String(config.learningRate),
    embedding_learning_rate:
      isCpt && config.embeddingLearningRate != null
        ? config.embeddingLearningRate
        : null,
    batch_size: config.batchSize,
    gradient_accumulation_steps: config.gradientAccumulation,
    warmup_steps: isEmbedding ? null : config.warmupSteps,
    warmup_ratio: isEmbedding ? 0.03 : null,
    max_steps: config.maxSteps,
    save_steps: config.saveSteps,
    eval_steps: config.evalSteps,
    weight_decay: config.weightDecay,
    // max_grad_norm omitted on purpose: the backend now honors an explicit value,
    // so hardcoding 0 here would pin every UI run to "clipping off" and override
    // that. Guarded by tests/training-start-payload-grad-norm.test.ts.
    max_grad_value: null,
    random_seed: config.randomSeed,
    packing: isEmbedding || isDecision ? false : config.packing,
    // Laya's recipe needs torch AdamW; Clef keeps its recipe's (8-bit) optimizer.
    optim:
      isDecision && config.decisionLayout !== "clef"
        ? "adamw_torch"
        : config.optimizerType,
    lr_scheduler_type: config.lrSchedulerType,
    use_lora: adapterMethod,
    lora_r: config.loraRank,
    lora_alpha: config.loraAlpha,
    lora_dropout: config.loraDropout,
    target_modules: adapterMethod ? config.targetModules : [],
    gradient_checkpointing: config.gradientCheckpointing,
    ...offloadPayload(config, trainingGpuIndices(system?.gpu), system),
    use_rslora: loraVariants && config.loraVariant === "rslora",
    use_loftq: loraVariants && config.loraVariant === "loftq",
    use_dora: loraVariants && config.loraVariant === "dora",
    // CPT always trains on full sequences (no chat format masking)
    train_on_completions:
      isEmbedding || isDecision || isCpt || isRawText
        ? false
        : config.trainOnCompletions,
    finetune_vision_layers: config.finetuneVisionLayers,
    finetune_language_layers: config.finetuneLanguageLayers,
    finetune_attention_modules: config.finetuneAttentionModules,
    finetune_mlp_modules: config.finetuneMLPModules,
    is_dataset_image: isEmbedding ? false : !!config.isDatasetImage,
    is_dataset_audio: isEmbedding ? false : config.isDatasetAudio,
    is_embedding: isEmbedding && !isDecision,
    is_decision: isDecision,
    model_subfolder: isDecision ? config.modelSubfolder : null,
    enable_wandb: config.enableWandb,
    wandb_token: config.enableWandb ? config.wandbToken.trim() || null : null,
    wandb_project: config.enableWandb
      ? config.wandbProject.trim() || null
      : null,
    enable_tensorboard: config.enableTensorboard,
    tensorboard_dir: config.enableTensorboard
      ? config.tensorboardDir.trim() || null
      : null,
  };
}
