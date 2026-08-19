// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  isRawTextDatasetFormat,
  toBackendTrainingType,
} from "../lib/training-methods";
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
  const isCptOrGrpo =
    config.trainingMethod === "cpt" || config.trainingMethod === "grpo";
  const adapterMethod = config.trainingMethod !== "full";
  const isQloraMethod = config.trainingMethod === "qlora";
  const isFourBitModel = (config.selectedModel ?? "")
    .toLowerCase()
    .includes("4bit");
  return (adapterMethod && isQloraMethod) || (isCptOrGrpo && isFourBitModel);
}

export function buildTrainingStartPayload(
  config: TrainingConfigState,
  hfToken: string | null,
): TrainingStartRequest {
  const isCpt = config.trainingMethod === "cpt";
  const isGrpo = config.trainingMethod === "grpo";
  const adapterMethod = config.trainingMethod !== "full";
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
    Object.keys(config.datasetManualMapping).length > 0
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
    training_type: toBackendTrainingType(config.trainingMethod),
    hf_token: hfToken,
    model_known_cached: config.modelKnownCached,
    model_local_path: config.modelKnownCached ? config.modelLocalPath : null,
    model_format: config.modelFormat,
    load_in_4bit: trainingLoadsIn4Bit(config),
    max_seq_length: config.contextLength,
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
    dataset_streaming: hfDataset ? config.datasetStreaming : false,
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
    packing: isEmbedding ? false : config.packing,
    optim: config.optimizerType,
    lr_scheduler_type: config.lrSchedulerType,
    use_lora: adapterMethod,
    lora_r: config.loraRank,
    lora_alpha: config.loraAlpha,
    lora_dropout: config.loraDropout,
    target_modules: adapterMethod ? config.targetModules : [],
    gradient_checkpointing: config.gradientCheckpointing,
    use_rslora: adapterMethod && config.loraVariant === "rslora",
    use_loftq: adapterMethod && config.loraVariant === "loftq",
    use_dora: adapterMethod && config.loraVariant === "dora",
    // CPT always trains on full sequences (no chat format masking)
    train_on_completions:
      isEmbedding || isCpt || isRawText || isGrpo
        ? false
        : config.trainOnCompletions,
    reward_functions: isGrpo ? config.rewardFunctions : [],
    num_generations: config.numGenerations,
    max_prompt_length: config.maxPromptLength,
    max_completion_length: config.maxCompletionLength,
    grpo_temperature: config.grpoTemperature,
    grpo_top_p: config.grpoTopP,
    grpo_beta: config.grpoBeta,
    finetune_vision_layers: config.finetuneVisionLayers,
    finetune_language_layers: config.finetuneLanguageLayers,
    finetune_attention_modules: config.finetuneAttentionModules,
    finetune_mlp_modules: config.finetuneMLPModules,
    is_dataset_image: isEmbedding ? false : !!config.isDatasetImage,
    is_dataset_audio: isEmbedding ? false : config.isDatasetAudio,
    is_embedding: isEmbedding,
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
