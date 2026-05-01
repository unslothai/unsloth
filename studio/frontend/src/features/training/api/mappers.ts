// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { TrainingConfigState } from "../types/config";
import type { TrainingStartRequest } from "../types/api";

const BACKEND_LORA_TYPE = "LoRA/QLoRA";
const BACKEND_FULL_TYPE = "Full Finetuning";

function parseSliceValue(value: string | null): number | null {
  if (value == null) return null;
  const trimmed = value.trim();
  if (!trimmed) return null;
  const num = Number(trimmed);
  if (!Number.isFinite(num) || !Number.isInteger(num) || num < 0) return null;
  return num;
}

export function toBackendTrainingType(trainingMethod: string): string {
  return trainingMethod === "full" ? BACKEND_FULL_TYPE : BACKEND_LORA_TYPE;
}

export function buildTrainingStartPayload(
  config: TrainingConfigState,
): TrainingStartRequest {
  const adapterMethod = config.trainingMethod !== "full";
  const isQloraMethod = config.trainingMethod === "qlora";
  const isEmbedding = config.isEmbeddingModel;
  const hfDataset = config.datasetSource === "huggingface" ? config.dataset : null;
  const localDatasets =
    config.datasetSource === "upload" && config.uploadedFile
      ? [config.uploadedFile]
      : [];
  let customFormatMapping: Record<string, unknown> | undefined =
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
    training_type: toBackendTrainingType(config.trainingMethod),
    hf_token: config.hfToken.trim() || null,
    load_in_4bit: adapterMethod ? isQloraMethod : false,
    max_seq_length: config.contextLength,
    trust_remote_code: config.trustRemoteCode ?? false,
    hf_dataset: hfDataset,
    subset: hfDataset ? config.datasetSubset : null,
    train_split: hfDataset ? config.datasetSplit : null,
    eval_split: hfDataset ? config.datasetEvalSplit : null,
    dataset_slice_start: parseSliceValue(config.datasetSliceStart),
    dataset_slice_end: parseSliceValue(config.datasetSliceEnd),
    local_datasets: localDatasets,
    local_eval_datasets:
      config.datasetSource === "upload" && config.uploadedEvalFile
        ? [config.uploadedEvalFile]
        : [],
    format_type: config.datasetFormat,
    custom_format_mapping: customFormatMapping,
    num_epochs: config.epochs,
    learning_rate: String(config.learningRate),
    batch_size: config.batchSize,
    gradient_accumulation_steps: config.gradientAccumulation,
    warmup_steps: isEmbedding ? null : config.warmupSteps,
    warmup_ratio: isEmbedding ? 0.03 : null,
    max_steps: config.maxSteps,
    save_steps: config.saveSteps,
    eval_steps: config.evalSteps,
    weight_decay: config.weightDecay,
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
    use_rslora: config.loraVariant === "rslora",
    use_loftq: config.loraVariant === "loftq",
    train_on_completions: isEmbedding ? false : config.trainOnCompletions,
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
