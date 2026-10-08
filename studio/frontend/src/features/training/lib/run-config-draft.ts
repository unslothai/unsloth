// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { translate } from "@/i18n";
import type { TrainingConfigState } from "../types/config";

function stringValue(value: unknown): string | null {
  return typeof value === "string" && value.trim() ? value : null;
}

function record(value: unknown): Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value)
    ? Object.fromEntries(Object.entries(value))
    : {};
}

/** Restore resource selections separately from hyperparameters. Never copy run identity,
 * output paths, credentials, or a previous remote-code approval into a new draft. */
export function runConfigDraftSelections(
  config: Record<string, unknown>,
): Partial<TrainingConfigState> & { selectedModel: string } {
  const selectedModel = stringValue(config.model_name);
  if (!selectedModel) {
    throw new Error(translate("studio.training.duplicateNoModel"));
  }
  const dataset = stringValue(config.hf_dataset);
  const uploadedFile = Array.isArray(config.local_datasets)
    ? stringValue(config.local_datasets[0])
    : null;
  const uploadedEvalFile = Array.isArray(config.local_eval_datasets)
    ? stringValue(config.local_eval_datasets[0])
    : null;
  const s3 = record(config.s3_dataset);
  const datasetSource =
    config.dataset_source === "s3" ? "s3" : dataset ? "huggingface" : "upload";
  const mapping = record(config.custom_format_mapping);
  const datasetManualMapping: Record<string, string> = {};
  for (const [column, role] of Object.entries(mapping)) {
    if (!column.startsWith("__") && typeof role === "string") {
      datasetManualMapping[column] = role;
    }
  }
  const datasetLabelMapping: Record<string, Record<string, string>> = {};
  for (const [column, labels] of Object.entries(
    record(mapping.__label_mapping),
  )) {
    datasetLabelMapping[column] = Object.fromEntries(
      Object.entries(record(labels)).filter(
        (entry): entry is [string, string] => typeof entry[1] === "string",
      ),
    );
  }
  const format = config.format_type;
  const decision = config.is_decision === true;
  const layout = config.decision_layout;
  const modelFormat = config.model_format;
  const modelType = decision
    ? "decision"
    : config.is_embedding === true
      ? "embeddings"
      : config.is_dataset_audio === true
        ? "audio"
        : config.is_dataset_image === true
          ? "vision"
          : "text";
  return {
    selectedModel,
    modelType,
    modelKnownCached: config.model_known_cached === true,
    modelLocalPath: stringValue(config.model_local_path),
    modelFormat:
      modelFormat === "gguf" ||
      modelFormat === "safetensors" ||
      modelFormat === "adapter" ||
      modelFormat === "checkpoint" ||
      modelFormat === "unknown"
        ? modelFormat
        : null,
    modelSubfolder: stringValue(config.model_subfolder),
    modelDefaultsAppliedFor: selectedModel,
    trainAsDecision: decision && layout === "llm",
    decisionLayout: decision
      ? layout === "llm" || layout === "clef"
        ? layout
        : "laya"
      : null,
    isVisionModel: modelType === "vision",
    isAudioModel: modelType === "audio",
    isEmbeddingModel: modelType === "embeddings",
    projectName: stringValue(config.project_name) ?? "",
    trainingMethod:
      config.training_type === "Continued Pretraining"
        ? "cpt"
        : config.training_type === "Full Finetuning"
          ? "full"
          : config.load_in_4bit === false
            ? "lora"
            : "qlora",
    datasetSource,
    dataset,
    uploadedFile,
    uploadedEvalFile,
    datasetKnownCached: config.dataset_known_cached === true,
    datasetLocalPath: stringValue(config.dataset_local_path),
    browseDatasetSelection:
      datasetSource === "huggingface"
        ? {
            source: "huggingface",
            dataset,
            knownCached: config.dataset_known_cached === true,
            localPath: stringValue(config.dataset_local_path),
          }
        : { source: "upload", uploadedFile },
    datasetSubset: stringValue(config.subset),
    datasetSplit: stringValue(config.train_split),
    datasetEvalSplit: stringValue(config.eval_split),
    datasetStreaming: config.dataset_streaming === true,
    datasetSliceStart:
      typeof config.dataset_slice_start === "number"
        ? String(config.dataset_slice_start)
        : null,
    datasetSliceEnd:
      typeof config.dataset_slice_end === "number"
        ? String(config.dataset_slice_end)
        : null,
    datasetFormat:
      format === "alpaca" ||
      format === "chatml" ||
      format === "sharegpt" ||
      format === "raw"
        ? format
        : "auto",
    datasetManualMapping,
    datasetSystemPrompt: stringValue(mapping.__system_prompt) ?? "",
    datasetLabelMapping,
    // Recheck availability/format through the existing Configure flow.
    isDatasetImage: null,
    isDatasetAudio: config.is_dataset_audio === true,
    s3Config:
      datasetSource === "s3"
        ? {
            bucket: stringValue(s3.bucket) ?? "",
            region: stringValue(s3.region) ?? "",
            prefix: stringValue(s3.prefix) ?? "",
            useIamRole: s3.use_iam_role === true,
          }
        : null,
    trustRemoteCode: false,
    approvedRemoteCodeFingerprint: null,
    wandbToken: "",
    tensorboardDir: "",
  };
}
