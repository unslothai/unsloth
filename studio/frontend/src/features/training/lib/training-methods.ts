// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { DatasetFormat, TrainingMethod } from "@/types/training";

const BACKEND_TRAINING_TYPE: Record<TrainingMethod, string> = {
  qlora: "LoRA/QLoRA",
  lora: "LoRA/QLoRA",
  full: "Full Finetuning",
  cpt: "Continued Pretraining",
};

const TRAINING_METHOD_LABELS: Record<TrainingMethod, string> = {
  qlora: "QLoRA",
  lora: "LoRA",
  full: "Full",
  cpt: "CPT",
};

export function toBackendTrainingType(trainingMethod: TrainingMethod): string {
  return BACKEND_TRAINING_TYPE[trainingMethod];
}

export function getTrainingMethodLabel(
  trainingMethod: TrainingMethod | string,
): string {
  if (Object.prototype.hasOwnProperty.call(TRAINING_METHOD_LABELS, trainingMethod)) {
    return TRAINING_METHOD_LABELS[trainingMethod as TrainingMethod];
  }
  return TRAINING_METHOD_LABELS.full;
}

export function parseBackendTrainingMethod(
  trainingType: unknown,
  loadIn4Bit: unknown,
): TrainingMethod {
  if (trainingType === "Continued Pretraining") return "cpt";
  if (trainingType === "LoRA/QLoRA") {
    return loadIn4Bit ? "qlora" : "lora";
  }
  return "full";
}

export function isRawTextDatasetFormat(
  datasetFormat: DatasetFormat,
): boolean {
  return datasetFormat === "raw";
}
