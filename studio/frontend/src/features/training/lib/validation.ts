// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { TrainingConfigState } from "../types/config";

export interface StartValidationResult {
  ok: boolean;
  message: string | null;
}

export function validateS3Source(
  config: TrainingConfigState,
): StartValidationResult {
  if (
    config.modelType === "vision" ||
    config.modelType === "audio" ||
    config.isVisionModel ||
    config.isAudioModel
  ) {
    return {
      ok: false,
      message: "S3 datasets are not supported for vision or audio training yet.",
    };
  }
  const s3 = config.s3Config;
  if (!s3 || !s3.bucket.trim()) {
    return { ok: false, message: "Enter an S3 bucket name first." };
  }
  const hasKeys = Boolean(s3.accessKeyId && s3.secretAccessKey);
  if (!s3.useIamRole && !hasKeys) {
    return { ok: false, message: "Provide S3 access keys or enable IAM role." };
  }
  return { ok: true, message: null };
}

export function validateTrainingConfig(
  config: TrainingConfigState,
): StartValidationResult {
  if (!config.selectedModel) {
    return { ok: false, message: "Select a base model first." };
  }

  if (config.datasetSource === "huggingface") {
    if (!config.dataset) {
      return { ok: false, message: "Select a Hugging Face dataset first." };
    }
  } else if (config.datasetSource === "upload") {
    if (!config.uploadedFile) {
      return { ok: false, message: "Select a local dataset first." };
    }
  } else if (config.datasetSource === "s3") {
    return validateS3Source(config);
  } else {
    return { ok: false, message: "Unsupported dataset source." };
  }

  return { ok: true, message: null };
}
