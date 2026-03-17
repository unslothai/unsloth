// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { TrainingConfigState } from "../types/config";

export interface StartValidationResult {
  ok: boolean;
  message: string | null;
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
  } else {
    return { ok: false, message: "Unsupported dataset source." };
  }


  return { ok: true, message: null };
}
