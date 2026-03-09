// SPDX-License-Identifier: AGPL-3.0-only - See /studio/LICENSE.AGPL-3.0
// Copyright © 2025 Unsloth AI

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

  // Eval steps requires an eval split to be selected
  if (config.evalSteps > 0 && !config.datasetEvalSplit) {
    return {
      ok: false,
      message:
        "Eval Steps is set but no Eval Split is selected. Choose an Eval Split or set Eval Steps to 0.",
    };
  }

  return { ok: true, message: null };
}
