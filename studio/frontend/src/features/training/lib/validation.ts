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
    return { ok: true, message: null };
  }

  if (config.datasetSource === "upload") {
    if (!config.uploadedFile) {
      return { ok: false, message: "Select a local dataset first." };
    }
    return { ok: true, message: null };
  }

  return {
    ok: false,
    message: "Unsupported dataset source.",
  };
}
