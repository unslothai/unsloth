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

  if (config.datasetSource !== "huggingface") {
    return {
      ok: false,
      message: "Only Hugging Face dataset source is enabled right now.",
    };
  }

  if (!config.dataset) {
    return { ok: false, message: "Select a Hugging Face dataset first." };
  }

  return { ok: true, message: null };
}
