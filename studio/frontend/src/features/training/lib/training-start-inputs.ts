


import type { TrainingStartRequest } from "../types/api";
import { isUntrainableModelFormat } from "./model-support";

export function normalizeTrainingStartPayloadForComparison(
  payload: TrainingStartRequest,
): TrainingStartRequest {
  const normalized = { ...payload };
  normalized.model_format = isUntrainableModelFormat(payload.model_format)
    ? payload.model_format
    : null;
  return normalized;
}
