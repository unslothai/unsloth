// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { TrainingStartRequest } from "../types/api";
import { isUntrainableModelFormat } from "./model-support";

export function normalizeTrainingStartPayloadForComparison(
  payload: TrainingStartRequest,
): TrainingStartRequest {
  const normalized = { ...payload };
  normalized.model_known_cached = false;
  normalized.model_local_path = null;
  normalized.dataset_known_cached = false;
  normalized.dataset_local_path = null;
  normalized.model_format = isUntrainableModelFormat(payload.model_format)
    ? payload.model_format
    : null;
  return normalized;
}
